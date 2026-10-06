# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Typed operator IR recorded only inside explicit compiled substeps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from hydroforge.core.errors import (
    ResourceCleanupError,
    SubstepCompileError,
    cleanup_on_exit,
)
from hydroforge.execution.aten import (
    COMPILED_ATEN,
    preallocated_replay_overload,
    validate_compiled_aten,
)
from hydroforge.execution.collectives import launch_recorded_collective_batch
from hydroforge.kernels.calls import recording_sink, routing
from hydroforge.kernels.metal import MetalCommandNode
from hydroforge.kernels.spec import buffer_access_semantics
from hydroforge.kernels.toolchain import CompileRequest, compile_calls


@dataclass(frozen=True, slots=True)
class _ValueRef:
    index: int
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class _StableRef:
    tensor: torch.Tensor


def _map(value: Any, function) -> Any:
    if isinstance(value, tuple):
        return tuple(_map(item, function) for item in value)
    if isinstance(value, list):
        return [_map(item, function) for item in value]
    if isinstance(value, dict):
        return {key: _map(item, function) for key, item in value.items()}
    return function(value)


def _leaves(value: Any):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    else:
        yield value


def _tensors(value: Any):
    return (item for item in _leaves(value) if isinstance(item, torch.Tensor))


def _refs(value: Any):
    return (
        item for item in _leaves(value) if isinstance(item, (_StableRef, _ValueRef))
    )


@dataclass(slots=True)
class TorchOperator:
    function: Any
    arguments: Any
    keywords: Any
    outputs: Any
    writes: tuple[torch.Tensor, ...]
    functional: Any = None
    # ``(arguments, keywords, resolved arguments, resolved keywords)``.
    _resolved: Any = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def _static(value: Any) -> Any:
        def resolve(item: Any) -> Any:
            if isinstance(item, (_StableRef, _ValueRef)):
                return item.tensor
            return item

        return _map(value, resolve)

    def static_values(self) -> tuple[Any, Any, Any]:
        """Return address-stable values for backend compilation."""
        outputs = self._static(self.outputs)
        if isinstance(outputs, (tuple, list)) and len(outputs) == 1:
            outputs = outputs[0]
        return (
            self._static(self.arguments),
            self._static(self.keywords),
            outputs,
        )

    def launch_static(self) -> None:
        """Launch with every reference bound to its address-stable tensor.

        Valid whenever no earlier replay value differs from its ``out=``
        buffer, i.e. a non-differentiable launch that starts without values.
        """

        arguments, keywords = self.arguments, self.keywords
        resolved = self._resolved
        if (
            resolved is None
            or resolved[0] is not arguments
            or resolved[1] is not keywords
        ):
            resolved = self._resolved = (
                arguments,
                keywords,
                self._static(arguments),
                self._static(keywords),
            )
        self.function(*resolved[2], **resolved[3])

    def launch(
        self, values: dict[int, torch.Tensor], *, differentiable: bool = False
    ) -> None:
        def resolve(value: Any) -> Any:
            if isinstance(value, _StableRef):
                return value.tensor
            if isinstance(value, _ValueRef):
                # Only a new producer's out= buffer is absent. Mutations of
                # an earlier local result must use that replay's live value.
                return values.get(value.index, value.tensor)
            return value

        functional = differentiable and self.functional is not None
        function = self.functional if functional else self.function
        keywords = (
            {name: value for name, value in self.keywords.items() if name != "out"}
            if functional
            else self.keywords
        )
        result = function(
            *_map(self.arguments, resolve),
            **_map(keywords, resolve),
        )

        # The recorder accepts only a single tensor result, including
        # mutations; multi-output operators and views have no lowering.
        for reference in _refs(self.outputs):
            if isinstance(reference, _ValueRef):
                values[reference.index] = result


@dataclass(frozen=True, slots=True)
class CompiledKernelCall(MetalCommandNode):
    """One fully validated and specialized native kernel invocation."""

    launch: Any
    reads: tuple[torch.Tensor, ...]
    writes: tuple[torch.Tensor, ...]
    entry: Any = None
    arguments: Any = None
    buffer_dtypes: Any = None
    call: Any = None
    temporaries: tuple[tuple[str, int], ...] = ()

    def record(self) -> None:
        """Record the specialized native launch through its dispatcher."""
        self.launch()


@dataclass(frozen=True, slots=True)
class DeferredKernelCall:
    """A validated kernel call awaiting its recording's compilation batch."""

    call: Any
    reads: tuple[torch.Tensor, ...]
    writes: tuple[torch.Tensor, ...]
    entry: Any


@dataclass(slots=True)
class CollectiveOperator:
    tensors: tuple[torch.Tensor, ...]
    abis: tuple[tuple[int, int, int], ...]
    operation: str
    reduction: str
    destination: int | None
    reads: tuple[torch.Tensor, ...]
    writes: tuple[torch.Tensor, ...]
    cuda_graph_capture_safe: bool = False
    scope: str = "spatial"
    temporaries: tuple[tuple[int, int], ...] = ()

    def launch(self, values: dict[int, torch.Tensor] | None = None) -> None:
        tensors = self.tensors
        if values is not None and self.temporaries:
            current = list(tensors)
            for position, index in self.temporaries:
                current[position] = values[index]
            tensors = tuple(current)
        launch_recorded_collective_batch(
            tensors,
            self.abis,
            operation=self.operation,
            reduction=self.reduction,
            destination=self.destination,
            scope=self.scope,
        )


@dataclass(slots=True)
class PredicateLoopOperator:
    """Hierarchical device-predicate loop embedded in an operator program."""

    loop: Any
    reads: tuple[torch.Tensor, ...]
    writes: tuple[torch.Tensor, ...]
    cuda_graph_capture_safe: bool = False

    def launch(self) -> None:
        self.loop.execute()

    def close(self, executor: Any) -> None:
        del executor
        self.loop.close()


def launch_operators(
    operators: tuple[Any, ...],
    values: dict[int, Any],
    *,
    differentiable: bool = False,
) -> None:
    """Launch operators in order; ``values`` carries their local results."""

    if not differentiable and not values:
        # Every local result is then its own ``out=`` buffer, so each
        # operator's references resolve once to the same stable tensors.
        for operator in operators:
            if isinstance(operator, TorchOperator):
                operator.launch_static()
            else:
                operator.launch()
        return
    for operator in operators:
        if isinstance(operator, TorchOperator):
            operator.launch(values, differentiable=differentiable)
        elif (
            differentiable
            and isinstance(operator, CompiledKernelCall)
            and operator.temporaries
        ):
            # Functional ATen results own fresh autograd storage. Rebind only
            # their addresses in the already validated native call; static
            # signatures, layouts and backend selection do not change.
            arguments = dict(operator.arguments)
            for name, index in operator.temporaries:
                arguments[name] = values[index]
            replace(operator.call, arguments=MappingProxyType(arguments)).compile()()
        elif differentiable and isinstance(operator, CollectiveOperator):
            operator.launch(values)
        else:
            operator.launch()


def rollback_tensors(operators: Iterable[Any]) -> tuple[torch.Tensor, ...]:
    """State a captured run must restore after warmup.

    A functional ATen producer allocates private storage and overwrites all
    of it before its consumers. Its old contents are never observable. A
    temporary produced outside this run is still preserved, so segmented
    captures cannot change the inputs supplied by an earlier segment.
    """
    operators = tuple(operators)
    private = {
        id(reference.tensor)
        for operator in operators
        if isinstance(operator, TorchOperator) and operator.functional is not None
        for reference in _refs(operator.outputs)
        if isinstance(reference, _ValueRef)
    }
    return tuple(
        dict.fromkeys(
            tensor
            for operator in operators
            for tensor in operator.writes
            if id(tensor) not in private
        )
    )


def _fingerprint(value: Any, aliases: dict[int, Any]) -> Any:
    """Hashable identity of recorded host values and tensor bindings."""

    if isinstance(value, torch.Tensor):
        return aliases.get(id(value), ("tensor", id(value)))
    if isinstance(value, _StableRef):
        return aliases.get(id(value.tensor), ("tensor", id(value.tensor)))
    if isinstance(value, _ValueRef):
        return ("value", value.index)
    if isinstance(value, float):
        return ("float", float(value).hex())
    if value is None or isinstance(value, (bool, int, str)):
        return (type(value).__name__, value)
    if isinstance(value, (tuple, list)):
        return tuple(_fingerprint(item, aliases) for item in value)
    if isinstance(value, dict):
        return tuple(
            (name, _fingerprint(item, aliases)) for name, item in sorted(value.items())
        )
    try:
        hash(value)
    except TypeError:
        return ("object", type(value).__qualname__)
    return ("object", value)


class OperatorProgram:
    """One ordered, fully bound substep operator list."""

    def __init__(self, operators: list[Any], *, eager: bool = False) -> None:
        self.operators = tuple(operators)
        self.eager = eager
        self._validate_temporary_uses()
        self.mutated_tensors = tuple(
            dict.fromkeys(
                tensor for operator in self.operators for tensor in operator.writes
            )
        )
        self.rollback_tensors = rollback_tensors(self.operators)
        tensors: list[torch.Tensor] = []
        for operator in self.operators:
            tensors.extend(getattr(operator, "reads", ()))
            tensors.extend(getattr(operator, "writes", ()))
            if isinstance(operator, TorchOperator):
                tensors.extend(
                    reference.tensor
                    for reference in (
                        *_refs(operator.arguments),
                        *_refs(operator.keywords),
                        *_refs(operator.outputs),
                    )
                )
        self.referenced_tensors = tuple(dict.fromkeys(tensors))
        self._referenced_tensor_ids = frozenset(map(id, self.referenced_tensors))
        self.cuda_graph_capture_safe = all(
            getattr(operator, "cuda_graph_capture_safe", True)
            for operator in self.operators
        )

    def fingerprint(self, aliases: dict[int, Any]) -> tuple[Any, ...]:
        """Identify everything recording froze into this program.

        ``aliases`` names per-recording draft controls, so two recordings of
        the same body compare equal exactly when their operator sequence,
        bindings and host scalars (for example values derived from the step
        duration) agree.
        """

        aliases = {
            **aliases,
            **{
                id(reference.tensor): ("value", reference.index)
                for operator in self.operators
                if isinstance(operator, TorchOperator)
                for reference in _refs(operator.outputs)
                if isinstance(reference, _ValueRef)
            },
        }
        identity = []
        for operator in self.operators:
            if isinstance(operator, TorchOperator):
                identity.append(
                    (
                        operator.function,
                        _fingerprint(operator.arguments, aliases),
                        _fingerprint(operator.keywords, aliases),
                    )
                )
            elif isinstance(operator, (CompiledKernelCall, DeferredKernelCall)):
                arguments = (
                    operator.call.arguments
                    if isinstance(operator, DeferredKernelCall)
                    else operator.arguments
                )
                identity.append(
                    (
                        "kernel",
                        id(operator.entry),
                        _fingerprint(dict(arguments), aliases),
                    )
                )
            elif isinstance(operator, PredicateLoopOperator):
                nested = operator.loop
                identity.append(
                    (
                        "predicate",
                        nested.maximum_steps,
                        nested.body.fingerprint(
                            {
                                **aliases,
                                id(nested.predicate): "predicate",
                                id(nested.counter): "predicate index",
                            }
                        ),
                    )
                )
            elif isinstance(operator, CollectiveOperator):
                identity.append(
                    (
                        "collective",
                        _fingerprint(operator.tensors, aliases),
                        operator.abis,
                        operator.operation,
                        operator.reduction,
                        operator.destination,
                        operator.scope,
                    )
                )
            else:
                identity.append(("operator", type(operator).__qualname__))
        return tuple(identity)

    def output_values(self) -> dict[int, torch.Tensor]:
        """Map each local result to its address-stable ``out=`` tensor.

        A run of this program's operators launched on its own observes, with
        these values, exactly what the whole-program launch passes along.
        """

        return dict(self._output_values)

    def references_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether this compiled program reads or writes ``tensor``."""

        return id(tensor) in self._referenced_tensor_ids

    def materialize(self, pending: Iterable[CompileRequest] = ()) -> None:
        """Compile the deferred kernel calls of this program and nested bodies.

        One batch per toolchain covers the whole lexical scope and the
        ``pending`` programs of the model (statistics, step fields, loop
        control).
        """

        programs: list[OperatorProgram] = []

        def collect(program: OperatorProgram) -> None:
            programs.append(program)
            for operator in program.operators:
                if isinstance(operator, PredicateLoopOperator):
                    collect(operator.loop.body)

        collect(self)
        deferred = [
            operator
            for program in programs
            for operator in program.operators
            if isinstance(operator, DeferredKernelCall)
        ]
        launches = iter(
            compile_calls([operator.call for operator in deferred], pending)
        )
        for program in programs:
            indices = {
                id(tensor): index for index, tensor in program._output_values.items()
            }
            program.operators = tuple(
                CompiledKernelCall(
                    next(launches),
                    operator.reads,
                    operator.writes,
                    operator.entry,
                    operator.call.arguments,
                    operator.call.buffer_dtypes,
                    operator.call,
                    temporaries=tuple(
                        (name, indices[id(value)])
                        for name, value in operator.call.arguments.items()
                        if isinstance(value, torch.Tensor) and id(value) in indices
                    ),
                )
                if isinstance(operator, DeferredKernelCall)
                else operator
                for operator in program.operators
            )

    def _validate_temporary_uses(self) -> None:
        """Seal local producer identities and collect consumers in one pass."""
        produced = {}
        origins = {}
        tensor_indices = {}
        consumed = set()
        for operator in self.operators:
            if isinstance(operator, TorchOperator):
                for reference in (
                    *_refs(operator.arguments),
                    *(
                        _ref
                        for _key, _value in operator.keywords.items()
                        if _key != "out"
                        for _ref in _refs(_value)
                    ),
                ):
                    if not isinstance(reference, _ValueRef):
                        continue
                    if produced.get(reference.index) is not reference:
                        raise SubstepCompileError(
                            "local result is foreign or used before its producer"
                        )
                    consumed.add(reference.index)
                for reference in _refs(operator.outputs):
                    if not isinstance(reference, _ValueRef):
                        continue
                    if type(reference.index) is not int or reference.index < 0:
                        raise SubstepCompileError(
                            "local result index must be a nonnegative exact int"
                        )
                    previous = produced.get(reference.index)
                    if previous is not None and previous is not reference:
                        raise SubstepCompileError(
                            "different local producers share a result index"
                        )
                    produced[reference.index] = reference
                    origins.setdefault(reference.index, operator.function._schema)
                    tensor_indices.setdefault(id(reference.tensor), set()).add(
                        reference.index
                    )
            else:
                for tensor in getattr(operator, "reads", ()):
                    consumed.update(tensor_indices.get(id(tensor), ()))
        for index in produced.keys() - consumed:
            schema = origins[index]
            qualified = schema.name + (
                f".{schema.overload_name}" if schema.overload_name else ""
            )
            raise SubstepCompileError(
                f"compiled substep discards local result of {qualified}; write it to registered model state or consume it with a later operator"
            )
        self._output_values = {
            index: reference.tensor for index, reference in produced.items()
        }

    def launch(self) -> None:
        if (
            self.eager
            and torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in self.referenced_tensors)
        ):
            # Model buffers keep stable addresses across substeps and forcing
            # updates. Autograd must save the values at use time, not aliases
            # of buffers a later substep or managed call will overwrite.
            with torch.autograd.graph.saved_tensors_hooks(torch.clone, lambda x: x):
                launch_operators(self.operators, {}, differentiable=True)
        else:
            launch_operators(self.operators, {})

    def close(self, executor: Any) -> None:
        operators, self.operators = self.operators, ()
        try:
            with cleanup_on_exit(
                "operator program",
                (
                    partial(operator.close, executor)
                    for operator in operators
                    if hasattr(operator, "close")
                ),
            ):
                pass
        finally:
            self._output_values.clear()
            self.mutated_tensors = ()
            self.rollback_tensors = ()
            self.referenced_tensors = ()
            self._referenced_tensor_ids = frozenset()


class _OperatorRecorder:
    """The call sink of one operator recording; nothing launches until its
    compilation batch."""

    recording = True

    def __init__(
        self,
        execution: Any,
        stable_tensors: tuple[torch.Tensor, ...],
        *,
        scope_kind: str,
    ) -> None:
        self.execution = execution
        self.scope_kind = scope_kind
        self.binder = execution.kernel_binding
        self.operators: list[Any] = []
        self.references: dict[int, _StableRef | _ValueRef] = {
            id(tensor): _StableRef(tensor) for tensor in stable_tensors
        }
        self.next_value = 0
        self.snapshots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def reference(self, tensor: torch.Tensor) -> _StableRef | _ValueRef:
        """Resolve one tensor actually observed by the lexical substep."""

        reference = self.references.get(id(tensor))
        if reference is None and self.execution.is_model_tensor(tensor):
            reference = _StableRef(tensor)
            self.references[id(tensor)] = reference
        if reference is None:
            raise SubstepCompileError(
                "compiled substep captured a tensor that is neither declared "
                "model state nor a prior operator result; copy caller inputs "
                "into model state outside the substep with an explicit "
                "tensor.copy_()"
            )
        return reference

    def snapshot_writes(self, tensors: Any) -> tuple[torch.Tensor, ...]:
        writes = tuple(dict.fromkeys(_tensors(tensors)))
        for tensor in writes:
            if id(tensor) not in self.snapshots:
                # Recording is transactional but must not introduce a hidden
                # device-to-host synchronization or a full host mirror of the
                # model state. The snapshot is cold-path temporary storage on
                # the tensor's own device and is released with the recorder.
                with _disable_current_modes(), torch.no_grad():
                    snapshot = tensor.detach().clone(
                        memory_format=torch.preserve_format,
                    )
                self.snapshots[id(tensor)] = (tensor, snapshot)
        return writes

    def restore(self) -> None:
        with (
            _disable_current_modes(),
            torch.no_grad(),
            cleanup_on_exit(
                "substep recorded tensors",
                (
                    partial(tensor.copy_, snapshot)
                    for tensor, snapshot in reversed(tuple(self.snapshots.values()))
                ),
            ),
        ):
            pass

    def call(self, registry: Any, arguments: dict[str, Any]) -> None:
        # Explicitly supplied tensors must already belong to model/runtime
        # state or be results produced earlier in this same operator program.
        # Automatically completed values are trusted because KernelBinder
        # resolves them from the compiled model namespace.
        for value in arguments.values():
            if isinstance(value, torch.Tensor):
                self.reference(value)
        # Binding may initialize cached geometry scalars with ordinary Torch
        # reductions. Those are cold-path compiler work, not substep operators.
        with _disable_current_modes():
            call = self.binder.bind(registry, arguments)
        bound = call.arguments
        for name, value in bound.items():
            if not isinstance(value, torch.Tensor):
                continue
            if name not in arguments and id(value) not in self.references:
                self.references[id(value)] = _StableRef(value)
        # Registered kernels are intercepted and do not execute while the IR
        # is recorded, so their write set needs no trace-time snapshot.
        buffers = call.implementation.spec.buffers
        reads = tuple(
            dict.fromkeys(
                bound[name]
                for name, access in buffers.items()
                if buffer_access_semantics(access).reads
                and isinstance(bound.get(name), torch.Tensor)
            )
        )
        writes = tuple(
            dict.fromkeys(
                bound[name]
                for name, access in buffers.items()
                if buffer_access_semantics(access).writes
                and isinstance(bound.get(name), torch.Tensor)
            )
        )
        self.operators.append(DeferredKernelCall(call, reads, writes, registry))

    def record_collective_batch(
        self,
        tensors: tuple[torch.Tensor, ...],
        abis: tuple[tuple[int, int, int], ...],
        reduction: str,
        *,
        operation: str = "all_reduce",
        destination: int | None = None,
        scope: str = "spatial",
    ) -> None:
        """Record one validated communication batch at its sequence point."""

        references = tuple(self.reference(tensor) for tensor in tensors)
        self.operators.append(
            CollectiveOperator(
                tensors=tensors,
                abis=abis,
                operation=operation,
                reduction=reduction,
                destination=destination,
                scope=scope,
                reads=tensors,
                writes=tensors,
                temporaries=tuple(
                    (position, reference.index)
                    for position, reference in enumerate(references)
                    if isinstance(reference, _ValueRef)
                ),
            )
        )

    def record_predicate_loop(self, loop: Any) -> None:
        """Append one nested predicate loop to the current lexical IR."""

        body = loop.body
        reads = body.referenced_tensors
        writes = tuple(dict.fromkeys((*loop.control_state, *body.mutated_tensors)))
        self.snapshot_writes(writes)
        self.operators.append(
            PredicateLoopOperator(loop=loop, reads=reads, writes=writes)
        )

    def encode(self, value: Any) -> Any:
        def encode_one(item: Any) -> Any:
            if not isinstance(item, torch.Tensor):
                return item
            return self.reference(item)

        return _map(value, encode_one)

    def encode_outputs(self, value: Any) -> Any:
        def encode_one(item: Any) -> Any:
            if not isinstance(item, torch.Tensor):
                return item
            reference = self.references.get(id(item))
            if reference is None:
                reference = _ValueRef(self.next_value, item)
                self.next_value += 1
                self.references[id(item)] = reference
            return reference

        return _map(value, encode_one)


class _TorchOperatorMode(TorchDispatchMode):
    def __init__(self, recorder: _OperatorRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        del types
        kwargs = kwargs or {}
        schema_name = function._schema.name
        overload = function._schema.overload_name
        if (schema_name, overload) not in COMPILED_ATEN:
            qualified = schema_name + (f".{overload}" if overload else "")
            raise SubstepCompileError(
                f"Torch operator {qualified!r} has no strict compiled substep lowering"
            )
        values_by_name = {}
        for index, argument in enumerate(function._schema.arguments):
            if index < len(args):
                values_by_name[argument.name] = args[index]
            elif argument.name in kwargs:
                values_by_name[argument.name] = kwargs[argument.name]
        write_values = tuple(
            values_by_name[argument.name]
            for argument in function._schema.arguments
            if argument.alias_info is not None
            and argument.alias_info.is_write
            and argument.name in values_by_name
        )
        # Validate every mutation against its pre-call shape/dtype.  PyTorch
        # ``out=`` overloads may otherwise resize registered model state before
        # the post-call contract sees it, invalidating native pointer captures.
        if write_values:
            if len(write_values) != 1 or not isinstance(
                write_values[0],
                torch.Tensor,
            ):
                raise SubstepCompileError(
                    "compiled ATen mutations must have exactly one tensor output"
                )
            validate_compiled_aten(
                function,
                args,
                kwargs,
                write_values[0],
            )
        writes = tuple(dict.fromkeys(_tensors(write_values)))
        encoded_args = self.recorder.encode(args)
        encoded_kwargs = self.recorder.encode(kwargs)
        # Registered producers are deferred. Running a real consumer here
        # would observe pre-producer values (including invalid scratch IDs).
        # Mutations already have a validated output; pure operations need
        # only metadata and one stable, uninitialized replay destination.
        if write_values:
            # Already validated against the mutated output above.
            result = write_values[0]
        else:
            with _disable_current_modes():

                def metadata(value):
                    if isinstance(value, torch.Tensor):
                        return torch.empty_strided(
                            value.shape,
                            value.stride(),
                            dtype=value.dtype,
                            device="meta",
                        )
                    return value

                meta_result = function(*_map(args, metadata), **_map(kwargs, metadata))
                device = next(_tensors((args, kwargs))).device
                result = torch.empty_like(meta_result, device=device)
            validate_compiled_aten(function, args, kwargs, result)
        outputs = self.recorder.encode_outputs(result)
        value_outputs = tuple(
            reference
            for reference in _refs(outputs)
            if isinstance(reference, _ValueRef)
        )
        functional = None
        if value_outputs and not write_values:
            if len(value_outputs) != 1 or outputs is not value_outputs[0]:
                raise SubstepCompileError(
                    "compiled ATen out-of-place operators must return exactly "
                    "one tensor"
                )
            replay = preallocated_replay_overload(function)
            if replay is None:
                raise SubstepCompileError(
                    f"Torch operator {schema_name}.{overload} has no explicit "
                    "preallocated replay overload"
                )
            functional, function = function, replay
            encoded_kwargs = dict(encoded_kwargs)
            encoded_kwargs["out"] = value_outputs[0]
        output_writes = tuple(
            reference.tensor
            for reference in _refs(outputs)
            if isinstance(reference, _ValueRef)
        )
        self.recorder.operators.append(
            TorchOperator(
                function,
                encoded_args,
                encoded_kwargs,
                outputs,
                tuple(dict.fromkeys((*writes, *output_writes))),
                functional,
            )
        )
        return result


class OperatorRecording:
    """One transactional recording scope used by model-authored substeps."""

    def __init__(
        self,
        execution: Any,
        *,
        arguments: tuple[Any, ...] = (),
        stable_tensors: tuple[torch.Tensor, ...] = (),
        scope_kind: str = "generic",
    ) -> None:
        tensor_arguments = tuple(
            value for value in arguments if isinstance(value, torch.Tensor)
        )
        stable = tuple(dict.fromkeys((*tensor_arguments, *stable_tensors)))
        self.recorder = _OperatorRecorder(
            execution,
            stable,
            scope_kind=scope_kind,
        )
        self.mode = _TorchOperatorMode(self.recorder)
        self.grad_scope = torch.no_grad()
        self.scope: Any = None
        self.parent = None
        self.program: OperatorProgram | None = None

    def __enter__(self) -> OperatorRecording:
        self.parent = recording_sink()
        scope = routing(self.recorder)
        scope.__enter__()
        try:
            self.grad_scope.__enter__()
            try:
                self.mode.__enter__()
            except BaseException:
                self.grad_scope.__exit__(None, None, None)
                raise
        except BaseException:
            scope.__exit__(None, None, None)
            raise
        self.scope = scope
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        failures: list[BaseException] = []
        try:
            self.mode.__exit__(exc_type, exc, traceback)
        except BaseException as error:
            failures.append(error)
        try:
            self.grad_scope.__exit__(exc_type, exc, traceback)
        except BaseException as error:
            failures.append(error)
        try:
            if self.scope is not None:
                scope, self.scope = self.scope, None
                scope.__exit__(None, None, None)
        except BaseException as error:
            failures.append(error)
        try:
            # Recording is transactional: compilation may never change the
            # physical state observed by the first live substep.
            self.recorder.restore()
        except BaseException as error:
            failures.append(error)
        if exc_type is not None or failures:
            # No program will own the nested loops this body already bound.
            for operator in self.recorder.operators:
                if isinstance(operator, PredicateLoopOperator):
                    try:
                        operator.close(self.recorder.execution.executor)
                    except BaseException as error:
                        failures.append(error)
        if failures:
            causes = (() if exc is None else (exc,)) + tuple(failures)
            error = ResourceCleanupError("substep recording rollback", causes)
            raise error from (exc if exc is not None else failures[0])
        if exc_type is None:
            program = OperatorProgram(
                self.recorder.operators,
                eager=self.recorder.execution.executor.name == "eager",
            )
            if self.recorder.scope_kind != "generic" and not program.operators:
                raise SubstepCompileError(
                    f"{self.recorder.scope_kind} produced an empty operator IR"
                )
            try:
                # A nested recording's calls compile with its parent's batch.
                if self.parent is None:
                    execution = self.recorder.execution
                    with _disable_current_modes():
                        program.materialize(execution.take_pending())
                        execution.step_fields.flush()
            except BaseException as primary:
                try:
                    program.close(self.recorder.execution.executor)
                except BaseException as cleanup:
                    raise ResourceCleanupError(
                        "native operator compilation",
                        (primary, cleanup),
                    ) from primary
                raise
            self.program = program


def record_operator_scope(
    execution: Any,
    *,
    arguments: tuple[Any, ...] = (),
    stable_tensors: tuple[torch.Tensor, ...] = (),
    scope_kind: str = "generic",
) -> OperatorRecording:
    """Open an operator recording transaction without requiring a callback."""
    return OperatorRecording(
        execution,
        arguments=arguments,
        stable_tensors=stable_tensors,
        scope_kind=scope_kind,
    )

"""Typed operator IR recorded only inside explicit compiled substeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from hydroforge.kernels.context import _ACTIVE_OPERATOR_RECORDER
from hydroforge.kernels.backends.metal.protocol import MetalCommandNode
from hydroforge.contracts import buffer_access_semantics


class SubstepCompileError(RuntimeError):
    """Raised when a substep contains an operator without strict lowering."""


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


def _tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _refs(value: Any):
    if isinstance(value, (_StableRef, _ValueRef)):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _refs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _refs(item)


def _tensor_abi(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tensor.data_ptr(), tensor.dtype, tensor.device, tensor.layout,
        tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
    )


@dataclass(slots=True)
class TorchOperator:
    function: Any
    arguments: Any
    keywords: Any
    outputs: Any
    writes: tuple[torch.Tensor, ...]

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
            self._static(self.arguments), self._static(self.keywords), outputs,
        )

    def launch(self, values: dict[int, torch.Tensor]) -> None:
        output_references = {
            id(reference)
            for reference in _refs(self.outputs)
            if isinstance(reference, _ValueRef)
        }

        def resolve(value: Any) -> Any:
            if isinstance(value, _StableRef):
                return value.tensor
            if isinstance(value, _ValueRef):
                # The current node's out= buffer is address-stable storage,
                # not a dependency. Every other value reference must have
                # been produced earlier in this exact replay. Falling back to
                # its trace-time tensor would silently consume stale data when
                # an operator dependency is malformed.
                if id(value) in output_references:
                    return value.tensor
                try:
                    return values[value.index]
                except KeyError as error:
                    raise RuntimeError(
                        "compiled ATen replay consumed a temporary before its "
                        f"producer ran (value {value.index})"
                    ) from error
            return value

        result = self.function(
            *_map(self.arguments, resolve), **_map(self.keywords, resolve),
        )

        def retain(reference: Any, value: Any) -> None:
            if isinstance(reference, _ValueRef):
                if value is not reference.tensor:
                    raise RuntimeError(
                        "compiled ATen replay did not write its preallocated "
                        "address-stable output"
                    )
                values[reference.index] = reference.tensor

        def walk(reference: Any, value: Any) -> None:
            if isinstance(reference, (tuple, list)):
                for ref_item, value_item in zip(reference, value, strict=True):
                    walk(ref_item, value_item)
            elif isinstance(reference, dict):
                for key in reference:
                    walk(reference[key], value[key])
            else:
                retain(reference, value)

        walk(self.outputs, result)


@dataclass(slots=True)
class KernelOperator(MetalCommandNode):
    launch: Any
    reads: tuple[torch.Tensor, ...]
    writes: tuple[torch.Tensor, ...]

    def record(self) -> None:
        """Record the specialized native launch through its dispatcher."""
        self.launch()


@dataclass(slots=True)
class CollectiveOperator:
    tensor: torch.Tensor
    operation: str
    reduction: str
    destination: int | None
    reads: tuple[torch.Tensor, ...]
    writes: tuple[torch.Tensor, ...]

    def launch(self) -> None:
        from hydroforge.execution.collectives import (
            _launch_all_reduce, _launch_reduce,
        )

        if self.operation == "all_reduce":
            _launch_all_reduce(self.tensor, self.reduction)
        elif self.operation == "reduce" and self.destination is not None:
            _launch_reduce(
                self.tensor, self.reduction, destination=self.destination,
            )
        else:
            raise RuntimeError(
                f"invalid recorded collective operation {self.operation!r}"
            )


@dataclass(slots=True)
class _MetalKernelSegment:
    icb: Any
    writes: tuple[torch.Tensor, ...]

    def launch(self) -> None:
        self.icb.replay()

    def replay(self, count: int) -> None:
        """Encode repeated executions in one Metal command submission."""
        self.icb.replay(count)


def capture_metal_commands(
    capture: Any, commands: tuple[Any, ...], *, cyclic: bool = False,
):
    """Compile one ordered command tuple into a single owned Metal ICB.

    ``cyclic`` places a barrier after the final command as well, making the
    command buffer safe to encode repeatedly in one command encoder.
    """

    from hydroforge.kernels.backends.metal.runtime import (
        MetalCommandSequence, record_metal_commands,
    )

    if not commands:
        raise SubstepCompileError("cannot capture an empty Metal command program")
    sequence = MetalCommandSequence()
    with record_metal_commands(sequence):
        for command in commands:
            if not isinstance(command, MetalCommandNode):
                raise TypeError(
                    "Metal ICB commands must implement the nominal "
                    f"MetalCommandNode contract, got {type(command).__name__}"
                )
            command.record()
        if cyclic:
            sequence.mark_barrier()
    return _MetalKernelSegment(
        capture.register(sequence.capture()),
        tuple(dict.fromkeys(
            tensor for command in commands for tensor in command.writes
        )),
    )


class OperatorProgram:
    """One ordered, fully bound substep operator list."""

    def __init__(self, operators: list[Any]) -> None:
        self.operators = tuple(operators)
        self._validate_temporary_uses()
        self._launch_operators = self.operators
        self._metal_segments: tuple[_MetalKernelSegment, ...] = ()
        self._metal_prepared = False
        self._metal_commands: tuple[Any, ...] | None = None
        self._metal_error_flags: tuple[torch.Tensor, ...] = ()
        self.mutated_tensors = tuple(dict.fromkeys(
            tensor
            for operator in self.operators
            for tensor in operator.writes
        ))
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
        self._binding_abi = tuple(
            (tensor, _tensor_abi(tensor))
            for tensor in dict.fromkeys(tensors)
        )

    def require_stable_bindings(self) -> None:
        """Reject storage/metadata drift once per outer-step program launch."""

        for tensor, expected in self._binding_abi:
            observed = _tensor_abi(tensor)
            if observed != expected:
                raise RuntimeError(
                    "compiled substep tensor storage or metadata changed after "
                    f"recording: expected {expected}, observed {observed}; "
                    "declared model tensors must be updated in place without "
                    "resize_() or set_()"
                )

    def references_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether this compiled program reads or writes ``tensor``."""

        return any(candidate is tensor for candidate, _abi in self._binding_abi)

    def _validate_temporary_uses(self) -> None:
        """Reject pure local results that no later operator can observe."""

        for index, operator in enumerate(self.operators):
            if not isinstance(operator, TorchOperator):
                continue
            produced = tuple(
                reference for reference in _refs(operator.outputs)
                if isinstance(reference, _ValueRef)
            )
            for reference in produced:
                consumed = False
                for later in self.operators[index + 1:]:
                    if isinstance(later, TorchOperator):
                        consumed = any(
                            candidate is reference
                            for candidate in (
                                *_refs(later.arguments), *_refs(later.keywords),
                            )
                        )
                    else:
                        consumed = any(
                            tensor is reference.tensor
                            for tensor in getattr(later, "reads", ())
                        )
                    if consumed:
                        break
                if not consumed:
                    schema = operator.function._schema
                    overload = schema.overload_name
                    qualified = schema.name + (
                        f".{overload}" if overload else ""
                    )
                    raise SubstepCompileError(
                        f"compiled substep discards local result of "
                        f"{qualified}; write it to registered model state or "
                        "consume it with a later operator"
                    )

    def launch(self) -> None:
        values: dict[int, torch.Tensor] = {}
        for operator in self._launch_operators:
            if isinstance(operator, (
                KernelOperator, CollectiveOperator, _MetalKernelSegment,
            )):
                operator.launch()
            else:
                operator.launch(values)

    def prepare_metal(self, capture: Any) -> None:
        """Online-lower the complete operator program into one native ICB."""
        if self._metal_prepared:
            return
        commands = self.metal_commands()
        if commands:
            segment = capture_metal_commands(capture, commands)
            self._metal_segments = (segment,)
            self._launch_operators = (segment,)
        else:
            self._metal_segments = ()
            self._launch_operators = ()
        self._metal_prepared = True

    def metal_commands(self) -> tuple[Any, ...]:
        """Online-lower operators without capturing, for loop-level fusion."""

        if self._metal_commands is not None:
            return self._metal_commands

        from hydroforge.execution.metal_aten import lower_metal_aten

        commands: list[Any] = []
        for operator in self.operators:
            if isinstance(operator, KernelOperator):
                commands.append(operator)
            elif isinstance(operator, CollectiveOperator):
                raise SubstepCompileError(
                    "Metal ICB substeps do not support distributed collectives"
                )
            else:
                commands.extend(lower_metal_aten(operator))
        self._metal_error_flags = tuple(dict.fromkeys(
            flag
            for command in commands
            for flag in getattr(command, "errors", ())
        ))
        self._metal_commands = tuple(commands)
        return self._metal_commands

    def reset_metal_errors(self) -> None:
        for flag in self._metal_error_flags:
            flag.zero_()

    def check_metal_errors(self) -> None:
        if any(int(flag.item()) != 0 for flag in self._metal_error_flags):
            raise IndexError("Metal scatter_add index is outside the output extent")

    def close(self, capture: Any) -> None:
        segments, self._metal_segments = self._metal_segments, ()
        for segment in segments:
            capture.release(segment.icb)
        # Commands may own online-lowering scratch tensors referenced by an
        # ICB (for example the Metal scatter bounds-error flag).  Drop those
        # references only after every ICB release has been attempted.
        self._launch_operators = self.operators
        self._metal_prepared = False
        self._metal_commands = None
        self._metal_error_flags = ()


class _OperatorRecorder:
    def __init__(
        self, execution: Any, stable_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        self.execution = execution
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
                "into model state outside the substep with copy_input()"
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
        with _disable_current_modes(), torch.no_grad():
            for tensor, snapshot in self.snapshots.values():
                tensor.copy_(snapshot)

    def record(self, entry: Any, arguments: dict[str, Any]) -> None:
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
            bound = self.binder.complete(entry, arguments)
            implementation = entry.implementation(self.execution.backend)
            specialized = implementation.specialize(
                bound, frozenset(),
                buffer_dtypes=self.binder.buffer_dtypes(entry, bound),
            )
        for name, value in bound.items():
            if not isinstance(value, torch.Tensor):
                continue
            if name not in arguments and id(value) not in self.references:
                self.references[id(value)] = _StableRef(value)
        launch = specialized
        # Registered kernels are intercepted and do not execute while the IR
        # is recorded, so their write set needs no trace-time snapshot.
        reads = tuple(dict.fromkeys(
            bound[name]
            for name, access in entry.metadata.buffers.items()
            if buffer_access_semantics(access).reads
            and isinstance(bound.get(name), torch.Tensor)
        ))
        writes = tuple(dict.fromkeys(
            bound[name]
            for name, access in entry.metadata.buffers.items()
            if buffer_access_semantics(access).writes
            and isinstance(bound.get(name), torch.Tensor)
        ))
        self.operators.append(KernelOperator(launch, reads, writes))

    def record_collective(
        self, tensor: torch.Tensor, reduction: str, *,
        operation: str = "all_reduce", destination: int | None = None,
    ) -> None:
        """Record one communication operation at its physical sequence point."""

        self.reference(tensor)
        if self.execution.capture_mode == "metal_icb":
            raise SubstepCompileError(
                "Metal ICB substeps do not support distributed collectives"
            )
        self.operators.append(CollectiveOperator(
            tensor=tensor, operation=operation, reduction=reduction,
            destination=destination,
            reads=(tensor,), writes=(tensor,),
        ))

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
        from hydroforge.execution.aten import supports_aten

        del types
        kwargs = kwargs or {}
        schema_name = function._schema.name
        overload = function._schema.overload_name
        if not supports_aten(self.recorder.execution, schema_name, overload):
            qualified = schema_name + (f".{overload}" if overload else "")
            raise SubstepCompileError(
                f"Torch operator {qualified!r} has no strict compiled "
                f"substep lowering for {self.recorder.execution.capture_mode}"
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
            if argument.alias_info is not None and argument.alias_info.is_write
            and argument.name in values_by_name
        )
        # Validate every mutation against its pre-call shape/dtype.  PyTorch
        # ``out=`` overloads may otherwise resize registered model state before
        # the post-call contract sees it, invalidating native pointer captures.
        if write_values:
            if len(write_values) != 1 or not isinstance(
                write_values[0], torch.Tensor,
            ):
                raise SubstepCompileError(
                    "compiled ATen mutations must have exactly one tensor output"
                )
            from hydroforge.execution.aten import validate_compiled_aten

            validate_compiled_aten(
                function, args, kwargs, write_values[0],
            )
        writes = self.recorder.snapshot_writes(write_values)
        encoded_args = self.recorder.encode(args)
        encoded_kwargs = self.recorder.encode(kwargs)
        result = function(*args, **kwargs)
        from hydroforge.execution.aten import validate_compiled_aten

        validate_compiled_aten(function, args, kwargs, result)
        outputs = self.recorder.encode_outputs(result)
        value_outputs = tuple(
            reference for reference in _refs(outputs)
            if isinstance(reference, _ValueRef)
        )
        if value_outputs:
            if len(value_outputs) != 1 or outputs is not value_outputs[0]:
                raise SubstepCompileError(
                    "compiled ATen out-of-place operators must return exactly "
                    "one tensor"
                )
            from hydroforge.execution.aten import preallocated_replay_overload

            replay = preallocated_replay_overload(function)
            if replay is None:
                raise SubstepCompileError(
                    f"Torch operator {schema_name}.{overload} has no explicit "
                    "preallocated replay overload"
                )
            function = replay
            encoded_kwargs = dict(encoded_kwargs)
            encoded_kwargs["out"] = value_outputs[0]
        output_writes = tuple(
            reference.tensor
            for reference in _refs(outputs)
            if isinstance(reference, _ValueRef)
        )
        self.recorder.operators.append(TorchOperator(
            function, encoded_args, encoded_kwargs, outputs,
            tuple(dict.fromkeys((*writes, *output_writes))),
        ))
        return result


def record_operator_program(
    model: Any,
    body,
    arguments: tuple[Any, ...],
    *,
    stable_tensors: tuple[torch.Tensor, ...] = (),
) -> OperatorProgram:
    """Trace one explicit substep, restoring every mutated model tensor."""
    with record_operator_scope(
        model, arguments=arguments, stable_tensors=stable_tensors,
    ) as recording:
        body(*arguments)
    return recording.program


class OperatorRecording:
    """One transactional recording scope used by model-authored substeps."""

    def __init__(
        self,
        model: Any,
        *,
        arguments: tuple[Any, ...] = (),
        stable_tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        execution = model._execution
        tensor_arguments = tuple(
            value for value in arguments if isinstance(value, torch.Tensor)
        )
        stable = tuple(dict.fromkeys((*tensor_arguments, *stable_tensors)))
        self.recorder = _OperatorRecorder(
            execution, stable,
        )
        self.mode = _TorchOperatorMode(self.recorder)
        self.token = None
        self.program: OperatorProgram | None = None

    def __enter__(self) -> OperatorRecording:
        self.token = _ACTIVE_OPERATOR_RECORDER.set(self.recorder)
        try:
            self.mode.__enter__()
        except BaseException:
            _ACTIVE_OPERATOR_RECORDER.reset(self.token)
            self.token = None
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        failures: list[BaseException] = []
        try:
            self.mode.__exit__(exc_type, exc, traceback)
        except BaseException as error:
            failures.append(error)
        try:
            if self.token is not None:
                _ACTIVE_OPERATOR_RECORDER.reset(self.token)
                self.token = None
        except BaseException as error:
            failures.append(error)
        try:
            # Recording is transactional: compilation may never change the
            # physical state observed by the first live substep.
            self.recorder.restore()
        except BaseException as error:
            failures.append(error)
        if failures:
            from hydroforge.contracts import ResourceCleanupError

            causes = (() if exc is None else (exc,)) + tuple(failures)
            error = ResourceCleanupError("substep recording rollback", causes)
            raise error from (exc if exc is not None else failures[0])
        if exc_type is None:
            self.program = OperatorProgram(self.recorder.operators)


def record_operator_scope(
    model: Any,
    *,
    arguments: tuple[Any, ...] = (),
    stable_tensors: tuple[torch.Tensor, ...] = (),
) -> OperatorRecording:
    """Open an operator recording transaction without requiring a callback."""
    return OperatorRecording(
        model, arguments=arguments, stable_tensors=stable_tensors,
    )

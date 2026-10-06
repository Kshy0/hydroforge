# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Explicit model-authored compiled sub-step scopes.

A scope body is entered once for each managed-method specialization to build
the operator IR.  Later outer steps skip the Python body and replay the
cached device program.  Registered-kernel identity and intercepted ATen
operators define the IR; Python function names have no execution meaning.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import torch
from pydantic import Field, FiniteFloat, PrivateAttr, field_validator, model_validator
from torch.utils._python_dispatch import _disable_current_modes

from hydroforge.core.devices import devices_match
from hydroforge.core.errors import SubstepCompileError, cleanup_on_exit
from hydroforge.core.validation import HydroForgeModel
from hydroforge.execution.channel import StepEvent
from hydroforge.execution.context import (
    InvocationScope,
    PredicateLoopFrame,
    SubstepFrame,
    specialization_key,
    validate_callback,
)
from hydroforge.execution.loops import AdaptiveLoop, FixedLoop, PredicateLoop
from hydroforge.execution.operators import record_operator_scope
from hydroforge.kernels.calls import recording_sink

_MISSING_PROGRAM = object()

INVALID_SUBSTEP_COUNT: Final[int] = (1 << 31) - 1

# Host-value variants of one lexical scope and the step triggers (duration,
# count) already matched to one of them.  Both are bounded so a stream of
# distinct durations or counts re-records instead of growing without limit.
_HOST_VARIANT: Final[str] = "host values"
_HOST_VARIANT_LIMIT: Final[int] = 8
_HOST_TRIGGER_LIMIT: Final[int] = 256


@dataclass(frozen=True, slots=True)
class Recorded:
    """One finished recording awaiting either program construction or disposal."""

    signature: tuple[Any, ...]
    owned: tuple[Any, ...]
    build: Callable[[], Any]


def close_owned(executor: Any, owned: tuple[Any, ...], *, scope: str) -> None:
    """Close recorded programs, retaining every cleanup error."""

    with cleanup_on_exit(
        scope,
        (
            (lambda program=program: program.close(executor))
            for program in owned
            if program is not None
        ),
    ):
        pass


class _HostVariant:
    """One built program of a scope and the cache key that owns it."""

    __slots__ = ("key", "program")

    def __init__(self, key: tuple[Any, ...], program: Any) -> None:
        self.key = key
        self.program = program


class HostVariants:
    """Recorded host values of one scope's primary program, its built
    variants and the step triggers (duration, count) already matched to one.

    Triggers map straight to their variant, so a warm step never hashes or
    compares a recorded signature.  ``recent`` orders the non-primary
    variants from least to most recently used.
    """

    __slots__ = ("signature", "entries", "recent", "triggers")

    def __init__(self, signature: tuple[Any, ...]) -> None:
        self.signature = signature
        self.entries: dict[tuple[Any, ...], _HostVariant] = {}
        self.recent: OrderedDict[_HostVariant, None] = OrderedDict()
        self.triggers: OrderedDict[tuple[Any, ...], _HostVariant] = OrderedDict()


def cached_program(
    execution: Any,
    key: tuple[Any, ...],
    trigger: tuple[Any, ...],
    record: Callable[[], Generator[Any, None, Recorded]],
) -> Generator[Any, None, Any]:
    """Resolve the program whose recorded host values match this invocation.

    Recording freezes every Python scalar a scope body computes (for example
    ``seconds / count``).  A program is therefore reused for a new duration
    or count only after re-entering the body once and proving that the
    recording is identical; otherwise the body gets its own program keyed by
    those host values.  Other host inputs must be declared through
    ``specialization=``.
    """

    variants = execution.host_variants.get(key)
    if variants is not None:
        entry = variants.triggers.get(trigger)
        if entry is not None:
            variants.triggers.move_to_end(trigger)
            if entry in variants.recent:
                variants.recent.move_to_end(entry)
            return entry.program
    recorded = yield from record()
    signature = recorded.signature
    if variants is None or signature == variants.signature:
        variant = key
    else:
        variant = (*key, (_HOST_VARIANT, signature))
    entry = None if variants is None else variants.entries.get(variant)
    if entry is not None:
        close_owned(execution.executor, recorded.owned, scope="verified recording")
    else:
        try:
            program = recorded.build()
        except BaseException:
            with cleanup_on_exit(
                "substep program construction",
                (
                    lambda: close_owned(
                        execution.executor, recorded.owned, scope="substep recording"
                    ),
                ),
            ):
                raise
        execution.programs[variant] = program
        entry = _HostVariant(variant, program)
        if variants is None:
            variants = execution.host_variants[key] = HostVariants(signature)
        variants.entries[variant] = entry
    if variant is not key:
        variants.recent[entry] = None
        variants.recent.move_to_end(entry)
    variants.triggers[trigger] = entry
    variants.triggers.move_to_end(trigger)
    if len(variants.triggers) > _HOST_TRIGGER_LIMIT:
        variants.triggers.popitem(last=False)
    _evict_host_variants(execution.programs, variants)
    return entry.program


def _evict_host_variants(programs: dict[Any, Any], variants: HostVariants) -> None:
    """Close the least recently used non-primary variants beyond the limit."""

    evicted = []
    while len(variants.recent) > _HOST_VARIANT_LIMIT:
        entry, _ = variants.recent.popitem(last=False)
        del variants.entries[entry.key]
        programs.pop(entry.key, None)
        triggers = variants.triggers
        for trigger in [
            trigger for trigger, value in triggers.items() if value is entry
        ]:
            del triggers[trigger]
        evicted.append(entry.program)
    # Bookkeeping is complete before any close can fail.
    with cleanup_on_exit(
        "host-variant eviction", (program.close for program in evicted)
    ):
        pass


class _FixedSubstepRequest(HydroForgeModel):
    count: int | None = Field(default=None, ge=1, lt=INVALID_SUBSTEP_COUNT)
    requested_sub_steps: int | None = Field(
        default=None,
        ge=1,
        lt=INVALID_SUBSTEP_COUNT,
        exclude=True,
    )
    final: Callable[[], None] | None = None
    specialization: Any = None
    scope_available: bool = Field(exclude=True)

    _count: int = PrivateAttr()
    _specialization: Any = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self):
        if not self.scope_available:
            raise ValueError(
                "a managed step may execute only one fixed/adaptive substep scope"
            )
        if self.count is not None and self.requested_sub_steps is not None:
            raise ValueError(
                "computed fixed substep count conflicts with explicit "
                "num_sub_steps request"
            )
        self._count = self.count or self.requested_sub_steps or 1
        self._specialization = specialization_key(self.specialization)
        return self

    @property
    def resolved_count(self) -> int:
        return self._count

    @property
    def specialization_key(self) -> Any:
        return self._specialization


@lru_cache(maxsize=128)
def _fixed_count(count: int | None, requested: int | None) -> int:
    """Validate an exact-int fixed count once per distinct request."""
    return _FixedSubstepRequest(
        count=count, requested_sub_steps=requested, scope_available=True
    ).resolved_count


@lru_cache(maxsize=128)
def _encoded_scalar(value: float, dtype: torch.dtype) -> float:
    """Round one host scalar exactly as the model dtype stores it."""
    return torch.tensor(value, dtype=dtype).item()


class _AdaptiveSubstepRequest(HydroForgeModel):
    candidate_dt: torch.Tensor
    dt: torch.Tensor
    maximum_dt: int | FiniteFloat = Field(gt=0)
    maximum_steps: int = Field(ge=1, lt=INVALID_SUBSTEP_COUNT)
    proposal: Callable[[], None]
    specialization: Any = None
    requested_sub_steps: int | None = Field(default=None, exclude=True)
    model_dtype: torch.dtype = Field(exclude=True)
    model_device: torch.device = Field(exclude=True)
    scope_available: bool = Field(exclude=True)

    _maximum_dt: float = PrivateAttr()
    _specialization: Any = PrivateAttr()

    @field_validator("maximum_dt", mode="before")
    @classmethod
    def _exact_maximum_dt(cls, value: Any):
        if type(value) not in (int, float):
            raise ValueError("adaptive maximum_dt must be an exact real scalar")
        return value

    @model_validator(mode="after")
    def _validate_request(self):
        if not self.scope_available:
            raise ValueError(
                "a managed step may execute only one fixed/adaptive substep scope"
            )
        if self.requested_sub_steps is not None:
            raise ValueError(
                "adaptive substeps conflict with explicit num_sub_steps "
                "request; omit num_sub_steps for adaptive timestepping"
            )
        for label, tensor in (
            ("candidate_dt", self.candidate_dt),
            ("dt", self.dt),
        ):
            if tensor.numel() != 1:
                raise ValueError(f"adaptive {label} must be a one-element tensor")
            if tensor.layout is not torch.strided or not tensor.is_contiguous():
                raise ValueError(
                    f"adaptive {label} must be a contiguous strided tensor"
                )
        if self.candidate_dt.dtype != self.model_dtype:
            raise ValueError("adaptive candidate_dt dtype must match model dtype")
        if self.dt.dtype != self.candidate_dt.dtype:
            raise ValueError("adaptive dt and candidate_dt must have identical dtype")
        if not devices_match(self.candidate_dt.device, self.model_device):
            raise ValueError("adaptive candidate_dt must be on the model device")
        if not devices_match(self.dt.device, self.candidate_dt.device):
            raise ValueError("adaptive dt and candidate_dt must share one device")
        try:
            maximum_dt = float(self.maximum_dt)
        except OverflowError as error:
            raise ValueError(
                "adaptive maximum_dt must be a finite real scalar"
            ) from error
        encoded = _encoded_scalar(maximum_dt, self.model_dtype)
        if not math.isfinite(encoded) or encoded <= 0:
            raise ValueError(
                "adaptive maximum_dt must remain finite and positive in "
                f"model dtype {self.model_dtype}"
            )
        if type(self.maximum_dt) is int and int(encoded) != self.maximum_dt:
            raise ValueError(
                "adaptive maximum_dt integer must be exactly representable "
                f"in model dtype {self.model_dtype}"
            )
        self._maximum_dt = maximum_dt
        self._specialization = specialization_key(self.specialization)
        return self

    @property
    def normalized_maximum_dt(self) -> float:
        return self._maximum_dt

    @property
    def specialization_key(self) -> Any:
        return self._specialization


class _PredicateLoopRequest(HydroForgeModel):
    maximum_steps: int = Field(ge=1, lt=INVALID_SUBSTEP_COUNT)


def _execute(context: Any, program: Any, *arguments: int | float) -> int:
    """Run one fixed or adaptive program and close its scope."""

    context.channel.event(StepEvent.SUBSTEP)
    completed = program.execute(*arguments, context)
    context.completed_substeps = completed
    context.scopes.pop()
    return completed


class _FixedScope(InvocationScope):
    def __init__(
        self,
        context: Any,
        *,
        key: tuple[Any, ...],
        count: int,
        final: Callable[[], None] | None,
    ) -> None:
        super().__init__(context)
        self.key = key
        self.count = count
        self.final = final
        self.completed = 0

    def _iterate(self) -> Generator[SubstepFrame, None, None]:
        context = self.context
        program = yield from cached_program(
            context.execution,
            self.key,
            (context.time_step, self.count),
            self._record,
        )
        self.completed = _execute(context, program, self.count, context.time_step)

    def _record(self) -> Generator[SubstepFrame, None, Recorded]:
        context = self.context
        execution = context.execution
        loop = FixedLoop(execution)
        controls = (loop.count, loop.counter, loop.weight)
        depth = len(context.scopes)
        with record_operator_scope(
            execution,
            stable_tensors=controls,
            scope_kind="fixed",
        ) as recording:
            yield loop.frame
            context.require_scopes_closed(depth)
        owned = [recording.program]
        try:
            if self.final is not None:
                final = record_operator_scope(
                    execution, stable_tensors=controls, scope_kind="fixed final"
                )
                # The recording rejects an empty final IR on exit.
                with final:
                    self.final()
                owned.append(final.program)
            aliases = dict(zip(map(id, controls), ("count", "index", "dt")))
            signature = tuple(program.fingerprint(aliases) for program in owned)
        except BaseException:
            with cleanup_on_exit(
                "fixed substep recording",
                (
                    lambda: close_owned(
                        execution.executor,
                        tuple(owned),
                        scope="fixed substep recording",
                    ),
                ),
            ):
                raise
        final_program = owned[1] if len(owned) > 1 else None
        return Recorded(
            signature,
            tuple(owned),
            lambda: loop.bind(owned[0], final_program),
        )


class _AdaptiveScope(InvocationScope):
    def __init__(
        self,
        context: Any,
        *,
        key: tuple[Any, ...],
        candidate_dt: torch.Tensor,
        dt: torch.Tensor,
        maximum_dt: float,
        maximum_steps: int,
        proposal: Callable[[], None],
    ) -> None:
        super().__init__(context)
        self.key = key
        self.candidate_dt = candidate_dt
        self.dt = dt
        self.maximum_dt = maximum_dt
        self.maximum_steps = maximum_steps
        self.proposal = proposal
        self.completed = 0

    def _iterate(self) -> Generator[SubstepFrame, None, None]:
        context = self.context
        program = yield from cached_program(
            context.execution,
            self.key,
            (context.time_step,),
            self._record,
        )
        self.completed = _execute(context, program, context.time_step)

    def _record(self) -> Generator[SubstepFrame, None, Recorded]:
        execution = self.context.execution
        loop = AdaptiveLoop(
            execution,
            candidate_dt=self.candidate_dt,
            dt=self.dt,
            maximum_dt=self.maximum_dt,
            maximum_steps=self.maximum_steps,
        )
        proposal = record_operator_scope(
            execution,
            stable_tensors=(loop.candidate,),
            scope_kind="adaptive proposal",
        )
        physics = record_operator_scope(
            execution,
            stable_tensors=(loop.counter, loop.time_step),
            scope_kind="adaptive physics",
        )
        with proposal:
            self.proposal()
        try:
            with physics:
                yield loop.frame
            owned = (proposal.program, physics.program)
            aliases = {id(loop.counter): "index"}
            signature = tuple(program.fingerprint(aliases) for program in owned)
        except BaseException:
            with cleanup_on_exit(
                "adaptive substep recording",
                (
                    lambda: close_owned(
                        execution.executor,
                        (proposal.program, physics.program),
                        scope="adaptive substep recording",
                    ),
                ),
            ):
                raise
        return Recorded(
            signature,
            owned,
            lambda: loop.bind(owned[0], owned[1]),
        )


class _PredicateScope(InvocationScope):
    def __init__(self, context: Any, *, maximum_steps: int) -> None:
        super().__init__(context)
        self.maximum_steps = maximum_steps

    def _iterate(self) -> Generator[PredicateLoopFrame, None, None]:
        parent = recording_sink()
        if parent is None:
            raise SubstepCompileError(
                "predicate loops must be nested directly inside a compiled "
                "fixed substep scope"
            )
        if parent.scope_kind != "fixed":
            raise SubstepCompileError(
                "predicate loops are supported only directly inside a fixed "
                f"substep; found {parent.scope_kind!r} operator scope"
            )
        context = self.context
        execution = context.execution
        context.scopes.append("predicate")
        loop = PredicateLoop(execution, maximum_steps=self.maximum_steps)
        recording = record_operator_scope(
            execution,
            stable_tensors=(loop.predicate, loop.counter, loop.continue_flag),
            scope_kind="predicate",
        )
        bound = False
        try:
            # The child recorder replaces, rather than stacks on, the parent
            # TorchDispatchMode.  Otherwise every child ATen operator would be
            # intercepted a second time by the outer recorder.
            with _disable_current_modes(), recording:
                yield PredicateLoopFrame(
                    index=loop.counter,
                    continue_flag=loop.predicate,
                )
            loop.bind(recording.program)
            bound = True
            parent.record_predicate_loop(loop)
            context.scopes.pop()
        except BaseException:
            if bound:
                loop.close()
            elif recording.program is not None:
                recording.program.close(execution.executor)
            raise


def fixed_scope(
    context: Any,
    *,
    count: object,
    specialization: Any,
    final: Callable[[], None] | None,
) -> _FixedScope:
    """Declare a fixed loop after decoding the shared count ABI.

    Count-producing device kernels may return :data:`INVALID_SUBSTEP_COUNT`
    to report an invalid or overflowing result without performing an unsafe
    integer cast.
    """
    validate_callback(final, arguments=0, label="fixed final")
    requested = context.requested_sub_steps
    try:
        # Warm steps validate only what changes: the host specialization.
        if context.substep_claimed or not (
            (count is None or type(count) is int) and (final is None or callable(final))
        ):
            raise ValueError
        resolved = _fixed_count(count, requested)
        specialized = specialization_key(specialization)
    except (TypeError, ValueError):
        request = _FixedSubstepRequest(
            count=count,
            requested_sub_steps=requested,
            final=final,
            specialization=specialization,
            scope_available=not context.substep_claimed,
        )
        resolved = request.resolved_count
        specialized = request.specialization_key
    key = context.claim_substep_scope(
        "fixed", (specialized, ("final", final is not None))
    )
    return _FixedScope(context, key=key, count=resolved, final=final)


def _tensor_identity(tensor: Any) -> tuple[Any, ...]:
    return (
        id(tensor),
        tensor.dtype,
        tensor.device,
        tensor.shape,
        tensor.layout,
        tensor.is_contiguous(),
    )


def adaptive_scope(
    context: Any,
    *,
    candidate_dt: torch.Tensor,
    dt: torch.Tensor,
    maximum_dt: float,
    maximum_steps: int,
    proposal: Callable[[], None],
    specialization: Any,
) -> _AdaptiveScope:
    validate_callback(proposal, arguments=0, label="adaptive proposal")
    execution = context.execution
    try:
        # Warm steps reuse the validation of identical controls and limits;
        # the tensor identity includes every checked tensor property.
        if (
            context.substep_claimed
            or context.requested_sub_steps is not None
            or not isinstance(candidate_dt, torch.Tensor)
            or not isinstance(dt, torch.Tensor)
        ):
            raise ValueError
        key = (
            _tensor_identity(candidate_dt),
            _tensor_identity(dt),
            type(maximum_dt),
            maximum_dt,
            type(maximum_steps),
            maximum_steps,
            specialization_key(specialization),
            callable(proposal),
        )
        validated = execution.adaptive_requests[key]
    except (KeyError, TypeError, ValueError):
        key = None
        request = _AdaptiveSubstepRequest(
            candidate_dt=candidate_dt,
            dt=dt,
            maximum_dt=maximum_dt,
            maximum_steps=maximum_steps,
            proposal=proposal,
            specialization=specialization,
            requested_sub_steps=context.requested_sub_steps,
            model_dtype=execution.dtype,
            model_device=execution.device,
            scope_available=not context.substep_claimed,
        )
        validated = (
            request.normalized_maximum_dt,
            request.maximum_steps,
            request.specialization_key,
        )
    # Tensor identity keys the cached program, so the controls must be
    # address-stable model state rather than a fresh tensor per step.
    for label, tensor in (("candidate_dt", candidate_dt), ("dt", dt)):
        if not execution.is_model_tensor(tensor):
            raise ValueError(
                f"adaptive {label} must be a declared model tensor; a new "
                "tensor per step would compile a new program every step"
            )
    if key is None:
        try:
            key = (
                _tensor_identity(candidate_dt),
                _tensor_identity(dt),
                type(maximum_dt),
                maximum_dt,
                type(maximum_steps),
                maximum_steps,
                validated[2],
                True,
            )
            requests = execution.adaptive_requests
            if len(requests) >= _HOST_TRIGGER_LIMIT:
                requests.clear()
            requests[key] = validated
        except TypeError:
            pass
    maximum, steps, specialized = validated
    scope_key = context.claim_substep_scope(
        "adaptive",
        (
            specialized,
            ("candidate_dt", id(candidate_dt)),
            ("dt", id(dt)),
            ("maximum_dt", maximum),
            ("maximum_steps", steps),
        ),
    )
    return _AdaptiveScope(
        context,
        key=scope_key,
        candidate_dt=candidate_dt,
        dt=dt,
        maximum_dt=maximum,
        maximum_steps=steps,
        proposal=proposal,
    )


def predicate_scope(context: Any, *, maximum_steps: int) -> _PredicateScope:
    """Declare a non-temporal loop controlled by a device scalar.

    The body must write ``frame.continue_flag`` on every iteration.  The loop
    executes at least once and stops when that scalar becomes zero or after
    ``maximum_steps`` iterations.  It must be nested inside the one lexical
    fixed substep owned by the managed step.
    """

    request = _PredicateLoopRequest(maximum_steps=maximum_steps)
    return _PredicateScope(context, maximum_steps=request.maximum_steps)

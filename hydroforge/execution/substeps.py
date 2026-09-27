"""Explicit model-authored compiled sub-step scopes."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

import torch
from pydantic import Field, FiniteFloat, PrivateAttr, field_validator, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.devices import devices_match

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


_MISSING_PROGRAM = object()
_MISSING_COUNT = object()

INVALID_SUBSTEP_COUNT: Final[int] = (1 << 31) - 1

# Host-value variants of one lexical scope and the step triggers (duration,
# count) already matched to one of them.  Both are bounded so a stream of
# distinct durations or counts re-records instead of growing without limit.
_HOST_VARIANT: Final[str] = "host values"
_HOST_VARIANT_LIMIT: Final[int] = 8
_HOST_TRIGGER_LIMIT: Final[int] = 256


@dataclass(frozen=True, slots=True)
class _Recorded:
    """One finished recording awaiting either program construction or disposal."""

    signature: tuple[Any, ...]
    owned: tuple[Any, ...]
    build: Callable[[], Any]


def _close_owned(
    capture: Any,
    owned: tuple[Any, ...],
    primary: BaseException | None,
    *,
    scope: str,
) -> None:
    """Close recorded programs, preserving ``primary`` and every cleanup error."""

    from hydroforge.contracts.errors import ResourceCleanupError

    failures = [] if primary is None else [primary]
    for program in owned:
        if program is None:
            continue
        try:
            program.close(capture)
        except BaseException as error:
            failures.append(error)
    if primary is None and failures:
        raise ResourceCleanupError(scope, failures) from failures[0]
    if len(failures) > 1:
        raise ResourceCleanupError(scope, failures) from primary


def _cached_program(
    execution: Any,
    key: tuple[Any, ...],
    trigger: tuple[Any, ...],
    record: Callable[[], Generator[Any, None, _Recorded]],
) -> Generator[Any, None, Any]:
    """Resolve the program whose recorded host values match this invocation.

    Recording freezes every Python scalar a scope body computes (for example
    ``seconds / count``).  A program is therefore reused for a new duration
    or count only after re-entering the body once and proving that the
    recording is identical; otherwise the body gets its own program keyed by
    those host values.  Other host inputs must be declared through
    ``specialization=``.
    """

    programs = execution.programs
    primary = programs.get(key)
    if primary is not None:
        triggers = getattr(primary, "host_triggers", {})
        if trigger in triggers:
            signature = triggers[trigger]
            program = (
                primary
                if signature == primary.host_signature
                else programs.get((*key, (_HOST_VARIANT, signature)))
            )
            if program is not None:
                triggers.move_to_end(trigger)
                return program
    recorded = yield from record()
    signature = recorded.signature
    if primary is None:
        variant = key
    elif signature == getattr(primary, "host_signature", None):
        variant = key
    else:
        variant = (*key, (_HOST_VARIANT, signature))
    program = programs.get(variant)
    if program is not None:
        _close_owned(
            execution.capture, recorded.owned, None, scope="verified recording"
        )
    else:
        try:
            program = recorded.build()
        except BaseException as error:
            _close_owned(
                execution.capture,
                recorded.owned,
                error,
                scope="substep program construction",
            )
            raise
        programs[variant] = program
        if primary is None:
            program.host_signature = signature
            program.host_triggers = OrderedDict()
            primary = program
        else:
            _evict_host_variants(programs, key, primary)
    triggers = getattr(primary, "host_triggers", None)
    if triggers is not None:
        triggers[trigger] = signature
        triggers.move_to_end(trigger)
        if len(triggers) > _HOST_TRIGGER_LIMIT:
            triggers.popitem(last=False)
    return program


def _evict_host_variants(programs: dict[Any, Any], key: tuple[Any, ...], primary: Any) -> None:
    size = len(key) + 1
    variants = [
        candidate
        for candidate in programs
        if len(candidate) == size
        and candidate[:-1] == key
        and isinstance(candidate[-1], tuple)
        and candidate[-1][:1] == (_HOST_VARIANT,)
    ]
    if len(variants) <= _HOST_VARIANT_LIMIT:
        return
    oldest = variants[0]
    evicted = programs.pop(oldest)
    signature = oldest[-1][1]
    triggers = getattr(primary, "host_triggers", {})
    for trigger in [trigger for trigger, value in triggers.items() if value == signature]:
        del triggers[trigger]
    evicted.close()


@dataclass(frozen=True, slots=True)
class SubstepFrame:
    """Compiler-owned scalar tensors visible only inside a sub-step body."""

    index: torch.Tensor
    dt: torch.Tensor


class _FixedFinalRecorder:
    def __init__(self, model: AbstractModel, *, stable_tensors) -> None:
        self.model = model
        self.stable_tensors = stable_tensors

    def record(self, callback: Callable[[], None]) -> Any:
        from hydroforge.execution.operators import record_operator_scope

        recording = record_operator_scope(
            self.model,
            stable_tensors=self.stable_tensors,
            scope_kind="fixed final",
        )
        with recording:
            callback()
        if not recording.program.operators:
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError(
                "fixed final callback produced an empty operator IR"
            )
        return recording.program


@dataclass(frozen=True, slots=True)
class PredicateLoopFrame:
    """Device state exposed to one nested predicate-loop body."""

    index: torch.Tensor
    continue_flag: torch.Tensor


def _specialization_key(value: Any) -> Any:
    """Make an explicit host specialization unambiguous and hashable."""
    if value is None:
        return None
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("substep float specialization must be finite")
        return float, value.hex()
    if type(value) in {bool, int, str}:
        return type(value), value
    if isinstance(value, tuple):
        return tuple(_specialization_key(item) for item in value)
    raise ValueError(
        "substep specialization must be None, bool, int, float, str, or a "
        "tuple composed from those exact scalar types"
    )


class _FixedSubstepRequest(HydroForgeModel):
    count: int | None = Field(default=None, ge=1, lt=INVALID_SUBSTEP_COUNT)
    requested_sub_steps: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        lt=INVALID_SUBSTEP_COUNT,
        exclude=True,
    )
    final: Callable[[], None] | None = None
    specialization: Any = None
    scope_available: bool = Field(strict=True, exclude=True)

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
        count = self.count or self.requested_sub_steps or 1
        object.__setattr__(self, "count", count)
        self._specialization = _specialization_key(self.specialization)
        return self

    @property
    def specialization_key(self) -> Any:
        return self._specialization


@lru_cache(maxsize=128)
def _plain_fixed_request(count: int | None, requested: int | None):
    return _FixedSubstepRequest(
        count=count, requested_sub_steps=requested, scope_available=True
    )


@lru_cache(maxsize=128)
def _encoded_scalar(value: float, dtype: torch.dtype) -> float:
    """Round one host scalar exactly as the model dtype stores it."""
    return torch.tensor(value, dtype=dtype).item()


class _AdaptiveSubstepRequest(HydroForgeModel):
    candidate_dt: torch.Tensor
    dt: torch.Tensor
    maximum_dt: int | FiniteFloat = Field(gt=0)
    maximum_steps: int = Field(strict=True, ge=1, lt=INVALID_SUBSTEP_COUNT)
    proposal: Callable[[], None]
    specialization: Any = None
    requested_sub_steps: int | None = Field(default=None, exclude=True)
    model_dtype: torch.dtype = Field(exclude=True)
    model_device: torch.device = Field(exclude=True)
    scope_available: bool = Field(strict=True, exclude=True)

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
        self._specialization = _specialization_key(self.specialization)
        return self

    @property
    def normalized_maximum_dt(self) -> float:
        return self._maximum_dt

    @property
    def specialization_key(self) -> Any:
        return self._specialization


class _PredicateLoopRequest(HydroForgeModel):
    maximum_steps: int = Field(strict=True, ge=1, lt=INVALID_SUBSTEP_COUNT)


class _PredicateIterationRequest(HydroForgeModel):
    """Validate the lexical compiler context at the iterator boundary."""

    parent: Any

    @model_validator(mode="after")
    def _validate_parent(self):
        if self.parent is None:
            raise ValueError(
                "predicate loops must be nested directly inside a compiled "
                "fixed substep scope"
            )
        if self.parent.scope_kind != "fixed":
            raise ValueError(
                "predicate loops are supported only directly inside a fixed "
                f"substep; found {self.parent.scope_kind!r} operator scope"
            )
        return self


class _FixedScope:
    def __init__(
        self,
        runtime: SubstepRuntime,
        *,
        key: tuple[Any, ...],
        count: int,
        duration: float,
        final: Callable[[], None] | None,
    ) -> None:
        self.runtime = runtime
        self.key = key
        self.count = count
        self.duration = duration
        self.final = final
        self.completed = 0

    def __iter__(self) -> Iterator[SubstepFrame]:
        program = yield from _cached_program(
            self.runtime.model._execution,
            self.key,
            (self.duration, self.count),
            self._record,
        )
        self.completed = self.runtime._execute_program(
            program, self.count, self.duration
        )

    def _record(self) -> Generator[SubstepFrame, None, _Recorded]:
        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.program import FixedSubstepProgram

        model = self.runtime.model
        draft = FixedSubstepProgram.recording_draft(model)
        controls = (draft.count, draft.counter, draft.weight)
        with record_operator_scope(
            model,
            stable_tensors=controls,
            scope_kind="fixed",
        ) as recording:
            yield draft.frame
            if self.runtime._pending_predicate_scopes:
                from hydroforge.execution.operators import SubstepCompileError

                raise SubstepCompileError(
                    "predicate loop scope was exited before recording "
                    "completed; do not break or return from a "
                    "step.predicate() loop"
                )
        owned = [recording.program]
        try:
            if self.final is not None:
                final = _FixedFinalRecorder(model, stable_tensors=controls)
                owned.append(final.record(self.final))
            aliases = dict(zip(map(id, controls), ("count", "index", "dt")))
            signature = tuple(program.fingerprint(aliases) for program in owned)
        except BaseException as primary:
            _close_owned(
                model._execution.capture,
                tuple(owned),
                primary,
                scope="fixed substep recording",
            )
            raise
        final_program = owned[1] if len(owned) > 1 else None
        return _Recorded(
            signature,
            tuple(owned),
            lambda: FixedSubstepProgram(model, draft, owned[0], final_program),
        )


class _AdaptiveScope:
    def __init__(
        self,
        runtime: SubstepRuntime,
        *,
        key: tuple[Any, ...],
        duration: float,
        candidate_dt: torch.Tensor,
        dt: torch.Tensor,
        maximum_dt: float,
        maximum_steps: int,
        proposal: Callable[[], None],
    ) -> None:
        self.runtime = runtime
        self.key = key
        self.duration = duration
        self.candidate_dt = candidate_dt
        self.dt = dt
        self.maximum_dt = maximum_dt
        self.maximum_steps = maximum_steps
        self.proposal = proposal
        self.completed = 0

    def __iter__(self) -> Iterator[SubstepFrame]:
        program = yield from _cached_program(
            self.runtime.model._execution,
            self.key,
            (self.duration,),
            self._record,
        )
        self.completed = self.runtime._execute_program(program, self.duration)

    def _record(self) -> Generator[SubstepFrame, None, _Recorded]:
        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.program import AdaptiveSubstepProgram

        model = self.runtime.model
        draft = AdaptiveSubstepProgram.recording_draft(
            candidate_dt=self.candidate_dt,
            dt=self.dt,
            maximum_dt=self.maximum_dt,
            maximum_steps=self.maximum_steps,
        )
        proposal = record_operator_scope(
            model,
            stable_tensors=(draft.candidate,),
            scope_kind="adaptive proposal",
        )
        physics = record_operator_scope(
            model,
            stable_tensors=(draft.counter, draft.time_step),
            scope_kind="adaptive physics",
        )
        with proposal:
            self.proposal()
        try:
            with physics:
                yield draft.frame
            owned = (proposal.program, physics.program)
            aliases = {id(draft.counter): "index"}
            signature = tuple(program.fingerprint(aliases) for program in owned)
        except BaseException as primary:
            _close_owned(
                model._execution.capture,
                (proposal.program, physics.program),
                primary,
                scope="adaptive substep recording",
            )
            raise
        return _Recorded(
            signature,
            owned,
            lambda: AdaptiveSubstepProgram(
                model, draft=draft, proposal=owned[0], body=owned[1]
            ),
        )


class _PredicateScope:
    def __init__(
        self,
        runtime: SubstepRuntime,
        *,
        maximum_steps: int,
    ) -> None:
        self.runtime = runtime
        self.maximum_steps = maximum_steps

    def __iter__(self) -> Iterator[PredicateLoopFrame]:
        from torch.utils._python_dispatch import _disable_current_modes

        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.program import PredicateLoopProgram
        from hydroforge.kernels.context import active_operator_recorder

        request = _PredicateIterationRequest(
            parent=active_operator_recorder(),
        )
        parent = request.parent
        self.runtime._pending_predicate_scopes += 1
        draft = PredicateLoopProgram.recording_draft(
            self.runtime.model,
            maximum_steps=self.maximum_steps,
        )
        recording = record_operator_scope(
            self.runtime.model,
            stable_tensors=(
                draft.predicate,
                draft.counter,
                draft.continue_flag,
            ),
            scope_kind="predicate",
        )
        program = None
        try:
            # The child recorder replaces, rather than stacks on, the parent
            # TorchDispatchMode.  Otherwise every child ATen operator would be
            # intercepted a second time by the outer recorder.
            with _disable_current_modes(), recording:
                yield PredicateLoopFrame(
                    index=draft.counter,
                    continue_flag=draft.predicate,
                )
            program = PredicateLoopProgram(
                self.runtime.model,
                maximum_steps=self.maximum_steps,
                draft=draft,
                body=recording.program,
            )
            parent.record_predicate_loop(program)
            self.runtime._pending_predicate_scopes -= 1
        except BaseException:
            if program is not None:
                program.close()
            elif recording.program is not None:
                recording.program.close(self.runtime.model._execution.capture)
            raise


class SubstepRuntime:
    """Declare compiled loops as ordinary readable Python ``for`` scopes.

    A scope body is entered once for each managed-method specialization to
    build the operator IR.  Later outer steps skip the Python body and replay
    the cached device program.  Registered-kernel identity and intercepted
    ATen operators define the IR; Python function names have no execution
    meaning.
    """

    def __init__(self, model: Any, step: Any) -> None:
        self.model = model
        self.step = step
        self._pending_predicate_scopes = 0

    def fixed(
        self,
        *,
        count: object = None,
        specialization: Any = None,
        final: Callable[[], None] | None = None,
    ) -> _FixedScope:
        """Declare a fixed loop after decoding the shared count ABI.

        Count-producing device kernels may return
        :data:`INVALID_SUBSTEP_COUNT` to report an invalid or overflowing
        result without performing an unsafe integer cast.
        """
        requested = self.step.requested_sub_steps
        if (
            final is None
            and specialization is None
            and not self.step._substep_scope_claimed
            and (count is None or type(count) is int)
            and (requested is None or type(requested) is int)
        ):
            request = _plain_fixed_request(count, requested)
        else:
            request = _FixedSubstepRequest(
                count=count,
                requested_sub_steps=requested,
                final=final,
                specialization=specialization,
                scope_available=not self.step._substep_scope_claimed,
            )
        duration, key = self.step.claim_substep_scope(
            kind="fixed",
            specialization=(
                request.specialization_key,
                ("final", request.final is not None),
            ),
        )
        return _FixedScope(
            self,
            key=key,
            count=request.count,
            duration=duration,
            final=request.final,
        )

    def adaptive(
        self,
        *,
        candidate_dt: torch.Tensor,
        dt: torch.Tensor,
        maximum_dt: float,
        maximum_steps: int,
        proposal: Callable[[], None],
        specialization: Any = None,
    ) -> _AdaptiveScope:
        request = _AdaptiveSubstepRequest(
            candidate_dt=candidate_dt,
            dt=dt,
            maximum_dt=maximum_dt,
            maximum_steps=maximum_steps,
            proposal=proposal,
            specialization=specialization,
            requested_sub_steps=self.step.requested_sub_steps,
            model_dtype=self.model.dtype,
            model_device=self.model.device,
            scope_available=not self.step._substep_scope_claimed,
        )
        # Tensor identity keys the cached program, so the controls must be
        # address-stable model state rather than a fresh tensor per step.
        for label, tensor in (("candidate_dt", request.candidate_dt), ("dt", request.dt)):
            if not self.model._execution.is_model_tensor(tensor):
                raise ValueError(
                    f"adaptive {label} must be a declared model tensor; a new "
                    "tensor per step would compile a new program every step"
                )
        duration, key = self.step.claim_substep_scope(
            kind="adaptive",
            specialization=(
                request.specialization_key,
                ("candidate_dt", id(request.candidate_dt)),
                ("dt", id(request.dt)),
                ("maximum_dt", request.normalized_maximum_dt),
                ("maximum_steps", request.maximum_steps),
            ),
        )
        return _AdaptiveScope(
            self,
            key=key,
            duration=duration,
            candidate_dt=request.candidate_dt,
            dt=request.dt,
            maximum_dt=request.normalized_maximum_dt,
            maximum_steps=request.maximum_steps,
            proposal=request.proposal,
        )

    def predicate(self, *, maximum_steps: int) -> _PredicateScope:
        """Declare a non-temporal loop controlled by a device scalar.

        The body must write ``frame.continue_flag`` on every iteration.  The
        loop executes at least once and stops when that scalar becomes zero or
        after ``maximum_steps`` iterations.  It must be nested inside the one
        lexical fixed substep owned by the managed step.
        """

        request = _PredicateLoopRequest(maximum_steps=maximum_steps)
        return _PredicateScope(
            self,
            maximum_steps=request.maximum_steps,
        )

    def _execute_program(self, program: Any, *arguments: int | float) -> int:
        """Coordinate and account for a fixed or adaptive substep execution."""

        if self.model.world_size > 1:
            from hydroforge.execution.step import (
                _DistributedStepEvent,
                _DistributedStepKind,
            )

            self.step.synchronize_distributed(
                _DistributedStepEvent(_DistributedStepKind.SUBSTEP)
            )
        completed = program.execute(*arguments, self.step)
        self.step.completed_substeps = completed
        return completed

"""Explicit model-authored compiled sub-step scopes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Final, Iterator

import torch

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


_MISSING_PROGRAM = object()
_MISSING_COUNT = object()

INVALID_SUBSTEP_COUNT: Final[int] = (1 << 31) - 1


@dataclass(frozen=True, slots=True)
class SubstepFrame:
    """Compiler-owned scalar tensors visible only inside a sub-step body."""

    index: torch.Tensor
    dt: torch.Tensor


class AdaptiveSubstepFrame:
    """Adaptive frame with an explicit proposal/physics phase boundary."""

    __slots__ = ("index", "dt", "_resolve")

    def __init__(self, frame: SubstepFrame, resolve) -> None:
        self.index = frame.index
        self.dt = frame.dt
        self._resolve = resolve

    def resolve_dt(self) -> None:
        """End dt proposal and begin the physics operator region."""
        self._resolve()


class _FixedFinalRecorder:
    def __init__(self, model: AbstractModel, *, stable_tensors) -> None:
        self.model = model
        self.stable_tensors = stable_tensors
        self.claimed = False
        self.program = None

    def record(self, callback: Callable[[], None]) -> None:
        if self.claimed:
            raise RuntimeError(
                "a fixed substep may declare only one final operator region"
            )
        if not callable(callback):
            raise TypeError("fixed substep final callback must be callable")
        from hydroforge.execution.operators import record_operator_scope

        self.claimed = True
        recording = record_operator_scope(
            self.model,
            stable_tensors=self.stable_tensors,
            scope_kind="fixed final",
        )
        with recording:
            callback()
        if recording.program is None:
            raise RuntimeError("fixed final recording did not complete")
        if not recording.program.operators:
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError(
                "fixed final callback produced an empty operator IR"
            )
        self.program = recording.program


@dataclass(frozen=True, slots=True)
class PredicateLoopFrame:
    """Device state exposed to one nested predicate-loop body."""

    index: torch.Tensor
    continue_flag: torch.Tensor


def _specialization_key(value: Any) -> Any:
    """Make an explicit host specialization unambiguous and hashable."""
    if value is None:
        return None
    if type(value) is float and not math.isfinite(value):
        raise ValueError("substep float specialization must be finite")
    if type(value) in {bool, int, float, str}:
        return type(value), value
    if isinstance(value, tuple):
        return tuple(_specialization_key(item) for item in value)
    raise TypeError(
        "substep specialization must be None, bool, int, float, str, or a "
        "tuple composed from those exact scalar types"
    )


class _FixedScope:
    def __init__(
        self, runtime: SubstepRuntime, *, key: tuple[Any, ...],
        count: int, duration: float, defer_final: bool,
    ) -> None:
        self.runtime = runtime
        self.key = key
        self.count = count
        self.duration = duration
        self.defer_final = defer_final
        self.completed = 0
        self._program = None
        self._operators = None
        self._last = None
        self._iterated = False

    def _execute(self) -> None:
        program = self._program
        if program is None:
            raise RuntimeError("fixed substep body has not been recorded")
        step = self.runtime.model._execution.active_step
        if step is None:
            raise RuntimeError("fixed substeps require @managed_step")
        if self.runtime.model.world_size > 1:
            from hydroforge.execution.step import (
                _DistributedStepEvent, _DistributedStepKind,
            )
            step.synchronize_distributed(_DistributedStepEvent(
                _DistributedStepKind.SUBSTEP,
            ))
        self.completed = program.execute(
            self.count, self.duration,
        )
        step.completed_substeps = self.completed

    def __iter__(self) -> Iterator[SubstepFrame]:
        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.program import FixedSubstepProgram

        programs = self.runtime.model._execution.programs
        program = programs.get(self.key, _MISSING_PROGRAM)
        self._iterated = True
        if program is _MISSING_PROGRAM:
            program = FixedSubstepProgram(self.runtime.model)
            final = _FixedFinalRecorder(
                self.runtime.model,
                stable_tensors=(
                    program.count, program.counter, program.weight,
                ),
            )
            frame = SubstepFrame(
                index=program.counter,
                dt=program.weight,
            )
            with record_operator_scope(
                self.runtime.model,
                stable_tensors=(
                    program.count, program.counter,
                    program.weight,
                ),
                scope_kind="fixed",
            ) as recording:
                yield frame
            if recording.program is None:
                raise RuntimeError("fixed substep recording did not complete")
            if self.defer_final:
                self._operators = recording.program
                self._last = final
            else:
                program.install(recording.program, final.program)
                programs[self.key] = program
        elif not isinstance(program, FixedSubstepProgram):
            raise RuntimeError("cached substep program has the wrong execution kind")
        elif program.operators is None:
            raise RuntimeError("cached fixed substep program is not installed")
        self._program = program
        if not self.defer_final:
            self._execute()

    def after(self, callback: Callable[[], None]) -> None:
        """Run one compiled callback after the fixed loop."""

        if not callable(callback):
            raise TypeError("fixed substep final callback must be callable")
        if not self.defer_final:
            raise RuntimeError(
                "fixed scope after() requires fixed(defer_final=True)"
            )
        if not self._iterated or self._program is None:
            raise RuntimeError(
                "fixed scope after() must follow the completed lexical loop"
            )
        if self._operators is not None:
            last = self._last
            if last is None:
                raise RuntimeError("deferred fixed final recorder is missing")
            last.record(callback)
            self._program.install(self._operators, last.program)
            self.runtime.model._execution.programs[self.key] = self._program
            self._operators = None
            self._last = None
        self._execute()


class _AdaptiveScope:
    def __init__(
        self, runtime: SubstepRuntime, *, key: tuple[Any, ...],
        duration: float, candidate_dt: torch.Tensor, dt: torch.Tensor,
        maximum_dt: float, maximum_steps: int,
    ) -> None:
        if type(duration) not in {int, float}:
            raise TypeError("adaptive substep duration must be an int or float")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("adaptive substep duration must be finite and positive")
        self.runtime = runtime
        self.key = key
        self.duration = float(duration)
        self.candidate_dt = candidate_dt
        self.dt = dt
        self.maximum_dt = maximum_dt
        self.maximum_steps = maximum_steps
        self.completed = 0

    def __iter__(self) -> Iterator[SubstepFrame]:
        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.program import AdaptiveSubstepProgram

        programs = self.runtime.model._execution.programs
        program = programs.get(self.key, _MISSING_PROGRAM)
        new_program = program is _MISSING_PROGRAM
        if new_program:
            program = AdaptiveSubstepProgram(
                self.runtime.model,
                candidate_dt=self.candidate_dt,
                dt=self.dt,
                maximum_dt=self.maximum_dt,
                maximum_steps=self.maximum_steps,
            )
        elif not isinstance(program, AdaptiveSubstepProgram):
            raise RuntimeError("cached substep program has the wrong execution kind")
        program.require_binding(
            candidate_dt=self.candidate_dt,
            dt=self.dt,
            maximum_dt=self.maximum_dt,
            maximum_steps=self.maximum_steps,
        )
        installed = (
            program.proposal_operators is not None
            and program.body_operators is not None
        )
        if new_program:
            proposal = record_operator_scope(
                self.runtime.model,
                stable_tensors=(program.candidate,),
                scope_kind="adaptive proposal",
            )
            physics = record_operator_scope(
                self.runtime.model,
                stable_tensors=(
                    program.counter, program.time_step,
                ),
                scope_kind="adaptive physics",
            )
            active: Any | None = proposal
            resolved = False
            proposal.__enter__()

            def resolve() -> None:
                nonlocal active, resolved
                if resolved:
                    raise RuntimeError("sub_step.resolve_dt() called more than once")
                # Relinquish ownership before exit: a failing transactional
                # rollback must never be exited a second time by the outer
                # exception path.
                current, active = active, None
                if current is not proposal:
                    raise RuntimeError("adaptive proposal recording is not active")
                proposal.__exit__(None, None, None)
                resolved = True
                physics.__enter__()
                active = physics

            frame = AdaptiveSubstepFrame(program.frame, resolve)
            try:
                yield frame
            except BaseException as exc:
                current, active = active, None
                if current is not None:
                    current.__exit__(type(exc), exc, exc.__traceback__)
                raise
            if not resolved:
                current, active = active, None
                if current is proposal:
                    proposal.__exit__(None, None, None)
                raise RuntimeError(
                    "adaptive substep must call sub_step.resolve_dt() exactly "
                    "once between dt proposal and physics"
                )
            current, active = active, None
            if current is not physics:
                raise RuntimeError("adaptive physics recording is not active")
            physics.__exit__(None, None, None)
            if proposal.program is None or physics.program is None:
                raise RuntimeError("adaptive substep recording did not complete")
            program.install(proposal.program, physics.program)
            programs[self.key] = program
        elif not installed:
            raise RuntimeError("cached adaptive substep program is not installed")
        step = self.runtime.model._execution.active_step
        if step is None:
            raise RuntimeError("adaptive substeps require @managed_step")
        if self.runtime.model.world_size > 1:
            from hydroforge.execution.step import (
                _DistributedStepEvent, _DistributedStepKind,
            )
            step.synchronize_distributed(_DistributedStepEvent(
                _DistributedStepKind.SUBSTEP,
            ))
        self.completed = program.execute(self.duration)
        step.completed_substeps = self.completed


class _PredicateScope:
    def __init__(
        self,
        runtime: SubstepRuntime,
        *,
        maximum_steps: int,
    ) -> None:
        if type(maximum_steps) is not int:
            raise TypeError("predicate loop maximum_steps must be an exact int")
        if maximum_steps < 1:
            raise ValueError("predicate loop maximum_steps must be positive")
        self.runtime = runtime
        self.maximum_steps = maximum_steps

    def __iter__(self) -> Iterator[PredicateLoopFrame]:
        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.program import PredicateLoopProgram
        from hydroforge.kernels.context import active_operator_recorder
        from torch.utils._python_dispatch import _disable_current_modes

        parent = active_operator_recorder()
        if parent is None:
            raise RuntimeError(
                "predicate loops must be nested directly inside a compiled "
                "fixed substep scope"
            )
        if parent.scope_kind != "fixed":
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError(
                "predicate loops are supported only directly inside a fixed "
                f"substep; found {parent.scope_kind!r} operator scope"
            )
        program = PredicateLoopProgram(
            self.runtime.model,
            maximum_steps=self.maximum_steps,
        )
        recording = record_operator_scope(
            self.runtime.model,
            stable_tensors=(
                program.predicate,
                program.counter,
                program.continue_flag,
            ),
            scope_kind="predicate",
        )
        try:
            # The child recorder replaces, rather than stacks on, the parent
            # TorchDispatchMode.  Otherwise every child ATen operator would be
            # intercepted a second time by the outer recorder.
            with _disable_current_modes(), recording:
                yield PredicateLoopFrame(
                    index=program.counter,
                    continue_flag=program.predicate,
                )
            if recording.program is None:
                raise RuntimeError("predicate loop recording did not complete")
            program.install(recording.program)
            parent.record_predicate_loop(program)
        except BaseException:
            program.close()
            raise


class SubstepRuntime:
    """Declare compiled loops as ordinary readable Python ``for`` scopes.

    A scope body is entered once for each managed-method specialization to
    build the operator IR.  Later outer steps skip the Python body and replay
    the cached device program.  Registered-kernel identity and intercepted
    ATen operators define the IR; Python function names have no execution
    meaning.
    """

    def __init__(self, model: AbstractModel) -> None:
        self.model = model

    @property
    def requested_sub_steps(self) -> int:
        """Return the requested fixed count, defaulting to one."""

        step = self.model._execution.active_step
        if step is None:
            raise RuntimeError(
                "requested_sub_steps is available only inside @managed_step"
            )
        raw = getattr(step, "requested_sub_steps", None)
        return 1 if raw is None else raw

    def fixed(
        self, *, count: object = _MISSING_COUNT, specialization: Any = None,
        defer_final: bool = False,
    ) -> _FixedScope:
        """Declare a fixed loop after decoding the shared count ABI.

        Count-producing device kernels may return
        :data:`INVALID_SUBSTEP_COUNT` to report an invalid or overflowing
        result without performing an unsafe integer cast.
        """
        if type(defer_final) is not bool:
            raise TypeError("defer_final must be an exact bool")
        step = self.model._execution.active_step
        requested = (
            None if step is None else getattr(step, "requested_sub_steps", None)
        )
        if count is _MISSING_COUNT:
            if step is None:
                raise RuntimeError("compiled substeps require @managed_step")
            count = self.requested_sub_steps
        elif requested is not None:
            raise ValueError(
                "computed fixed substep count conflicts with explicit "
                "num_sub_steps request"
            )
        if type(count) is not int:
            raise TypeError("fixed substep count must be an int")
        if count < 1:
            raise ValueError("fixed substep count must be positive")
        if count == INVALID_SUBSTEP_COUNT:
            raise RuntimeError(
                "fixed substep count received HydroForge's reserved "
                "invalid-count sentinel"
            )
        if count > INVALID_SUBSTEP_COUNT:
            raise ValueError("fixed substep count must fit in a signed int32")
        if step is None:
            raise RuntimeError("compiled substeps require @managed_step")
        duration, key = self._claim_scope(
            kind="fixed",
            specialization=(
                _specialization_key(specialization),
                ("defer_final", defer_final),
            ),
        )
        return _FixedScope(
            self, key=key,
            count=count, duration=duration, defer_final=defer_final,
        )

    def adaptive(
        self, *, candidate_dt: torch.Tensor, dt: torch.Tensor,
        maximum_dt: float, maximum_steps: int,
        specialization: Any = None,
    ) -> _AdaptiveScope:
        step = self.model._execution.active_step
        if step is None:
            raise RuntimeError("compiled substeps require @managed_step")
        if getattr(step, "requested_sub_steps", None) is not None:
            raise ValueError(
                "adaptive substeps conflict with explicit num_sub_steps "
                "request; omit num_sub_steps for adaptive timestepping"
            )
        duration, key = self._claim_scope(
            kind="adaptive",
            specialization=_specialization_key(specialization),
        )
        return _AdaptiveScope(
            self, key=key, duration=duration,
            candidate_dt=candidate_dt, dt=dt,
            maximum_dt=maximum_dt, maximum_steps=maximum_steps,
        )

    def predicate(self, *, maximum_steps: int) -> _PredicateScope:
        """Declare a non-temporal loop controlled by a device scalar.

        The body must write ``frame.continue_flag`` on every iteration.  The
        loop executes at least once and stops when that scalar becomes zero or
        after ``maximum_steps`` iterations.  It must be nested inside the one
        lexical fixed substep owned by the managed step.
        """

        return _PredicateScope(
            self,
            maximum_steps=maximum_steps,
        )

    def _claim_scope(
        self, *, kind: str, specialization: Any,
    ) -> tuple[float, tuple[Any, ...]]:
        """Claim the managed method's cached compiled-program identity."""

        step = self.model._execution.active_step
        if step is None:
            raise RuntimeError("compiled substeps require @managed_step")
        duration, key = step.claim_substep_scope(
            kind=kind, specialization=specialization,
        )
        if (
            type(duration) is not float
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise RuntimeError(
                "active managed step has no valid compiler-owned time_step"
            )
        return duration, key

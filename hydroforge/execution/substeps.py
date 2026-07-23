"""Explicit model-authored compiled sub-step scopes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import torch


_MISSING_PROGRAM = object()


@dataclass(frozen=True, slots=True)
class SubstepFrame:
    """Compiler-owned scalar tensors visible only inside a sub-step body."""

    index: torch.Tensor
    dt: torch.Tensor
    midpoint: torch.Tensor


class AdaptiveSubstepFrame:
    """Adaptive frame with an explicit proposal/physics phase boundary."""

    __slots__ = ("index", "dt", "midpoint", "_resolve")

    def __init__(self, frame: SubstepFrame, resolve) -> None:
        self.index = frame.index
        self.dt = frame.dt
        self.midpoint = frame.midpoint
        self._resolve = resolve

    def resolve_dt(self) -> None:
        """End dt proposal and begin the physics operator region."""
        self._resolve()


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
        count: int, duration: float,
    ) -> None:
        if type(count) is not int:
            raise TypeError("fixed substep count must be an int")
        if count < 1:
            raise ValueError("fixed substep count must be positive")
        if type(duration) not in {int, float}:
            raise TypeError("fixed substep duration must be an int or float")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("fixed substep duration must be finite and positive")
        self.runtime = runtime
        self.key = key
        self.requested_count = int(count)
        self.duration = float(duration)
        self.completed = 0

    def __iter__(self) -> Iterator[SubstepFrame]:
        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.program import FixedSubstepProgram

        programs = self.runtime.model._execution.programs
        program = programs.get(self.key, _MISSING_PROGRAM)
        if program is _MISSING_PROGRAM:
            program = FixedSubstepProgram(self.runtime.model)
            with record_operator_scope(
                self.runtime.model,
                stable_tensors=(
                    program.count, program.counter,
                    program.weight, program.midpoint,
                ),
            ) as recording:
                yield program.frame
            if recording.program is None:
                raise RuntimeError("fixed substep recording did not complete")
            program.install(recording.program)
            programs[self.key] = program
        elif not isinstance(program, FixedSubstepProgram):
            raise RuntimeError("cached substep program has the wrong execution kind")
        elif program.operators is None:
            raise RuntimeError("cached fixed substep program is not installed")
        step = self.runtime.model._execution.active_step
        if step is None:
            raise RuntimeError("fixed substeps require @managed_step")
        if self.runtime.model.world_size > 1:
            step.synchronize_distributed(1)
        self.completed = program.execute(
            self.requested_count, self.duration,
        )
        step.completed_substeps = self.completed


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
            )
            physics = record_operator_scope(
                self.runtime.model,
                stable_tensors=(
                    program.counter, program.time_step, program.fraction,
                ),
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
            step.synchronize_distributed(1)
        self.completed = program.execute(self.duration)
        step.completed_substeps = self.completed


class SubstepRuntime:
    """Declare compiled loops as ordinary readable Python ``for`` scopes.

    A scope body is entered once for each managed-method specialization to
    build the operator IR.  Later outer steps skip the Python body and replay
    the cached device program.  Registered-kernel identity and intercepted
    ATen operators define the IR; Python function names have no execution
    meaning.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def fixed(
        self, *, count: int, specialization: Any = None,
    ) -> _FixedScope:
        if type(count) is not int:
            raise TypeError("fixed substep count must be an int")
        if count < 1:
            raise ValueError("fixed substep count must be positive")
        duration, key = self._claim_scope(
            kind="fixed", specialization=_specialization_key(specialization),
        )
        return _FixedScope(
            self, key=key,
            count=count, duration=duration,
        )

    def adaptive(
        self, *, candidate_dt: torch.Tensor, dt: torch.Tensor,
        maximum_dt: float, maximum_steps: int,
        specialization: Any = None,
    ) -> _AdaptiveScope:
        duration, key = self._claim_scope(
            kind="adaptive",
            specialization=_specialization_key(specialization),
        )
        return _AdaptiveScope(
            self, key=key, duration=duration,
            candidate_dt=candidate_dt, dt=dt,
            maximum_dt=maximum_dt, maximum_steps=maximum_steps,
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

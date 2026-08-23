"""Cached outer-step lifecycle and streaming-statistics coordination."""

from __future__ import annotations

import inspect
from hashlib import sha256
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from functools import wraps
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import torch
import torch.distributed as dist
from pydantic import Field, PrivateAttr, ValidationInfo, model_validator

from hydroforge.contracts.events import emit
from hydroforge.contracts.errors import (
    ResourceCleanupError,
    distributed_failure_error,
)
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.devices import devices_match
from hydroforge.contracts.temporal import (
    DateLike,
    SimulationStep,
    date_calendar,
    timedelta_microseconds,
)
from hydroforge.execution.windows import (
    StatisticsWindowController,
    WindowDecision,
)

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


_F = TypeVar("_F", bound=Callable[..., Any])


_ACTIVE_MANAGED_STEP: ContextVar[_StepRuntime | None] = ContextVar(
    "hydroforge_active_managed_step",
    default=None,
)


def _managed_step_active() -> bool:
    """Return whether the caller is inside one executing managed step."""

    return _ACTIVE_MANAGED_STEP.get() is not None


class _DistributedStepKind(IntEnum):
    ABORT = 0
    SUBSTEP = 1
    USER_STEP_COMPLETE = 2
    STEP_FINALIZED = 3
    BEGIN = 4


@dataclass(frozen=True, slots=True)
class _DistributedStepEvent:
    """One logical managed-step handshake before int64 wire encoding."""

    kind: int
    signature: tuple[int, int, int] = (0, 0, 0)
    failed: bool = False

    def wire(self, sequence: int) -> tuple[int, int, int, int, int]:
        """Encode the stable five-int process-group protocol."""

        if self.failed:
            return sequence, -1, 0, 0, 0
        return sequence, int(self.kind), *self.signature


def synchronize_collective(
    kind: int,
    signature: tuple[int, int, int],
) -> None:
    """Synchronize an eager framework collective with its managed step."""

    step = _ACTIVE_MANAGED_STEP.get()
    if step is not None:
        step.synchronize_distributed(_DistributedStepEvent(kind, signature))


@dataclass
class _StepState:
    elapsed: float = 0.0
    start_time: Any = None
    pending_outer_first: bool = False


class _StepRuntime:
    """Private state for one managed outer step."""

    def __init__(self, model: AbstractModel, execution: Any) -> None:
        self.model = model
        self.execution = execution
        topology = model._process_topology
        self.world_size = topology.world_size
        self.rank = topology.rank
        self.schedule = getattr(model, "simulation_schedule", None)
        self.statistics = execution.statistics
        self.state = _StepState()
        plan = model._statistics_plan
        self.controller = (
            None if plan is None else StatisticsWindowController(plan, self.schedule)
        )
        self._distributed_sequence = 0
        self._distributed_terminal = False
        self._distributed_input: torch.Tensor | None = None
        self._distributed_outputs: tuple[torch.Tensor, ...] = ()
        self.stat_is_last = True
        self.stat_is_outer_last = True
        self.scheduled_step: SimulationStep | None = None
        self.requested_sub_steps: int | None = None

    def prepare_invocation(self) -> None:
        """Reset and validate the rank-synchronous managed-step protocol."""

        self._distributed_sequence = 0
        self._distributed_terminal = False
        world_size = self.world_size
        if world_size == 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "multi-rank managed steps require an initialized "
                "torch.distributed process group"
            )
        backend = str(dist.get_backend()).lower()
        accelerator_collective = "nccl" in backend or "xccl" in backend
        sync_device = (
            self.execution.device
            if accelerator_collective
            else torch.device("cpu")
        )
        if (
            self._distributed_input is None
            or not devices_match(self._distributed_input.device, sync_device)
            or len(self._distributed_outputs) != world_size
        ):
            self._distributed_input = torch.empty(
                5,
                dtype=torch.int64,
                device=sync_device,
            )
            self._distributed_outputs = tuple(
                torch.empty_like(self._distributed_input) for _ in range(world_size)
            )

    def synchronize_distributed(
        self,
        event: _DistributedStepEvent,
    ) -> None:
        """Match the next rank event or propagate a peer's local failure."""

        if self.world_size == 1 or self._distributed_terminal:
            return
        source = self._distributed_input
        wire = event.wire(self._distributed_sequence)
        for index, value in enumerate(wire):
            source[index] = value
        dist.all_gather(list(self._distributed_outputs), source)
        observed = tuple(
            tuple(map(int, value.tolist())) for value in self._distributed_outputs
        )
        self._distributed_sequence += 1
        failed_ranks = tuple(
            rank for rank, value in enumerate(observed) if value[1] < 0
        )
        if failed_ranks or len(set(observed)) != 1:
            self._distributed_terminal = True
            if event.failed:
                return
            if failed_ranks:
                raise RuntimeError(
                    "managed step failed on peer rank(s) before distributed "
                    f"event {self._distributed_sequence - 1}: {failed_ranks}"
                )
            raise RuntimeError(
                "managed-step distributed event or collective ABI differs "
                "across ranks: "
                f"{observed}"
            )

    def abort_distributed(self) -> None:
        """Publish a caught local failure at the next synchronization event."""

        self.synchronize_distributed(
            _DistributedStepEvent(
                _DistributedStepKind.ABORT,
                failed=True,
            )
        )

    def snapshot_state(self) -> tuple[tuple[Any, ...], tuple[Any, Any] | None]:
        state = self.state
        local = (
            state.elapsed,
            state.start_time,
            state.pending_outer_first,
        )
        controller = (
            None if self.controller is None else self.controller.snapshot_state()
        )
        return local, controller

    def restore_snapshot_state(
        self,
        snapshot: tuple[tuple[Any, ...], tuple[Any, Any] | None],
    ) -> None:
        local, controller = snapshot
        (
            self.state.elapsed,
            self.state.start_time,
            self.state.pending_outer_first,
        ) = local
        if self.controller is not None and controller is not None:
            self.controller.restore_snapshot_state(controller)

    def begin(
        self,
        *,
        current_time: Any,
        time_step: timedelta | None,
        output_enabled: bool | None,
        num_sub_steps: int | None = None,
        program_owner: _ManagedStepDescriptor,
        scheduled_step: SimulationStep | None,
    ) -> _StepRuntime:
        state = self.state
        self.current_time = current_time
        self.requested_sub_steps = num_sub_steps
        self._substep_scope_claimed = False
        self._outer_scope_count = 0
        self._pending_outer_scopes = 0
        self.completed_substeps = None
        self._substep_program_owner = program_owner
        self.scheduled_step = scheduled_step
        if scheduled_step is not None:
            time_step = scheduled_step.end - scheduled_step.start
        microseconds = timedelta_microseconds(time_step, label="time_step")
        self.duration = time_step
        self.time_step = microseconds / 1_000_000
        spinup = scheduled_step is not None and scheduled_step.is_spin_up
        self.spinup = spinup
        if output_enabled is None:
            output_enabled = not spinup
        if spinup:
            self.stat_is_first = self.stat_is_last = False
            outer_first = outer_last = False
            output_enabled = False
        elif self.controller is not None:
            if scheduled_step is None:
                decision = WindowDecision(
                    bool(output_enabled),
                    True,
                    True,
                    True,
                    True,
                )
            else:
                decision = self.controller.resolve(
                    step=scheduled_step,
                    output_enabled=output_enabled,
                )
            output_enabled = decision.output_enabled
            self.stat_is_first = decision.first
            self.stat_is_last = decision.last
            outer_first = decision.outer_first
            outer_last = decision.outer_last
        else:
            self.stat_is_first = self.stat_is_last = True
            outer_first = outer_last = True
        if outer_first and not self.stat_is_last:
            state.pending_outer_first = True
            outer_first = False
        if state.pending_outer_first and self.stat_is_last:
            outer_first = True
            state.pending_outer_first = False
        self.stat_is_outer_first = outer_first
        self.stat_is_outer_last = outer_last
        if self.stat_is_first:
            state.elapsed = 0.0
            state.start_time = current_time
        self.output_enabled = bool(output_enabled)
        self.run_statistics = self.statistics.enabled(self.output_enabled)
        self.total_weight = (
            0.0 if self.stat_is_first else state.elapsed
        ) + self.time_step
        self.flags = self._flags(
            self.stat_is_first,
            self.stat_is_last,
            self.stat_is_outer_first,
            self.stat_is_outer_last,
        )
        return self

    def commit_clock(self) -> None:
        """Publish the next model time only after the full step succeeds."""

        if self.scheduled_step is not None:
            next_time = self.scheduled_step.end
        elif self.current_time is not None:
            next_time = self.current_time + self.duration
        else:
            return
        self.model._set_runtime_current_time(next_time)

    def claim_substep_scope(
        self,
        *,
        kind: str,
        specialization: Any,
    ) -> tuple[float, tuple[Any, ...]]:
        """Claim this managed method's sole cached compilation scope."""

        self._substep_scope_claimed = True
        return self.time_step, (
            self._substep_program_owner,
            kind,
            specialization,
        )

    def claim_outer_scope(
        self,
        *,
        specialization: Any,
        site: Any = None,
    ) -> tuple[Any, ...]:
        """Return the stable cache key for one lexical outer operator scope."""

        if site is None:
            site = self._outer_scope_count
            self._outer_scope_count += 1
        return (
            self._substep_program_owner,
            "outer",
            site,
            specialization,
        )

    def begin_outer_scope_execution(self) -> None:
        """Track one outer scope until its recorded program has launched."""

        self._pending_outer_scopes += 1

    def complete_outer_scope_execution(self) -> None:
        """Mark one outer scope as fully recorded and launched."""

        self._pending_outer_scopes -= 1

    def require_outer_scopes_completed(self) -> None:
        """Reject ``break``/``return`` that silently skips an outer program."""

        if self._pending_outer_scopes:
            raise RuntimeError(
                "outer operator scope was exited before recording and launch "
                "completed; do not break or return from a step.outer() "
                "loop"
            )

    @property
    def program_owner(self) -> Any:
        """Stable owner identity used by explicitly named cached programs."""

        return self._substep_program_owner

    @staticmethod
    def _flags(first: bool, last: bool, outer_first: bool, outer_last: bool) -> int:
        return (
            int(first)
            | (int(last) << 1)
            | (int(outer_first) << 2)
            | (int(outer_last) << 3)
        )

    def sample_fixed(self, *, sub_step: int, num_sub_steps: int, weight: float) -> None:
        if self.run_statistics:
            self.statistics.sample(
                sub_step=sub_step,
                num_sub_steps=num_sub_steps,
                flags=self.flags,
                weight=weight,
                total_weight=self.total_weight,
            )
            self.state.elapsed += float(weight)

    def sample_adaptive(
        self, *, weight: float, first_event: bool, last_event: bool
    ) -> None:
        if self.run_statistics:
            flags = self._flags(
                self.stat_is_first and first_event,
                self.stat_is_last and last_event,
                self.stat_is_outer_first and last_event,
                self.stat_is_outer_last and last_event,
            )
            self.statistics.sample(
                sub_step=0,
                num_sub_steps=1,
                flags=flags,
                weight=weight,
                total_weight=self.total_weight,
            )
            self.state.elapsed += float(weight)

    def advance_device(self, elapsed: float) -> None:
        """Account for statistics already folded into a device-side loop."""
        if self.run_statistics:
            self.state.elapsed += float(elapsed)

    def finish(self) -> None:
        self.require_outer_scopes_completed()
        if self._substep_scope_claimed and self.completed_substeps is None:
            raise RuntimeError(
                "compiled substep scope was exited before recording and "
                "execution completed; do not break or return from a "
                "step.fixed/adaptive loop"
            )
        if self.run_statistics and not self._substep_scope_claimed:
            raise RuntimeError(
                "statistics were enabled but the managed step executed no "
                "step.fixed/adaptive scope"
            )
        if not self.stat_is_last:
            return
        if self.run_statistics:
            output_time = (
                self.state.start_time
                if self.state.start_time is not None
                else self.current_time
            )
            self.statistics.finish(output_time)
        self.state.elapsed = 0.0
        self.state.start_time = None


_FRAMEWORK_STEP_PARAMETERS = frozenset(
    {
        "output_enabled",
        "time_step",
        "num_sub_steps",
    }
)
_MANAGED_STEP_CONTEXT = "hydroforge_managed_step_runtime"


class ManagedStep(HydroForgeModel):
    """Validated physical-step identity passed to model-authored code.

    Driver inputs and schedule resolution are complete before this object is
    constructed.  Its private runtime references only expose authoring scopes
    owned by this exact invocation, so downstream physics no longer reaches
    through mutable ``AbstractModel`` active-step properties.
    """

    current_time: DateLike | None = None
    duration: timedelta
    output_enabled: bool = Field(strict=True)
    requested_sub_steps: int = Field(strict=True, ge=1, lt=(1 << 31) - 1)
    is_spin_up: bool = Field(strict=True)

    _substeps: Any = PrivateAttr()
    _outer: Any = PrivateAttr()

    @model_validator(mode="after")
    def _bind_framework_runtime(self, info: ValidationInfo) -> ManagedStep:
        binding = (
            info.context.get(_MANAGED_STEP_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if (
            not isinstance(binding, tuple)
            or len(binding) != 2
            or not isinstance(binding[1], _StepRuntime)
        ):
            raise ValueError(
                "ManagedStep is framework-owned and is created only by @managed_step"
            )
        model, runtime = binding
        from hydroforge.execution.outer import OuterRuntime
        from hydroforge.execution.substeps import SubstepRuntime

        self._substeps = SubstepRuntime(model, runtime)
        self._outer = OuterRuntime(model, runtime)
        return self

    @classmethod
    def _from_runtime(
        cls,
        model: AbstractModel,
        runtime: _StepRuntime,
    ) -> ManagedStep:
        """Bind the already validated step transaction without revalidation."""

        from hydroforge.execution.outer import OuterRuntime
        from hydroforge.execution.substeps import SubstepRuntime

        result = cls.model_construct(
            current_time=runtime.current_time,
            duration=runtime.duration,
            output_enabled=runtime.output_enabled,
            requested_sub_steps=(
                1
                if runtime.requested_sub_steps is None
                else runtime.requested_sub_steps
            ),
            is_spin_up=runtime.spinup,
        )
        result._substeps = SubstepRuntime(model, runtime)
        result._outer = OuterRuntime(model, runtime)
        return result

    def fixed(
        self,
        *,
        count: object = None,
        specialization: Any = None,
        final: Callable[[], None] | None = None,
    ):
        """Declare one validated fixed physical-substep scope."""

        return self._substeps.fixed(
            count=count,
            specialization=specialization,
            final=final,
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
    ):
        """Declare one validated adaptive physical-substep scope."""

        return self._substeps.adaptive(
            candidate_dt=candidate_dt,
            dt=dt,
            maximum_dt=maximum_dt,
            maximum_steps=maximum_steps,
            proposal=proposal,
            specialization=specialization,
        )

    def predicate(self, *, maximum_steps: int):
        """Declare one nested device-predicate loop."""

        return self._substeps.predicate(maximum_steps=maximum_steps)

    def outer(self, *, specialization: Any = None):
        """Declare one cached once-per-outer-step operator scope."""

        return self._outer.once(specialization=specialization)


class _ManagedStepEntryRequest(HydroForgeModel):
    """Validate invocation state before the runtime transaction is touched."""

    available: bool = Field(strict=True, exclude=True)

    @model_validator(mode="after")
    def _validate_available(self):
        if not self.available:
            raise ValueError("nested @managed_step calls are not supported")
        return self


class _ManagedScheduleRequest(HydroForgeModel):
    """Resolve one driver clock value against the validated schedule."""

    schedule: Any = Field(exclude=True)
    current_time: Any

    _step: SimulationStep | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve(self):
        if self.schedule is None:
            return self
        if self.current_time == self.schedule._end:
            raise ValueError("simulation schedule is exhausted")
        try:
            index = self.schedule._index_at(self.current_time)
        except KeyError:
            raise ValueError(
                f"model current_time {self.current_time!r} is not the start "
                "of a simulation schedule step"
            ) from None
        self._step = self.schedule._step_at_trusted(index)
        return self

    @property
    def step(self) -> SimulationStep | None:
        return self._step


class _ManagedStepRequest(HydroForgeModel):
    """The complete driver-owned input to one managed-step invocation."""

    time_step: timedelta | None = None
    time_step_supplied: bool = Field(strict=True, exclude=True)
    num_sub_steps: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        lt=(1 << 31) - 1,
    )
    output_enabled: bool | None = Field(default=None, strict=True)
    schedule_configured: bool = Field(strict=True, exclude=True)
    spinup: bool = Field(strict=True, exclude=True)
    statistics_configured: bool = Field(strict=True, exclude=True)
    current_time_available: bool = Field(strict=True, exclude=True)

    @model_validator(mode="after")
    def _validate_schedule_ownership(self) -> _ManagedStepRequest:
        if self.schedule_configured and self.time_step_supplied:
            raise ValueError(
                "time_step is derived from simulation_schedule and must not be provided"
            )
        if not self.schedule_configured and self.time_step is None:
            raise ValueError(
                "time_step is required when simulation_schedule is not configured"
            )
        if (
            self.time_step is not None
            and timedelta_microseconds(
                self.time_step,
                label="time_step",
            )
            <= 0
        ):
            raise ValueError("time_step must be positive")
        if self.spinup and self.output_enabled is True:
            raise ValueError("spin-up requires output_enabled=False")
        effective_output = (
            not self.spinup if self.output_enabled is None else self.output_enabled
        )
        if (
            self.statistics_configured
            and effective_output
            and not self.current_time_available
        ):
            raise ValueError(
                "current_time must be provided when statistics output is enabled"
            )
        return self


class _ManagedStepDeclaration(HydroForgeModel):
    """One validated model-authored managed-step function declaration."""

    function: Callable

    @model_validator(mode="after")
    def _validate_signature(self) -> _ManagedStepDeclaration:
        from hydroforge.execution.boundaries import is_between_steps_api

        if is_between_steps_api(self.function):
            raise ValueError("@managed_step cannot decorate a @between_steps method")
        parameters = tuple(inspect.signature(self.function).parameters.values())
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )
        if (
            len(positional) != 2
            or positional[0].name != "self"
            or positional[1].name != "step"
            or any(
                parameter.kind
                in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
                for parameter in parameters
            )
        ):
            raise ValueError(
                "@managed_step implementations must accept exactly "
                "(self, step); the decorator owns the driver-facing request"
            )
        return self


class _ManagedStepDescriptor:
    def __init__(self, declaration: _ManagedStepDeclaration) -> None:
        self.function = declaration.function
        self.protocol_name = (
            f"{self.function.__module__}.{self.function.__qualname__}"
        )
        self.protocol_code = int.from_bytes(
            sha256(self.protocol_name.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        ) & ((1 << 63) - 1)

    def compile(self, model: AbstractModel) -> _CompiledStepPolicy:
        return _CompiledStepPolicy(model, self)

    def validate_invocation(
        self,
        model: AbstractModel,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _ValidatedStepInvocation:
        """Validate driver input before model runtime materialization."""

        _ManagedStepEntryRequest(
            available=_ACTIVE_MANAGED_STEP.get() is None,
        )
        framework_values = dict(kwargs)
        if len(args) != 1:
            framework_values["unexpected_positional_arguments"] = args[1:]
        schedule = model.simulation_schedule
        current_time = (
            model._current_time
            if model._runtime_materialized
            else schedule.execution_start
            if schedule is not None
            else model.initial_time
        )
        scheduled_step = _ManagedScheduleRequest(
            schedule=schedule,
            current_time=current_time,
        ).step
        request = _ManagedStepRequest.model_validate(
            {
                **framework_values,
                "time_step_supplied": "time_step" in framework_values,
                "schedule_configured": schedule is not None,
                "spinup": (scheduled_step is not None and scheduled_step.is_spin_up),
                "statistics_configured": model._statistics_plan is not None,
                "current_time_available": current_time is not None,
            }
        )
        return _ValidatedStepInvocation(
            current_time=current_time,
            scheduled_step=scheduled_step,
            request=request,
        )


@dataclass(frozen=True, slots=True)
class _ValidatedStepInvocation:
    """Canonical driver request ready for the trusted step transaction."""

    current_time: Any
    scheduled_step: SimulationStep | None
    request: _ManagedStepRequest

    def distributed_signature(self) -> tuple[Any, ...]:
        """Return the exact driver and schedule identity shared by all ranks."""

        request = self.request
        time_step_us = (
            None
            if request.time_step is None
            else timedelta_microseconds(request.time_step, label="time_step")
        )
        return (
            _distributed_date_identity(self.current_time),
            _distributed_step_identity(self.scheduled_step),
            time_step_us,
            request.time_step_supplied,
            request.num_sub_steps,
            request.output_enabled,
            request.schedule_configured,
            request.spinup,
            request.statistics_configured,
            request.current_time_available,
        )


def _distributed_date_identity(value: DateLike | None) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (
        type(value).__module__,
        type(value).__qualname__,
        date_calendar(value),
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
    )


def _distributed_step_identity(
    step: SimulationStep | None,
) -> tuple[Any, ...] | None:
    if step is None:
        return None
    return (
        step.index,
        _distributed_date_identity(step.start),
        _distributed_date_identity(step.end),
        _distributed_date_identity(step.source_start),
        _distributed_date_identity(step.source_end),
        step.phase,
        step.spinup_cycle,
        step.source_index,
        step.reuse_index,
        step.reuse_count,
    )


class _CompiledStepPolicy:
    """Cached forcing, window, lifecycle and progress policy for one method."""

    def __init__(
        self,
        model: AbstractModel,
        descriptor: _ManagedStepDescriptor,
    ) -> None:
        self.model = model
        self.execution = model._execution
        self.descriptor = descriptor
        self._parameter_transaction = model._parameters.step_transaction
        self._execute_parameter_change_plan = model._execute_parameter_changes
        self._rank = model.rank
        if self._rank == 0:
            self._progress_start = model._progress_start
            self._progress_tick = model._progress_tick
            self._format_progress = model._format_progress
        else:
            self._progress_start = None
            self._progress_tick = None
            self._format_progress = None

    def _coordinate_failure(
        self,
        context: _StepRuntime,
        snapshot: Any,
        error: BaseException,
        *,
        poison: bool,
    ) -> BaseException:
        """Publish failure, restore temporal state, and return its full cause."""

        primary_error = error
        try:
            context.abort_distributed()
        except BaseException as coordination_error:
            error = ResourceCleanupError(
                "managed-step distributed failure propagation",
                (error, coordination_error),
            )
        if poison:
            self.execution.poison(error, phase="managed-step execution")
        try:
            context.restore_snapshot_state(snapshot)
        except BaseException as rollback_error:
            error = ResourceCleanupError(
                "managed-step temporal rollback",
                (error, rollback_error),
            )
        if error is primary_error:
            return primary_error
        return error

    def execute(self, invocation: _ValidatedStepInvocation) -> Any:
        model = self.model
        context = self.execution.step
        context.prepare_invocation()
        failure = self.execution.failure
        if failure is not None:
            error = self.execution.poisoned_error(failure)
            try:
                context.abort_distributed()
            except BaseException as coordination_error:
                combined = ResourceCleanupError(
                    "managed-step entry failure propagation",
                    (error, coordination_error),
                )
                raise combined from error
            raise error
        snapshot = context.snapshot_state()
        current_time = invocation.current_time
        scheduled_step = invocation.scheduled_step
        request = invocation.request
        entered_user_step = False
        try:
            self.execution.statistics.check_background_failures(current_time)
            context.begin(
                current_time=current_time,
                time_step=request.time_step,
                output_enabled=request.output_enabled,
                num_sub_steps=request.num_sub_steps,
                program_owner=self.descriptor,
                scheduled_step=scheduled_step,
            )
            managed = ManagedStep._from_runtime(model, context)
            context.synchronize_distributed(
                _DistributedStepEvent(
                    _DistributedStepKind.BEGIN,
                    (
                        self.descriptor.protocol_code,
                        (
                            -1
                            if scheduled_step is None
                            else scheduled_step.index
                        ),
                        (
                            0
                            if context.requested_sub_steps is None
                            else context.requested_sub_steps
                        )
                        << 6
                        | (context.flags << 2)
                        | (int(context.output_enabled) << 1)
                        | int(context.spinup),
                    ),
                ),
            )
            if self._rank == 0:
                self._progress_start()
            with self._parameter_transaction():
                if not context.spinup:
                    self._execute_parameter_change_plan(current_time)
                from hydroforge.kernels.registry import automatic_kernel_binding

                token = _ACTIVE_MANAGED_STEP.set(context)
                try:
                    binding_scope = automatic_kernel_binding(
                        self.execution.kernel_binding,
                    )
                    with binding_scope:
                        # From here onward model-authored outer Torch work and
                        # compiled physics may mutate address-stable state.
                        # There is no affordable generic rollback proof for an
                        # arbitrary failure, so the instance must fail closed.
                        entered_user_step = True
                        result = self.descriptor.function(model, managed)
                finally:
                    _ACTIVE_MANAGED_STEP.reset(token)
            context.synchronize_distributed(
                _DistributedStepEvent(
                    _DistributedStepKind.USER_STEP_COMPLETE,
                )
            )
            context.finish()
            self.execution.statistics.check_background_failures(current_time)
            if self._rank == 0:
                if self._progress_tick():
                    progress = self._format_progress()
                    emit(
                        model,
                        "progress",
                        "step.completed",
                        "Processed step",
                        current_time=current_time,
                        adaptive_time_step=context.completed_substeps,
                        progress=progress,
                    )
            context.synchronize_distributed(
                _DistributedStepEvent(
                    _DistributedStepKind.STEP_FINALIZED,
                )
            )
            context.commit_clock()
            return result
        except BaseException as error:
            resolved = self._coordinate_failure(
                context,
                snapshot,
                error,
                poison=entered_user_step or context.world_size > 1,
            )
            if resolved is error:
                raise
            raise resolved from error


def compile_step_policies(model: AbstractModel) -> None:
    """Compile every managed method after module initialization."""
    execution = model._execution
    execution.step = _StepRuntime(model, execution)
    seen: set[str] = set()
    for cls in type(model).__mro__:
        for name, method in vars(cls).items():
            if name in seen:
                continue
            seen.add(name)
            descriptor = getattr(method, "__hydroforge_managed_step__", None)
            if descriptor is not None:
                execution.step_policies[descriptor] = descriptor.compile(model)


def managed_step(function: _F) -> _F:
    """Compile step lifecycle once; the hot wrapper performs direct lookups."""
    declaration = _ManagedStepDeclaration(function=function)
    descriptor = _ManagedStepDescriptor(declaration)

    @wraps(function)
    def wrapper(*args, **kwargs):
        model = args[0] if args else kwargs["self"]
        invocation: _ValidatedStepInvocation | None = None
        validation_error: BaseException | None = None
        try:
            invocation = descriptor.validate_invocation(model, args, kwargs)
        except BaseException as error:
            validation_error = error
        if model.world_size > 1:
            validation_failures = model._gather_distributed_failures(
                validation_error,
                phase=(
                    "managed-step.invocation:"
                    f"{descriptor.protocol_name}"
                ),
                signature=(
                    None
                    if invocation is None
                    else (
                        model._runtime_materialized,
                        *invocation.distributed_signature(),
                    )
                ),
            )
            if any(
                failure is not None for failure in validation_failures
            ):
                if validation_error is not None:
                    raise validation_error
                raise distributed_failure_error(
                    "distributed managed-step invocation validation",
                    validation_failures,
                )
        elif validation_error is not None:
            raise validation_error
        model._ensure_runtime_materialized()
        return model._execution.step_policies[descriptor].execute(
            cast(_ValidatedStepInvocation, invocation),
        )

    authored = inspect.signature(function)
    parameters = [next(iter(authored.parameters.values()))]
    framework_options = (
        inspect.Parameter(
            "time_step",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=timedelta | None,
        ),
        inspect.Parameter(
            "num_sub_steps",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=int | None,
        ),
        inspect.Parameter(
            "output_enabled",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=bool | None,
        ),
    )
    parameters.extend(framework_options)
    wrapper.__signature__ = authored.replace(parameters=parameters)  # type: ignore[attr-defined]
    setattr(wrapper, "__hydroforge_managed_step__", descriptor)
    return cast(_F, wrapper)

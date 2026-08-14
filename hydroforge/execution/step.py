"""Cached outer-step lifecycle and streaming-statistics coordination."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import torch
import torch.distributed as dist

from hydroforge.contracts.events import emit
from hydroforge.contracts.temporal import (
    SimulationStep,
    timedelta_microseconds,
)
from hydroforge.execution.parameters import (
    ParameterChangeEffect, ParameterPlanRuntime,
)
from hydroforge.execution.windows import StatisticsWindowController

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


_F = TypeVar("_F", bound=Callable[..., Any])


_ACTIVE_MANAGED_STEP: ContextVar[_StepRuntime | None] = ContextVar(
    "hydroforge_active_managed_step", default=None,
)


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

    def __post_init__(self) -> None:
        kind = self.kind
        if isinstance(kind, _DistributedStepKind):
            kind = int(kind)
            object.__setattr__(self, "kind", kind)
        if type(kind) is not int or kind < 0:
            raise ValueError("distributed managed-step event kind must be >= 0")
        if (
            not isinstance(self.signature, tuple)
            or len(self.signature) != 3
            or any(
                type(value) is not int or value < 0
                for value in self.signature
            )
        ):
            raise ValueError(
                "distributed managed-step signature must contain three "
                "non-negative exact ints"
            )
        if type(self.failed) is not bool:
            raise TypeError("distributed managed-step failed flag must be bool")

    def wire(self, sequence: int) -> tuple[int, int, int, int, int]:
        """Encode the stable five-int process-group protocol."""

        if type(sequence) is not int or sequence < 0:
            raise ValueError("distributed managed-step sequence must be >= 0")
        if self.failed:
            return sequence, -1, 0, 0, 0
        return sequence, self.kind, *self.signature


def synchronize_collective(
    kind: int, signature: tuple[int, int, int],
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
        plan = getattr(model, "statistics_plan", None)
        self.controller = (
            None if plan is None
            else StatisticsWindowController(plan, self.schedule)
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
        sync_device = (
            self.execution.device if "nccl" in backend
            else torch.device("cpu")
        )
        if (
            self._distributed_input is None
            or self._distributed_input.device != sync_device
            or len(self._distributed_outputs) != world_size
        ):
            self._distributed_input = torch.empty(
                5, dtype=torch.int64, device=sync_device,
            )
            self._distributed_outputs = tuple(
                torch.empty_like(self._distributed_input)
                for _ in range(world_size)
            )

    def synchronize_distributed(
        self, event: _DistributedStepEvent,
    ) -> None:
        """Match the next rank event or propagate a peer's local failure."""

        if self.world_size == 1 or self._distributed_terminal:
            return
        if not isinstance(event, _DistributedStepEvent):
            raise TypeError(
                "distributed managed-step synchronization requires an event"
            )
        source = self._distributed_input
        if source is None or not self._distributed_outputs:
            raise RuntimeError(
                "distributed managed-step synchronization was not prepared"
            )
        wire = event.wire(self._distributed_sequence)
        for index, value in enumerate(wire):
            source[index] = value
        dist.all_gather(list(self._distributed_outputs), source)
        observed = tuple(
            tuple(map(int, value.tolist()))
            for value in self._distributed_outputs
        )
        self._distributed_sequence += 1
        failed_ranks = tuple(
            rank for rank, value in enumerate(observed)
            if value[1] < 0
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

        self.synchronize_distributed(_DistributedStepEvent(
            _DistributedStepKind.ABORT, failed=True,
        ))

    def snapshot_state(self) -> tuple[tuple[Any, ...], tuple[Any, Any] | None]:
        state = self.state
        local = (
            state.elapsed,
            state.start_time,
            state.pending_outer_first,
        )
        controller = (
            None if self.controller is None
            else self.controller.snapshot_state()
        )
        return local, controller

    def restore_snapshot_state(
        self, snapshot: tuple[tuple[Any, ...], tuple[Any, Any] | None],
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
        self, *, current_time: Any, time_step: timedelta | None,
        output_enabled: bool | None,
        num_sub_steps: int | None = None,
        program_owner: _ManagedStepDescriptor,
    ) -> _StepRuntime:
        state = self.state
        if not isinstance(program_owner, _ManagedStepDescriptor):
            raise TypeError(
                "managed-step program owner must be a compiled descriptor"
            )
        self.current_time = current_time
        self.requested_sub_steps = _validate_requested_sub_steps(num_sub_steps)
        self._substep_scope_claimed = False
        self._outer_scope_count = 0
        self.completed_substeps = None
        self._substep_program_owner = program_owner
        scheduled_step = self._resolve_model_schedule(current_time=current_time)
        self.scheduled_step = scheduled_step
        if scheduled_step is not None:
            if time_step is not None:
                raise TypeError(
                    "time_step is derived from simulation_schedule and must "
                    "not be provided"
                )
            time_step = scheduled_step.end - scheduled_step.start
        elif time_step is None:
            raise TypeError(
                "time_step is required when simulation_schedule is not "
                "configured"
            )
        microseconds = timedelta_microseconds(time_step, label="time_step")
        if microseconds <= 0:
            raise ValueError("time_step must be positive")
        self.duration = time_step
        self.time_step = microseconds / 1_000_000
        spinup = scheduled_step is not None and scheduled_step.is_spin_up
        self.spinup = spinup
        if output_enabled is None:
            output_enabled = not spinup
        elif type(output_enabled) is not bool:
            raise TypeError("output_enabled must be an exact bool when provided")
        if spinup and output_enabled:
            raise ValueError("spin-up requires output_enabled=False")
        if spinup:
            self.stat_is_first = self.stat_is_last = False
            outer_first = outer_last = False
            output_enabled = False
        elif self.controller is not None:
            if scheduled_step is None:
                raise RuntimeError(
                    "statistics controller requires a scheduled model step"
                )
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
        if self.run_statistics and current_time is None:
            raise ValueError(
                "current_time must be provided when statistics output is enabled"
            )
        self.total_weight = (
            (0.0 if self.stat_is_first else state.elapsed) + self.time_step
        )
        self.flags = self._flags(
            self.stat_is_first, self.stat_is_last,
            self.stat_is_outer_first, self.stat_is_outer_last,
        )
        return self

    def _resolve_model_schedule(
        self, *, current_time: Any,
    ) -> SimulationStep | None:
        """Resolve the current call from the sole runtime clock."""

        schedule = self.schedule
        if schedule is None:
            return None
        if current_time == schedule.end:
            raise RuntimeError("simulation schedule is exhausted")
        try:
            index = schedule.index_at(current_time)
        except KeyError:
            raise ValueError(
                f"model current_time {current_time!r} is not the start of a "
                "simulation schedule step"
            ) from None
        step = schedule.step_at(index)
        return step

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
        self, *, kind: str, specialization: Any,
    ) -> tuple[float, tuple[Any, ...]]:
        """Claim this managed method's sole cached compilation scope."""

        if kind not in {"fixed", "adaptive"}:
            raise ValueError(f"unknown compiled substep kind {kind!r}")
        if self._substep_scope_claimed:
            raise RuntimeError(
                "a managed step may execute only one substep scope; combine "
                "the operators into one lexical loop"
            )
        self._substep_scope_claimed = True
        return self.time_step, (
            self._substep_program_owner, kind, specialization,
        )

    def claim_outer_scope(
        self, *, specialization: Any, site: Any = None,
    ) -> tuple[Any, ...]:
        """Return the stable cache key for one lexical outer operator scope."""

        if site is None:
            site = self._outer_scope_count
            self._outer_scope_count += 1
        return (
            self._substep_program_owner, "outer", site, specialization,
        )

    @property
    def program_owner(self) -> Any:
        """Stable owner identity used by explicitly named cached programs."""

        return self._substep_program_owner

    @staticmethod
    def _flags(first: bool, last: bool, outer_first: bool, outer_last: bool) -> int:
        return (
            int(first) | (int(last) << 1)
            | (int(outer_first) << 2) | (int(outer_last) << 3)
        )

    def sample_fixed(self, *, sub_step: int, num_sub_steps: int, weight: float) -> None:
        if self.run_statistics:
            self.statistics.sample(
                sub_step=sub_step, num_sub_steps=num_sub_steps, flags=self.flags,
                weight=weight, total_weight=self.total_weight,
            )
            self.state.elapsed += float(weight)

    def sample_adaptive(self, *, weight: float, first_event: bool, last_event: bool) -> None:
        if self.run_statistics:
            flags = self._flags(
                self.stat_is_first and first_event,
                self.stat_is_last and last_event,
                self.stat_is_outer_first and last_event,
                self.stat_is_outer_last and last_event,
            )
            self.statistics.sample(
                sub_step=0, num_sub_steps=1, flags=flags,
                weight=weight, total_weight=self.total_weight,
            )
            self.state.elapsed += float(weight)

    def advance_device(self, elapsed: float) -> None:
        """Account for statistics already folded into a device-side loop."""
        if self.run_statistics:
            self.state.elapsed += float(elapsed)

    def finish(self) -> None:
        if self._substep_scope_claimed and self.completed_substeps is None:
            raise RuntimeError(
                "compiled substep scope was exited before recording and "
                "execution completed; do not break or return from a "
                "self.substeps.fixed/adaptive loop"
            )
        if self.run_statistics and not self._substep_scope_claimed:
            raise RuntimeError(
                "statistics were enabled but the managed step executed no "
                "self.substeps.fixed/adaptive scope"
            )
        if not self.stat_is_last:
            return
        if self.run_statistics:
            output_time = (
                self.state.start_time
                if self.state.start_time is not None else self.current_time
            )
            if output_time is None:
                raise ValueError(
                    "current_time must be provided when finalizing statistics"
                )
            self.statistics.finish(output_time)
        self.state.elapsed = 0.0
        self.state.start_time = None


_MISSING = object()
_FRAMEWORK_STEP_PARAMETERS = frozenset({
    "current_time",
    "spinup",
    "output_enabled",
    "time_step",
    "num_sub_steps",
})


def _validate_requested_sub_steps(value: Any) -> int | None:
    """Validate the driver-owned fixed sub-step request before user code."""

    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("num_sub_steps must be an exact int or None")
    if value < 1:
        raise ValueError("num_sub_steps must be positive when provided")
    from hydroforge.execution.substeps import INVALID_SUBSTEP_COUNT

    if value >= INVALID_SUBSTEP_COUNT:
        raise ValueError(
            "num_sub_steps must be smaller than HydroForge's reserved "
            "invalid-count sentinel"
        )
    return value


class _StepCallLayout:
    """Validate that model-authored signatures contain only model inputs."""

    def __init__(self, function: Callable) -> None:
        self.signature = inspect.signature(function)
        parameters = tuple(self.signature.parameters.values())
        names = {parameter.name for parameter in parameters}
        if "self" not in names:
            raise TypeError("@managed_step requires a self parameter")
        owned = names.intersection(_FRAMEWORK_STEP_PARAMETERS)
        if owned:
            raise TypeError(
                "@managed_step framework parameters must not appear in the "
                f"model method signature: {sorted(owned)}"
            )

    def validate_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Validate model arguments before opening the managed lifecycle."""

        self.signature.bind(*args, **kwargs)


class _ManagedStepDescriptor:
    def __init__(self, function: Callable) -> None:
        self.function = function
        self.layout = _StepCallLayout(function)

    def compile(self, model: AbstractModel) -> _CompiledStepPolicy:
        return _CompiledStepPolicy(model, self)


class _CompiledStepPolicy:
    """Cached forcing, window, lifecycle and progress policy for one method."""

    def __init__(
        self, model: AbstractModel, descriptor: _ManagedStepDescriptor,
    ) -> None:
        self.model = model
        self.execution = model._execution
        self.descriptor = descriptor
        parameters = getattr(model, "_parameters", None)
        if parameters is not None and not isinstance(
            parameters, ParameterPlanRuntime,
        ):
            raise TypeError(
                "model._parameters must be ParameterPlanRuntime when present"
            )
        self._parameter_transaction = (
            nullcontext if parameters is None else parameters.step_transaction
        )
        self._execute_parameter_change_plan = model.execute_parameter_change_plan
        self._rank = model.rank
        if self._rank == 0:
            self._progress_start = getattr(model, "progress_start", None)
            self._progress_tick = getattr(model, "progress_tick", None)
            self._format_progress = getattr(model, "format_progress", None)
        else:
            self._progress_start = None
            self._progress_tick = None
            self._format_progress = None

    def _coordinate_failure(
        self, context: _StepRuntime, snapshot: Any,
        error: BaseException, *, poison: bool,
    ) -> BaseException:
        """Publish failure, restore temporal state, and return its full cause."""

        primary_error = error
        try:
            context.abort_distributed()
        except BaseException as coordination_error:
            from hydroforge.contracts import ResourceCleanupError

            error = ResourceCleanupError(
                "managed-step distributed failure propagation",
                (error, coordination_error),
            )
        if poison:
            self.execution.poison(error, phase="managed-step execution")
        try:
            context.restore_snapshot_state(snapshot)
        except BaseException as rollback_error:
            from hydroforge.contracts import ResourceCleanupError

            error = ResourceCleanupError(
                "managed-step temporal rollback", (error, rollback_error),
            )
        if error is primary_error:
            return primary_error
        return error

    def execute(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        model = self.model
        if self.execution.active_step is not None:
            raise RuntimeError("nested @managed_step calls are not supported")
        context = self.execution.step
        context.prepare_invocation()
        try:
            self.execution.require_open()
        except BaseException as error:
            try:
                context.abort_distributed()
            except BaseException as coordination_error:
                from hydroforge.contracts import ResourceCleanupError

                combined = ResourceCleanupError(
                    "managed-step entry failure propagation",
                    (error, coordination_error),
                )
                raise combined from error
            raise
        snapshot = context.snapshot_state()
        try:
            kwargs = dict(kwargs)
            if "current_time" in kwargs:
                raise TypeError(
                    "current_time is owned by the managed-step runtime and "
                    "must not be provided"
                )
            if "spinup" in kwargs:
                raise TypeError(
                    "spinup is derived from simulation_schedule and must not "
                    "be provided"
                )
            output_enabled = kwargs.pop("output_enabled", None)
            if output_enabled is not None and type(output_enabled) is not bool:
                raise TypeError(
                    "output_enabled must be an exact bool when provided"
                )
            supplied_time_step = kwargs.pop("time_step", _MISSING)
            requested_sub_steps = _validate_requested_sub_steps(
                kwargs.pop("num_sub_steps", None),
            )
            self.descriptor.layout.validate_call(args, kwargs)
            if context.schedule is not None:
                if supplied_time_step is not _MISSING:
                    raise TypeError(
                        "time_step is derived from simulation_schedule and "
                        "must not be provided"
                    )
                time_step = None
            elif supplied_time_step is _MISSING:
                raise TypeError(
                    "time_step is required when simulation_schedule is not "
                    "configured"
                )
            else:
                time_step = supplied_time_step
            current_time = getattr(model, "current_time", None)
        except BaseException as error:
            resolved = self._coordinate_failure(
                context, snapshot, error, poison=context.world_size > 1,
            )
            if resolved is error:
                raise
            raise resolved from error
        entered_user_step = False
        try:
            self.execution.statistics.check_background_failures(current_time)
            context.begin(
                current_time=current_time,
                time_step=time_step,
                output_enabled=output_enabled,
                num_sub_steps=requested_sub_steps,
                program_owner=self.descriptor,
            )
            context.synchronize_distributed(
                _DistributedStepEvent(
                    _DistributedStepKind.BEGIN,
                    (
                        int(context.spinup), int(context.output_enabled),
                        (
                            0 if context.requested_sub_steps is None
                            else context.requested_sub_steps
                        ) << 4 | context.flags,
                    ),
                ),
            )
            if self._rank == 0:
                if self._progress_start is None:
                    raise RuntimeError(
                        "rank-zero managed models must define progress_start()"
                    )
                self._progress_start()
            with self._parameter_transaction():
                if not context.spinup:
                    parameter_effect = self._execute_parameter_change_plan(
                        current_time,
                    )
                    if not isinstance(parameter_effect, ParameterChangeEffect):
                        raise TypeError(
                            "execute_parameter_change_plan() must return "
                            "ParameterChangeEffect"
                        )
                self.execution.active_step = context
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
                        result = self.descriptor.function(
                            *args, **kwargs,
                        )
                finally:
                    _ACTIVE_MANAGED_STEP.reset(token)
            if self._rank == 0:
                if self._progress_tick is None or self._format_progress is None:
                    raise RuntimeError(
                        "rank-zero managed models must define progress_tick() "
                        "and format_progress()"
                    )
                self._progress_tick()
                progress = self._format_progress()
                emit(
                    model, "progress", "step.completed", "Processed step",
                    current_time=current_time,
                    adaptive_time_step=context.completed_substeps,
                    progress=progress,
                )
            context.synchronize_distributed(_DistributedStepEvent(
                _DistributedStepKind.USER_STEP_COMPLETE,
            ))
            context.finish()
            self.execution.statistics.check_background_failures(current_time)
            context.synchronize_distributed(_DistributedStepEvent(
                _DistributedStepKind.STEP_FINALIZED,
            ))
            context.commit_clock()
            return result
        except BaseException as error:
            resolved = self._coordinate_failure(
                context, snapshot, error,
                poison=entered_user_step or context.world_size > 1,
            )
            if resolved is error:
                raise
            raise resolved from error
        finally:
            self.execution.active_step = None


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
    from hydroforge.execution.boundaries import is_between_steps_api

    if is_between_steps_api(function):
        raise TypeError("@managed_step cannot decorate a @between_steps method")
    descriptor = _ManagedStepDescriptor(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        model = args[0] if args else kwargs["self"]
        return model._execution.step_policies[descriptor].execute(args, kwargs)

    authored = inspect.signature(function)
    parameters = list(authored.parameters.values())
    insertion = next(
        (
            index for index, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
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
    parameters[insertion:insertion] = framework_options
    wrapper.__signature__ = authored.replace(parameters=parameters)  # type: ignore[attr-defined]
    setattr(wrapper, "__hydroforge_managed_step__", descriptor)
    return cast(_F, wrapper)

"""Cached outer-step lifecycle and streaming-statistics coordination."""

from __future__ import annotations

import inspect
import math
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import torch
import torch.distributed as dist

from hydroforge.contracts.events import emit
from hydroforge.contracts.temporal import (
    StatisticsFlags,
    require_calendar,
)
from hydroforge.execution.parameters import (
    ParameterChangeEffect, ParameterPlanRuntime,
)
from hydroforge.execution.windows import StatisticsWindowController


_ACTIVE_MANAGED_STEP: ContextVar[_StepRuntime | None] = ContextVar(
    "hydroforge_active_managed_step", default=None,
)


def synchronize_collective(
    kind: int, signature: tuple[int, int, int],
) -> None:
    """Synchronize an eager framework collective with its managed step."""

    step = _ACTIVE_MANAGED_STEP.get()
    if step is not None:
        step.synchronize_distributed(kind, signature=signature)


@dataclass
class _StepState:
    elapsed: float = 0.0
    start_time: Any = None
    pending_outer_first: bool = False
    schedule_step: int | None = None
    output_active: bool = False


class _StepRuntime:
    """Private state for one managed outer step."""

    def __init__(self, model: Any, execution: Any) -> None:
        self.model = model
        self.execution = execution
        self.world_size = getattr(model, "world_size", 1)
        self.rank = getattr(model, "rank", 0)
        self.schedule = getattr(model, "simulation_schedule", None)
        self.output_start_time = getattr(model, "output_start_time", None)
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("model world_size must be an exact positive int")
        self.statistics = execution.statistics
        self.state = _StepState()
        plan = getattr(model, "statistics_plan", None)
        self.controller = (
            None if plan is None else StatisticsWindowController(plan)
        )
        self._distributed_sequence = 0
        self._distributed_terminal = False
        self._distributed_input: torch.Tensor | None = None
        self._distributed_outputs: tuple[torch.Tensor, ...] = ()
        self.stat_is_last = True
        self.stat_is_outer_last = True

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
        observed_world = dist.get_world_size()
        observed_rank = dist.get_rank()
        if observed_world != world_size or observed_rank != self.rank:
            raise RuntimeError(
                "model rank topology disagrees with the initialized process "
                f"group: model=({self.model.rank}, {world_size}), "
                f"group=({observed_rank}, {observed_world})"
            )
        backend = str(dist.get_backend()).lower()
        sync_device = (
            self.execution.device if "nccl" in backend
            else torch.device("cpu")
        )
        if (
            self._distributed_input is None
            or self._distributed_input.device != sync_device
            or len(self._distributed_outputs) != observed_world
        ):
            self._distributed_input = torch.empty(
                5, dtype=torch.int64, device=sync_device,
            )
            self._distributed_outputs = tuple(
                torch.empty_like(self._distributed_input)
                for _ in range(world_size)
            )

    def synchronize_distributed(
        self, kind: int, *, signature: tuple[int, int, int] = (0, 0, 0),
        failed: bool = False,
    ) -> None:
        """Match the next rank event or propagate a peer's local failure."""

        if self.world_size == 1 or self._distributed_terminal:
            return
        if type(kind) is not int or kind < 0:
            raise ValueError("distributed managed-step event kind must be >= 0")
        if (
            not isinstance(signature, tuple) or len(signature) != 3
            or any(type(value) is not int or value < 0 for value in signature)
        ):
            raise ValueError(
                "distributed managed-step signature must contain three "
                "non-negative exact ints"
            )
        source = self._distributed_input
        if source is None or not self._distributed_outputs:
            raise RuntimeError(
                "distributed managed-step synchronization was not prepared"
            )
        source[0] = self._distributed_sequence
        source[1] = -1 if failed else kind
        source[2] = 0 if failed else signature[0]
        source[3] = 0 if failed else signature[1]
        source[4] = 0 if failed else signature[2]
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
            if failed:
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

        self.synchronize_distributed(0, failed=True)

    def snapshot_state(self) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
        state = self.state
        local = (
            state.elapsed,
            state.start_time,
            state.pending_outer_first,
            state.schedule_step,
            state.output_active,
        )
        controller = (
            None if self.controller is None
            else self.controller.snapshot_state()
        )
        return local, controller

    def restore_snapshot_state(
        self, snapshot: tuple[tuple[Any, ...], dict[str, Any] | None],
    ) -> None:
        local, controller = snapshot
        (
            self.state.elapsed,
            self.state.start_time,
            self.state.pending_outer_first,
            self.state.schedule_step,
            self.state.output_active,
        ) = local
        if self.controller is not None and controller is not None:
            self.controller.restore_snapshot_state(controller)

    def begin(
        self, *, current_time: Any, time_step: float,
        output_enabled: bool,
        program_owner: _ManagedStepDescriptor,
        spinup: bool = False,
        override: StatisticsFlags | None = None,
    ) -> _StepRuntime:
        state = self.state
        if not isinstance(program_owner, _ManagedStepDescriptor):
            raise TypeError(
                "managed-step program owner must be a compiled descriptor"
            )
        self.current_time = current_time
        self._substep_scope_claimed = False
        self._outer_scope_count = 0
        self.completed_substeps = None
        self._substep_program_owner = program_owner
        if type(time_step) not in {int, float}:
            raise TypeError("time_step must be an int or float")
        self.time_step = float(time_step)
        if not math.isfinite(self.time_step) or self.time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        if type(spinup) is not bool:
            raise TypeError("spinup must be an exact bool")
        if spinup and output_enabled:
            raise ValueError("spin-up requires output_enabled=False")
        if spinup and state.schedule_step is not None:
            raise ValueError(
                "spin-up cannot resume after the main schedule starts"
            )
        if spinup and override is not None:
            raise ValueError("spin-up does not accept manual statistics flags")
        if spinup and self.schedule is not None:
            require_calendar(
                current_time, self.schedule.calendar, label="model current_time",
            )
        if not spinup and self.controller is None:
            self._validate_model_schedule(
                current_time=current_time,
            )
        if (
            self.output_start_time is not None and current_time is not None
            and current_time < self.output_start_time
        ):
            output_enabled = False
        if override is not None and not output_enabled and any((
            override.first, override.last,
            override.outer_first, override.outer_last,
        )):
            raise ValueError(
                "manual statistics flags must be false when output is disabled"
            )
        if spinup:
            self.stat_is_first = self.stat_is_last = False
            outer_first = outer_last = False
            output_enabled = False
        elif self.controller is not None:
            decision = self.controller.resolve(
                current_time=current_time,
                time_step=self.time_step,
                output_enabled=output_enabled,
                override=override,
            )
            if current_time >= self.schedule.start:
                state.schedule_step = self.schedule.index_at(current_time)
            output_enabled = decision.output_enabled
            flags = decision.flags
            self.stat_is_first = flags.first
            self.stat_is_last = flags.last
            outer_first = flags.outer_first
            outer_last = flags.outer_last
        elif override is not None:
            StatisticsWindowController._validate_override(override)
            self.stat_is_first = override.first
            self.stat_is_last = override.last
            outer_first = override.outer_first
            outer_last = override.outer_last
        else:
            self.stat_is_first = self.stat_is_last = True
            outer_first = outer_last = True
        state.output_active = bool(output_enabled)
        if override is None:
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

    def _validate_model_schedule(self, *, current_time: Any) -> None:
        """Advance a schedule cursor when statistics does not already own it."""

        schedule = self.schedule
        if schedule is None:
            return
        require_calendar(
            current_time, schedule.calendar, label="model current_time",
        )
        try:
            index = schedule.index_at(current_time)
        except KeyError as error:
            raise ValueError(
                f"current_time {current_time!r} is not a model schedule boundary"
            ) from error
        step = schedule.step_at(index)
        if abs(step.duration_seconds - self.time_step) > 1e-9:
            raise ValueError(
                f"time_step {self.time_step} differs from scheduled "
                f"duration {step.duration_seconds} at {current_time!r}"
            )
        previous = self.state.schedule_step
        expected = 0 if previous is None else previous + 1
        if index != expected:
            raise ValueError(
                f"model schedule moved from step {previous} to {index}; "
                f"expected {expected}"
            )
        self.state.schedule_step = index

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

    def claim_outer_scope(self, *, specialization: Any) -> tuple[Any, ...]:
        """Return the stable cache key for one lexical outer operator scope."""

        ordinal = self._outer_scope_count
        self._outer_scope_count += 1
        return (
            self._substep_program_owner, "outer", ordinal, specialization,
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


class _StepCallLayout:
    """Precomputed positional/default access for one step signature."""

    def __init__(self, function: Callable) -> None:
        parameters = tuple(inspect.signature(function).parameters.values())
        self.positions = {
            parameter.name: index for index, parameter in enumerate(parameters)
        }
        self.defaults = {
            parameter.name: parameter.default
            for parameter in parameters
            if parameter.default is not inspect.Parameter.empty
        }
        if "self" not in self.positions or "time_step" not in self.positions:
            raise TypeError("@managed_step requires self and time_step parameters")

    def get(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], name: str,
        default: Any = _MISSING,
    ) -> Any:
        if name in kwargs:
            return kwargs[name]
        position = self.positions.get(name)
        if position is not None and position < len(args):
            return args[position]
        if name in self.defaults:
            return self.defaults[name]
        if default is not _MISSING:
            return default
        raise TypeError(f"missing required step argument {name!r}")

    def replace(
        self, args: tuple[Any, ...], kwargs: dict[str, Any],
        name: str, value: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if name not in self.positions:
            return args, kwargs
        position = self.positions[name]
        if position < len(args):
            positional = list(args)
            positional[position] = value
            return tuple(positional), kwargs
        keywords = dict(kwargs)
        keywords[name] = value
        return args, keywords


class _ManagedStepDescriptor:
    def __init__(self, function: Callable) -> None:
        self.function = function
        self.layout = _StepCallLayout(function)

    def compile(self, model: Any) -> _CompiledStepPolicy:
        return _CompiledStepPolicy(model, self)


class _CompiledStepPolicy:
    """Cached forcing, window, lifecycle and progress policy for one method."""

    def __init__(self, model: Any, descriptor: _ManagedStepDescriptor) -> None:
        self.model = model
        self.execution = model._execution
        self.descriptor = descriptor
        self.layout = descriptor.layout
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
        layout = self.layout
        try:
            kwargs = dict(kwargs)
            flag_names = (
                "stat_is_first", "stat_is_last",
                "stat_is_outer_first", "stat_is_outer_last",
            )
            present = tuple(name for name in flag_names if name in kwargs)
            if present and len(present) != len(flag_names):
                missing = sorted(set(flag_names).difference(present))
                raise TypeError(
                    "manual statistics control requires all four flags; "
                    f"missing={missing}"
                )
            override = None
            if present:
                values = tuple(kwargs.pop(name) for name in flag_names)
                if any(type(value) is not bool for value in values):
                    raise TypeError("manual statistics flags must be bool values")
                override = StatisticsFlags(*values)
            spinup = kwargs.pop("spinup", False)
            if type(spinup) is not bool:
                raise TypeError("spinup must be an exact bool")
            current_time = layout.get(args, kwargs, "current_time", None)
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
                time_step=layout.get(args, kwargs, "time_step"),
                output_enabled=layout.get(args, kwargs, "output_enabled", True),
                spinup=spinup,
                program_owner=self.descriptor,
                override=override,
            )
            context.synchronize_distributed(
                4,
                signature=(
                    int(spinup), int(context.output_enabled), context.flags,
                ),
            )
            if self._rank == 0:
                if self._progress_start is None:
                    raise RuntimeError(
                        "rank-zero managed models must define progress_start()"
                    )
                self._progress_start()
            with self._parameter_transaction():
                if not spinup:
                    parameter_effect = self._execute_parameter_change_plan(
                        current_time,
                    )
                    if not isinstance(parameter_effect, ParameterChangeEffect):
                        raise TypeError(
                            "execute_parameter_change_plan() must return "
                            "ParameterChangeEffect"
                        )
                self.execution.active_step = context
                call_args, call_kwargs = layout.replace(
                    args, kwargs, "output_enabled", context.output_enabled,
                )
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
                            *call_args, **call_kwargs,
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
            context.synchronize_distributed(2)
            context.finish()
            self.execution.statistics.check_background_failures(current_time)
            context.synchronize_distributed(3)
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


def compile_step_policies(model: Any) -> None:
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


def managed_step(function: Callable) -> Callable:
    """Compile step lifecycle once; the hot wrapper performs direct lookups."""
    from hydroforge.execution.boundaries import is_between_steps_api

    if is_between_steps_api(function):
        raise TypeError("@managed_step cannot decorate a @between_steps method")
    descriptor = _ManagedStepDescriptor(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        model = args[0] if args else kwargs["self"]
        return model._execution.step_policies[descriptor].execute(args, kwargs)

    setattr(wrapper, "__hydroforge_managed_step__", descriptor)
    return wrapper

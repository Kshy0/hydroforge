"""Cached outer-step lifecycle and streaming-statistics coordination."""

from __future__ import annotations

import inspect
import math
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable

import cftime
import torch
import torch.distributed as dist

from hydroforge.contracts.events import emit
from hydroforge.contracts.temporal import (
    StatisticsFlags,
    require_calendar,
    timedelta_microseconds,
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
    next_time: Any = None
    pending_outer_first: bool = False
    inner_bucket: int | None = None
    outer_bucket: int | None = None
    schedule_step: int | None = None
    output_active: bool = False


class _WindowPolicy:
    """Calendar-aligned statistics positions derived without caller flags."""

    def __init__(self, model: Any, state: _StepState) -> None:
        self.state = state
        self.calendar = model.calendar
        if (
            model.statistics_interval is None
            and model.statistics_outer_interval is not None
        ):
            raise ValueError(
                "statistics_outer_interval requires statistics_interval"
            )
        self.inner = (
            None if model.statistics_interval is None
            else timedelta_microseconds(
                model.statistics_interval, label="statistics_interval",
            )
        )
        self.outer = (
            self.inner if model.statistics_outer_interval is None
            else timedelta_microseconds(
                model.statistics_outer_interval,
                label="statistics_outer_interval",
            )
        )

    def _instant_microseconds(self, current_time: Any) -> int:
        require_calendar(
            current_time, self.calendar, label="model current_time",
        )
        if isinstance(current_time, cftime.datetime):
            epoch = cftime.datetime(1970, 1, 1, calendar=self.calendar)
        elif isinstance(current_time, datetime):
            epoch = datetime(1970, 1, 1)
        else:
            raise TypeError("model current_time must be a datetime value")
        return timedelta_microseconds(
            current_time - epoch, label="model calendar offset",
        )

    def _bucket(self, current_time: Any, interval: int | None) -> int:
        if interval is None:
            return 0
        if current_time is None:
            raise ValueError(
                "current_time is required when statistics_interval is configured"
            )
        return self._instant_microseconds(current_time) // interval

    def position(
        self, *, current_time: Any, time_step: float,
        output_enabled: bool, final_step: bool,
    ) -> tuple[bool, bool, bool, bool]:
        state = self.state
        was_active = state.output_active
        tracks_time = self.inner is not None or self.outer is not None
        if (
            tracks_time
            and output_enabled
            and was_active
            and current_time != state.next_time
        ):
            raise ValueError(
                f"implicit statistics expected the next model step at "
                f"{state.next_time!r}, got {current_time!r}"
            )
        if self.inner is None:
            inner_first = inner_last = True
            inner_bucket = 0
        else:
            inner_bucket = self._bucket(current_time, self.inner)
            inner_first = state.inner_bucket != inner_bucket
            end_bucket = self._validated_bucket_offset(
                current_time, float(time_step), self.inner,
                label="statistics_interval",
            )
            inner_last = end_bucket != inner_bucket or final_step

        output_starts = bool(output_enabled and not state.output_active)
        if not output_enabled:
            state.output_active = False
            state.next_time = None
        elif output_starts:
            inner_first = True
            state.output_active = True

        if self.outer is None:
            outer_bucket = inner_bucket
            outer_first = inner_first
            outer_last = inner_last
        else:
            outer_bucket = self._bucket(current_time, self.outer)
            outer_first = state.outer_bucket != outer_bucket
            outer_last = (
                self._validated_bucket_offset(
                    current_time, float(time_step), self.outer,
                    label="statistics_outer_interval",
                ) != outer_bucket
                or final_step
            )
        if output_starts:
            outer_first = True

        state.inner_bucket = inner_bucket
        state.outer_bucket = outer_bucket
        if output_enabled and tracks_time:
            state.next_time = current_time + timedelta(seconds=float(time_step))
        return inner_first, inner_last, outer_first, outer_last

    def _validated_bucket_offset(
        self, current_time: Any, offset: float, interval: int, *, label: str,
    ) -> int:
        try:
            offset_delta = timedelta(seconds=offset)
        except OverflowError as exc:
            raise ValueError("time_step is outside timedelta range") from exc
        offset_microseconds = timedelta_microseconds(
            offset_delta, label="time_step",
        )
        start = self._instant_microseconds(current_time)
        end = start + offset_microseconds
        start_bucket = start // interval
        end_bucket = end // interval
        if end_bucket != start_bucket and (
            end_bucket != start_bucket + 1 or end % interval != 0
        ):
            raise ValueError(
                f"model step starting at {current_time!r} crosses a {label} "
                "boundary without ending exactly at the next boundary; use "
                "an aligned step or an explicit StatisticsPlan"
            )
        return end_bucket


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
        self.window = _WindowPolicy(model, self.state)
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

    def checkpoint_state(self) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
        state = self.state
        local = (
            state.elapsed,
            state.start_time,
            state.next_time,
            state.pending_outer_first,
            state.inner_bucket,
            state.outer_bucket,
            state.schedule_step,
            state.output_active,
        )
        controller = (
            None if self.controller is None
            else self.controller.checkpoint_state()
        )
        return local, controller

    def restore_checkpoint_state(
        self, snapshot: tuple[tuple[Any, ...], dict[str, Any] | None],
    ) -> None:
        local, controller = snapshot
        (
            self.state.elapsed,
            self.state.start_time,
            self.state.next_time,
            self.state.pending_outer_first,
            self.state.inner_bucket,
            self.state.outer_bucket,
            self.state.schedule_step,
            self.state.output_active,
        ) = local
        if self.controller is not None and controller is not None:
            self.controller.restore_checkpoint_state(controller)

    @property
    def statistics_control(self) -> str:
        """Return the one persisted temporal-control dialect for this model."""

        return "plan" if self.controller is not None else "implicit"

    def persisted_statistics_state(self) -> dict[str, Any] | None:
        """Return the explicit-plan cursor; implicit mode resumes at a boundary."""

        return (
            None if self.controller is None
            else self.controller.checkpoint_state()
        )

    def validate_persisted_statistics_state(
        self, state: dict[str, Any] | None,
    ) -> None:
        if self.controller is None:
            if state is not None:
                raise ValueError(
                    "implicit statistics control cannot restore a plan cursor"
                )
            return
        if state is None:
            raise ValueError("statistics-plan checkpoint is missing its cursor")
        self.controller.validate_checkpoint_state(state)

    def open_statistics_windows(self, *, has_outer: bool) -> tuple[bool, bool]:
        """Report accumulator windows that cannot be restored without payloads."""

        if self.controller is not None:
            return self.controller.open_windows
        if not self.state.output_active:
            return False, False
        return (
            not self.stat_is_last,
            bool(has_outer and not self.stat_is_outer_last),
        )

    def restore_persisted_statistics_state(
        self, state: dict[str, Any] | None,
    ) -> None:
        """Restore a validated closed-window cursor without stale local state."""

        self.validate_persisted_statistics_state(state)
        self.state = _StepState()
        self.window = _WindowPolicy(self.model, self.state)
        if self.controller is not None:
            self.controller.restore_checkpoint_state(state)

    def begin(
        self, *, current_time: Any, time_step: float,
        output_enabled: bool, final_step: bool,
        program_owner: _ManagedStepDescriptor,
        override: StatisticsFlags | None = None,
    ) -> _StepRuntime:
        model = self.model
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
        if self.controller is None:
            self._validate_model_schedule(
                current_time=current_time,
                output_enabled=output_enabled,
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
        if self.controller is not None:
            decision = self.controller.resolve(
                current_time=current_time,
                time_step=self.time_step,
                output_enabled=output_enabled,
                override=override,
            )
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
            (
                self.stat_is_first,
                self.stat_is_last,
                outer_first,
                outer_last,
            ) = self.window.position(
                current_time=current_time, time_step=self.time_step,
                output_enabled=output_enabled, final_step=final_step,
            )
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

    def _validate_model_schedule(
        self, *, current_time: Any, output_enabled: bool,
    ) -> None:
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
            # Before the first main-schedule step, disabled-output calls are
            # explicit spin-up and intentionally do not advance the cursor.
            if (
                not output_enabled
                and self.state.schedule_step is None
                and current_time < schedule.start
            ):
                return
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
        self.completed_steps = 0
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
            self._progress_tick = getattr(model, "progress_tick", None)
            self._format_progress = getattr(model, "format_progress", None)
        else:
            self._progress_tick = None
            self._format_progress = None

    def _proposed_completion(self) -> tuple[bool, int]:
        """Return final-step status and counter without mutating live state."""

        total_steps = self.execution.total_steps
        if total_steps <= 0:
            return False, self.completed_steps + 1
        base = 0 if self.completed_steps >= total_steps else self.completed_steps
        proposed = base + 1
        return proposed >= total_steps, proposed

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
            context.restore_checkpoint_state(snapshot)
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
        snapshot = context.checkpoint_state()
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
            current_time = layout.get(args, kwargs, "current_time", None)
        except BaseException as error:
            resolved = self._coordinate_failure(
                context, snapshot, error, poison=context.world_size > 1,
            )
            if resolved is error:
                raise
            raise resolved from error
        entered_user_step = False
        final_step, proposed_completed_steps = self._proposed_completion()
        try:
            self.execution.statistics.check_background_failures(current_time)
            context.begin(
                current_time=current_time,
                time_step=layout.get(args, kwargs, "time_step"),
                output_enabled=layout.get(args, kwargs, "output_enabled", True),
                final_step=final_step,
                program_owner=self.descriptor,
                override=override,
            )
            with self._parameter_transaction():
                parameter_effect = self._execute_parameter_change_plan(current_time)
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
            self.completed_steps = proposed_completed_steps
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

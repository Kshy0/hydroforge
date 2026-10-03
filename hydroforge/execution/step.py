"""Managed-step invocations: driver validation, the step transaction and its
authoring scopes.

A managed method validates each distinct driver request once per compiled
policy; a warm step then reuses the cached request and runs the transaction
without constructing validation models.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from pydantic import Field, ValidationInfo, model_validator

from hydroforge.contracts.schedule import SimulationStep
from hydroforge.core.errors import ResourceCleanupError, SubstepCompileError
from hydroforge.core.events import emit
from hydroforge.core.identity import canonical, digest63
from hydroforge.core.time import DateLike, timedelta_microseconds
from hydroforge.core.validation import HydroForgeModel
from hydroforge.execution.channel import StagedPreflight, StepEvent
from hydroforge.execution.context import (
    ACTIVE_STEP,
    is_between_steps_api,
    validate_synchronous_function,
)
from hydroforge.execution.outer import outer_scope
from hydroforge.execution.substeps import adaptive_scope, fixed_scope, predicate_scope
from hydroforge.kernels.calls import routing

if TYPE_CHECKING:
    from hydroforge.execution.session import ModelRuntime


_F = TypeVar("_F", bound=Callable[..., Any])

# Messages for an authoring scope left by ``break`` or ``return`` before its
# program was recorded and launched, keyed by the innermost open scope.
_OPEN_SCOPE_ERRORS: dict[str, tuple[type[Exception], str]] = {
    "substep": (
        RuntimeError,
        "compiled substep scope was exited before recording and execution "
        "completed; do not break or return from a step.fixed/adaptive loop",
    ),
    "outer": (
        RuntimeError,
        "outer operator scope was exited before recording and launch "
        "completed; do not break or return from a step.outer() loop",
    ),
    "predicate": (
        SubstepCompileError,
        "predicate loop scope was exited before recording completed; do not "
        "break or return from a step.predicate() loop",
    ),
}


class StepContext:
    """The managed step executing on one runtime.

    Created once per materialization; ``begin`` binds each invocation.
    ``scopes`` lists the authoring scopes entered and not yet completed.
    """

    def __init__(self, runtime: ModelRuntime) -> None:
        plan = runtime.plan
        self.execution = runtime.execution
        self.executor = runtime.execution.executor
        self.channel = runtime.channel
        self.clock = runtime.clock
        self.mesh = plan.parallel
        self.statistics = runtime.statistics
        self.options_key = plan.options.specialization_key()
        self.invocation: ManagedStep | None = None
        self.scopes: list[str] = []
        self.owner: Any = None
        self.current_time: Any = None
        self.scheduled_step: SimulationStep | None = None
        self.duration: timedelta | None = None
        self.time_step = 0.0
        self.requested_sub_steps: int | None = None
        self.spinup = False
        self.output_enabled = False
        self.sampling = False
        self.flags = 0
        self.substep_claimed = False
        self.completed_substeps: int | None = None
        self._outer_sites: dict[Any, int] = {}

    def snapshot(self) -> tuple[Any, ...]:
        """Capture everything ``begin`` changes, for a failed invocation."""

        statistics = self.statistics
        return (
            self.clock.time,
            self.clock.schedule_index,
            None if statistics is None else statistics.snapshot(),
        )

    def require_invocation(self, invocation: ManagedStep) -> None:
        """Reject expired steps and scopes, including reuse during a later step."""

        if (
            invocation is None
            or self.invocation is not invocation
            or ACTIVE_STEP.get() is not self
        ):
            raise RuntimeError("managed scope belongs to an inactive invocation")

    def restore(self, snapshot: tuple[Any, ...]) -> None:
        self.clock.time, self.clock.schedule_index, windows = snapshot
        if windows is not None:
            self.statistics.restore(windows)

    def begin(self, invocation: _Invocation, owner: Any) -> None:
        request = invocation.request
        step = invocation.scheduled_step
        self.owner = owner
        self.current_time = invocation.current_time
        self.scheduled_step = step
        self.requested_sub_steps = request.num_sub_steps
        self.substep_claimed = False
        self.completed_substeps = None
        self.scopes.clear()
        self._outer_sites.clear()
        duration = request.time_step if step is None else step.end - step.start
        self.duration = duration
        self.time_step = timedelta_microseconds(duration) / 1_000_000
        self.spinup = spinup = step is not None and step.is_spin_up
        enabled = (
            not spinup if request.output_enabled is None else request.output_enabled
        )
        statistics = self.statistics
        if statistics is None:
            self.output_enabled = enabled
            self.sampling = False
            self.flags = 0
        else:
            self.output_enabled = self.sampling = statistics.begin_step(
                step, enabled=enabled, time=invocation.current_time
            )
            self.flags = statistics.step_flags

    def commit_clock(self) -> None:
        """Publish the next model time only after the full step succeeds."""

        if self.scheduled_step is not None:
            self.clock.schedule_index = self.scheduled_step.index + 1
        elif self.current_time is not None:
            self.clock.time = self.current_time + self.duration

    def claim_substep_scope(self, kind: str, specialization: Any) -> tuple[Any, ...]:
        """Claim this managed method's sole cached substep scope."""

        self.substep_claimed = True
        self.scopes.append("substep")
        return (self.owner, kind, self.options_key, specialization)

    def claim_outer_scope(self, site: Any, specialization: Any) -> tuple[Any, ...]:
        """Return the cache key of one outer scope and open it.

        A lexical site reached several times in one invocation (a loop or a
        shared helper) records one program per occurrence, in order.
        """

        occurrence = self._outer_sites.get(site, 0)
        self._outer_sites[site] = occurrence + 1
        self.scopes.append("outer")
        return (self.owner, "outer", site, occurrence, self.options_key, specialization)

    def require_scopes_closed(self, depth: int = 0) -> None:
        """Reject a scope exited by ``break``/``return`` before completion."""

        if len(self.scopes) > depth:
            error, message = _OPEN_SCOPE_ERRORS[self.scopes[-1]]
            raise error(message)

    def sample(self, *, first: bool, last: bool, weight: float) -> None:
        """Run one host-issued statistics sample when the step collects output."""

        if self.sampling:
            statistics = self.statistics
            phase = statistics.sample(first=first, last=last, weight=weight)
            self.executor.sample(statistics.launch, phase)

    @property
    def fold(self) -> bool:
        """Whether a captured loop must fold statistics into every iteration."""

        return self.sampling and self.statistics.per_substep

    def finish(self) -> None:
        self.require_scopes_closed()
        if self.sampling and not self.substep_claimed:
            raise RuntimeError(
                "statistics were enabled but the managed step executed no "
                "step.fixed/adaptive scope"
            )
        if self.statistics is not None:
            self.statistics.finish_step()


class ManagedStep:
    """Physical-step identity passed to model-authored code.

    Driver inputs and schedule resolution are complete before the framework
    creates it; its scopes act only on the invocation that created it.
    """

    __slots__ = (
        "current_time",
        "duration",
        "output_enabled",
        "requested_sub_steps",
        "is_spin_up",
        "_context",
    )

    current_time: DateLike | None
    duration: timedelta
    output_enabled: bool
    requested_sub_steps: int
    is_spin_up: bool

    def __init__(self, context: StepContext) -> None:
        step = context.scheduled_step
        self.current_time = context.current_time if step is None else step.start
        self.duration = context.duration
        self.output_enabled = context.output_enabled
        requested = context.requested_sub_steps
        self.requested_sub_steps = 1 if requested is None else requested
        self.is_spin_up = context.spinup
        self._context = context

    def fixed(
        self,
        *,
        count: object = None,
        specialization: Any = None,
        final: Callable[[], None] | None = None,
    ):
        """Declare one validated fixed physical-substep scope."""

        self._context.require_invocation(self)
        return fixed_scope(
            self._context, count=count, specialization=specialization, final=final
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

        self._context.require_invocation(self)
        return adaptive_scope(
            self._context,
            candidate_dt=candidate_dt,
            dt=dt,
            maximum_dt=maximum_dt,
            maximum_steps=maximum_steps,
            proposal=proposal,
            specialization=specialization,
        )

    def predicate(self, *, maximum_steps: int):
        """Declare one nested device-predicate loop."""

        self._context.require_invocation(self)
        return predicate_scope(self._context, maximum_steps=maximum_steps)

    def outer(self, *, specialization: Any = None):
        """Declare one cached once-per-outer-step operator scope."""

        self._context.require_invocation(self)
        caller = sys._getframe(1)
        return outer_scope(
            self._context,
            site=(caller.f_code, caller.f_lasti),
            specialization=specialization,
        )


_FRAMEWORK_STEP_PARAMETERS = frozenset({"output_enabled", "time_step", "num_sub_steps"})


@dataclass(frozen=True, slots=True)
class _Conditions:
    """Runtime facts a driver request is validated against."""

    time_step_supplied: bool
    schedule_configured: bool
    spinup: bool
    statistics_configured: bool
    current_time_available: bool


class _StepRequest(HydroForgeModel):
    """The complete driver-owned input to one managed-step invocation."""

    time_step: timedelta | None = None
    num_sub_steps: int | None = Field(default=None, ge=1, lt=(1 << 31) - 1)
    output_enabled: bool | None = None

    @model_validator(mode="after")
    def _validate_schedule_ownership(self, info: ValidationInfo) -> _StepRequest:
        conditions = info.context
        if conditions.schedule_configured and conditions.time_step_supplied:
            raise ValueError(
                "time_step is derived from simulation_schedule and must not be provided"
            )
        if not conditions.schedule_configured and self.time_step is None:
            raise ValueError(
                "time_step is required when simulation_schedule is not configured"
            )
        if self.time_step is not None and timedelta_microseconds(self.time_step) <= 0:
            raise ValueError("time_step must be positive")
        if conditions.spinup and self.output_enabled is True:
            raise ValueError("spin-up requires output_enabled=False")
        enabled = (
            not conditions.spinup
            if self.output_enabled is None
            else self.output_enabled
        )
        if (
            conditions.statistics_configured
            and enabled
            and not conditions.current_time_available
        ):
            raise ValueError(
                "current_time must be provided when statistics output is enabled"
            )
        return self


class _Invocation:
    """One validated driver request resolved against the runtime clock."""

    __slots__ = ("current_time", "scheduled_step", "request", "key")

    def __init__(self, current_time, scheduled_step, request, key) -> None:
        self.current_time = current_time
        self.scheduled_step = scheduled_step
        self.request = request
        self.key = key

    def signature(self) -> Any:
        """Return the exact driver and schedule identity shared by all ranks."""

        request = self.request
        return canonical(
            (
                self.current_time,
                self.scheduled_step,
                request.time_step,
                request.num_sub_steps,
                request.output_enabled,
                self.key[6:],
            )
        )


class _ManagedStepDeclaration(HydroForgeModel):
    """One validated model-authored managed-step function declaration."""

    function: Callable

    @model_validator(mode="after")
    def _validate_signature(self) -> _ManagedStepDeclaration:
        if is_between_steps_api(self.function):
            raise ValueError("@managed_step cannot decorate a @between_steps method")
        validate_synchronous_function(self.function, decorator="@managed_step")
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
        self.protocol_name = f"{self.function.__module__}.{self.function.__qualname__}"
        self.protocol_code = digest63(self.protocol_name)
        self.requests: dict[tuple[Any, ...], _StepRequest] = {}

    def compile(self, runtime: ModelRuntime) -> _StepPolicy:
        return _StepPolicy(runtime, self)

    def validate(
        self,
        runtime: ModelRuntime,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _Invocation:
        """Resolve one driver request; each distinct request validates once.

        Only the plan and the clock are read, so a declared runtime validates
        its first invocation before materializing.
        """

        if ACTIVE_STEP.get() is not None:
            raise ValueError("nested @managed_step calls are not supported")
        if len(args) != 1:
            kwargs = {**kwargs, "unexpected_positional_arguments": args[1:]}
        plan = runtime.plan
        clock = runtime.clock
        schedule = plan.schedule
        statistics_configured = plan.output.windows is not None
        step = None
        if schedule is not None:
            if clock.schedule_index >= len(schedule):
                raise ValueError("simulation schedule is exhausted")
            step = schedule._step_at_trusted(clock.schedule_index)
        current_time = clock.time if step is None else step.start
        duration = kwargs.get("time_step")
        count = kwargs.get("num_sub_steps")
        output = kwargs.get("output_enabled")
        key = (
            type(duration),
            duration,
            type(count),
            count,
            type(output),
            output,
            "time_step" in kwargs,
            step is not None and step.is_spin_up,
            current_time is not None,
            schedule is not None,
            statistics_configured,
        )
        cacheable = (
            kwargs.keys() <= _FRAMEWORK_STEP_PARAMETERS
            and (duration is None or type(duration) is timedelta)
            and (count is None or type(count) is int)
            and (output is None or type(output) is bool)
        )
        request = self.requests.get(key) if cacheable else None
        if request is None:
            request = _StepRequest.model_validate(
                kwargs,
                context=_Conditions(
                    time_step_supplied="time_step" in kwargs,
                    schedule_configured=schedule is not None,
                    spinup=key[7],
                    statistics_configured=statistics_configured,
                    current_time_available=key[8],
                ),
            )
            if cacheable and len(self.requests) < 64:
                self.requests[key] = request
        return _Invocation(current_time, step, request, key)


class _StepPolicy:
    """Cached driver requests and the step transaction of one method."""

    def __init__(
        self, runtime: ModelRuntime, descriptor: _ManagedStepDescriptor
    ) -> None:
        plan = runtime.plan
        self.runtime = runtime
        self.model = runtime.owner
        self.execution = runtime.execution
        self.context = runtime.execution.step
        self.channel = runtime.channel
        self.descriptor = descriptor
        parameters = runtime.parameters
        self._parameter_transaction = parameters.step_transaction
        self._execute_parameter_change_plan = parameters.execute_parameter_change_plan
        self.progress = runtime.progress if plan.rank == 0 else None

    def _fail(
        self, snapshot: Any, error: BaseException, *, poison: bool
    ) -> BaseException:
        """Publish failure, restore temporal state, and return its full cause."""

        try:
            self.channel.abort()
        except BaseException as coordination_error:
            error = ResourceCleanupError(
                "managed-step distributed failure propagation",
                (error, coordination_error),
            )
        try:
            self.context.restore(snapshot)
        except BaseException as rollback_error:
            error = ResourceCleanupError(
                "managed-step temporal rollback",
                (error, rollback_error),
            )
        if poison or isinstance(error, ResourceCleanupError):
            self.execution.poison(error, phase="managed-step execution")
        return error

    def execute(
        self,
        invocation: _Invocation,
        preflight: StagedPreflight | None = None,
    ) -> Any:
        channel = self.channel
        execution = self.execution
        context = self.context
        channel.open_step(preflight)
        if preflight is not None and preflight.error is not None:
            channel.abort()
            raise preflight.error
        failure = execution.failure
        if failure is not None:
            error = execution.poisoned_error(failure)
            try:
                channel.abort()
            except BaseException as coordination_error:
                if coordination_error is channel.rejection:
                    raise
                combined = ResourceCleanupError(
                    "managed-step entry failure propagation",
                    (error, coordination_error),
                )
                raise combined from error
            raise error
        snapshot = context.snapshot()
        statistics = context.statistics
        current_time = invocation.current_time
        progress = self.progress
        entered_user_step = False
        preparation_failed = False
        try:
            if statistics is not None:
                statistics.sink.poll(current_time)
            try:
                if context.mesh is not None:
                    context.mesh.validate_live()
                if self.runtime.structure_hooks:
                    self.runtime.update_structure()
            except BaseException:
                preparation_failed = True
                raise
            context.begin(invocation, self.descriptor)
            managed = ManagedStep(context)
            step = invocation.scheduled_step
            requested = context.requested_sub_steps
            channel.event(
                StepEvent.BEGIN,
                (
                    self.descriptor.protocol_code,
                    -1 if step is None else step.index,
                    (0 if requested is None else requested) << 6
                    | context.flags << 2
                    | int(context.output_enabled) << 1
                    | int(context.spinup),
                ),
            )
            if progress is not None:
                progress.begin_step(step)
            with self._parameter_transaction():
                if not context.spinup:
                    self._execute_parameter_change_plan(current_time)
                token = ACTIVE_STEP.set(context)
                context.invocation = managed
                try:
                    with routing(execution.kernel_binding):
                        # From here onward model-authored outer Torch work and
                        # compiled physics may mutate address-stable state.
                        # There is no affordable generic rollback proof for an
                        # arbitrary failure, so the instance must fail closed.
                        entered_user_step = True
                        execution.step_fields.prepare(
                            managed.current_time, context.time_step
                        )
                        result = self.descriptor.function(self.model, managed)
                finally:
                    context.invocation = None
                    ACTIVE_STEP.reset(token)
                # Statistics output is the only rank-visible effect of
                # finish(); without it the final handshake alone rejects a
                # peer's body failure before the clock commits.
                if statistics is not None and statistics.step_closes:
                    channel.event(StepEvent.USER_STEP_COMPLETE)
                context.finish()
                if statistics is not None:
                    statistics.sink.poll(current_time)
                if progress is not None and progress.progress_tick(step):
                    emit(
                        self.runtime,
                        "progress",
                        "step.completed",
                        "Processed step",
                        current_time=managed.current_time,
                        is_spin_up=managed.is_spin_up,
                        adaptive_time_step=context.completed_substeps,
                        progress=progress.format_progress(step),
                    )
                channel.event(StepEvent.STEP_FINALIZED)
                context.commit_clock()
            return result
        except BaseException as error:
            poison = preparation_failed or entered_user_step or channel.distributed
            resolved = self._fail(
                snapshot,
                error,
                poison=poison and error is not channel.rejection,
            )
            if resolved is error:
                raise
            raise resolved from error


def compile_step_policies(runtime: ModelRuntime) -> None:
    """Compile every managed method after module initialization."""
    execution = runtime.execution
    execution.step = StepContext(runtime)
    # Shadowed managed steps stay reachable through ``super()`` from a plain
    # override, so every managed descriptor in the MRO needs its policy.
    for cls in type(runtime.owner).__mro__:
        for method in vars(cls).values():
            descriptor = getattr(method, "__hydroforge_managed_step__", None)
            if descriptor is not None and descriptor not in execution.step_policies:
                execution.step_policies[descriptor] = descriptor.compile(runtime)


def managed_step(function: _F) -> _F:
    """Compile step lifecycle once; the hot wrapper performs direct lookups."""
    declaration = _ManagedStepDeclaration(function=function)
    descriptor = _ManagedStepDescriptor(declaration)
    method_name = function.__qualname__
    phase = f"managed-step.invocation:{descriptor.protocol_name}"
    scope = "distributed managed-step invocation validation"

    @wraps(function)
    def wrapper(*args, **kwargs):
        # ``ModelRuntime.of`` without importing the session that imports us.
        runtime = (args[0] if args else kwargs["self"]).__pydantic_private__["_runtime"]
        channel = runtime.channel
        materialized = runtime.state == "materialized"
        if materialized and not channel.distributed:
            return runtime.execution.step_policies[descriptor].execute(
                descriptor.validate(runtime, args, kwargs)
            )
        invocation: _Invocation | None = None
        error: BaseException | None = None
        try:
            invocation = descriptor.validate(runtime, args, kwargs)
        except BaseException as caught:
            error = caught
        signature = (
            None
            if invocation is None or not channel.distributed
            else (materialized, invocation.signature())
        )
        preflight = None
        if channel.distributed and materialized:
            # A warm step publishes its preflight inside the BEGIN handshake.
            preflight = StagedPreflight(error, phase, scope, signature)
        else:
            channel.preflight(error, phase=phase, scope=scope, signature=signature)
            runtime.ensure_materialized(method_name)
        return runtime.execution.step_policies[descriptor].execute(
            cast(_Invocation, invocation), preflight
        )

    authored = inspect.signature(function)
    parameters = [next(iter(authored.parameters.values()))]
    parameters.extend(
        inspect.Parameter(
            name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=annotation,
        )
        for name, annotation in (
            ("time_step", timedelta | None),
            ("num_sub_steps", int | None),
            ("output_enabled", bool | None),
        )
    )
    wrapper.__signature__ = authored.replace(parameters=parameters)  # type: ignore[attr-defined]
    setattr(wrapper, "__hydroforge_managed_step__", descriptor)
    return cast(_F, wrapper)

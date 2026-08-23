"""Transactional execution boundaries for public model-authored APIs."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any, Callable, TypeVar, cast

from pydantic import model_validator

from hydroforge.contracts.errors import (
    ResourceCleanupError,
    distributed_failure_error,
)
from hydroforge.contracts.validation import HydroForgeModel


_F = TypeVar("_F", bound=Callable[..., Any])


class _BetweenStepsDeclaration(HydroForgeModel):
    """One validated between-step function declaration."""

    function: Callable

    @model_validator(mode="after")
    def _validate_function(self) -> _BetweenStepsDeclaration:
        if getattr(self.function, "__hydroforge_managed_step__", None) is not None:
            raise ValueError(
                "@between_steps cannot decorate a @managed_step method"
            )
        parameters = tuple(inspect.signature(self.function).parameters.values())
        if (
            not parameters
            or parameters[0].name != "self"
            or parameters[0].kind not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ):
            raise ValueError(
                "@between_steps requires self as the first positional parameter"
            )
        return self


def between_steps(function: _F) -> _F:
    """Guard a model-authored mutation that is valid only between steps.

    Argument binding and runtime-health failures are coordinated before the
    body runs. Once the body is entered, any local failure makes mutation
    atomicity unprovable: every rank is notified and the model is permanently
    poisoned so it cannot be stepped again.
    """

    declaration = _BetweenStepsDeclaration(function=function)
    function = declaration.function
    signature = inspect.signature(function)
    protocol_name = f"{function.__module__}.{function.__qualname__}"

    @wraps(function)
    def guarded(self, *args, **kwargs):
        from hydroforge.execution.step import _managed_step_active

        if _managed_step_active():
            raise RuntimeError(
                "@between_steps APIs cannot be called from an active "
                "@managed_step"
            )

        invocation_error: BaseException | None = None
        try:
            signature.bind(self, *args, **kwargs)
        except BaseException as error:
            invocation_error = error
        if self.world_size > 1:
            invocation_failures = self._gather_distributed_failures(
                invocation_error,
                phase=f"between-steps.invocation:{protocol_name}",
                signature=(self._runtime_materialized,),
            )
            if any(
                failure is not None for failure in invocation_failures
            ):
                if invocation_error is not None:
                    raise invocation_error
                raise distributed_failure_error(
                    "distributed between-steps invocation validation",
                    invocation_failures,
                )
        elif invocation_error is not None:
            raise invocation_error

        health_error: BaseException | None = None
        try:
            self._ensure_healthy_runtime()
        except BaseException as error:
            health_error = error
        if self.world_size > 1:
            health_failures = self._gather_distributed_failures(
                health_error,
                phase=f"between-steps.health:{protocol_name}",
            )
            if any(failure is not None for failure in health_failures):
                if health_error is not None:
                    raise health_error
                raise distributed_failure_error(
                    "distributed between-steps runtime health validation",
                    health_failures,
                )
        elif health_error is not None:
            raise health_error

        result: Any = None
        body_error: BaseException | None = None
        try:
            result = function(self, *args, **kwargs)
        except BaseException as error:
            body_error = error

        poison_phase = f"between-steps body:{protocol_name}"
        if self.world_size > 1:
            try:
                body_failures = self._gather_distributed_failures(
                    body_error,
                    phase=f"between-steps.body:{protocol_name}",
                )
            except BaseException as coordination_error:
                failure = (
                    coordination_error
                    if body_error is None
                    else ResourceCleanupError(
                        "between-steps body failure coordination",
                        (body_error, coordination_error),
                    )
                )
                self._execution.poison(failure, phase=poison_phase)
                if failure is coordination_error:
                    raise
                raise failure from coordination_error
            if any(failure is not None for failure in body_failures):
                failure = (
                    body_error
                    if body_error is not None
                    else distributed_failure_error(
                        "distributed between-steps body",
                        body_failures,
                    )
                )
                self._execution.poison(failure, phase=poison_phase)
                raise failure
        elif body_error is not None:
            self._execution.poison(body_error, phase=poison_phase)
            raise body_error
        return result

    guarded.__hydroforge_between_steps__ = True
    return cast(_F, guarded)


def is_between_steps_api(value: Any) -> bool:
    """Read the nominal marker without invoking descriptors or user code."""

    try:
        marker = inspect.getattr_static(
            value, "__hydroforge_between_steps__",
        )
    except AttributeError:
        return False
    return marker is True

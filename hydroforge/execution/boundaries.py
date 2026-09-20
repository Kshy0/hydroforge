"""Transactional execution boundaries for public model-authored APIs."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import model_validator

from hydroforge.contracts.errors import (
    ResourceCleanupError,
    distributed_failure_error,
)
from hydroforge.contracts.validation import HydroForgeModel

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel

_F = TypeVar("_F", bound=Callable[..., Any])


def coordinate_preflight(
    model: AbstractModel,
    error: BaseException | None,
    *,
    phase: str,
    scope: str,
    signature: tuple[Any, ...] | None = None,
) -> None:
    """Coordinate an entry check before its guarded side effects begin."""

    if model.world_size > 1:
        failures = model._gather_distributed_failures(
            error, phase=phase, signature=signature
        )
        if any(failure is not None for failure in failures):
            if error is not None:
                raise error
            raise distributed_failure_error(scope, failures)
    elif error is not None:
        raise error


def _validate_synchronous_function(function: Callable, *, decorator: str) -> None:
    def is_deferred(implementation: Callable) -> bool:
        return (
            inspect.iscoroutinefunction(implementation)
            or inspect.isgeneratorfunction(implementation)
            or inspect.isasyncgenfunction(implementation)
        )

    if is_deferred(inspect.unwrap(function, stop=is_deferred)):
        raise ValueError(f"{decorator} requires a synchronous non-generator function")


class _BetweenStepsDeclaration(HydroForgeModel):
    """One validated between-step function declaration."""

    function: Callable

    @model_validator(mode="after")
    def _validate_function(self) -> _BetweenStepsDeclaration:
        if getattr(self.function, "__hydroforge_managed_step__", None) is not None:
            raise ValueError("@between_steps cannot decorate a @managed_step method")
        _validate_synchronous_function(self.function, decorator="@between_steps")
        parameters = tuple(inspect.signature(self.function).parameters.values())
        if (
            not parameters
            or parameters[0].name != "self"
            or parameters[0].kind
            not in {
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
    parameter_names = set(signature.parameters).difference({"self"})
    parameters = tuple(signature.parameters.values())[1:]
    simple_binding = all(
        parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in parameters
    )
    keyword_names = frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    )
    required_names = frozenset(
        parameter.name
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
    )
    defaults = {
        parameter.name: parameter.default
        for parameter in parameters
        if parameter.default is not inspect.Parameter.empty
    }

    @wraps(function)
    def guarded(self, *args, **kwargs):
        from hydroforge.execution.step import _managed_step_active

        if _managed_step_active():
            raise RuntimeError(
                "@between_steps APIs cannot be called from an active @managed_step"
            )

        invocation_error: BaseException | None = None
        try:
            if (
                simple_binding
                and not args
                and kwargs.keys() <= keyword_names
                and required_names <= kwargs.keys()
            ):
                arguments = {"self": self, **defaults, **kwargs}
            else:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                arguments = bound.arguments
            from hydroforge.contracts.options import OptionsConfig

            options = getattr(self, "options", None)
            if isinstance(options, OptionsConfig):
                options.validate_forcing_arguments(
                    parameter_names,
                    arguments,
                )
        except BaseException as error:
            invocation_error = error
        coordinate_preflight(
            self,
            invocation_error,
            phase=f"between-steps.invocation:{protocol_name}",
            scope="distributed between-steps invocation validation",
            signature=(self._runtime_materialized,) if self.world_size > 1 else None,
        )

        health_error: BaseException | None = None
        try:
            self._ensure_healthy_runtime()
        except BaseException as error:
            health_error = error
        coordinate_preflight(
            self,
            health_error,
            phase=f"between-steps.health:{protocol_name}",
            scope="distributed between-steps runtime health validation",
        )

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
            value,
            "__hydroforge_between_steps__",
        )
    except AttributeError:
        return False
    return marker is True

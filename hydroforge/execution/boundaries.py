# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Transactional execution boundaries for public model-authored APIs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, TypeVar, cast

from pydantic import model_validator

from hydroforge.core.errors import (
    ResourceCleanupError,
    distributed_failure_error,
)
from hydroforge.core.validation import HydroForgeModel
from hydroforge.execution.context import (
    managed_step_active,
    validate_synchronous_function,
)
from hydroforge.execution.session import ModelRuntime
from hydroforge.kernels.calls import routing

_F = TypeVar("_F", bound=Callable[..., Any])


class _BetweenStepsDeclaration(HydroForgeModel):
    """One validated between-step function declaration."""

    function: Callable

    @model_validator(mode="after")
    def _validate_function(self) -> _BetweenStepsDeclaration:
        if getattr(self.function, "__hydroforge_managed_step__", None) is not None:
            raise ValueError("@between_steps cannot decorate a @managed_step method")
        validate_synchronous_function(self.function, decorator="@between_steps")
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

    In a distributed run every rank must call the same between-step APIs in
    the same order. Arguments are this rank's own data (for example its
    spatial shard of new inputs): the framework compares only the method
    identity and call order across ranks, never the argument values.
    """

    declaration = _BetweenStepsDeclaration(function=function)
    function = declaration.function
    signature = inspect.signature(function)
    method_name = function.__qualname__
    protocol_name = f"{function.__module__}.{method_name}"
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
        if managed_step_active():
            raise RuntimeError(
                "@between_steps APIs cannot be called from an active @managed_step"
            )

        runtime = ModelRuntime.of(self)
        channel = runtime.channel
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

            runtime.plan.options.validate_forcing_arguments(
                parameter_names,
                arguments,
            )
        except BaseException as error:
            invocation_error = error
        channel.preflight(
            invocation_error,
            phase=f"between-steps.invocation:{protocol_name}",
            scope="distributed between-steps invocation validation",
            signature=(runtime.state == "materialized",),
        )

        health_error: BaseException | None = None
        try:
            runtime.require_healthy(method_name)
        except BaseException as error:
            health_error = error
        channel.preflight(
            health_error,
            phase=f"between-steps.health:{protocol_name}",
            scope="distributed between-steps runtime health validation",
        )

        result: Any = None
        body_error: BaseException | None = None
        # Registered kernels called by the body bind like inside a step.
        execution = runtime.execution
        try:
            with (
                nullcontext()
                if execution is None
                else routing(execution.kernel_binding)
            ):
                result = function(self, *args, **kwargs)
        except BaseException as error:
            body_error = error

        poison_phase = f"between-steps body:{protocol_name}"
        try:
            body_failures = channel.gather(
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
            runtime.execution.poison(failure, phase=poison_phase)
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
            runtime.execution.poison(failure, phase=poison_phase)
            raise failure
        return result

    guarded.__hydroforge_between_steps__ = True
    return cast(_F, guarded)


@contextmanager
def specialization_update(model: Any, *, phase: str) -> Iterator[None]:
    """Publish between-step changes to values compiled kernels specialize on.

    Kernel bindings, compiled programs and captures of the materialized
    ``model`` are released before the body assigns new non-tensor values
    (for example module scalars bound as kernel arguments). A failure of the
    release or of the body poisons the model instead of letting a stale
    specialization run. Tensor storage changes use ``update_structure``.
    """

    if managed_step_active():
        raise RuntimeError(
            "specialization updates cannot run inside an active @managed_step"
        )
    runtime = ModelRuntime.of(model)
    what = f"{type(model).__name__}.specialization_update"
    runtime.require_materialized(what)
    runtime.require_healthy(what)
    execution = runtime.execution
    try:
        execution.invalidate()
        yield
    except BaseException as error:
        execution.poison(error, phase=phase)
        raise

"""Nominal execution-boundary contracts for public model APIs."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any


def between_steps(function):
    """Declare an API valid only at a healthy between-step boundary."""

    if getattr(function, "__hydroforge_managed_step__", None) is not None:
        raise TypeError("@between_steps cannot decorate a @managed_step method")

    @wraps(function)
    def guarded(self, *args, **kwargs):
        self._execution.require_between_steps(function.__name__)
        return function(self, *args, **kwargs)

    guarded.__hydroforge_between_steps__ = True
    return guarded


def is_between_steps_api(value: Any) -> bool:
    """Read the nominal marker without invoking descriptors or user code."""

    try:
        marker = inspect.getattr_static(
            value, "__hydroforge_between_steps__",
        )
    except AttributeError:
        return False
    return marker is True

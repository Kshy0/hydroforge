# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Leaf primitives shared by the modules of the execution package."""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from types import FunctionType, MethodType
from typing import Any

import torch

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.core.identity import canonical

# The managed-step context executing on this thread, if any.
ACTIVE_STEP: ContextVar[Any] = ContextVar(
    "hydroforge_active_managed_step", default=None
)


def managed_step_active() -> bool:
    """Return whether the caller is inside one executing managed step."""

    return ACTIVE_STEP.get() is not None


class InvocationScope(ABC):
    """A scope whose iterator may advance only during its creating invocation."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.invocation = context.invocation
        self._started = False

    def __iter__(self) -> Generator[Any, None, None]:
        self.context.require_invocation(self.invocation)
        if self._started:
            raise RuntimeError("managed scope has already been consumed")
        self._started = True
        iterator = self._iterate()
        with cleanup_on_exit("managed scope iterator", (iterator.close,)):
            while True:
                self.context.require_invocation(self.invocation)
                try:
                    frame = next(iterator)
                except StopIteration:
                    return
                yield frame

    @abstractmethod
    def _iterate(self) -> Generator[Any, None, None]: ...


@dataclass(frozen=True, slots=True)
class SubstepFrame:
    """Compiler-owned scalar tensors visible only inside a sub-step body."""

    index: torch.Tensor
    dt: torch.Tensor


@dataclass(frozen=True, slots=True)
class PredicateLoopFrame:
    """Device state exposed to one nested predicate-loop body."""

    index: torch.Tensor
    continue_flag: torch.Tensor


def specialization_key(value: Any) -> Any:
    """Validate an explicit host specialization and return its identity."""

    def plain(item: Any) -> bool:
        if type(item) is tuple:
            return all(map(plain, item))
        if type(item) is float:
            return math.isfinite(item)
        return item is None or type(item) in {bool, int, str}

    if not plain(value):
        raise ValueError(
            "substep specialization must be None, bool, int, finite float, "
            "str, or a tuple composed from those exact scalar types"
        )
    return canonical(value)


def close_runner(
    executor: Any, runner: Any, programs: Iterable[Any], *, scope: str
) -> None:
    """Close a runner, then the recorded programs its captures reference."""

    with cleanup_on_exit(
        scope,
        (
            *(() if runner is None else (runner.close,)),
            *(
                partial(program.close, executor)
                for program in programs
                if program is not None
            ),
        ),
    ):
        pass


def validate_synchronous_function(function: Callable, *, decorator: str) -> None:
    """Reject coroutine and generator implementations of a step API."""

    def is_deferred(implementation: Callable) -> bool:
        return (
            inspect.iscoroutinefunction(implementation)
            or inspect.isgeneratorfunction(implementation)
            or inspect.isasyncgenfunction(implementation)
        )

    implementation = function
    seen: set[int] = set()
    while True:
        if id(implementation) in seen:
            raise ValueError(f"{decorator} has a cyclic callable wrapper")
        seen.add(id(implementation))
        if is_deferred(implementation):
            raise ValueError(
                f"{decorator} requires a synchronous non-generator function"
            )
        # Preserve partial's wrapped function before falling back to __call__.
        # Inspecting partial.__call__ loses coroutine/generator information.
        if isinstance(implementation, partial):
            implementation = implementation.func
        elif hasattr(implementation, "__wrapped__"):
            implementation = implementation.__wrapped__
        elif not inspect.isroutine(implementation):
            implementation = implementation.__call__
        else:
            return


def validate_callback(function: Callable | None, *, arguments: int, label: str) -> None:
    """Validate the fixed invocation shape before registering a callback."""
    if function is None:
        return
    if not callable(function):
        raise TypeError(f"{label} must be callable")
    validate_synchronous_function(function, decorator=label)
    implementation = function.__func__ if isinstance(function, MethodType) else function
    if isinstance(implementation, FunctionType) and not any(
        hasattr(implementation, name)
        for name in ("__wrapped__", "__signature__", "__text_signature__")
    ):
        code = implementation.__code__
        bound = int(isinstance(function, MethodType))
        if code.co_argcount == arguments + bound and code.co_kwonlyargcount == 0:
            # The common loop callback has exactly the invoked positional
            # parameters. Read the live function ABI without signature objects
            # or a cache that could outlive changes to code/defaults/wrappers.
            return
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        # Extension callables without inspectable signatures are checked on call.
        return
    try:
        signature.bind(*(None for _ in range(arguments)))
    except TypeError as error:
        raise ValueError(
            f"{label} must accept {arguments} positional argument(s)"
        ) from error


def is_between_steps_api(value: Any) -> bool:
    """Read the ``@between_steps`` marker without invoking user code."""

    try:
        marker = inspect.getattr_static(value, "__hydroforge_between_steps__")
    except AttributeError:
        return False
    return marker is True

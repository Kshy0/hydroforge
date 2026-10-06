# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The one route of registered kernel calls of a model.

A registered kernel goes to the innermost call sink as
``sink.call(registry, arguments)``: the recorder of an operator recording, or
the model's kernel binder, which launches at once. The binder is installed
for managed steps, ``initialize_model_state`` and ``@between_steps`` bodies.  A sink's
``recording`` tells callers whether nothing launches until compilation.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Protocol, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


class KernelCallSink(Protocol):
    recording: bool

    def call(self, registry: Any, arguments: dict[str, Any]) -> None: ...


_SINK: ContextVar[KernelCallSink | None] = ContextVar(
    "hydroforge_kernel_call_sink", default=None
)


class routing:
    """Route registered kernel calls to ``sink`` for the ``with`` block."""

    __slots__ = ("sink", "token")

    def __init__(self, sink: KernelCallSink) -> None:
        self.sink = sink

    def __enter__(self) -> KernelCallSink:
        self.token = _SINK.set(self.sink)
        return self.sink

    def __exit__(self, *exc: object) -> None:
        _SINK.reset(self.token)


def current_sink() -> KernelCallSink | None:
    """The innermost call sink of this thread, if any."""

    return _SINK.get()


def recording_sink() -> KernelCallSink | None:
    """The innermost call sink when it records instead of launching."""

    sink = _SINK.get()
    return sink if sink is not None and sink.recording else None


def compiled_operator_entry(function: _F) -> _F:
    """Mark a framework function as one nominal substep IR operator entry."""

    function.__hydroforge_compiled_operator__ = True
    return function

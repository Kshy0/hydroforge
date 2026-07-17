"""Lazy Objective-C++ bridge for native Metal function constants."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


_recording_sequence: ContextVar[Any] = ContextVar(
    "hydroforge_metal_recording_sequence", default=None,
)


@cache
def load_metal_kernel():
    if sys.platform != "darwin":
        raise RuntimeError("Native Metal kernels are only available on macOS")

    from torch.utils.cpp_extension import load

    # cpp_extension invokes helper programs by name; make the active Python
    # environment discoverable without imposing a package-build dependency.
    os.environ["PATH"] = (
        f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
    )
    source = Path(__file__).with_suffix(".mm")
    return load(
        name="hydroforge_metal_kernel",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++20", "-fno-objc-arc", "-fblocks"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )


@dataclass
class MetalCommandSequence:
    """A fixed ordered group of prepared Metal kernel calls."""

    commands: list[tuple[Callable, dict[str, Any], bool]] = field(
        default_factory=list,
    )
    prepared_commands: list[tuple[Any, int, int, int, int, bool]] = field(
        default_factory=list,
    )

    def add(self, dispatcher: Callable, *, barrier: bool = True, **kwargs) -> None:
        prepare = getattr(dispatcher, "prepare", None)
        if prepare is None:
            raise TypeError("dispatcher was not created by make_metal_dispatcher")
        self.commands.append((dispatcher, kwargs, barrier))

    def add_prepared(
        self,
        prepared: tuple[Any, int, int, int, int],
        *,
        barrier: bool = True,
    ) -> None:
        """Record an already-specialized launch from the normal dispatcher."""
        self.prepared_commands.append((*prepared, barrier))

    def mark_barrier(self) -> None:
        """Place a dependency barrier after the most recently recorded command."""
        if not self.prepared_commands:
            raise RuntimeError("No Metal command has been recorded")
        self.prepared_commands[-1] = (*self.prepared_commands[-1][:-1], True)

    def _prepare(self):
        if not self.commands and not self.prepared_commands:
            raise ValueError("Metal command sequence is empty")
        prepared = [
            (*dispatcher.prepare(**kwargs), barrier)
            for dispatcher, kwargs, barrier in self.commands
        ]
        prepared.extend(self.prepared_commands)
        native = prepared[0][0]
        if any(item[0] is not native for item in prepared[1:]):
            raise RuntimeError("Metal commands use different native runtimes")
        return (
            native,
            [item[1] for item in prepared],
            [item[2] for item in prepared],
            [item[3] for item in prepared],
            [item[4] for item in prepared],
            [item[5] for item in prepared],
        )

    def dispatch(self) -> None:
        native, pipelines, bindings, threads, groups, barriers = self._prepare()
        native.dispatch_sequence(
            pipelines, bindings, threads, groups, barriers,
        )

    def capture(self) -> "MetalICB":
        native, pipelines, bindings, threads, groups, barriers = self._prepare()
        graph_id = native.create_icb(
            pipelines, bindings, threads, groups, barriers,
        )
        return MetalICB(native, graph_id)


@dataclass
class MetalICB:
    """Persistent fixed-address Metal indirect command buffer."""

    _native: Any
    graph_id: int
    _closed: bool = field(default=False, init=False)

    def replay(self, replays: int = 1) -> None:
        if self._closed:
            raise RuntimeError("Metal ICB has been released")
        self._native.replay_icb(self.graph_id, int(replays))

    def close(self) -> None:
        if not self._closed:
            self._native.release_icb(self.graph_id)
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may unload the extension before this object.
            pass


@contextmanager
def record_metal_commands(sequence: MetalCommandSequence):
    """Record launches made by existing Metal dispatchers without executing them."""
    token = _recording_sequence.set(sequence)
    try:
        yield sequence
    finally:
        _recording_sequence.reset(token)


def recording_metal_sequence() -> MetalCommandSequence | None:
    """Return the sequence active in this Python context, if any."""
    return _recording_sequence.get()


def mark_metal_barrier() -> None:
    """Mark a model-declared dependency boundary while recording an ICB."""
    sequence = _recording_sequence.get()
    if sequence is not None:
        sequence.mark_barrier()

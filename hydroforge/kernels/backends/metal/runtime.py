"""Lazy Objective-C++ bridge for native Metal function constants."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from hydroforge.contracts import ResourceCleanupError


_recording_sequence: ContextVar[Any] = ContextVar(
    "hydroforge_metal_recording_sequence", default=None,
)


def _raise_failures(scope: str, failures: list[BaseException]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    error = ResourceCleanupError(scope, failures)
    raise error from failures[0]


def metal_resource_identity(value: Any) -> Any:
    """Identify the underlying allocation so distinct tensor views alias."""
    if not isinstance(value, torch.Tensor):
        return type(value), id(value)
    pointer = value.untyped_storage().data_ptr()
    if pointer == 0:
        return "empty_tensor", id(value)
    device = value.device
    return "tensor_storage", device.type, device.index, pointer


@cache
def load_metal_kernel():
    if sys.platform != "darwin":
        raise RuntimeError("Native Metal kernels are only available on macOS")

    from torch.utils.cpp_extension import load

    # cpp_extension invokes helper programs by name; make the active Python
    # environment discoverable without imposing a package-build dependency.
    old_path = os.environ.get("PATH")
    executable_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = (
        executable_dir if old_path is None else f"{executable_dir}:{old_path}"
    )
    source = Path(__file__).with_suffix(".mm")
    try:
        return load(
            name="hydroforge_metal_kernel",
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++20", "-fno-objc-arc", "-fblocks"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
    finally:
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path


@dataclass
class MetalCommandSequence:
    """A fixed ordered group of prepared Metal kernel calls."""

    prepared_commands: list[tuple[Any, int, int, int, int, bool]] = field(
        default_factory=list,
    )
    _pending_reads: set[int] = field(default_factory=set, init=False)
    _pending_writes: set[int] = field(default_factory=set, init=False)

    def add_prepared(
        self,
        prepared: tuple[Any, int, int, int, int],
        *,
        barrier: bool = True,
        reads: tuple[Any, ...] = (),
        writes: tuple[Any, ...] = (),
    ) -> None:
        """Record an already-specialized launch from the normal dispatcher."""
        read_ids = {metal_resource_identity(value) for value in reads}
        write_ids = {metal_resource_identity(value) for value in writes}
        if (
            self._pending_writes.intersection(read_ids | write_ids)
            or self._pending_reads.intersection(write_ids)
        ):
            self.mark_barrier()
        self.prepared_commands.append((*prepared, barrier))
        if barrier:
            self._pending_reads.clear()
            self._pending_writes.clear()
        else:
            self._pending_reads.update(read_ids)
            self._pending_writes.update(write_ids)

    def mark_barrier(self) -> None:
        """Place a dependency barrier after the most recently recorded command."""
        if not self.prepared_commands:
            raise RuntimeError("No Metal command has been recorded")
        self.prepared_commands[-1] = (*self.prepared_commands[-1][:-1], True)
        self._pending_reads.clear()
        self._pending_writes.clear()

    def _prepare(self):
        if not self.prepared_commands:
            raise ValueError("Metal command sequence is empty")
        prepared = []
        for item in self.prepared_commands:
            if item[3] == 0:
                # A zero-width dispatch is a semantic no-op. Preserve an
                # explicit barrier by moving it to the nearest real command;
                # otherwise it must not occupy an ICB command slot because
                # Metal requires positive indirect dispatch dimensions.
                if item[5] and prepared:
                    prepared[-1] = (*prepared[-1][:-1], True)
                continue
            prepared.append(item)
        if not prepared:
            return None, [], [], [], [], []
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

    def close(self) -> None:
        """Release every prepared binding that was not transferred to an ICB."""

        prepared, self.prepared_commands = self.prepared_commands, []
        self._pending_reads.clear()
        self._pending_writes.clear()
        failures: list[BaseException] = []
        for native, _pipeline, binding, _threads, _groups, _barrier in prepared:
            try:
                native.release_argument_binding(binding)
            except BaseException as error:
                failures.append(error)
        if failures:
            error = ResourceCleanupError("Metal argument bindings", failures)
            raise error from failures[0]

    def dispatch(self) -> None:
        failures: list[BaseException] = []
        try:
            native, pipelines, bindings, threads, groups, barriers = self._prepare()
            if native is not None:
                native.dispatch_sequence(
                    pipelines, bindings, threads, groups, barriers,
                )
        except BaseException as error:
            failures.append(error)
        try:
            self.close()
        except BaseException as error:
            failures.append(error)
        _raise_failures("Metal command dispatch", failures)

    def capture(self) -> "MetalICB | MetalNoOpICB":
        failures: list[BaseException] = []
        native = None
        graph_id = None
        empty = False
        try:
            native, pipelines, bindings, threads, groups, barriers = self._prepare()
            if native is None:
                empty = True
            else:
                graph_id = native.create_icb(
                    pipelines, bindings, threads, groups, barriers,
                )
        except BaseException as error:
            failures.append(error)
        try:
            self.close()
        except BaseException as error:
            failures.append(error)
        if failures and graph_id is not None:
            try:
                native.release_icb(graph_id)
            except BaseException as error:
                failures.append(error)
        _raise_failures("Metal ICB capture", failures)
        return MetalNoOpICB() if empty else MetalICB(native, graph_id)


@dataclass
class MetalNoOpICB:
    """An address-stable empty program produced by zero-width operators."""

    _closed: bool = field(default=False, init=False)

    def replay(self, replays: int = 1) -> None:
        if self._closed:
            raise RuntimeError("Metal no-op ICB has been released")
        if type(replays) is not int or replays < 1:
            raise ValueError("Metal ICB replay count must be a positive int")

    def close(self) -> None:
        self._closed = True


@dataclass
class MetalICB:
    """Persistent fixed-address Metal indirect command buffer."""

    _native: Any
    graph_id: int
    _closed: bool = field(default=False, init=False)

    def replay(self, replays: int = 1) -> None:
        if self._closed:
            raise RuntimeError("Metal ICB has been released")
        if type(replays) is not int or replays < 1:
            raise ValueError("Metal ICB replay count must be a positive int")
        self._native.replay_icb(self.graph_id, replays)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._native.release_icb(self.graph_id)

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
    except BaseException as primary:
        try:
            sequence.close()
        except BaseException as cleanup_error:
            error = ResourceCleanupError(
                "Metal command recording", (primary, cleanup_error),
            )
            raise error from primary
        raise
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

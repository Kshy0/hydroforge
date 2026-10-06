# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Metal toolchain: the Objective-C++ bridge, libraries, pipelines and ICBs.

The bridge (``metal.mm``) is built once per environment as a PyTorch
extension.  An MSL library compiles once per source text and origin
(:func:`library`); each kernel and function-constant specialization of a
library is one pipeline, cached by its dispatcher.  Argument bindings, command
sequences and indirect command buffers belong to their launches.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

import torch

from hydroforge.core.errors import ResourceCleanupError, cleanup_on_exit
from hydroforge.kernels.toolchain.cache import (
    acquire_compile_lock,
    clear_abandoned_torch_lock,
    release_compile_lock,
    serialized_compilation,
    temporary_environment,
)
from hydroforge.platform.backend import METAL

_recording_sequence: ContextVar[Any] = ContextVar(
    "hydroforge_metal_recording_sequence",
    default=None,
)


def metal_resource_identity(value: Any) -> tuple[Any, ...]:
    """Identify the underlying allocation so distinct tensor views alias."""
    if not isinstance(value, torch.Tensor):
        return type(value), id(value)
    pointer = value.untyped_storage().data_ptr()
    if pointer == 0:
        return "empty_tensor", id(value)
    device = value.device
    return "tensor_storage", device.type, device.index, pointer


@cache
@serialized_compilation()
def load_metal_kernel():
    if sys.platform != "darwin":
        raise RuntimeError("Native Metal kernels are only available on macOS")

    from torch.utils.cpp_extension import _get_build_directory, load

    # cpp_extension invokes helper programs by name; make the active Python
    # environment discoverable without imposing a package-build dependency.
    old_path = os.environ.get("PATH")
    executable_dir = str(Path(sys.executable).parent)
    build_path = executable_dir if old_path is None else f"{executable_dir}:{old_path}"
    source = Path(__file__).with_suffix(".mm")
    name = "hydroforge_metal_kernel"
    build_dir = Path(_get_build_directory(name, verbose=False))
    lock_path = build_dir / ".hydroforge_compile.lock"
    # Torch's own build lock waits forever on a file left by an interrupted
    # compile. Holding HydroForge's bounded, liveness-checked lock proves any
    # torch lock in this directory is abandoned before torch sees it.
    token, _ = acquire_compile_lock(lock_path)
    with cleanup_on_exit(
        "Metal compile lock", (lambda: release_compile_lock(lock_path, token),)
    ):
        clear_abandoned_torch_lock(build_dir)
        with temporary_environment({"PATH": build_path}):
            return load(
                name=name,
                sources=[str(source)],
                extra_cflags=["-O3", "-std=c++20", "-fno-objc-arc", "-fblocks"],
                extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
                build_directory=str(build_dir),
                verbose=False,
            )


MetalOrigin = Literal["physics", "aten", "framework"]

# Safe math still contracts ``a * b + c`` within one statement; this pragma,
# leading the source, turns contraction off for the whole library.
_NO_CONTRACTION = "#pragma METAL fp contract(off)\n"

_LIBRARIES: dict[tuple[str, bool, str], int] = {}
_LIBRARY_LOCK = threading.Lock()


def prepare_source(
    source: str, *, origin: MetalOrigin, encoding: str = "native"
) -> str:
    """The exact source saved, hashed, and submitted to the native compiler."""
    if encoding not in {"native", "float32x2"}:
        raise ValueError(f"unknown Metal encoding {encoding!r}")
    if encoding == "float32x2" and METAL.math.physics_fast_math:
        raise ValueError("Metal float32x2 requires HYDROFORGE_FAST_MATH=0")
    if encoding == "float32x2":
        # Comments are removed by preprocessing, including ones directly
        # adjacent to a pragma. Generated pair programs need no _Pragma;
        # reject that indirection rather than attempting macro expansion.
        spliced = source.replace("\\\r\n", "").replace("\\\n", "")
        checked = re.sub(r"/\*.*?\*/|//[^\n]*", " ", spliced, flags=re.DOTALL)
        unsafe = re.search(
            r"(?mi)^\s*#\s*pragma\s+(?:METAL\s+fp\s+(?:contract\s*\(\s*(?:on|fast)\s*\)|"
            r"math_mode\s*\(\s*(?:relaxed|fast)\s*\))|STDC\s+FP_CONTRACT\s+ON)(?=\s|$)",
            checked,
        )
        if unsafe or re.search(r"\b(?:_Pragma|__pragma)\b", checked):
            raise ValueError(
                "Metal float32x2 source cannot re-enable unsafe math or contraction"
            )
    if origin == "framework" or encoding == "float32x2":
        if not source.startswith(_NO_CONTRACTION):
            source = _NO_CONTRACTION + source
    return source


def library(source: str, *, origin: MetalOrigin, encoding: str = "native") -> int:
    """Compile one MSL library once per source text and origin.

    Metal otherwise compiles with fast math.  Physics libraries follow the
    math mode; ATen stand-ins for torch operators and framework libraries
    keep safe math so their NaN and finiteness checks survive.  Framework
    libraries (statistics, step fields, loop control) also never contract
    multiplies and adds, so their results do not depend on how the generated
    code is structured.
    """

    source = prepare_source(source, origin=origin, encoding=encoding)
    fast_math = origin == "physics" and METAL.math.physics_fast_math
    key = (hashlib.sha256(source.encode()).hexdigest(), fast_math, encoding)
    with _LIBRARY_LOCK:
        compiled = _LIBRARIES.get(key)
        if compiled is None:
            compiled = load_metal_kernel().compile_library(source, fast_math)
            _LIBRARIES[key] = compiled
    return compiled


@dataclass
class MetalCommandSequence:
    """A fixed ordered group of prepared Metal kernel calls."""

    prepared_commands: list[tuple[Any, int, int, int, int, bool]] = field(
        default_factory=list,
    )
    _pending_reads: set[tuple[Any, ...]] = field(default_factory=set, init=False)
    _pending_writes: set[tuple[Any, ...]] = field(default_factory=set, init=False)

    def add_prepared(
        self,
        prepared: tuple[Any, int, int, int, int],
        *,
        barrier: bool = True,
        reads: tuple[Any, ...] = (),
        writes: tuple[Any, ...] = (),
    ) -> None:
        """Record an already-specialized launch from the normal dispatcher."""
        if len(prepared) != 5 or type(barrier) is not bool:
            raise TypeError(
                "Metal prepared commands require five fields and an exact bool barrier"
            )
        if type(prepared[4]) is not int or not 1 <= prepared[4] <= 1024:
            raise ValueError(
                "Metal command group size must be an exact int in [1, 1024]"
            )
        read_ids = {metal_resource_identity(value) for value in reads}
        write_ids = {metal_resource_identity(value) for value in writes}
        if self._pending_writes.intersection(
            read_ids | write_ids
        ) or self._pending_reads.intersection(write_ids):
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
        if self.prepared_commands:
            self.prepared_commands[-1] = (*self.prepared_commands[-1][:-1], True)
        self._pending_reads.clear()
        self._pending_writes.clear()

    def _prepare(self):
        if not self.prepared_commands:
            return None, [], [], [], [], []
        prepared = []
        for item in self.prepared_commands:
            threads = item[3]
            METAL.validate_extent("Metal command sequence", threads, 0)
            if threads == 0:
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
        with cleanup_on_exit("Metal command dispatch", (self.close,)):
            native, pipelines, bindings, threads, groups, barriers = self._prepare()
            if native is not None:
                native.dispatch_sequence(
                    pipelines,
                    bindings,
                    threads,
                    groups,
                    barriers,
                )

    def capture(self) -> MetalICB | MetalNoOpICB:
        native = graph_id = None
        try:
            with cleanup_on_exit("Metal ICB capture", (self.close,)):
                native, pipelines, bindings, threads, groups, barriers = self._prepare()
                if native is not None:
                    graph_id = native.create_icb(
                        pipelines,
                        bindings,
                        threads,
                        groups,
                        barriers,
                    )
        except BaseException:
            if graph_id is None:
                raise
            with cleanup_on_exit(
                "Metal ICB capture", (lambda: native.release_icb(graph_id),)
            ):
                raise
        return MetalNoOpICB() if native is None else MetalICB(native, graph_id)


@dataclass
class MetalNoOpICB:
    """An address-stable empty program produced by zero-width operators."""

    def replay(self, replays: int = 1) -> None:
        if type(replays) is not int or replays < 1:
            raise ValueError("Metal replay count must be a positive exact int")

    def close(self) -> None:
        return None


@dataclass
class MetalICB:
    """Persistent fixed-address Metal indirect command buffer."""

    _native: Any
    graph_id: int
    _closed: bool = field(default=False, init=False)

    def replay(self, replays: int = 1) -> None:
        if type(replays) is not int or replays < 1:
            raise ValueError("Metal replay count must be a positive exact int")
        if self._closed:
            raise RuntimeError("Metal ICB is closed")
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
                "Metal command recording",
                (primary, cleanup_error),
            )
            raise error from primary
        raise
    finally:
        _recording_sequence.reset(token)


def recording_metal_sequence() -> MetalCommandSequence | None:
    """Return the sequence active in this Python context, if any."""
    return _recording_sequence.get()

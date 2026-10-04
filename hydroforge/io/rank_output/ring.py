"""The preallocated host ring that carries output rows to rank files.

A writer owns one ring for its whole run.  Every output stream holds
``depth`` batch slots of ``batch`` rows: one slot fills while the others are
appended to its file, and a slot is reused only after that append finished,
so the ring capacity bounds all retained output memory.  Finalized rows are
copied straight into their slot (asynchronously from a CUDA device once the
ring is page-locked); output workers attach the shared ring by name and
append each slot from it, receiving only a small descriptor per batch.
"""

from __future__ import annotations

import errno
import math
import mmap
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from typing import Self

import numpy as np
import torch

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.platform.host_memory import (
    register_host_memory,
    unregister_host_memory,
)

# Ring size the batch plan targets; only a ring within it is page-locked.
RING_BYTES = 512 * 1024 * 1024
# Most rows one append writes; it also sets the HDF5 time chunk.
MAX_BATCH = 30
_ALIGNMENT = 64


@dataclass(frozen=True, slots=True)
class RingSlots:
    """Placement of one stream's ``depth`` batch slots in the ring."""

    offset: int
    depth: int
    batch: int
    row_shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def row_bytes(self) -> int:
        return math.prod(self.row_shape) * self.dtype.itemsize

    def slot_offset(self, slot: int) -> int:
        return self.offset + slot * self.batch * self.row_bytes


def plan_output_batches(
    step_bytes: Mapping[str, int],
    *,
    max_pending_steps: int,
    background_writes: bool = True,
) -> dict[str, tuple[int, int]]:
    """Return each output's ``(depth, batch)`` for :data:`RING_BYTES`.

    ``step_bytes`` is the bytes one step adds to an output (all components).
    A batch is ``min(MAX_BATCH, max_pending_steps)`` rows and ``depth *
    batch`` never exceeds ``max_pending_steps``, so up to that many steps wait
    for their append while the model runs ahead.  Over the ring budget the
    largest outputs first give up slots (down to two, which still overlap
    filling and appending), then halve their batch. In-process writes use
    one slot because appending blocks the owner; their next fill waits for
    that append. Outputs too large even for one-row slots keep that layout:
    their ring exceeds the budget and is not page-locked.
    """

    plans = {}
    for name, size in step_bytes.items():
        batch = min(MAX_BATCH, max_pending_steps, RING_BYTES // max(1, 2 * size))
        batch = max(1, batch)
        depth = max(1, max_pending_steps // batch) if background_writes else 1
        plans[name] = [depth, batch]

    def footprint(name: str) -> int:
        depth, batch = plans[name]
        return depth * batch * step_bytes[name]

    total = sum(map(footprint, plans))
    while total > RING_BYTES:
        deep = [name for name, (depth, _batch) in plans.items() if depth > 2]
        wide = [name for name, (_depth, batch) in plans.items() if batch > 1]
        if not deep and not wide:
            break
        name = max(deep or wide, key=lambda item: (footprint(item), item))
        before = footprint(name)
        if deep:
            plans[name][0] -= 1
        else:
            plans[name][1] //= 2
        total -= before - footprint(name)
    return {name: (depth, batch) for name, (depth, batch) in plans.items()}


def page_lockable(
    step_bytes: Mapping[str, int], plans: Mapping[str, tuple[int, int]]
) -> bool:
    """Whether the planned slots fit :data:`RING_BYTES` and may be page-locked."""

    footprint = sum(
        depth * batch * step_bytes[name] for name, (depth, batch) in plans.items()
    )
    return footprint <= RING_BYTES


def place_slots(
    streams: Mapping[str, tuple[tuple[int, ...], np.dtype, int, int]],
) -> tuple[dict[str, RingSlots], int]:
    """Place each stream's slots of ``(row_shape, dtype, depth, batch)``.

    Returns the placements and the offset where the ring's flag rows start.
    """

    placements = {}
    offset = 0
    for key, (row_shape, dtype, depth, batch) in streams.items():
        slots = RingSlots(offset, depth, batch, row_shape, np.dtype(dtype))
        placements[key] = slots
        offset += -(-depth * batch * slots.row_bytes // _ALIGNMENT) * _ALIGNMENT
    return placements, offset


def _require_shared_memory(nbytes: int) -> None:
    """Fail up front when ``/dev/shm`` cannot back the whole ring."""

    if not os.path.isdir("/dev/shm"):
        return
    status = os.statvfs("/dev/shm")
    available = status.f_bavail * status.f_frsize
    if nbytes > available:
        raise OSError(
            errno.ENOSPC,
            f"statistics output workers need a {nbytes}-byte shared-memory ring "
            f"but /dev/shm has {available} bytes free; enlarge /dev/shm or use "
            "OutputConfig(workers=0) to write output in the model process",
        )


class OutputRing:
    """One writer's ring: shared with its workers, or private without them.

    On a CUDA device a ring within its budget is page-locked for
    asynchronous copies; ``registration_error`` says why a ring is not, and
    copies into it then block.
    """

    def __init__(
        self, memory: SharedMemory | mmap.mmap, nbytes: int, device: torch.device
    ) -> None:
        self._memory = memory
        self._device = device
        buffer = memory.buf if isinstance(memory, SharedMemory) else memory
        self.buffer: np.ndarray | None = np.frombuffer(buffer, np.uint8, count=nbytes)
        self.registered = False
        self.registration_error: str | None = None

    @classmethod
    def create(
        cls, nbytes: int, *, shared: bool, device: torch.device, page_lock: bool
    ) -> Self:
        nbytes = max(nbytes, 1)
        if shared:
            _require_shared_memory(nbytes)
            memory = SharedMemory(create=True, size=nbytes)
        else:
            memory = mmap.mmap(-1, nbytes)
        ring = cls(memory, nbytes, device)
        if device.type == "cuda":
            if not page_lock:
                ring.registration_error = (
                    f"the output rows need more than the {RING_BYTES}-byte "
                    "page-locked ring budget"
                )
            else:
                try:
                    ring.registration_error = register_host_memory(
                        ring.buffer.ctypes.data, nbytes, device
                    )
                except BaseException:
                    with cleanup_on_exit("statistics output ring", (ring.close,)):
                        raise
            ring.registered = ring.registration_error is None
        return ring

    @property
    def name(self) -> str | None:
        """Shared-memory name workers attach, or ``None`` for a private ring."""

        memory = self._memory
        return memory.name if isinstance(memory, SharedMemory) else None

    def array(self, offset: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Return a view of the ring; drop it before :meth:`close`."""

        count = math.prod(shape) * np.dtype(dtype).itemsize
        return self.buffer[offset : offset + count].view(dtype).reshape(shape)

    def close(self) -> None:
        """Unregister, then unmap and unlink a shared ring; all are attempted.

        Pending device copies into the ring must have completed.  A private
        ring is unmapped once its last view is released; a shared one stays
        referenced here, so a view that outlives a failed unmap stays valid.
        """

        base, self.buffer = self.buffer, None
        if base is None:
            return
        address = base.ctypes.data
        registered, self.registered = self.registered, False
        del base
        actions = []
        if registered:
            actions.append(partial(unregister_host_memory, address, self._device))
        memory = self._memory
        if isinstance(memory, SharedMemory):
            actions.extend((memory.close, memory.unlink))
        else:
            self._memory = None
        with cleanup_on_exit("statistics output ring", actions):
            pass


def _attach(name: str) -> SharedMemory:
    """Attach without taking over the owner's unlink responsibility."""

    if sys.version_info >= (3, 13):
        return SharedMemory(name=name, track=False)
    # Spawned workers share the owner's resource tracker; unregistering here
    # would also remove the owner's registration before its unlink.
    return SharedMemory(name=name)


_ATTACHED: dict[str, SharedMemory] = {}


def attached_ring(name: str) -> memoryview:
    """Return the buffer of a shared ring, attaching once per process."""

    memory = _ATTACHED.get(name)
    if memory is None:
        memory = _ATTACHED[name] = _attach(name)
    return memory.buf


def detach_rings() -> None:
    """Close every ring mapping this process attached."""

    memories = tuple(_ATTACHED.values())
    _ATTACHED.clear()
    with cleanup_on_exit(
        "attached statistics output rings", (m.close for m in memories)
    ):
        pass

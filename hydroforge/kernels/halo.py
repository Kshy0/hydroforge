# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The halo exchange as one registered kernel (:mod:`hydroforge.parallel.halo`).

:func:`attach` gives a :class:`~hydroforge.parallel.halo.HaloRuntime` its
``registry``: CUDA launches of ``halo.cu`` (peer stores, staging for NCCL
and Gloo links) on the CUDA backend, torch operations elsewhere, both with
the wire format the plan's message records describe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

import hydroforge.parallel
from hydroforge.kernels.cuda import (
    CudaKernel,
    CudaLaunch,
    CudaSource,
    int32,
    int64,
    pointer,
    struct,
)
from hydroforge.kernels.registry import BackendRegistry
from hydroforge.kernels.spec import (
    HALO_FIELDS,
    HALO_HEADER_WORDS,
    HALO_QUIET_WORDS,
    HaloArgument,
    KernelSpec,
    constant,
)
from hydroforge.kernels.torch import TorchKernel
from hydroforge.parallel.halo import (
    _ADD_TYPES,
    CHUNK,
    DEPTH,
    KEY,
    KINDS,
    MAX_FIELDS,
    PHASES,
    HaloRuntime,
    _aligned,
    _Message,
    _View,
    plain_site,
    unpack_layouts,
)


def attach(runtime: HaloRuntime) -> HaloRuntime:
    """Give ``runtime`` its registered exchange kernel; returns it."""

    runtime.registry = _registry(runtime)
    runtime.plain_site = HaloArgument(plain_site(runtime.plan.world))
    return runtime


# The device API of kernels declaring ``halo`` or ``publishes``
# (``inline_includes``); its comment documents the protocol.
ACCESS_HEADER = Path(hydroforge.parallel.__file__).resolve().parent / "halo_access.cuh"


def publishable(runtime: HaloRuntime) -> str:
    """Why kernels cannot publish or read rings, or ``""``: peer memory on
    the CUDA backend and every pair of every rank on a peer link (alike on
    every rank, so every rank records the same messages)."""

    device = runtime.device
    if runtime.backend != "cuda" or device.type != "cuda" or runtime.memory is None:
        return "publication needs peer memory on the cuda backend"
    if runtime.communicates:
        return "some pair of ranks is reached over nccl or host links"
    return ""


class Published(NamedTuple):
    """The message of one publication (or exchange): its fields and layouts
    in message order."""

    fields: tuple[torch.Tensor, ...]
    layouts: tuple[Any, ...]


def senders(runtime: HaloRuntime, published: Published) -> tuple[int, ...]:
    """The peers whose message of ``published`` this rank receives."""

    views = _field_views(published.fields, published.layouts)
    receives = runtime.messages[("mirror", "receive")][1]
    return tuple(m.peer for m in receives if m.nbytes(views))


def consumes(layout: Any, distance: int, depth: int) -> bool:
    """Whether a kernel reading a field ``distance`` rows away reads every
    ghost row of it (so writes them all back): slab fields read at the
    slab's full ``depth``; partition key fields never (their ghosts are the
    ends of whatever references the model holds, which one kernel need not
    read)."""

    return layout.kind != KEY and bool(depth) and distance >= depth


def _interior(rows: Sequence[int]) -> tuple[int, int]:
    """The longest run [a, b) of consecutive ``rows`` (sorted)."""

    best, start = (0, 0), 0
    for i, row in enumerate(rows):
        if i == 0 or row != rows[i - 1] + 1:
            start = row
        if row + 1 - start > best[1] - best[0]:
            best = (start, row + 1)
    return best


def _field_views(fields, layouts):
    return [_View.of(t, layout) for t, layout in zip(fields, layouts, strict=True)]


def _segments(messages, views):
    """Each message's (byte offset, rows) of every field's segment."""

    result = []
    for message in messages:
        offsets, rows, offset = [0] * MAX_FIELDS, [0] * MAX_FIELDS, 0
        for f, view in enumerate(views):
            count = message.count(view.kind)
            offsets[f], rows[f] = offset, count
            offset += _aligned(view.elements(count) * view.size)
        result.append((offsets, rows))
    return result


def _row_entries(messages, view) -> tuple[list[int], list[int]]:
    """Row pointers over the stored rows of ``view`` and their (message,
    position) entries in ``messages``."""

    entries: list[list[tuple[int, int]]] = [[] for _ in range(view.extent)]
    for m, message in enumerate(messages):
        for position, row in enumerate(message.rows.get(view.kind, ())):
            entries[int(row)].append((m, position))
    rowptr = [0]
    for found in entries:
        rowptr.append(rowptr[-1] + len(found))
    return rowptr, [value for found in entries for pair in found for value in pair]


def _entry(offset: int, peer: int, outer: int) -> tuple[int, int]:
    """A ring element descriptor (``HfRing``): the byte offset of its row's
    element (outer 0, inner 0) in slot 0 from the window holding the ring,
    then the peer (low half) and the elements per outer index (high half)."""

    if not -(2**31) <= peer < 2**31 or not 0 <= outer < 2**31:
        raise ValueError(f"halo descriptor out of range: peer={peer} outer={outer}")
    return offset, (peer & 0xFFFFFFFF) | (outer << 32)


def _outside(extent: int, band: tuple[int, int]) -> list[int]:
    """The rows outside the interior band, in the order kernels index them
    (``row < lo ? row : row - (hi - lo)``)."""

    return [r for r in range(extent) if not band[0] <= r < band[1]]


def access_argument(
    runtime: HaloRuntime | None,
    device: torch.device,
    publish: Sequence[tuple[torch.Tensor, Any]] = (),
    reads: Sequence[tuple[torch.Tensor, Any, Published | None, Mapping[int, int]]] = (),
    commit: Sequence[tuple[int, int, int]] = (),
    *,
    checks: bool = True,
) -> HaloArgument:
    """The ``halo`` argument of one call (``HfHalo`` of halo_access.cuh),
    prepared once per call site: the published fields (tensor, layout), the
    read fields (tensor, layout, the message in the ring holding its ghost
    rows (or ``None``: read them from the field) and, per peer, that
    message's number past ``received``), the prologue's (peer, credits,
    counted) entries, and whether publishing waits for slot credits
    (``checks``: not in a loop body proven free of them).

    Everything a thread needs on every element sits in the struct (passed
    by value: the interior band of each field); the tables describe only
    rows outside it, with each row's ring address precomputed. Arguments
    of equal content are one object (kept by the runtime)."""

    if runtime is None or not (
        publish or commit or any(p is not None for _, _, p, _ in reads)
    ):
        return HaloArgument.quiet()  # every read from the field
    key = (
        tuple((id(t), layout) for t, layout in publish),
        tuple(
            (
                id(t),
                layout,
                None if p is None else (tuple(map(id, p.fields)), p.layouts),
                tuple(sorted(skip.items())),
            )
            for t, layout, p, skip in reads
        ),
        tuple(commit),
        checks,
    )
    prepared = runtime.__dict__.setdefault("access_arguments", {})
    found = prepared.get(key)
    if found is None:
        found = prepared[key] = _access_argument(
            runtime, device, publish, reads, commit, checks
        )
    return found


def _access_argument(runtime, device, publish, reads, commit, checks) -> HaloArgument:
    plan, capacity = runtime.plan, runtime.capacity
    words = list(HALO_QUIET_WORDS)
    tables: list[int] = []
    pointers: list[tuple[int, int]] = []  # (word index, table offset)

    def table(values: list[int]) -> int:
        """Append ``values`` to the tables; a placeholder for their address."""

        offset = len(tables)
        tables.extend(values)
        return offset

    window = runtime.memory.addresses[plan.rank]
    assert DEPTH & (DEPTH - 1) == 0, "ring depth must be a power of two"
    words[:7] = [
        runtime.peers.data_ptr(),
        runtime.state.data_ptr(),
        window,
        capacity,
        DEPTH - 1,  # slot mask (the depth is a power of two)
        plan.world,
        plan.rank,
    ]
    words[11] = int(checks)
    keep: list[Any] = [runtime.peers, runtime.state]
    if commit:
        pointers.append((8, table([len(commit), *(v for e in commit for v in e)])))
    if publish:
        tensors = [t for t, _ in publish]
        layouts = [layout for _, layout in publish]
        views = _field_views(tensors, layouts)
        # Messages without rows of these fields are neither sent nor
        # counted, as by the send kernel.
        sends = [m for m in runtime.messages[("mirror", "send")][1] if m.nbytes(views)]
        runtime._check_capacity(tensors, layouts, "mirror")
        words[7] = sum(v.elements(m.count(v.kind)) for m in sends for v in views)
        pointers.append((9, table([m.peer for m in sends])))
        words[10] = len(sends)
        segments = _segments(sends, views)
        for f, view in enumerate(views):
            rowptr, flat = _row_entries(sends, view)
            band = _interior(
                [r for r in range(view.extent) if rowptr[r + 1] == rowptr[r]]
            )
            outside = _outside(view.extent, band)
            pointer_table, entries = [0], []
            for row in outside:
                for e in range(rowptr[row], rowptr[row + 1]):
                    m, position = flat[2 * e], flat[2 * e + 1]
                    message, (offsets, rows) = sends[m], segments[m]
                    entries += _entry(
                        runtime.ring
                        + message.ring * DEPTH * capacity
                        + offsets[f]
                        + position * view.inner * view.size,
                        message.peer,
                        rows[f] * view.inner,
                    )
                pointer_table.append(len(entries) // 2)
            at = HALO_HEADER_WORDS + 4 * (HALO_FIELDS + f)
            words[at : at + 2] = band
            pointers.append((at + 2, table(pointer_table)))
            pointers.append((at + 3, table(entries)))
    if reads:
        receives = runtime.messages[("mirror", "receive")][1]
        for k, (tensor, layout, published, skip) in enumerate(reads):
            if published is None:
                continue  # interior everywhere: read from the field
            view = _View.of(tensor, layout)
            # The incoming message holds the publication's fields, alike on
            # every rank, with the receive rows of each field's kind.
            group_views = _field_views(published.fields, published.layouts)
            f = [id(t) for t in published.fields].index(id(tensor))
            incoming = [m for m in receives if m.nbytes(group_views)]
            segments = _segments(incoming, group_views)
            rowptr, flat = _row_entries(incoming, view)
            band = _interior(
                [r for r in range(view.extent) if rowptr[r + 1] == rowptr[r]]
            )
            ghosts: list[int] = []
            for row in _outside(view.extent, band):
                if rowptr[row] == rowptr[row + 1]:
                    ghosts += _entry(0, -1, 0)  # not in the ring: the field
                    continue
                m, position = flat[2 * rowptr[row]], flat[2 * rowptr[row] + 1]
                message, (offsets, rows) = incoming[m], segments[m]
                ghosts += _entry(
                    runtime.ring
                    + message.ring * DEPTH * capacity
                    + offsets[f]
                    + position * view.inner * view.size,
                    message.peer,
                    rows[f] * view.inner,
                )
            at = HALO_HEADER_WORDS + 4 * k
            words[at : at + 2] = band
            pointers.append((at + 2, table(ghosts)))
            pointers.append(
                (at + 3, table([int(skip.get(p, 0)) for p in range(plan.world)]))
            )
    tensor = torch.tensor(np.array(tables or [0], np.int64), device=device)
    for at, offset in pointers:
        words[at] = tensor.data_ptr() + 8 * offset
    return HaloArgument(tuple(words), (*keep, tensor))


# --------------------------------------------------------------------------- #
# The registered kernel
# --------------------------------------------------------------------------- #
_PARAMETERS = {
    **{f"f{i}": "read_write" for i in range(MAX_FIELDS)},
    "clear": "read_write",
    # Peers write these; no rollback may restore them.
    "window": "remote",
    "state": "remote",
    "send_stage": "remote",
    "recv_stage": "remote",
    "peers": "read",
    "rows": "read",
    "records": "read",
    "num_entries": "index",
    "nf": constant("int32"),
    "accumulate": constant("bool"),
    "has_clear": constant("bool"),
    "layout_low": constant("int32"),
    "layout_high": constant("int32"),
    "clear_layout": constant("int32"),
    # The schedule's phase, counts and credits (parallel.halo.site_words).
    "site": "halo",
}
_SPEC = KernelSpec(name="halo_exchange", size=("num_entries",), parameters=_PARAMETERS)
# halo.cu and peer.cuh ship with hydroforge.parallel.
_DIR = Path(hydroforge.parallel.__file__).resolve().parent
_THREADS = 512


def _views(arguments: Mapping[str, Any]) -> tuple[list[_View], _View | None]:
    count = arguments["nf"]
    layouts = unpack_layouts(arguments["layout_low"], arguments["layout_high"], count)
    views = [_View.of(arguments[f"f{i}"], layouts[i]) for i in range(count)]
    clear = None
    if arguments["has_clear"]:
        (layout,) = unpack_layouts(arguments["clear_layout"], 0, 1)
        clear = _View.of(arguments["clear"], layout)
    return views, clear


def _fields_struct(views: Sequence[_View], clear: _View | None, written: int):
    slots: list[_View | None] = [*views, *(None,) * (MAX_FIELDS - len(views)), clear]

    def each(function):
        return [function(v) if v is not None else function(None) for v in slots]

    return struct(
        *each(lambda v: pointer(None if v is None else v.tensor)),
        *each(lambda v: int64(0 if v is None else v.outer)),
        *each(lambda v: int64(0 if v is None else v.extent)),
        *each(lambda v: int64(1 if v is None else v.inner)),
        *each(lambda v: int32(1 if v is None else v.size)),
        *each(lambda v: int32(0 if v is None else _ADD_TYPES.get(v.tensor.dtype, 0))),
        *each(lambda v: int32(0 if v is None else KINDS.index(v.kind))),
        int32(len(views)),
        int32(int(clear is not None)),
        int32(written),
    )


def _blocks(message: _Message, views: Sequence[_View], clear: _View | None) -> int:
    longest = sum(v.elements(message.count(v.kind)) for v in views)
    if clear is not None:
        longest = max(longest, clear.elements(message.count(clear.kind)))
    return -(-longest // CHUNK)


def _nccl_step(runtime: HaloRuntime, sends, receives):
    """NCCL send/receive (or host-staged Gloo) of the non-peer messages."""

    import torch.distributed as dist

    group, capacity = runtime.group, runtime.capacity
    if not sends and not receives:
        # This call's fields have no rows on these links.
        return lambda: None
    host = any(m.link == "host" for m, _n in (*sends, *receives))
    # Messages name group ranks; point-to-point calls take global ranks.
    peer = {
        m.peer: m.peer if group is None else dist.get_global_rank(group, m.peer)
        for m, _n in (*sends, *receives)
    }

    if host:
        # Gloo moves host memory: the staging slots themselves on a CPU
        # device, else host mirrors of them (pinned), allocated once.
        send_host, recv_host = runtime.host_stages()
        send_slots = [
            (slice(m.stage * capacity, m.stage * capacity + n), peer[m.peer])
            for m, n in sends
        ]
        recv_slots = [
            (slice(m.stage * capacity, m.stage * capacity + n), peer[m.peer])
            for m, n in receives
        ]
        mirrored = send_host is not runtime.send_stage

    def run() -> None:
        if host:
            requests = []
            for rows, rank in send_slots:
                if mirrored:
                    send_host[rows].copy_(runtime.send_stage[rows])
                requests.append(dist.isend(send_host[rows], rank, group))
            for rows, rank in recv_slots:
                requests.append(dist.irecv(recv_host[rows], rank, group))
            for request in requests:
                request.wait()
            if mirrored:
                # Synchronous copies: the mirrors are free again on return.
                for rows, _rank in recv_slots:
                    runtime.recv_stage[rows].copy_(recv_host[rows])
            return
        operations = [
            dist.P2POp(
                dist.isend,
                runtime.send_stage[m.stage * capacity : m.stage * capacity + n],
                peer[m.peer],
                group,
            )
            for m, n in sends
        ] + [
            dist.P2POp(
                dist.irecv,
                runtime.recv_stage[m.stage * capacity : m.stage * capacity + n],
                peer[m.peer],
                group,
            )
            for m, n in receives
        ]
        for request in dist.batch_isend_irecv(operations):
            request.wait()

    return run


def cuda_steps(runtime: HaloRuntime, arguments: Mapping[str, Any]) -> list[Any]:
    """The launches (and NCCL step) of one exchange on the CUDA backend; its
    first launch runs the site's prologue (halo.cu)."""

    views, clear = _views(arguments)
    mode = "accumulate" if arguments["accumulate"] else "mirror"
    flag = "true" if arguments["accumulate"] else "false"
    site = arguments["site"]
    words = plain_site(runtime.plan.world) if site is None else site.words
    phase_index, written, checks, *table = words
    phase = PHASES[phase_index]
    fields = _fields_struct(views, clear, written)
    ring = struct(
        pointer(arguments["window"]),
        pointer(arguments["peers"]),
        pointer(arguments["state"]),
        pointer(arguments["send_stage"]),
        pointer(arguments["recv_stage"]),
        pointer(arguments["rows"]),
        pointer(arguments["records"]),
        int64(runtime.capacity),
        int64(runtime.ring),
        int64(runtime.entries),
        pointer(runtime.site_table(table)),
        int32(runtime.plan.world),
        int32(runtime.plan.rank),
        int32(DEPTH),
        int32(CHUNK),
        int32(checks),
    )

    def staged(direction: str) -> list[tuple[_Message, int]]:
        found = runtime.messages[(mode, direction)][1]
        return [
            (m, m.nbytes(views)) for m in found if m.link != "peer" and m.nbytes(views)
        ]

    send_first, sends = runtime.messages[(mode, "send")]
    receive_first, receives = runtime.messages[(mode, "receive")]
    steps: list[Any] = []

    def grid(messages, extra=None) -> int:
        return max([1, *(_blocks(m, views, extra) for m in messages)])

    def begin() -> int32:
        """1 for the exchange's first launch."""

        return int32(int(not any(isinstance(s, CudaLaunch) for s in steps)))

    remote = staged("send") or staged("receive")
    if phase != "both":
        # Ring messages the schedule sends or receives apart (publishable).
        if remote:
            raise ValueError(f"a {phase} phase exchanges over peer links only")
        if phase == "send" and sends:
            # Messages a receiver finds pending (never added: no clearing).
            steps.append(
                CudaLaunch(
                    "halo_send<false>",
                    (grid(sends, clear), len(sends)),
                    _THREADS,
                    (fields, ring, int32(send_first), int32(1)),
                )
            )
        elif phase == "receive" and receives:
            steps.append(
                CudaLaunch(
                    "halo_receive<false>",
                    (grid(receives), len(receives)),
                    _THREADS,
                    (fields, ring, int32(receive_first), int32(0), int32(1), int32(1)),
                )
            )
        elif phase in ("book", "drain", "flush"):
            steps.append(CudaLaunch(f"halo_{phase}", (1, 1), 32, (ring,)))
        return steps
    if mode == "mirror" and not remote and sends and receives:
        steps.append(
            CudaLaunch(
                "halo_mirror",
                (grid([*sends, *receives]), len(sends) + len(receives)),
                _THREADS,
                (
                    fields,
                    ring,
                    int32(send_first),
                    int32(len(sends)),
                    int32(receive_first),
                ),
            )
        )
        return steps
    if sends:
        steps.append(
            CudaLaunch(
                f"halo_send<{flag}>",
                (grid(sends, clear), len(sends)),
                _THREADS,
                (fields, ring, int32(send_first), begin()),
            )
        )
    if remote:
        steps.append(
            _nccl_step(
                runtime,
                staged("send"),
                staged("receive"),
            )
        )
    if not views:
        return steps  # zeroing the ghost rows of clear only: no receive
    if receives and mode == "mirror":
        steps.append(
            CudaLaunch(
                "halo_receive<false>",
                (grid(receives), len(receives)),
                _THREADS,
                (fields, ring, int32(receive_first), int32(0), int32(1), begin()),
            )
        )
    elif receives:
        # Owners add their peers' contributions in ascending peer order, a
        # message whose rows repeat in one pass per entry of a row.
        for index, message in enumerate(receives):
            for step in range(message.passes):
                steps.append(
                    CudaLaunch(
                        "halo_receive<true>",
                        (grid([message]), 1),
                        _THREADS,
                        (
                            fields,
                            ring,
                            int32(receive_first + index),
                            int32(step),
                            int32(int(step == message.passes - 1)),
                            begin(),
                        ),
                    )
                )
    return steps


def _registry(runtime: HaloRuntime) -> BackendRegistry:
    """The exchange of ``runtime`` as one registered kernel."""

    def cuda_launch(**arguments):
        return cuda_steps(runtime, arguments)

    def halo_exchange(**arguments):
        for step in _torch_steps(runtime, arguments):
            step()

    cuda = CudaKernel(
        CudaSource(path=_DIR / "halo.cu", include_root=_DIR),
        launch=_named(cuda_launch, "BLOCK_SIZE"),
        communicates=runtime.communicates,
        capturable=runtime.capturable,
    )
    torch_kernel = TorchKernel(
        _named(halo_exchange),
        compile=False,
        communicates=runtime.communicates,
        capturable=runtime.capturable,
    )
    return BackendRegistry(
        _SPEC,
        {
            "cuda": cuda,
            **{name: torch_kernel for name in ("triton", "torch", "metal")},
        },
    )


def _named(function, *extra: str):
    """``function(**arguments)`` with the spec's parameters (and ``extra``)
    as its signature."""

    import inspect

    names = (*_SPEC.parameter_names, *extra)
    parameters = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in names
    ]

    def wrapper(*args, **kwargs):
        bound = dict(zip(names, args, strict=False))
        bound.update(kwargs)
        return function(**bound)

    wrapper.__signature__ = inspect.Signature(parameters)
    wrapper.__name__ = function.__name__
    return wrapper


def _torch_steps(runtime: HaloRuntime, arguments: Mapping[str, Any]):
    """The exchange in torch operations, the same wire format as halo.cu."""

    views, clear = _views(arguments)
    mode = "accumulate" if arguments["accumulate"] else "mirror"
    site = arguments.get("site")
    if site is not None and PHASES[site.words[0]] != "both":
        raise ValueError("only peer links of the cuda backend exchange in phases")
    capacity = runtime.capacity
    index = runtime.index
    repeats = runtime.repeats

    sends = runtime.messages[(mode, "send")][1]
    receives = runtime.messages[(mode, "receive")][1]

    def segments(message: _Message, direction: str):
        """(field, its rows, byte offset, bytes) of a message, in wire order."""

        offset = 0
        for view in views:
            rows = message.count(view.kind)
            nbytes = view.elements(rows) * view.size
            if rows:
                key = (mode, direction, message.peer, view.kind)
                yield view, index[key], offset, nbytes
            offset += _aligned(nbytes)

    def send() -> None:
        for message in sends:
            base = message.stage * capacity
            for view, rows, offset, nbytes in segments(message, "send"):
                shaped = view.tensor.view(view.outer, view.extent, view.inner)
                block = shaped.index_select(1, rows).contiguous().view(-1)
                runtime.send_stage[base + offset : base + offset + nbytes].copy_(
                    block.view(torch.uint8)
                )
                if mode == "accumulate":
                    shaped.index_fill_(1, rows, 0)
            if clear is not None and message.count(clear.kind):
                rows = index[(mode, "send", message.peer, clear.kind)]
                clear.tensor.view(clear.outer, clear.extent, clear.inner).index_fill_(
                    1, rows, 0
                )

    def receive() -> None:
        for message in receives:
            base = message.stage * capacity
            for view, rows, offset, nbytes in segments(message, "receive"):
                raw = runtime.recv_stage[base + offset : base + offset + nbytes]
                values = raw.view(view.tensor.dtype).view(view.outer, -1, view.inner)
                shaped = view.tensor.view(view.outer, view.extent, view.inner)
                passes = repeats.get((mode, "receive", message.peer, view.kind))
                if passes is not None:
                    # Each pass adds into a row once, in the CUDA order.
                    for targets, entries in passes:
                        shaped.index_add_(1, targets, values.index_select(1, entries))
                elif mode == "accumulate":
                    shaped.index_add_(1, rows, values)
                else:
                    shaped.index_copy_(1, rows, values)

    staged_sends = [(m, m.nbytes(views)) for m in sends if m.nbytes(views)]
    staged_receives = [(m, m.nbytes(views)) for m in receives if m.nbytes(views)]
    return (send, _nccl_step(runtime, staged_sends, staged_receives), receive)


__all__ = [
    "ACCESS_HEADER",
    "Published",
    "access_argument",
    "attach",
    "consumes",
    "cuda_steps",
    "publishable",
    "senders",
]

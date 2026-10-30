# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The halo exchange as one registered kernel (:mod:`hydroforge.parallel.halo`).

:func:`attach` gives a :class:`~hydroforge.parallel.halo.HaloRuntime` its
``registry``: CUDA launches of ``halo.cu`` (peer stores, staging for NCCL
and Gloo links) on the CUDA backend, torch operations elsewhere, both with
the wire format the plan's message records describe. The exchange is a
program step of its own: model kernels never take part in it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from hydroforge.kernels.spec import KernelSpec, constant
from hydroforge.kernels.torch import TorchKernel
from hydroforge.parallel.halo import (
    _ADD_TYPES,
    CHUNK,
    DEPTH,
    KINDS,
    MAX_FIELDS,
    HaloRuntime,
    _aligned,
    _Message,
    _View,
    code_layout,
)


def attach(runtime: HaloRuntime) -> HaloRuntime:
    """Give ``runtime`` its registered exchange kernel; returns it."""

    runtime.registry = _registry(runtime)
    return runtime


# --------------------------------------------------------------------------- #
# The registered kernel
# --------------------------------------------------------------------------- #
_PARAMETERS = {
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
    "clear_layout": constant("int32"),
}
# halo.cu and peer.cuh ship with hydroforge.parallel.
_DIR = Path(hydroforge.parallel.__file__).resolve().parent
_THREADS = 512


def _views(arguments: Mapping[str, Any]) -> tuple[list[_View], _View | None]:
    count = arguments["nf"]
    layouts = [code_layout(arguments[f"layout{i}"]) for i in range(count)]
    views = [_View.of(arguments[f"f{i}"], layouts[i]) for i in range(count)]
    clear = None
    if arguments["has_clear"]:
        clear = _View.of(arguments["clear"], code_layout(arguments["clear_layout"]))
    return views, clear


def _fields_struct(views: Sequence[_View], clear: _View | None):
    """HaloFields of halo.cu, by value."""

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
        *each(lambda v: int32(0 if v is None else v.depth)),
        int32(len(views)),
        int32(int(clear is not None)),
    )


def _blocks(message: _Message, views: Sequence[_View], clear: _View | None) -> int:
    longest = sum(v.elements(message.count(v.kind, v.depth)) for v in views)
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
    """The launches (and NCCL step) of one exchange on the CUDA backend
    (halo.cu)."""

    views, clear = _views(arguments)
    mode = "accumulate" if arguments["accumulate"] else "mirror"
    flag = "true" if arguments["accumulate"] else "false"
    fields = _fields_struct(views, clear)
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
        int32(runtime.plan.world),
        int32(runtime.plan.rank),
        int32(DEPTH),
        int32(CHUNK),
        int32(runtime.record),
        int32(runtime.plan.levels),
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

    remote = staged("send") or staged("receive")
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
                (fields, ring, int32(send_first)),
            )
        )
    if remote:
        steps.append(_nccl_step(runtime, staged("send"), staged("receive")))
    if not views:
        return steps  # zeroing the ghost rows of clear only: no receive
    if receives and mode == "mirror":
        steps.append(
            CudaLaunch(
                "halo_receive<false>",
                (grid(receives), len(receives)),
                _THREADS,
                (fields, ring, int32(receive_first), int32(0), int32(1)),
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

    spec = KernelSpec(
        name="halo_exchange",
        size=("num_entries",),
        parameters={
            **{f"f{i}": "read_write" for i in range(MAX_FIELDS)},
            **{f"layout{i}": constant("int32") for i in range(MAX_FIELDS)},
            **_PARAMETERS,
        },
    )
    cuda = CudaKernel(
        CudaSource(path=_DIR / "halo.cu", include_root=_DIR),
        launch=_named(cuda_launch, spec, "BLOCK_SIZE"),
        communicates=runtime.communicates,
        capturable=runtime.capturable,
    )
    torch_kernel = TorchKernel(
        _named(halo_exchange, spec),
        compile=False,
        communicates=runtime.communicates,
        capturable=runtime.capturable,
    )
    return BackendRegistry(
        spec,
        {
            "cuda": cuda,
            **{name: torch_kernel for name in ("triton", "torch", "metal")},
        },
    )


def _named(function, spec, *extra: str):
    """``function(**arguments)`` with the spec's parameters (and ``extra``)
    as its signature."""

    import inspect

    names = (*spec.parameter_names, *extra)
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
    capacity = runtime.capacity
    index = runtime.index
    repeats = runtime.repeats

    sends = runtime.messages[(mode, "send")][1]
    receives = runtime.messages[(mode, "receive")][1]

    def segments(message: _Message, direction: str):
        """(field, its rows, byte offset, bytes) of a message, in wire order."""

        offset = 0
        for view in views:
            rows = message.count(view.kind, view.depth)
            nbytes = view.elements(rows) * view.size
            if rows:
                key = (mode, direction, message.peer, view.kind)
                yield view, index[key][:rows], offset, nbytes
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


__all__ = ["attach", "cuda_steps"]

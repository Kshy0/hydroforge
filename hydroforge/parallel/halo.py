# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Ghost rows and GPU-initiated halo exchange of the partition coordinate.

A partition group need not be closed under the references to the partition
key: when one splits connected structure (for example a river basin), each
rank also holds read-only *ghost* rows of the remote partition-key rows that
its rows reference. Ghost rows follow the owned rows of every field aligned
to the coordinate, so local arrays are ``owned + ghosts`` long and
:meth:`AbstractModule.owned_rows` is the extent kernels iterate. The ghosts
are derived from the global partition on every rank alike, without
collectives; when every group is closed there are none, and no buffer,
symmetric memory or exchange kernel exists.

:meth:`AbstractModel.halo_exchange_` moves fields between owners and ghosts
inside a managed step: ``"mirror"`` copies owner values to ghosts and
``"accumulate"`` adds what landed on ghosts to their owners (zeroing the
ghosts). It is a registered kernel, recorded and CUDA-graph capturable. A
rank without messages in an exchange launches nothing.

The transport is GPU-initiated and CUDA-only. Senders store straight into
the receiving peer's buffer and release an arrival signal the receiver
acquires (:mod:`hydroforge.parallel.symmetric`). The ``cuda`` backend does
so in CUDA C++ over CUDA IPC mappings (ranks of one node); the ``triton``
backend allocates NVSHMEM memory and, across nodes or with
``HYDROFORGE_HALO_TRANSPORT=put``, puts each message with a signal.

Signals count arrivals against each receiver's own count, so the protocol
stays consistent under any number of identical executions on every rank
(such as capture warm-ups); its buffers use the ``remote`` access mode,
which rollbacks never restore.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from hydroforge.kernels.cuda import (
    CudaKernel,
    CudaLaunch,
    CudaSource,
    ctype,
    int32,
    pointer,
)
from hydroforge.kernels.registry import BackendRegistry
from hydroforge.kernels.spec import KernelSpec, constant
from hydroforge.kernels.torch import TorchKernel
from hydroforge.kernels.triton import TritonProgram
from hydroforge.parallel.symmetric import (
    PeerBuffers,
    RankFacts,
    gather_rank_facts,
    ipc_problem,
    nvshmem_problem,
)

MAX_FIELDS = 3
SLOTS = 8
CHUNK = 1024  # entries one thread block moves


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
def ghost_rows(
    rank_rows: Sequence[np.ndarray],
    references: Sequence[tuple[np.ndarray, Sequence[np.ndarray]]],
) -> list[np.ndarray]:
    """Ghost rows of every rank, sorted global row indices.

    ``rank_rows[r]`` holds rank ``r``'s owned global rows of the coordinate;
    each reference is ``(targets, local_rows)``: the global target row of
    every source row, and every rank's owned source rows.
    """

    world = len(rank_rows)
    owner = np.full(sum(len(r) for r in rank_rows), -1, np.int64)
    for r, rows in enumerate(rank_rows):
        owner[rows] = r
    ghosts = []
    for r in range(world):
        targets = [t[rows[r]] for t, rows in references]
        targets = np.concatenate(targets) if targets else np.zeros(0, np.int64)
        targets = targets[(targets >= 0) & (owner[targets] != r)]
        ghosts.append(np.unique(targets))
    return ghosts


@dataclass(frozen=True)
class _Direction:
    """Offsets into the plan's index array of one direction's messages."""

    out_peer: int
    out_start: int
    out_dst: int
    out_idx: int
    in_peer: int
    in_start: int
    in_off: int
    in_idx: int
    n_out: int
    n_in: int
    chunks: int  # most chunks of one message


class HaloPlan:
    """Pairwise owner <-> ghost messages of one rank, from the global layout."""

    def __init__(
        self, rank: int, rank_rows: Sequence[np.ndarray], ghosts: Sequence[np.ndarray]
    ) -> None:
        world = len(rank_rows)
        self.rank, self.world = rank, world
        self.n_owned = len(rank_rows[rank])
        self.n_ghosts = len(ghosts[rank])
        owner = np.full(sum(len(r) for r in rank_rows), -1, np.int64)
        local = np.zeros_like(owner)
        for r, rows in enumerate(rank_rows):
            owner[rows] = r
            local[rows] = np.arange(len(rows))
        # pairs[q][p]: (q-local owned rows, p-local ghost rows) of p's ghosts owned by q.
        pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        counts = np.zeros((world, world), np.int64)
        for p, rows in enumerate(ghosts):
            for q in np.unique(owner[rows]):
                mine = np.flatnonzero(owner[rows] == q)
                pairs[(int(q), p)] = (local[rows[mine]], len(rank_rows[p]) + mine)
                counts[q, p] = len(mine)
        self.counts = counts
        self.total_ghosts = int(counts.sum())
        self.index: list[int] = []
        me = rank
        sends = {p: pairs[(me, p)][0] for p in range(world) if (me, p) in pairs}
        recvs = {q: pairs[(q, me)][1] for q in range(world) if (q, me) in pairs}
        self.mirror = self._direction(sends, recvs, counts)  # owner -> ghost
        self.accumulate = self._direction(recvs, sends, counts.T)  # ghost -> owner
        self.capacity = (
            max(int(counts.sum(0).max()), int(counts.sum(1).max()), 1) * MAX_FIELDS
        )
        self.max_chunks = max(-(-int(counts.max()) // CHUNK), 1)

    def _append(self, values) -> int:
        offset = len(self.index)
        self.index.extend(int(v) for v in values)
        return offset

    def _direction(self, sends, recvs, counts) -> _Direction:
        me = self.rank
        out_peer, out_start, out_dst, out_idx = [], [0], [], []
        for p in sorted(sends):
            out_peer.append(p)
            out_idx.extend(sends[p])
            out_start.append(len(out_idx))
            # My segment of p's receive buffer follows lower-ranked senders.
            out_dst.append(int(counts[:me, p].sum()))
        in_peer, in_start, in_off, in_idx = [], [0], [], []
        for q in sorted(recvs):
            in_peer.append(q)
            in_off.append(int(counts[:q, me].sum()))
            in_idx.extend(recvs[q])
            in_start.append(len(in_idx))
        longest = max([len(v) for v in (*sends.values(), *recvs.values())], default=0)
        a = self._append
        return _Direction(
            a(out_peer),
            a(out_start),
            a(out_dst),
            a(out_idx),
            a(in_peer),
            a(in_start),
            a(in_off),
            a(in_idx),
            len(out_peer),
            len(in_peer),
            max(-(-longest // CHUNK), 1),
        )


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
_SPEC = KernelSpec(
    name="halo_exchange",
    size=("num_chunks", "num_messages"),
    parameters={
        "f0": "read_write",
        "f1": "read_write",
        "f2": "read_write",
        "clear": "read_write",
        "index": "read",
        "stage": "read_write",
        # Peers write these; no rollback may restore them.
        "recv": "remote",
        "signals": "remote",
        "expected": "remote",
        # Peers' receive buffers and signals, as P2P addresses (direct mode).
        "recv_peers": "read",
        "signal_peers": "read",
        **dict.fromkeys(
            (
                "out_peer",
                "out_start",
                "out_dst",
                "out_idx",
                "in_peer",
                "in_start",
                "in_off",
                "in_idx",
            ),
            "int32",
        ),
        "slot_offset": "int32",
        "signal_offset": "int32",
        "num_chunks": "index",
        "num_messages": "index",
        "n_out": "int32",
        "my_pe": "int32",
        "nf": constant("int32"),
        "accumulate": constant("bool"),
        "has_clear": constant("bool"),
        "direct": constant("bool"),
        "max_chunks": constant("int32"),
        "num_pes": constant("int32"),
    },
)


def _kernel():
    import triton
    import triton.language as tl
    from torch.distributed._symmetric_memory import _nvshmem_triton as nvshmem

    # nvshmem_common_transport.h / nvshmem_constants.h.
    signal_add = tl.constexpr(10)
    cmp_ge = tl.constexpr(5)
    chunk = tl.constexpr(CHUNK)

    @nvshmem.requires_nvshmem
    @triton.jit
    def halo_exchange_kernel(
        f0,
        f1,
        f2,
        clear,
        index,
        stage,
        recv,
        signals,
        expected,
        recv_peers,
        signal_peers,
        out_peer,
        out_start,
        out_dst,
        out_idx,
        in_peer,
        in_start,
        in_off,
        in_idx,
        slot_offset,
        signal_offset,
        n_out,
        my_pe,
        nf: tl.constexpr,
        accumulate: tl.constexpr,
        has_clear: tl.constexpr,
        direct: tl.constexpr,
        max_chunks: tl.constexpr,
        num_pes: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)  # chunk of the message
        m = tl.program_id(1)  # message: sends first, then receives
        stage = stage + slot_offset
        recv = recv + slot_offset
        if m < n_out:
            peer = tl.load(index + out_peer + m)
            base = tl.load(index + out_start + m)
            start = base + c * chunk
            end = tl.minimum(tl.load(index + out_start + m + 1), start + chunk)
            if start < end:
                if direct:
                    # NVLink: write into the peer's receive buffer itself.
                    peer_recv = tl.load(recv_peers + peer).to(
                        tl.pointer_type(tl.float64), bitcast=True
                    )
                    out = (
                        peer_recv
                        + slot_offset
                        + (tl.load(index + out_dst + m) - base) * nf
                    )
                else:
                    out = stage
                for first in range(start, end, BLOCK):
                    offs = first + tl.arange(0, BLOCK)
                    mask = offs < end
                    row = tl.load(index + out_idx + offs, mask=mask, other=0)
                    slot = offs * nf
                    v = tl.load(f0 + row, mask=mask, other=0.0)
                    tl.store(out + slot, v.to(tl.float64), mask=mask)
                    if accumulate:
                        tl.store(f0 + row, tl.zeros_like(v), mask=mask)
                    if nf > 1:
                        v = tl.load(f1 + row, mask=mask, other=0.0)
                        tl.store(out + slot + 1, v.to(tl.float64), mask=mask)
                        if accumulate:
                            tl.store(f1 + row, tl.zeros_like(v), mask=mask)
                    if nf > 2:
                        v = tl.load(f2 + row, mask=mask, other=0.0)
                        tl.store(out + slot + 2, v.to(tl.float64), mask=mask)
                        if accumulate:
                            tl.store(f2 + row, tl.zeros_like(v), mask=mask)
                    if has_clear:
                        v = tl.load(clear + row, mask=mask, other=0.0)
                        tl.store(clear + row, tl.zeros_like(v), mask=mask)
                tl.debug_barrier()
                if direct:
                    # One system-scope release orders the whole block's stores
                    # (cumulative after the barrier) before the arrival.
                    peer_signal = tl.load(signal_peers + peer).to(
                        tl.pointer_type(tl.int64), bitcast=True
                    )
                    lane = tl.arange(0, BLOCK)
                    tl.atomic_add(
                        peer_signal
                        + signal_offset
                        + (my_pe * max_chunks + c)
                        + lane * 0,
                        tl.where(lane == 0, 1, 0).to(tl.int64),
                        mask=lane == 0,
                        sem="release",
                        scope="sys",
                    )
                else:
                    dst = tl.load(index + out_dst + m) + (start - base)
                    nvshmem.putmem_signal_block(
                        recv + dst * nf,
                        stage + start * nf,
                        (end - start).to(tl.int64) * nf * 8,
                        signals + signal_offset + (my_pe * max_chunks + c),
                        1,
                        signal_add,
                        peer,
                    )
        else:
            j = m - n_out
            peer = tl.load(index + in_peer + j)
            base = tl.load(index + in_start + j)
            start = base + c * chunk
            end = tl.minimum(tl.load(index + in_start + j + 1), start + chunk)
            if start < end:
                count = expected + signal_offset + (peer * max_chunks + c)
                arrivals = tl.load(count) + 1
                nvshmem.signal_wait_until(
                    signals + signal_offset + (peer * max_chunks + c),
                    cmp_ge,
                    arrivals,
                )
                tl.store(count, arrivals)
                origin = tl.load(index + in_off + j) - base
                for first in range(start, end, BLOCK):
                    offs = first + tl.arange(0, BLOCK)
                    mask = offs < end
                    row = tl.load(index + in_idx + offs, mask=mask, other=0)
                    slot = (origin + offs) * nf
                    # Written by a peer during this kernel: bypass caches.
                    v = tl.load(recv + slot, mask=mask, other=0.0, volatile=True)
                    if accumulate:
                        tl.atomic_add(f0 + row, v.to(f0.dtype.element_ty), mask=mask)
                    else:
                        tl.store(f0 + row, v.to(f0.dtype.element_ty), mask=mask)
                    if nf > 1:
                        v = tl.load(
                            recv + slot + 1, mask=mask, other=0.0, volatile=True
                        )
                        if accumulate:
                            tl.atomic_add(
                                f1 + row, v.to(f1.dtype.element_ty), mask=mask
                            )
                        else:
                            tl.store(f1 + row, v.to(f1.dtype.element_ty), mask=mask)
                    if nf > 2:
                        v = tl.load(
                            recv + slot + 2, mask=mask, other=0.0, volatile=True
                        )
                        if accumulate:
                            tl.atomic_add(
                                f2 + row, v.to(f2.dtype.element_ty), mask=mask
                            )
                        else:
                            tl.store(f2 + row, v.to(f2.dtype.element_ty), mask=mask)

    return halo_exchange_kernel


def _prepare(arguments, buffer_dtypes):
    del buffer_dtypes
    kernel = _kernel()
    grid = (arguments["num_chunks"], arguments["num_messages"])
    values = {
        k: v
        for k, v in arguments.items()
        if k not in ("num_chunks", "num_messages", "BLOCK_SIZE")
    }
    # Compile, and initialize the module's NVSHMEM state, before any capture.
    kernel.jit_func.warmup(
        **values, BLOCK=128, grid=grid, extern_libs=kernel.extern_libs
    )
    launcher = kernel[grid]

    def launch():
        launcher(**values, BLOCK=128)

    return launch


_DIR = Path(__file__).resolve().parent


def _cuda_launch(
    f0,
    f1,
    f2,
    clear,
    index,
    stage,
    recv,
    signals,
    expected,
    recv_peers,
    signal_peers,
    out_peer,
    out_start,
    out_dst,
    out_idx,
    in_peer,
    in_start,
    in_off,
    in_idx,
    slot_offset,
    signal_offset,
    num_chunks,
    num_messages,
    n_out,
    my_pe,
    nf,
    accumulate,
    has_clear,
    direct,
    max_chunks,
    num_pes,
    BLOCK_SIZE,
):
    del stage, num_pes, BLOCK_SIZE
    if not direct:
        raise ValueError(
            "the cuda backend's halo exchange stores into peers mapped "
            "peer-to-peer; the triton backend puts through NVSHMEM"
        )
    flag = {True: "true", False: "false"}
    kernel = (
        f"halo_exchange<{ctype(f0.dtype)}, {ctype(f1.dtype)}, {ctype(f2.dtype)}, "
        f"{ctype(clear.dtype)}, {nf}, {flag[accumulate]}, {flag[has_clear]}>"
    )
    arguments = (
        *(
            pointer(t)
            for t in (
                f0,
                f1,
                f2,
                clear,
                index,
                recv,
                signals,
                expected,
                recv_peers,
                signal_peers,
            )
        ),
        *(
            int32(v)
            for v in (
                out_peer,
                out_start,
                out_dst,
                out_idx,
                in_peer,
                in_start,
                in_off,
                in_idx,
                slot_offset,
                signal_offset,
                n_out,
                my_pe,
                max_chunks,
                CHUNK,
            )
        ),
    )
    return (CudaLaunch(kernel, (num_chunks, num_messages), 256, arguments),)


_CUDA_KERNEL = CudaKernel(
    CudaSource(path=_DIR / "halo.cu", include_root=_DIR), launch=_cuda_launch
)
# Direct stores into CUDA IPC mappings: the CUDA C++ kernel.
halo_exchange_cuda = BackendRegistry(_SPEC, {"cuda": _CUDA_KERNEL})
# NVSHMEM memory: direct stores within an NVLink domain, puts across nodes.
halo_exchange_nvshmem = BackendRegistry(_SPEC, {"triton": TritonProgram(_prepare)})


# --------------------------------------------------------------------------- #
# NCCL send/recv
# --------------------------------------------------------------------------- #
_NCCL_SPEC = KernelSpec(
    name="halo_exchange_nccl",
    size=("num_entries",),
    parameters={
        "f0": "read_write",
        "f1": "read_write",
        "f2": "read_write",
        "clear": "read_write",
        "send": "read_write",
        "recv": "read_write",
        "sources": "read",
        "targets": "read",
        "num_entries": "index",
        "nf": constant("int32"),
        "accumulate": constant("bool"),
        "has_clear": constant("bool"),
    },
)


@dataclass(frozen=True)
class _Messages:
    """One direction's rows and per-peer message lengths, in peer order."""

    sources: torch.Tensor
    targets: torch.Tensor
    sends: tuple[tuple[int, int], ...]
    receives: tuple[tuple[int, int], ...]


def _messages(plan: HaloPlan, d: _Direction, device: torch.device) -> _Messages:
    index = plan.index

    def segments(peer: int, start: int, count: int) -> tuple[tuple[int, int], ...]:
        bounds = index[start : start + count + 1]
        peers = index[peer : peer + count]
        return tuple((int(p), int(b - a)) for p, a, b in zip(peers, bounds, bounds[1:]))

    sends = segments(d.out_peer, d.out_start, d.n_out)
    receives = segments(d.in_peer, d.in_start, d.n_in)
    out_rows = index[d.out_idx : d.out_idx + sum(n for _p, n in sends)]
    in_rows = index[d.in_idx : d.in_idx + sum(n for _p, n in receives)]
    return _Messages(
        torch.tensor(out_rows, dtype=torch.int64, device=device),
        torch.tensor(in_rows, dtype=torch.int64, device=device),
        sends,
        receives,
    )


def _nccl_exchange(
    mirror: _Messages, accumulate_messages: _Messages, group: Any
) -> BackendRegistry:
    """The NCCL send/recv exchange of one plan, as a registered kernel.

    Pack the rows to send, exchange every peer's message in one NCCL group
    and apply the received rows peer by peer, so accumulated sums keep one
    order. NCCL point-to-point operations are CUDA-graph capturable.
    """

    import torch.distributed as dist

    def halo_exchange_nccl(
        f0,
        f1,
        f2,
        clear,
        send,
        recv,
        sources,
        targets,
        num_entries,
        nf,
        accumulate,
        has_clear,
    ):
        del num_entries
        messages = accumulate_messages if accumulate else mirror
        fields = (f0, f1, f2)[:nf]
        packed = send[: sources.numel() * nf].view(-1, nf)
        for k, field in enumerate(fields):
            packed[:, k] = field.index_select(0, sources)
            if accumulate:
                field.index_fill_(0, sources, 0)
        if has_clear:
            clear.index_fill_(0, sources, 0)
        operations, offset = [], 0
        for peer, count in messages.sends:
            message = send[offset * nf : (offset + count) * nf]
            operations.append(dist.P2POp(dist.isend, message, peer, group))
            offset += count
        offset = 0
        for peer, count in messages.receives:
            message = recv[offset * nf : (offset + count) * nf]
            operations.append(dist.P2POp(dist.irecv, message, peer, group))
            offset += count
        if operations:
            for request in dist.batch_isend_irecv(operations):
                request.wait()
        received = recv[: targets.numel() * nf].view(-1, nf)
        offset = 0
        for _peer, count in messages.receives:
            rows = targets[offset : offset + count]
            part = received[offset : offset + count]
            for k, field in enumerate(fields):
                values = part[:, k].to(field.dtype)
                if accumulate:
                    field.index_add_(0, rows, values)
                else:
                    field.index_copy_(0, rows, values)
            offset += count

    kernel = TorchKernel(halo_exchange_nccl, compile=False)
    return BackendRegistry(_NCCL_SPEC, {"cuda": kernel, "triton": kernel})


# --------------------------------------------------------------------------- #
# Transport selection
# --------------------------------------------------------------------------- #
TRANSPORTS = ("peer", "nvshmem", "nccl")
"""Halo transports, fastest first."""


def transport_problems(
    backend: str, facts: tuple[RankFacts, ...], group_backend: str
) -> dict[str, str | None]:
    """Why each transport cannot connect these ranks (``None``: it can).

    Decided from facts every rank gathered alike: ``peer`` stores into
    peers' memory over CUDA IPC (one node, outside WSL); ``nvshmem`` puts
    with signals (:func:`~hydroforge.parallel.symmetric.nvshmem_problem`);
    ``nccl`` needs an NCCL process group.
    """

    ipc = ipc_problem(facts)
    nvshmem = nvshmem_problem(backend, facts)
    peer = ipc
    if peer is None and backend == "triton":
        # The triton kernel stores into NVSHMEM-allocated peers.
        peer = nvshmem
    return {
        "peer": peer,
        "nvshmem": nvshmem,
        "nccl": None if group_backend == "nccl" else "needs an NCCL process group",
    }


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class HaloRuntime:
    """The halo plan of one rank, its transport and recorded exchanges.

    Construction is collective over ``group``: every rank gathers the others'
    :class:`RankFacts` and takes the fastest transport they all support
    (:data:`TRANSPORTS`), or the one ``HYDROFORGE_HALO_TRANSPORT`` names.
    ``transport`` is the choice and ``skipped`` why faster ones were not.
    """

    def __init__(
        self, plan: HaloPlan, device: torch.device, backend: str, group: Any
    ) -> None:
        import torch.distributed as dist

        if device.type != "cuda" or backend not in ("cuda", "triton"):
            raise ValueError("halo exchange needs the cuda or triton backend on CUDA")
        self.plan = plan
        self.backend = backend
        self.signal_slot = plan.world * plan.max_chunks
        self._slots: dict[tuple[Any, ...], int] = {}
        requested = os.environ.get("HYDROFORGE_HALO_TRANSPORT", "auto")
        if requested not in ("auto", *TRANSPORTS):
            raise ValueError(
                f"HYDROFORGE_HALO_TRANSPORT must be auto or one of {TRANSPORTS}, "
                f"got {requested!r}"
            )
        self.facts = gather_rank_facts(group, device)
        problems = transport_problems(
            backend, self.facts, str(dist.get_backend(group)).lower()
        )
        self.skipped: dict[str, str] = {}
        self.transport: str | None = None
        for name in TRANSPORTS if requested == "auto" else (requested,):
            problem = problems[name] or self._open(name, device, group)
            if problem is None:
                self.transport = name
                break
            self.skipped[name] = problem
        if self.transport is None:
            reasons = "; ".join(f"{name}: {why}" for name, why in self.skipped.items())
            raise ValueError(f"no halo transport connects these ranks ({reasons})")

    def _open(self, name: str, device: torch.device, group: Any) -> str | None:
        """Allocate one transport's buffers; why it cannot connect, or ``None``."""

        plan = self.plan
        self.index = torch.tensor(plan.index or [0], dtype=torch.int32, device=device)
        if name == "nccl":
            mirror = _messages(plan, plan.mirror, device)
            accumulate = _messages(plan, plan.accumulate, device)
            size = max(1, plan.capacity)
            self.send = torch.zeros(size, dtype=torch.float64, device=device)
            self.recv = torch.zeros(size, dtype=torch.float64, device=device)
            self.messages = {"mirror": mirror, "accumulate": accumulate}
            self.registry = _nccl_exchange(mirror, accumulate, group)
            self.direct = False
            return None
        if getattr(self, "buffers", None) is None:
            self.buffers = PeerBuffers(
                group,
                device,
                {
                    "recv": (SLOTS * plan.capacity, torch.float64),
                    "signals": (SLOTS * self.signal_slot, torch.int64),
                },
                "CUDA" if self.backend == "cuda" else "NVSHMEM",
            )
        if name == "peer" and not self.buffers.mapped:
            return "the peers' memory could not be mapped"
        self.direct = name == "peer"
        self.registry = (
            halo_exchange_cuda if self.backend == "cuda" else halo_exchange_nvshmem
        )
        self.recv = self.buffers.tensors["recv"]
        self.signals = self.buffers.tensors["signals"]
        self.recv_peers = self.buffers.peers["recv"]
        self.signal_peers = self.buffers.peers["signals"]
        self.expected = torch.zeros_like(self.signals)
        # Only NVSHMEM puts read from a local staging buffer.
        self.stage = torch.zeros(
            1 if self.direct else SLOTS * plan.capacity,
            dtype=torch.float64,
            device=device,
        )
        return None

    def tensors(self) -> tuple[torch.Tensor, ...]:
        if self.transport == "nccl":
            messages = self.messages.values()
            return (
                self.index,
                self.send,
                self.recv,
                *(m.sources for m in messages),
                *(m.targets for m in messages),
            )
        return (
            self.index,
            self.recv,
            self.signals,
            self.expected,
            self.stage,
            self.recv_peers,
            self.signal_peers,
        )

    def exchange(
        self,
        fields: Sequence[torch.Tensor],
        mode: Literal["mirror", "accumulate"],
        clear: torch.Tensor | None = None,
    ) -> None:
        if not 1 <= len(fields) <= MAX_FIELDS:
            raise ValueError(f"halo_exchange_ moves 1 to {MAX_FIELDS} fields")
        if mode not in ("mirror", "accumulate"):
            raise ValueError("halo exchange mode must be 'mirror' or 'accumulate'")
        if clear is not None and mode != "accumulate":
            raise ValueError("clear applies to accumulate exchanges")
        plan = self.plan
        # Every call site owns one receive slot, numbered alike on every
        # rank: a slot is reused only by the same exchange of the next
        # sub-step.
        key = (mode, *(id(f) for f in fields))
        slot = self._slots.setdefault(key, len(self._slots))
        if slot >= SLOTS:
            raise ValueError(f"at most {SLOTS} distinct halo exchanges per model")
        d = plan.accumulate if mode == "accumulate" else plan.mirror
        if d.n_out + d.n_in == 0:
            return
        f = [*fields, *(fields[0],) * (MAX_FIELDS - len(fields))]
        if self.transport == "nccl":
            messages = self.messages[mode]
            self.registry(
                f0=f[0],
                f1=f[1],
                f2=f[2],
                clear=f[0] if clear is None else clear,
                send=self.send,
                recv=self.recv,
                sources=messages.sources,
                targets=messages.targets,
                num_entries=messages.sources.numel() + messages.targets.numel(),
                nf=len(fields),
                accumulate=mode == "accumulate",
                has_clear=clear is not None,
            )
            return
        self.registry(
            f0=f[0],
            f1=f[1],
            f2=f[2],
            clear=f[0] if clear is None else clear,
            index=self.index,
            stage=self.stage,
            recv=self.recv,
            signals=self.signals,
            expected=self.expected,
            recv_peers=self.recv_peers,
            signal_peers=self.signal_peers,
            out_peer=d.out_peer,
            out_start=d.out_start,
            out_dst=d.out_dst,
            out_idx=d.out_idx,
            in_peer=d.in_peer,
            in_start=d.in_start,
            in_off=d.in_off,
            in_idx=d.in_idx,
            slot_offset=slot * plan.capacity,
            signal_offset=slot * self.signal_slot,
            num_chunks=d.chunks,
            num_messages=d.n_out + d.n_in,
            n_out=d.n_out,
            my_pe=plan.rank,
            nf=len(fields),
            accumulate=mode == "accumulate",
            has_clear=clear is not None,
            direct=self.direct,
            max_chunks=plan.max_chunks,
            num_pes=plan.world,
        )


__all__ = [
    "TRANSPORTS",
    "HaloPlan",
    "HaloRuntime",
    "ghost_rows",
    "halo_exchange_cuda",
    "halo_exchange_nvshmem",
    "transport_problems",
]

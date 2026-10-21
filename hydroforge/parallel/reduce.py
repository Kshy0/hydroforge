# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Device all-reductions of a process group over the fastest transport.

``all_reduce_`` of model tensors on CUDA ranks runs as a registered kernel
rather than a process-group collective behind a managed-step handshake: as
a kernel it is recorded and CUDA-graph capturable, so a substep loop that
reduces (such as an adaptive sub-step count) stays captured. The transport
is chosen once per process group, like the halo's, from the facts every
rank gathers alike (:class:`~hydroforge.parallel.symmetric.RankFacts`):

``peer``
    each rank stores its values into slot ``rank`` of every rank's memory,
    signals them and reduces the slots in rank order (CUDA C++,
    ``all_reduce.cu``, on ``cuda``; Triton on ``triton``; one node outside
    WSL);
``nvshmem``
    the same protocol with NVSHMEM puts and signals (``triton``; across
    nodes with InfiniBand/RoCE on every node);
``nccl``
    an NCCL all-reduce of the tensor, any size and dtype.

``peer`` and ``nvshmem`` serve a minimum, maximum or sum of at most
:data:`CAPACITY` values; every rank reduces in rank order, so all hold the
same bits. A request no transport serves, a reduction to one destination,
temporaries and other devices keep the process-group path.
``HYDROFORGE_REDUCE_TRANSPORT`` names one transport (``auto``, ``peer``,
``nvshmem``, ``nccl``) or ``process_group`` to keep every collective on the
process-group path. Buffers peers write use the ``remote`` access mode,
which rollbacks never restore.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

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
    gather_rank_facts,
    ipc_problem,
    nvshmem_problem,
)

CAPACITY = 64  # values one device reduction carries
DTYPES = frozenset({torch.int32, torch.int64, torch.float32, torch.float64})
TRANSPORTS = ("peer", "nvshmem", "nccl")
"""All-reduce transports, fastest first."""

_OPS = {"min": 0, "max": 1, "sum": 2}

_SPEC = KernelSpec(
    name="peer_all_reduce",
    size=("numel",),
    parameters={
        "values": "read_write",
        # Peers write these; no rollback may restore them.
        "slots": "remote",
        "signals": "remote",
        "calls": "remote",
        "stage": "read_write",
        "slot_peers": "read",
        "signal_peers": "read",
        "numel": "index",
        "my_pe": "int32",
        "num_pes": "int32",
        "op": constant("int32"),
    },
)

_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# peer: CUDA C++ stores into mapped peers
# --------------------------------------------------------------------------- #
def _cuda_launch(
    values,
    slots,
    signals,
    calls,
    stage,
    slot_peers,
    signal_peers,
    numel,
    my_pe,
    num_pes,
    op,
    BLOCK_SIZE,
):
    del stage, BLOCK_SIZE
    kernel = f"peer_all_reduce<{ctype(values.dtype)}, {op}>"
    threads = min(-(-num_pes * numel // 32) * 32, 1024)
    arguments = (
        *(
            pointer(t)
            for t in (values, slots, signals, calls, slot_peers, signal_peers)
        ),
        *(int32(v) for v in (numel, my_pe, num_pes, CAPACITY)),
    )
    return (CudaLaunch(kernel, 1, threads, arguments),)


_CUDA_KERNEL = CudaKernel(
    CudaSource(path=_DIR / "all_reduce.cu", include_root=_DIR), launch=_cuda_launch
)


# --------------------------------------------------------------------------- #
# Triton: direct stores into mapped peers (peer) or NVSHMEM puts (nvshmem)
# --------------------------------------------------------------------------- #
@cache
def _triton_kernels():
    import triton
    import triton.language as tl
    from torch.distributed._symmetric_memory import _nvshmem_triton as nvshmem

    # nvshmem_common_transport.h / nvshmem_constants.h.
    signal_add = tl.constexpr(10)
    cmp_ge = tl.constexpr(5)

    @triton.jit
    def encode(mine, floating: tl.constexpr):
        # Values travel as the bits of a float64 or an int64, exactly.
        if floating:
            return mine.to(tl.float64).to(tl.int64, bitcast=True)
        return mine.to(tl.int64)

    @triton.jit
    def combine(
        values,
        slots,
        base,
        lanes,
        mask,
        world,
        op: tl.constexpr,
        floating: tl.constexpr,
        capacity: tl.constexpr,
    ):
        element = values.dtype.element_ty
        result = tl.zeros([capacity], dtype=element)
        for peer in range(world):
            # Written by a peer during this kernel: bypass caches.
            raw = tl.load(
                slots + (base + peer) * capacity + lanes,
                mask=mask,
                other=0,
                volatile=True,
            )
            if floating:
                value = raw.to(tl.float64, bitcast=True).to(element)
            else:
                value = raw.to(element)
            if peer == 0:
                result = value
            elif op == 2:
                result = result + value
            elif op == 1:
                result = tl.where(value > result, value, result)
            else:
                result = tl.where(value < result, value, result)
        tl.store(values + lanes, result, mask=mask)

    @triton.jit(do_not_specialize=["numel", "rank"])
    def peer_all_reduce_kernel(
        values,
        slots,
        signals,
        calls,
        slot_peers,
        signal_peers,
        numel,
        rank,
        world,
        op: tl.constexpr,
        floating: tl.constexpr,
        capacity: tl.constexpr,
    ):
        call = tl.load(calls)
        base = (call & 1) * world
        lanes = tl.arange(0, capacity)
        mask = lanes < numel
        bits = encode(tl.load(values + lanes, mask=mask, other=0), floating)
        for peer in range(world):
            slot = tl.load(slot_peers + peer).to(
                tl.pointer_type(tl.int64), bitcast=True
            )
            tl.store(slot + (base + rank) * capacity + lanes, bits, mask=mask)
        tl.debug_barrier()
        for peer in range(world):
            # One system-scope release orders the block's stores (cumulative
            # after the barrier) before the arrival.
            signal = tl.load(signal_peers + peer).to(
                tl.pointer_type(tl.int64), bitcast=True
            )
            tl.atomic_add(
                signal + lanes * 0,
                tl.where(lanes == 0, 1, 0).to(tl.int64),
                mask=lanes == 0,
                sem="release",
                scope="sys",
            )
        target = (call + 1) * world
        arrived = tl.atomic_add(signals, 0, sem="acquire", scope="sys")
        while arrived < target:
            arrived = tl.atomic_add(signals, 0, sem="acquire", scope="sys")
        combine(values, slots, base, lanes, mask, world, op, floating, capacity)
        tl.store(calls, call + 1)

    @nvshmem.requires_nvshmem
    @triton.jit(do_not_specialize=["numel", "rank"])
    def nvshmem_all_reduce_kernel(
        values,
        slots,
        signals,
        calls,
        stage,
        numel,
        rank,
        world,
        op: tl.constexpr,
        floating: tl.constexpr,
        capacity: tl.constexpr,
    ):
        call = tl.load(calls)
        base = (call & 1) * world
        lanes = tl.arange(0, capacity)
        mask = lanes < numel
        bits = encode(tl.load(values + lanes, mask=mask, other=0), floating)
        tl.store(stage + lanes, bits, mask=mask)
        tl.debug_barrier()
        for peer in range(world):
            nvshmem.putmem_signal_block(
                slots + (base + rank) * capacity,
                stage,
                numel.to(tl.int64) * 8,
                signals,
                1,
                signal_add,
                peer,
            )
        nvshmem.signal_wait_until(signals, cmp_ge, (call + 1) * world)
        combine(values, slots, base, lanes, mask, world, op, floating, capacity)
        tl.store(calls, call + 1)

    return peer_all_reduce_kernel, nvshmem_all_reduce_kernel


def _triton_prepare(direct: bool):
    def prepare(arguments, buffer_dtypes):
        del buffer_dtypes
        kernel = _triton_kernels()[0 if direct else 1]
        names = ("values", "slots", "signals", "calls")
        names += ("slot_peers", "signal_peers") if direct else ("stage",)
        values = {name: arguments[name] for name in names}
        values.update(
            numel=arguments["numel"],
            rank=arguments["my_pe"],
            world=arguments["num_pes"],
            op=arguments["op"],
            floating=arguments["values"].dtype.is_floating_point,
            capacity=CAPACITY,
        )
        # Compile (and initialize NVSHMEM state) before any capture.
        jit = kernel.jit_func if hasattr(kernel, "jit_func") else kernel
        extra = {"extern_libs": kernel.extern_libs} if not direct else {}
        jit.warmup(**values, grid=(1,), **extra)
        launcher = kernel[(1,)]

        def launch():
            launcher(**values)

        return launch

    return prepare


peer_all_reduce = BackendRegistry(
    _SPEC, {"cuda": _CUDA_KERNEL, "triton": TritonProgram(_triton_prepare(True))}
)
nvshmem_all_reduce = BackendRegistry(
    _SPEC, {"triton": TritonProgram(_triton_prepare(False))}
)


# --------------------------------------------------------------------------- #
# nccl: a recorded NCCL collective
# --------------------------------------------------------------------------- #
_NCCL_SPEC = KernelSpec(
    name="nccl_all_reduce",
    size=("numel",),
    parameters={
        "values": "read_write",
        "numel": "index",
        "op": constant("int32"),
    },
)


def _nccl_all_reduce(group: Any) -> BackendRegistry:
    """NCCL's all-reduce over ``group`` as a registered kernel (capturable)."""

    import torch.distributed as dist

    operations = (dist.ReduceOp.MIN, dist.ReduceOp.MAX, dist.ReduceOp.SUM)

    def nccl_all_reduce(values, numel, op):
        del numel
        dist.all_reduce(values, op=operations[op], group=group)

    kernel = TorchKernel(nccl_all_reduce, compile=False)
    return BackendRegistry(
        _NCCL_SPEC, {"cuda": kernel, "triton": kernel, "torch": kernel}
    )


# --------------------------------------------------------------------------- #
# Selection and runtime
# --------------------------------------------------------------------------- #
def transport_problems(
    backend: str, facts: tuple, group_backend: str
) -> dict[str, str | None]:
    """Why each transport cannot connect these ranks (``None``: it can)."""

    kernels = backend in ("cuda", "triton")
    return {
        "peer": (
            ipc_problem(facts)
            if kernels
            else "device reductions need the cuda or triton backend"
        ),
        "nvshmem": nvshmem_problem(backend, facts),
        "nccl": None if group_backend == "nccl" else "needs an NCCL process group",
    }


def _small(tensors: tuple[torch.Tensor, ...], reduction: str) -> bool:
    return reduction in _OPS and all(
        tensor.dtype in DTYPES and tensor.numel() <= CAPACITY for tensor in tensors
    )


class ReduceRuntime:
    """The all-reduce transports of one process group.

    Construction is collective over ``group``: every rank gathers the others'
    facts and keeps, in order, the transports they all support (or the one
    ``HYDROFORGE_REDUCE_TRANSPORT`` names). ``transports`` lists them,
    ``skipped`` why the others were not taken.
    """

    def __init__(self, group: Any, device: torch.device, backend: str) -> None:
        import torch.distributed as dist

        requested = os.environ.get("HYDROFORGE_REDUCE_TRANSPORT", "auto")
        if requested not in ("auto", "process_group", *TRANSPORTS):
            raise ValueError(
                "HYDROFORGE_REDUCE_TRANSPORT must be auto, process_group or one "
                f"of {TRANSPORTS}, got {requested!r}"
            )
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.skipped: dict[str, str] = {}
        self.transports: tuple[str, ...] = ()
        self.buffers: PeerBuffers | None = None
        if requested == "process_group":
            self.skipped = dict.fromkeys(TRANSPORTS, "HYDROFORGE_REDUCE_TRANSPORT")
            return
        if torch.version.hip is not None or device.type != "cuda":
            self.skipped = dict.fromkeys(TRANSPORTS, "device reductions need CUDA")
            return
        facts = gather_rank_facts(group, device)
        problems = transport_problems(
            backend, facts, str(dist.get_backend(group)).lower()
        )
        chosen = []
        for name in TRANSPORTS if requested == "auto" else (requested,):
            if name == "nvshmem" and "peer" in chosen:
                # Both serve the same requests; the faster one is open.
                self.skipped[name] = "peer serves its requests"
                continue
            problem = problems[name] or self._open(name, group, device)
            if problem is None:
                chosen.append(name)
            else:
                self.skipped[name] = problem
        self.transports = tuple(chosen)
        self.calls = torch.zeros(1, dtype=torch.int64, device=device)

    def _open(self, name: str, group: Any, device: torch.device) -> str | None:
        """Allocate one transport's buffers; why it cannot connect, or ``None``."""

        if name == "nccl":
            self.nccl = _nccl_all_reduce(group)
            return None
        import torch.distributed._symmetric_memory as symm_mem

        memory = "NVSHMEM" if name == "nvshmem" else symm_mem.get_backend(device)
        try:
            buffers = PeerBuffers(
                group,
                device,
                {
                    "slots": (2 * self.world * CAPACITY, torch.int64),
                    "signals": (1, torch.int64),
                    "stage": (CAPACITY, torch.int64),
                },
                memory or "CUDA",
            )
        except ValueError as error:
            return str(error)
        if name == "peer" and not buffers.mapped:
            return "the peers' memory could not be mapped"
        self.buffers = buffers
        return None

    def transport(
        self, tensors: tuple[torch.Tensor, ...], reduction: str
    ) -> str | None:
        """The transport serving a batch, alike on every rank, or ``None``."""

        small = _small(tensors, reduction)
        for name in self.transports:
            if name == "nccl" or small:
                return name
        return None

    def tensors(self) -> tuple[torch.Tensor, ...]:
        if self.buffers is None:
            return ()
        return (*self.buffers.all_tensors(), self.calls)

    def all_reduce(self, tensor: torch.Tensor, reduction: str, transport: str) -> None:
        """Reduce ``tensor`` in place on every rank (a registered kernel)."""

        if transport == "nccl":
            self.nccl(values=tensor, numel=tensor.numel(), op=_OPS[reduction])
            return
        buffers = self.buffers
        registry = peer_all_reduce if transport == "peer" else nvshmem_all_reduce
        registry(
            values=tensor,
            slots=buffers.tensors["slots"],
            signals=buffers.tensors["signals"],
            calls=self.calls,
            stage=buffers.tensors["stage"],
            slot_peers=buffers.peers["slots"],
            signal_peers=buffers.peers["signals"],
            numel=tensor.numel(),
            my_pe=self.rank,
            num_pes=self.world,
            op=_OPS[reduction],
        )


__all__ = [
    "CAPACITY",
    "TRANSPORTS",
    "ReduceRuntime",
    "nvshmem_all_reduce",
    "peer_all_reduce",
    "transport_problems",
]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Device buffers allocated alike on every rank of a process group.

GPU-initiated communication (halo exchange, device reductions) writes into
other ranks' buffers. :class:`PeerBuffers` allocates them through torch's
symmetric memory and records every peer's address when it is mapped into
this process (peer-to-peer over NVLink or PCIe), so kernels store straight
into it and release a signal word the receiver acquires.

The memory backend is process-wide and fixed by the first allocation:
``"CUDA"`` maps peers through CUDA IPC (ranks of one node, no NVSHMEM
bootstrap), ``"NVSHMEM"`` also reaches peers of other nodes through NVSHMEM
puts. Every rank agrees whether all peers are mapped; :func:`peers_share_node`
decides beforehand whether CUDA IPC can map them at all.
"""

from __future__ import annotations

import ctypes
import platform
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

Memory = Literal["CUDA", "NVSHMEM"]


@dataclass(frozen=True, slots=True)
class RankFacts:
    """What one rank's process can use for GPU-initiated communication.

    Gathered once and decided on alike by every rank, so a transport that
    would fail on some rank (or diverge between them) is ruled out before
    anything is allocated or initialized.
    """

    node: str
    wsl: bool
    driver: int
    capability: tuple[int, int]
    rdma: bool  # an RDMA device NVSHMEM's remote transports drive
    nvshmem: bool


def _driver_version() -> int:
    """The CUDA driver's version (``13040`` for 13.4), 0 when unknown."""

    try:
        version = ctypes.c_int()
        ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version))
        return int(version.value)
    except OSError:
        return 0


# RDMA drivers NVSHMEM's remote transports drive: Mellanox ConnectX (IBRC,
# IBGDA) and AWS EFA (libfabric). Omni-Path (hfi1) and others are not.
_NVSHMEM_RDMA = ("mlx5_core", "mlx4_core", "efa")


def _rdma() -> bool:
    infiniband = Path("/sys/class/infiniband")
    if not infiniband.is_dir():
        return False
    for device in infiniband.iterdir():
        driver = device / "device" / "driver"
        if driver.exists() and driver.resolve().name in _NVSHMEM_RDMA:
            return True
    return False


def _local_facts(device: torch.device) -> RankFacts:
    boot = Path("/proc/sys/kernel/random/boot_id")
    try:
        import torch.distributed._symmetric_memory as symm_mem

        nvshmem = bool(symm_mem.is_nvshmem_available())
    except Exception:  # noqa: BLE001 - absent or unusable NVSHMEM
        nvshmem = False
    return RankFacts(
        node=boot.read_text().strip() if boot.is_file() else socket.gethostname(),
        wsl="microsoft" in platform.release().lower(),
        driver=_driver_version(),
        capability=tuple(torch.cuda.get_device_capability(device)),
        rdma=_rdma(),
        nvshmem=nvshmem,
    )


def gather_rank_facts(group: Any, device: torch.device) -> tuple[RankFacts, ...]:
    """Every rank's :class:`RankFacts`, in rank order (collective)."""

    import torch.distributed as dist

    facts: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(facts, _local_facts(device), group=group)
    return tuple(facts)


def ipc_problem(facts: tuple[RankFacts, ...]) -> str | None:
    """Why CUDA IPC cannot map every rank's memory, or ``None``."""

    if len({fact.node for fact in facts}) > 1:
        return "the ranks span nodes"
    if any(fact.wsl for fact in facts):
        return "WSL has no CUDA IPC"
    if min(fact.capability for fact in facts) < (7, 0):
        return "peer signals need compute capability 7.0"
    return None


def nvshmem_problem(backend: str, facts: tuple[RankFacts, ...]) -> str | None:
    """Why NVSHMEM kernels cannot connect these ranks, or ``None``.

    They are Triton kernels; every rank needs NVSHMEM and one CUDA driver
    version (NVSHMEM's initialization takes driver-dependent steps that must
    match) and, across nodes, on every node an RDMA device its remote
    transports drive (Mellanox InfiniBand/RoCE, AWS EFA).
    """

    drivers = sorted({fact.driver for fact in facts})
    if backend != "triton":
        return "NVSHMEM kernels need the triton backend"
    if not all(fact.nvshmem for fact in facts):
        missing = [rank for rank, fact in enumerate(facts) if not fact.nvshmem]
        return f"NVSHMEM is unavailable on ranks {missing}"
    if len(drivers) > 1:
        return f"the ranks run different CUDA drivers {drivers}"
    single = len({fact.node for fact in facts}) == 1
    if not single and not all(fact.rdma for fact in facts):
        return (
            "NVSHMEM reaches other nodes over Mellanox InfiniBand/RoCE or EFA, "
            "and some node has none"
        )
    if single:
        # Within a node NVSHMEM maps peers like CUDA IPC.
        return ipc_problem(facts)
    return None


def peers_share_node(group: Any, device: torch.device | None = None) -> bool:
    """Whether every rank of ``group`` can map the others' memory over CUDA IPC.

    Collective over ``group``; decided before any symmetric allocation, since
    a rendezvous that cannot map a peer fails instead of reporting it.
    """

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return ipc_problem(gather_rank_facts(group, device)) is None


def symmetric_memory(memory: Memory, device: torch.device) -> Any:
    """Select the process-wide symmetric-memory backend, once."""

    import torch.distributed._symmetric_memory as symm_mem

    if symm_mem.get_backend(device) != memory:
        try:
            symm_mem.set_backend(memory)
        except Exception as error:
            raise ValueError(
                f"symmetric memory already uses {symm_mem.get_backend(device)!r} "
                f"in this process; {memory!r} cannot be selected"
            ) from error
    return symm_mem


class PeerBuffers:
    """Named symmetric buffers of one group and their peers' addresses.

    ``tensors[name]`` is this rank's buffer; ``peers[name]`` an int64 device
    tensor of every group rank's address of it (zeros unless ``mapped``).
    Buffers start zeroed, on every rank, before any rank returns.
    """

    def __init__(
        self,
        group: Any,
        device: torch.device,
        buffers: Mapping[str, tuple[int, torch.dtype]],
        memory: Memory,
    ) -> None:
        import torch.distributed as dist

        symm_mem = symmetric_memory(memory, device)
        if not symm_mem.is_symm_mem_enabled_for_group(group.group_name):
            symm_mem.enable_symm_mem_for_group(group.group_name)
        self.memory = memory
        self.world = dist.get_world_size(group)
        self.tensors: dict[str, torch.Tensor] = {}
        addresses: dict[str, list[int]] = {}
        for name, (numel, dtype) in buffers.items():
            tensor = symm_mem.empty(max(numel, 1), dtype=dtype, device=device)
            handle = symm_mem.rendezvous(tensor, group=group)
            self.tensors[name] = tensor
            addresses[name] = list(handle.buffer_ptrs)
        mapped = all(
            len(values) == self.world and all(values) for values in addresses.values()
        )
        agreed = torch.tensor([int(mapped)], dtype=torch.int32, device=device)
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
        self.mapped = bool(agreed.item())
        self.peers = {
            name: torch.tensor(
                values if self.mapped else [0] * self.world,
                dtype=torch.int64,
                device=device,
            )
            for name, values in addresses.items()
        }
        for tensor in self.tensors.values():
            tensor.zero_()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def all_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.tensors.values(), *self.peers.values())


__all__ = [
    "Memory",
    "PeerBuffers",
    "RankFacts",
    "gather_rank_facts",
    "ipc_problem",
    "nvshmem_problem",
    "peers_share_node",
    "symmetric_memory",
]

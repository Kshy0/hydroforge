# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""NCCL symmetric windows: the peers' memory NCCL maps for load and store.

From NCCL 2.29 the host API returns, for a window registered alike on every
rank, the address of each peer's copy that this process can load from and
store to: the *LSA* (load/store accessible) peers NCCL connects over NVLink,
PCIe peer-to-peer or a multi-node NVLink fabric. Kernels then exchange data
by plain stores, whatever made the peer reachable; a peer outside the LSA
team has no address and is reached otherwise.

The library is found, in order, at ``HYDROFORGE_NCCL`` (a ``libnccl.so.2``,
or a directory holding one, such as a private
``pip install --target DIR nvidia-nccl-cu12``), then as the NCCL torch
loaded. A private copy loads beside torch's under its own symbol scope; its
communicator is separate from the process group's. :func:`nccl_library`
reports why no usable library was found.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch

MINIMUM_VERSION = 22900
"""NCCL 2.29: host-side peer pointers and communicator properties."""

_WIN_COLL_SYMMETRIC = 0x01
_WIN_ALIGNMENT = 4096
_API_MAGIC = 0xCAFEBEEF
_GIN_TYPES = {0: "none", 2: "proxy", 3: "gdaki", 4: "gpi"}


def version_text(code: int) -> str:
    """``23007`` as ``2.30.7``."""

    return f"{code // 10000}.{code // 100 % 100}.{code % 100}"


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


class _Properties(ctypes.Structure):
    """``ncclCommProperties_t`` as of NCCL 2.32.3; earlier versions fill a
    prefix of it, and later ones are asked for this version's layout."""

    VERSION = 23203
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("rank", ctypes.c_int),
        ("nRanks", ctypes.c_int),
        ("cudaDev", ctypes.c_int),
        ("nvmlDev", ctypes.c_int),
        ("deviceApiSupport", ctypes.c_bool),
        ("multimemSupport", ctypes.c_bool),
        ("ginType", ctypes.c_int),
        ("nLsaTeams", ctypes.c_int),
        ("hostRmaSupport", ctypes.c_bool),
        ("railedGinType", ctypes.c_int),
        ("commHash", ctypes.c_uint64),
        ("ginMinStride", ctypes.c_int),
        ("ginConnectionType", ctypes.c_int),
        ("ginSupport", ctypes.c_bool * 64),
        ("devCommRuntimeVersionSize", ctypes.c_size_t),
        ("cftSupport", ctypes.c_bool),
        ("cftMulticastSupport", ctypes.c_bool),
        ("cftCountedSupport", ctypes.c_bool),
    ]


@dataclass(frozen=True)
class NcclLibrary:
    """One loaded ``libnccl`` with the window API."""

    path: str
    version: int
    handle: Any

    @property
    def version_text(self) -> str:
        return version_text(self.version)

    def check(self, result: int, what: str) -> None:
        if result != 0:
            message = self.handle.ncclGetErrorString(result)
            raise RuntimeError(
                f"NCCL {self.version_text} {what} failed: "
                f"{message.decode() if message else result}"
            )


def _candidates() -> list[Path]:
    found: list[Path] = []
    configured = os.environ.get("HYDROFORGE_NCCL", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            hits = sorted(path.rglob("libnccl.so*"))
            if not hits:
                raise ValueError(f"HYDROFORGE_NCCL={configured}: no libnccl.so in it")
            found.append(hits[0])
        else:
            found.append(path)
    # The NCCL torch loaded (``/proc/self/maps`` names its file).
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            name = line.rsplit(" ", 1)[-1]
            if "/libnccl.so" in name:
                found.append(Path(name))
                break
    return found


def _load(path: Path) -> NcclLibrary:
    loaded = {
        line.rsplit(" ", 1)[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libnccl.so" in line
    }
    mode = os.RTLD_LOCAL
    if str(path.resolve()) not in {str(Path(p).resolve()) for p in loaded}:
        # A private copy beside torch's NCCL binds its own symbols first.
        mode |= os.RTLD_DEEPBIND
    handle = ctypes.CDLL(str(path), mode=mode)
    handle.ncclGetErrorString.restype = ctypes.c_char_p
    version = ctypes.c_int()
    if handle.ncclGetVersion(ctypes.byref(version)) != 0:
        raise RuntimeError(f"{path}: ncclGetVersion failed")
    return NcclLibrary(str(path), int(version.value), handle)


@cache
def nccl_library() -> tuple[NcclLibrary | None, str | None]:
    """The first NCCL with windows and peer pointers, or why there is none."""

    problems = []
    for path in _candidates():
        try:
            library = _load(path)
        except OSError as error:
            problems.append(f"{path}: {error}")
            continue
        if library.version < MINIMUM_VERSION:
            problems.append(
                f"{path} is NCCL {library.version_text}; peer pointers need "
                f">= {version_text(MINIMUM_VERSION)}"
            )
            continue
        if not hasattr(library.handle, "ncclGetPeerDevicePointer"):
            problems.append(f"{path} lacks ncclGetPeerDevicePointer")
            continue
        return library, None
    if not problems:
        problems.append("no libnccl found (set HYDROFORGE_NCCL)")
    return None, "; ".join(problems)


class NcclComm:
    """A communicator of the library's own, over the ranks of ``group``, or
    over its ranks ``members`` (ascending, this rank among them).

    Collective over all of ``group`` either way: the first member makes the
    unique id, and every rank gathers the ids of every communicator being
    made, so ranks of ``group`` may make one communicator each over
    disjoint ``members`` (one per node) without creating process groups.
    """

    def __init__(
        self,
        library: NcclLibrary,
        group: Any,
        device: torch.device,
        members: list[int] | None = None,
    ) -> None:
        import torch.distributed as dist

        self.library = library
        self.device = device
        everyone = dist.get_world_size(group)
        members = list(range(everyone)) if members is None else list(members)
        self.rank = members.index(dist.get_rank(group))
        self.world = len(members)
        unique = _UniqueId()
        if self.rank == 0:
            library.check(
                library.handle.ncclGetUniqueId(ctypes.byref(unique)), "ncclGetUniqueId"
            )
        # All 128 bytes: the c_char array's value would stop at a zero byte.
        mine = ctypes.string_at(ctypes.byref(unique), 128) if self.rank == 0 else None
        if self.world == everyone:
            payload = [mine]
            dist.broadcast_object_list(
                payload, src=dist.get_global_rank(group, 0), group=group
            )
            mine = payload[0]
        else:
            ids: list[Any] = [None] * everyone
            dist.all_gather_object(ids, mine, group=group)
            mine = ids[members[0]]
        unique = _UniqueId.from_buffer_copy(mine)
        self.handle = ctypes.c_void_p()
        with torch.cuda.device(device):
            library.check(
                library.handle.ncclCommInitRank(
                    ctypes.byref(self.handle), self.world, unique, self.rank
                ),
                "ncclCommInitRank",
            )

    def properties(self) -> dict[str, Any]:
        """Device API support and the GIN type NCCL found for this rank."""

        version = min(self.library.version, _Properties.VERSION)
        props = _Properties(ctypes.sizeof(_Properties), _API_MAGIC, version)
        query = getattr(self.library.handle, "ncclCommQueryProperties", None)
        if query is None:
            return {}
        result = query(self.handle, ctypes.byref(props))
        if result != 0:
            return {"properties": f"ncclCommQueryProperties failed ({result})"}
        found = {
            "device_api": bool(props.deviceApiSupport),
            "multimem": bool(props.multimemSupport),
            "gin": _GIN_TYPES.get(int(props.ginType), str(props.ginType)),
        }
        if version >= 23203:
            found["gin_supported"] = [
                name
                for code, name in _GIN_TYPES.items()
                if code and props.ginSupport[code]
            ]
        return found

    def close(self) -> None:
        if self.handle:
            self.library.check(
                self.library.handle.ncclCommDestroy(self.handle), "ncclCommDestroy"
            )
            self.handle = ctypes.c_void_p()


class _DeviceArray:
    """A raw device allocation seen by torch through the array interface."""

    def __init__(self, address: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (address, False),
            "version": 3,
            "strides": None,
            "stream": None,
        }


class NcclWindow:
    """``nbytes`` of ``ncclMemAlloc`` memory registered as a symmetric window
    (collective), zeroed, with every rank's address of it.

    ``peers[r]`` is rank ``r``'s copy as this process loads and stores it, 0
    when NCCL cannot map it; ``tensor`` is the local copy as uint8.
    """

    def __init__(self, comm: NcclComm, nbytes: int) -> None:
        library = comm.library
        handle = library.handle
        self.comm = comm
        self.nbytes = -(-max(int(nbytes), 1) // _WIN_ALIGNMENT) * _WIN_ALIGNMENT
        self.address = ctypes.c_void_p()
        self.window = ctypes.c_void_p()
        with torch.cuda.device(comm.device):
            library.check(
                handle.ncclMemAlloc(
                    ctypes.byref(self.address), ctypes.c_size_t(self.nbytes)
                ),
                "ncclMemAlloc",
            )
            library.check(
                handle.ncclCommWindowRegister(
                    comm.handle,
                    self.address,
                    ctypes.c_size_t(self.nbytes),
                    ctypes.byref(self.window),
                    _WIN_COLL_SYMMETRIC,
                ),
                "ncclCommWindowRegister",
            )
        self.peers: list[int] = []
        if not self.window:
            # Without the device API (a communicator spanning nodes NCCL
            # cannot reach from the device) registration yields no window.
            library.check(handle.ncclMemFree(self.address), "ncclMemFree")
            self.address = ctypes.c_void_p()
            self.tensor = None
            return
        for peer in range(comm.world):
            pointer = ctypes.c_void_p()
            result = handle.ncclGetPeerDevicePointer(
                self.window, ctypes.c_size_t(0), peer, ctypes.byref(pointer)
            )
            if peer == comm.rank:
                library.check(result, "ncclGetPeerDevicePointer")
            # A peer outside the LSA team has no address: NCCL returns NULL,
            # or (2.32) an invalid-argument error.
            self.peers.append(int(pointer.value or 0) if result == 0 else 0)
        self.tensor = torch.as_tensor(
            _DeviceArray(int(self.address.value), self.nbytes), device=comm.device
        )
        self.tensor.zero_()

    @property
    def registered(self) -> bool:
        return bool(self.window)

    def close(self) -> None:
        if self.window:
            library = self.comm.library
            library.check(
                library.handle.ncclCommWindowDeregister(self.comm.handle, self.window),
                "ncclCommWindowDeregister",
            )
            library.check(library.handle.ncclMemFree(self.address), "ncclMemFree")
            self.window = ctypes.c_void_p()


class PeerWindow:
    """``nbytes`` every rank of ``group`` can store into from the ranks NCCL
    maps (collective).

    One communicator over the whole group first: its load/store team may
    span nodes (a multi-node NVLink fabric). When some rank gets no window
    (NCCL has no device API across these nodes), one communicator per node
    instead, so the ranks of a node still reach each other. ``peers[r]`` is
    group rank ``r``'s window as this process maps it, 0 when unreachable;
    ``scope`` is ``"group"`` or ``"node"``; ``properties`` what NCCL reported
    for the whole group, ``node_properties`` for this rank's node.

    ``group`` may be a subgroup of WORLD. The per-node communicators are
    made over ``group`` itself (:class:`NcclComm` ``members``), so no
    process group is created: ranks outside ``group`` take no part, and the
    default numbering of later process groups stays aligned on every rank.
    """

    def __init__(
        self,
        library: NcclLibrary,
        group: Any,
        device: torch.device,
        nbytes: int,
        nodes: list[Any],
    ) -> None:
        import torch.distributed as dist

        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        self._closers: list[Any] = []
        comm = NcclComm(library, group, device)
        self.properties = comm.properties()
        self.node_properties: dict[str, Any] = {}
        window = NcclWindow(comm, nbytes)
        everyone: list[Any] = [None] * world
        dist.all_gather_object(everyone, window.registered, group=group)
        if all(everyone):
            self.scope = "group"
            self._keep(window, comm)
            self.tensor, self.peers = window.tensor, window.peers
            return
        window.close()
        comm.close()
        self.scope = "node"
        self.peers = [0] * world
        self.tensor = None
        # One communicator per node, every rank of the group making its own.
        members = [r for r in range(world) if nodes[r] == nodes[rank]]
        local = NcclComm(library, group, device, members)
        self.node_properties = local.properties()
        window = NcclWindow(local, nbytes)
        self._keep(window, local)
        if window.registered:
            self.tensor = window.tensor
            for position, peer in enumerate(window.peers):
                self.peers[members[position]] = peer

    def _keep(self, window: NcclWindow, comm: NcclComm) -> None:
        self._closers = [window.close, comm.close, *self._closers]

    def close(self) -> None:
        for close in self._closers:
            close()
        self._closers = []


__all__ = [
    "MINIMUM_VERSION",
    "NcclComm",
    "NcclLibrary",
    "NcclWindow",
    "PeerWindow",
    "nccl_library",
    "version_text",
]

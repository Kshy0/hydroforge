# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Ghost rows of a decomposed axis and their exchange between ranks.

A model decomposes one axis over the spatial ranks in either of two ways,
and both give every rank read-only *ghost* rows of rows other ranks own:

- a **partition** of the partition-key coordinate whose groups split
  connected structure (a river basin cut between ranks): each rank holds
  ghost rows of the remote key rows its rows reference, after its owned
  rows (:func:`ghost_rows`; kind ``key``, the field's leading axis);
- a **slab** of a structured-grid axis (:class:`~hydroforge.contracts.slab.SlabAxis`):
  contiguous row blocks with ``halo`` ghost rows on each side
  (kinds ``cell`` and ``face`` for its staggered axis).

:meth:`AbstractModel.halo_exchange_` exchanges up to :data:`MAX_FIELDS`
fields of any of these kinds in one call: ``"mirror"`` gives ghost rows
their owners' values, ``"accumulate"`` adds what ghost rows hold to their
owners (zeroing the ghosts, and those of ``clear``). Each element travels in
its own bits, so integers and floats arrive exactly. It is a registered
kernel, recorded and CUDA-graph capturable.

Every pair of ranks exchanging rows takes its fastest link (``links``):

``peer``
    stores straight into the peer's memory (CUDA backend): the peers NCCL
    (>= 2.29) maps into a symmetric window — NVLink, PCIe peer-to-peer, a
    multi-node NVLink fabric — or, without such an NCCL, the ranks of one
    node over CUDA IPC (:mod:`hydroforge.parallel.nccl`);
``nccl``
    NCCL send/receive of the process group (other nodes, other backends);
``host``
    staged through the host over a Gloo group (CPU emulation); no CUDA
    graph captures it (``capturable``), so programs holding it run eagerly.

``HYDROFORGE_HALO_TRANSPORT`` (``auto``, ``peer``, ``nccl``, ``host``)
forces one link for every pair; ``HYDROFORGE_PEER_MEMORY`` (``auto``,
``nccl``, ``ipc``) the provider of peer memory.

Peers write into a ring of :data:`DEPTH` message slots per directed pair:
the sender stores message ``n`` into slot ``n % DEPTH`` only after the
receiver credited message ``n - DEPTH`` (finished with it), so no message
overwrites one still unread, however far a rank runs ahead (``halo.cu``).
A loop body whose own waits already order every reuse after the read
(the schedule proves it per pair) skips the credits.
Kernels publish into the rings and read from them
(:mod:`hydroforge.kernels.halo`), so a receiver may finish with a message
long after it arrived: when it counts and credits each one, and which
message every read wants, the recorded schedule decides
(:mod:`hydroforge.execution.halo_schedule`); each exchange then carries a
*site* argument, :data:`PLAIN_SITE` (receive the next message, count and
credit it) outside recordings.
A periodic slab over two ranks can hold two ghosts of one owned row on the
same peer, so the accumulate message from that peer lists the row twice:
its owner then adds the message in passes (the first entry of every row,
then the second), each pass touching a row once, so every contribution
lands, in the same order on every backend.
Counters live on the device and count against each rank's own, so any
number of identical executions on every rank (capture warm-ups included)
keep the protocol consistent; its buffers use the ``remote`` access mode,
which rollbacks never restore.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

MAX_FIELDS = 8
# Ring slots of a directed pair (a power of two): a sender may run this many
# messages ahead of the receiver's credits.
DEPTH = 4
CHUNK = 2048  # elements one block moves (four per thread)
KINDS = ("key", "cell", "face")
KEY, CELL, FACE = KINDS
LINKS = ("peer", "nccl", "host")
_LINK_CODE = {name: code for code, name in enumerate(LINKS)}
_REPEATS = 32  # halo.cu: an accumulate receive adds into some row twice
# Phases of one exchange launch (halo.cu): sends and receives, sends only
# (a message the schedule receives later), receives only (a receipt),
# neither (only the prologue's counts and credits); and around a loop whose
# body stores without credits (hydroforge.execution.halo_schedule), a wait
# for every credit of what was sent before it, and the prologue's counts
# then every credit set to what was counted.
PHASES = ("both", "send", "receive", "book", "drain", "flush")
_RECORD = 4 + 2 * len(KINDS)
_HEADER_ALIGN = 256
# halo.cu accumulates these element types.
_ADD_TYPES = {torch.float32: 1, torch.float64: 2, torch.int32: 3, torch.int64: 4}


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
class RowLayout:
    """One rank's stored rows of one kind: the global row each holds, and
    whether this rank owns it (else it is a ghost of its owner's)."""

    stored: np.ndarray
    owned: np.ndarray

    @property
    def extent(self) -> int:
        return int(self.stored.size)


def partition_layouts(
    rank_rows: Sequence[np.ndarray], ghosts: Sequence[np.ndarray]
) -> list[RowLayout]:
    """Owned key rows, then ghost rows, of every rank."""

    return [
        RowLayout(
            np.concatenate((rows, extra)).astype(np.int64),
            np.arange(rows.size + extra.size) < rows.size,
        )
        for rows, extra in zip(rank_rows, ghosts, strict=True)
    ]


class HaloPlan:
    """Which rows every pair of ranks exchanges, from all ranks' layouts.

    ``pairs[k][(q, p)]`` holds, for kind ``k``, the rows of ``p``'s ghosts
    that ``q`` owns: ``(q-local owned rows, p-local ghost rows)`` in ``p``'s
    stored order. A mirror sends the first to ``p``'s second; an accumulate
    the reverse. ``neighbours[r]`` are the ranks ``r`` exchanges with.
    """

    def __init__(self, rank: int, layouts: Mapping[str, Sequence[RowLayout]]) -> None:
        unknown = set(layouts).difference(KINDS)
        if unknown:
            raise ValueError(f"unknown halo kinds {sorted(unknown)}")
        self.rank = rank
        self.world = len(next(iter(layouts.values())))
        self.layouts = dict(layouts)
        self.pairs: dict[str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
        world = self.world
        self.counts = np.zeros((len(KINDS), world, world), np.int64)
        for kind, ranks in layouts.items():
            k = KINDS.index(kind)
            size = 1 + max((int(r.stored.max()) for r in ranks if r.extent), default=0)
            owner = np.full(size, -1, np.int64)
            local = np.zeros(size, np.int64)
            for r, layout in enumerate(ranks):
                rows = np.flatnonzero(layout.owned)
                owner[layout.stored[rows]] = r
                local[layout.stored[rows]] = rows
            pairs = {}
            for p, layout in enumerate(ranks):
                ghosts = np.flatnonzero(~layout.owned)
                globals_ = layout.stored[ghosts]
                owners = owner[globals_]
                if (owners < 0).any():
                    missing = globals_[owners < 0][:5].tolist()
                    raise ValueError(f"{kind} ghost rows {missing} have no owner")
                for q in np.unique(owners):
                    if q == p:
                        raise ValueError(
                            f"rank {p} holds a {kind} ghost of its own row"
                        )
                    pick = owners == q
                    pairs[(int(q), p)] = (local[globals_[pick]], ghosts[pick])
                    self.counts[k, q, p] = int(pick.sum())
            self.pairs[kind] = pairs
        linked = self.counts.sum(0)
        linked = (linked + linked.T) > 0
        self.neighbours = tuple(
            tuple(np.flatnonzero(linked[r]).tolist()) for r in range(world)
        )

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(kind for kind in KINDS if kind in self.layouts)

    def extent(self, kind: str) -> int:
        return self.layouts[kind][self.rank].extent

    def owned(self, kind: str) -> int:
        return int(self.layouts[kind][self.rank].owned.sum())

    @property
    def ghosts(self) -> int:
        return sum(self.extent(k) - self.owned(k) for k in self.kinds)

    @property
    def ghosts_total(self) -> int:
        return int(self.counts.sum())

    def rows(
        self, direction: Literal["send", "receive"], mode: str, peer: int
    ) -> dict[str, np.ndarray]:
        """This rank's local rows of the message to (send) or from
        (receive) ``peer``, by kind, in the order both ends agree on."""

        me = self.rank
        mirror = mode == "mirror"
        out = {}
        for kind, pairs in self.pairs.items():
            if (direction == "send") == mirror:
                entry = pairs.get((me, peer))  # my owned rows peer ghosts
                rows = None if entry is None else entry[0]
            else:
                entry = pairs.get((peer, me))  # my ghosts peer owns
                rows = None if entry is None else entry[1]
            if rows is not None and rows.size:
                out[kind] = rows
        return out

    def ring_slot(self, sender: int, receiver: int) -> int:
        """The sender's slot in the receiver's ring."""

        return self.neighbours[receiver].index(sender)

    def pair_rows(self, a: int, b: int) -> np.ndarray:
        """Rows of every kind a message between ``a`` and ``b`` may carry."""

        return np.maximum(self.counts[:, a, b], self.counts[:, b, a])


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #
@dataclass
class _PeerMemory:
    """A window every rank allocated alike, and its peers' addresses."""

    provider: str
    tensor: torch.Tensor
    addresses: list[int]
    closers: list[Any] = field(default_factory=list)

    def close(self) -> None:
        for close in self.closers:
            close()
        self.closers = []


def _open_peer_memory(
    nbytes: int, group: Any, device: torch.device, details: dict[str, Any]
) -> tuple[_PeerMemory | None, str | None]:
    """Peer-mapped memory (collective), or why there is none."""

    import torch.distributed as dist

    from hydroforge.parallel.nccl import PeerWindow, nccl_library
    from hydroforge.parallel.symmetric import (
        PeerBuffers,
        gather_rank_facts,
        ipc_problem,
    )

    requested = os.environ.get("HYDROFORGE_PEER_MEMORY", "auto")
    if requested not in ("auto", "nccl", "ipc"):
        raise ValueError(
            f"HYDROFORGE_PEER_MEMORY must be auto, nccl or ipc, got {requested!r}"
        )
    library, problem = nccl_library()
    states: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(
        states, None if library is None else library.version, group=group
    )
    problems = {}
    if requested in ("auto", "nccl"):
        if all(state is not None for state in states):
            from hydroforge.parallel.topology import node_label

            nodes: list[Any] = [None] * dist.get_world_size(group)
            dist.all_gather_object(nodes, node_label(), group=group)
            window = PeerWindow(library, group, device, nbytes, nodes)
            details["nccl"] = library.version_text
            details["window_scope"] = window.scope
            details.update(window.properties)
            if window.node_properties:
                details["node"] = window.node_properties
            if window.tensor is not None:
                window.tensor.zero_()
            # Every window is zeroed before any peer stores into it.
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
            tensor = window.tensor
            if tensor is None:
                # A rank alone on its node keeps a window of its own.
                tensor = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            return _PeerMemory("nccl", tensor, window.peers, [window.close]), None
        missing = [r for r, state in enumerate(states) if state is None]
        problems["nccl"] = (
            problem if library is None else f"ranks {missing} have no NCCL >= 2.29"
        )
    if requested in ("auto", "ipc"):
        ipc = ipc_problem(gather_rank_facts(group, device))
        if ipc is None:
            buffers = PeerBuffers(
                group, device, {"window": (nbytes, torch.uint8)}, "CUDA"
            )
            if buffers.mapped:
                addresses = buffers.peers["window"].tolist()
                return _PeerMemory("ipc", buffers.tensors["window"], addresses), None
            ipc = "the peers' memory could not be mapped"
        problems["ipc"] = ipc
    return None, "; ".join(f"{k}: {v}" for k, v in problems.items())


# --------------------------------------------------------------------------- #
# Calls
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldLayout:
    """Where an exchanged field's decomposed axis is: its kind and its axis
    counted from the last dimension."""

    kind: str
    axis: int


def pack_layouts(layouts: Sequence[FieldLayout]) -> tuple[int, int]:
    """Each field's (kind, axis) in two int32 compile-time constants."""

    words = [0, 0]
    for index, layout in enumerate(layouts):
        if not 0 <= layout.axis < 16:
            raise ValueError("halo fields have at most sixteen dimensions")
        code = (layout.axis << 2) | KINDS.index(layout.kind)
        words[index // 4] |= code << (8 * (index % 4))
    return words[0], words[1]


def unpack_layouts(low: int, high: int, count: int) -> list[FieldLayout]:
    words = (low, high)
    result = []
    for index in range(count):
        code = (words[index // 4] >> (8 * (index % 4))) & 0xFF
        result.append(FieldLayout(KINDS[code & 3], code >> 2))
    return result


@dataclass(frozen=True)
class _View:
    """A field as (outer, extent, inner) around its decomposed axis."""

    tensor: torch.Tensor
    kind: str
    outer: int
    extent: int
    inner: int

    @classmethod
    def of(cls, tensor: torch.Tensor, layout: FieldLayout) -> _View:
        position = tensor.dim() - 1 - layout.axis
        shape = tuple(tensor.shape)
        return cls(
            tensor,
            layout.kind,
            int(np.prod(shape[:position], dtype=np.int64)),
            shape[position],
            int(np.prod(shape[position + 1 :], dtype=np.int64)),
        )

    @property
    def size(self) -> int:
        return self.tensor.element_size()

    def elements(self, rows: int) -> int:
        return rows * self.outer * self.inner


def _aligned(nbytes: int) -> int:
    return -(-nbytes // 16) * 16


@dataclass(frozen=True)
class _Message:
    """One message of this rank: its peer, link and rows by kind."""

    peer: int
    link: str
    rows: dict[str, np.ndarray]
    ring: int  # the sender's slot in the receiver's ring
    stage: int  # its staging slot (non-peer links)
    passes: int = 1  # accumulate receives: entries of its most listed row

    def count(self, kind: str) -> int:
        rows = self.rows.get(kind)
        return 0 if rows is None else int(rows.size)

    def nbytes(self, views: Sequence[_View]) -> int:
        return sum(_aligned(v.elements(self.count(v.kind)) * v.size) for v in views)


def occurrences(rows: np.ndarray) -> np.ndarray:
    """How many earlier entries of ``rows`` hold each entry's row: the pass
    in which an accumulate adds it."""

    order = np.argsort(rows, kind="stable")
    ranked = rows[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    run = np.arange(rows.size) - np.repeat(starts, np.diff(np.r_[starts, rows.size]))
    result = np.empty(rows.size, np.int64)
    result[order] = run
    return result


def message_capacity(plan: HaloPlan, row_bytes: Mapping[str, int]) -> int:
    """Bytes of one message slot: the largest message of any pair."""

    largest = 1
    for a in range(plan.world):
        for b in plan.neighbours[a]:
            rows = plan.pair_rows(a, b)
            for kind in plan.kinds:
                largest = max(
                    largest, int(rows[KINDS.index(kind)]) * row_bytes[kind] + 16
                )
    return -(-MAX_FIELDS * largest // _HEADER_ALIGN) * _HEADER_ALIGN


def window_layout(plan: HaloPlan, capacity: int) -> tuple[int, int]:
    """Byte offset of the ring in a window and the window's size: arrivals
    and credits of every rank, then :data:`DEPTH` slots per neighbour."""

    ring = -(-16 * plan.world // _HEADER_ALIGN) * _HEADER_ALIGN
    slots = max((len(n) for n in plan.neighbours), default=0)
    return ring, ring + slots * DEPTH * capacity


def site_words(
    world: int,
    phase: str = "both",
    written: int = (1 << MAX_FIELDS) - 1,
    skip: Mapping[int, int] | None = None,
    settle: Mapping[int, int] | None = None,
    post: Mapping[int, int] | None = None,
    prologue: Sequence[tuple[int, int, int]] = (),
    *,
    checks: bool = True,
) -> tuple[int, ...]:
    """The words of an exchange's site argument: its phase (``PHASES``),
    the mask of fields its receives write, whether its sends wait for slot
    credits (``checks``: not in a loop body proven free of them), then its
    site table (halo.cu): per peer the message's place past ``received``
    (``skip``), the messages its receive counts (``settle``) and credits
    (``post``), then the prologue's (peer, credits, counted) entries."""

    table = []
    for values in (skip, settle, post):
        table += [int((values or {}).get(p, 0)) for p in range(world)]
    table.append(len(prologue))
    for entry in prologue:
        table.extend(int(value) for value in entry)
    return (PHASES.index(phase), int(written), int(checks), *table)


def plain_site(world: int) -> tuple[int, ...]:
    """The site words of an exchange outside recordings: every field
    written, the next message received, counted and credited."""

    every = dict.fromkeys(range(world), 1)
    return site_words(world, settle=every, post=every)


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class HaloRuntime:
    """The halo of one rank: its plan, each pair's link and the exchange.

    Construction is collective over ``group``. ``row_bytes[kind]`` bounds the
    bytes of one row of any field of that kind (every element of the field
    at one index of its decomposed axis); it sizes the message slots.
    """

    def __init__(
        self,
        plan: HaloPlan,
        device: torch.device,
        backend: str,
        group: Any,
        row_bytes: Mapping[str, int],
    ) -> None:
        import torch.distributed as dist

        from hydroforge.parallel.topology import node_label

        self.plan = plan
        self.device = device
        self.backend = backend
        self.group = group
        self.details: dict[str, Any] = {}
        self.skipped: dict[str, str] = {}
        # Fields recorded mirrors asked for, and those pruned as unchanged;
        # recorded publications, ghost reads from rings and ring receipts
        # (hydroforge.execution.halo_schedule).
        self.mirrors_requested = 0
        self.mirrors_pruned = 0
        self.publications = 0
        self.fused_reads = 0
        self.receipts = 0
        self.credit_free = 0  # loop bodies proven to need no slot credits
        rank, world = plan.rank, plan.world
        if dist.get_world_size(group) != world or dist.get_rank(group) != rank:
            raise ValueError(
                f"the halo plan is rank {rank} of {world}, but the process group "
                f"is rank {dist.get_rank(group)} of {dist.get_world_size(group)}"
            )
        # The node every rank runs on, whatever placement assumed.
        labels: list[Any] = [None] * world
        dist.all_gather_object(labels, node_label(), group=group)
        self.nodes = tuple(labels)
        bounds: list[Any] = [None] * world
        dist.all_gather_object(bounds, dict(row_bytes), group=group)
        self.row_bytes = {
            kind: max(int(b.get(kind, 0)) for b in bounds) for kind in plan.kinds
        }
        self.capacity = message_capacity(plan, self.row_bytes)
        group_backend = str(dist.get_backend(group)).lower()
        requested = os.environ.get("HYDROFORGE_HALO_TRANSPORT", "auto")
        if requested not in ("auto", *LINKS):
            raise ValueError(
                f"HYDROFORGE_HALO_TRANSPORT must be auto or one of {LINKS}, "
                f"got {requested!r}"
            )
        fallback = "nccl" if group_backend == "nccl" else "host"
        if requested in ("nccl", "host") and requested != fallback:
            raise ValueError(
                f"the {requested} halo transport needs a "
                f"{'NCCL' if requested == 'nccl' else 'Gloo'} process group"
            )
        self.ring, self.window_bytes = window_layout(plan, self.capacity)
        self.memory: _PeerMemory | None = None
        reach = [False] * world
        capable = requested in ("auto", "peer")
        if capable and (backend != "cuda" or device.type != "cuda"):
            self.skipped["peer"] = "peer stores need the cuda backend on CUDA devices"
            capable = False
        # Every rank takes part in allocating peer memory, or none does.
        flags: list[Any] = [None] * world
        dist.all_gather_object(flags, capable, group=group)
        if all(flags) and any(plan.neighbours):
            self.memory, problem = _open_peer_memory(
                self.window_bytes, group, device, self.details
            )
            if self.memory is None:
                self.skipped["peer"] = problem
            else:
                self.details["memory"] = self.memory.provider
                reach = [address != 0 for address in self.memory.addresses]
        elif capable:
            self.skipped["peer"] = "some rank cannot store into peers"
        matrix: list[Any] = [None] * world
        dist.all_gather_object(matrix, reach, group=group)
        self.links: dict[int, str] = {}
        for peer in plan.neighbours[rank]:
            direct = matrix[rank][peer] and matrix[peer][rank]
            if requested == "peer" and not direct:
                raise ValueError(
                    f"HYDROFORGE_HALO_TRANSPORT=peer, but rank {peer} is not "
                    f"load/store reachable from rank {rank} "
                    f"({self.skipped.get('peer', 'outside the LSA team')})"
                )
            self.links[peer] = (
                "peer" if direct and requested in ("auto", "peer") else fallback
            )
        everywhere: list[Any] = [None] * world
        dist.all_gather_object(
            everywhere, sorted(set(self.links.values())), group=group
        )
        self.communicates = any(
            link != "peer" for links in everywhere for link in links
        )
        # Host staging (``.cpu()``, Gloo requests) runs on the host at every
        # exchange: no CUDA graph can replay it.
        self.capturable = not any("host" in links for links in everywhere)
        self._build_tables()
        # The registered exchange kernel (hydroforge.kernels.halo.attach),
        # attached by the layer that owns kernels.
        self.registry: Any = None

    # ------------------------------------------------------------------ #
    def _build_tables(self) -> None:
        plan, device = self.plan, self.device
        rank = plan.rank
        rows: list[np.ndarray] = []
        passes: list[np.ndarray] = []
        cursor = 0
        records: list[list[int]] = []
        self.messages: dict[tuple[str, str], tuple[int, list[_Message]]] = {}
        stages = {"send": 0, "receive": 0}
        for mode in ("mirror", "accumulate"):
            for direction in ("send", "receive"):
                first = len(records)
                found = []
                stage = 0
                for peer in plan.neighbours[rank]:
                    kinds = plan.rows(direction, mode, peer)
                    if not kinds:
                        continue
                    link = self.links[peer]
                    ring = (
                        plan.ring_slot(rank, peer)
                        if direction == "send"
                        else plan.ring_slot(peer, rank)
                    )
                    # Only an owner's rows of one accumulate message repeat.
                    adds = mode == "accumulate" and direction == "receive"
                    entries = {
                        kind: occurrences(values) if adds else np.zeros_like(values)
                        for kind, values in kinds.items()
                    }
                    count = 1 + max(int(e.max(initial=0)) for e in entries.values())
                    message = _Message(
                        peer, link, kinds, ring, -1 if link == "peer" else stage, count
                    )
                    if link != "peer":
                        stage += 1
                    code = _LINK_CODE[link] | (_REPEATS if count > 1 else 0)
                    record = [peer, code, ring, message.stage]
                    offsets, counts = [], []
                    for kind in KINDS:
                        values = kinds.get(kind, np.zeros(0, np.int64))
                        offsets.append(cursor)
                        counts.append(int(values.size))
                        rows.append(values.astype(np.int64))
                        passes.append(entries.get(kind, np.zeros(0, np.int64)))
                        cursor += int(values.size)
                    records.append(record + offsets + counts)
                    found.append(message)
                stages[direction] = max(stages[direction], stage)
                self.messages[(mode, direction)] = (first, found)
        table = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        if table.size and table.max() >= 2**31:
            raise ValueError("halo rows exceed int32")
        # Every entry's row, then the pass that adds it (halo.cu).
        self.entries = max(1, table.size)
        self.rows = torch.tensor(
            np.concatenate((table, np.concatenate(passes)))
            if table.size
            else np.zeros(2, np.int64),
            dtype=torch.int32,
            device=device,
        )
        self.records = torch.tensor(
            np.array(records, np.int32).reshape(-1)
            if records
            else np.zeros(1, np.int32),
            dtype=torch.int32,
            device=device,
        )
        # Index tensors of the torch path, by (mode, direction, peer, kind).
        self.index = {
            (mode, direction, m.peer, kind): torch.tensor(
                r, dtype=torch.int64, device=device
            )
            for (mode, direction), (_first, found) in self.messages.items()
            for m in found
            for kind, r in m.rows.items()
        }
        # The torch path's passes over repeated rows: (rows, entries) of each.
        self.repeats = {}
        for m in self.messages[("accumulate", "receive")][1]:
            for kind, r in m.rows.items():
                entry = occurrences(r)
                if entry.max(initial=0) == 0:
                    continue
                self.repeats[("accumulate", "receive", m.peer, kind)] = [
                    tuple(
                        torch.tensor(v, dtype=torch.int64, device=device)
                        for v in (r[entry == p], np.flatnonzero(entry == p))
                    )
                    for p in range(int(entry.max()) + 1)
                ]
        world = plan.world
        # sent, received, sends done and receives done of every peer.
        self.state = torch.zeros(4 * world, dtype=torch.int64, device=device)
        # Site tables by content (site_table), kept so recorded calls keep
        # their addresses.
        self.sites: dict[tuple[int, ...], torch.Tensor] = {}
        # The site argument of exchanges outside recordings (a HaloArgument
        # of plain_site's words, from hydroforge.kernels.halo.attach).
        self.plain_site: Any = None
        self.send_stage = torch.zeros(
            max(1, stages["send"] * self.capacity), dtype=torch.uint8, device=device
        )
        self.recv_stage = torch.zeros(
            max(1, stages["receive"] * self.capacity), dtype=torch.uint8, device=device
        )
        if self.memory is not None:
            self.window = self.memory.tensor
            self.peers = torch.tensor(
                self.memory.addresses, dtype=torch.int64, device=device
            )
        else:
            self.window = torch.zeros(1, dtype=torch.uint8, device=device)
            self.peers = torch.zeros(world, dtype=torch.int64, device=device)

    def host_stages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The host memory Gloo sends from and receives into: the staging
        slots on a CPU device, else pinned mirrors made on first use."""

        if self.send_stage.device.type == "cpu":
            return self.send_stage, self.recv_stage
        mirrors = getattr(self, "_host_mirrors", None)
        if mirrors is None:
            pin = torch.cuda.is_available()
            mirrors = self._host_mirrors = tuple(
                torch.empty(stage.numel(), dtype=torch.uint8, pin_memory=pin)
                for stage in (self.send_stage, self.recv_stage)
            )
        return mirrors

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.window,
            self.peers,
            self.state,
            self.send_stage,
            self.recv_stage,
            self.rows,
            self.records,
            *self.index.values(),
            *(t for passes in self.repeats.values() for pair in passes for t in pair),
            # Site tables of exchanges and per-call access tables (kernels.halo).
            *self.sites.values(),
            *(
                tensor
                for argument in getattr(self, "access_arguments", {}).values()
                for tensor in argument.keep
                if isinstance(tensor, torch.Tensor)
            ),
        )

    def summary(self) -> dict[str, Any]:
        """Links, providers and message counts, for reports."""

        plan = self.plan
        sends = {
            str(m.peer): {k: int(v.size) for k, v in m.rows.items()}
            for m in self.messages[("mirror", "send")][1]
        }
        receives = {
            str(m.peer): {k: int(v.size) for k, v in m.rows.items()}
            for m in self.messages[("mirror", "receive")][1]
        }
        # Ghost rows whose owner runs on another node.
        nodes = np.asarray(self.nodes)
        apart = nodes[:, None] != nodes[None, :]
        across = int(plan.counts[:, apart].sum())
        return {
            "kinds": list(plan.kinds),
            "ghosts": plan.ghosts,
            "ghosts_total": plan.ghosts_total,
            "ghosts_across_nodes": across,
            "links": {str(p): link for p, link in self.links.items()},
            "skipped": dict(self.skipped),
            **self.details,
            "message_bytes": self.capacity,
            "mirrors_requested": self.mirrors_requested,
            "mirrors_pruned": self.mirrors_pruned,
            "publications": self.publications,
            "fused_reads": self.fused_reads,
            "receipts": self.receipts,
            "credit_free": self.credit_free,
            "sends": sends,
            "receives": receives,
        }

    def close(self) -> None:
        if self.memory is not None:
            self.memory.close()
            self.memory = None

    # ------------------------------------------------------------------ #
    def site_table(self, words: Sequence[int]) -> torch.Tensor:
        """The device copy of a site table (halo.cu), one per content."""

        key = tuple(int(w) for w in words)
        table = self.sites.get(key)
        if table is None:
            from torch.utils._python_dispatch import _disable_current_modes

            with _disable_current_modes():
                table = self.sites[key] = torch.tensor(
                    key, dtype=torch.int64, device=self.device
                )
        return table

    def exchange(
        self,
        fields: Sequence[torch.Tensor],
        layouts: Sequence[FieldLayout],
        mode: Literal["mirror", "accumulate"],
        clear: torch.Tensor | None = None,
        clear_layout: FieldLayout | None = None,
        *,
        site: Any = None,
    ) -> None:
        """Exchange ``fields``. ``site`` (a ``HaloArgument`` whose words
        :func:`site_words` lays out) comes from the recorded schedule; the
        default receives, counts and credits the next message."""
        arguments = self.exchange_arguments(
            fields, layouts, mode, clear, clear_layout, site=site
        )
        if arguments is not None:
            self.registry(**arguments)

    def exchange_arguments(
        self,
        fields: Sequence[torch.Tensor],
        layouts: Sequence[FieldLayout],
        mode: Literal["mirror", "accumulate"],
        clear: torch.Tensor | None = None,
        clear_layout: FieldLayout | None = None,
        *,
        site: Any = None,
    ) -> dict[str, Any] | None:
        """The exchange kernel's arguments for :meth:`exchange`, validated
        (``None``: this rank exchanges nothing in ``mode``). An accumulate
        of no fields only zeroes the ghost rows of ``clear``."""
        if not 1 <= len(fields) <= MAX_FIELDS and not (
            not fields and clear is not None and mode == "accumulate"
        ):
            raise ValueError(f"halo_exchange_ moves 1 to {MAX_FIELDS} fields")
        if mode not in ("mirror", "accumulate"):
            raise ValueError("halo exchange mode must be 'mirror' or 'accumulate'")
        if clear is not None and mode != "accumulate":
            raise ValueError("clear applies to accumulate exchanges")
        for tensor, layout in zip(fields, layouts, strict=True):
            self._check(tensor, layout, mode)
        if clear is not None:
            self._check(clear, clear_layout, mode, clearing=True)
        if not (
            self.messages[(mode, "send")][1] or self.messages[(mode, "receive")][1]
        ):
            return None
        self._check_capacity(fields, layouts, mode)
        if self.registry is None:
            raise RuntimeError(
                "this halo has no exchange kernel; attach one with "
                "hydroforge.kernels.halo.attach(runtime)"
            )
        low, high = pack_layouts(layouts)
        filler = fields[0] if fields else clear
        padded = [*fields, *(filler,) * (MAX_FIELDS - len(fields))]
        return dict(
            **{f"f{i}": padded[i] for i in range(MAX_FIELDS)},
            clear=filler if clear is None else clear,
            window=self.window,
            peers=self.peers,
            state=self.state,
            send_stage=self.send_stage,
            recv_stage=self.recv_stage,
            rows=self.rows,
            records=self.records,
            num_entries=int(self.rows.numel()),
            nf=len(fields),
            accumulate=mode == "accumulate",
            has_clear=clear is not None,
            layout_low=low,
            layout_high=high,
            clear_layout=0 if clear_layout is None else pack_layouts([clear_layout])[0],
            site=self.plain_site if site is None else site,
        )

    def _check_capacity(
        self,
        fields: Sequence[torch.Tensor],
        layouts: Sequence[FieldLayout],
        mode: str,
    ) -> None:
        """Reject an exchange whose message outgrows its slot: kernels store
        messages by raw address, so a larger one would overwrite the next
        slot or run past the allocation."""

        views = [_View.of(t, layout) for t, layout in zip(fields, layouts)]
        for direction in ("send", "receive"):
            for message in self.messages[(mode, direction)][1]:
                nbytes = message.nbytes(views)
                if nbytes <= self.capacity:
                    continue
                widths = {
                    f"{v.kind} field {i}": v.outer * v.inner * v.size
                    for i, v in enumerate(views)
                }
                towards = "to" if direction == "send" else "from"
                raise ValueError(
                    f"halo message {towards} rank "
                    f"{message.peer} holds {nbytes} bytes, more than the "
                    f"{self.capacity}-byte slot sized at startup for rows of at "
                    f"most {self.row_bytes} bytes (these fields' rows: {widths} "
                    "bytes); exchange fewer or narrower fields per call"
                )

    def _check(
        self,
        tensor: torch.Tensor,
        layout: FieldLayout,
        mode: str,
        clearing: bool = False,
    ) -> None:
        if layout.kind not in self.plan.kinds:
            raise ValueError(f"the halo has no {layout.kind} rows")
        view = _View.of(tensor, layout)
        if view.extent != self.plan.extent(layout.kind):
            raise ValueError(
                f"a {layout.kind} field holds {view.extent} rows along its "
                f"decomposed axis; this rank stores {self.plan.extent(layout.kind)}"
            )
        if mode == "accumulate" and not clearing and tensor.dtype not in _ADD_TYPES:
            raise ValueError(
                f"accumulate exchanges add float32, float64, int32 or int64 "
                f"fields, not {tensor.dtype}"
            )


__all__ = [
    "CHUNK",
    "DEPTH",
    "KINDS",
    "LINKS",
    "MAX_FIELDS",
    "PHASES",
    "FieldLayout",
    "HaloPlan",
    "HaloRuntime",
    "RowLayout",
    "ghost_rows",
    "occurrences",
    "pack_layouts",
    "partition_layouts",
    "plain_site",
    "site_words",
    "unpack_layouts",
]

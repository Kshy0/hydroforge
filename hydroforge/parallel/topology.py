# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Which node every spatial rank runs on.

Placement prefers the links inside a node: the halo partition keeps the
pieces of a cut group on the ranks of one node where it can, and a slab
orders its row blocks so consecutive blocks share a node. Every rank
derives both from the same :class:`Topology`, gathered once.

A node is one operating-system instance (its boot id, else its host name).
``HYDROFORGE_NODES`` overrides the discovery, for tests and for emulating
several nodes on one: a comma-separated node label per rank
(``0,0,1,1``), or one integer, the ranks per node (``2``).
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Topology:
    """The node of every rank, numbered in order of their first rank."""

    nodes: tuple[int, ...]

    @property
    def world(self) -> int:
        return len(self.nodes)

    @property
    def node_count(self) -> int:
        return len(set(self.nodes))

    def members(self, node: int) -> tuple[int, ...]:
        """Ranks of ``node``, ascending."""

        return tuple(rank for rank, value in enumerate(self.nodes) if value == node)

    def positions(self) -> tuple[int, ...]:
        """Block position of every rank: node by node, ranks ascending
        within a node, so consecutive positions cross nodes only between
        the last rank of one node and the first of the next."""

        order = sorted(range(self.world), key=lambda rank: (self.nodes[rank], rank))
        position = [0] * self.world
        for index, rank in enumerate(order):
            position[rank] = index
        return tuple(position)

    @classmethod
    def single(cls, world: int = 1) -> Topology:
        return cls((0,) * world)

    @classmethod
    def of(cls, labels: list[Any]) -> Topology:
        """Number arbitrary node labels by their first rank."""

        numbers: dict[Any, int] = {}
        return cls(tuple(numbers.setdefault(label, len(numbers)) for label in labels))


def node_label() -> str:
    """This process's node: its boot id, else its host name."""

    boot = Path("/proc/sys/kernel/random/boot_id")
    return boot.read_text().strip() if boot.is_file() else socket.gethostname()


def _override(ranks: list[int], world: int) -> Topology | None:
    """The overridden nodes of global ``ranks`` of ``world`` processes."""

    value = os.environ.get("HYDROFORGE_NODES", "").strip()
    if not value:
        return None
    if "," not in value:
        size = int(value)
        if size < 1:
            raise ValueError("HYDROFORGE_NODES must count at least one rank per node")
        return Topology.of([rank // size for rank in ranks])
    labels = [item.strip() for item in value.split(",")]
    if len(labels) != world:
        raise ValueError(
            f"HYDROFORGE_NODES names {len(labels)} nodes for {world} ranks"
        )
    return Topology.of([labels[rank] for rank in ranks])


def gather_topology(group: Any = None) -> Topology:
    """The node of every rank of ``group`` (default WORLD), in group rank
    order (collective over ``group`` when distributed). ``HYDROFORGE_NODES``
    names the nodes of the global ranks."""

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return _override([0], 1) or Topology.single()
    world = dist.get_world_size(group)
    labels: list[Any] = [None] * world
    dist.all_gather_object(labels, node_label(), group=group)
    ranks = [
        r if group is None else dist.get_global_rank(group, r) for r in range(world)
    ]
    return _override(ranks, dist.get_world_size()) or Topology.of(labels)


__all__ = ["Topology", "gather_topology", "node_label"]

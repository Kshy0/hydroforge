# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Deterministic dependency ordering shared by declaration compilers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from graphlib import CycleError, TopologicalSorter
from typing import TypeVar

_Node = TypeVar("_Node", bound=Hashable)


def dependency_order(
    nodes: Iterable[_Node],
    dependencies: Callable[[_Node], Iterable[_Node]],
    *,
    cycle_message: Callable[[tuple[_Node, ...]], str],
) -> tuple[_Node, ...]:
    """Return ``nodes`` and their transitive dependencies, dependencies first.

    Insertion order breaks ties, so the result is deterministic.  A cycle
    raises ``ValueError(cycle_message(cycle))``.
    """

    sorter: TopologicalSorter[_Node] = TopologicalSorter()
    pending = list(nodes)
    seen = set(pending)
    while pending:
        node = pending.pop(0)
        edges = tuple(dependencies(node))
        sorter.add(node, *edges)
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                pending.append(edge)
    try:
        return tuple(sorter.static_order())
    except CycleError as error:
        raise ValueError(cycle_message(tuple(error.args[1]))) from error

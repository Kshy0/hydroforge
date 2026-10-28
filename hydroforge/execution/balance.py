# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Partition pieces: whole groups when they balance, else cut groups.

The rows of the partition key are first placed by group, greedily (largest
group on the least loaded rank). When the most loaded rank then exceeds the
mean by more than the tolerance and the model exchanges a halo, placement
repeats largest group first, and a group that does not fit the least loaded
rank under the tolerated load is cut. Each group is spanned by a tree of its
tree references (``downstream``; other references join what they leave
apart), and a cut moves a subtree (a sub-basin) or a subtree less one below
it (a reach between confluences) to the rank: among the pieces that fit, the
one crossing the fewest references per moved row, so a rank fills with a few
cheap sub-basins. The rank keeps receiving pieces of the group until it is
full; the rest of the group continues on the next one. Every rank computes
the same pieces from the same global inputs.

Ranks on several nodes are filled node first: the groups (cut only if they
do not balance over the nodes, each weighted by its ranks) are placed on
nodes, then each node's rows on its ranks, so a group cut to balance the
ranks of a node exchanges its halo within that node, over its fastest links.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Pieces:
    """The partition ids of the key rows and their placement.

    ``ids`` holds one partition id per key row: the group id when groups
    stay whole, else a piece number. ``cut`` counts the edges between rows
    placed on different ranks.
    """

    ids: np.ndarray
    piece_ids: np.ndarray
    ranks: np.ndarray
    split: bool
    pieces: int
    imbalance: float
    cut: int


@njit(cache=True)
def _groups(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense group of every row and the size of each group."""

    order = np.argsort(values, kind="mergesort")
    dense = np.empty(values.size, np.int64)
    sizes = np.zeros(values.size, np.int64)
    group = -1
    for position in range(order.size):
        row = order[position]
        if position == 0 or values[row] != values[order[position - 1]]:
            group += 1
        dense[row] = group
        sizes[group] += 1
    return dense, sizes[: group + 1]


@njit(cache=True)
def _greedy(weights: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Largest group first on the least loaded bin (load per weight)."""

    order = np.argsort(sizes, kind="mergesort")
    loads = np.zeros(weights.size, np.float64)
    ranks = np.empty(sizes.size, np.int64)
    for position in range(order.size - 1, -1, -1):
        item = order[position]
        rank = int(np.argmin(loads / weights))
        ranks[item] = rank
        loads[rank] += sizes[item]
    return ranks


@njit(cache=True)
def _find(root: np.ndarray, v: int) -> int:
    while root[v] != v:
        root[v] = root[root[v]]
        v = root[v]
    return v


@njit(cache=True)
def _forest(
    dense: np.ndarray,
    child: np.ndarray,
    target: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A spanning tree of each group: parents, Euler ranges, Euler order.

    Tree references (``child -> target``) span first, the other edges only
    join what they leave apart (river trees a bifurcation connects); each
    tree is rooted at a row whose reference is itself (a mouth).
    """

    n = dense.size
    root = np.arange(n)
    chosen_a = np.empty(n, np.int64)
    chosen_b = np.empty(n, np.int64)
    count = 0
    for edges in range(2):
        left, right = (child, target) if edges == 0 else (a, b)
        for k in range(left.size):
            u, w = left[k], right[k]
            if u == w or dense[u] != dense[w]:
                continue
            ru, rw = _find(root, u), _find(root, w)
            if ru != rw:
                root[ru] = rw
                chosen_a[count], chosen_b[count] = u, w
                count += 1
    pointer = np.zeros(n + 1, np.int64)
    for k in range(count):
        pointer[chosen_a[k] + 1] += 1
        pointer[chosen_b[k] + 1] += 1
    for v in range(n):
        pointer[v + 1] += pointer[v]
    fill = pointer[:-1].copy()
    neighbours = np.empty(pointer[n], np.int64)
    for k in range(count):
        u, w = chosen_a[k], chosen_b[k]
        neighbours[fill[u]] = w
        fill[u] += 1
        neighbours[fill[w]] = u
        fill[w] += 1
    starts = np.empty(n + child.size, np.int64)
    found = 0
    for k in range(child.size):
        if child[k] == target[k]:
            starts[found] = child[k]
            found += 1
    for v in range(n):
        starts[found] = v
        found += 1
    parent = np.full(n, -1, np.int64)
    seen = np.zeros(n, np.bool_)
    enter = np.empty(n, np.int64)
    leave = np.empty(n, np.int64)
    rows = np.empty(n, np.int64)
    stack = np.empty(n, np.int64)
    cursor = np.empty(n, np.int64)
    clock = 0
    for k in range(found):
        start = starts[k]
        if seen[start]:
            continue
        seen[start] = True
        depth = 0
        stack[0] = start
        cursor[0] = pointer[start]
        enter[start] = clock
        rows[clock] = start
        clock += 1
        while depth >= 0:
            v = stack[depth]
            if cursor[depth] < pointer[v + 1]:
                u = neighbours[cursor[depth]]
                cursor[depth] += 1
                if seen[u]:
                    continue
                seen[u] = True
                parent[u] = v
                depth += 1
                stack[depth] = u
                cursor[depth] = pointer[u]
                enter[u] = clock
                rows[clock] = u
                clock += 1
            else:
                leave[v] = clock - 1
                depth -= 1
    return parent, enter, leave, rows


@njit(cache=True)
def _lca(
    parent: np.ndarray, rows: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Lowest common ancestor of each edge's ends in the spanning forest
    (Tarjan's offline algorithm over the Euler order), or -1 across trees."""

    n = parent.size
    pointer = np.zeros(n + 1, np.int64)
    for k in range(a.size):
        pointer[a[k] + 1] += 1
        pointer[b[k] + 1] += 1
    for v in range(n):
        pointer[v + 1] += pointer[v]
    fill = pointer[:-1].copy()
    queries = np.empty(pointer[n], np.int64)
    for k in range(a.size):
        queries[fill[a[k]]] = k
        fill[a[k]] += 1
        queries[fill[b[k]]] = k
        fill[b[k]] += 1
    root = np.arange(n)
    ancestor = np.arange(n)
    done = np.zeros(n, np.bool_)
    tree = np.full(n, -1, np.int64)
    lca = np.full(a.size, -1, np.int64)
    # Preorder reversed finishes every child before its parent; a finished
    # row joins its parent's set, whose ancestor is the parent.
    for position in range(n):
        v = rows[position]
        tree[v] = v if parent[v] < 0 else tree[parent[v]]
    for position in range(n - 1, -1, -1):
        v = rows[position]
        done[v] = True
        for k in range(pointer[v], pointer[v + 1]):
            e = queries[k]
            u = b[e] if a[e] == v else a[e]
            if done[u] and lca[e] < 0 and tree[u] == tree[v]:
                lca[e] = ancestor[_find(root, u)]
        p = parent[v]
        if p >= 0:
            rp, rv = _find(root, p), _find(root, v)
            root[rv] = rp
            ancestor[rp] = p
    return lca


@njit(cache=True)
def _cheaper(cost: int, size: int, best_cost: int, best_size: int) -> bool:
    """Fewer crossing edges per moved row, then more rows."""

    if cost * best_size != best_cost * size:
        return cost * best_size < best_cost * size
    return size > best_size


@njit(cache=True)
def _split(
    weights: np.ndarray,
    dense: np.ndarray,
    sizes: np.ndarray,
    parent: np.ndarray,
    enter: np.ndarray,
    leave: np.ndarray,
    rows: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Piece of every row and rank of every piece (see the module doc).

    A cut moves the remaining rows of a subtree ``x``, or of ``x`` less a
    subtree ``y`` below it (a reach between two confluences, or with ``x``
    the whole group, everything but ``y``), that fit the rank's tolerated
    load: the one crossing the fewest edges per moved row, so a rank fills
    with a few cheap sub-basins rather than one expensive cut. ``cross[v]``,
    the remaining edges with one end in ``v``'s remaining subtree, is the
    subtree sum of +1 at both ends of every remaining edge and -2 at their
    lowest common ancestor; ``x`` less ``y`` is costed ``cross[x] +
    cross[y]``.
    """

    n = dense.size
    groups = sizes.size
    world = weights.size
    # Each bin's share of the rows follows its weight.
    mean = n * weights / weights.sum()
    capacity = mean * (1.0 + tolerance)
    window = np.maximum(1.0, 0.5 * tolerance * mean)
    # Rows of each group in Euler order, and its edges.
    first = np.zeros(groups + 1, np.int64)
    for v in range(n):
        first[dense[v] + 1] += 1
    for g in range(groups):
        first[g + 1] += first[g]
    fill = first[:-1].copy()
    members = np.empty(n, np.int64)
    for position in range(n):
        v = rows[position]
        members[fill[dense[v]]] = v
        fill[dense[v]] += 1
    inside = np.zeros(a.size, np.bool_)
    edge_first = np.zeros(groups + 1, np.int64)
    for k in range(a.size):
        inside[k] = dense[a[k]] == dense[b[k]]
        if inside[k]:
            edge_first[dense[a[k]] + 1] += 1
    for g in range(groups):
        edge_first[g + 1] += edge_first[g]
    fill = edge_first[:-1].copy()
    edge_a = np.empty(edge_first[groups], np.int64)
    edge_b = np.empty(edge_first[groups], np.int64)
    for k in range(a.size):
        if inside[k]:
            g = dense[a[k]]
            edge_a[fill[g]], edge_b[fill[g]] = a[k], b[k]
            fill[g] += 1
    edge_lca = _lca(parent, rows, edge_a, edge_b)
    # Remaining rows in each row's Euler range.
    subtree = np.ones(n, np.int64)
    for position in range(n - 1, -1, -1):
        v = rows[position]
        if parent[v] >= 0:
            subtree[parent[v]] += subtree[v]
    cross = np.zeros(n, np.int64)
    piece = dense.copy()
    taken = np.zeros(n, np.bool_)
    ranks = np.full(groups + 2 * n, -1, np.int64)
    loads = np.zeros(world, np.float64)
    pieces = groups
    slots = 8
    above = np.empty(slots, np.int64)
    for g in np.argsort(-sizes, kind="mergesort"):
        left = sizes[g]
        r = -1
        while True:
            # A rank keeps receiving pieces of the group until it is full.
            if r < 0 or loads[r] >= mean[r] - window[r]:
                r = int(np.argmin(loads / weights))
            if loads[r] + left <= capacity[r] or left <= 1:
                ranks[g] = r
                loads[r] += left
                break
            limit = capacity[r] - loads[r]
            for position in range(first[g], first[g + 1]):
                cross[members[position]] = 0
            for e in range(edge_first[g], edge_first[g + 1]):
                u, w = edge_a[e], edge_b[e]
                if taken[u] or taken[w]:
                    continue
                cross[u] += 1
                cross[w] += 1
                if edge_lca[e] >= 0:
                    cross[edge_lca[e]] -= 2
            for position in range(first[g + 1] - 1, first[g] - 1, -1):
                v = members[position]
                if parent[v] >= 0:
                    cross[parent[v]] += cross[v]
            # The cut crossing the fewest edges per moved row; y = -1 for a
            # subtree.
            best_x, best_y, best_size, best_cost = -1, -1, 0, 0
            ups = 0
            for position in range(first[g], first[g + 1]):
                v = members[position]
                size = subtree[v]
                if taken[v] or size >= left:
                    continue
                if size > limit:
                    # The cheapest subtrees too large to move, to less a subtree.
                    if ups < slots:
                        above[ups] = v
                        ups += 1
                    else:
                        worst = 0
                        for k in range(1, ups):
                            if cross[above[k]] > cross[above[worst]]:
                                worst = k
                        if cross[v] < cross[above[worst]]:
                            above[worst] = v
                    continue
                if best_x < 0 or _cheaper(cross[v], size, best_cost, best_size):
                    best_x, best_y, best_size, best_cost = v, -1, size, cross[v]
            # The whole remaining group less a subtree is a candidate too.
            for position in range(first[g], first[g + 1]):
                v = members[position]
                if not taken[v] and parent[v] < 0 and ups < slots:
                    above[ups] = v
                    ups += 1
            for k in range(ups):
                x = above[k]
                whole = subtree[x]
                for position in range(enter[x] + 1, leave[x] + 1):
                    y = rows[position]
                    if taken[y]:
                        continue
                    size = whole - subtree[y]
                    if size > limit or size >= left or size <= 0:
                        continue
                    cost = cross[x] + cross[y]
                    if best_x < 0 or _cheaper(cost, size, best_cost, best_size):
                        best_x, best_y, best_size, best_cost = x, y, size, cost
            if best_x < 0:
                ranks[g] = r
                loads[r] += left
                break
            x, y = best_x, best_y
            moved = subtree[x] - (subtree[y] if y >= 0 else 0)
            hole_low = enter[y] if y >= 0 else 1
            hole_high = leave[y] if y >= 0 else 0
            for position in range(enter[x], leave[x] + 1):
                u = rows[position]
                if not taken[u] and not hole_low <= position <= hole_high:
                    taken[u] = True
                    piece[u] = pieces
            ranks[pieces] = r
            loads[r] += moved
            pieces += 1
            left -= moved
            # Keep every row's count of remaining rows in its range.
            u = x
            while u >= 0:
                subtree[u] -= moved
                u = parent[u]
            if y >= 0:
                u = parent[y]
                while u != x:
                    subtree[u] -= subtree[y]
                    u = parent[u]
    return piece, ranks[:pieces]


def _imbalance(weights: np.ndarray, ranks: np.ndarray, sizes: np.ndarray) -> float:
    """How far the heaviest bin's load per weight exceeds the mean: all
    rows over all weight (a node of three ranks weighs three)."""

    loads = np.bincount(ranks, weights=sizes, minlength=weights.size)
    total = loads.sum()
    if not total:
        return 0.0
    return float((loads / weights).max() / (total / weights.sum()) - 1.0)


def _place(
    weights: np.ndarray,
    dense: np.ndarray,
    sizes: np.ndarray,
    tree: tuple[np.ndarray, np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Piece of every row (dense), bin of every piece, and whether any group
    was cut: whole groups when they balance, else cut groups."""

    ranks = _greedy(weights, sizes)
    if tolerance is None or _imbalance(weights, ranks, sizes) <= tolerance:
        return dense, ranks, False
    parent, enter, leave, rows = _forest(dense, *tree, a, b)
    piece, ranks = _split(
        weights, dense, sizes, parent, enter, leave, rows, a, b, float(tolerance)
    )
    return piece, ranks, True


def _within(rows: np.ndarray, pairs: tuple[np.ndarray, np.ndarray], size: int):
    """The pairs with both ends in ``rows``, renumbered to positions in it."""

    local = np.full(size, -1, np.int64)
    local[rows] = np.arange(rows.size)
    left, right = local[pairs[0]], local[pairs[1]]
    keep = (left >= 0) & (right >= 0)
    return left[keep], right[keep]


def balance(
    world: int,
    groups: np.ndarray,
    tree: tuple[np.ndarray, np.ndarray],
    edges: tuple[np.ndarray, np.ndarray],
    tolerance: float | None,
    nodes: tuple[int, ...] | None = None,
) -> Pieces:
    """Partition ids of the key rows over ``world`` ranks.

    ``groups`` is the partition group of every key row; ``tree`` the
    ``(row, target)`` pairs of the references within the key coordinate
    (``downstream``), along which groups are cut; ``edges`` the row pairs of
    every crossing reference, which a cut separates at a cost.
    ``tolerance`` is the allowed fraction above the mean load; ``None``
    never cuts a group. ``nodes`` holds the node of every rank: groups are
    then placed on nodes first, and cut within a node where they can.
    """

    dense, sizes = _groups(groups)
    a, b = edges
    keep = a != b
    a, b = a[keep], b[keep]
    ones = np.ones(world, np.float64)
    if nodes is None or len(set(nodes)) == 1:
        piece, ranks, split = _place(ones, dense, sizes, tree, a, b, tolerance)
        ids = groups if not split else piece
        return _pieces(world, ids, piece, ranks, a, b, split=split)
    if tolerance is None or _imbalance(ones, _greedy(ones, sizes), sizes) <= tolerance:
        # Whole groups balance the ranks: no halo, nothing to keep in a node.
        return _pieces(world, groups, dense, _greedy(ones, sizes), a, b, split=False)
    labels = sorted(set(nodes))
    members = [[r for r in range(world) if nodes[r] == label] for label in labels]
    weights = np.array([len(m) for m in members], np.float64)
    piece, node_of, split = _place(weights, dense, sizes, tree, a, b, tolerance)
    final = np.empty(dense.size, np.int64)
    ranks: list[int] = []
    for node, ranks_of_node in enumerate(members):
        rows = np.flatnonzero(node_of[piece] == node)
        if rows.size == 0:
            continue
        local_dense, local_sizes = _groups(piece[rows])
        local_tree = _within(rows, tree, dense.size)
        local_a, local_b = _within(rows, (a, b), dense.size)
        local_piece, local_ranks, cut = _place(
            np.ones(len(ranks_of_node), np.float64),
            local_dense,
            local_sizes,
            local_tree,
            local_a,
            local_b,
            tolerance,
        )
        split = split or cut
        final[rows] = len(ranks) + local_piece
        ranks.extend(ranks_of_node[int(r)] for r in local_ranks)
    return _pieces(world, final, final, np.array(ranks, np.int64), a, b, split=split)


def _pieces(
    world: int,
    ids: np.ndarray,
    dense: np.ndarray,
    ranks: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    split: bool,
) -> Pieces:
    row_ranks = ranks[dense]
    sizes = np.bincount(dense, minlength=ranks.size)
    order = np.unique(ids, return_index=True)[1]
    return Pieces(
        ids=np.ascontiguousarray(ids, dtype=np.int64),
        piece_ids=np.ascontiguousarray(ids[order], dtype=np.int64),
        ranks=np.ascontiguousarray(ranks[dense[order]], dtype=np.int64),
        split=split,
        pieces=int(ranks.size),
        imbalance=_imbalance(np.ones(world, np.float64), ranks, sizes),
        cut=int(np.count_nonzero(row_ranks[a] != row_ranks[b])),
    )

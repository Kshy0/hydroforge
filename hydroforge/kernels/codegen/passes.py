# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Statement reordering on kernel IR for block-program printers.

A block program (Triton) runs a statement for all lanes at once, so an
``If`` on a uniform condition is a real branch and every statement outside
one runs whatever the phase.  Two passes make the branches do the work:

:func:`hoist`
    gathers statements into shared branches: each ``If`` on a sample-phase
    test moves down into a later ``If`` on the same test, or on a test that
    implies or is implied by it, merging their branches, or else to the
    latest place it can reach, so unconditional work (and its loads) comes
    first; never across a ``Guard``.
:func:`sink`
    moves each ``Let`` whose value only one branch of a later phase ``If``
    uses into that branch, so the phase that does not take it skips the
    loads.

Both move a statement only past statements it commutes with: neither writes
a buffer or local the other reads or writes (:func:`effects`).  Buffers are
compared by name, so distinct storage slots never alias.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields, is_dataclass

from hydroforge.kernels.codegen.ir import (
    PHASE,
    Assign,
    AtomicAdd,
    Block,
    Evaluate,
    Expr,
    ForK,
    Guard,
    If,
    Let,
    Load,
    PhaseTest,
    Stmt,
    Store,
    Var,
    While,
)


@dataclass(frozen=True, slots=True)
class Effects:
    """What a statement touches: buffers and locals it reads and writes."""

    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    uses: frozenset[str] = frozenset()
    defines: frozenset[str] = frozenset()

    def __or__(self, other: Effects) -> Effects:
        return Effects(
            self.reads | other.reads,
            self.writes | other.writes,
            self.uses | other.uses,
            self.defines | other.defines,
        )

    def commutes(self, other: Effects) -> bool:
        """Whether the two statements may run in either order."""

        return not (
            self.writes & (other.reads | other.writes)
            or other.writes & self.reads
            or self.defines & (other.uses | other.defines)
            or other.defines & self.uses
        )


def subexpressions(node: Expr) -> Iterator[Expr]:
    """The direct operands of ``node``."""

    for field in fields(node):
        value = getattr(node, field.name)
        items = value if isinstance(value, tuple) else (value,)
        yield from (item for item in items if is_dataclass(item))


def _expression(node: Expr) -> Effects:
    reads, uses = set(), set()
    pending = [node]
    while pending:
        item = pending.pop()
        if isinstance(item, Load):
            reads.add(item.buffer)
        elif isinstance(item, Var):
            uses.add(item.name)
        elif isinstance(item, PhaseTest):
            # A phase test reads the kernel's sample phase local.
            uses.add(PHASE.name)
        pending.extend(subexpressions(item))
    return Effects(reads=frozenset(reads), uses=frozenset(uses))


def effects(statements: Stmt | Sequence[Stmt]) -> Effects:
    """The buffers and locals ``statements`` read and write."""

    return _effects(statements, None)


def _effects(statements: Stmt | Sequence[Stmt], cache: dict | None) -> Effects:
    """:func:`effects`, memoized per statement object in ``cache``."""

    if not isinstance(statements, Sequence):
        return _statement(statements, cache)
    result = Effects()
    for node in statements:
        result |= _statement(node, cache)
    return result


def _statement(node: Stmt, cache: dict | None) -> Effects:
    if cache is not None:
        known = cache.get(id(node))
        if known is not None and known[0] is node:
            return known[1]
    match node:
        case Let() | Assign():
            touched = _expression(node.value) | Effects(
                defines=frozenset((node.var.name,))
            )
        case Store() | AtomicAdd():
            touched = (
                _expression(node.index)
                | _expression(node.value)
                | Effects(writes=frozenset((node.buffer,)))
            )
        case If():
            touched = (
                _expression(node.condition)
                | _effects(node.then, cache)
                | _effects(node.orelse, cache)
            )
        case ForK():
            touched = _effects(node.body, cache) | Effects(
                defines=frozenset((node.var.name,))
            )
        case While():
            touched = _expression(node.condition) | _effects(node.body, cache)
        case Evaluate():
            touched = _expression(node.call)
        case Guard():
            touched = _expression(node.condition)
        case Block():
            touched = _effects(node.body, cache)
        case _:
            raise TypeError(f"not a statement: {node!r}")
    if cache is not None:
        # The node is kept with its effects so its id is not reused.
        cache[id(node)] = (node, touched)
    return touched


Implies = Callable[[PhaseTest, PhaseTest], bool]


def _phase_if(node: Stmt) -> bool:
    return isinstance(node, If) and isinstance(node.condition, PhaseTest)


def _within(node: Stmt, rewrite: Callable[[Sequence[Stmt]], tuple[Stmt, ...]]):
    """``node`` with ``rewrite`` applied to each of its statement bodies."""

    match node:
        case If():
            return If(node.condition, rewrite(node.then), rewrite(node.orelse))
        case ForK():
            return ForK(node.var, node.count, rewrite(node.body))
        case While():
            return While(node.condition, rewrite(node.body), per_lane=node.per_lane)
        case Block():
            return Block(rewrite(node.body))
    return node


def hoist(statements: Sequence[Stmt], implies: Implies) -> tuple[Stmt, ...]:
    """Gather each phase ``If`` into the nearest later one it may join.

    An ``If(c, a, b)`` joins a later ``If(c, x, y)`` as ``If(c, a + x, b + y)``.
    With ``implies(c, d)`` it nests at the start of the branch of a later
    ``If(d, x, y)``, and with ``implies(d, c)`` the later ``If`` nests at the
    end of its branch: the nested test never holds where the outer does not,
    and the nested ``If`` has no else branch.
    An ``If`` that joins none moves to the latest place it can reach.  It moves
    down only past statements it commutes with.
    """

    return _hoist(statements, implies, {})


def _hoist(
    statements: Sequence[Stmt], implies: Implies, cache: dict
) -> tuple[Stmt, ...]:
    result: list[Stmt] = []
    for node in reversed(statements):
        node = _within(node, lambda body: _hoist(body, implies, cache))
        position, joined = 0, None
        if _phase_if(node):
            moved = _effects(node, cache)
            for position, target in enumerate(result):
                joined = _join(node, target, implies, cache)
                if joined is not None:
                    result[position] = joined
                    break
                if isinstance(target, Guard) or not moved.commutes(
                    _effects(target, cache)
                ):
                    break
            else:
                position = len(result)
            if joined is not None:
                continue
        result.insert(position, node)
    return tuple(result)


def _join(node: If, target: Stmt, implies: Implies, cache: dict) -> If | None:
    """``node`` and the later ``target`` as one ``If``, if they may join."""

    if not _phase_if(target):
        return None
    if target.condition == node.condition:
        return If(
            node.condition,
            _hoist((*node.then, *target.then), implies, cache),
            _hoist((*node.orelse, *target.orelse), implies, cache),
        )
    # A nested ``If`` must have no else branch: it would no longer run where
    # the outer test fails.
    if not node.orelse and implies(node.condition, target.condition):
        return If(
            target.condition,
            _hoist((node, *target.then), implies, cache),
            target.orelse,
        )
    if not target.orelse and implies(target.condition, node.condition):
        return If(
            node.condition,
            _hoist((*node.then, target), implies, cache),
            node.orelse,
        )
    return None


def sink(statements: Sequence[Stmt]) -> tuple[Stmt, ...]:
    """Move each ``Let`` into the one phase-``If`` branch that uses it.

    The ``Let`` moves when exactly one later statement uses its local, no
    later statement assigns it, that statement is a phase ``If`` whose
    condition does not use it, only one of its branches does, and every
    statement in between commutes with the ``Let``.
    """

    return _sink(statements, {})


def _sink(statements: Sequence[Stmt], cache: dict) -> tuple[Stmt, ...]:
    result = [_within(node, lambda body: _sink(body, cache)) for node in statements]
    for position in range(len(result) - 1, -1, -1):
        node = result[position]
        if not isinstance(node, Let):
            continue
        name = node.var.name
        later = [
            (index, _statement(result[index], cache))
            for index in range(position + 1, len(result))
        ]
        users = [index for index, touched in later if name in touched.uses]
        if len(users) != 1 or any(name in touched.defines for _, touched in later):
            continue
        target = result[users[0]]
        if not _phase_if(target) or name in _expression(target.condition).uses:
            continue
        in_then = name in _effects(target.then, cache).uses
        if in_then == (name in _effects(target.orelse, cache).uses):
            continue
        moved = _effects(node, cache)
        if not all(
            moved.commutes(touched) for index, touched in later if index < users[0]
        ):
            continue
        if in_then:
            target = If(
                target.condition, _sink((node, *target.then), cache), target.orelse
            )
        else:
            target = If(
                target.condition, target.then, _sink((node, *target.orelse), cache)
            )
        result[users[0]] = target
        del result[position]
    return tuple(result)

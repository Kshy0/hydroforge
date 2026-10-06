# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The statistics reduction state machine, written once as kernel IR.

One sample of one variable element updates, in order, the recorded state
of each inner reduction and then every output operation.  A simple
operation folds the sampled value into its window; a compound operation
folds the closing inner result into its outer window at ``INNER_LAST``.
A settle closes inner windows without a sample: each inner result is read
from its recorded state and the compound operations fold it as a sample at
``INNER_LAST`` would.  Recorded states need no reset at a close: the next
window's first sample (``INNER_FIRST``) restarts them, and a settle only
reads a window that has samples.  Sample weights accumulate in
``SampleContext.weight_dtype``, which may be wider than the value dtype.  Printers spell the statements; every
dialect runs the same operation sequence, so their results differ only where
their NaN helpers do.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hydroforge.core.expr import Reduction
from hydroforge.kernels.codegen.ir import (
    Assign,
    Binary,
    Call,
    Compare,
    Const,
    Expr,
    ForK,
    If,
    Let,
    Load,
    Logical,
    Names,
    PhaseTest,
    Select,
    Stmt,
    Store,
    Unary,
    Var,
    cast,
)
from hydroforge.statistics.lowering import LoweredOperation
from hydroforge.statistics.phases import SampleFlags
from hydroforge.statistics.storage import (
    INDEX_DTYPE,
    InnerSlots,
    OperationSlots,
    StoragePlan,
)

INNER_FIRST = PhaseTest(int(SampleFlags.INNER_FIRST))
INNER_LAST = PhaseTest(int(SampleFlags.INNER_LAST))
OUTER_FIRST = PhaseTest(int(SampleFlags.OUTER_FIRST))
STEP_LAST = PhaseTest(int(SampleFlags.STEP_LAST))


@dataclass(frozen=True, slots=True)
class SampleContext:
    """The element one sample updates and the controls it reads.

    ``offset`` addresses the element in every slot of the variable (top-k
    slots hold ``k`` entries per element); ``names`` allocates the locals of
    the kernel and ``prefix`` makes them readable.  ``weight_dtype`` is the
    dtype of the sample weight slots (the value dtype when ``None``).
    """

    offset: Expr
    weight: Expr
    macro_steps: Expr
    macro_index: Expr
    names: Names
    prefix: str
    weight_dtype: torch.dtype | None = None

    def local(self, role: str, dtype: torch.dtype) -> Var:
        return self.names.var(f"{self.prefix}_{role}", dtype)


def _extremum(reduction: Reduction) -> str:
    return "nan_max" if reduction is Reduction.MAX else "nan_min"


def _better(reduction: Reduction) -> str:
    return ">" if reduction is Reduction.MAX else "<"


def _isnan(value: Expr) -> Expr:
    return Call("isnan", (value,), torch.bool)


def _weighted_mean(
    old: Expr,
    weight_slot: str,
    value: Expr,
    dtype: torch.dtype,
    ctx: SampleContext,
    role: str,
) -> tuple[tuple[Let, Store], Expr]:
    """Fold ``value`` into the weighted mean ``old`` of the open window.

    Returns the load and the store of the accumulated weight, which enclose
    every use of the returned mean, and the mean.

    The accumulated weight keeps ``ctx.weight_dtype``: in the value dtype
    alone a float32 sum stops growing once it is 2**24 times the sample
    weight, which turns a long window's mean into a moving average.
    """

    weight_dtype = dtype if ctx.weight_dtype is None else ctx.weight_dtype
    offset = ctx.offset
    old_weight = ctx.local(role, weight_dtype)
    mean = Call(
        "weighted_mean",
        (old, cast(old_weight, dtype), cast(value, dtype), cast(ctx.weight, dtype)),
        dtype,
    )
    return (
        Let(
            old_weight,
            Select(
                INNER_FIRST,
                Const(0, weight_dtype),
                Load(weight_slot, offset, weight_dtype),
            ),
        ),
        Store(
            weight_slot,
            offset,
            Binary("+", old_weight, cast(ctx.weight, weight_dtype)),
        ),
    ), mean


def inner(
    reduction: Reduction,
    value: Expr,
    dtype: torch.dtype,
    slots: InnerSlots,
    ctx: SampleContext,
) -> tuple[tuple[Stmt, ...], Expr]:
    """Record one sample in an inner window; return the statements and the
    window's result, which is valid at ``INNER_LAST``."""

    offset = ctx.offset
    state = Load(slots.state, offset, dtype)
    zero = Const(0, dtype)
    if reduction is Reduction.LAST:
        return (If(STEP_LAST, (Store(slots.state, offset, value),)),), value
    if reduction is Reduction.FIRST:
        result = ctx.local("first_inner", dtype)
        return (
            If(INNER_FIRST, (Store(slots.state, offset, value),)),
            Let(result, state),
        ), result
    result = ctx.local(f"{reduction.value}_inner", dtype)
    old = ctx.local(f"{reduction.value}_inner_old", dtype)
    if reduction in {Reduction.MAX, Reduction.MIN}:
        return (
            Let(old, state),
            Let(
                result,
                Select(
                    INNER_FIRST, value, Call(_extremum(reduction), (old, value), dtype)
                ),
            ),
            Store(slots.state, offset, result),
        ), result
    statements: list[Stmt] = [Let(old, Select(INNER_FIRST, zero, state))]
    if reduction is Reduction.SUM:
        weight = cast(ctx.weight, dtype)
        statements.append(
            Let(result, Binary("+", old, Binary("*", cast(value, dtype), weight)))
        )
    else:
        weighted, mean = _weighted_mean(
            old, slots.weight, value, dtype, ctx, "mean_inner_weight"
        )
        load, store = weighted
        statements += [load, Let(result, mean), store]
    statements.append(Store(slots.state, offset, result))
    return tuple(statements), result


def _simple(
    operation: LoweredOperation,
    value: Expr,
    dtype: torch.dtype,
    slots: OperationSlots,
    ctx: SampleContext,
) -> tuple[Stmt, ...]:
    offset, out = ctx.offset, slots.output
    zero = Const(0, dtype)
    match operation.outer:
        case Reduction.MEAN:
            old = ctx.local("mean_old", dtype)
            weighted, mean = _weighted_mean(
                old, slots.sample_weight, value, dtype, ctx, "mean_weight"
            )
            return (
                Let(old, Select(INNER_FIRST, zero, Load(out, offset, dtype))),
                weighted[0],
                Store(out, offset, mean),
                weighted[1],
            )
        case Reduction.SUM:
            old = ctx.local("sum_old", dtype)
            return (
                Let(old, Select(INNER_FIRST, zero, Load(out, offset, dtype))),
                Store(
                    out,
                    offset,
                    Binary(
                        "+",
                        old,
                        Binary("*", cast(value, dtype), cast(ctx.weight, dtype)),
                    ),
                ),
            )
        case Reduction.MAX | Reduction.MIN:
            folded = Call(
                _extremum(operation.outer), (Load(out, offset, dtype), value), dtype
            )
            return (
                If(
                    INNER_FIRST,
                    (Store(out, offset, value),),
                    (Store(out, offset, folded),),
                ),
            )
        case Reduction.FIRST:
            return (If(INNER_FIRST, (Store(out, offset, value),)),)
    return (If(STEP_LAST, (Store(out, offset, value),)),)


def _arg(
    operation: LoweredOperation,
    value: Expr,
    dtype: torch.dtype,
    slots: OperationSlots,
    ctx: SampleContext,
) -> tuple[Stmt, ...]:
    offset, index = ctx.offset, ctx.macro_index
    out, aux = slots.output, slots.aux
    better = Compare(_better(operation.outer), value, Load(aux, offset, dtype))
    if not dtype.is_floating_point:
        return (
            If(
                OUTER_FIRST,
                (Store(out, offset, index), Store(aux, offset, value)),
                (If(better, (Store(aux, offset, value), Store(out, offset, index))),),
            ),
        )
    old = ctx.local(f"{operation.spelling}_old", dtype)
    return (
        If(
            OUTER_FIRST,
            (
                Store(out, offset, Const(-1, INDEX_DTYPE)),
                Store(aux, offset, Const(float("nan"), dtype)),
            ),
        ),
        Let(old, Load(aux, offset, dtype)),
        If(
            Logical(
                "and",
                (
                    Unary("not", _isnan(value)),
                    Logical(
                        "or",
                        (_isnan(old), Compare(_better(operation.outer), value, old)),
                    ),
                ),
            ),
            (Store(aux, offset, value), Store(out, offset, index)),
        ),
    )


def _top(
    operation: LoweredOperation,
    value: Expr,
    dtype: torch.dtype,
    slots: OperationSlots,
    ctx: SampleContext,
) -> tuple[Stmt, ...]:
    """Insert the value into the sorted top-k entries of its element.

    Entries start missing (NaN); an arg reduction also carries the macro-step
    index and prefers the earlier step among equal values.
    """

    k, name = operation.k, operation.spelling
    base = ctx.local(f"{name}_base", INDEX_DTYPE)
    rank = ctx.local(f"{name}_rank", INDEX_DTYPE)
    entry = Binary("+", base, rank)
    # (slot, carried entry, stored entry, inserted value, missing entry)
    lanes = [
        (
            slots.aux if operation.stores_index else slots.output,
            ctx.local(f"{name}_new", dtype),
            ctx.local(f"{name}_old", dtype),
            value,
            Const(float("nan"), dtype),
        )
    ]
    new, old = lanes[0][1], lanes[0][2]
    precedes: Expr = Compare(_better(operation.outer), new, old)
    if operation.stores_index:
        new_index = ctx.local(f"{name}_new_index", INDEX_DTYPE)
        old_index = ctx.local(f"{name}_old_index", INDEX_DTYPE)
        lanes.append(
            (
                slots.output,
                new_index,
                old_index,
                ctx.macro_index,
                Const(-1, INDEX_DTYPE),
            )
        )
        ties = Logical(
            "and", (Compare("==", new, old), Compare("<", new_index, old_index))
        )
        precedes = Logical("or", (precedes, ties))
    insert = Logical(
        "and", (Unary("not", _isnan(new)), Logical("or", (_isnan(old), precedes)))
    )
    return (
        Let(base, Binary("*", ctx.offset, Const(k, INDEX_DTYPE))),
        *(Let(carried, start) for _slot, carried, _stored, start, _missing in lanes),
        If(
            OUTER_FIRST,
            (
                ForK(
                    rank,
                    k,
                    tuple(Store(slot, entry, missing) for slot, *_, missing in lanes),
                ),
            ),
        ),
        ForK(
            rank,
            k,
            (
                *(
                    Let(stored, Load(slot, entry, stored.type))
                    for slot, _carried, stored, _start, _missing in lanes
                ),
                If(
                    insert,
                    (
                        *(Store(slot, entry, carried) for slot, carried, *_ in lanes),
                        *(
                            Assign(carried, stored)
                            for _slot, carried, stored, *_ in lanes
                        ),
                    ),
                ),
            ),
        ),
    )


def _compound(
    operation: LoweredOperation,
    value: Expr,
    dtype: torch.dtype,
    slots: OperationSlots,
    ctx: SampleContext,
) -> tuple[Stmt, ...]:
    offset, out = ctx.offset, slots.output
    if operation.k > 1:
        return _top(operation, value, dtype, slots, ctx)
    if operation.stores_index:
        return _arg(operation, value, dtype, slots, ctx)
    match operation.outer:
        case Reduction.MEAN:
            count = cast(ctx.macro_steps, dtype)
            one = Const(1, dtype)
            candidate = cast(value, dtype)
            folded = Call(
                "weighted_mean",
                (Load(out, offset, dtype), Binary("-", count, one), candidate, one),
                dtype,
            )
            return (Store(out, offset, Select(OUTER_FIRST, candidate, folded)),)
        case Reduction.SUM:
            candidate = cast(value, dtype)
            return (
                If(
                    OUTER_FIRST,
                    (Store(out, offset, candidate),),
                    (
                        Store(
                            out,
                            offset,
                            Binary("+", Load(out, offset, dtype), candidate),
                        ),
                    ),
                ),
            )
        case Reduction.MAX | Reduction.MIN:
            folded = Call(
                _extremum(operation.outer), (Load(out, offset, dtype), value), dtype
            )
            return (
                If(
                    OUTER_FIRST,
                    (Store(out, offset, value),),
                    (Store(out, offset, folded),),
                ),
            )
        case Reduction.FIRST:
            return (If(OUTER_FIRST, (Store(out, offset, value),)),)
    return (Store(out, offset, value),)


def sample(
    operation: LoweredOperation,
    value: Expr,
    dtype: torch.dtype,
    slots: OperationSlots,
    ctx: SampleContext,
) -> tuple[Stmt, ...]:
    """Update one output operation from a sampled or inner-window value."""

    if operation.inner is None:
        return _simple(operation, value, dtype, slots, ctx)
    return (If(INNER_LAST, _compound(operation, value, dtype, slots, ctx)),)


def _inner_reductions(operations: tuple[LoweredOperation, ...]) -> list[Reduction]:
    return sorted(
        {operation.inner for operation in operations if operation.inner is not None},
        key=lambda item: item.value,
    )


def variable_update(
    variable: str,
    operations: tuple[LoweredOperation, ...],
    value: Expr,
    dtype: torch.dtype,
    ctx: SampleContext,
) -> tuple[Stmt, ...]:
    """One sample of every operation of ``variable`` at one element."""

    statements: list[Stmt] = []
    results: dict[Reduction, Expr] = {}
    for reduction in _inner_reductions(operations):
        slots = StoragePlan.inner_slots(variable, reduction, operations)
        recorded, results[reduction] = inner(reduction, value, dtype, slots, ctx)
        statements.extend(recorded)
    for operation in operations:
        if (
            operation.inner is None
            and operation.outer is Reduction.MEAN
            and Reduction.MEAN in results
        ):
            # The inner mean already recorded this window in the same slots.
            continue
        sampled = value if operation.inner is None else results[operation.inner]
        statements.extend(
            sample(
                operation,
                sampled,
                dtype,
                StoragePlan.operation_slots(variable, operation),
                ctx,
            )
        )
    return tuple(statements)


def variable_settle(
    variable: str,
    operations: tuple[LoweredOperation, ...],
    dtype: torch.dtype,
    ctx: SampleContext,
) -> tuple[Stmt, ...]:
    """Close the inner windows of ``variable`` at one element without a sample.

    Simple operations already hold their window results; each compound
    operation folds its inner window's recorded result.  The next window's
    first sample restarts the recorded state.
    """

    statements: list[Stmt] = []
    results: dict[Reduction, Expr] = {}
    for reduction in _inner_reductions(operations):
        slots = StoragePlan.inner_slots(variable, reduction, operations)
        result = ctx.local(f"{reduction.value}_inner", dtype)
        statements.append(Let(result, Load(slots.state, ctx.offset, dtype)))
        results[reduction] = result
    for operation in operations:
        if operation.inner is not None:
            statements.extend(
                sample(
                    operation,
                    results[operation.inner],
                    dtype,
                    StoragePlan.operation_slots(variable, operation),
                    ctx,
                )
            )
    return tuple(statements)

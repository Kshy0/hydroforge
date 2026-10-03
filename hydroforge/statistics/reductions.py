"""The statistics reduction state machine, written once as kernel IR.

One sample of one variable element updates, in order, the recorded state
of each inner reduction and then every output operation.  A simple
operation folds the sampled value into its window; a compound operation
folds the closing inner result into its outer window at ``INNER_LAST``.
A settle closes inner windows without a sample: each inner result is read
from its recorded state, which restarts, and the compound operations fold it
as a sample at ``INNER_LAST`` would.  Printers spell the statements; every
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
    the kernel and ``prefix`` makes them readable.
    """

    offset: Expr
    weight: Expr
    macro_steps: Expr
    macro_index: Expr
    names: Names
    prefix: str

    def local(self, role: str, dtype: torch.dtype) -> Var:
        return self.names.var(f"{self.prefix}_{role}", dtype)


def _extremum(reduction: Reduction) -> str:
    return "nan_max" if reduction is Reduction.MAX else "nan_min"


def _better(reduction: Reduction) -> str:
    return ">" if reduction is Reduction.MAX else "<"


def _isnan(value: Expr) -> Expr:
    return Call("isnan", (value,), torch.bool)


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
    if reduction in {Reduction.MAX, Reduction.MIN}:
        old = ctx.local(f"{reduction.value}_inner_old", dtype)
        reset = Const(float("-inf" if reduction is Reduction.MAX else "inf"), dtype)
        return (
            Let(old, state),
            Let(
                result,
                Select(
                    INNER_FIRST, value, Call(_extremum(reduction), (old, value), dtype)
                ),
            ),
            If(
                INNER_LAST,
                (Store(slots.state, offset, reset),),
                (Store(slots.state, offset, result),),
            ),
        ), result
    weight = cast(ctx.weight, dtype)
    old = ctx.local(f"{reduction.value}_inner_old", dtype)
    statements: list[Stmt] = [Let(old, Select(INNER_FIRST, zero, state))]
    if reduction is Reduction.SUM:
        statements.append(
            Let(result, Binary("+", old, Binary("*", cast(value, dtype), weight)))
        )
        statements.append(
            If(
                INNER_LAST,
                (Store(slots.state, offset, zero),),
                (Store(slots.state, offset, result),),
            )
        )
        return tuple(statements), result
    old_weight = ctx.local("mean_inner_weight", dtype)
    statements += [
        Let(old_weight, Select(INNER_FIRST, zero, Load(slots.weight, offset, dtype))),
        Let(
            result,
            Call("weighted_mean", (old, old_weight, cast(value, dtype), weight), dtype),
        ),
        If(
            INNER_LAST,
            (Store(slots.state, offset, zero), Store(slots.weight, offset, zero)),
            (
                Store(slots.state, offset, result),
                Store(slots.weight, offset, Binary("+", old_weight, weight)),
            ),
        ),
    ]
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
            weight = cast(ctx.weight, dtype)
            old = ctx.local("mean_old", dtype)
            old_weight = ctx.local("mean_weight", dtype)
            return (
                Let(old, Select(INNER_FIRST, zero, Load(out, offset, dtype))),
                Let(
                    old_weight,
                    Select(INNER_FIRST, zero, Load(slots.sample_weight, offset, dtype)),
                ),
                Store(
                    out,
                    offset,
                    Call(
                        "weighted_mean",
                        (old, old_weight, cast(value, dtype), weight),
                        dtype,
                    ),
                ),
                Store(
                    slots.sample_weight,
                    offset,
                    Select(INNER_LAST, zero, Binary("+", old_weight, weight)),
                ),
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
        recorded, results[reduction] = inner(
            reduction, value, dtype, StoragePlan.inner_slots(variable, reduction), ctx
        )
        statements.extend(recorded)
    for operation in operations:
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
    operation folds its inner window's recorded result, and the inner state
    restarts as a closing sample leaves it.
    """

    statements: list[Stmt] = []
    results: dict[Reduction, Expr] = {}
    for reduction in _inner_reductions(operations):
        slots = StoragePlan.inner_slots(variable, reduction)
        result = ctx.local(f"{reduction.value}_inner", dtype)
        statements.append(Let(result, Load(slots.state, ctx.offset, dtype)))
        if reduction in {Reduction.MAX, Reduction.MIN}:
            reset = float("-inf" if reduction is Reduction.MAX else "inf")
            statements.append(Store(slots.state, ctx.offset, Const(reset, dtype)))
        elif reduction in {Reduction.MEAN, Reduction.SUM}:
            zero = Const(0, dtype)
            statements.extend(
                Store(slot, ctx.offset, zero)
                for slot in (slots.state, slots.weight)
                if slot is not None
            )
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

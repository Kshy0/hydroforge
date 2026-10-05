"""Statistics lowered to kernel IR before any backend spelling.

:func:`plan_statistics` turns a lowered schedule into scatter pre-kernels
(zero, atomic add and, for scatter means, divide), and one sample body and,
with compound statistics, one settle body per variable of every output
group.  An output group becomes one kernel entry once a member mapping is
chosen (:meth:`StatisticsKernelPlan.group_kernels`):

``threads``
    One thread per saved point, member and level of an indexed group.
``loop``
    One thread per saved point, looping over members and levels.
``tiles``
    A block of lanes per saved point, unrolling members; level variables
    are ``[point, level]`` tiles whose level axis is contiguous.

A full-layout group maps one thread to each element in every mapping.
Sample bodies address their element through the locals ``t`` (member),
``point`` (saved point), ``idx`` (source index of the point), ``level``
and, in the full layout, ``linear``; the mapping defines them.  A scatter
add runs one thread per source point and member, or one lane per source
point unrolling members (:meth:`StatisticsKernelPlan.scatter_kernels`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from hydroforge.core.expr import (
    Expression,
    ExpressionSource,
    Reduction,
    ScatterSource,
    TensorSource,
)
from hydroforge.core.naming import sanitize_symbol
from hydroforge.kernels.codegen.expr import lower_expression
from hydroforge.kernels.codegen.ir import (
    PHASE,
    AtomicAdd,
    Binary,
    Block,
    Compare,
    Const,
    Expr,
    ForK,
    Guard,
    If,
    KernelFunction,
    Let,
    Load,
    Logical,
    Names,
    Param,
    PhaseTest,
    Select,
    Stmt,
    Store,
    ThreadIndex,
    TileIndex,
    Var,
    cast,
)
from hydroforge.statistics.lowering import (
    LoweredVariable,
    OutputLayout,
    StatisticsLowering,
)
from hydroforge.statistics.phases import (
    CONTROL_LAYOUT,
    CONTROL_MACRO_INDEX,
    CONTROL_MACRO_STEPS,
    CONTROL_PHASE,
    CONTROL_WEIGHT,
    KERNEL_CONTROLS,
)
from hydroforge.statistics.reductions import (
    SampleContext,
    variable_settle,
    variable_update,
)
from hydroforge.statistics.storage import COUNT_DTYPE, StoragePlan, variable_slots

if TYPE_CHECKING:
    from hydroforge.statistics.layout import StatisticsVariableLayout

_INDEX = torch.int64
LINEAR = Var("linear", _INDEX)
MEMBER = Var("t", _INDEX)
POINT = Var("point", _INDEX)
POINT_LINEAR = Var("point_linear", _INDEX)
SOURCE = Var("idx", _INDEX)
LEVEL = Var("level", _INDEX)
POINTS = Var("n_saved_points", _INDEX)
ELEMENTS = Var("n_elements", _INDEX)
TOTAL = Var("total", _INDEX)
SOURCE_SIZE = Var("source_size", _INDEX)
TARGET_SIZE = Var("target_size", _INDEX)
_SCATTER_SOURCE = Var("src", _INDEX)
_SCATTER_TARGET = Var("dst", _INDEX)
_CONTROL_NAMES = {
    CONTROL_WEIGHT: "weight",
    CONTROL_MACRO_STEPS: "num_macro_steps",
    CONTROL_PHASE: PHASE.name,
    CONTROL_MACRO_INDEX: "macro_step_index",
}
_RESERVED = (
    *(var.name for var in (LINEAR, MEMBER, POINT, POINT_LINEAR, SOURCE, LEVEL)),
    *(var.name for var in (POINTS, ELEMENTS)),
    *(var.name for var in (TOTAL, SOURCE_SIZE, TARGET_SIZE)),
    _SCATTER_SOURCE.name,
    _SCATTER_TARGET.name,
    *_CONTROL_NAMES.values(),
)


@dataclass(frozen=True, slots=True)
class StatisticsCompileContext:
    """Explicit read-only bindings needed by backend code generation.

    Tensor handles refer to runtime-owned storage; code generation neither
    allocates model state nor retains the model or its lifecycle service.
    """

    device: torch.device
    rank: int
    ensemble_size: int
    save_kernels: bool
    kernels_dir: Path | None
    variables: frozenset[str]
    layouts: Mapping[str, StatisticsVariableLayout]
    storage: Mapping[str, torch.Tensor]
    tensors: Mapping[str, torch.Tensor]
    symbol_names: Mapping[str, str]
    control_dtype: torch.dtype

    def symbol(self, name: str) -> str:
        return self.symbol_names.get(name) or sanitize_symbol(name)

    def buffer(self, name: str) -> torch.Tensor:
        tensor = self.tensors.get(name)
        return self.storage[name] if tensor is None else tensor

    def member_stride(self, name: str, logical_rank: int = 1) -> int:
        """The member stride of a bound buffer; 0 for a member-shared one."""

        tensor = self.buffer(name)
        if self.ensemble_size > 1 and tensor.ndim == logical_rank + 1:
            return int(tensor.shape[1])
        return 0

    def shared_plane(self, name: str, output: str) -> int | None:
        """The member plane of a full output that a shared buffer repeats."""

        layout = self.layouts[output]
        if not layout.batched or self.member_stride(
            name, logical_rank=layout.actual_ndim - 1
        ):
            return None
        return max(1, prod(layout.actual_shape[1:]))


@dataclass(frozen=True, slots=True)
class Count:
    """``factor`` times the saved points of the output index ``points``."""

    factor: int
    points: str | None = None

    def value(self, states: Mapping[str, torch.Tensor]) -> int:
        if self.points is None:
            return self.factor
        return self.factor * states[self.points].numel()


@dataclass(frozen=True, slots=True)
class StatisticsKernel:
    """One entry: buffers bind by parameter name, scalars by ``scalars``."""

    function: KernelFunction
    scalars: Mapping[str, Count]
    extent: Count
    phase_mask: int | None


@dataclass(frozen=True, slots=True)
class VariableBody:
    """One variable's sample body and, with compound statistics, its settle
    body (empty otherwise)."""

    name: str
    batched: bool
    level_axis: bool
    levels: int
    numel: int
    body: tuple[Stmt, ...]
    settle: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class GroupKernel:
    """One output group's buffers and variable bodies, before mapping."""

    name: str
    output_index: Param | None
    buffers: tuple[Param, ...]
    controls: tuple[Stmt, ...]
    variables: tuple[VariableBody, ...]
    phase_mask: int | None
    ensemble_size: int


@dataclass(frozen=True, slots=True)
class ScatterKernels:
    """The pre-kernels of one scatter source.

    ``add`` maps one thread to each source point and member; ``add_members``
    maps one lane to each source point and unrolls the members.
    """

    zero: StatisticsKernel
    add: StatisticsKernel
    add_members: StatisticsKernel
    divide: StatisticsKernel | None


Members = Literal["threads", "loop", "tiles"]


@dataclass(frozen=True, slots=True)
class StatisticsKernelPlan:
    lowering: StatisticsLowering
    scatters: tuple[ScatterKernels, ...]
    groups: tuple[GroupKernel, ...]

    def scatter_kernels(self, *, unrolled: bool) -> tuple[StatisticsKernel, ...]:
        """Every scatter pre-kernel in launch order."""

        return tuple(
            kernel
            for scatter in self.scatters
            for kernel in (
                scatter.zero,
                scatter.add_members if unrolled else scatter.add,
                scatter.divide,
            )
            if kernel is not None
        )

    def group_kernels(self, members: Members) -> tuple[StatisticsKernel, ...]:
        """One sample kernel per output group."""

        return tuple(_entry(group, members) for group in self.groups)

    def settle_kernels(self, members: Members) -> tuple[StatisticsKernel, ...]:
        """One settle kernel per output group with compound statistics."""

        return tuple(
            _entry(
                replace(
                    group,
                    name=f"{group.name}_settle",
                    variables=tuple(
                        replace(variable, body=variable.settle)
                        for variable in group.variables
                        if variable.settle
                    ),
                    phase_mask=None,
                ),
                members,
            )
            for group in self.groups
            if any(variable.settle for variable in group.variables)
        )

    def kernels(
        self, members: Literal["threads", "loop"]
    ) -> tuple[StatisticsKernel, ...]:
        """Every sample kernel of a thread program in launch order: the
        scatters, one thread per source point and member, then the groups."""

        return (*self.scatter_kernels(unrolled=False), *self.group_kernels(members))


def _control_params(dtype: torch.dtype) -> dict[str, Param]:
    dtypes = {name: dtype if kind is None else kind for name, kind in CONTROL_LAYOUT}
    return {name: Param(name, dtypes[name], "read") for name in KERNEL_CONTROLS}


def _control_var(param: Param) -> Var:
    return Var(_CONTROL_NAMES[param.name], param.type)


def _read_control(param: Param) -> Let:
    return Let(_control_var(param), Load(param.name, Const(0), param.type))


def _gated(mask: int | None, control: Param) -> tuple[tuple[Param, ...], tuple]:
    """A pre-kernel's phase parameter and the return of skipped samples."""

    if mask is None:
        return (), ()
    return (control,), (_read_control(control), Guard(PhaseTest(mask)))


class _Values:
    """Loads and expressions of one sample, each computed once."""

    def __init__(
        self,
        context: StatisticsCompileContext,
        lowering: StatisticsLowering,
        names: Names,
        prefix: str,
        dtype: torch.dtype,
        offset: Callable[[str], Expr],
    ) -> None:
        self.context = context
        self.sources = lowering.ir.sources
        self.names = names
        self.prefix = prefix
        self.dtype = dtype
        self.offset = offset
        self.statements: list[Stmt] = []
        self.values: dict[str, Var] = {}

    def expression(
        self, expression: Expression, dtype: torch.dtype | None = None
    ) -> Expr:
        names = {name: self.value(name) for name in expression.dependencies}
        return lower_expression(
            expression, names, self.dtype if dtype is None else dtype
        )

    def value(self, name: str) -> Var:
        known = self.values.get(name)
        if known is not None:
            return known
        source = self.sources.get(name) or TensorSource(name)
        if isinstance(source, ExpressionSource):
            # Preserve each virtual field's resolved precision boundary. The
            # consumer's expression casts this value to its own arithmetic type.
            dtype = self.context.layouts[name].dtype
            value = self.expression(source.expression, dtype)
        else:
            key = (
                source.name
                if isinstance(source, TensorSource)
                else StoragePlan.scatter_buffer(name)
            )
            dtype = self.context.buffer(key).dtype
            value = Load(key, self.offset(key), dtype)
        local = self.names.var(f"{self.prefix}_{self.context.symbol(name)}", dtype)
        self.statements.append(Let(local, cast(value, dtype)))
        self.values[name] = local
        return local


def _saved_offset() -> Expr:
    """The saved-point row of the member in indexed storage."""

    return Binary("+", Binary("*", MEMBER, POINTS), POINT)


def _level_offset(row: Expr, width: Const) -> Expr:
    return Binary("+", Binary("*", row, width), LEVEL)


def _member_offset(stride: int, index: Expr) -> Expr:
    if stride == 0:
        return index
    return Binary("+", Binary("*", MEMBER, Const(stride, _INDEX)), index)


def _scatter(
    context: StatisticsCompileContext,
    lowering: StatisticsLowering,
    name: str,
    source: ScatterSource,
    control: Param,
) -> ScatterKernels:
    buffer = StoragePlan.scatter_buffer(name)
    count = (
        StoragePlan.scatter_count(name) if source.reduction is Reduction.MEAN else None
    )
    dtype = context.storage[buffer].dtype
    target_width = context.storage[buffer].shape[-1]
    members = context.ensemble_size if context.layouts[name].batched else 1
    sources = context.tensors[source.index].numel()
    mask = lowering.scatter_phase_mask(name)
    gate_params, gate = _gated(mask, control)
    symbol = context.symbol(name)
    total = Param(TOTAL.name, _INDEX)
    head = (*gate, Let(LINEAR, ThreadIndex()), Guard(Compare("<", LINEAR, TOTAL)))

    def kernel(role, params, body, extent, **scalars) -> StatisticsKernel:
        return StatisticsKernel(
            KernelFunction(f"hf_scatter_{role}_{symbol}", tuple(params), body),
            {"total": Count(extent), **scalars},
            Count(extent),
            mask,
        )

    counts = () if count is None else (count,)
    zero = kernel(
        "zero",
        (
            Param(buffer, dtype, "write"),
            *(Param(item, COUNT_DTYPE, "write") for item in counts),
            *gate_params,
            total,
        ),
        (
            *head,
            Store(buffer, LINEAR, Const(0, dtype)),
            *(Store(item, LINEAR, Const(0, COUNT_DTYPE)) for item in counts),
        ),
        target_width * members,
    )

    names = Names(_RESERVED)
    values = _Values(
        context,
        lowering,
        names,
        "scatter",
        dtype,
        lambda key: _member_offset(context.member_stride(key), _SCATTER_SOURCE),
    )
    value = names.var("contribution", dtype)
    values.statements.append(Let(value, cast(values.expression(source.value), dtype)))
    leaves = [key for key in lowering.ir.scatter_inputs(name) if key != source.index]
    index_dtype = context.tensors[source.index].dtype
    target_offset = Binary("+", Binary("*", MEMBER, TARGET_SIZE), _SCATTER_TARGET)
    add_params = (
        Param(buffer, dtype, "atomic_add"),
        Param(source.index, index_dtype, "read"),
        *(Param(item, COUNT_DTYPE, "atomic_add") for item in counts),
        *(Param(key, context.buffer(key).dtype, "read") for key in leaves),
        *gate_params,
        Param(SOURCE_SIZE.name, _INDEX),
        Param(TARGET_SIZE.name, _INDEX),
    )
    target = (
        Let(
            _SCATTER_TARGET,
            cast(Load(source.index, _SCATTER_SOURCE, index_dtype), _INDEX),
        ),
        Guard(
            Logical(
                "and",
                (
                    Compare(">=", _SCATTER_TARGET, Const(0, _INDEX)),
                    Compare("<", _SCATTER_TARGET, TARGET_SIZE),
                ),
            )
        ),
    )
    contribution = (
        *values.statements,
        AtomicAdd(buffer, target_offset, value),
        *(AtomicAdd(item, target_offset, Const(1, COUNT_DTYPE)) for item in counts),
    )
    sizes = {"source_size": Count(sources), "target_size": Count(target_width)}
    add = kernel(
        "add",
        (*add_params, total),
        (
            *head,
            Let(MEMBER, Binary("/", LINEAR, SOURCE_SIZE)),
            Let(
                _SCATTER_SOURCE,
                Binary("-", LINEAR, Binary("*", MEMBER, SOURCE_SIZE)),
            ),
            *target,
            *contribution,
        ),
        sources * members,
        **sizes,
    )
    add_members = StatisticsKernel(
        KernelFunction(
            f"hf_scatter_add_{symbol}",
            add_params,
            (
                *gate,
                Let(_SCATTER_SOURCE, ThreadIndex()),
                Guard(Compare("<", _SCATTER_SOURCE, SOURCE_SIZE)),
                *target,
                ForK(MEMBER, members, contribution),
            ),
        ),
        sizes,
        Count(sources),
        mask,
    )
    if count is None:
        return ScatterKernels(zero, add, add_members, None)
    number = Var("count", COUNT_DTYPE)
    mean = Binary("/", Load(buffer, LINEAR, dtype), cast(number, dtype))
    divide = kernel(
        "divide",
        (
            Param(buffer, dtype, "read_write"),
            Param(count, COUNT_DTYPE, "read"),
            *gate_params,
            total,
        ),
        (
            *head,
            Let(number, Load(count, LINEAR, COUNT_DTYPE)),
            Store(
                buffer,
                LINEAR,
                Select(
                    Compare(">", number, Const(0, COUNT_DTYPE)),
                    mean,
                    Const(float("nan"), dtype),
                ),
            ),
        ),
        target_width * members,
    )
    return ScatterKernels(zero, add, add_members, divide)


def _variable(
    context: StatisticsCompileContext,
    lowering: StatisticsLowering,
    variable: LoweredVariable,
    names: Names,
    controls: Mapping[str, Param],
) -> VariableBody:
    name = variable.variable.name
    layout = context.layouts[name]
    dtype = layout.dtype
    prefix = context.symbol(name)
    levels = 1
    statements: list[Stmt] = []
    match variable.layout:
        case OutputLayout.FULL:
            offset = LINEAR

            def source_offset(key: str) -> Expr:
                plane = context.shared_plane(key, name)
                if plane is None:
                    return LINEAR
                return Binary("%", LINEAR, Const(plane, _INDEX))

        case OutputLayout.INDEXED_VECTOR:
            offset = names.var(f"{prefix}_offset", _INDEX)
            statements.append(Let(offset, _saved_offset()))

            def source_offset(key: str) -> Expr:
                return _member_offset(context.member_stride(key), SOURCE)

        case _:
            levels = layout.actual_shape[-1]
            width = Const(levels, _INDEX)
            offset = names.var(f"{prefix}_offset", _INDEX)
            statements.append(Let(offset, _level_offset(_saved_offset(), width)))

            def source_offset(key: str) -> Expr:
                point = _member_offset(context.member_stride(key, 2), SOURCE)
                return _level_offset(point, width)

    ctx = SampleContext(
        offset=offset,
        weight=_control_var(controls[CONTROL_WEIGHT]),
        macro_steps=_control_var(controls[CONTROL_MACRO_STEPS]),
        macro_index=_control_var(controls[CONTROL_MACRO_INDEX]),
        names=names,
        prefix=prefix,
    )
    head = tuple(statements)
    values = _Values(context, lowering, names, prefix, dtype, source_offset)
    value = values.value(name)
    statements.extend(values.statements)
    statements.extend(variable_update(name, variable.operations, value, dtype, ctx))
    settle: tuple[Stmt, ...] = ()
    if any(operation.inner is not None for operation in variable.operations):
        settle = (*head, *variable_settle(name, variable.operations, dtype, ctx))
    return VariableBody(
        name,
        layout.batched,
        variable.layout is OutputLayout.INDEXED_LEVEL,
        levels,
        prod(layout.actual_shape),
        tuple(statements),
        settle,
    )


def _group(
    context: StatisticsCompileContext,
    lowering: StatisticsLowering,
    position: int,
    output_index: str,
    variables: tuple[LoweredVariable, ...],
    controls: Mapping[str, Param],
) -> GroupKernel:
    full = output_index == "__full__"
    names = Names(_RESERVED)
    index = (
        None
        if full
        else Param(output_index, context.tensors[output_index].dtype, "read")
    )
    buffers = {} if index is None else {output_index: index}
    inputs = sorted(
        {
            key
            for variable in variables
            for key in lowering.ir.materialized_inputs(variable.variable.name)
        }
    )
    for key in inputs:
        buffers.setdefault(key, Param(key, context.buffer(key).dtype, "read"))
    for variable in variables:
        name = variable.variable.name
        for slot in variable_slots(
            name, variable.variable.operations, (), context.layouts[name].dtype
        ):
            buffers[slot.name] = Param(
                slot.name, context.storage[slot.name].dtype, "read_write"
            )
    buffers.update(controls)
    bodies = tuple(
        _variable(context, lowering, variable, names, controls)
        for variable in variables
    )
    return GroupKernel(
        # The position keeps names distinct when selections sanitize alike.
        name=(
            "hf_full"
            if full
            else f"hf_indexed{position}_{context.symbol(output_index)}"
        ),
        output_index=index,
        buffers=tuple(buffers.values()),
        controls=tuple(_read_control(param) for param in controls.values()),
        variables=bodies,
        phase_mask=lowering.group_phase_mask(output_index),
        ensemble_size=context.ensemble_size,
    )


def plan_statistics(
    context: StatisticsCompileContext, lowering: StatisticsLowering
) -> StatisticsKernelPlan:
    """Lower every scatter and output group of ``lowering`` to kernel IR."""

    controls = _control_params(context.control_dtype)
    scatters = tuple(
        _scatter(
            context, lowering, scatter.name, scatter.source, controls[CONTROL_PHASE]
        )
        for scatter in lowering.ir.ordered_scatters()
    )
    groups = tuple(
        _group(context, lowering, position, output_index, variables, controls)
        for position, (output_index, variables) in enumerate(
            lowering.grouped_variables.items()
        )
    )
    return StatisticsKernelPlan(lowering, scatters, groups)


def _all(conditions: Sequence[Expr]) -> Expr | None:
    if len(conditions) < 2:
        return conditions[0] if conditions else None
    return Logical("and", tuple(conditions))


def _entry(group: GroupKernel, members: Members) -> StatisticsKernel:
    if group.output_index is None:
        return _full_entry(group, members)
    if members == "tiles":
        return _tile_entry(group)
    if members == "loop":
        return _loop_entry(group)
    return _thread_entry(group)


def _full_entry(group: GroupKernel, members: Members) -> StatisticsKernel:
    """One thread per element; members are part of the flat index.

    Tiles leave the variables spanning every element unbounded, so their
    statements share one block.
    """

    extent = max(variable.numel for variable in group.variables)
    body = (
        Let(LINEAR, ThreadIndex()),
        Guard(Compare("<", LINEAR, ELEMENTS)),
        *group.controls,
        *(
            node
            for variable in group.variables
            for node in (
                variable.body
                if members == "tiles" and variable.numel == extent
                else (
                    If(
                        Compare("<", LINEAR, Const(variable.numel, _INDEX)),
                        variable.body,
                    ),
                )
            )
        ),
    )
    params = (*group.buffers, Param(ELEMENTS.name, _INDEX))
    return StatisticsKernel(
        KernelFunction(group.name, params, body),
        {ELEMENTS.name: Count(extent)},
        Count(extent),
        group.phase_mask,
    )


def _indexed(
    group: GroupKernel, body: tuple[Stmt, ...], factor: int
) -> StatisticsKernel:
    index = group.output_index.name
    return StatisticsKernel(
        KernelFunction(group.name, (*group.buffers, Param(POINTS.name, _INDEX)), body),
        {POINTS.name: Count(1, index)},
        Count(factor, index),
        group.phase_mask,
    )


def _source_index(group: GroupKernel) -> Let:
    index = group.output_index
    return Let(SOURCE, cast(Load(index.name, POINT, index.type), _INDEX))


_FIRST_MEMBER = Compare("==", MEMBER, Const(0, _INDEX))


def _loop_entry(group: GroupKernel) -> StatisticsKernel:
    """One thread per saved point, looping over members and levels."""

    batched: list[Stmt] = []
    shared: list[Stmt] = []
    for variable in group.variables:
        body = variable.body
        if variable.level_axis:
            body = (ForK(LEVEL, variable.levels, body),)
        (batched if variable.batched else shared).append(
            body[0] if len(body) == 1 else Block(body)
        )
    if shared:
        batched.append(If(_FIRST_MEMBER, tuple(shared)))
    body = (
        Let(POINT, ThreadIndex()),
        Guard(Compare("<", POINT, POINTS)),
        _source_index(group),
        *group.controls,
        ForK(MEMBER, group.ensemble_size, tuple(batched)),
    )
    return _indexed(group, body, 1)


def _tile_entry(group: GroupKernel) -> StatisticsKernel:
    """A block of lanes per saved point, unrolling members.

    Level variables run on ``[point, level]`` tiles over a power-of-two level
    axis, so each point's levels are contiguous; a group with any of them
    addresses its points as a column.
    """

    batched: list[Stmt] = []
    shared: list[Stmt] = []
    for variable in group.variables:
        body = variable.body
        if variable.level_axis:
            width = 1 << (variable.levels - 1).bit_length()
            levels = Compare("<", LEVEL, Const(variable.levels, _INDEX))
            body = (Block((Let(LEVEL, TileIndex(1, width)), If(levels, body))),)
        (batched if variable.batched else shared).extend(body)
    if shared:
        batched.append(If(_FIRST_MEMBER, tuple(shared)))
    tiled = any(variable.level_axis for variable in group.variables)
    body = (
        Let(POINT, TileIndex(0) if tiled else ThreadIndex()),
        Guard(Compare("<", POINT, POINTS)),
        _source_index(group),
        *group.controls,
        ForK(MEMBER, group.ensemble_size, tuple(batched)),
    )
    return _indexed(group, body, 1)


def _thread_entry(group: GroupKernel) -> StatisticsKernel:
    """One thread per saved point, member and level.

    Vector variables use one thread per point and member; only level
    variables spread over the level axis, so no vector lane idles.
    """

    ensemble = group.ensemble_size
    levels = max(variable.levels for variable in group.variables)
    width = Const(max(1, levels), _INDEX)
    parts = []
    for level_axis in (False, True):
        variables = [
            variable
            for variable in group.variables
            if variable.level_axis == level_axis
        ]
        if not variables:
            continue
        guarded = []
        for variable in variables:
            condition = _all(
                (
                    *(
                        (Compare("<", LEVEL, Const(variable.levels, _INDEX)),)
                        if level_axis
                        else ()
                    ),
                    *(() if variable.batched else (_FIRST_MEMBER,)),
                )
            )
            guarded.append(
                Block(variable.body)
                if condition is None
                else If(condition, variable.body)
            )
        point = Binary("/", LINEAR, width) if level_axis else LINEAR
        parts.append(
            Block(
                (
                    Let(POINT_LINEAR, point),
                    Let(MEMBER, Binary("/", POINT_LINEAR, POINTS)),
                    Let(POINT, Binary("-", POINT_LINEAR, Binary("*", MEMBER, POINTS))),
                    *((Let(LEVEL, Binary("%", LINEAR, width)),) if level_axis else ()),
                    If(
                        Compare("<", MEMBER, Const(ensemble, _INDEX)),
                        (_source_index(group), *guarded),
                    ),
                )
            )
        )
    total = Binary("*", POINTS, Const(ensemble * levels, _INDEX))
    body = (
        Let(LINEAR, ThreadIndex()),
        Guard(Compare("<", LINEAR, total)),
        *group.controls,
        *parts,
    )
    return _indexed(group, body, ensemble * levels)

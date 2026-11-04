# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Cold-path compilation of exact statistics tensor layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from hydroforge.contracts.fields import RuntimeTensorMetadata, concrete_tensor_dtype
from hydroforge.contracts.precision import Precisions
from hydroforge.core.expr import (
    Expression,
    ExpressionSource,
    Reduction,
    ScatterSource,
    TensorSource,
)
from hydroforge.statistics.ir import StatisticsProgram


@dataclass(frozen=True, slots=True)
class AxisMap:
    """Output row ``o`` of an axis reads source row ``table[o]``, or
    ``start + step * o`` without a table."""

    table: str | None = None
    start: int = 0
    step: int = 1

    @property
    def identity(self) -> bool:
        return self.table is None and self.start == 0 and self.step == 1


@dataclass(frozen=True, slots=True)
class ViewMap:
    """Where element ``e`` of an expression of logical ``shape`` reads one
    of its views (:class:`~hydroforge.core.expr.FieldView`) of ``field``.

    A slice reads element ``start_d + step_d * i_d`` along every axis ``d``
    of the field; a gather the flat element ``index[e]``. ``field_shape``
    is the field's logical shape, ``batched`` whether it has members.
    """

    field: str
    shape: tuple[int, ...]
    field_shape: tuple[int, ...]
    batched: bool
    starts: tuple[int, ...] = ()
    steps: tuple[int, ...] = ()
    index: str | None = None


@dataclass(frozen=True, slots=True)
class StatisticsVariableLayout:
    """Fully resolved storage and source addressing for one output variable."""

    actual_shape: tuple[int, ...]
    dtype: torch.dtype
    batched: bool
    stride_input: int
    scatter_extent: int | None = None
    scatter_source_size: int | None = None
    # Logical source shape and how each logical axis reads it; ``axes`` is
    # None when the output is the whole source.
    source_shape: tuple[int, ...] = ()
    axes: tuple[AxisMap, ...] | None = None
    gather: str | None = None
    # The views of an expression source, by key.
    views: Mapping[str, ViewMap] = MappingProxyType({})

    @property
    def actual_ndim(self) -> int:
        return len(self.actual_shape)


@dataclass(frozen=True, slots=True)
class StatisticsCompilation:
    """Complete immutable input to the statistics execution runtime."""

    variable_ops: Mapping[str, tuple[str, ...]]
    program: StatisticsProgram
    layouts: Mapping[str, StatisticsVariableLayout]


@dataclass(frozen=True, slots=True)
class _SourceLayout:
    shape: tuple[int, ...]
    logical_rank: int
    dtype: torch.dtype
    batched: bool
    scatter_extent: int | None = None
    scatter_source_size: int | None = None
    views: Mapping[str, ViewMap] = MappingProxyType({})

    @property
    def logical_axis(self) -> int:
        return 1 if self.batched else 0

    @property
    def logical_extent(self) -> int:
        return self.shape[self.logical_axis]


class _StatisticsLayoutCompiler:
    def __init__(
        self,
        program: StatisticsProgram,
        *,
        tensors: Mapping[str, torch.Tensor],
        fields: Mapping[str, RuntimeTensorMetadata],
        scatter_extents: Mapping[str, int],
        ensemble_size: int,
        precisions: Precisions,
    ) -> None:
        self.program = program
        self.tensors = tensors
        self.fields = fields
        self.scatter_extents = scatter_extents
        self.ensemble_size = ensemble_size
        self.precisions = precisions
        self.sources: dict[str, _SourceLayout] = {}

    def compile(
        self,
        variables: Mapping[str, list[str] | tuple[str, ...]],
    ) -> Mapping[str, StatisticsVariableLayout]:
        layouts = {name: self._selected_layout(name) for name in variables}
        for name, source in self.program.sources.items():
            if name in layouts or not isinstance(
                source,
                (ExpressionSource, ScatterSource),
            ):
                continue
            materialized = self._source_layout(name)
            layouts[name] = StatisticsVariableLayout(
                actual_shape=materialized.shape,
                dtype=materialized.dtype,
                batched=materialized.batched,
                stride_input=(
                    materialized.logical_extent if materialized.batched else 0
                ),
                scatter_extent=materialized.scatter_extent,
                scatter_source_size=materialized.scatter_source_size,
                views=materialized.views,
            )
        return MappingProxyType(layouts)

    def _tensor_layout(self, name: str) -> _SourceLayout:
        tensor = self.tensors[name]
        metadata = self.fields[name].tensor
        logical_rank = len(metadata.shape)
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(
                f"statistics tensor {name!r} must be contiguous strided storage"
            )
        if tensor.dtype != self._declared_dtype(name):
            raise TypeError(
                f"statistics tensor {name!r} has a different dtype than its declaration"
            )
        if tensor.ndim not in {logical_rank, logical_rank + 1}:
            raise ValueError(f"statistics tensor {name!r} has an incompatible rank")
        if tensor.ndim == logical_rank + 1 and tensor.shape[0] != self.ensemble_size:
            raise ValueError(
                f"statistics tensor {name!r} has an incompatible ensemble extent"
            )
        shape = tuple(int(value) for value in tensor.shape)
        batched = tensor.ndim == logical_rank + 1
        return _SourceLayout(
            shape=shape,
            logical_rank=logical_rank,
            dtype=tensor.dtype,
            batched=batched,
        )

    def _declared_dtype(self, name: str) -> torch.dtype:
        metadata = self.fields[name].tensor
        return concrete_tensor_dtype(
            metadata.dtype,
            self.precisions.field_dtype,
            self.precisions.storage_dtype,
        )

    def _source_layout(self, name: str) -> _SourceLayout:
        cached = self.sources.get(name)
        if cached is not None:
            return cached
        source = self.program.sources.get(name) or TensorSource(name)
        if isinstance(source, TensorSource):
            layout = (
                self._tensor_layout(name)
                if source.name == name
                else self._source_layout(source.name)
            )
        elif isinstance(source, ExpressionSource):
            layout = self._expression_layout(
                name,
                source.expression,
            )
        else:
            layout = self._scatter_layout(name, source)
        metadata = self.fields[name]
        declared = metadata.resolved_shape
        if declared is not None:
            logical_shape = layout.shape[int(layout.batched) :]
            if logical_shape != declared:
                raise ValueError(
                    f"statistics field {name!r} resolves to logical shape "
                    f"{logical_shape}, but its owner declares {declared}"
                )
        self.sources[name] = layout
        return layout

    def _expression_layout(
        self,
        name: str,
        expression: Expression,
    ) -> _SourceLayout:
        # Every operand and view reads the expression's logical shape.
        shapes: dict[str, tuple[tuple[int, ...], bool]] = {}
        for item in expression.operands:
            layout = self._source_layout(item)
            shapes[item] = (layout.shape[int(layout.batched) :], layout.batched)
        views = {}
        for view in expression.views:
            field = self._source_layout(view.field)
            field_shape = field.shape[int(field.batched) :]
            if view.index is None:
                ranges = [
                    range(*slice(*bounds).indices(extent))
                    for bounds, extent in zip(view.slices, field_shape)
                ] + [range(extent) for extent in field_shape[len(view.slices) :]]
                shape = tuple(len(item) for item in ranges)
                if 0 in shape:
                    raise ValueError(
                        f"statistics expression {name!r}: {view.key} is empty "
                        f"for {view.field!r} of shape {field_shape}"
                    )
                mapped = ViewMap(
                    view.field,
                    shape,
                    field_shape,
                    field.batched,
                    starts=tuple(item.start for item in ranges),
                    steps=tuple(item.step for item in ranges),
                )
            else:
                index = self.tensors[view.index]
                if index.ndim != 1 or index.dtype not in {torch.int32, torch.int64}:
                    raise ValueError(
                        f"statistics expression {name!r}: {view.index!r} must be "
                        "a one-dimensional integer index"
                    )
                shape = (index.numel(),)
                mapped = ViewMap(
                    view.field, shape, field_shape, field.batched, index=view.index
                )
            views[view.key] = mapped
            shapes[view.key] = (shape, field.batched)
        reference_shape = next(iter(shapes.values()))[0]
        for item, (shape, _batched) in shapes.items():
            if shape != reference_shape:
                raise ValueError(
                    f"statistics expression {name!r} has incompatible resolved "
                    f"shape for {item!r}: {shape}; expected logical shape "
                    f"{reference_shape}"
                )
        batched = any(item for _shape, item in shapes.values())
        dtype = self._declared_dtype(name)
        return _SourceLayout(
            (self.ensemble_size, *reference_shape) if batched else reference_shape,
            len(reference_shape),
            dtype,
            batched,
            views=MappingProxyType(views),
        )

    def _scatter_layout(
        self,
        name: str,
        source: ScatterSource,
    ) -> _SourceLayout:
        value = self._expression_layout(
            name,
            source.value,
        )
        # The declared target domain, not the contributors' largest index:
        # consumers read the buffer at every target.
        index = self.tensors[source.index]
        if (
            index.ndim != 1
            or index.dtype not in {torch.int32, torch.int64}
            or index.numel() != value.logical_extent
        ):
            raise ValueError(
                f"statistics scatter {name!r} requires one shared integer index per contributor"
            )
        if (
            source.reduction is Reduction.MEAN
            and value.logical_extent > torch.iinfo(torch.int32).max
        ):
            raise OverflowError(
                "statistics scatter mean contributor count exceeds int32"
            )
        extent = self.scatter_extents[name]
        shape = (self.ensemble_size, extent) if value.batched else (extent,)
        return _SourceLayout(
            shape=shape,
            logical_rank=1,
            dtype=value.dtype,
            batched=value.batched,
            scatter_extent=extent,
            scatter_source_size=value.logical_extent,
        )

    def _selected_layout(self, name: str) -> StatisticsVariableLayout:
        source = self._source_layout(name)
        source_shape = source.shape[int(source.batched) :]
        output_axes = self.fields[name].output_axes
        gather = self.fields[name].output_gather
        axes = None
        if gather is not None:
            actual_shape = (*source.shape[:int(source.batched)], output_axes[0].extent)
            if not output_axes[0].extent:
                # No point on this rank: nothing is read, nothing addressed.
                gather = None
        elif output_axes is None:
            actual_shape = source.shape
        else:
            if len(output_axes) != len(source_shape):
                raise ValueError(
                    f"statistics output {name!r} has {len(output_axes)} output axes "
                    f"for logical shape {source_shape}"
                )
            for axis, extent in zip(output_axes, source_shape, strict=True):
                last = axis.start + axis.step * (axis.extent - 1)
                if axis.table is None and axis.extent and last >= extent:
                    raise ValueError(
                        f"statistics output {name!r} axis {axis.name!r} exceeds "
                        f"its source extent {extent}"
                    )
            maps = tuple(
                AxisMap(axis.table, axis.start, axis.step) for axis in output_axes
            )
            extents = tuple(axis.extent for axis in output_axes)
            # An output with no element on this rank (a view of rows it does
            # not own) reads nothing, so it needs no addressing.
            if 0 not in extents and (
                not all(item.identity for item in maps) or extents != source_shape
            ):
                axes = maps
            actual_shape = (
                *source.shape[: int(source.batched)],
                *(axis.extent for axis in output_axes),
            )
        return StatisticsVariableLayout(
            actual_shape=actual_shape,
            # 16-bit fields accumulate (and are saved) in float32.
            dtype=(
                torch.float32
                if source.dtype in (torch.float16, torch.bfloat16)
                else source.dtype
            ),
            batched=source.batched,
            stride_input=source.logical_extent if source.batched else 0,
            scatter_extent=source.scatter_extent,
            scatter_source_size=source.scatter_source_size,
            source_shape=source_shape,
            axes=axes,
            gather=gather,
            views=source.views,
        )


def compile_statistics(
    variable_ops: Mapping[str, list[str] | tuple[str, ...]],
    program: StatisticsProgram,
    *,
    tensors: Mapping[str, torch.Tensor],
    fields: Mapping[str, RuntimeTensorMetadata],
    scatter_extents: Mapping[str, int],
    ensemble_size: int,
    precisions: Precisions,
) -> StatisticsCompilation:
    """Compile exact layouts from construction-time validated semantics."""

    normalized = {
        variable: tuple(sorted(operations))
        for variable, operations in variable_ops.items()
    }
    immutable_ops = MappingProxyType(normalized)
    layouts = _StatisticsLayoutCompiler(
        program,
        tensors=tensors,
        fields=fields,
        scatter_extents=scatter_extents,
        ensemble_size=ensemble_size,
        precisions=precisions,
    ).compile(immutable_ops)
    return StatisticsCompilation(immutable_ops, program, layouts)


__all__: list[str] = []

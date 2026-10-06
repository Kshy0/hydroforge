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
from hydroforge.core.expr import (
    Expression,
    ExpressionSource,
    Reduction,
    ScatterSource,
    TensorSource,
)
from hydroforge.statistics.ir import StatisticsProgram


@dataclass(frozen=True, slots=True)
class StatisticsVariableLayout:
    """Fully resolved storage and source addressing for one output variable."""

    actual_shape: tuple[int, ...]
    dtype: torch.dtype
    batched: bool
    stride_input: int
    scatter_extent: int | None = None
    scatter_source_size: int | None = None

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
        base_dtype: torch.dtype,
        mixed_precision: bool,
    ) -> None:
        self.program = program
        self.tensors = tensors
        self.fields = fields
        self.scatter_extents = scatter_extents
        self.ensemble_size = ensemble_size
        self.base_dtype = base_dtype
        self.mixed_precision = mixed_precision
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
            self.base_dtype,
            self.mixed_precision,
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
        dependencies = expression.dependencies
        layouts = tuple(self._source_layout(item) for item in dependencies)
        reference = next((layout for layout in layouts if layout.batched), layouts[0])
        reference_shape = reference.shape[int(reference.batched) :]
        for dependency, layout in zip(dependencies, layouts, strict=True):
            if layout.shape[int(layout.batched) :] != reference_shape:
                raise ValueError(
                    f"statistics expression {name!r} has incompatible resolved "
                    f"shape for {dependency!r}: {layout.shape}; "
                    f"expected logical shape {reference_shape}"
                )
        dtype = self._declared_dtype(name)
        return _SourceLayout(
            reference.shape,
            reference.logical_rank,
            dtype,
            reference.batched,
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

    def _selection(self, name: str) -> torch.Tensor | None:
        output_index = self.fields[name].output_index
        if output_index is None:
            return None
        return self.tensors[output_index]

    def _selected_layout(self, name: str) -> StatisticsVariableLayout:
        source = self._source_layout(name)
        selection = self._selection(name)
        if selection is None:
            actual_shape = source.shape
        else:
            values = list(source.shape)
            values[source.logical_axis] = int(selection.numel())
            actual_shape = tuple(values)
        return StatisticsVariableLayout(
            actual_shape=actual_shape,
            dtype=source.dtype,
            batched=source.batched,
            stride_input=source.logical_extent if source.batched else 0,
            scatter_extent=source.scatter_extent,
            scatter_source_size=source.scatter_source_size,
        )


def compile_statistics(
    variable_ops: Mapping[str, list[str] | tuple[str, ...]],
    program: StatisticsProgram,
    *,
    tensors: Mapping[str, torch.Tensor],
    fields: Mapping[str, RuntimeTensorMetadata],
    scatter_extents: Mapping[str, int],
    ensemble_size: int,
    base_dtype: torch.dtype,
    mixed_precision: bool,
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
        base_dtype=base_dtype,
        mixed_precision=mixed_precision,
    ).compile(immutable_ops)
    return StatisticsCompilation(immutable_ops, program, layouts)


__all__: list[str] = []

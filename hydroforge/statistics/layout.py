"""Cold-path compilation of exact statistics tensor layouts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from hydroforge.contracts.fields import concrete_tensor_dtype
from hydroforge.statistics.ir import (
    Expression, ExpressionSource, ScatterSource, StatisticsProgram,
    TensorSource,
)


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
    def __init__(self, aggregator: Any, program: StatisticsProgram) -> None:
        self.aggregator = aggregator
        self.program = program
        self.num_trials = aggregator.num_trials
        self.sources: dict[str, _SourceLayout] = {}

    def compile(
        self, variables: Mapping[str, list[str] | tuple[str, ...]],
    ) -> Mapping[str, StatisticsVariableLayout]:
        layouts = {
            name: self._selected_layout(name)
            for name in variables
        }
        for name, source in self.program.sources.items():
            if name in layouts or not isinstance(
                source, (ExpressionSource, ScatterSource),
            ):
                continue
            materialized = self._source_layout(name)
            layouts[name] = StatisticsVariableLayout(
                actual_shape=materialized.shape,
                dtype=materialized.dtype,
                batched=materialized.batched,
                stride_input=(
                    materialized.logical_extent
                    if materialized.batched else 0
                ),
                scatter_extent=materialized.scatter_extent,
                scatter_source_size=materialized.scatter_source_size,
            )
        return MappingProxyType(layouts)

    def _field_info(self, name: str):
        return self.aggregator._field_registry[name]

    def _tensor_layout(self, name: str) -> _SourceLayout:
        tensor = self.aggregator._tensor_registry[name]
        metadata = self._field_info(name).tensor
        logical_rank = len(metadata.shape)
        shape = tuple(int(value) for value in tensor.shape)
        batched = tensor.ndim == logical_rank + 1
        return _SourceLayout(
            shape=shape, logical_rank=logical_rank,
            dtype=tensor.dtype, batched=batched,
        )

    def _declared_dtype(self, name: str) -> torch.dtype:
        metadata = self._field_info(name).tensor
        return concrete_tensor_dtype(
            metadata.dtype,
            self.aggregator.base_dtype,
            self.aggregator.mixed_precision,
        )

    def _source_layout(self, name: str) -> _SourceLayout:
        cached = self.sources.get(name)
        if cached is not None:
            return cached
        source = self.program.sources.get(name, TensorSource(name))
        if isinstance(source, TensorSource):
            layout = self._tensor_layout(source.name)
        elif isinstance(source, ExpressionSource):
            layout = self._expression_layout(
                name, source.expression,
            )
        else:
            layout = self._scatter_layout(name, source)
        self.sources[name] = layout
        return layout

    def _expression_layout(
        self,
        name: str,
        expression: Expression,
    ) -> _SourceLayout:
        dependencies = expression.dependencies
        layouts = tuple(self._source_layout(item) for item in dependencies)
        reference = layouts[0]
        dtype = self._declared_dtype(name)
        return _SourceLayout(
            reference.shape, reference.logical_rank, dtype, reference.batched,
        )

    def _scatter_layout(
        self, name: str, source: ScatterSource,
    ) -> _SourceLayout:
        index = self.aggregator._tensor_registry[source.index]
        value = self._expression_layout(
            name, source.value,
        )
        source_size = value.logical_extent
        if index.numel() == 0:
            extent = 0
        else:
            upper = int(index.max().item())
            extent = upper + 1
        # The scatter index only describes contributors, not the full target
        # domain.
        output_index = self._field_info(name).output_index
        if output_index is not None:
            selection = self.aggregator._tensor_registry.get(output_index)
            if selection is not None and selection.numel():
                extent = max(extent, int(selection.max().item()) + 1)
        shape = (
            (self.num_trials, extent) if value.batched else (extent,)
        )
        return _SourceLayout(
            shape=shape, logical_rank=1, dtype=value.dtype,
            batched=value.batched, scatter_extent=extent,
            scatter_source_size=source_size,
        )

    def _selection(self, name: str, extent: int) -> torch.Tensor | None:
        output_index = self._field_info(name).output_index
        if output_index is None:
            return None
        return self.aggregator._tensor_registry[output_index]

    def _selected_layout(self, name: str) -> StatisticsVariableLayout:
        source = self._source_layout(name)
        selection = self._selection(name, source.logical_extent)
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
    aggregator: Any,
    variable_ops: Mapping[str, list[str] | tuple[str, ...]],
    program: StatisticsProgram,
) -> StatisticsCompilation:
    """Compile exact layouts from construction-time validated semantics."""

    normalized = {
        variable: tuple(sorted(operations))
        for variable, operations in variable_ops.items()
    }
    immutable_ops = MappingProxyType(normalized)
    layouts = _StatisticsLayoutCompiler(
        aggregator, program,
    ).compile(immutable_ops)
    return StatisticsCompilation(immutable_ops, program, layouts)


__all__: list[str] = []

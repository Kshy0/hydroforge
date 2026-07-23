"""Cold-path compilation of exact statistics tensor layouts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from hydroforge.statistics.ir import (
    ExpressionSource, ScatterSource, StatisticsProgram, TensorSource,
    compile_statistics_program,
)


@dataclass(frozen=True, slots=True)
class StatisticsVariableLayout:
    """Fully resolved storage and source addressing for one output variable."""

    actual_shape: tuple[int, ...]
    dtype: torch.dtype
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
        self.num_trials = int(aggregator.num_trials)
        self.sources: dict[str, _SourceLayout] = {}
        self.resolving: set[str] = set()

    def compile(
        self, variables: Mapping[str, list[str] | tuple[str, ...]],
    ) -> Mapping[str, StatisticsVariableLayout]:
        layouts = {
            name: self._selected_layout(name)
            for name in variables
        }
        return MappingProxyType(layouts)

    def _field_info(self, name: str):
        try:
            return self.aggregator._field_registry[name]
        except KeyError as error:
            raise ValueError(
                f"Statistics field {name!r} has no registered metadata"
            ) from error

    def _tensor_layout(self, name: str) -> _SourceLayout:
        try:
            tensor = self.aggregator._tensor_registry[name]
        except KeyError as error:
            raise ValueError(
                f"Statistics source {name!r} has no registered tensor"
            ) from error
        metadata = self._field_info(name).tensor
        logical_rank = len(metadata.shape)
        shape = tuple(int(value) for value in tensor.shape)
        if tensor.ndim == logical_rank:
            batched = False
        elif (
            self.num_trials > 1
            and tensor.ndim == logical_rank + 1
            and tensor.shape[0] == self.num_trials
        ):
            batched = True
        else:
            expected = (
                f"rank {logical_rank}, or rank {logical_rank + 1} with "
                f"leading num_trials={self.num_trials}"
                if self.num_trials > 1 else f"rank {logical_rank}"
            )
            raise ValueError(
                f"Statistics tensor {name!r} has shape {shape}; its declared "
                f"tensor_shape requires {expected}"
            )
        if logical_rank < 1:
            raise ValueError(
                f"Statistics tensor {name!r} must declare at least one "
                "logical dimension"
            )
        return _SourceLayout(
            shape=shape, logical_rank=logical_rank,
            dtype=tensor.dtype, batched=batched,
        )

    def _source_layout(self, name: str) -> _SourceLayout:
        cached = self.sources.get(name)
        if cached is not None:
            return cached
        if name in self.resolving:
            raise ValueError(
                f"cyclic statistics layout dependency involving {name!r}"
            )
        self.resolving.add(name)
        try:
            source = self.program.sources.get(name, TensorSource(name))
            if isinstance(source, TensorSource):
                layout = self._tensor_layout(source.name)
            elif isinstance(source, ExpressionSource):
                layout = self._expression_layout(
                    name, source.expression.dependencies,
                )
            elif isinstance(source, ScatterSource):
                layout = self._scatter_layout(name, source)
            else:
                raise TypeError(
                    f"Unsupported statistics source {type(source).__name__}"
                )
        finally:
            self.resolving.remove(name)
        self.sources[name] = layout
        return layout

    def _expression_layout(
        self, name: str, dependencies: tuple[str, ...],
    ) -> _SourceLayout:
        if not dependencies:
            raise ValueError(
                f"Statistics expression {name!r} has no tensor dependency"
            )
        layouts = tuple(self._source_layout(item) for item in dependencies)
        reference = layouts[0]
        incompatible = {
            dependency: layout.shape
            for dependency, layout in zip(dependencies, layouts, strict=True)
            if (
                layout.shape != reference.shape
                or layout.logical_rank != reference.logical_rank
                or layout.batched != reference.batched
            )
        }
        if incompatible:
            raise ValueError(
                f"Statistics expression {name!r} requires identical dependency "
                f"layouts; reference {dependencies[0]!r}={reference.shape}, "
                f"incompatible={incompatible}"
            )
        floating = [layout.dtype for layout in layouts if layout.dtype.is_floating_point]
        if not floating:
            raise TypeError(
                f"Statistics expression {name!r} has no floating-point "
                "dependency from which to define its value dtype"
            )
        dtype = floating[0]
        for dependency_dtype in floating[1:]:
            dtype = torch.promote_types(dtype, dependency_dtype)
        return _SourceLayout(
            reference.shape, reference.logical_rank, dtype, reference.batched,
        )

    def _scatter_layout(
        self, name: str, source: ScatterSource,
    ) -> _SourceLayout:
        index_layout = self._tensor_layout(source.index)
        index = self.aggregator._tensor_registry[source.index]
        if index_layout.batched or index_layout.logical_rank != 1:
            raise ValueError(
                f"Statistics scatter index {source.index!r} must be a shared "
                "one-dimensional tensor"
            )
        if index.dtype not in {torch.int32, torch.int64}:
            raise TypeError(
                f"Statistics scatter index {source.index!r} must have int32 "
                f"or int64 dtype, got {index.dtype}"
            )
        value = self._expression_layout(name, source.value.dependencies)
        if value.logical_rank != 1:
            raise ValueError(
                f"Statistics scatter value {name!r} must have one logical "
                f"dimension, got rank {value.logical_rank}"
            )
        source_size = value.logical_extent
        if index.numel() != source_size:
            raise ValueError(
                f"Statistics scatter {name!r} source length {source_size} "
                f"differs from index length {index.numel()}"
            )
        if index.numel() == 0:
            extent = 0
        else:
            lower = int(index.min().item())
            upper = int(index.max().item())
            if lower < 0:
                raise ValueError(
                    f"Statistics scatter index {source.index!r} contains "
                    f"negative value {lower}"
                )
            extent = upper + 1
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
        try:
            index = self.aggregator._tensor_registry[output_index]
        except KeyError as error:
            raise ValueError(
                f"Statistics output index {output_index!r} for {name!r} is "
                "not registered"
            ) from error
        if index.ndim != 1 or index.dtype not in {torch.int32, torch.int64}:
            raise TypeError(
                f"Statistics output index {output_index!r} must be a "
                "one-dimensional int32 or int64 tensor"
            )
        if index.numel():
            lower = int(index.min().item())
            upper = int(index.max().item())
            if lower < 0 or upper >= extent:
                raise ValueError(
                    f"Statistics output index {output_index!r} for {name!r} "
                    f"has range [{lower}, {upper}], outside [0, {extent})"
                )
        return index

    def _selected_layout(self, name: str) -> StatisticsVariableLayout:
        source = self._source_layout(name)
        info = self._field_info(name)
        if len(info.tensor.shape) != source.logical_rank:
            raise ValueError(
                f"Statistics field {name!r} declares logical rank "
                f"{len(info.tensor.shape)}, but its expression has rank "
                f"{source.logical_rank}"
            )
        if self.num_trials > 1 and not source.batched:
            raise ValueError(
                f"Dynamic statistics field {name!r} is shared in a "
                "multi-trial model; save it as static or make its source "
                "explicitly trial-batched"
            )
        selection = self._selection(name, source.logical_extent)
        if selection is None:
            if isinstance(
                self.program.sources.get(name), ScatterSource,
            ):
                raise ValueError(
                    f"Full-output scatter virtual variable {name!r} is not "
                    "supported; declare an output selection"
                )
            actual_shape = source.shape
        else:
            values = list(source.shape)
            values[source.logical_axis] = int(selection.numel())
            actual_shape = tuple(values)
        maximum_rank = 3 if self.num_trials > 1 else 2
        if len(actual_shape) > maximum_rank:
            raise ValueError(
                f"Statistics field {name!r} has resolved shape {actual_shape}; "
                f"only rank <= {maximum_rank} is supported"
            )
        return StatisticsVariableLayout(
            actual_shape=actual_shape,
            dtype=source.dtype,
            stride_input=source.logical_extent if source.batched else 0,
            scatter_extent=source.scatter_extent,
            scatter_source_size=source.scatter_source_size,
        )


def compile_statistics_layouts(
    aggregator: Any,
    program: StatisticsProgram,
    variables: Mapping[str, list[str] | tuple[str, ...]],
) -> Mapping[str, StatisticsVariableLayout]:
    """Compile all shape, dtype and source-stride decisions exactly once."""

    return _StatisticsLayoutCompiler(aggregator, program).compile(variables)


def compile_statistics(
    aggregator: Any,
    variable_ops: Mapping[str, list[str] | tuple[str, ...]],
) -> StatisticsCompilation:
    """Compile operations, expressions and exact layouts before execution."""

    if not isinstance(variable_ops, Mapping) or not variable_ops:
        raise ValueError("statistics program must contain at least one variable")
    normalized: dict[str, tuple[str, ...]] = {}
    for variable, operations in variable_ops.items():
        if not isinstance(variable, str) or not variable:
            raise TypeError("statistics variable names must be non-empty strings")
        if not isinstance(operations, (list, tuple)) or not operations:
            raise TypeError(
                f"statistics operations for {variable!r} must be a non-empty "
                "list or tuple"
            )
        invalid = [
            operation for operation in operations
            if not isinstance(operation, str) or not operation
        ]
        if invalid:
            raise TypeError(
                f"statistics operations for {variable!r} must be non-empty "
                f"strings, got {invalid!r}"
            )
        canonical = tuple(sorted(operation.lower() for operation in operations))
        if len(canonical) != len(set(canonical)):
            raise ValueError(
                f"statistics operations for {variable!r} contain duplicates: "
                f"{canonical}"
            )
        normalized[variable] = canonical
    immutable_ops = MappingProxyType(normalized)
    program = compile_statistics_program(aggregator, immutable_ops)
    layouts = compile_statistics_layouts(
        aggregator, program, immutable_ops,
    )
    return StatisticsCompilation(immutable_ops, program, layouts)


__all__ = [
    "StatisticsCompilation", "StatisticsVariableLayout",
    "compile_statistics", "compile_statistics_layouts",
]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Typed, backend-neutral statistics intermediate representation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from hydroforge.core.expr import (
    ExpressionSource,
    ScatterSource,
    StatisticOperation,
    TensorSource,
    ValueSource,
)
from hydroforge.statistics.storage import StoragePlan

if TYPE_CHECKING:
    from hydroforge.contracts.fields import RuntimeTensorMetadata
    from hydroforge.statistics.layout import StatisticsVariableLayout


@dataclass(frozen=True)
class StatisticsProgram:
    """Shape-independent statistics semantics compiled before allocation."""

    operations: Mapping[str, tuple[StatisticOperation, ...]]
    sources: Mapping[str, ValueSource]

    def dependencies(self, name: str) -> tuple[str, ...]:
        source = self.sources.get(name) or TensorSource(name)
        if isinstance(source, TensorSource):
            return (source.name,)
        expression = (
            source.expression if isinstance(source, ExpressionSource) else source.value
        )
        return expression.dependencies

    def leaf_tensors(self, name: str) -> tuple[str, ...]:
        """Return concrete model tensors required to evaluate ``name``."""
        leaves: set[str] = set()
        visited: set[str] = set()

        def visit(field: str) -> None:
            if field in visited:
                return
            visited.add(field)
            source = self.sources.get(field) or TensorSource(field)
            if isinstance(source, TensorSource):
                leaves.add(source.name)
                return
            if isinstance(source, ScatterSource):
                leaves.add(source.index)
                dependencies = source.value.dependencies
            else:
                dependencies = source.expression.dependencies
            for dependency in dependencies:
                visit(dependency)

        visit(name)
        return tuple(sorted(leaves))


@dataclass(frozen=True, slots=True)
class StatisticsDeclaration:
    """Construction-time statistics semantics owned by a validated model."""

    program: StatisticsProgram
    static_names: tuple[str, ...]
    netcdf_options: Mapping[str, Mapping[str, Any]]

    @property
    def variable_ops(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                name: tuple(operation.spelling for operation in operations)
                for name, operations in self.program.operations.items()
            }
        )


@dataclass(frozen=True)
class MaterializedScatter:
    """One reachable scatter source that must run before aggregation."""

    name: str
    source: ScatterSource


@dataclass(frozen=True)
class StatisticVariable:
    name: str
    safe_name: str
    source: ValueSource
    operations: tuple[StatisticOperation, ...]
    tensor_shape: tuple[Any, ...]
    actual_shape: tuple[int, ...]
    actual_ndim: int
    output_group: str


@dataclass(frozen=True)
class StatisticsIR:
    """Complete aggregation program consumed by backend syntax emitters."""

    variables: tuple[StatisticVariable, ...]
    by_name: Mapping[str, StatisticVariable]
    grouped_variables: Mapping[str, tuple[StatisticVariable, ...]]
    sources: Mapping[str, ValueSource]

    def materialized_inputs(self, name: str) -> tuple[str, ...]:
        """Return leaf buffers read by the main aggregation kernel."""
        inputs: set[str] = set()
        visited: set[str] = set()

        def visit(field: str) -> None:
            if field in visited:
                return
            visited.add(field)
            source = self.sources.get(field) or TensorSource(field)
            if isinstance(source, TensorSource):
                inputs.add(source.name)
            elif isinstance(source, ScatterSource):
                inputs.add(StoragePlan.scatter_buffer(field))
            else:
                for dependency in source.expression.dependencies:
                    visit(dependency)

        visit(name)
        return tuple(sorted(inputs))

    def scatter_inputs(self, name: str) -> tuple[str, ...]:
        """Return source and index buffers for one scatter pre-kernel."""
        source = self.sources[name]
        inputs = {source.index}
        for dependency in source.value.dependencies:
            inputs.update(self.materialized_inputs(dependency))
        return tuple(sorted(inputs))

    def ordered_scatters(self) -> tuple[MaterializedScatter, ...]:
        """Topologically order scatter materializations by virtual dependency."""
        result: list[MaterializedScatter] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            source = self.sources.get(name) or TensorSource(name)
            if isinstance(source, TensorSource):
                return
            dependencies = (
                source.value.dependencies
                if isinstance(source, ScatterSource)
                else source.expression.dependencies
            )
            for dependency in dependencies:
                visit(dependency)
            if isinstance(source, ScatterSource):
                result.append(MaterializedScatter(name, source))

        for variable in self.variables:
            visit(variable.name)
        return tuple(result)


def build_statistics_ir(
    program: StatisticsProgram,
    *,
    fields: Mapping[str, RuntimeTensorMetadata],
    layouts: Mapping[str, StatisticsVariableLayout],
    symbol_names: Mapping[str, str],
) -> StatisticsIR:

    variables: list[StatisticVariable] = []
    groups: dict[str, list[StatisticVariable]] = {}
    for name in sorted(program.operations):
        info = fields[name]
        metadata = info.tensor
        layout = layouts[name]
        group = info.output_index or "__full__"
        variable = StatisticVariable(
            name=name,
            safe_name=symbol_names[name],
            source=program.sources.get(name) or TensorSource(name),
            operations=program.operations[name],
            tensor_shape=metadata.shape,
            actual_shape=layout.actual_shape,
            actual_ndim=layout.actual_ndim,
            output_group=group,
        )
        variables.append(variable)
        groups.setdefault(group, []).append(variable)

    by_name = MappingProxyType({variable.name: variable for variable in variables})
    grouped = MappingProxyType({key: tuple(value) for key, value in groups.items()})
    return StatisticsIR(
        tuple(variables),
        by_name,
        grouped,
        program.sources,
    )

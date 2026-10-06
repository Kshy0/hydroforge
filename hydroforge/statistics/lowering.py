# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Backend-neutral execution schedule for statistics code generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType

from hydroforge.core.expr import (
    ExpressionSource,
    Reduction,
    ScatterSource,
    StatisticOperation,
)
from hydroforge.statistics.ir import StatisticsIR, StatisticVariable
from hydroforge.statistics.phases import SampleFlags


class OutputLayout(StrEnum):
    """Physical layout selected before any backend emits source."""

    __str__ = Enum.__str__
    __format__ = Enum.__format__

    FULL = "full"
    INDEXED_VECTOR = "indexed_vector"
    INDEXED_LEVEL = "indexed_level"


class SamplePhase(StrEnum):
    """When the source value participates in an operation."""

    __str__ = Enum.__str__
    __format__ = Enum.__format__

    EVERY_SUBSTEP = "every_substep"
    INNER_FIRST = "inner_first"
    INNER_LAST = "inner_last"
    STEP_LAST = "step_last"


_PHASE_BITS = {
    SamplePhase.INNER_FIRST: int(SampleFlags.INNER_FIRST),
    SamplePhase.INNER_LAST: int(SampleFlags.INNER_LAST),
    SamplePhase.STEP_LAST: int(SampleFlags.STEP_LAST),
}


def sample_phase_mask(phases: Iterable[SamplePhase]) -> int | None:
    """Return the phase bits needing a launch, or ``None`` for every sample."""
    mask = 0
    for phase in phases:
        if phase is SamplePhase.EVERY_SUBSTEP:
            return None
        mask |= _PHASE_BITS[phase]
    return mask


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    """One normalized operation with all scheduling decisions resolved."""

    spelling: str
    phase: SamplePhase
    value_phase: SamplePhase
    outer: Reduction
    inner: Reduction | None
    k: int
    stores_index: bool

    @property
    def compound(self) -> bool:
        return self.inner is not None


@dataclass(frozen=True, slots=True)
class LoweredVariable:
    """One variable's layout, source schedule and output operations."""

    variable: StatisticVariable
    layout: OutputLayout
    operations: tuple[LoweredOperation, ...]

    @property
    def value_phases(self) -> frozenset[SamplePhase]:
        """Samples at which the source value is read."""
        return frozenset(operation.value_phase for operation in self.operations)

    @property
    def launch_phases(self) -> frozenset[SamplePhase]:
        """Samples at which any source read or state update happens."""
        return self.value_phases | {operation.phase for operation in self.operations}


@dataclass(frozen=True, slots=True)
class StatisticsLowering:
    """The sole semantic input consumed by backend syntax emitters.

    ``per_substep`` says whether any operation needs samples before the last
    substep of a step, so a device loop must fold statistics into every
    iteration; ``compound`` whether a window close can require a settle fold.
    """

    ir: StatisticsIR
    variables: tuple[LoweredVariable, ...]
    by_name: Mapping[str, LoweredVariable]
    grouped_variables: Mapping[str, tuple[LoweredVariable, ...]]
    per_substep: bool
    compound: bool

    def operations(self, name: str) -> tuple[LoweredOperation, ...]:
        """Return the only backend-visible operation schedule for ``name``."""
        return self.by_name[name].operations

    @property
    def groups(self) -> Mapping[str, tuple[str, ...]]:
        """Return backend launch groups without exposing the shape IR."""
        return MappingProxyType(
            {
                group: tuple(item.variable.name for item in variables)
                for group, variables in self.grouped_variables.items()
            }
        )

    def group_phase_mask(self, group: str) -> int | None:
        """Return when one output group's update kernel has any effect."""
        return sample_phase_mask(
            phase
            for variable in self.grouped_variables[group]
            for phase in variable.launch_phases
        )

    def scatter_phase_mask(self, name: str) -> int | None:
        """Return when any consumer reads one materialized scatter buffer."""
        phases: set[SamplePhase] = set()
        for variable in self.variables:
            if name in _source_closure(self.ir, variable.variable.name):
                phases.update(variable.value_phases)
        return sample_phase_mask(phases)


def _layout(variable: StatisticVariable) -> OutputLayout:
    if variable.output_group == "__full__":
        return OutputLayout.FULL
    if len(variable.tensor_shape) == 2:
        return OutputLayout.INDEXED_LEVEL
    return OutputLayout.INDEXED_VECTOR


def _phase(operation: StatisticOperation) -> SamplePhase:
    if operation.compound:
        return SamplePhase.INNER_LAST
    match operation.outer:
        case Reduction.FIRST:
            return SamplePhase.INNER_FIRST
        case Reduction.LAST:
            return SamplePhase.STEP_LAST
        case _:
            return SamplePhase.EVERY_SUBSTEP


def _value_phase(operation: StatisticOperation) -> SamplePhase:
    """``last`` values are recorded at every step's last sample.

    A window whose closing step collects no output is folded from recorded
    state, so its last sampled value must already be stored.
    """
    match operation.inner:
        case None:
            return _phase(operation)
        case Reduction.LAST:
            return SamplePhase.STEP_LAST
        case Reduction.FIRST:
            return SamplePhase.INNER_FIRST
        case _:
            return SamplePhase.EVERY_SUBSTEP


def _source_closure(ir: StatisticsIR, name: str) -> frozenset[str]:
    names: set[str] = set()
    pending = [name]
    while pending:
        field = pending.pop()
        if field in names:
            continue
        names.add(field)
        source = ir.sources.get(field)
        if isinstance(source, ScatterSource):
            pending.extend(source.value.dependencies)
        elif isinstance(source, ExpressionSource):
            pending.extend(source.expression.dependencies)
    return frozenset(names)


def lower_statistics(ir: StatisticsIR) -> StatisticsLowering:
    """Resolve layouts and sample phases once before backend generation."""
    variables: list[LoweredVariable] = []
    groups: dict[str, list[LoweredVariable]] = {}
    for variable in ir.variables:
        layout = _layout(variable)
        operations = tuple(
            LoweredOperation(
                spelling=operation.spelling,
                phase=_phase(operation),
                value_phase=_value_phase(operation),
                outer=operation.outer,
                inner=operation.inner,
                k=operation.k,
                stores_index=operation.stores_index,
            )
            for operation in variable.operations
        )
        lowered = LoweredVariable(
            variable=variable, layout=layout, operations=operations
        )
        variables.append(lowered)
        groups.setdefault(variable.output_group, []).append(lowered)
    return StatisticsLowering(
        ir=ir,
        variables=tuple(variables),
        by_name=MappingProxyType(
            {variable.variable.name: variable for variable in variables}
        ),
        grouped_variables=MappingProxyType(
            {name: tuple(group) for name, group in groups.items()}
        ),
        per_substep=any(
            phase in {SamplePhase.EVERY_SUBSTEP, SamplePhase.INNER_FIRST}
            for variable in variables
            for phase in variable.launch_phases
        ),
        compound=any(
            operation.compound
            for variable in variables
            for operation in variable.operations
        ),
    )

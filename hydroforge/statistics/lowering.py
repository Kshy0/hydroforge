"""Backend-neutral execution schedule for statistics code generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from hydroforge.statistics.ir import (
    Reduction,
    StatisticOperation,
    StatisticVariable,
    StatisticsIR,
)


class OutputLayout(str, Enum):
    """Physical layout selected before any backend emits source."""

    FULL = "full"
    INDEXED_VECTOR = "indexed_vector"
    INDEXED_LEVEL = "indexed_level"


class SamplePhase(str, Enum):
    """When the source value participates in an operation."""

    EVERY_SUBSTEP = "every_substep"
    INNER_FIRST = "inner_first"
    INNER_LAST = "inner_last"
    MIDDLE = "middle"


class ReductionAction(str, Enum):
    """Backend-independent mutation performed for one reduction."""

    WEIGHTED_MEAN = "weighted_mean"
    WEIGHTED_SUM = "weighted_sum"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    TAKE_FIRST = "take_first"
    TAKE_LAST = "take_last"
    TAKE_MIDDLE = "take_middle"


@dataclass(frozen=True, slots=True)
class ReductionPlan:
    """Fully lowered reduction behavior consumed by syntax emitters."""

    reduction: Reduction
    action: ReductionAction
    initialization: str

    @property
    def value(self) -> str:
        return self.reduction.value


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    """One normalized operation with all scheduling decisions resolved."""

    spelling: str
    phase: SamplePhase
    outer: ReductionPlan
    inner: ReductionPlan | None
    k: int
    stores_index: bool

    @property
    def output(self) -> Reduction:
        return self.outer.reduction

    @property
    def compound(self) -> bool:
        return self.inner is not None

    @property
    def value_reduction(self) -> Reduction | None:
        return None if self.inner is None else self.inner.reduction


@dataclass(frozen=True, slots=True)
class LoweredVariable:
    """One variable's layout, source schedule and output operations."""

    variable: StatisticVariable
    layout: OutputLayout
    operations: tuple[LoweredOperation, ...]
    inner_reductions: tuple[Reduction, ...]
    needs_unconditional_value: bool


@dataclass(frozen=True, slots=True)
class StatisticsLowering:
    """The sole semantic input consumed by backend syntax emitters."""

    ir: StatisticsIR
    variables: tuple[LoweredVariable, ...]
    by_name: Mapping[str, LoweredVariable]
    grouped_variables: Mapping[str, tuple[LoweredVariable, ...]]
    required_flags: frozenset[str]

    def operations(self, name: str) -> tuple[LoweredOperation, ...]:
        """Return the only backend-visible operation schedule for ``name``."""
        return self.by_name[name].operations

    @property
    def groups(self) -> Mapping[str, tuple[str, ...]]:
        """Return backend launch groups without exposing the shape IR."""
        return MappingProxyType({
            group: tuple(item.variable.name for item in variables)
            for group, variables in self.grouped_variables.items()
        })

    def inner_reductions(self, name: str) -> tuple[Reduction, ...]:
        return self.by_name[name].inner_reductions

    def variables_by_inner(
        self, names: Iterable[str],
    ) -> Mapping[Reduction, tuple[str, ...]]:
        """Group variables by compiled inner schedule in stable order."""
        grouped: dict[Reduction, list[str]] = {}
        for name in names:
            for reduction in self.by_name[name].inner_reductions:
                grouped.setdefault(reduction, []).append(name)
        return MappingProxyType({
            reduction: tuple(variables)
            for reduction, variables in grouped.items()
        })

    def split_indexed(
        self, names: Iterable[str],
    ) -> tuple[list[str], list[str]]:
        """Partition an indexed launch group by normalized output layout."""
        vectors: list[str] = []
        levels: list[str] = []
        for name in names:
            layout = self.by_name[name].layout
            if layout is OutputLayout.INDEXED_VECTOR:
                vectors.append(name)
            elif layout is OutputLayout.INDEXED_LEVEL:
                levels.append(name)
            else:
                raise ValueError(
                    f"full-output variable {name!r} entered an indexed group"
                )
        return vectors, levels


def _layout(variable: StatisticVariable, num_trials: int) -> OutputLayout:
    if variable.output_group == "__full__":
        return OutputLayout.FULL
    level_ndim = 3 if num_trials > 1 else 2
    if variable.actual_ndim == level_ndim:
        return OutputLayout.INDEXED_LEVEL
    return OutputLayout.INDEXED_VECTOR


def _phase(operation: StatisticOperation) -> SamplePhase:
    if operation.compound:
        return SamplePhase.INNER_LAST
    match operation.outer:
        case Reduction.FIRST:
            return SamplePhase.INNER_FIRST
        case Reduction.LAST:
            return SamplePhase.INNER_LAST
        case Reduction.MID:
            return SamplePhase.MIDDLE
        case _:
            return SamplePhase.EVERY_SUBSTEP


def _reduction_plan(reduction: Reduction) -> ReductionPlan:
    action, initialization = {
        Reduction.MEAN: (ReductionAction.WEIGHTED_MEAN, "zero"),
        Reduction.SUM: (ReductionAction.WEIGHTED_SUM, "zero"),
        Reduction.MAX: (ReductionAction.MAXIMUM, "negative_infinity"),
        Reduction.MIN: (ReductionAction.MINIMUM, "positive_infinity"),
        Reduction.FIRST: (ReductionAction.TAKE_FIRST, "zero"),
        Reduction.LAST: (ReductionAction.TAKE_LAST, "zero"),
        Reduction.MID: (ReductionAction.TAKE_MIDDLE, "zero"),
    }[reduction]
    return ReductionPlan(reduction, action, initialization)


def _validate_layout(
    variable: StatisticVariable,
    layout: OutputLayout,
) -> None:
    if layout is OutputLayout.INDEXED_LEVEL:
        unsupported = next((
            operation for operation in variable.operations
            if operation.compound or operation.k > 1 or operation.stores_index
        ), None)
        if unsupported is not None:
            raise ValueError(
                f"Level variable {variable.name!r} does not support compound, "
                f"top-k, or arg operation {unsupported.spelling!r}"
            )
    if layout is OutputLayout.FULL:
        unsupported = next((
            operation for operation in variable.operations
            if operation.k > 1 or operation.stores_index
        ), None)
        if unsupported is not None:
            raise ValueError(
                f"Full-output variable {variable.name!r} does not support "
                f"top-k or arg operation {unsupported.spelling!r}"
            )


def lower_statistics(
    ir: StatisticsIR,
    *,
    num_trials: int,
) -> StatisticsLowering:
    """Resolve layouts and sample phases once before backend generation."""
    variables: list[LoweredVariable] = []
    groups: dict[str, list[LoweredVariable]] = {}
    for variable in ir.variables:
        layout = _layout(variable, num_trials)
        _validate_layout(variable, layout)
        operations = tuple(
            LoweredOperation(
                spelling=operation.spelling,
                phase=_phase(operation),
                outer=_reduction_plan(operation.outer),
                inner=(
                    None if operation.inner is None
                    else _reduction_plan(operation.inner)
                ),
                k=operation.k,
                stores_index=operation.stores_index,
            )
            for operation in variable.operations
        )
        lowered = LoweredVariable(
            variable=variable,
            layout=layout,
            operations=operations,
            inner_reductions=tuple(dict.fromkeys(
                operation.inner
                for operation in variable.operations
                if operation.inner is not None
            )),
            needs_unconditional_value=any(
                operation.phase is SamplePhase.EVERY_SUBSTEP
                or (
                    operation.value_reduction is not None
                    and operation.value_reduction
                    not in {Reduction.FIRST, Reduction.LAST, Reduction.MID}
                )
                for operation in operations
            ),
        )
        variables.append(lowered)
        groups.setdefault(variable.output_group, []).append(lowered)
    flags: set[str] = set()
    for variable in variables:
        for operation in variable.operations:
            if operation.compound:
                flags.update({
                    "is_inner_last", "is_outer_first", "is_outer_last",
                })
                continue
            match operation.output:
                case Reduction.FIRST | Reduction.MAX | Reduction.MIN | Reduction.SUM:
                    flags.update({"is_inner_first", "is_inner_last"})
                case Reduction.LAST:
                    flags.add("is_inner_last")
                case Reduction.MID:
                    flags.update({"is_middle", "is_inner_last"})
                case Reduction.MEAN:
                    flags.update({
                        "is_inner_first", "is_inner_last", "is_outer_last",
                    })
    return StatisticsLowering(
        ir=ir,
        variables=tuple(variables),
        by_name=MappingProxyType({
            variable.variable.name: variable for variable in variables
        }),
        grouped_variables=MappingProxyType({
            name: tuple(group) for name, group in groups.items()
        }),
        required_flags=frozenset(flags),
    )

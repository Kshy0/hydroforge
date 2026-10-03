"""Accumulator storage of compiled statistics and the one source of its names.

Every slot a statistics program allocates, binds or publishes is named by a
:class:`StoragePlan` rule: operation outputs ``{variable}_{operation}``,
arg-reduction values ``..._aux``, the sample weight of a simple mean, inner
window states and weights, and scatter buffers and counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from hydroforge.core.expr import Reduction, ScatterSource, StatisticOperation

if TYPE_CHECKING:
    from hydroforge.statistics.ir import StatisticsProgram
    from hydroforge.statistics.layout import StatisticsVariableLayout

INDEX_DTYPE = torch.int64
COUNT_DTYPE = torch.int32


class StorageInitialization(StrEnum):
    __str__ = Enum.__str__
    __format__ = Enum.__format__

    ZERO = "zero"
    NEGATIVE_INFINITY = "negative_infinity"
    POSITIVE_INFINITY = "positive_infinity"


def storage_initialization(reduction: Reduction) -> StorageInitialization:
    if reduction is Reduction.MAX:
        return StorageInitialization.NEGATIVE_INFINITY
    if reduction is Reduction.MIN:
        return StorageInitialization.POSITIVE_INFINITY
    return StorageInitialization.ZERO


@dataclass(frozen=True, slots=True)
class StorageSlot:
    """One allocation; ``owner`` is its statistic variable or scatter field."""

    name: str
    owner: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    initialization: StorageInitialization
    output: bool


@dataclass(frozen=True, slots=True)
class OperationSlots:
    """The slots one output operation updates."""

    output: str
    aux: str | None
    sample_weight: str | None


@dataclass(frozen=True, slots=True)
class InnerSlots:
    """The recorded state of one inner window reduction."""

    state: str
    weight: str | None


@dataclass(frozen=True, slots=True)
class StoragePlan:
    """Every storage slot of one program, in allocation order."""

    slots: Mapping[str, StorageSlot]

    @staticmethod
    def output(variable: str, operation: str) -> str:
        return f"{variable}_{operation}"

    @staticmethod
    def aux(variable: str, operation: str) -> str:
        return f"{variable}_{operation}_aux"

    @staticmethod
    def sample_weight(variable: str) -> str:
        return f"{variable}_mean_sample_weight_state"

    @staticmethod
    def inner_state(variable: str, inner: Reduction) -> str:
        return f"{variable}_{inner.value}_inner_state"

    @staticmethod
    def inner_weight(variable: str, inner: Reduction) -> str:
        return f"{variable}_{inner.value}_weight_state"

    @staticmethod
    def scatter_buffer(name: str) -> str:
        return f"__scatter_buf_{name}"

    @staticmethod
    def scatter_count(name: str) -> str:
        return f"__scatter_cnt_{name}"

    @classmethod
    def operation_slots(
        cls, variable: str, operation: StatisticOperation
    ) -> OperationSlots:
        return OperationSlots(
            cls.output(variable, operation.spelling),
            cls.aux(variable, operation.spelling) if operation.stores_index else None,
            cls.sample_weight(variable)
            if operation.inner is None and operation.outer is Reduction.MEAN
            else None,
        )

    @classmethod
    def inner_slots(cls, variable: str, inner: Reduction) -> InnerSlots:
        return InnerSlots(
            cls.inner_state(variable, inner),
            cls.inner_weight(variable, inner) if inner is Reduction.MEAN else None,
        )

    def owned_by(self, owners: Iterable[str]) -> tuple[str, ...]:
        """Names of the slots of ``owners``, in allocation order."""

        owners = frozenset(owners)
        return tuple(name for name, slot in self.slots.items() if slot.owner in owners)


def variable_slots(
    variable: str,
    operations: Iterable[StatisticOperation],
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[StorageSlot, ...]:
    """Slots of one statistic variable of ``shape`` and value ``dtype``."""

    slots: dict[str, StorageSlot] = {}

    def add(name, slot_shape, slot_dtype, initialization, output=False) -> None:
        if name not in slots:
            slots[name] = StorageSlot(
                name, variable, slot_shape, slot_dtype, initialization, output
            )

    zero = StorageInitialization.ZERO
    for operation in operations:
        operation_shape = shape + (operation.k,) if operation.k > 1 else shape
        initialization = storage_initialization(operation.outer)
        names = StoragePlan.operation_slots(variable, operation)
        if operation.stores_index:
            add(names.output, operation_shape, INDEX_DTYPE, zero, output=True)
            add(names.aux, operation_shape, dtype, initialization)
        else:
            add(names.output, operation_shape, dtype, initialization, output=True)
        if names.sample_weight is not None:
            add(names.sample_weight, shape, dtype, zero)
        if operation.inner is None:
            continue
        # A ``last`` inner state records the last sampled value for a window
        # closed by a step that collects no output.
        inner = StoragePlan.inner_slots(variable, operation.inner)
        add(inner.state, shape, dtype, storage_initialization(operation.inner))
        if inner.weight is not None:
            add(inner.weight, shape, dtype, zero)
    return tuple(slots.values())


def build_storage_plan(
    program: StatisticsProgram,
    layouts: Mapping[str, StatisticsVariableLayout],
    variables: Iterable[str],
    ensemble_size: int,
) -> StoragePlan:
    """Scatter buffers of every scatter source, then each variable's slots."""

    slots: dict[str, StorageSlot] = {}
    zero = StorageInitialization.ZERO
    for name, source in program.sources.items():
        if not isinstance(source, ScatterSource):
            continue
        layout = layouts[name]
        extent = layout.scatter_extent
        shape = (ensemble_size, extent) if layout.batched else (extent,)
        buffer = StoragePlan.scatter_buffer(name)
        slots[buffer] = StorageSlot(buffer, name, shape, layout.dtype, zero, False)
        if source.reduction is Reduction.MEAN:
            count = StoragePlan.scatter_count(name)
            slots[count] = StorageSlot(count, name, shape, COUNT_DTYPE, zero, False)
    for variable in variables:
        layout = layouts[variable]
        for slot in variable_slots(
            variable,
            program.operations[variable],
            layout.actual_shape,
            layout.dtype,
        ):
            slots[slot.name] = slot
    return StoragePlan(MappingProxyType(slots))

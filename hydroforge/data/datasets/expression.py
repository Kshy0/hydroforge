"""Lazy arithmetic expressions over compatible forcing datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
import torch

from hydroforge.contracts.temporal import DatasetTemporalContract
from hydroforge.data.datasets.base import _close_dataset_tree
from hydroforge.data.datasets.gridded import GriddedDataset


_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "mul": lambda left, right: left * right,
    "div": lambda left, right: left / right,
}

_REFERENCE_CONFIGURATION = frozenset({
    "start_date", "end_date", "time_interval", "chunk_len", "out_dtype",
    "spin_up_cycles", "spin_up_start_date", "spin_up_end_date", "calendar",
    "clip_negative", "model_step", "upsampling", "reuse_count",
})
_NO_SPATIAL_SELECTION = object()


@dataclass(frozen=True, slots=True)
class _SpatialIdentity:
    kind: str
    coordinates: tuple[np.ndarray, np.ndarray]


def _spatial_identity(dataset: Any, *, position: int) -> _SpatialIdentity | None:
    if isinstance(dataset, DatasetExpression):
        dataset._assert_spatial_selection_current()
        return dataset._expression_spatial_identity
    coordinate_reader = getattr(dataset, "get_coordinates", None)
    if coordinate_reader is None:
        return None
    if not callable(coordinate_reader):
        raise TypeError(
            f"dataset operand {position} get_coordinates must be callable"
        )
    coordinates = coordinate_reader()
    if not isinstance(coordinates, tuple) or len(coordinates) != 2:
        raise TypeError(
            f"dataset operand {position} get_coordinates() must return two "
            "coordinate arrays"
        )
    arrays = tuple(np.asarray(value) for value in coordinates)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(
            f"dataset operand {position} spatial coordinates must be "
            "one-dimensional"
        )
    is_gridded = isinstance(dataset, GriddedDataset) or bool(
        getattr(dataset, "_gridded", False)
    )
    selected = getattr(dataset, "_local_indices", None)
    if selected is not None:
        indices = np.asarray(selected)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError(
                f"dataset operand {position} spatial selection must be a "
                "one-dimensional integer array"
            )
        indices = indices.astype(np.int64, copy=False)
        if is_gridded:
            longitude, latitude = arrays
            total = longitude.size * latitude.size
            if np.any((indices < 0) | (indices >= total)):
                raise ValueError(
                    f"dataset operand {position} spatial selection contains "
                    "an out-of-range grid index"
                )
            x = indices % longitude.size
            y = indices // longitude.size
            arrays = (longitude[x], latitude[y])
            kind = "selected_grid"
        else:
            if arrays[0].shape != arrays[1].shape:
                raise ValueError(
                    f"dataset operand {position} point coordinates must have "
                    "matching shapes"
                )
            if np.any((indices < 0) | (indices >= arrays[0].size)):
                raise ValueError(
                    f"dataset operand {position} spatial selection contains "
                    "an out-of-range point index"
                )
            identifiers = arrays[0][indices]
            arrays = (
                identifiers,
                np.arange(identifiers.size, dtype=np.int64),
            )
            kind = "point"
    else:
        kind = "grid" if is_gridded else "point"
        if not is_gridded:
            if arrays[0].shape != arrays[1].shape:
                raise ValueError(
                    f"dataset operand {position} point coordinates must have "
                    "matching shapes"
                )
            arrays = (
                arrays[0],
                np.arange(arrays[0].size, dtype=np.int64),
            )
    return _SpatialIdentity(
        kind=kind,
        coordinates=arrays,
    )


def _same_coordinate_members(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> bool:
    for left_axis, right_axis in zip(left, right, strict=True):
        if left_axis.shape != right_axis.shape:
            return False
        try:
            if not np.array_equal(np.sort(left_axis), np.sort(right_axis)):
                return False
        except TypeError:
            return False
    return True


class DatasetExpression(torch.utils.data.Dataset):
    """One immutable lazy arithmetic tree with an explicit reference dataset."""

    def __init__(self, left: Any, operation: str, right: Any) -> None:
        if operation not in _OPERATIONS:
            raise ValueError(f"unknown dataset operation {operation!r}")
        datasets = tuple(
            value for value in (left, right) if not isinstance(value, Number)
        )
        if not datasets:
            raise TypeError("a dataset expression requires at least one dataset")
        for position, dataset in enumerate(datasets):
            missing = tuple(
                name for name in (
                    "temporal_contract", "chunk_plan", "simulation_schedule",
                )
                if not hasattr(dataset, name)
            )
            if missing:
                raise TypeError(
                    f"dataset operand {position} is missing forcing contract "
                    f"fields {missing}"
                )
        self.left = left
        self.right = right
        self.operation = operation
        self.reference = datasets[0]
        self._temporal_contract = DatasetTemporalContract.combine({
            f"operand_{index}": dataset.temporal_contract
            for index, dataset in enumerate(datasets)
        })
        self._chunk_plan = self.reference.chunk_plan
        self._simulation_schedule = self.reference.simulation_schedule
        self._expression_spatial_identity = _spatial_identity(
            self.reference, position=0,
        )
        self.supports_time_aggregation = bool(
            getattr(self.reference, "supports_time_aggregation", False)
        )
        for position, dataset in enumerate(datasets[1:], start=1):
            self._validate_compatible(dataset, position)
        self._expression_datasets = datasets
        self._spatial_selection_snapshot = tuple(
            self._selection_handle(dataset) for dataset in datasets
        )

    def __getattr__(self, name: str) -> Any:
        if name in _REFERENCE_CONFIGURATION:
            return getattr(self.reference, name)
        raise AttributeError(name)

    @staticmethod
    def _selection_handle(dataset: Any) -> Any:
        if isinstance(dataset, DatasetExpression):
            dataset._assert_spatial_selection_current()
            return dataset
        return getattr(dataset, "_local_indices", _NO_SPATIAL_SELECTION)

    def _assert_spatial_selection_current(self) -> None:
        """Reject evaluation after any operand installs a new local mapping."""

        for position, (dataset, expected) in enumerate(zip(
            self._expression_datasets,
            self._spatial_selection_snapshot,
            strict=True,
        )):
            if isinstance(dataset, DatasetExpression):
                dataset._assert_spatial_selection_current()
                continue
            current = getattr(dataset, "_local_indices", _NO_SPATIAL_SELECTION)
            if current is not expected:
                raise RuntimeError(
                    f"dataset operand {position} spatial selection changed "
                    "after expression construction; rebuild the expression "
                    "after installing all local mappings"
                )

    def _validate_compatible(self, other: Any, position: int) -> None:
        if (
            other.simulation_schedule.cadence
            != self._simulation_schedule.cadence
        ):
            raise ValueError(
                f"dataset operand {position} has a different model cadence"
            )
        if other.chunk_plan.chunk_len != self._chunk_plan.chunk_len:
            raise ValueError(
                f"dataset operand {position} has a different chunk length"
            )
        spatial_identity = _spatial_identity(other, position=position)
        reference_identity = self._expression_spatial_identity
        if spatial_identity is None and reference_identity is None:
            return
        if (
            spatial_identity is None
            or reference_identity is None
            or spatial_identity.kind != reference_identity.kind
        ):
            raise ValueError(
                f"dataset operand {position} uses a different spatial domain"
            )
        if all(
            np.array_equal(reference_axis, other_axis)
            for reference_axis, other_axis in zip(
                reference_identity.coordinates,
                spatial_identity.coordinates,
                strict=True,
            )
        ):
            return
        if _same_coordinate_members(
            reference_identity.coordinates, spatial_identity.coordinates,
        ):
            raise ValueError(
                f"dataset operand {position} uses a different spatial "
                "coordinate order"
            )
        raise ValueError(
            f"dataset operand {position} uses a different spatial domain"
        )

    @staticmethod
    def _value(operand: Any, index: int) -> Any:
        return operand if isinstance(operand, Number) else operand[index]

    def __getitem__(self, index: int):
        self._assert_spatial_selection_current()
        return _OPERATIONS[self.operation](
            self._value(self.left, index), self._value(self.right, index),
        )

    def __len__(self) -> int:
        return len(self.reference)

    def get_data(self, current_time: Any, chunk_len: int):
        self._assert_spatial_selection_current()

        def read(operand: Any):
            return (
                operand
                if isinstance(operand, Number)
                else operand.get_data(current_time, chunk_len)
            )
        return _OPERATIONS[self.operation](read(self.left), read(self.right))

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.reference.get_coordinates()

    @property
    def data_size(self) -> int:
        return self.reference.data_size

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.reference.grid_shape

    @property
    def temporal_contract(self):
        return self._temporal_contract

    @property
    def chunk_plan(self):
        return self._chunk_plan

    @property
    def simulation_schedule(self):
        return self._simulation_schedule

    def close(self) -> None:
        _close_dataset_tree(self, scope="dataset expression resources")

    def _close_children(self) -> tuple[object, ...]:
        return tuple(
            operand for operand in (self.left, self.right)
            if not isinstance(operand, Number)
        )

    def _combine(self, other: Any, operation: str, *, reverse: bool = False):
        left, right = (other, self) if reverse else (self, other)
        return DatasetExpression(left, operation, right)

    def __add__(self, other):
        return self._combine(other, "add")

    def __radd__(self, other):
        return self._combine(other, "add", reverse=True)

    def __sub__(self, other):
        return self._combine(other, "sub")

    def __rsub__(self, other):
        return self._combine(other, "sub", reverse=True)

    def __mul__(self, other):
        return self._combine(other, "mul")

    def __rmul__(self, other):
        return self._combine(other, "mul", reverse=True)

    def __truediv__(self, other):
        return self._combine(other, "div")

    def __rtruediv__(self, other):
        return self._combine(other, "div", reverse=True)

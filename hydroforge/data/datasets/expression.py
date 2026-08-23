"""Lazy arithmetic expressions over compatible forcing datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import PrivateAttr, field_validator, model_validator

from hydroforge.contracts.temporal import _combine_temporal_domains_trusted
from hydroforge.data.datasets.base import (
    AbstractDataset,
    _close_dataset_tree,
    _validated_dataset_index,
)
from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.gridded import GriddedDataset
from hydroforge.data.numeric import (
    canonical_floating_array,
    exact_numeric_array_equal,
)


_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "mul": lambda left, right: left * right,
    "div": lambda left, right: left / right,
}

_DatasetOperand = Any


def _is_scalar(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _finalize_expression_array(
    value: Any,
    *,
    out_dtype: str,
    label: str,
) -> np.ndarray:
    return canonical_floating_array(
        value,
        dtype=out_dtype,
        label=label,
    )


def _canonical_expression_scalar(
    value: Any,
    *,
    out_dtype: str,
    label: str,
) -> float:
    canonical = canonical_floating_array(
        value,
        dtype=out_dtype,
        label=label,
    )
    return float(canonical.item())


def _evaluate_expression(
    operation: str,
    left: Any,
    right: Any,
    *,
    left_is_scalar: bool,
    right_is_scalar: bool,
    out_dtype: str,
) -> Any:
    left_mapping = isinstance(left, Mapping)
    right_mapping = isinstance(right, Mapping)
    if left_mapping or right_mapping:
        if left_mapping and right_mapping:
            if set(left) != set(right):
                raise ValueError(
                    "dataset expression mappings must have identical variable names"
                )
            return {
                name: _evaluate_expression(
                    operation,
                    left[name],
                    right[name],
                    left_is_scalar=False,
                    right_is_scalar=False,
                    out_dtype=out_dtype,
                )
                for name in left
            }
        if left_mapping and right_is_scalar:
            return {
                name: _evaluate_expression(
                    operation,
                    block,
                    right,
                    left_is_scalar=False,
                    right_is_scalar=True,
                    out_dtype=out_dtype,
                )
                for name, block in left.items()
            }
        if right_mapping and left_is_scalar:
            return {
                name: _evaluate_expression(
                    operation,
                    left,
                    block,
                    left_is_scalar=True,
                    right_is_scalar=False,
                    out_dtype=out_dtype,
                )
                for name, block in right.items()
            }
        raise TypeError(
            "dataset expression operands must return matching mappings or "
            "numeric arrays"
        )

    if not left_is_scalar and not isinstance(left, (np.ndarray, np.number)):
        raise TypeError("left dataset expression operand must return a numeric array")
    if not right_is_scalar and not isinstance(right, (np.ndarray, np.number)):
        raise TypeError("right dataset expression operand must return a numeric array")
    if np.ma.isMaskedArray(left) or np.ma.isMaskedArray(right):
        raise TypeError("dataset expression operands must not be masked arrays")
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        not left_is_scalar
        and not right_is_scalar
        and left_array.shape != right_array.shape
    ):
        raise ValueError(
            "dataset expression operands must return identical shapes; got "
            f"{left_array.shape} and {right_array.shape}"
        )
    # Evaluate in the dataset's declared output dtype.  Performing arithmetic
    # first would inherit an accidental integer or lower-precision operand
    # dtype, so overflow/underflow could occur before the result validator ever
    # sees the intended value.
    left_array = _finalize_expression_array(
        left_array,
        out_dtype=out_dtype,
        label="left dataset expression operand",
    )
    right_array = _finalize_expression_array(
        right_array,
        out_dtype=out_dtype,
        label="right dataset expression operand",
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = _OPERATIONS[operation](left_array, right_array)
    if np.isinf(result).any() and not (operation == "div" and np.any(right_array == 0)):
        if out_dtype == "float32":
            raise OverflowError(
                "dataset expression result contains values outside float32 range"
            )
        raise OverflowError("dataset expression result overflowed float64")
    return _finalize_expression_array(
        result,
        out_dtype=out_dtype,
        label="dataset expression result",
    )


@dataclass(frozen=True, slots=True)
class _SpatialIdentity:
    kind: str
    coordinates: tuple[np.ndarray, np.ndarray]


def _spatial_identity(dataset: Any, *, position: int) -> _SpatialIdentity | None:
    if isinstance(dataset, DatasetExpression):
        return dataset._expression_spatial_identity
    coordinate_reader = getattr(dataset, "get_coordinates", None)
    if coordinate_reader is None:
        return None
    if not callable(coordinate_reader):
        raise ValueError(f"dataset operand {position} get_coordinates must be callable")
    coordinates = coordinate_reader()
    if not isinstance(coordinates, tuple) or len(coordinates) != 2:
        raise ValueError(
            f"dataset operand {position} get_coordinates() must return two "
            "coordinate arrays"
        )
    if any(np.ma.isMaskedArray(value) for value in coordinates):
        raise ValueError(
            f"dataset operand {position} spatial coordinates must not be masked"
        )
    arrays = tuple(np.asarray(value) for value in coordinates)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(
            f"dataset operand {position} spatial coordinates must be one-dimensional"
        )
    gridded_marker = getattr(dataset, "_gridded", False)
    if type(gridded_marker) is not bool:
        raise ValueError(f"dataset operand {position} _gridded must be an exact bool")
    is_gridded = isinstance(dataset, GriddedDataset) or gridded_marker
    selected = getattr(dataset, "local_indices", None)
    if selected is not None:
        if np.ma.isMaskedArray(selected):
            raise ValueError(
                f"dataset operand {position} spatial selection must not be masked"
            )
        indices = np.asarray(selected)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(
                f"dataset operand {position} spatial selection must be a "
                "one-dimensional integer array"
            )
        if (
            indices.dtype.kind == "u"
            and indices.size
            and np.any(indices > np.iinfo(np.int64).max)
        ):
            raise ValueError(
                f"dataset operand {position} spatial selection exceeds int64"
            )
        indices = np.array(indices, dtype=np.int64, order="C", copy=True)
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
    frozen = tuple(np.array(axis, order="C", copy=True) for axis in arrays)
    for axis in frozen:
        axis.setflags(write=False)
    return _SpatialIdentity(
        kind=kind,
        coordinates=frozen,
    )


def _same_coordinate_members(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> bool:
    for left_axis, right_axis in zip(left, right, strict=True):
        if left_axis.shape != right_axis.shape:
            return False
        try:
            if not exact_numeric_array_equal(
                np.sort(left_axis),
                np.sort(right_axis),
            ):
                return False
        except TypeError:
            return False
    return True


class DatasetExpression(AbstractDataset):
    """One immutable lazy arithmetic tree with an explicit reference dataset."""

    left: Any
    operation: Literal["add", "sub", "mul", "div"]
    right: Any

    _reference: AbstractDataset | DatasetExpression = PrivateAttr()
    _out_dtype: str = PrivateAttr()
    _left_operand: Any = PrivateAttr()
    _right_operand: Any = PrivateAttr()
    _expression_spatial_identity: _SpatialIdentity | None = PrivateAttr()
    _supports_time_aggregation: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _derive_dataset_identity(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        reference = next(
            (
                operand
                for operand in (value.get("left"), value.get("right"))
                if not _is_scalar(operand)
                and isinstance(operand, (AbstractDataset, DatasetExpression))
            ),
            None,
        )
        if reference is None:
            return value
        payload = dict(value)
        derived = reference._dataset_identity_arguments()
        for name, expected in derived.items():
            if name in payload and payload[name] != expected:
                raise ValueError(f"dataset expression {name} must match its operands")
            payload[name] = expected
        return payload

    @field_validator("operation")
    @classmethod
    def _validate_operation(cls, operation: str) -> str:
        if type(operation) is not str or operation not in _OPERATIONS:
            raise ValueError(f"unknown dataset operation {operation!r}")
        return operation

    @model_validator(mode="after")
    def _validate_expression(self):
        left = self.left
        right = self.right
        invalid_scalars = [
            value
            for value in (left, right)
            if isinstance(value, (bool, int, float, complex, np.number))
            and not _is_scalar(value)
        ]
        if invalid_scalars:
            raise ValueError(
                "dataset expression scalars must be finite real numbers; "
                f"got {invalid_scalars!r}"
            )
        left_is_scalar = _is_scalar(left)
        right_is_scalar = _is_scalar(right)
        datasets = tuple(
            value
            for value, is_scalar in (
                (left, left_is_scalar),
                (right, right_is_scalar),
            )
            if not is_scalar
        )
        if not datasets:
            raise ValueError("a dataset expression requires at least one dataset")
        for position, dataset in enumerate(datasets):
            if not isinstance(dataset, (AbstractDataset, DatasetExpression)):
                raise ValueError(
                    f"dataset operand {position} must be an AbstractDataset "
                    "or DatasetExpression"
                )
        reference = datasets[0]
        out_dtype = getattr(reference, "out_dtype", None)
        if type(out_dtype) is not str or out_dtype not in {"float32", "float64"}:
            raise ValueError(
                "reference dataset out_dtype must be 'float32' or 'float64'"
            )
        _canonical_expression_scalar(
            left,
            out_dtype=out_dtype,
            label="left dataset expression scalar",
        ) if left_is_scalar else left
        _canonical_expression_scalar(
            right,
            out_dtype=out_dtype,
            label="right dataset expression scalar",
        ) if right_is_scalar else right
        _combine_temporal_domains_trusted(
            {
                f"operand_{index}": dataset._temporal_domain
                for index, dataset in enumerate(datasets)
            }
        )
        reference_identity = _spatial_identity(reference, position=0)
        supports_time_aggregation = getattr(
            reference,
            "supports_time_aggregation",
            False,
        )
        if type(supports_time_aggregation) is not bool:
            raise ValueError(
                "reference dataset supports_time_aggregation must be an exact bool"
            )
        for position, dataset in enumerate(datasets[1:], start=1):
            self._validate_compatible(
                dataset,
                position,
                reference=reference,
                out_dtype=out_dtype,
                schedule=reference.simulation_schedule,
                chunk_plan=reference.chunk_plan,
                reference_identity=reference_identity,
            )
        return self

    @model_validator(mode="after")
    def _compile_expression(self):
        left_is_scalar = _is_scalar(self.left)
        right_is_scalar = _is_scalar(self.right)
        datasets = tuple(
            value
            for value, is_scalar in (
                (self.left, left_is_scalar),
                (self.right, right_is_scalar),
            )
            if not is_scalar
        )
        reference = datasets[0]
        out_dtype = reference.out_dtype
        self._reference = reference
        self._out_dtype = out_dtype
        self._left_operand = (
            _canonical_expression_scalar(
                self.left,
                out_dtype=out_dtype,
                label="left dataset expression scalar",
            )
            if left_is_scalar
            else self.left
        )
        self._right_operand = (
            _canonical_expression_scalar(
                self.right,
                out_dtype=out_dtype,
                label="right dataset expression scalar",
            )
            if right_is_scalar
            else self.right
        )
        temporal_domain = _combine_temporal_domains_trusted(
            {
                f"operand_{index}": dataset._temporal_domain
                for index, dataset in enumerate(datasets)
            }
        )
        if temporal_domain != self._temporal_domain:
            raise ValueError("dataset expression timeline must match its operands")
        self._expression_spatial_identity = _spatial_identity(
            reference,
            position=0,
        )
        self._supports_time_aggregation = reference.supports_time_aggregation
        return self

    @property
    def reference(self) -> AbstractDataset | DatasetExpression:
        return self._reference

    @property
    def supports_time_aggregation(self) -> bool:
        return self._supports_time_aggregation

    @staticmethod
    def _validate_compatible(
        other: Any,
        position: int,
        *,
        reference: AbstractDataset | DatasetExpression,
        out_dtype: str,
        schedule: Any,
        chunk_plan: Any,
        reference_identity: _SpatialIdentity | None,
    ) -> None:
        other_dtype = getattr(other, "out_dtype", None)
        if other_dtype != out_dtype:
            raise ValueError(
                f"dataset operand {position} has out_dtype {other_dtype!r}, "
                f"expected {out_dtype!r}"
            )
        other_contract = (
            other.reference._temporal_domain
            if isinstance(other, DatasetExpression)
            else other._temporal_domain
        )
        if other_contract != reference._temporal_domain:
            raise ValueError(
                f"dataset operand {position} has a different temporal contract"
            )
        if other.simulation_schedule.cadence != schedule.cadence:
            raise ValueError(
                f"dataset operand {position} has a different model cadence"
            )
        if other.chunk_plan.chunk_len != chunk_plan.chunk_len:
            raise ValueError(f"dataset operand {position} has a different chunk length")
        spatial_identity = _spatial_identity(other, position=position)
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
            exact_numeric_array_equal(reference_axis, other_axis)
            for reference_axis, other_axis in zip(
                reference_identity.coordinates,
                spatial_identity.coordinates,
                strict=True,
            )
        ):
            return
        if _same_coordinate_members(
            reference_identity.coordinates,
            spatial_identity.coordinates,
        ):
            raise ValueError(
                f"dataset operand {position} uses a different spatial coordinate order"
            )
        raise ValueError(f"dataset operand {position} uses a different spatial domain")

    @staticmethod
    def _value(operand: Any, chunk: SourceChunk) -> Any:
        if _is_scalar(operand):
            return operand
        operand_chunk = operand.chunk_plan._at_trusted(chunk.index)
        return operand._get_chunk_trusted(operand_chunk)

    def __getitem__(self, index: int):
        chunk = self._chunk_plan._at_trusted(
            _validated_dataset_index(self, index),
        )
        return self._get_chunk_trusted(chunk)

    def _get_chunk_trusted(self, chunk: SourceChunk):
        """Evaluate one framework-produced consumer request."""

        return _evaluate_expression(
            self.operation,
            self._value(self._left_operand, chunk),
            self._value(self._right_operand, chunk),
            left_is_scalar=_is_scalar(self._left_operand),
            right_is_scalar=_is_scalar(self._right_operand),
            out_dtype=self._out_dtype,
        )

    def __len__(self) -> int:
        return len(self.reference)

    def _read_chunk(self, chunk: SourceChunk):
        """Evaluate one framework-produced raw source request."""

        def read(operand: Any):
            return (
                operand
                if _is_scalar(operand)
                else operand._read_chunk_trusted(
                    operand.chunk_plan._at_trusted(chunk.index)
                )
            )

        return _evaluate_expression(
            self.operation,
            read(self._left_operand),
            read(self._right_operand),
            left_is_scalar=_is_scalar(self._left_operand),
            right_is_scalar=_is_scalar(self._right_operand),
            out_dtype=self._out_dtype,
        )

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.reference.get_coordinates()

    @property
    def data_size(self) -> int:
        return self.reference.data_size

    @property
    def _grid_shape(self) -> tuple[int, int]:
        return self.reference._grid_shape

    def close(self) -> None:
        _close_dataset_tree(self, scope="dataset expression resources")

    def _close_children(self) -> tuple[object, ...]:
        return tuple(
            operand
            for operand in (self._left_operand, self._right_operand)
            if not _is_scalar(operand)
        )

    def _combine(self, other: Any, operation: str, *, reverse: bool = False):
        left, right = (other, self) if reverse else (self, other)
        return DatasetExpression(
            left=left,
            operation=operation,
            right=right,
        )

    def __add__(self, other: _DatasetOperand):
        return self._combine(other, "add")

    def __radd__(self, other: _DatasetOperand):
        return self._combine(other, "add", reverse=True)

    def __sub__(self, other: _DatasetOperand):
        return self._combine(other, "sub")

    def __rsub__(self, other: _DatasetOperand):
        return self._combine(other, "sub", reverse=True)

    def __mul__(self, other: _DatasetOperand):
        return self._combine(other, "mul")

    def __rmul__(self, other: _DatasetOperand):
        return self._combine(other, "mul", reverse=True)

    def __truediv__(self, other: _DatasetOperand):
        return self._combine(other, "div")

    def __rtruediv__(self, other: _DatasetOperand):
        return self._combine(other, "div", reverse=True)

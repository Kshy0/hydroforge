"""Lazy arithmetic expressions over compatible forcing datasets."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Number
from typing import Any

import numpy as np
import torch

from hydroforge.data.datasets.base import _close_dataset_tree


_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "mul": lambda left, right: left * right,
    "div": lambda left, right: left / right,
}

_DATASET_CONTRACT = (
    "start_date", "end_date", "time_interval", "chunk_len", "out_dtype",
    "spin_up_cycles", "spin_up_start_date", "spin_up_end_date", "calendar",
    "clip_negative",
)


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
                name for name in (*_DATASET_CONTRACT, "temporal_contract")
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
        for attribute in _DATASET_CONTRACT:
            setattr(self, attribute, getattr(self.reference, attribute))
        self.supports_time_aggregation = bool(
            getattr(self.reference, "supports_time_aggregation", False)
        )
        for position, dataset in enumerate(datasets[1:], start=1):
            self._validate_compatible(dataset, position)

    def _validate_compatible(self, other: Any, position: int) -> None:
        reference = self.reference
        for attribute in _DATASET_CONTRACT:
            if getattr(other, attribute) != getattr(reference, attribute):
                raise ValueError(
                    f"dataset operand {position} has different {attribute}"
                )
        if len(other) != len(reference):
            raise ValueError(f"dataset operand {position} has different length")

    @staticmethod
    def _value(operand: Any, index: int) -> Any:
        return operand if isinstance(operand, Number) else operand[index]

    def __getitem__(self, index: int):
        return _OPERATIONS[self.operation](
            self._value(self.left, index), self._value(self.right, index),
        )

    def __len__(self) -> int:
        return len(self.reference)

    def get_data(self, current_time: Any, chunk_len: int):
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
    def total_steps(self) -> int:
        return self.reference.total_steps

    def time_iter(self):
        return self.reference.time_iter()

    def temporal_contract(self):
        return self.reference.temporal_contract()

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

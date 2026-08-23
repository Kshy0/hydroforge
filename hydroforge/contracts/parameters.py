"""Immutable public declarations for scheduled parameter changes."""

from __future__ import annotations

from datetime import datetime
import math
from typing import ClassVar, Self

import cftime
import torch
from pydantic import field_serializer, model_validator

from hydroforge.contracts.temporal import _require_date, date_calendar
from hydroforge.contracts.validation import HydroForgeModel


ParameterScalar = bool | int | float
ParameterValue = ParameterScalar | torch.Tensor


def _owned_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise ValueError(f"{label} must contain only finite values")
    return value.detach().clone(memory_format=torch.preserve_format)


class ParameterChange(HydroForgeModel):
    """One complete scheduled SET or ADD parameter declaration."""

    variable: str
    start: datetime | cftime.datetime
    active_steps: int = 1
    delta: ParameterValue = 0.0
    target_value: ParameterValue | None = None
    target_ids: tuple[int, ...] | torch.Tensor | None = None
    target_id_field: str | None = None

    _PUBLIC_TENSOR_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "delta", "target_value", "target_ids",
    })

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if (
            name in ParameterChange._PUBLIC_TENSOR_FIELDS
            and isinstance(value, torch.Tensor)
        ):
            return value.detach().clone(memory_format=torch.preserve_format)
        return value

    def __iter__(self):
        for name, value in super().__iter__():
            if (
                name in self._PUBLIC_TENSOR_FIELDS
                and isinstance(value, torch.Tensor)
            ):
                value = value.detach().clone(
                    memory_format=torch.preserve_format,
                )
            yield name, value

    @field_serializer("delta", "target_value", "target_ids")
    def _serialize_owned_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone(memory_format=torch.preserve_format)
        return value

    def _trusted_value(self, name: str) -> ParameterValue | tuple[int, ...] | None:
        """Return owned declaration storage to the semantic compiler only."""

        return object.__getattribute__(self, name)

    @model_validator(mode="after")
    def _validate_change(self) -> Self:
        names = self.variable.split(".")
        if not names or any(not name.isidentifier() for name in names):
            raise ValueError(
                "parameter change variable must be a dotted attribute path"
            )
        _require_date(self.start, label="parameter change start")
        date_calendar(self.start)
        if self.active_steps < 1:
            raise ValueError("parameter change active_steps must be positive")
        target_value = self._trusted_value("target_value")
        delta = self._trusted_value("delta")
        for label, value in (
            ("parameter delta", delta),
            ("parameter target_value", target_value),
        ):
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if target_value is not None and self.active_steps != 1:
            raise ValueError(
                "SET parameter changes are one-shot and require active_steps=1"
            )
        if target_value is not None and not (
            type(delta) is float and delta == 0.0
        ):
            raise ValueError(
                "SET parameter changes cannot also define delta"
            )
        if self.target_id_field is not None:
            names = self.target_id_field.split(".")
            if any(not name.isidentifier() for name in names):
                raise ValueError(
                    "parameter target_id_field must be a dotted attribute path"
                )
        if isinstance(delta, torch.Tensor):
            object.__setattr__(
                self,
                "delta",
                _owned_tensor(delta, label="parameter delta"),
            )
        if isinstance(target_value, torch.Tensor):
            object.__setattr__(
                self,
                "target_value",
                _owned_tensor(
                    target_value, label="parameter target_value",
                ),
            )
        target_ids = self._trusted_value("target_ids")
        if isinstance(target_ids, tuple):
            if not target_ids:
                raise ValueError("parameter target_ids must not be empty")
            if len(target_ids) != len(set(target_ids)):
                raise ValueError("parameter target_ids must be unique")
        elif isinstance(target_ids, torch.Tensor):
            if target_ids.ndim != 1:
                raise ValueError("parameter target_ids must be one-dimensional")
            if target_ids.dtype not in {
                torch.int8, torch.uint8, torch.int16, torch.uint16,
                torch.int32, torch.uint32, torch.int64,
            }:
                raise ValueError("parameter target_ids must contain integers")
            if torch.unique(target_ids).numel() != target_ids.numel():
                raise ValueError("parameter target_ids must be unique")
            object.__setattr__(
                self,
                "target_ids",
                _owned_tensor(target_ids, label="parameter target_ids"),
            )
        return self

    @property
    def is_set(self) -> bool:
        return self._trusted_value("target_value") is not None


__all__ = ["ParameterChange", "ParameterScalar", "ParameterValue"]

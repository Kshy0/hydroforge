"""Immutable public declarations for scheduled parameter changes."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Self

import cftime
import torch
from pydantic import (
    Field,
    FiniteFloat,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from hydroforge.contracts.naming import DottedPath
from hydroforge.contracts.temporal import _require_date, date_calendar
from hydroforge.contracts.validation import HydroForgeModel

ParameterScalar = bool | int | FiniteFloat
ParameterValue = ParameterScalar | torch.Tensor


def _owned_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.layout is not torch.strided:
        raise ValueError(f"{label} must use torch.strided layout")
    if value.is_floating_point():
        if value.device.type == "meta":
            raise ValueError(
                f"{label} must contain materialized values, not meta tensors"
            )
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{label} must contain only finite values")
    return value.detach().clone(memory_format=torch.preserve_format)


class ParameterChange(HydroForgeModel):
    """One complete scheduled SET or ADD parameter declaration."""

    variable: DottedPath
    start: datetime | cftime.datetime
    active_steps: int = Field(default=1, ge=1)
    delta: ParameterValue = 0.0
    target_value: ParameterValue | None = None
    target_ids: tuple[int, ...] | torch.Tensor | None = None
    target_id_field: DottedPath | None = None

    _PUBLIC_TENSOR_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "delta",
            "target_value",
            "target_ids",
        }
    )

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if name in ParameterChange._PUBLIC_TENSOR_FIELDS and isinstance(
            value, torch.Tensor
        ):
            return value.detach().clone(memory_format=torch.preserve_format)
        return value

    def __iter__(self):
        for name, value in super().__iter__():
            if name in self._PUBLIC_TENSOR_FIELDS and isinstance(value, torch.Tensor):
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

    @field_validator("start")
    @classmethod
    def _validate_start(cls, value: datetime | cftime.datetime):
        _require_date(value, label="parameter change start")
        date_calendar(value)
        return value

    @field_validator("delta", "target_value")
    @classmethod
    def _own_update(cls, value: ParameterValue | None, info: ValidationInfo):
        if isinstance(value, torch.Tensor):
            return _owned_tensor(value, label=f"parameter {info.field_name}")
        return value

    @model_validator(mode="after")
    def _validate_change(self) -> Self:
        target_value = self._trusted_value("target_value")
        delta = self._trusted_value("delta")
        if target_value is not None and self.active_steps != 1:
            raise ValueError(
                "SET parameter changes are one-shot and require active_steps=1"
            )
        if target_value is not None and not (type(delta) is float and delta == 0.0):
            raise ValueError("SET parameter changes cannot also define delta")
        return self

    @field_validator("target_ids")
    @classmethod
    def _own_target_ids(cls, target_ids: tuple[int, ...] | torch.Tensor | None):
        if isinstance(target_ids, tuple):
            if not target_ids:
                raise ValueError("parameter target_ids must not be empty")
            if len(target_ids) != len(set(target_ids)):
                raise ValueError("parameter target_ids must be unique")
        elif isinstance(target_ids, torch.Tensor):
            if target_ids.layout is not torch.strided:
                raise ValueError("parameter target_ids must use torch.strided layout")
            if target_ids.device.type == "meta":
                raise ValueError(
                    "parameter target_ids must contain materialized values, not meta tensors"
                )
            if target_ids.ndim != 1:
                raise ValueError("parameter target_ids must be one-dimensional")
            if target_ids.dtype not in {
                torch.int8,
                torch.uint8,
                torch.int16,
                torch.uint16,
                torch.int32,
                torch.uint32,
                torch.int64,
            }:
                raise ValueError("parameter target_ids must contain integers")
            if target_ids.numel() == 0:
                raise ValueError("parameter target_ids must not be empty")
            if torch.unique(target_ids).numel() != target_ids.numel():
                raise ValueError("parameter target_ids must be unique")
            return target_ids.detach().clone(memory_format=torch.preserve_format)
        return target_ids

    @property
    def is_set(self) -> bool:
        return self._trusted_value("target_value") is not None


__all__ = ["ParameterChange", "ParameterScalar", "ParameterValue"]

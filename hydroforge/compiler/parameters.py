"""Construction-time compilation of scheduled parameter changes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import TYPE_CHECKING, Mapping

import cftime
import numpy as np
import torch

from hydroforge.contracts.fields import (
    ModuleFieldSchema,
    concrete_tensor_dtype,
)
from hydroforge.contracts.parameters import ParameterChange, ParameterValue
from hydroforge.data.distributed import _find_indices_in_trusted

if TYPE_CHECKING:
    from hydroforge.compiler.partition import _PartitionSemanticCompiler
    from hydroforge.model.model import AbstractModel


@dataclass(frozen=True, slots=True)
class _ParameterChangePlan:
    """One complete rank-local instruction compiled from public input."""

    variable_name: str
    module_name: str
    field_name: str
    start_time: datetime | cftime.datetime
    active_steps: int
    delta: ParameterValue
    target_value: ParameterValue | None
    target_ids: tuple[int, ...] | None
    target_id_field: str | None
    local_indices: tuple[int, ...] | None
    index_axis: int

    @property
    def is_set_value(self) -> bool:
        return self.target_value is not None


@dataclass(frozen=True, slots=True)
class _ResolvedParameterField:
    module_name: str
    schema: ModuleFieldSchema


class ParameterSemanticCompiler:
    """Resolve every parameter declaration before runtime materialization."""

    def __init__(
        self,
        model: AbstractModel,
        partition: _PartitionSemanticCompiler,
        *,
        input_axes: Mapping[str, int],
    ) -> None:
        self.model = model
        self.partition = partition
        self.input_axes = input_axes
        self._qualified, self._unqualified = self._field_index()

    def compile(
        self, changes: tuple[ParameterChange, ...],
    ) -> tuple[_ParameterChangePlan, ...]:
        plans = tuple(sorted(
            (self._compile_change(change) for change in changes),
            key=lambda item: item.start_time,
        ))
        self._validate_set_conflicts(plans)
        return plans

    def _field_index(self) -> tuple[
        dict[str, _ResolvedParameterField],
        dict[str, _ResolvedParameterField | None],
    ]:
        qualified: dict[str, _ResolvedParameterField] = {}
        unqualified: dict[str, _ResolvedParameterField | None] = {}
        schema = self.model._compiled_schema()
        for module_name in self.model.opened_modules:
            for field in schema.fields(module_name):
                tensor = field.tensor
                if tensor is None or not self.model._is_tensor_field_active(
                    module_name, field,
                ):
                    continue
                resolved = _ResolvedParameterField(module_name, field)
                qualified[f"{module_name}.{field.name}"] = resolved
                previous = unqualified.get(field.name)
                if previous is None and field.name not in unqualified:
                    unqualified[field.name] = resolved
                else:
                    unqualified[field.name] = None
        return qualified, unqualified

    def _resolve_field(
        self,
        name: str,
        *,
        owner_module: str | None = None,
        label: str,
    ) -> _ResolvedParameterField:
        if "." in name:
            resolved = self._qualified.get(name)
        else:
            resolved = (
                self._qualified.get(f"{owner_module}.{name}")
                if owner_module is not None else None
            )
            if resolved is None:
                resolved = self._unqualified.get(name)
        if resolved is None:
            raise ValueError(
                f"{label} {name!r} was not found unambiguously in an "
                "opened module"
            )
        return resolved

    def _compile_change(
        self, change: ParameterChange,
    ) -> _ParameterChangePlan:
        resolved = self._resolve_field(
            change.variable,
            label="parameter change variable",
        )
        field = resolved.schema
        tensor = field.tensor
        if tensor.category != "param":
            raise ValueError(
                f"parameter change variable {change.variable!r} declares "
                f"category={tensor.category!r}, expected 'param'"
            )
        if tensor.mode == "discard":
            raise ValueError(
                f"parameter change variable {change.variable!r} cannot use "
                "mode='discard'"
            )
        source_shape = self.model._input.get_var_shape(field.name)
        index_axis = self.input_axes.get(field.name)
        if index_axis is None:
            index_axis = self.partition.logical_axis(
                field.name, field, source_shape,
            )
        local_shape = list(source_shape)
        group = self.partition.variable_groups.get(field.name)
        local_rows: np.ndarray | None = None
        if group is not None:
            local_rows = self.partition.rank_indices(group)
            local_shape[index_axis] = int(local_rows.size)

        expected_dtype = concrete_tensor_dtype(
            tensor.dtype, self.model.dtype, self.model.mixed_precision,
        )
        expected_device = (
            torch.device("cpu")
            if tensor.mode == "cpu" else torch.device(self.model.device)
        )

        target_ids: tuple[int, ...] | None = None
        local_indices: tuple[int, ...] | None = None
        local_positions: tuple[int, ...] | None = None
        update_shape = tuple(local_shape)
        resolved_id_name: str | None = None
        if change._trusted_value("target_ids") is not None:
            (
                target_ids,
                local_indices,
                local_positions,
                resolved_id_name,
            ) = self._compile_target_ids(
                change,
                parameter=resolved,
                source_shape=source_shape,
                index_axis=index_axis,
                local_rows=local_rows,
            )
            requested_shape = list(local_shape)
            requested_shape[index_axis] = len(target_ids)
            update_shape = tuple(requested_shape)

        target_value = change._trusted_value("target_value")
        delta = change._trusted_value("delta")
        is_set = target_value is not None
        raw_value = target_value if is_set else delta
        value = self._validate_update_value(
            raw_value,
            expected_shape=update_shape,
            expected_dtype=expected_dtype,
            expected_device=expected_device,
            variable_name=change.variable,
            is_set=is_set,
        )
        if (
            isinstance(value, torch.Tensor)
            and value.ndim != 0
            and local_positions is not None
        ):
            positions = torch.tensor(
                local_positions,
                dtype=torch.int64,
                device=value.device,
            )
            value = value.index_select(index_axis, positions).contiguous()

        return _ParameterChangePlan(
            variable_name=change.variable,
            module_name=resolved.module_name,
            field_name=field.name,
            start_time=change.start,
            active_steps=change.active_steps,
            delta=delta if is_set else value,
            target_value=value if is_set else None,
            target_ids=target_ids,
            target_id_field=resolved_id_name,
            local_indices=local_indices,
            index_axis=index_axis,
        )

    def _compile_target_ids(
        self,
        change: ParameterChange,
        *,
        parameter: _ResolvedParameterField,
        source_shape: tuple[int, ...],
        index_axis: int,
        local_rows: np.ndarray | None,
    ) -> tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], str,
    ]:
        parameter_tensor = parameter.schema.tensor
        id_name = change.target_id_field or parameter_tensor.dim_coords
        if id_name is None:
            raise ValueError(
                f"parameter change variable {change.variable!r} needs "
                "target_id_field because it has no dim_coords"
            )
        resolved_id = self._resolve_field(
            id_name,
            owner_module=parameter.module_name,
            label="parameter target ID field",
        )
        id_field = resolved_id.schema
        id_tensor = id_field.tensor
        if not id_tensor.is_key:
            raise ValueError(
                f"parameter target ID field {id_name!r} must declare "
                "is_key=True"
            )
        if len(id_tensor.shape) != 1:
            raise ValueError(
                f"parameter target ID field {id_name!r} must be "
                "one-dimensional"
            )
        parameter_coordinate = (
            parameter_tensor.dim_coords.rsplit(".", 1)[-1]
            if parameter_tensor.dim_coords else None
        )
        id_coordinate = (
            id_field.name
            if id_tensor.is_coordinate
            else (
                id_tensor.dim_coords.rsplit(".", 1)[-1]
                if id_tensor.dim_coords else None
            )
        )
        if parameter_coordinate != id_coordinate:
            raise ValueError(
                f"parameter target ID field {id_name!r} is not aligned to "
                f"{change.variable!r} coordinate {parameter_coordinate!r}"
            )

        id_shape = self.model._input.get_var_shape(id_field.name)
        if id_shape != (source_shape[index_axis],):
            raise ValueError(
                f"parameter target ID field {id_name!r} shape {id_shape} is "
                f"not co-indexed with {change.variable!r} axis "
                f"length {source_shape[index_axis]}"
            )
        id_values_tensor = self.model._input[id_field.name]
        id_values = id_values_tensor.detach().cpu().numpy().reshape(-1)
        if np.unique(id_values).size != id_values.size:
            raise ValueError(
                f"parameter target ID field {id_name!r} contains duplicate IDs"
            )

        target_ids = self._canonical_target_ids(
            change._trusted_value("target_ids"),
            expected_dtype=id_values_tensor.dtype,
            variable_name=change.variable,
        )
        target_array = np.asarray(target_ids, dtype=id_values.dtype)
        global_indices = _find_indices_in_trusted(target_array, id_values)
        missing = global_indices < 0
        if np.any(missing):
            missing_ids = target_array[missing][:10].tolist()
            raise ValueError(
                f"target_ids for {change.variable!r} were not found in "
                f"{id_name!r}: {missing_ids}"
            )

        if local_rows is None:
            local_rows = np.arange(id_values.size, dtype=np.int64)
        local_by_global = {
            int(global_index): local_index
            for local_index, global_index in enumerate(local_rows.tolist())
        }
        local_positions: list[int] = []
        local_indices: list[int] = []
        for request_position, global_index in enumerate(global_indices.tolist()):
            local_index = local_by_global.get(int(global_index))
            if local_index is None:
                continue
            local_positions.append(request_position)
            local_indices.append(local_index)
        return (
            target_ids,
            tuple(local_indices),
            tuple(local_positions),
            f"{resolved_id.module_name}.{id_field.name}",
        )

    @staticmethod
    def _canonical_target_ids(
        values: tuple[int, ...] | torch.Tensor,
        *,
        expected_dtype: torch.dtype,
        variable_name: str,
    ) -> tuple[int, ...]:
        if isinstance(values, tuple):
            return values
        del expected_dtype, variable_name
        return tuple(int(value) for value in values.tolist())

    @staticmethod
    def _validate_update_value(
        value: ParameterValue,
        *,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
        expected_device: torch.device,
        variable_name: str,
        is_set: bool,
    ) -> ParameterValue:
        if isinstance(value, torch.Tensor):
            if expected_dtype is torch.bool and not is_set:
                raise ValueError(
                    f"boolean parameter {variable_name!r} supports SET only"
                )
            if value.layout is not torch.strided:
                raise ValueError(
                    f"parameter {variable_name!r} update tensor must use "
                    "torch.strided layout"
                )
            if not value.is_contiguous():
                raise ValueError(
                    f"parameter {variable_name!r} update tensor must be "
                    "contiguous"
                )
            if value.ndim != 0 and tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"parameter {variable_name!r} update tensor must be "
                    f"scalar or have shape {expected_shape}; got "
                    f"{tuple(value.shape)}"
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"parameter {variable_name!r} update tensor must use "
                    f"dtype {expected_dtype}; got {value.dtype}"
                )
            device_matches = bool(
                value.device.type == expected_device.type
                and (
                    value.device.index is None
                    or expected_device.index is None
                    or value.device.index == expected_device.index
                )
            )
            if not device_matches:
                raise ValueError(
                    f"parameter {variable_name!r} update tensor must be on "
                    f"device {expected_device}; got {value.device}"
                )
            return value.detach().clone(memory_format=torch.preserve_format)

        if expected_dtype is torch.bool:
            if not is_set or type(value) is not bool:
                raise ValueError(
                    f"boolean parameter {variable_name!r} requires an exact "
                    "bool SET value"
                )
            return value
        if expected_dtype.is_floating_point:
            if type(value) is not float:
                raise ValueError(
                    f"floating parameter {variable_name!r} update must be an "
                    "exact float or matching tensor"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"parameter {variable_name!r} update must be finite"
                )
            if abs(value) > torch.finfo(expected_dtype).max:
                raise ValueError(
                    f"parameter {variable_name!r} update is outside "
                    f"{expected_dtype} range"
                )
            encoded = torch.tensor(value, dtype=expected_dtype).item()
            if value != 0.0 and encoded == 0.0:
                raise ValueError(
                    f"parameter {variable_name!r} update underflows "
                    f"{expected_dtype} storage"
                )
            return value
        if expected_dtype in {
            torch.int8, torch.uint8, torch.int16, torch.uint16,
            torch.int32, torch.uint32, torch.int64,
        }:
            if type(value) is not int:
                raise ValueError(
                    f"integer parameter {variable_name!r} update must be an "
                    "exact int or matching tensor"
                )
            limits = torch.iinfo(expected_dtype)
            if value < limits.min or value > limits.max:
                raise ValueError(
                    f"parameter {variable_name!r} update is outside "
                    f"{expected_dtype} range"
                )
            return value
        raise ValueError(
            f"parameter {variable_name!r} has unsupported dtype "
            f"{expected_dtype}"
        )

    @staticmethod
    def _validate_set_conflicts(
        plans: tuple[_ParameterChangePlan, ...],
    ) -> None:
        for index, item in enumerate(plans):
            if not item.is_set_value:
                continue
            for existing in plans[:index]:
                if not (
                    existing.is_set_value
                    and existing.module_name == item.module_name
                    and existing.field_name == item.field_name
                    and existing.start_time == item.start_time
                ):
                    continue
                if existing.target_ids is None or item.target_ids is None:
                    raise ValueError(
                        f"parameter {item.variable_name!r} has overlapping SET "
                        f"plans at {item.start_time}: a global SET conflicts "
                        "with every other SET"
                    )
                if set(item.target_ids).intersection(existing.target_ids):
                    raise ValueError(
                        f"parameter {item.variable_name!r} has overlapping SET "
                        f"target_ids at {item.start_time}"
                    )


__all__: list[str] = []

"""Construction-time compilation of scheduled parameter changes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import cftime
import numpy as np
import torch

from hydroforge.compiler.partition import _searchsorted_batch
from hydroforge.contracts.fields import (
    ModuleFieldSchema,
    concrete_tensor_dtype,
)
from hydroforge.contracts.parameters import ParameterChange, ParameterValue
from hydroforge.model.tensors import ModuleTensors

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


@dataclass(frozen=True, slots=True)
class _TargetIdLookup:
    """Validated global-to-local lookup shared by changes on one ID field."""

    id_values: np.ndarray
    order: np.ndarray
    sorted_ids: np.ndarray
    local_by_global: np.ndarray
    local_extent: int

    def global_indices(self, ids: np.ndarray) -> np.ndarray:
        position = _searchsorted_batch(self.sorted_ids, ids)
        valid = position < self.sorted_ids.size
        hit = np.zeros(ids.shape, dtype=bool)
        hit[valid] = self.sorted_ids[position[valid]] == ids[valid]
        index = np.full(ids.shape, -1, dtype=np.int64)
        index[hit] = self.order[position[hit]]
        return index


class ParameterSemanticCompiler:
    """Resolve every parameter declaration before runtime materialization."""

    def __init__(
        self,
        model: AbstractModel,
        partition: _PartitionSemanticCompiler,
    ) -> None:
        self.model = model
        self.partition = partition
        self._qualified, self._unqualified = self._field_index()
        self._target_id_lookups: dict[tuple[str, str | None], _TargetIdLookup] = {}

    def _parameter_shape(self, resolved: _ResolvedParameterField) -> tuple[int, ...]:
        view = self.model._data.prepare_modules()[resolved.module_name]
        tensors = ModuleTensors(view)
        field = resolved.schema
        shape = tensors._expected_shape(field.name)
        if shape is None:
            raise ValueError(f"Inactive parameter {field.name!r}")
        if field.name in view.model_fields_set:
            value = getattr(view, field.name)
            if isinstance(value, torch.Tensor):
                if tuple(value.shape) != shape:
                    tensors._resolve_batch_shape(field, value, shape)
                return tuple(value.shape)
        return shape

    def compile(
        self,
        changes: tuple[ParameterChange, ...],
    ) -> tuple[_ParameterChangePlan, ...]:
        plans = tuple(
            sorted(
                (self._compile_change(change) for change in changes),
                key=lambda item: item.start_time,
            )
        )
        self._validate_set_conflicts(plans)
        return plans

    def _field_index(
        self,
    ) -> tuple[
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
                    module_name,
                    field,
                ):
                    continue
                resolved = _ResolvedParameterField(module_name, field)
                qualified[f"{module_name}.{field.name}"] = resolved
                unqualified[field.name] = (
                    None if field.name in unqualified else resolved
                )
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
                if owner_module is not None
                else None
            )
            if resolved is None:
                resolved = self._unqualified.get(name)
        if resolved is None:
            raise ValueError(
                f"{label} {name!r} was not found unambiguously in an opened module"
            )
        return resolved

    def _compile_change(
        self,
        change: ParameterChange,
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
        local_shape = self._parameter_shape(resolved)
        index_axis = len(local_shape) - len(tensor.shape)
        group = self.partition.variable_groups.get(field.name)
        local_rows: np.ndarray | None = None
        if group is not None:
            local_rows = self.partition.rank_indices(group)

        expected_dtype = concrete_tensor_dtype(
            tensor.dtype,
            self.model.dtype,
            self.model.mixed_precision,
        )
        expected_device = (
            torch.device("cpu")
            if tensor.mode == "cpu"
            else torch.device(self.model.device)
        )

        target_ids: tuple[int, ...] | None = None
        local_indices: tuple[int, ...] | None = None
        local_positions: tuple[int, ...] | None = None
        update_shape = local_shape
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
                local_shape=local_shape,
                index_axis=index_axis,
                group=group,
                local_rows=local_rows,
            )
            requested_shape = list(local_shape)
            requested_shape[index_axis] = len(target_ids)
            update_shape = tuple(requested_shape)

        target_value = change._trusted_value("target_value")
        delta = change._trusted_value("delta")
        is_set = target_value is not None
        raw_value = target_value if is_set else delta
        if (
            self.model.parallel is not None
            and index_axis == 1
            and isinstance(raw_value, torch.Tensor)
            and raw_value.ndim == len(update_shape)
        ):
            if raw_value.shape[0] != self.model.ensemble_size:
                raise ValueError(
                    "ensemble parameter changes require the global member axis"
                )
            raw_value = raw_value[self.model.parallel.member_slice]
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
        local_shape: tuple[int, ...],
        index_axis: int,
        group: str | None,
        local_rows: np.ndarray | None,
    ) -> tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        str,
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
                f"parameter target ID field {id_name!r} must declare is_key=True"
            )
        if len(id_tensor.shape) != 1:
            raise ValueError(
                f"parameter target ID field {id_name!r} must be one-dimensional"
            )
        parameter_coordinate = (
            parameter_tensor.dim_coords.rsplit(".", 1)[-1]
            if parameter_tensor.dim_coords
            else None
        )
        id_coordinate = (
            id_field.name
            if id_tensor.is_coordinate
            else (
                id_tensor.dim_coords.rsplit(".", 1)[-1]
                if id_tensor.dim_coords
                else None
            )
        )
        if parameter_coordinate != id_coordinate:
            raise ValueError(
                f"parameter target ID field {id_name!r} is not aligned to "
                f"{change.variable!r} coordinate {parameter_coordinate!r}"
            )

        qualified_id = f"{resolved_id.module_name}.{id_field.name}"
        lookup = self._target_id_lookups.get((qualified_id, group))
        if lookup is None:
            lookup = self._target_id_lookup(
                resolved_id,
                id_name=id_name,
                local_rows=local_rows,
            )
            self._target_id_lookups[(qualified_id, group)] = lookup
        if lookup.local_extent != local_shape[index_axis]:
            raise ValueError(
                f"parameter target ID field {id_name!r} shape "
                f"({lookup.local_extent},) "
                f"is not co-indexed with {change.variable!r} axis "
                f"length {local_shape[index_axis]}"
            )

        requested_ids = change._trusted_value("target_ids")
        target_ids = (
            requested_ids
            if isinstance(requested_ids, tuple)
            else tuple(requested_ids.tolist())
        )
        target_array = np.asarray(target_ids, dtype=lookup.id_values.dtype)
        global_indices = lookup.global_indices(target_array)
        missing = global_indices < 0
        if np.any(missing):
            missing_ids = target_array[missing][:10].tolist()
            raise ValueError(
                f"target_ids for {change.variable!r} were not found in "
                f"{id_name!r}: {missing_ids}"
            )

        local = lookup.local_by_global[global_indices]
        present = local >= 0
        return (
            target_ids,
            tuple(local[present].tolist()),
            tuple(np.flatnonzero(present).tolist()),
            qualified_id,
        )

    def _target_id_lookup(
        self,
        resolved_id: _ResolvedParameterField,
        *,
        id_name: str,
        local_rows: np.ndarray | None,
    ) -> _TargetIdLookup:
        id_field = resolved_id.schema
        local_view = self.model._data.prepare_modules()[resolved_id.module_name]
        local_id_tensor = getattr(local_view, id_field.name)
        if not isinstance(local_id_tensor, torch.Tensor):
            raise ValueError(f"parameter target ID field {id_name!r} must be a tensor")
        if local_id_tensor.ndim != 1 or local_id_tensor.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError(
                f"parameter target ID field {id_name!r} must be a one-dimensional "
                "integer tensor"
            )
        order, sorted_ids, unique = self.partition.sorted_global_key(
            id_field.name,
            lambda: self.model._input[id_field.name],
        )
        if not unique:
            raise ValueError(
                f"parameter target ID field {id_name!r} contains duplicate IDs"
            )
        id_values = sorted_ids
        lookup = _TargetIdLookup(
            id_values=id_values,
            order=order,
            sorted_ids=sorted_ids,
            local_by_global=np.empty(0, dtype=np.int64),
            local_extent=0,
        )

        if local_rows is None:
            local_rows = np.arange(id_values.size, dtype=np.int64)
        local_id_values = local_id_tensor.detach().cpu().numpy()
        prepared_global_indices = lookup.global_indices(
            local_id_values.astype(id_values.dtype, copy=False)
        )
        owned = np.zeros(id_values.size, dtype=bool)
        owned[local_rows] = True
        if np.any(prepared_global_indices < 0) or not np.all(
            owned[prepared_global_indices]
        ):
            raise ValueError(
                f"prepared parameter target ID field {id_name!r} contains IDs "
                "outside its rank-local input partition"
            )
        local_by_global = np.full(id_values.size, -1, dtype=np.int64)
        local_by_global[prepared_global_indices] = np.arange(
            prepared_global_indices.size,
            dtype=np.int64,
        )
        # Global IDs are unique, so a collapsed scatter means a local duplicate.
        if np.count_nonzero(local_by_global >= 0) != prepared_global_indices.size:
            raise ValueError(
                f"prepared parameter target ID field {id_name!r} contains "
                "duplicate IDs"
            )
        return _TargetIdLookup(
            id_values=id_values,
            order=order,
            sorted_ids=sorted_ids,
            local_by_global=local_by_global,
            local_extent=int(prepared_global_indices.size),
        )

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
                    f"parameter {variable_name!r} update tensor must be contiguous"
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
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.uint16,
            torch.int32,
            torch.uint32,
            torch.int64,
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
            f"parameter {variable_name!r} has unsupported dtype {expected_dtype}"
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

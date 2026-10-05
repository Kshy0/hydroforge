# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Rank-local compilation and transactional execution of parameter changes."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Any

import cftime
import numpy as np
import torch

from hydroforge.compiler.parameters import validate_set_targets
from hydroforge.contracts.fields import concrete_tensor_dtype
from hydroforge.contracts.parameters import ParameterValue, validate_parameter_scalar
from hydroforge.core.devices import devices_match
from hydroforge.core.errors import ResourceCleanupError
from hydroforge.declare.module import AbstractModule
from hydroforge.declare.tensors import ModulePayload, ModuleTensors
from hydroforge.execution.partition import _searchsorted_batch

if TYPE_CHECKING:
    from hydroforge.compiler.fields import FieldEntry
    from hydroforge.compiler.parameters import ParameterTarget
    from hydroforge.execution.session import ModelRuntime


_FRAMEWORK_ATTRIBUTES = frozenset(dir(AbstractModule))


class _ReadRecorder:
    """Stand-in ``self`` that records the declared tensor reads of one formula.

    Declared tensor and reference-index reads are recorded by qualified name.
    Sibling modules resolve to their own recorders, and methods or properties
    authored by the module class run against the recorder so that helper
    reads are recorded as well. Scalars, framework methods and private state
    are read from the module itself.
    """

    __slots__ = ("_recorded_module", "_recorded_fields", "_recorded_reads")

    def __init__(
        self,
        module: Any,
        fields: Mapping[int, Mapping[str, str]],
        reads: set[str],
    ) -> None:
        self._recorded_module = module
        self._recorded_fields = fields
        self._recorded_reads = reads

    @property
    def __class__(self) -> type:
        # isinstance() and zero-argument super() see the real module class.
        return type(self._recorded_module)

    def __getattr__(self, name: str) -> Any:
        module = self._recorded_module
        qualified = self._recorded_fields.get(id(module), {}).get(name)
        if qualified is not None:
            self._recorded_reads.add(qualified)
            return getattr(module, name)
        if name in module.spec().references:
            sibling = getattr(module, name)
            if sibling is None:
                return None
            return _ReadRecorder(sibling, self._recorded_fields, self._recorded_reads)
        attribute = inspect.getattr_static(type(module), name, None)
        if (
            isinstance(attribute, (FunctionType, property))
            and name not in _FRAMEWORK_ATTRIBUTES
        ):
            return attribute.__get__(self, type(module))
        return getattr(module, name)


@dataclass(frozen=True, slots=True)
class _ParameterChangePlan:
    """One complete rank-local instruction compiled from public input."""

    target: ParameterTarget
    variable_name: str
    module_name: str
    field_name: str
    start_time: datetime | cftime.datetime
    active_steps: int
    delta: ParameterValue
    target_value: ParameterValue | None
    local_indices: tuple[int, ...] | None
    index_axis: int


@dataclass(frozen=True, slots=True)
class _TargetIdLookup:
    """Validated global-to-local lookup shared by changes on one ID field."""

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


class LocalParameterCompiler:
    """Resolve compiled parameter targets against rank-local module payloads."""

    def __init__(
        self,
        runtime: ModelRuntime,
        payloads: Mapping[str, ModulePayload],
    ) -> None:
        self.runtime = runtime
        self.payloads = payloads
        self._target_id_lookups: dict[tuple[str, str | None], _TargetIdLookup] = {}

    def compile(
        self, targets: tuple[ParameterTarget, ...]
    ) -> tuple[_ParameterChangePlan, ...]:
        plans = tuple(self._compile_change(target) for target in targets)
        validate_set_targets((item.target, None, item.local_indices) for item in plans)
        return plans

    def _parameter_shape(self, field: FieldEntry) -> tuple[int, ...]:
        view = self.payloads[field.module]
        tensors = ModuleTensors(view)
        shape = tensors._expected_shape(field.name)
        if shape is None:
            raise ValueError(f"Inactive parameter {field.name!r}")
        if field.name in view.model_fields_set:
            value = getattr(view, field.name)
            if isinstance(value, torch.Tensor):
                if tuple(value.shape) != shape:
                    tensors._resolve_batch_shape(field.spec, value, shape)
                return tuple(value.shape)
        return shape

    def _compile_change(self, target: ParameterTarget) -> _ParameterChangePlan:
        plan = self.runtime.plan
        change = target.change
        field = target.field
        tensor = field.tensor
        local_shape = self._parameter_shape(field)
        index_axis = len(local_shape) - len(tensor.shape)
        group = plan.fields.variable_groups.get(field.name)
        local_rows = (
            None if group is None else self.runtime.partition.rank_indices(group)
        )
        local_indices: tuple[int, ...] | None = None
        local_positions: tuple[int, ...] | None = None
        update_shape = local_shape
        if target.target_ids is not None:
            local_indices, local_positions = self._compile_target_ids(
                target,
                local_shape=local_shape,
                index_axis=index_axis,
                group=group,
                local_rows=local_rows,
            )
            requested_shape = list(local_shape)
            requested_shape[index_axis] = len(target.target_ids)
            update_shape = tuple(requested_shape)

        value = self.bind_update_value(
            plan, target, update_shape, index_axis, local_positions
        )
        is_set = target.is_set_value
        return _ParameterChangePlan(
            target=target,
            variable_name=change.variable,
            module_name=field.module,
            field_name=field.name,
            start_time=target.start,
            active_steps=change.active_steps,
            delta=change._trusted_value("delta") if is_set else value,
            target_value=value if is_set else None,
            local_indices=local_indices,
            index_axis=index_axis,
        )

    @staticmethod
    def bind_update_value(
        plan: Any,
        target: ParameterTarget,
        update_shape: tuple[int, ...],
        index_axis: int,
        local_positions: tuple[int, ...] | None,
    ) -> ParameterValue:
        """Bind the owned declaration to one shape and member/ID selection."""

        change = target.change
        tensor = target.field.tensor
        is_set = target.is_set_value
        raw_value = change._trusted_value("target_value" if is_set else "delta")
        if (
            plan.parallel is not None
            and index_axis == 1
            and isinstance(raw_value, torch.Tensor)
            and raw_value.ndim == len(update_shape)
        ):
            if raw_value.shape[0] != plan.ensemble_size:
                raise ValueError(
                    "ensemble parameter changes require the global member axis"
                )
            raw_value = raw_value[plan.parallel.member_slice]
        value = LocalParameterCompiler._validate_update_value(
            raw_value,
            expected_shape=update_shape,
            expected_dtype=concrete_tensor_dtype(
                tensor.dtype, plan.dtype, plan.mixed_precision
            ),
            expected_device=torch.device("cpu")
            if tensor.mode == "cpu"
            else plan.device,
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

        return value

    def _compile_target_ids(
        self,
        target: ParameterTarget,
        *,
        local_shape: tuple[int, ...],
        index_axis: int,
        group: str | None,
        local_rows: np.ndarray | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        change = target.change
        id_field = target.id_field
        id_name = change.target_id_field or target.field.tensor.dim_coords
        lookup = self._target_id_lookups.get((id_field.qualified, group))
        if lookup is None:
            lookup = self._target_id_lookup(
                id_field, id_name=id_name, local_rows=local_rows
            )
            self._target_id_lookups[(id_field.qualified, group)] = lookup
        if lookup.local_extent != local_shape[index_axis]:
            raise ValueError(
                f"parameter target ID field {id_name!r} shape "
                f"({lookup.local_extent},) "
                f"is not co-indexed with {change.variable!r} axis "
                f"length {local_shape[index_axis]}"
            )
        target_ids = target.target_ids
        limits = np.iinfo(lookup.sorted_ids.dtype)
        outside = [
            value for value in target_ids if not limits.min <= value <= limits.max
        ]
        if outside:
            raise ValueError(
                f"target_ids for {change.variable!r} are outside the "
                f"{lookup.sorted_ids.dtype} range of {id_name!r}: {outside[:10]}"
            )
        target_array = np.asarray(target_ids, dtype=lookup.sorted_ids.dtype)
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
            tuple(local[present].tolist()),
            tuple(np.flatnonzero(present).tolist()),
        )

    def _target_id_lookup(
        self,
        id_field: FieldEntry,
        *,
        id_name: str,
        local_rows: np.ndarray | None,
    ) -> _TargetIdLookup:
        runtime = self.runtime
        local_id_tensor = getattr(self.payloads[id_field.module], id_field.name)
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
        order, sorted_ids, unique = runtime.partition.sorted_global_key(
            id_field.name,
            lambda: runtime.input[id_field.name],
        )
        if not unique:
            raise ValueError(
                f"parameter target ID field {id_name!r} contains duplicate IDs"
            )
        local_by_global = np.full(sorted_ids.size, -1, dtype=np.int64)
        lookup = _TargetIdLookup(
            order=order,
            sorted_ids=sorted_ids,
            local_by_global=local_by_global,
            local_extent=local_id_tensor.numel(),
        )

        if local_rows is None:
            local_rows = np.arange(sorted_ids.size, dtype=np.int64)
        local_id_values = local_id_tensor.detach().cpu().numpy()
        prepared_global_indices = lookup.global_indices(
            local_id_values.astype(sorted_ids.dtype, copy=False)
        )
        owned = np.zeros(sorted_ids.size, dtype=bool)
        owned[local_rows] = True
        if np.any(prepared_global_indices < 0) or not np.all(
            owned[prepared_global_indices]
        ):
            raise ValueError(
                f"prepared parameter target ID field {id_name!r} contains IDs "
                "outside its rank-local input partition"
            )
        local_by_global[prepared_global_indices] = np.arange(
            prepared_global_indices.size,
            dtype=np.int64,
        )
        # Global IDs are unique, so a collapsed scatter means a local duplicate.
        if np.count_nonzero(local_by_global >= 0) != prepared_global_indices.size:
            raise ValueError(
                f"prepared parameter target ID field {id_name!r} contains duplicate IDs"
            )
        return lookup

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
            if not devices_match(value.device, expected_device):
                raise ValueError(
                    f"parameter {variable_name!r} update tensor must be on "
                    f"device {expected_device}; got {value.device}"
                )
            return value

        return validate_parameter_scalar(
            value, dtype=expected_dtype, variable_name=variable_name, is_set=is_set
        )


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One runtime binding of an already compiled parameter instruction."""

    target: ParameterTarget
    variable_name: str
    start_time: datetime | cftime.datetime
    active_steps: int
    delta: ParameterValue
    target_value: ParameterValue | None
    module: Any
    attr_name: str
    indices: torch.Tensor | None
    index_axis: int

    @property
    def is_set_value(self) -> bool:
        return self.target_value is not None

    @property
    def is_incremental(self) -> bool:
        return not self.is_set_value


@dataclass(slots=True)
class ActivePlan:
    """Bookkeeping wrapper around an instruction currently being executed."""

    item: PlanItem
    steps_executed: int = 0


class ParameterChangeEffect(Enum):
    """Exact execution consequence of one parameter-plan evaluation."""

    UNCHANGED = "unchanged"
    UPDATED = "updated"


@dataclass(frozen=True, slots=True)
class _TensorSnapshot:
    values: torch.Tensor
    indices: torch.Tensor | None
    index_axis: int


def _tensor_byte_ranges(
    tensor: torch.Tensor, indices: torch.Tensor | None = None, axis: int = 0
) -> tuple[tuple[int, int], ...]:
    """Exact occupied byte intervals, including offsets and selected rows.

    Model parameters are contiguous. Keep a strided fallback for resident
    aliases so a shared allocation alone never implies overlapping values.
    This is cold-path validation, never part of a recorded numerical kernel.
    """

    if tensor.numel() == 0:
        return ()
    size = tensor.element_size()
    pointer = tensor.data_ptr()
    if tensor.is_contiguous():
        if indices is None:
            return ((pointer, pointer + tensor.numel() * size),)
        rows = sorted(set(indices.tolist()))
        block = tensor.stride(axis) * size
        outer = tensor.numel() // (tensor.shape[axis] * tensor.stride(axis))
        ranges = (
            (
                pointer + (prefix * tensor.shape[axis] + row) * block,
                pointer + (prefix * tensor.shape[axis] + row + 1) * block,
            )
            for prefix in range(outer)
            for row in rows
        )
    else:
        selected = None if indices is None else set(indices.tolist())
        ranges = (
            (
                pointer
                + sum(i * stride for i, stride in zip(index, tensor.stride())) * size,
                pointer
                + (sum(i * stride for i, stride in zip(index, tensor.stride())) + 1)
                * size,
            )
            for index in np.ndindex(tuple(tensor.shape))
            if selected is None or index[axis] in selected
        )
        ranges = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _ranges_overlap(left, right) -> bool:
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][0] < right[j][1] and right[j][0] < left[i][1]:
            return True
        if left[i][1] <= right[j][0]:
            i += 1
        else:
            j += 1
    return False


def _validate_resident_sets(items, replacements=None) -> None:
    """Reject same-time SETs that write any common resident bytes."""

    replacements = {} if replacements is None else replacements
    groups = {}
    for item in items:
        if not item.is_set_value:
            continue
        current = getattr(item.module, item.attr_name)
        tensor = replacements.get(id(current), current)
        ranges = _tensor_byte_ranges(tensor, item.indices, item.index_axis)
        previous = groups.setdefault((item.start_time, tensor.device), [])
        if any(_ranges_overlap(ranges, other) for other in previous):
            raise ValueError(
                f"parameter {item.variable_name!r} has overlapping SET "
                f"targets in shared storage at {item.start_time}"
            )
        previous.append(ranges)


class ParameterPlanRuntime:
    """Apply rank-local plans compiled by ``LocalParameterCompiler``."""

    def __init__(
        self,
        runtime: ModelRuntime,
        plans: tuple[_ParameterChangePlan, ...],
    ) -> None:
        self.runtime = runtime
        self._plans = tuple(self._bind(item) for item in plans)
        _validate_resident_sets(self._plans)
        self._aliases: dict[str, set[str]] = {}
        self.dependencies: Mapping[str, tuple[str, ...]] = MappingProxyType({})
        self._derived: dict[str, tuple[Any, str, cached_property]] = {}
        self._dependency_revision: int | None = None
        self._active_plans: list[ActivePlan] = []
        self._next_plan_idx = 0
        self._step_transaction_snapshots: list[tuple[Any, str, Any]] = []

    def _bind(self, item: _ParameterChangePlan) -> PlanItem:
        module = self.runtime.modules[item.module_name]
        target = getattr(module, item.field_name)
        indices = (
            None
            if item.local_indices is None
            else torch.tensor(
                item.local_indices,
                dtype=torch.int64,
                device=target.device,
            )
        )
        return PlanItem(
            target=item.target,
            variable_name=item.variable_name,
            start_time=item.start_time,
            active_steps=item.active_steps,
            delta=item.delta,
            target_value=item.target_value,
            module=module,
            attr_name=item.field_name,
            indices=indices,
            index_axis=item.index_axis,
        )

    def prepare_rebind(
        self, replacements: Mapping[int, torch.Tensor]
    ) -> tuple[tuple[PlanItem, ...], list[ActivePlan]] | None:
        """Prepare live ID/value bindings before a structural mutation.

        Completed instructions do not constrain later layouts. Source values
        come from the declaration so an ID returning to this rank recovers its
        original update, while active-step counters keep their current values.
        """

        pending = self._plans[self._next_plan_idx :]
        active = [
            p for p in self._active_plans if p.steps_executed < p.item.active_steps
        ]
        items = {id(item): item for item in (*pending, *(p.item for p in active))}
        updated: dict[int, PlanItem] = {}
        lookups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for identity, item in items.items():
            target = item.target
            current = getattr(item.module, item.attr_name)
            id_field = target.id_field
            ids = (
                None
                if id_field is None
                else getattr(self.runtime.modules[id_field.module], id_field.name)
            )
            if id(current) not in replacements and (
                ids is None or id(ids) not in replacements
            ):
                continue
            candidate = replacements.get(id(current), current)
            shape = list(candidate.shape)
            positions = None
            indices = None
            if target.target_ids is not None:
                if ids is None:
                    raise ValueError(
                        f"cannot rebind {item.variable_name!r}: target ID field was discarded"
                    )
                candidate_ids = replacements.get(id(ids), ids)
                if candidate_ids.numel() != shape[item.index_axis]:
                    raise ValueError(
                        f"target IDs for {item.variable_name!r} no longer align to its parameter axis"
                    )
                lookup = lookups.get(id(candidate_ids))
                if lookup is None:
                    values = candidate_ids.detach().cpu().numpy()
                    order = np.argsort(values, kind="stable")
                    lookup = lookups[id(candidate_ids)] = (order, values[order])
                order, ordered = lookup
                requested = np.asarray(target.target_ids, dtype=ordered.dtype)
                found = _searchsorted_batch(ordered, requested)
                present = found < ordered.size
                present[present] &= ordered[found[present]] == requested[present]
                positions = tuple(np.flatnonzero(present).tolist())
                indices = torch.tensor(
                    order[found[present]], dtype=torch.int64, device=candidate.device
                )
                shape[item.index_axis] = len(target.target_ids)
            value = LocalParameterCompiler.bind_update_value(
                self.runtime.plan, target, tuple(shape), item.index_axis, positions
            )
            updated[identity] = replace(
                item,
                indices=indices,
                delta=item.delta if item.is_set_value else value,
                target_value=value if item.is_set_value else None,
            )
        if not updated:
            return None
        rebound = tuple(updated.get(identity, item) for identity, item in items.items())
        _validate_resident_sets(rebound, replacements)
        validate_set_targets(
            (item.target, None, None if item.indices is None else item.indices.tolist())
            for item in rebound
            if item.is_set_value
        )
        return (
            tuple(updated.get(id(item), item) for item in self._plans),
            [
                ActivePlan(updated.get(id(p.item), p.item), p.steps_executed)
                for p in self._active_plans
            ],
        )

    def install_rebind(
        self, candidate: tuple[tuple[PlanItem, ...], list[ActivePlan]] | None
    ) -> None:
        """Publish already prepared bindings without changing the plan cursor."""

        if candidate is not None:
            self._plans, self._active_plans = candidate

    @staticmethod
    def _evaluate_derived(
        module: Any, name: str, descriptor: cached_property
    ) -> torch.Tensor:
        current = getattr(module, name)
        fresh = descriptor.func(module)
        if (
            not isinstance(current, torch.Tensor)
            or not isinstance(fresh, torch.Tensor)
            or fresh.shape != current.shape
            or fresh.dtype != current.dtype
            or fresh.device != current.device
            or fresh.layout != torch.strided
            or current.layout != torch.strided
            or not fresh.is_contiguous()
            or not current.is_contiguous()
        ):
            raise ValueError(
                f"{module.module_name}.{name}: parameter refresh must preserve "
                "tensor shape, dtype, device and contiguous layout"
            )
        return fresh

    def _record_dependencies(self) -> None:
        """Discover direct field reads before the first write, without replacing caches.

        Formulas must be pure and have fixed field dependencies. Scalars and
        configuration remain fixed; this is not a Python control-flow tracer.
        A structural revision requires a fresh graph at the next parameter event.
        """
        revision = self.runtime.execution.structural_revision
        if self._dependency_revision == revision:
            return
        fields: dict[int, dict[str, str]] = {}
        declared: dict[str, tuple[Any, str, Any]] = {}
        candidates = {}
        for module_name in self.runtime.plan.modules:
            module = self.runtime.modules[module_name]
            module_fields = fields.setdefault(id(module), {})
            for field in module.spec().tensor_fields.values():
                qualified = f"{module_name}.{field.name}"
                module_fields[field.name] = qualified
                declared[qualified] = (module, field.name, field.tensor)
                if (
                    field.computed
                    and field.tensor.category == "derived_param"
                    and module._is_tensor_field_active(field.name)
                ):
                    descriptor = getattr(type(module), field.name)
                    if not isinstance(descriptor, cached_property):
                        raise ValueError(
                            f"{qualified}: parameter refresh requires a cached derived parameter"
                        )
                    if isinstance(module.__dict__.get(field.name), torch.Tensor):
                        candidates[qualified] = (module, field.name, descriptor)
            for name, metadata in module._binding.plan.reference_indices.items():
                qualified = f"{module_name}.{name}"
                module_fields[name] = qualified
                declared[qualified] = (module, name, metadata)
        dependencies = {}
        with torch.inference_mode():
            for qualified, (module, name, descriptor) in candidates.items():
                reads: set[str] = set()
                try:
                    # Invoke only the formula: reading the result buffer here
                    # would falsely record a self-dependency.
                    descriptor.func(_ReadRecorder(module, fields, reads))
                except Exception as error:
                    raise ValueError(
                        f"cannot record parameter dependencies for {qualified}: {error}"
                    ) from error
                for dependency in reads:
                    source, source_name, metadata = declared[dependency]
                    if metadata.category not in {
                        "param",
                        "derived_param",
                        "topology",
                    }:
                        raise ValueError(
                            f"{qualified}: derived parameters cannot depend on runtime field {dependency}"
                        )
                    if not isinstance(getattr(source, source_name), torch.Tensor):
                        raise ValueError(
                            f"{qualified}: dependency {dependency} must remain resident"
                        )
                    if (
                        metadata.category == "derived_param"
                        and dependency not in candidates
                    ):
                        raise ValueError(
                            f"{qualified}: dependency {dependency} has no resident derived formula"
                        )
                dependencies[qualified] = tuple(sorted(reads))
        try:
            order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            raise ValueError(
                f"cyclic parameter tensor dependencies: {error.args[1]}"
            ) from error
        # Qualified names can refer to the same tensor or overlapping views.
        # Rebuild from live resident buffers after every structural revision.
        resident = {}
        for qualified, (module, name, _metadata) in declared.items():
            value = module.__dict__.get(name)
            if isinstance(value, torch.Tensor):
                resident[qualified] = (value.device, _tensor_byte_ranges(value))
        aliases = {name: {name} for name in resident}
        names = tuple(resident)
        for index, name in enumerate(names):
            device, ranges = resident[name]
            for other in names[:index]:
                other_device, other_ranges = resident[other]
                if device == other_device and _ranges_overlap(ranges, other_ranges):
                    aliases[name].add(other)
                    aliases[other].add(name)
        # Publish only a complete graph; a failed discovery remains retryable.
        self._aliases = aliases
        self._derived = {name: candidates[name] for name in order if name in candidates}
        self.dependencies = MappingProxyType(dependencies)
        self._dependency_revision = revision

    def _refresh_derived_parameters(self, changed: set[str]) -> None:
        changed.update(
            alias for name in tuple(changed) for alias in self._aliases.get(name, ())
        )
        with torch.inference_mode():
            for qualified, (module, name, descriptor) in self._derived.items():
                if changed.isdisjoint(self.dependencies[qualified]):
                    continue
                fresh = self._evaluate_derived(module, name, descriptor)
                current = getattr(module, name)
                self._step_transaction_snapshots.append(
                    (
                        module,
                        name,
                        _TensorSnapshot(current.detach().clone(), None, 0),
                    )
                )
                current.copy_(fresh)
                changed.update(self._aliases.get(qualified, (qualified,)))
            for module_name in self.runtime.plan.modules:
                self.runtime.modules[module_name].validate_parameters()

    @staticmethod
    def _apply_tensor_value(
        target: torch.Tensor,
        value: ParameterValue,
        indices: torch.Tensor | None,
        *,
        is_set: bool,
        index_axis: int = 0,
    ) -> None:
        if indices is None:
            if is_set:
                if isinstance(value, torch.Tensor) and value.ndim != 0:
                    target.copy_(value)
                else:
                    target.fill_(value)
            else:
                target.add_(value)
            return
        selection = [slice(None)] * target.ndim
        selection[index_axis] = indices
        selected = tuple(selection)
        if is_set:
            target[selected] = value
        else:
            target[selected] += value

    def _apply_grouped_changes(
        self,
        module: Any,
        attr: str,
        plans: list[ActivePlan],
    ) -> None:
        current = getattr(module, attr)
        for active in sorted(plans, key=lambda item: item.item.is_incremental):
            item = active.item
            value = item.target_value if item.is_set_value else item.delta
            self._apply_tensor_value(
                current,
                value,
                item.indices,
                is_set=item.is_set_value,
                index_axis=item.index_axis,
            )

    @staticmethod
    def _snapshot_value(
        value: torch.Tensor,
        plans: list[ActivePlan],
    ) -> _TensorSnapshot:
        indexed = [active.item.indices for active in plans]
        if all(indices is not None for indices in indexed):
            index_axis = plans[0].item.index_axis
            indices = torch.unique(torch.cat(indexed))
            values = value.detach().index_select(index_axis, indices)
            return _TensorSnapshot(values, indices, index_axis)
        return _TensorSnapshot(
            value.detach().clone(memory_format=torch.preserve_format),
            None,
            0,
        )

    @staticmethod
    def _restore_value(
        module: Any,
        attr: str,
        snapshot: _TensorSnapshot,
    ) -> None:
        current = getattr(module, attr)
        if snapshot.indices is None:
            current.copy_(snapshot.values)
        else:
            current.index_copy_(
                snapshot.index_axis,
                snapshot.indices,
                snapshot.values,
            )

    def step_transaction(self):
        """Keep parameter application atomic with one managed model step."""

        if not self._plans:
            return nullcontext()
        return self._step_transaction()

    def _restore_snapshots(self, snapshots) -> list[BaseException]:
        failures: list[BaseException] = []
        for module, attr, snapshot in reversed(snapshots):
            try:
                self._restore_value(module, attr, snapshot)
            except BaseException as error:
                failures.append(error)
        return failures

    @contextmanager
    def _step_transaction(self):
        cursor = (
            self._next_plan_idx,
            tuple(
                ActivePlan(active.item, active.steps_executed)
                for active in self._active_plans
            ),
        )
        self._step_transaction_snapshots = []
        try:
            yield
        except BaseException as step_error:
            rollback_errors = self._restore_snapshots(self._step_transaction_snapshots)
            self._next_plan_idx, active_plans = cursor
            self._active_plans = list(active_plans)
            if rollback_errors:
                error = ResourceCleanupError(
                    "managed-step parameter rollback",
                    (step_error, *rollback_errors),
                )
                raise error from step_error
            raise
        finally:
            self._step_transaction_snapshots = []

    def execute_parameter_change_plan(
        self,
        current_time: datetime | cftime.datetime | None,
    ) -> ParameterChangeEffect:
        """Apply one transactional plan step."""

        if current_time is None or not self._plans:
            return ParameterChangeEffect.UNCHANGED
        next_plan_idx = self._next_plan_idx
        active_plans = list(self._active_plans)
        while next_plan_idx < len(self._plans):
            plan = self._plans[next_plan_idx]
            if current_time >= plan.start_time:
                active_plans.append(ActivePlan(item=plan))
                next_plan_idx += 1
            else:
                break
        active_plans = [
            active
            for active in active_plans
            if active.steps_executed < active.item.active_steps
        ]
        if not active_plans:
            self._next_plan_idx = next_plan_idx
            self._active_plans = active_plans
            return ParameterChangeEffect.UNCHANGED

        grouped: dict[tuple[int, str], list[ActivePlan]] = {}
        for active in active_plans:
            key = (id(active.item.module), active.item.attr_name)
            grouped.setdefault(key, []).append(active)

        self._record_dependencies()

        snapshots: list[tuple[Any, str, Any]] = []
        for (_, attr), plans in grouped.items():
            module = plans[0].item.module
            current = getattr(module, attr)
            snapshots.append(
                (
                    module,
                    attr,
                    self._snapshot_value(current, plans),
                )
            )
        self._step_transaction_snapshots = snapshots
        try:
            # SET precedes increment even when different logical fields
            # share storage, just as it does for one field's grouped plans.
            for is_set in (True, False):
                for (_, attr), plans in grouped.items():
                    selected = [p for p in plans if p.item.is_set_value == is_set]
                    if selected:
                        self._apply_grouped_changes(plans[0].item.module, attr, selected)
            self._refresh_derived_parameters(
                {
                    f"{plans[0].item.module.module_name}.{attr}"
                    for (_, attr), plans in grouped.items()
                }
            )
        except BaseException as apply_error:
            rollback_errors = self._restore_snapshots(snapshots)
            if rollback_errors:
                error = ResourceCleanupError(
                    "parameter change rollback",
                    (apply_error, *rollback_errors),
                )
                raise error from apply_error
            raise
        for active in active_plans:
            active.steps_executed += 1
        self._next_plan_idx = next_plan_idx
        self._active_plans = active_plans
        return ParameterChangeEffect.UPDATED


__all__: list[str] = []

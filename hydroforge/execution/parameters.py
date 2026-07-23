# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cftime
import torch

from hydroforge.contracts import ResourceCleanupError
from hydroforge.contracts.temporal import require_calendar

# ---------------------------------------------------------------------------
# PlanItem / ActivePlan
# ---------------------------------------------------------------------------

@dataclass
class PlanItem:
    """A single parameter-change instruction."""

    # User inputs
    variable_name: str
    start_time: Union[datetime, cftime.datetime]
    active_steps: int = 1
    delta: Union[float, torch.Tensor] = 0.0
    target_value: Optional[Union[float, torch.Tensor]] = None
    target_ids: Optional[Union[List[int], torch.Tensor]] = None
    target_id_field: Optional[str] = None

    # Cached execution context (resolved once)
    _module: Optional[Any] = None
    _attr_name: str = ""
    _indices: Optional[torch.Tensor] = None
    _index_axis: int = 0
    _resolved_id_field: Optional[str] = None
    _is_ready: bool = False

    @property
    def is_set_value(self) -> bool:
        return self.target_value is not None

    @property
    def is_incremental(self) -> bool:
        return not self.is_set_value


@dataclass
class ActivePlan:
    """Bookkeeping wrapper around a PlanItem that is currently being executed."""

    item: PlanItem
    steps_executed: int = 0
    executed_once: bool = False


class ParameterChangeEffect(Enum):
    """Exact execution consequence of one parameter-plan evaluation."""

    UNCHANGED = "unchanged"
    UPDATED = "updated"


@dataclass(frozen=True, slots=True)
class _TensorSnapshot:
    values: torch.Tensor
    indices: torch.Tensor | None
    index_axis: int


# ---------------------------------------------------------------------------
# ParameterPlanRuntime
# ---------------------------------------------------------------------------

class ParameterPlanRuntime:
    """Own runtime parameter-change planning for one model.

    Expects the host class to expose:
      - ``self.variable_map``  (cached_property)
      - ``self.rank``          (int field)
    """

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self._sealed = False
        self._last_time: datetime | cftime.datetime | None = None
        self._plans: list[PlanItem] = []
        self._active_plans: list[ActivePlan] = []
        self._next_plan_idx = 0
        self._step_transaction_active = False
        self._step_transaction_executed = False
        self._step_transaction_snapshots: list[tuple[Any, str, Any]] = []

    @property
    def rank(self):
        return self.owner.rank

    @property
    def variable_map(self):
        return self.owner.variable_map

    @property
    def _calendar(self) -> str:
        return self.owner.calendar

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_update_value(
        target: Any,
        value: float | torch.Tensor,
        indices: torch.Tensor | None,
        *,
        variable_name: str,
        index_axis: int = 0,
    ) -> None:
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"parameter {variable_name!r} target must be a torch.Tensor"
            )

        if indices is not None and target.ndim == 0:
            raise ValueError(
                f"target_ids cannot be applied to scalar tensor parameter "
                f"{variable_name!r}"
            )
        if indices is not None and not 0 <= index_axis < target.ndim:
            raise ValueError(
                f"parameter {variable_name!r} target-ID axis {index_axis} "
                f"is outside rank {target.ndim}"
            )
        if isinstance(value, torch.Tensor):
            if value.device != target.device:
                raise ValueError(
                    f"parameter {variable_name!r} update is on {value.device}, "
                    f"expected {target.device}"
                )
            if value.dtype != target.dtype:
                raise TypeError(
                    f"parameter {variable_name!r} update has dtype {value.dtype}, "
                    f"expected {target.dtype}; implicit precision conversion is "
                    "forbidden"
                )
            if value.layout is not torch.strided or not value.is_contiguous():
                raise ValueError(
                    f"parameter {variable_name!r} update must be a contiguous "
                    "strided tensor"
                )
            if value.ndim == 0:
                return
            expected = (
                target.shape
                if indices is None
                else torch.Size((
                    *target.shape[:index_axis], indices.numel(),
                    *target.shape[index_axis + 1:],
                ))
            )
            if value.shape != expected:
                raise ValueError(
                    f"parameter {variable_name!r} update shape "
                    f"{tuple(value.shape)} must be scalar or exactly "
                    f"{tuple(expected)}"
                )
            return
        if type(value) not in {int, float}:
            raise TypeError(
                f"parameter {variable_name!r} update must be an exact int, "
                f"float, or tensor, got {type(value).__name__}"
            )
        if type(value) is float and not math.isfinite(value):
            raise ValueError(
                f"parameter {variable_name!r} update must be finite"
            )

    @staticmethod
    def _apply_tensor_value(
        target: torch.Tensor,
        value: float | torch.Tensor,
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
        self, module: Any, attr: str, plans: List[ActivePlan],
    ) -> None:
        current_val = getattr(module, attr)
        if not isinstance(current_val, torch.Tensor):
            raise TypeError(
                f"resolved parameter {attr!r} is no longer a torch.Tensor"
            )

        # Set operations precede increments. Each plan remains explicit; tensor
        # values are never collapsed into a Python truth-value or silently cast.
        for active in sorted(plans, key=lambda item: item.item.is_incremental):
            item = active.item
            value = item.target_value if item.is_set_value else item.delta

            self._apply_tensor_value(
                current_val, value, item._indices,
                is_set=item.is_set_value, index_axis=item._index_axis,
            )

    @staticmethod
    def _snapshot_value(value: Any, plans: list[ActivePlan]) -> Any:
        if not isinstance(value, torch.Tensor):
            raise TypeError("resolved parameter snapshot target must be a tensor")
        indexed = [active.item._indices for active in plans]
        if all(indices is not None for indices in indexed):
            axes = {active.item._index_axis for active in plans}
            if len(axes) != 1:
                raise RuntimeError(
                    "parameter plans for one tensor resolved different ID axes"
                )
            index_axis = axes.pop()
            indices = torch.unique(torch.cat(indexed))
            values = value.index_select(index_axis, indices).detach().clone(
                memory_format=torch.preserve_format,
            )
            return _TensorSnapshot(values, indices, index_axis)
        return _TensorSnapshot(
            value.detach().clone(memory_format=torch.preserve_format),
            None, 0,
        )

    @staticmethod
    def _restore_value(module: Any, attr: str, snapshot: Any) -> None:
        current = getattr(module, attr)
        if isinstance(current, torch.Tensor):
            if not isinstance(snapshot, _TensorSnapshot):
                raise TypeError("tensor parameter rollback snapshot is not a tensor")
            if snapshot.indices is None:
                current.copy_(snapshot.values)
            else:
                current.index_copy_(
                    snapshot.index_axis,
                    snapshot.indices.to(dtype=torch.int64),
                    snapshot.values,
                )

    def _time_number(self, value: datetime | cftime.datetime) -> float:
        return float(cftime.date2num(
            value, "seconds since 1970-01-01 00:00:00",
            calendar=self._calendar,
        ))

    def _restore_time(self, value: float) -> datetime | cftime.datetime:
        if self._calendar == "standard":
            return datetime(1970, 1, 1) + timedelta(seconds=value)
        return cftime.num2date(
            value, "seconds since 1970-01-01 00:00:00",
            calendar=self._calendar,
            only_use_cftime_datetimes=True,
        )

    @staticmethod
    def _value_signature(value: float | torch.Tensor) -> dict[str, Any]:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            return {
                "kind": "tensor",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        return {"kind": type(value).__name__, "value": value}

    def _plan_fingerprint(self) -> str:
        definitions = []
        for item in self._plans:
            value = item.target_value if item.is_set_value else item.delta
            target_ids = item.target_ids
            definitions.append({
                "variable": item.variable_name,
                "start": self._time_number(item.start_time),
                "active_steps": item.active_steps,
                "operation": "set" if item.is_set_value else "add",
                "value": self._value_signature(value),
                "target_ids": (
                    None if target_ids is None
                    else self._value_signature(target_ids)
                ),
                "target_id_field": item.target_id_field,
            })
        encoded = json.dumps(
            definitions, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the exact cursor for this already-defined temporal program."""

        plan_indices = {id(item): index for index, item in enumerate(self._plans)}
        return {
            "fingerprint": self._plan_fingerprint(),
            "sealed": self._sealed,
            "next_plan_idx": self._next_plan_idx,
            "last_time": (
                None if self._last_time is None
                else self._time_number(self._last_time)
            ),
            "active": [{
                "plan": plan_indices[id(active.item)],
                "steps_executed": active.steps_executed,
                "executed_once": active.executed_once,
            } for active in self._active_plans],
        }

    def validate_checkpoint_state(
        self, state: Any,
    ) -> tuple[bool, int, datetime | cftime.datetime | None, tuple[ActivePlan, ...]]:
        """Validate a persisted cursor without changing the live plan."""

        required = {
            "fingerprint", "sealed", "next_plan_idx", "last_time", "active",
        }
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError(
                "checkpoint parameter-plan state has an invalid schema"
            )
        if state["fingerprint"] != self._plan_fingerprint():
            raise ValueError(
                "checkpoint parameter-change plan does not match the model plan"
            )
        sealed = state["sealed"]
        if type(sealed) is not bool:
            raise TypeError("checkpoint parameter-plan sealed flag must be bool")
        next_index = state["next_plan_idx"]
        if type(next_index) is not int or not 0 <= next_index <= len(self._plans):
            raise ValueError("checkpoint parameter-plan cursor is outside the plan")
        last_number = state["last_time"]
        if last_number is not None and type(last_number) not in {int, float}:
            raise TypeError("checkpoint parameter-plan last_time must be numeric")
        if last_number is not None and not math.isfinite(last_number):
            raise ValueError("checkpoint parameter-plan last_time must be finite")
        if (last_number is None) != (not sealed):
            raise ValueError(
                "checkpoint parameter-plan last_time is inconsistent with sealing"
            )
        active_state = state["active"]
        if not isinstance(active_state, list):
            raise TypeError("checkpoint active parameter plans must be a list")
        active: list[ActivePlan] = []
        seen: set[int] = set()
        for entry in active_state:
            if not isinstance(entry, dict) or set(entry) != {
                "plan", "steps_executed", "executed_once",
            }:
                raise ValueError("checkpoint active parameter plan is malformed")
            index = entry["plan"]
            steps = entry["steps_executed"]
            executed = entry["executed_once"]
            if (
                type(index) is not int or index in seen
                or not 0 <= index < next_index
            ):
                raise ValueError("checkpoint active parameter-plan index is invalid")
            item = self._plans[index]
            if type(steps) is not int or not 0 <= steps <= item.active_steps:
                raise ValueError(
                    "checkpoint active parameter-plan step count is invalid"
                )
            if type(executed) is not bool or executed != (steps > 0):
                raise ValueError(
                    "checkpoint active parameter-plan execution flag is invalid"
                )
            seen.add(index)
            active.append(ActivePlan(item, steps, executed))
        last_time = (
            None if last_number is None
            else self._restore_time(float(last_number))
        )
        return sealed, next_index, last_time, tuple(active)

    def restore_checkpoint_state(
        self,
        state: tuple[
            bool, int, datetime | cftime.datetime | None, tuple[ActivePlan, ...],
        ],
    ) -> None:
        """Commit a cursor returned by :meth:`validate_checkpoint_state`."""

        sealed, next_index, last_time, active = state
        self._sealed = sealed
        self._next_plan_idx = next_index
        self._last_time = last_time
        self._active_plans = list(active)

    def _resolve_id_tensor(self, module: Any, id_attr: Optional[str]) -> Optional[torch.Tensor]:
        """Resolve the ID tensor from a module given an attribute path."""
        if not id_attr:
            return None

        if "." in id_attr:
            parts = id_attr.split(".")
            curr = module
            found = True
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            if found and isinstance(curr, torch.Tensor):
                return curr
        elif hasattr(module, id_attr):
            val = getattr(module, id_attr)
            if isinstance(val, torch.Tensor):
                return val

        if id_attr in self.variable_map:
            mod_inst, field_name, _ = self.variable_map[id_attr]
            if hasattr(mod_inst, field_name):
                val = getattr(mod_inst, field_name)
                if isinstance(val, torch.Tensor):
                    return val

        return None

    def _resolve_parameter(
        self, variable_name: str,
    ) -> tuple[Any, str, Any, int]:
        if variable_name not in self.variable_map:
            raise ValueError(
                f"ParameterChangePlan Error: Variable {variable_name!r} "
                "not found in model."
            )
        module, attr, id_attr = self.variable_map[variable_name]
        field = module.get_tensor_schema(attr)
        if (
            field is None or field.tensor is None
            or field.tensor.category != "param"
        ):
            category = (
                None if field is None or field.tensor is None
                else field.tensor.category
            )
            raise ValueError(
                f"parameter mutation requires category='param'; "
                f"{variable_name!r} declares category={category!r}"
            )
        value = getattr(module, attr)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"declared parameter {variable_name!r} must be an initialized "
                f"torch.Tensor, got {type(value).__name__}"
            )
        shape = field.tensor.shape
        if type(shape) is not tuple:
            raise TypeError(
                f"parameter {variable_name!r} tensor shape contract must be "
                "an exact tuple"
            )
        declared_rank = len(shape)
        if value.ndim == declared_rank:
            index_axis = 0
        elif value.ndim == declared_rank + 1:
            num_trials = getattr(module, "num_trials", None)
            if (
                type(num_trials) is not int
                or num_trials < 2
                or value.shape[0] != num_trials
            ):
                raise ValueError(
                    f"parameter {variable_name!r} has an undeclared leading "
                    f"axis of length {value.shape[0]}; a trial axis requires "
                    "module.num_trials to match"
                )
            index_axis = 1
        else:
            raise ValueError(
                f"parameter {variable_name!r} rank {value.ndim} differs from "
                f"declared rank {declared_rank} with at most one trial axis"
            )
        return module, attr, id_attr, index_axis

    @staticmethod
    def _normalize_target_ids(
        target_ids: List[int] | torch.Tensor,
        id_tensor: torch.Tensor,
        *,
        variable_name: str,
    ) -> torch.Tensor:
        if id_tensor.dtype not in {torch.int32, torch.int64}:
            raise TypeError(
                f"ID field for {variable_name!r} must be int32 or int64, got "
                f"{id_tensor.dtype}"
            )
        if isinstance(target_ids, list):
            if any(type(value) is not int for value in target_ids):
                raise TypeError(
                    f"target_ids for {variable_name!r} must contain exact ints"
                )
            target_ids = torch.tensor(
                target_ids, dtype=id_tensor.dtype, device=id_tensor.device,
            )
        elif not isinstance(target_ids, torch.Tensor):
            raise TypeError(
                f"target_ids for {variable_name!r} must be a list or tensor"
            )
        if target_ids.ndim != 1 or target_ids.numel() == 0:
            raise ValueError(
                f"target_ids for {variable_name!r} must be a non-empty 1-D tensor"
            )
        if target_ids.device != id_tensor.device:
            raise ValueError(
                f"target_ids for {variable_name!r} are on {target_ids.device}, "
                f"expected {id_tensor.device}"
            )
        if target_ids.dtype != id_tensor.dtype:
            raise TypeError(
                f"target_ids for {variable_name!r} have dtype "
                f"{target_ids.dtype}, expected {id_tensor.dtype}"
            )
        if target_ids.layout is not torch.strided or not target_ids.is_contiguous():
            raise ValueError(
                f"target_ids for {variable_name!r} must be contiguous and strided"
            )
        if torch.unique(target_ids).numel() != target_ids.numel():
            raise ValueError(
                f"target_ids for {variable_name!r} contain duplicates"
            )
        return target_ids

    def _assert_field_is_key(self, ref: str, ctx: str) -> None:
        """Assert that ``ref`` (bare or ``module.field``) refers to a field
        declared with ``is_key=True``.  Raises ValueError otherwise."""
        bare = ref.split(".")[-1]
        if bare not in self.variable_map:
            raise ValueError(
                f"ParameterChangePlan Error: {ctx} '{ref}' does not "
                f"resolve to any declared field."
            )
        mod_inst, field_name, _ = self.variable_map[bare]
        field = mod_inst.get_tensor_schema(field_name)
        if field is None or not field.tensor.is_key:
            raise ValueError(
                f"ParameterChangePlan Error: {ctx} '{ref}' is not a key "
                f"field. PlanItem can only target fields declared with "
                f"is_key=True (1D unique integer keys)."
            )

    def _resolve_plan_item(self, item: PlanItem) -> None:
        from hydroforge.data.distributed import find_indices_in_torch

        module, attr, id_attr, index_axis = self._resolve_parameter(
            item.variable_name,
        )
        item._module = module
        item._attr_name = attr
        item._index_axis = index_axis
        target = getattr(module, attr)

        if item.target_ids is not None:
            lookup_attr = item.target_id_field or id_attr
            if not lookup_attr:
                raise ValueError(
                    f"ParameterChangePlan Error: Variable "
                    f"'{item.variable_name}' has no dim_coords and no "
                    f"target_id_field was provided; cannot resolve "
                    f"target_ids."
                )

            self._assert_field_is_key(
                lookup_attr,
                ctx=("target_id_field" if item.target_id_field
                     else f"dim_coords of '{item.variable_name}'"),
            )

            id_tensor = self._resolve_id_tensor(module, lookup_attr)
            if id_tensor is None:
                raise ValueError(
                    f"ParameterChangePlan Error: Cannot find ID tensor "
                    f"'{lookup_attr}' for variable "
                    f"'{item.variable_name}'."
                )
            if id_tensor.numel() != target.shape[index_axis]:
                raise ValueError(
                    f"ParameterChangePlan Error: ID field '{lookup_attr}' "
                    f"length {id_tensor.numel()} is not co-indexed with "
                    f"parameter '{item.variable_name}' axis {index_axis} "
                    f"length {target.shape[index_axis]}"
                )

            item.target_ids = self._normalize_target_ids(
                item.target_ids, id_tensor,
                variable_name=item.variable_name,
            ).detach().clone()
            indices = find_indices_in_torch(item.target_ids, id_tensor)
            if torch.any(indices < 0):
                raise ValueError(
                    f"ParameterChangePlan Error: Some target_ids for "
                    f"{item.variable_name} were not found in {lookup_attr}."
                )
            item._indices = indices
            item._resolved_id_field = lookup_attr

        value = item.target_value if item.is_set_value else item.delta
        self._validate_update_value(
            target, value, item._indices, variable_name=item.variable_name,
            index_axis=index_axis,
        )
        if isinstance(value, torch.Tensor):
            owned_value = value.detach().clone(
                memory_format=torch.preserve_format,
            )
            if item.is_set_value:
                item.target_value = owned_value
            else:
                item.delta = owned_value

        item._is_ready = True

    def _validate_set_conflict(self, item: PlanItem) -> None:
        if not item.is_set_value:
            return
        for existing in self._plans:
            if not (
                existing.is_set_value
                and existing.variable_name == item.variable_name
                and existing.start_time == item.start_time
            ):
                continue
            if existing._indices is None or item._indices is None:
                raise ValueError(
                    f"parameter {item.variable_name!r} has overlapping SET "
                    f"plans at {item.start_time}: a global SET conflicts with "
                    "every other SET"
                )
            if torch.isin(item._indices, existing._indices).any().item():
                raise ValueError(
                    f"parameter {item.variable_name!r} has overlapping SET "
                    f"target_ids at {item.start_time}"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def step_transaction(self):
        """Extend parameter-plan atomicity through one managed model step.

        Parameter values are updated before physics so kernels see the new
        state.  Their existing apply-time snapshots remain live until the
        managed step succeeds; a later compilation/physics failure restores
        both values and the plan cursor, making a retry exact.
        """

        if self._step_transaction_active:
            raise RuntimeError("nested parameter-plan transactions are forbidden")
        cursor = (
            self._sealed,
            self._last_time,
            self._next_plan_idx,
            tuple(
                ActivePlan(
                    active.item, active.steps_executed, active.executed_once,
                )
                for active in self._active_plans
            ),
        )
        self._step_transaction_active = True
        self._step_transaction_executed = False
        self._step_transaction_snapshots = []
        try:
            yield
        except BaseException as step_error:
            rollback_errors: list[BaseException] = []
            for module, attr, snapshot in reversed(
                self._step_transaction_snapshots,
            ):
                try:
                    self._restore_value(module, attr, snapshot)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            (
                self._sealed,
                self._last_time,
                self._next_plan_idx,
                active_plans,
            ) = cursor
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
            self._step_transaction_executed = False
            self._step_transaction_active = False

    def execute_parameter_change_plan(
        self, current_time: Union[datetime, cftime.datetime] | None,
    ) -> ParameterChangeEffect:
        """Apply one transactional plan step and describe capture consequences."""
        if self._step_transaction_active:
            if self._step_transaction_executed:
                raise RuntimeError(
                    "parameter plan may execute only once per managed step"
                )
            self._step_transaction_executed = True
        if current_time is None:
            if self._plans:
                raise ValueError(
                    "current_time is required when parameter-change plans exist"
                )
            return ParameterChangeEffect.UNCHANGED
        if not isinstance(current_time, (datetime, cftime.datetime)):
            raise TypeError(
                "parameter-plan current_time must be datetime or cftime.datetime"
            )
        require_calendar(
            current_time, self.owner.calendar,
            label="parameter-plan current_time",
        )
        if self._last_time is not None and current_time < self._last_time:
            raise ValueError("parameter-plan current_time must be non-decreasing")
        self._sealed = True
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
            active for active in active_plans
            if active.steps_executed < active.item.active_steps
        ]
        if not active_plans:
            self._next_plan_idx = next_plan_idx
            self._active_plans = active_plans
            self._last_time = current_time
            return ParameterChangeEffect.UNCHANGED

        grouped_plans: Dict[Tuple[int, str], List[ActivePlan]] = {}
        for active in active_plans:
            if not active.item._is_ready:
                raise RuntimeError(
                    f"parameter plan {active.item.variable_name!r} was not "
                    "resolved before execution"
                )
            key = (id(active.item._module), active.item._attr_name)
            grouped_plans.setdefault(key, []).append(active)

        snapshots: list[tuple[Any, str, Any]] = []
        for (_, attr), plans in grouped_plans.items():
            module = plans[0].item._module
            current = getattr(module, attr)
            snapshots.append((
                module, attr, self._snapshot_value(current, plans),
            ))
        if self._step_transaction_active:
            self._step_transaction_snapshots = snapshots
        try:
            for (_, attr), plans in grouped_plans.items():
                self._apply_grouped_changes(plans[0].item._module, attr, plans)
        except BaseException as apply_error:
            rollback_errors: list[BaseException] = []
            for module, attr, snapshot in reversed(snapshots):
                try:
                    self._restore_value(module, attr, snapshot)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                error = ResourceCleanupError(
                    "parameter change rollback",
                    (apply_error, *rollback_errors),
                )
                raise error from apply_error
            raise
        for active in active_plans:
            active.steps_executed += 1
            active.executed_once = True
        self._next_plan_idx = next_plan_idx
        self._active_plans = active_plans
        self._last_time = current_time
        return ParameterChangeEffect.UPDATED

    def add_parameter_change_plan(
        self,
        variable_name: str,
        start_time: Union[datetime, cftime.datetime],
        active_steps: int = 1,
        delta: Union[float, torch.Tensor] = 0.0,
        target_value: Optional[Union[float, torch.Tensor]] = None,
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
        target_id_field: Optional[str] = None,
    ) -> None:
        """Add a parameter change plan.

        ``target_id_field`` overrides the lookup field for ``target_ids``
        (must be co-indexed with the variable's default ``dim_coords``).
        """
        if self._sealed:
            raise RuntimeError(
                "parameter plans are immutable after model stepping begins"
            )
        if not isinstance(variable_name, str):
            raise TypeError("parameter plan variable_name must be a str")
        if not variable_name:
            raise ValueError("parameter plan variable_name must be non-empty")
        if target_id_field is not None and not isinstance(target_id_field, str):
            raise TypeError("target_id_field must be a str or None")
        if target_id_field == "":
            raise ValueError("target_id_field must be non-empty when provided")
        if not isinstance(start_time, (datetime, cftime.datetime)):
            raise TypeError(
                "parameter plan start_time must be datetime or cftime.datetime"
            )
        require_calendar(
            start_time, self.owner.calendar,
            label="parameter plan start_time",
        )
        if type(active_steps) is not int:
            raise TypeError("active_steps must be an exact int")
        if active_steps < 1:
            raise ValueError("active_steps must be >= 1")
        if target_value is not None and active_steps != 1:
            raise ValueError(
                "SET parameter plans are one-shot and require active_steps=1"
            )
        if target_value is not None and not (
            type(delta) is float and delta == 0.0
        ):
            raise ValueError(
                "SET parameter plans cannot also define delta"
            )

        item = PlanItem(
            variable_name=variable_name,
            start_time=start_time,
            active_steps=active_steps,
            delta=delta,
            target_value=target_value,
            target_ids=target_ids,
            target_id_field=target_id_field,
        )

        self._resolve_plan_item(item)
        self._validate_set_conflict(item)
        self._plans.append(item)
        self._plans.sort(key=lambda x: x.start_time)
        self._next_plan_idx = 0
        self._active_plans.clear()

    def get_variable(self, variable_name: str) -> Any:
        if variable_name not in self.variable_map:
            raise ValueError(f"Variable '{variable_name}' not found in model.")
        module, attr, _ = self.variable_map[variable_name]
        return getattr(module, attr)

    def set_variable_value(
        self,
        variable_name: str,
        value: Union[float, torch.Tensor],
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> ParameterChangeEffect:
        """Directly set the value of a variable for specific IDs immediately."""
        from hydroforge.data.distributed import find_indices_in_torch

        module, attr, id_attr, index_axis = self._resolve_parameter(variable_name)
        current_val = getattr(module, attr)

        if target_ids is None:
            self._validate_update_value(
                current_val, value, None, variable_name=variable_name,
            )
            self._apply_tensor_value(
                current_val, value, None, is_set=True,
            )
            return ParameterChangeEffect.UPDATED

        if not id_attr:
            raise ValueError(
                f"variable {variable_name!r} has no declared ID field"
            )
        self._assert_field_is_key(
            id_attr, ctx=f"dim_coords of '{variable_name}'",
        )
        id_tensor = self._resolve_id_tensor(module, id_attr)
        if id_tensor is None:
            raise ValueError(
                f"Cannot resolve ID tensor '{id_attr}' for variable "
                f"'{variable_name}', so target_ids cannot be used."
            )
        if id_tensor.numel() != current_val.shape[index_axis]:
            raise ValueError(
                f"ID field '{id_attr}' length {id_tensor.numel()} is not "
                f"co-indexed with parameter '{variable_name}' axis "
                f"{index_axis} length {current_val.shape[index_axis]}"
            )

        target_ids = self._normalize_target_ids(
            target_ids, id_tensor, variable_name=variable_name,
        )

        indices = find_indices_in_torch(target_ids, id_tensor)
        if torch.any(indices < 0):
            raise ValueError(f"Some target_ids for '{variable_name}' were not found in '{id_attr}'.")
        self._validate_update_value(
            current_val, value, indices, variable_name=variable_name,
            index_axis=index_axis,
        )
        self._apply_tensor_value(
            current_val, value, indices, is_set=True,
            index_axis=index_axis,
        )
        return ParameterChangeEffect.UPDATED

    def summarize_plan(self) -> None:
        """Emit the already-validated parameter-plan summary."""
        from hydroforge.contracts.events import emit

        if not self._plans:
            emit(
                self.owner, "info", "parameter.plan_summary",
                "No parameter change plans defined", rank=self.rank, plans=[],
            )
            return

        sorted_plans = sorted(self._plans, key=lambda x: x.start_time)

        rows = []
        for plan in sorted_plans:
            type_str = "SET" if plan.is_set_value else "ADD"
            val = plan.target_value if plan.is_set_value else plan.delta
            if isinstance(val, torch.Tensor):
                val_str = f"{val.item():.4g}" if val.numel() == 1 else "Tensor"
            else:
                val_str = f"{val:.4g}"
            dur_str = f"{plan.active_steps}" if plan.is_incremental else "-"
            if plan.target_ids is None:
                target_str = "ALL"
            else:
                count = len(plan.target_ids)
                id_attr_name = "IDs"
                if plan._resolved_id_field:
                    id_attr_name = plan._resolved_id_field
                elif plan.target_id_field:
                    id_attr_name = plan.target_id_field
                elif plan.variable_name in self.variable_map:
                    _, _, id_attr = self.variable_map[plan.variable_name]
                    if id_attr:
                        id_attr_name = id_attr
                if count <= 5:
                    ids_list = (
                        plan.target_ids.tolist()
                        if isinstance(plan.target_ids, torch.Tensor)
                        else plan.target_ids
                    )
                    target_str = f"{str(ids_list)} ({id_attr_name})"
                else:
                    target_str = f"{count} {id_attr_name}"
            rows.append({
                "time": str(plan.start_time), "variable": plan.variable_name,
                "type": type_str, "value": val_str, "steps": dur_str,
                "target": target_str,
            })

        emit(
            self.owner, "info", "parameter.plan_summary",
            "Parameter change plan", rank=self.rank, plans=rows,
        )

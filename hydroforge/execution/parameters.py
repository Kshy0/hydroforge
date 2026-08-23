# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Trusted execution of construction-time compiled parameter changes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import cftime
import torch

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.contracts.parameters import ParameterValue

if TYPE_CHECKING:
    from hydroforge.compiler.parameters import _ParameterChangePlan


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One runtime binding of an already compiled parameter instruction."""

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


class ParameterPlanRuntime:
    """Apply rank-local plans compiled by ``ParameterSemanticCompiler``."""

    def __init__(
        self,
        owner: Any,
        plans: tuple[_ParameterChangePlan, ...],
    ) -> None:
        self.owner = owner
        self._plans = tuple(self._bind(item) for item in plans)
        self._active_plans: list[ActivePlan] = []
        self._next_plan_idx = 0
        self._step_transaction_snapshots: list[tuple[Any, str, Any]] = []

    def _bind(self, item: _ParameterChangePlan) -> PlanItem:
        module = self.owner._modules[item.module_name]
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
        self, module: Any, attr: str, plans: list[ActivePlan],
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
        value: torch.Tensor, plans: list[ActivePlan],
    ) -> _TensorSnapshot:
        indexed = [active.item.indices for active in plans]
        if all(indices is not None for indices in indexed):
            index_axis = plans[0].item.index_axis
            indices = torch.unique(torch.cat([
                item for item in indexed if item is not None
            ]))
            values = value.index_select(index_axis, indices).detach().clone(
                memory_format=torch.preserve_format,
            )
            return _TensorSnapshot(values, indices, index_axis)
        return _TensorSnapshot(
            value.detach().clone(memory_format=torch.preserve_format),
            None,
            0,
        )

    @staticmethod
    def _restore_value(
        module: Any, attr: str, snapshot: _TensorSnapshot,
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

    @contextmanager
    def step_transaction(self):
        """Keep parameter application atomic with one managed model step."""

        cursor = (
            self._next_plan_idx,
            tuple(
                ActivePlan(
                    active.item,
                    active.steps_executed,
                    active.executed_once,
                )
                for active in self._active_plans
            ),
        )
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
        self, current_time: datetime | cftime.datetime | None,
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
            active for active in active_plans
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

        snapshots: list[tuple[Any, str, Any]] = []
        for (_, attr), plans in grouped.items():
            module = plans[0].item.module
            current = getattr(module, attr)
            snapshots.append((
                module,
                attr,
                self._snapshot_value(current, plans),
            ))
        self._step_transaction_snapshots = snapshots
        try:
            for (_, attr), plans in grouped.items():
                self._apply_grouped_changes(plans[0].item.module, attr, plans)
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
        return ParameterChangeEffect.UPDATED


__all__: list[str] = []

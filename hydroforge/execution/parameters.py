# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Trusted execution of construction-time compiled parameter changes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import cftime
import torch

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.contracts.parameters import ParameterValue
from hydroforge.model.tensors import _PARAMETER_TENSOR_READS

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
        self.dependencies: Mapping[str, tuple[str, ...]] = MappingProxyType({})
        self._derived: dict[str, tuple[Any, str, cached_property]] = {}
        self._dependency_revision: int | None = None
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
        revision = self.owner._execution.structural_revision
        if self._dependency_revision == revision:
            return
        fields = {}
        schemas = {}
        candidates = {}
        for module_name in self.owner.opened_modules:
            module = self.owner._modules[module_name]
            for field in module.tensor_schema():
                qualified = f"{module_name}.{field.name}"
                fields[id(module), field.name] = qualified
                schemas[qualified] = (module, field)
                if (
                    field.computed
                    and field.tensor.category == "derived_param"
                    and module._is_tensor_field_active(field)
                ):
                    descriptor = getattr(type(module), field.name)
                    if not isinstance(descriptor, cached_property):
                        raise ValueError(
                            f"{qualified}: parameter refresh requires a cached derived parameter"
                        )
                    if isinstance(module.__dict__.get(field.name), torch.Tensor):
                        candidates[qualified] = (module, field.name, descriptor)
            for name in module._reference_index_fields(
                opened_modules=module.opened_modules,
                field_demand=module._field_demand,
            ):
                qualified = f"{module_name}.{name}"
                fields[id(module), name] = qualified
                schemas[qualified] = (
                    module,
                    module._get_tensor_schema(
                        name,
                        opened_modules=module.opened_modules,
                        field_demand=module._field_demand,
                    ),
                )
        dependencies = {}
        with torch.inference_mode():
            for qualified, (module, name, descriptor) in candidates.items():
                reads: set[str] = set()
                token = _PARAMETER_TENSOR_READS.set((fields, reads))
                try:
                    # Invoke only the formula: reading the result buffer here
                    # would falsely record a self-dependency.
                    descriptor.func(module)
                except Exception as error:
                    raise ValueError(
                        f"cannot record parameter dependencies for {qualified}: {error}"
                    ) from error
                finally:
                    _PARAMETER_TENSOR_READS.reset(token)
                for dependency in reads:
                    source, schema = schemas[dependency]
                    if schema.tensor.category not in {
                        "param",
                        "derived_param",
                        "topology",
                    }:
                        raise ValueError(
                            f"{qualified}: derived parameters cannot depend on runtime field {dependency}"
                        )
                    if not isinstance(getattr(source, schema.name), torch.Tensor):
                        raise ValueError(
                            f"{qualified}: dependency {dependency} must remain resident"
                        )
                    if (
                        schema.tensor.category == "derived_param"
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
        # Publish only a complete graph; a failed discovery remains retryable.
        self._derived = {name: candidates[name] for name in order if name in candidates}
        self.dependencies = MappingProxyType(dependencies)
        self._dependency_revision = revision

    def _refresh_derived_parameters(self, changed: set[str]) -> None:
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
                changed.add(qualified)
            for module_name in self.owner.opened_modules:
                self.owner._modules[module_name].validate_parameters()

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
            indices = torch.unique(
                torch.cat([item for item in indexed if item is not None])
            )
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
            for (_, attr), plans in grouped.items():
                self._apply_grouped_changes(plans[0].item.module, attr, plans)
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
            active.executed_once = True
        self._next_plan_idx = next_plan_idx
        self._active_plans = active_plans
        return ParameterChangeEffect.UPDATED


__all__: list[str] = []

"""Cold-path compiler for one concrete model specialization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from hydroforge.compiler.plan import (
    FieldNamespace,
    FieldOwner,
    ExecutionPlan,
    KernelPlan,
    ModelPlan,
    ModulePlan,
    RuntimePlan,
    StatisticsPlan,
)
from hydroforge.contracts.kernel_field import KernelField
from hydroforge.contracts.fields import tensor_is_active


class ModelCompiler:
    """Compile declarative modules into an immutable execution contract."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def compile(self) -> ModelPlan:
        model = self.model
        missing = set(model.opened_modules).difference(model._modules)
        if missing:
            raise RuntimeError(
                "opened modules were not constructed: " + ", ".join(sorted(missing))
            )
        order = tuple(model.opened_modules)
        dependencies = {
            name: tuple(model.module_list[name].dependencies)
            for name in order
        }
        runtime = model._execution
        fields = FieldNamespace(self._field_owners())
        aggregator = model._statistics.aggregator
        variables = (
            () if aggregator is None else tuple(sorted(aggregator._variables))
        )
        return ModelPlan(
            modules=ModulePlan(order=order, dependencies=dependencies),
            capabilities=frozenset(model._capabilities),
            fields=fields,
            runtime=RuntimePlan(
                backend=runtime.backend,
                device=runtime.device,
                capture_mode=runtime.capture_mode,
            ),
            kernels=KernelPlan(fields=fields),
            execution=ExecutionPlan(
                policy_count=len(model._execution.step_policies),
            ),
            statistics=StatisticsPlan(
                enabled=aggregator is not None,
                variables=variables,
            ),
        )

    def _field_owners(self) -> dict[str, tuple[FieldOwner, ...]]:
        model = self.model
        index: dict[str, list[FieldOwner]] = {}
        for module_name, module in model._modules.items():
            fields = {
                name
                for name, schema in module.field_schema_map().items()
                if (
                    schema.tensor is None
                    or (
                        not schema.tensor.expression
                        and tensor_is_active(
                            schema.tensor, getattr(model, "opened_modules", ()),
                        )
                    )
                )
            } | set(module.get_reference_index_fields()) | {
                name
                for name in type(module).model_fields
                if name not in model.module_list
            }
            for field_name in fields:
                index.setdefault(field_name, []).append(FieldOwner(
                    module_name=module_name,
                    field_name=field_name,
                    owner=module,
                ))
        for field_name in model.__class__.model_fields:
            if field_name not in model.module_list:
                index.setdefault(field_name, []).append(FieldOwner(
                    module_name="model",
                    field_name=field_name,
                    owner=model,
                ))
                value = getattr(model, field_name)
                if (
                    isinstance(value, BaseModel)
                    and type(value).model_config.get("frozen") is True
                ):
                    for nested_name in type(value).model_fields:
                        index.setdefault(nested_name, []).append(FieldOwner(
                            module_name=f"model.{field_name}",
                            field_name=nested_name,
                            owner=value,
                        ))
        for cls in reversed(type(model).__mro__):
            for field_name, descriptor in vars(cls).items():
                if not isinstance(descriptor, KernelField):
                    continue
                index.setdefault(field_name, []).append(FieldOwner(
                    module_name="model",
                    field_name=field_name,
                    owner=model,
                ))
        return {
            field_name: tuple(owners)
            for field_name, owners in index.items()
        }

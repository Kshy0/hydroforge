"""Static resolution of kernel parameters against one model plan.

A parameter a call site omits binds by name: a step field, an optional value
or buffer whose feature is off, ``ensemble_size``, a compile-time source, a
``batched_<field>`` flag or the field ``<name>`` (without a ``_ptr`` suffix)
of its one owner.  The plan decides each rule and every value it can know;
execution only reads the remaining values from the constructed model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from hydroforge.compiler.fields import BindingSource
from hydroforge.compiler.selection import resolve_block_size
from hydroforge.contracts.fields import concrete_tensor_dtype
from hydroforge.contracts.step_fields import StepField
from hydroforge.kernels.spec import (
    ConfigValue,
    KernelSpec,
    KernelWorkspace,
    LiteralValue,
    ModuleEnabled,
    ModuleFlag,
    OptionCode,
    OutputRequested,
)

if TYPE_CHECKING:
    from hydroforge.compiler.plan import ModelPlan


@dataclass(frozen=True, slots=True)
class Fixed:
    """A value the plan knows; ``source`` and ``owner`` describe its origin."""

    value: Any
    source: str
    owner: str | None


@dataclass(frozen=True, slots=True)
class StepFieldValue:
    field: StepField


@dataclass(frozen=True, slots=True)
class FieldValue:
    """The field of its one owner; an ``Enum`` binds its value.

    Buffer virtuals among ``owners`` bind only once materialized.  An
    ``optional`` field binds ``None`` when no owner holds it.
    """

    field: str
    owners: tuple[BindingSource, ...]
    optional: bool = False


@dataclass(frozen=True, slots=True)
class Batched:
    """Whether the field of its one owner has a leading member axis."""

    field: str
    owners: tuple[BindingSource, ...]


@dataclass(frozen=True, slots=True)
class ModuleFlagValue:
    """One exact bool field of a constructed module."""

    parameter: str
    module: str
    field: str


@dataclass(frozen=True, slots=True)
class Gated:
    """``disabled`` while the bool ``feature`` parameter is off, else ``source``."""

    feature: str
    disabled: Any
    source: Source


@dataclass(frozen=True, slots=True)
class Unresolved:
    """A binding rule that fails whenever the parameter is resolved."""

    error: type[Exception]
    message: str


Source = (
    Fixed
    | StepFieldValue
    | FieldValue
    | Batched
    | ModuleFlagValue
    | Gated
    | Unresolved
    | KernelWorkspace
)


@dataclass(frozen=True, slots=True)
class KernelBinding:
    """How every parameter of one spec binds in one model.

    ``buffer_dtypes`` holds the declared dtype of each buffer, ``None`` where
    the bound tensor's own dtype applies.
    """

    sources: Mapping[str, Source]
    buffer_dtypes: Mapping[str, torch.dtype | None | Unresolved]
    block_size: int


class KernelBindingPlan:
    """The kernel bindings of one model plan, compiled once per spec."""

    def __init__(self, plan: ModelPlan) -> None:
        self.plan = plan
        self._bindings: dict[int, tuple[KernelSpec, KernelBinding]] = {}

    def bind(self, spec: KernelSpec) -> KernelBinding:
        cached = self._bindings.get(id(spec))
        if cached is not None:
            return cached[1]
        plan = self.plan
        binding = KernelBinding(
            MappingProxyType(
                {name: self._source(spec, name) for name in spec.parameters}
            ),
            MappingProxyType(
                {name: self._declared_dtype(spec, name) for name in spec.buffers}
            ),
            resolve_block_size(
                plan.backend,
                plan.backend_requirement,
                plan.block_size,
                kernel=spec.block_sizes.get(plan.backend.name),
            ),
        )
        self._bindings[id(spec)] = (spec, binding)
        return binding

    def _source(self, spec: KernelSpec, name: str) -> Source:
        if name in spec.workspace and spec.optional.get(name) is None:
            return spec.workspace[name]
        if name in spec.step_fields:
            return StepFieldValue(spec.step_fields[name])
        if name in spec.optional_values:
            feature, disabled = spec.optional_values[name]
            return self._gate(spec, feature, disabled, name)
        if name in spec.optional:
            feature = spec.optional[name]
            if feature is None:
                field = name.removesuffix("_ptr")
                return FieldValue(
                    field, self.plan.fields.binding.get(field, ()), optional=True
                )
            return self._gate(spec, feature, None, name)
        return self._value(spec, name)

    def _gate(self, spec: KernelSpec, feature: str, disabled: Any, name: str) -> Source:
        enabled = self._compile_time(spec, feature)
        if isinstance(enabled, Fixed):
            if enabled.value:
                return self._value(spec, name)
            return Fixed(disabled, "optional", feature)
        if isinstance(enabled, Unresolved):
            return enabled
        return Gated(feature, disabled, self._value(spec, name))

    def _value(self, spec: KernelSpec, name: str) -> Source:
        if name in spec.workspace:
            return spec.workspace[name]
        plan = self.plan
        if name == "ensemble_size":
            members = plan.local_ensemble_size
            return Fixed(1 if members is None else members, "model_config", "model")
        if name in spec.compile_time_sources:
            return self._compile_time(spec, name)
        field = name.removesuffix("_ptr")
        if field.startswith("batched_"):
            source = field.removeprefix("batched_")
            owners = plan.fields.binding.get(source, ())
            if not owners:
                declared = [
                    module
                    for module, module_spec in plan.spec.modules.items()
                    if source in module_spec.fields
                ]
                if len(declared) == 1 and declared[0] not in plan.modules:
                    return Fixed(False, "batched", declared[0])
            return Batched(source, owners)
        return FieldValue(field, plan.fields.binding.get(field, ()))

    def _compile_time(self, spec: KernelSpec, name: str) -> Source:
        plan = self.plan
        source = spec.compile_time_sources.get(name)
        if source is None:
            return Unresolved(
                KeyError,
                f"kernel compile-time parameter {name!r} has no explicit "
                "compile-time source",
            )
        if isinstance(source, (ConfigValue, OptionCode, LiteralValue)):
            if isinstance(source, ConfigValue):
                value = plan.options.value(source.path)
            elif isinstance(source, OptionCode):
                value = plan.options.option_code(source.path)
            else:
                value = source.value
            return Fixed(value, "compile_time", name)
        module_spec = plan.spec.modules.get(source.module)
        if module_spec is None:
            return Unresolved(
                KeyError,
                f"kernel feature {name!r} references unknown model module "
                f"{source.module!r}",
            )
        if isinstance(source, ModuleEnabled):
            return Fixed(source.module in plan.modules, "compile_time", name)
        if isinstance(source, ModuleFlag):
            if source.module not in plan.modules:
                # Closed optional modules resolve field features to false.
                return Fixed(False, "compile_time", name)
            if source.field not in module_spec.fields and not hasattr(
                module_spec.module_type, source.field
            ):
                return Unresolved(
                    KeyError,
                    f"kernel feature {name!r} references unknown field "
                    f"{source.module}.{source.field}",
                )
            return ModuleFlagValue(name, source.module, source.field)
        assert isinstance(source, OutputRequested)
        schema = module_spec.tensor_fields.get(source.field)
        if source.field not in module_spec.reference_indices and (
            schema is None or schema.tensor.expression
        ):
            return Unresolved(
                KeyError,
                f"kernel feature {name!r} references unknown materialized tensor "
                f"field {source.module}.{source.field}",
            )
        if source.module not in plan.modules:
            return Fixed(False, "compile_time", name)
        binding = plan.fields.modules[source.module]
        return Fixed(
            source.field in binding.active and source.field in binding.observed,
            "compile_time",
            name,
        )

    def _concrete(self, kind: str) -> torch.dtype:
        plan = self.plan
        return concrete_tensor_dtype(kind, plan.dtype, plan.mixed_precision)

    def _declared_dtype(
        self, spec: KernelSpec, name: str
    ) -> torch.dtype | None | Unresolved:
        """The dtype the model declares for one buffer, if any."""

        plan = self.plan
        if name in spec.workspace:
            dtype = spec.workspace[name].dtype
            return plan.dtype if dtype == "precision" else getattr(torch, dtype)
        if name in spec.step_fields:
            dtype = spec.step_fields[name].dtype
            return plan.dtype if dtype == "precision" else getattr(torch, dtype)
        field = name.removesuffix("_ptr")
        optional = name in spec.optional
        typed = []
        for source in plan.fields.binding.get(field, ()):
            module_spec = plan.spec.modules.get(source.owner)
            if module_spec is None:
                continue
            schema = module_spec.tensor_fields.get(field)
            metadata = (
                schema.tensor
                if schema is not None
                else plan.fields.modules[source.owner].reference_indices.get(field)
            )
            if metadata is not None:
                typed.append((source.owner, self._concrete(metadata.dtype)))
        if len(typed) == 1:
            return typed[0][1]
        if len(typed) > 1:
            return Unresolved(
                ValueError,
                f"buffer {name!r} has ambiguous dtype declarations in "
                f"{[module for module, _dtype in typed]}",
            )
        feature = spec.optional.get(name)
        if feature is None:
            return self._global_dtype(name, field) if optional else None
        source = spec.compile_time_sources.get(feature)
        if not isinstance(source, (ModuleEnabled, ModuleFlag, OutputRequested)):
            return None
        module_spec = plan.spec.modules.get(source.module)
        if module_spec is None:
            return self._global_dtype(name, field) if optional else None
        if field in module_spec.reference_indices:
            return self._concrete("idx")
        schema = module_spec.tensor_fields.get(field)
        if schema is None:
            return self._global_dtype(name, field) if optional else None
        return self._concrete(schema.tensor.dtype)

    def _global_dtype(self, name: str, field: str) -> torch.dtype | None | Unresolved:
        """The one declaration of ``field`` among every module of the model."""

        declared = []
        for module, module_spec in self.plan.spec.modules.items():
            if field in module_spec.reference_indices:
                declared.append((module, "idx"))
                continue
            schema = module_spec.tensor_fields.get(field)
            if schema is not None and not schema.tensor.expression:
                declared.append((module, schema.tensor.dtype))
        if len(declared) > 1:
            return Unresolved(
                ValueError,
                f"optional buffer {name!r} has ambiguous declarations in "
                f"{[module for module, _kind in declared]}",
            )
        return self._concrete(declared[0][1]) if declared else None


__all__ = ["KernelBinding", "KernelBindingPlan"]

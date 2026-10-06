# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Frozen declaration specs built once per module and model class."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

from hydroforge.contracts.fields import TensorMetadata
from hydroforge.contracts.runtime import BackendRequirement, ModuleRequirement
from hydroforge.contracts.step_fields import StepFieldProvider
from hydroforge.core.events import EventSink
from hydroforge.core.expr import Expression

DimensionToken: TypeAlias = str | int


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One declared Pydantic or computed field of a module class."""

    module_name: str
    name: str
    tensor: TensorMetadata | None
    computed: bool
    required: bool
    excluded: bool
    annotation: Any
    description: str

    @property
    def shape(self) -> tuple[DimensionToken, ...]:
        return () if self.tensor is None else self.tensor.shape

    @property
    def dtype(self) -> str:
        return "" if self.tensor is None else self.tensor.dtype

    @property
    def category(self) -> str | None:
        return None if self.tensor is None else self.tensor.category

    @property
    def output(self) -> str | None:
        return None if self.tensor is None else self.tensor.output

    @property
    def selects(self) -> str | None:
        return None if self.tensor is None else self.tensor.selects


@dataclass(frozen=True, slots=True)
class ReferenceIndexSpec:
    """One ``ReferenceIndexField``; its tensor metadata depends on selection."""

    name: str
    reference: str
    inverse: bool
    device: bool


@dataclass(frozen=True, slots=True)
class ModuleRefSpec:
    """One typed sibling-module reference declared by ``module_ref``."""

    name: str
    module_type: type
    optional: bool

    def validate_type(self, actual: type | None) -> None:
        """References are nominally typed: subclasses retain the declared API."""

        if actual is None:
            if not self.optional:
                raise ValueError(f"Missing required module reference {self.name!r}")
        elif not issubclass(actual, self.module_type):
            raise TypeError(
                f"Module reference {self.name!r} requires a subclass of "
                f"{self.module_type.__name__}, got {actual.__name__}"
            )


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """Complete class-level declaration of one module."""

    module_type: type
    name: str
    description: str
    conflicts: tuple[str, ...]
    fields: Mapping[str, FieldSpec]
    tensor_fields: Mapping[str, FieldSpec]
    reference_indices: Mapping[str, ReferenceIndexSpec]
    references: Mapping[str, ModuleRefSpec]
    required_modules: tuple[str, ...]
    kernel_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StepFieldPlan:
    """Validated host providers and device expressions of outer-step fields.

    ``expressions`` is ordered so that every expression follows its
    dependencies.
    """

    providers: Mapping[str, StepFieldProvider]
    expressions: Mapping[str, Expression]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Complete class-level declaration of one model.

    ``authored_fields`` are the instance fields a model class adds to the
    framework declaration; kernels may bind them by name.
    """

    model_type: type
    modules: Mapping[str, ModuleSpec]
    required_modules: tuple[str, ...]
    backend_requirements: Mapping[str, BackendRequirement]
    module_requirements: Mapping[str, ModuleRequirement]
    partition_key: str | None
    partition_group: str
    step_fields: StepFieldPlan
    kernel_fields: tuple[str, ...]
    authored_fields: tuple[str, ...]


def _resolve_dimension(
    dimensions: Mapping[DimensionToken, Any],
    dimension: DimensionToken,
) -> Any:
    """Resolve a logical dimension, including ``module.attribute`` tokens."""
    try:
        return dimensions[dimension]
    except KeyError:
        if isinstance(dimension, str) and "." in dimension:
            try:
                return dimensions[dimension.rsplit(".", 1)[1]]
            except KeyError:
                raise KeyError(dimension) from None
        raise


@dataclass(frozen=True, slots=True)
class ModuleSchema:
    """Declared fields grouped by their owning module."""

    modules: Mapping[str, tuple[FieldSpec, ...]]

    def resolve_dimensions(
        self,
        dimensions: Mapping[DimensionToken, str],
        *,
        include: Callable[[FieldSpec], bool] | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        """Translate logical tensor shapes into consumer-specific dimensions."""
        resolved: dict[str, dict[str, tuple[str, ...]]] = {}
        for module_name, fields in self.modules.items():
            module_fields: dict[str, tuple[str, ...]] = {}
            for field_spec in fields:
                if field_spec.tensor is None:
                    continue
                if include is not None and not include(field_spec):
                    continue
                try:
                    module_fields[field_spec.name] = tuple(
                        str(dimension)
                        if isinstance(dimension, int)
                        else _resolve_dimension(dimensions, dimension)
                        for dimension in field_spec.shape
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"{module_name}.{field_spec.name} uses unresolved dimension "
                        f"{exc.args[0]!r}"
                    ) from exc
            resolved[module_name] = module_fields
        return resolved

    def fields(self, module_name: str) -> tuple[FieldSpec, ...]:
        """Return fields owned by ``module_name``."""
        try:
            return self.modules[module_name]
        except KeyError as exc:
            raise KeyError(f"Module {module_name!r} is absent from schema") from exc


@dataclass(frozen=True, slots=True)
class ReferenceTarget:
    """The resolved local target field of one reference-index source."""

    module: str
    field: str

    @property
    def qualified(self) -> str:
        return f"{self.module}.{self.field}"


@dataclass(frozen=True, slots=True)
class ModuleBindingPlan:
    """Static field selection of one module inside one model specialization."""

    active: frozenset[str]
    required_output: frozenset[str]
    observed: frozenset[str]
    reference_targets: Mapping[str, ReferenceTarget]
    reference_indices: Mapping[str, TensorMetadata]
    batched_forcing: frozenset[str]


@dataclass(frozen=True, slots=True)
class ModuleBinding:
    """Everything a module construction receives besides its field values.

    ``references`` holds the already constructed sibling modules declared by
    ``module_ref`` (``None`` when closed); ``prepared`` marks a payload that
    already passed ``prepare_module_input``; ``defaults`` carries default
    values a payload view already evaluated for its siblings.
    """

    plan: ModuleBindingPlan
    references: Mapping[str, Any]
    event_sink: EventSink
    prepared: bool = False
    defaults: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

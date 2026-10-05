"""Field selection of one model specialization: demand, activation and names."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
from pydantic import BaseModel

from hydroforge.compiler.partition import compile_partition
from hydroforge.compiler.selection import Selection
from hydroforge.contracts.conditions import conditions_satisfied, module_conditions
from hydroforge.contracts.fields import (
    FieldDemandPlan,
    PartitionSchema,
    TensorMetadata,
    concrete_tensor_dtype,
    tensor_is_active,
)
from hydroforge.core.expr import ExpressionSource, TensorSource, parse_value_source
from hydroforge.declare.spec import (
    FieldSpec,
    ModelSpec,
    ModuleBindingPlan,
    ModuleSpec,
    ReferenceIndexSpec,
    ReferenceTarget,
)

if TYPE_CHECKING:
    from hydroforge.declare.model import ModelDeclaration

_Entry = TypeVar("_Entry")


class FieldNameResolver(Generic[_Entry]):
    """The one qualified/bare-name precedence rule of model fields.

    Qualified names always resolve. A bare name resolves when exactly one
    candidate declares it, or when exactly one expression virtual does.
    """

    def __init__(self) -> None:
        self.entries: dict[str, _Entry] = {}
        self.virtual_names: set[str] = set()
        self.ambiguous_names: set[str] = set()

    def install(
        self,
        module_name: str,
        field_name: str,
        entry: _Entry,
        *,
        expression_virtual: bool,
    ) -> None:
        self.entries[f"{module_name}.{field_name}"] = entry
        if expression_virtual:
            if field_name in self.virtual_names:
                self.entries.pop(field_name, None)
                self.ambiguous_names.add(field_name)
            else:
                self.entries[field_name] = entry
                self.virtual_names.add(field_name)
                self.ambiguous_names.discard(field_name)
        elif (
            field_name not in self.virtual_names
            and field_name not in self.ambiguous_names
        ):
            if field_name in self.entries:
                self.entries.pop(field_name)
                self.ambiguous_names.add(field_name)
            else:
                self.entries[field_name] = entry


@dataclass(frozen=True, slots=True)
class FieldEntry:
    """One tensor field or active reference index of an opened module.

    ``tensor`` is the declared metadata, or the metadata a reference index
    resolves against its selected target.
    """

    module: str
    name: str
    spec: FieldSpec | ReferenceIndexSpec
    tensor: TensorMetadata
    active: bool

    @property
    def qualified(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def reference_index(self) -> bool:
        return isinstance(self.spec, ReferenceIndexSpec)

    @property
    def computed(self) -> bool:
        return self.reference_index or self.spec.computed

    @property
    def excluded(self) -> bool:
        return not self.reference_index and self.spec.excluded

    @property
    def description(self) -> str:
        return (
            f"Derived local index {self.name}"
            if self.reference_index
            else self.spec.description
        )

    @property
    def expression_virtual(self) -> bool:
        return self.tensor.category == "virtual" and bool(self.tensor.expression)


@dataclass(frozen=True, slots=True)
class InputFieldSpec:
    """One external input: the first active declaration of its name.

    ``dtype`` is the exact runtime dtype of a tensor field and ``None`` for
    a scalar field.
    """

    field: FieldSpec
    dtype: torch.dtype | None


@dataclass(frozen=True, slots=True)
class BindingSource:
    """One owner that a kernel ABI name can bind to.

    ``owner`` is a module name, ``"model"``, or ``"model.<field>"`` for a
    field of a frozen model-authored record. A ``lazy`` field is a buffer
    virtual that binds only once initialization has materialized it.
    """

    owner: str
    lazy: bool = False


@dataclass(frozen=True, slots=True)
class FieldPlan:
    """Every field-level decision of one model specialization."""

    entries: Mapping[str, FieldEntry]
    names: Mapping[str, FieldEntry]
    demand: FieldDemandPlan
    modules: Mapping[str, ModuleBindingPlan]
    partition: PartitionSchema
    variable_groups: Mapping[str, str]
    inverse_sources: frozenset[str]
    inputs: Mapping[str, InputFieldSpec]
    binding: Mapping[str, tuple[BindingSource, ...]]


def _field_is_active(
    field: FieldSpec,
    opened_modules: tuple[str, ...] | None,
    field_demand: FieldDemandPlan | None,
    conditions: Mapping[str, bool] | None = None,
) -> bool:
    """Apply tensor residency; without a demand plan output fields are candidates."""

    return opened_modules is None or tensor_is_active(
        field.tensor,
        opened_modules,
        conditions=conditions,
        output_required=(
            field_demand is None
            or field_demand.is_required(field.module_name, field.name)
        ),
    )


def active_reference_indices(
    spec: ModuleSpec,
    opened_modules: tuple[str, ...] | None,
    field_demand: FieldDemandPlan | None = None,
    conditions: Mapping[str, bool] | None = None,
) -> tuple[ReferenceIndexSpec, ...]:
    """Return reference indices whose source field is active."""

    return tuple(
        index
        for index in spec.reference_indices.values()
        if _field_is_active(
            spec.tensor_fields[index.reference],
            opened_modules,
            field_demand,
            conditions,
        )
    )


def reference_target(
    spec: ModuleSpec,
    reference: str,
    *,
    opened_modules: tuple[str, ...] | None = None,
    field_demand: FieldDemandPlan | None = None,
    module_specs: Mapping[str, ModuleSpec] | None = None,
    conditions: Mapping[str, bool] | None = None,
) -> FieldSpec:
    """Resolve the one active target field of a reference-index source."""

    target_name = spec.tensor_fields[reference].tensor.references
    owners = {spec.name: spec}
    owners.update(
        {
            name: (
                declaration.module_type.spec()
                if module_specs is None
                else module_specs[name]
            )
            for name, declaration in spec.references.items()
            if opened_modules is None or name in opened_modules
        }
    )
    parts = target_name.split(".")
    if len(parts) > 1:
        owner = owners.get(parts[-2])
        owners = {} if owner is None else {owner.name: owner}
    candidates = [
        target
        for owner in owners.values()
        if (target := owner.tensor_fields.get(parts[-1])) is not None
        and _field_is_active(target, opened_modules, field_demand, conditions)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Reference target {target_name!r} for "
            f"{spec.name}.{reference} resolves to "
            f"{len(candidates)} opened tensor fields; qualify the "
            "target with its module name or provide opened_modules"
        )
    if not candidates[0].tensor.is_coordinate:
        raise ValueError(f"Reference target {target_name!r} must be a coordinate key")
    return candidates[0]


def reference_index_metadata(
    spec: ModuleSpec,
    index: ReferenceIndexSpec,
    *,
    opened_modules: tuple[str, ...] | None = None,
    field_demand: FieldDemandPlan | None = None,
    module_specs: Mapping[str, ModuleSpec] | None = None,
    conditions: Mapping[str, bool] | None = None,
) -> TensorMetadata:
    """Resolve the tensor metadata of one derived reference index."""

    source = spec.tensor_fields[index.reference]
    axis = (
        reference_target(
            spec,
            index.reference,
            opened_modules=opened_modules,
            field_demand=field_demand,
            module_specs=module_specs,
            conditions=conditions,
        )
        if index.inverse
        else source
    )
    shape = axis.tensor.shape
    coordinate = axis.name if axis.tensor.is_coordinate else axis.tensor.dim_coords
    if axis.module_name != spec.name:
        shape = tuple(
            f"{axis.module_name}.{dimension}"
            if isinstance(dimension, str) and "." not in dimension
            else dimension
            for dimension in shape
        )
        if coordinate is not None and "." not in coordinate:
            coordinate = f"{axis.module_name}.{coordinate}"
    return TensorMetadata(
        shape=shape,
        dtype="idx",
        dim_coords=coordinate,
        category="topology",
        mode="device" if index.device else "cpu",
        output="disabled",
        depends_on=source.tensor.depends_on,
        required_by=source.tensor.required_by,
    )


def bind_module_fields(
    spec: ModuleSpec,
    opened_modules: tuple[str, ...],
    field_demand: FieldDemandPlan,
    batched_forcing: Iterable[str] = (),
    *,
    module_specs: Mapping[str, ModuleSpec] | None = None,
    conditions: Mapping[str, bool] | None = None,
) -> ModuleBindingPlan:
    """Select the active fields and reference-index targets of one module."""

    active = {
        name
        for name, field in spec.tensor_fields.items()
        if _field_is_active(field, opened_modules, field_demand, conditions)
    }
    targets: dict[str, ReferenceTarget] = {}
    indices: dict[str, TensorMetadata] = {}
    for index in active_reference_indices(
        spec, opened_modules, field_demand, conditions
    ):
        active.add(index.name)
        target = reference_target(
            spec,
            index.reference,
            opened_modules=opened_modules,
            field_demand=field_demand,
            module_specs=module_specs,
            conditions=conditions,
        )
        targets[index.reference] = ReferenceTarget(target.module_name, target.name)
        indices[index.name] = reference_index_metadata(
            spec,
            index,
            opened_modules=opened_modules,
            field_demand=field_demand,
            module_specs=module_specs,
            conditions=conditions,
        )
    return ModuleBindingPlan(
        active=frozenset(active),
        required_output=field_demand.required_for(spec.name),
        observed=field_demand.observed_for(spec.name),
        reference_targets=MappingProxyType(targets),
        reference_indices=MappingProxyType(indices),
        batched_forcing=frozenset(batched_forcing),
    )


def source_dependencies(source: Any) -> tuple[str, ...]:
    """Return the field names one parsed value source reads."""

    if isinstance(source, TensorSource):
        return (source.name,)
    if isinstance(source, ExpressionSource):
        return source.expression.dependencies
    return (*source.value.dependencies, source.index)


def _field_demand(
    spec: ModelSpec,
    opened: tuple[str, ...],
    declaration: ModelDeclaration,
    conditions: Mapping[str, bool],
) -> FieldDemandPlan:
    """Resolve output requests to their concrete field dependencies."""

    output = declaration.output
    if not output.variables and not output.materialized:
        return FieldDemandPlan.empty()
    modules = spec.modules
    resolver: FieldNameResolver[tuple[str, str]] = FieldNameResolver()
    for module_name in opened:
        module = modules[module_name]
        for field in module.tensor_fields.values():
            if field.excluded or not conditions_satisfied(
                field.tensor.depends_on, opened, conditions
            ):
                continue
            resolver.install(
                module_name,
                field.name,
                (module_name, field.name),
                expression_virtual=bool(
                    field.tensor.category == "virtual" and field.tensor.expression
                ),
            )
        for index in active_reference_indices(module, opened, conditions=conditions):
            resolver.install(
                module_name,
                index.name,
                (module_name, index.name),
                expression_virtual=False,
            )
    resolve = resolver.entries.get
    known = set(resolver.entries)
    required: dict[str, set[str]] = {}
    observed: dict[str, set[str]] = {}
    visited: set[tuple[str, str]] = set()

    def tensor_metadata(field: tuple[str, str]) -> TensorMetadata | None:
        declared = modules[field[0]].tensor_fields.get(field[1])
        return None if declared is None else declared.tensor

    def visit(field: tuple[str, str]) -> None:
        if field in visited:
            return
        visited.add(field)
        module_name, field_name = field
        observed.setdefault(module_name, set()).add(field_name)
        required.setdefault(module_name, set()).add(field_name)
        module = modules[module_name]
        index = module.reference_indices.get(field_name)
        if index is not None:
            visit((module_name, index.reference))
            target = reference_target(
                module,
                index.reference,
                opened_modules=opened,
                module_specs=modules,
                conditions=conditions,
            )
            visit((target.module_name, target.name))
            return
        tensor = tensor_metadata(field)
        if tensor is None or tensor.category != "virtual" or not tensor.expression:
            return
        source = parse_value_source(tensor.expression, known)
        for dependency in source_dependencies(source):
            dependency_field = resolve(dependency)
            if dependency_field is not None:
                visit(dependency_field)

    for name in output.materialized:
        field = resolve(name)
        if field is None:
            raise ValueError(
                f"materialized output {name!r} is unknown, ambiguous, or inactive"
            )
        tensor = tensor_metadata(field)
        if tensor is None or tensor.output == "disabled":
            raise ValueError(f"materialized output {name!r} is disabled for output")
        visit(field)
    for items in output.variables.values():
        for item in items:
            direct = isinstance(item, str)
            name = item if direct else next(iter(item))
            field = resolve(name)
            if direct:
                if field is None:
                    raise ValueError(
                        f"statistics field {name!r} is unknown, ambiguous, or inactive"
                    )
                tensor = tensor_metadata(field)
                if tensor is None or tensor.output == "disabled":
                    raise ValueError(
                        f"statistics field {name!r} is disabled for output"
                    )
                visit(field)
                continue
            if field is not None:
                tensor = tensor_metadata(field)
                if tensor is not None and (
                    tensor.depends_on or tensor.required_by or tensor.output_only
                ):
                    observed.setdefault(field[0], set()).add(field[1])
                    if tensor_is_active(
                        tensor, opened, output_required=False, conditions=conditions
                    ):
                        continue
            source = parse_value_source(next(iter(item.values())), known)
            for dependency in source_dependencies(source):
                dependency_field = resolve(dependency)
                if dependency_field is not None:
                    visit(dependency_field)
    return FieldDemandPlan(required, observed)


def _check_namespace(
    definitions: dict[str, FieldSpec],
    field: FieldSpec,
    known_modules: Mapping[str, Any],
    active: bool,
) -> None:
    """Reject conflicting declarations of one bare name across modules.

    Expression virtuals may share a name with their source in another module;
    that is the standard subcell-to-cell aggregation pattern.
    """

    tensor = field.tensor
    if tensor is not None:
        unknown = sorted(
            set(
                (*module_conditions(tensor.depends_on), *tensor.required_by)
            ).difference(known_modules)
        )
        if unknown:
            raise ValueError(
                f"Tensor field {field.module_name}.{field.name} depends on unknown modules: {unknown}"
            )
    if field.excluded or not active:
        return
    # Virtual lookup precedence belongs to FieldNameResolver. This table
    # tracks physical owners only, regardless of module declaration order.
    if tensor is not None and tensor.category == "virtual" and tensor.expression:
        return
    previous = definitions.get(field.name)
    if previous is None:
        definitions[field.name] = field
        return
    if (
        tensor is not None
        and previous.tensor is not None
        and tensor.category == "init_state"
        and previous.tensor.category == "init_state"
    ):
        raise ValueError(
            f"checkpoint state name {field.name!r} is declared by both {previous.module_name!r} and {field.module_name!r}; state ownership must be unique"
        )
    if field.annotation != previous.annotation or tensor != previous.tensor:
        raise ValueError(
            f"Namespace conflict for {field.name!r}: {previous.module_name} and {field.module_name} declare different types or tensor metadata"
        )


def plan_fields(
    spec: ModelSpec, selection: Selection, declaration: ModelDeclaration
) -> FieldPlan:
    """Select, name, validate and index every field of the opened modules."""

    opened = selection.modules
    demand = _field_demand(spec, opened, declaration, selection.conditions)
    modules = MappingProxyType(
        {
            name: bind_module_fields(
                spec.modules[name],
                opened,
                demand,
                declaration.ensemble_forcing_fields.get(name, ()),
                module_specs=spec.modules,
                conditions=selection.conditions,
            )
            for name in opened
        }
    )
    definitions: dict[str, FieldSpec] = {}
    entries: dict[str, FieldEntry] = {}
    names: FieldNameResolver[FieldEntry] = FieldNameResolver()
    inputs: dict[str, InputFieldSpec] = {}
    module_sources: dict[str, list[tuple[str, BindingSource]]] = {}
    for module_name in opened:
        module = spec.modules[module_name]
        binding = modules[module_name]
        sources = module_sources[module_name] = []
        for name, field in module.fields.items():
            tensor = field.tensor
            active = tensor is None or name in binding.active
            _check_namespace(definitions, field, spec.modules, active)
            if tensor is None:
                sources.append((name, BindingSource(module_name)))
            else:
                entry = FieldEntry(module_name, name, field, tensor, active)
                entries[entry.qualified] = entry
                if active:
                    names.install(
                        module_name,
                        name,
                        entry,
                        expression_virtual=entry.expression_virtual,
                    )
                    if not tensor.expression:
                        sources.append(
                            (
                                name,
                                BindingSource(
                                    module_name, lazy=tensor.category == "virtual"
                                ),
                            )
                        )
            if (
                active
                and not field.computed
                and not field.excluded
                and (tensor is None or tensor.category != "forcing")
                and name not in inputs
            ):
                inputs[name] = InputFieldSpec(
                    field,
                    None
                    if tensor is None
                    else concrete_tensor_dtype(
                        tensor.dtype, selection.dtype, selection.mixed_precision
                    ),
                )
        for name, metadata in binding.reference_indices.items():
            entry = FieldEntry(
                module_name, name, module.reference_indices[name], metadata, True
            )
            entries[entry.qualified] = entry
            names.install(module_name, name, entry, expression_virtual=False)
            sources.append((name, BindingSource(module_name)))
        sources.extend(
            (name, BindingSource(module_name)) for name in module.kernel_fields
        )

    # Kernel ABI owners follow module construction order, then the model.
    binding_sources: dict[str, list[BindingSource]] = {}
    for module_name in selection.module_order:
        for name, source in module_sources[module_name]:
            binding_sources.setdefault(name, []).append(source)
    for name in spec.authored_fields:
        binding_sources.setdefault(name, []).append(BindingSource("model"))
        value = getattr(declaration, name)
        if (
            isinstance(value, BaseModel)
            and type(value).model_config.get("frozen") is True
        ):
            for nested in type(value).model_fields:
                binding_sources.setdefault(nested, []).append(
                    BindingSource(f"model.{name}")
                )
    for name in spec.kernel_fields:
        binding_sources.setdefault(name, []).append(BindingSource("model"))

    partition, variable_groups = compile_partition(
        tuple(
            entry
            for entry in entries.values()
            if entry.active and not entry.reference_index
        ),
        spec.partition_key,
        modules={name: spec.modules[name] for name in opened},
    )
    return FieldPlan(
        entries=MappingProxyType(entries),
        names=MappingProxyType(names.entries),
        demand=demand,
        modules=modules,
        partition=partition,
        variable_groups=variable_groups,
        inverse_sources=frozenset(
            index.reference
            for name, binding in modules.items()
            for index in spec.modules[name].reference_indices.values()
            if index.inverse and index.name in binding.active
        ),
        inputs=MappingProxyType(inputs),
        binding=MappingProxyType(
            {name: tuple(owners) for name, owners in binding_sources.items()}
        ),
    )

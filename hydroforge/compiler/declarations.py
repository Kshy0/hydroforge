from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from pydantic import TypeAdapter

from hydroforge.compiler.namespace import FieldNameResolver
from hydroforge.contracts.fields import FieldDemandPlan, tensor_is_active
from hydroforge.contracts.options import OptionsConfig
from hydroforge.contracts.runtime import (
    DEFAULT_BACKEND_REQUIREMENT,
    DEFAULT_MODULE_REQUIREMENT,
    RUNTIME_BACKEND_REQUIREMENTS,
    _effective_block_size,
)
from hydroforge.contracts.temporal import (
    EveryStep,
    StatisticsPlan,
    canonical_calendar,
    normalize_calendar_dates,
)
from hydroforge.contracts.validation import _immutable_dict
from hydroforge.statistics.ir import ExpressionSource, TensorSource, parse_value_source

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


def _default_mixed_precision(
    backend: str, device: torch.device, *, xpu_supports_fp64: bool | None = None
) -> bool:
    """Return the native accelerator default for hpfloat model storage."""
    if backend == "cuda" and device.type == "cuda":
        return True
    if backend != "triton":
        return False
    if device.type == "cuda":
        return True
    if device.type != "xpu":
        return False
    if xpu_supports_fp64 is None:
        xpu_supports_fp64 = _xpu_supports_fp64(device)
    return xpu_supports_fp64


def _xpu_supports_fp64(device: torch.device) -> bool:
    """Return compiler-relevant XPU FP64 support or fail before lowering."""
    runtime = getattr(torch, "xpu", None)
    properties_getter = getattr(runtime, "get_device_properties", None)
    if properties_getter is None:
        raise RuntimeError(
            "this PyTorch XPU runtime cannot report FP64 capability; HydroForge cannot safely select float64 Triton storage"
        )
    try:
        properties = properties_getter(device)
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"cannot query FP64 capability for XPU device {str(device)!r}"
        ) from error
    supported = getattr(properties, "has_fp64", None)
    if type(supported) is not bool:
        raise RuntimeError(
            f"XPU device {str(device)!r} did not expose an exact has_fp64 capability; HydroForge cannot safely select float64 Triton storage"
        )
    return supported


def validate_ensemble_forcing_fields(model: AbstractModel) -> AbstractModel:
    declaration = model.ensemble_forcing_fields
    if declaration and model.ensemble_size is None:
        raise ValueError("ensemble_forcing_fields require ensemble_size")
    module_types = model._module_types()
    opened = frozenset(model.opened_modules)
    for module_name, field_names in declaration.items():
        if module_name not in opened:
            raise ValueError(f"member forcing module {module_name!r} is not open")
        module_type = module_types[module_name]
        for field_name in field_names:
            schema = module_type._get_tensor_schema(
                field_name,
                opened_modules=model.opened_modules,
            )
            if schema is None or schema.tensor is None:
                raise ValueError(
                    f"unknown member forcing field {module_name}.{field_name}"
                )
            if schema.tensor.category != "forcing":
                raise ValueError(
                    f"member forcing field {module_name}.{field_name} has category {schema.tensor.category!r}, expected 'forcing'"
                )
            if not tensor_is_active(schema.tensor, model.opened_modules):
                raise ValueError(
                    f"member forcing field {module_name}.{field_name} is inactive"
                )
    object.__setattr__(model, "ensemble_forcing_fields", _immutable_dict(declaration))
    return model


def validate_module_requirements(model: AbstractModel) -> AbstractModel:
    for name in model.opened_modules:
        rule = model.module_requirements.get(name, DEFAULT_MODULE_REQUIREMENT)
        if not rule.ensemble and model.ensemble_size is not None:
            raise ValueError(f"module {name!r} does not support ensemble members")
    return model


def include_option_required_modules(cls, data: Any) -> Any:
    """Materialize modules implied by selected options before validation."""
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    options_field = cls.model_fields["options"]
    raw_options = (
        values["options"]
        if "options" in values
        else options_field.get_default(call_default_factory=True)
    )
    annotation = options_field.annotation
    options = (
        raw_options
        if isinstance(raw_options, OptionsConfig)
        else TypeAdapter(annotation).validate_python(raw_options)
    )
    values["options"] = options
    opened_field = cls.model_fields["opened_modules"]
    opened = (
        values["opened_modules"]
        if "opened_modules" in values
        else opened_field.get_default(call_default_factory=True)
    )
    if type(opened) is not tuple:
        return values
    module_types = cls._module_types()
    required = {
        module for modules in options.required_modules().values() for module in modules
    }
    pending = list(required)
    while pending:
        module = pending.pop()
        module_type = module_types.get(module)
        if module_type is None:
            continue
        for dependency in module_type._required_modules():
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    selected = list(opened)
    for module in module_types:
        if module in required and module not in selected:
            selected.append(module)
    for module in sorted(required.difference(module_types)):
        if module not in selected:
            selected.append(module)
    values["opened_modules"] = tuple(selected)
    return values


def validate_option_module_requirements(model: AbstractModel) -> AbstractModel:
    opened = set(model.opened_modules)
    for path, required in model.options.required_modules().items():
        missing = sorted(set(required).difference(opened))
        if missing:
            raise ValueError(
                f"option {path}={model.options.choice(path)!r} requires opened_modules to include {missing}"
            )
    return model


def validate_runtime_declaration(model: AbstractModel) -> AbstractModel:
    """Canonicalize every model/runtime choice before initialization."""
    from hydroforge.contracts.step_fields import (
        _StepFieldExpressions,
        _StepFieldProviders,
    )
    from hydroforge.kernels.registry import (
        _backend_device_types,
        _resolve_model_backend_trusted,
    )

    model._step_field_providers = _StepFieldProviders(
        providers=model.step_field_providers
    ).providers
    model._step_field_expressions = _StepFieldExpressions(
        expressions=model.step_field_expressions,
        host_sources=frozenset(model._step_field_providers),
    )._compiled

    backend = _resolve_model_backend_trusted(model.device)
    model._backend = backend
    required_devices = _backend_device_types(backend)
    if required_devices is not None and model.device.type not in required_devices:
        required_label = (
            repr(required_devices[0])
            if len(required_devices) == 1
            else " or ".join(repr(item) for item in required_devices)
        )
        raise ValueError(
            f"HydroForge backend {backend!r} requires a {required_label} model device, got {str(model.device)!r}"
        )
    mixed_precision = model.mixed_precision
    needs_xpu_fp64_capability = model.device.type == "xpu" and (
        mixed_precision is None or mixed_precision or model.precision == "float64"
    )
    xpu_supports_fp64 = (
        _xpu_supports_fp64(model.device) if needs_xpu_fp64_capability else None
    )
    if mixed_precision is None:
        mixed_precision = _default_mixed_precision(
            backend, model.device, xpu_supports_fp64=xpu_supports_fp64
        )
        object.__setattr__(model, "mixed_precision", mixed_precision)
    if (
        model.device.type == "xpu"
        and (model.precision == "float64" or mixed_precision)
        and (xpu_supports_fp64 is False)
    ):
        raise ValueError(
            f"XPU device {str(model.device)!r} does not support FP64, but the model requests float64 storage through precision or mixed_precision"
        )
    if model.result_device is None:
        object.__setattr__(model, "result_device", torch.device("cpu"))
    if model.variables_to_save:
        plan = (
            StatisticsPlan() if model.statistics_plan is None else model.statistics_plan
        )
    else:
        if model.statistics_plan is not None:
            raise ValueError("statistics_plan requires a non-empty variables_to_save")
        plan = None
    schedule = model.simulation_schedule
    if (
        plan is not None
        and schedule is None
        and (
            not (
                isinstance(plan.inner, EveryStep)
                and isinstance(plan._effective_outer, EveryStep)
            )
        )
    ):
        raise ValueError(
            "calendar or explicit statistics windows require simulation_schedule"
        )
    if plan is not None and schedule is not None:
        from hydroforge.execution.windows import (
            bind_statistics_plan_schedule,
            validate_statistics_window_schedule,
        )

        plan = bind_statistics_plan_schedule(plan, schedule)
        if model.statistics_plan is not None:
            object.__setattr__(model, "statistics_plan", plan)
        validate_statistics_window_schedule(plan, schedule)
    model._statistics_plan = plan
    if schedule is not None:
        if model.initial_time is not None:
            raise ValueError(
                "initial_time must not be configured together with simulation_schedule"
            )
        if model.calendar is not None:
            configured = canonical_calendar(model.calendar)
            if configured != schedule.calendar:
                raise ValueError(
                    f"model calendar {configured!r} differs from simulation schedule {schedule.calendar!r}"
                )
        calendar = schedule.calendar
    else:
        calendar, normalized, _defaulted = normalize_calendar_dates(
            {"model initial_time": model.initial_time}, calendar=model.calendar
        )
        object.__setattr__(model, "initial_time", normalized["model initial_time"])
    object.__setattr__(model, "calendar", calendar)
    runtime_rule = RUNTIME_BACKEND_REQUIREMENTS.get(
        backend, DEFAULT_BACKEND_REQUIREMENT
    )
    model_rule = model.backend_requirements.get(backend, DEFAULT_BACKEND_REQUIREMENT)
    runtime_rule._validate_precision(model.precision, mixed_precision, backend=backend)
    model_rule._validate_precision(model.precision, mixed_precision, backend=backend)
    model.options.validate_backend(backend)
    if model.BLOCK_SIZE is not None or backend == "metal":
        block_size = _effective_block_size(model.BLOCK_SIZE, backend=backend)
        if backend == "metal":
            object.__setattr__(model, "BLOCK_SIZE", block_size)
        model_rule._validate_block_size(block_size, backend=backend)
    if not model_rule.ensemble and model.ensemble_size is not None:
        raise ValueError(f"backend {backend!r} does not support ensemble members")
    return model


def resolved_module_order(model: AbstractModel) -> tuple[str, ...]:
    """Return the deterministic dependency order for validated modules."""
    from graphlib import CycleError, TopologicalSorter

    module_types = model._module_types()
    opened = frozenset(model.opened_modules)
    sorter: TopologicalSorter[str] = TopologicalSorter()
    for name in model.opened_modules:
        references = module_types[name]._module_reference_fields().values()
        sorter.add(
            name,
            *(
                reference.module_name
                for reference in references
                if reference.module_name in opened
            ),
        )
    try:
        return tuple(sorter.static_order())
    except CycleError as error:
        raise ValueError(
            f"opened module references must form an acyclic construction graph: {error.args[1]}"
        ) from error


def compile_output_tensor_activation(model: AbstractModel) -> AbstractModel:
    """Resolve output requests to their concrete field dependencies."""
    if model._statistics_plan is None and (not model.materialized_outputs):
        model._field_demand = FieldDemandPlan.empty()
        return model
    opened = frozenset(model.opened_modules)
    schema = type(model)._compiled_schema()
    module_types = model._module_types()
    resolver: FieldNameResolver[tuple[str, str]] = FieldNameResolver()

    def install(field: Any) -> None:
        resolver.install(
            field.module_name,
            field.name,
            (field.module_name, field.name),
            expression_virtual=bool(
                field.tensor.category == "virtual" and field.tensor.expression
            ),
        )

    for module_name in model.opened_modules:
        excluded = set(module_types[module_name].nc_excluded_fields)
        for field in schema.fields(module_name):
            tensor = field.tensor
            if (
                tensor is None
                or field.excluded
                or field.name in excluded
                or (not all(dependency in opened for dependency in tensor.depends_on))
            ):
                continue
            install(field)
        for field_name in module_types[module_name]._reference_index_fields(
            opened_modules=model.opened_modules,
        ):
            resolver.install(
                module_name,
                field_name,
                (module_name, field_name),
                expression_virtual=False,
            )
    required: dict[str, set[str]] = {}
    observed: dict[str, set[str]] = {}
    known = set(resolver.entries)
    visited: set[tuple[str, str]] = set()

    def tensor_metadata(field: tuple[str, str]):
        module_name, field_name = field
        schema = module_types[module_name]._tensor_schema_map().get(field_name)
        return None if schema is None else schema.tensor

    def source_dependencies(source: Any) -> tuple[str, ...]:
        if isinstance(source, TensorSource):
            return (source.name,)
        if isinstance(source, ExpressionSource):
            return source.expression.dependencies
        return (*source.value.dependencies, source.index)

    def resolve_field(name: str) -> tuple[str, str] | None:
        return resolver.entries.get(name)

    def visit_field(field: tuple[str, str]) -> None:
        if field in visited:
            return
        visited.add(field)
        module_name, field_name = field
        observed.setdefault(module_name, set()).add(field_name)
        required.setdefault(module_name, set()).add(field_name)
        module_type = module_types[module_name]
        descriptor = module_type._reference_index_fields().get(field_name)
        if descriptor is not None:
            visit_field((module_name, descriptor.reference))
            target = module_type._reference_target_schema(
                descriptor.reference,
                opened_modules=model.opened_modules,
            )
            visit_field((target.module_name, target.name))
            return
        tensor = tensor_metadata(field)
        if tensor is None or tensor.category != "virtual" or (not tensor.expression):
            return
        source = parse_value_source(tensor.expression, known)
        for dependency in source_dependencies(source):
            dependency_field = resolve_field(dependency)
            if dependency_field is not None:
                visit_field(dependency_field)

    for name in model.materialized_outputs:
        field = resolve_field(name)
        if field is None:
            raise ValueError(
                f"materialized output {name!r} is unknown, ambiguous, or inactive"
            )
        tensor = tensor_metadata(field)
        if tensor is None or tensor.output == "disabled":
            raise ValueError(f"materialized output {name!r} is disabled for output")
        visit_field(field)
    for items in model.variables_to_save.values():
        for item in items:
            direct = isinstance(item, str)
            name = item if direct else next(iter(item))
            field = resolve_field(name)
            if direct:
                if field is None:
                    continue
                tensor = tensor_metadata(field)
                if tensor is None or tensor.output == "disabled":
                    raise ValueError(
                        f"statistics field {name!r} is disabled for output"
                    )
                visit_field(field)
                continue
            expression = next(iter(item.values()))
            if field is not None:
                tensor = tensor_metadata(field)
                if tensor is not None and (
                    tensor.depends_on or tensor.required_by or tensor.output_only
                ):
                    module_name, field_name = field
                    observed.setdefault(module_name, set()).add(field_name)
                    if tensor_is_active(
                        tensor, model.opened_modules, output_required=False
                    ):
                        continue
            source = parse_value_source(expression, known)
            for dependency in source_dependencies(source):
                dependency_field = resolve_field(dependency)
                if dependency_field is not None:
                    visit_field(dependency_field)
    model._field_demand = FieldDemandPlan.from_sets(required, observed)
    return model


def validate_namespace(model: AbstractModel) -> AbstractModel:
    """
    Check for namespace conflicts across all opened modules.

    Virtual fields with an ``expr`` (scatter / plain aggregation outputs)
    are allowed to share a name with their source counterpart in another
    module — this is the standard subcell→cell aggregation pattern.
    """
    field_definitions = {}
    schema = model._compiled_schema()
    module_types = model._module_types()
    for module_name in model.opened_modules:
        excluded = set(module_types[module_name].nc_excluded_fields)
        for field in schema.fields(module_name):
            if field.tensor is not None:
                unknown_dependencies = sorted(
                    set(
                        (*field.tensor.depends_on, *field.tensor.required_by)
                    ).difference(module_types)
                )
                if unknown_dependencies:
                    raise ValueError(
                        f"Tensor field {module_name}.{field.name} depends on unknown modules: {unknown_dependencies}"
                    )
            if field.excluded or field.name in excluded:
                continue
            if field.tensor is not None and (
                not model._is_tensor_field_active(module_name, field)
            ):
                continue
            previous = field_definitions.get(field.name)
            if previous is None:
                field_definitions[field.name] = field
                continue
            new_virtual = bool(
                field.tensor is not None
                and field.tensor.category == "virtual"
                and field.tensor.expression
            )
            old_virtual = bool(
                previous.tensor is not None
                and previous.tensor.category == "virtual"
                and previous.tensor.expression
            )
            if new_virtual or old_virtual:
                if new_virtual and (not old_virtual):
                    field_definitions[field.name] = field
                continue
            if (
                field.tensor is not None
                and previous.tensor is not None
                and (field.tensor.category == "init_state")
                and (previous.tensor.category == "init_state")
            ):
                raise ValueError(
                    f"checkpoint state name {field.name!r} is declared by both {previous.module_name!r} and {module_name!r}; state ownership must be unique"
                )
            if (
                field.annotation != previous.annotation
                or field.tensor != previous.tensor
            ):
                raise ValueError(
                    f"Namespace conflict for {field.name!r}: {previous.module_name} and {module_name} declare different types or tensor metadata"
                )
    model._namespace_declaration = MappingProxyType(field_definitions)
    return model


def validate_modules(cls, v: Any) -> tuple[str, ...]:
    """Validate module names are valid"""
    if type(v) is not tuple:
        raise ValueError("opened_modules must be an exact tuple")
    if any(type(module) is not str or not module for module in v):
        raise ValueError("opened_modules must contain non-empty exact strings")
    if not v:
        raise ValueError(
            "No modules opened. Please specify at least one module in opened_modules."
        )
    if len(v) != len(set(v)):
        raise ValueError("opened_modules must not contain duplicates")
    module_types = cls._module_types()
    for module in v:
        if module not in module_types:
            raise ValueError(
                f"Invalid module name: {module}. Available modules: {list(module_types)}"
            )
    missing_model_modules = [
        name
        for name, reference in cls._module_reference_fields().items()
        if not reference.optional and name not in v
    ]
    if missing_model_modules:
        raise ValueError(
            f"Missing required model modules in opened_modules: {missing_model_modules}. Available modules: {v}"
        )
    for module in v:
        module_class = module_types[module]
        references = module_class._module_reference_fields().values()
        unknown_references = sorted(
            {
                reference.module_name
                for reference in references
                if reference.module_name not in module_types
            }
        )
        if unknown_references:
            raise ValueError(
                f"Module '{module}' declares references to unknown modules: {unknown_references}. Available modules: {list(module_types)}"
            )
        required = module_class._required_modules()
        missing_deps = [dep for dep in required if dep not in v]
        if missing_deps:
            raise ValueError(
                f"Module '{module}' has missing required modules in opened_modules: {missing_deps}. Required modules: {required}. Available modules: {v}"
            )
        present_conflicts = [
            conflict
            for conflict in module_class.conflicts
            if conflict in v and conflict != module
        ]
        if present_conflicts:
            raise ValueError(
                f"Module '{module}' conflicts with modules present in opened_modules: {present_conflicts}. These modules cannot be enabled together."
            )
    return v


def validate_input_contract(model: AbstractModel) -> AbstractModel:
    """Bind external storage to the complete validated model schema.

    Scheduled parameter validation prepares the actual rank-local inputs and
    model/module hooks once. It does not allocate default model state,
    initialize a backend, open persistent resources or start execution.
    """
    from hydroforge.compiler.data import ModelDataCompiler
    from hydroforge.compiler.model import _ModelSemanticPlan
    from hydroforge.compiler.parameters import ParameterSemanticCompiler
    from hydroforge.compiler.partition import (
        PartitionCompiler,
        _PartitionSemanticCompiler,
    )
    from hydroforge.data.model_input import ModelInput

    model._input = ModelInput(model)
    partition = _PartitionSemanticCompiler(model)
    partition_schema = partition.schema
    variable_groups = partition.variable_groups
    input_axes = model._input.compile_partition_axes(partition)
    partition.validate_global_reference_integrity()
    reference_targets, inverse_sources = partition.compile_reference_targets()
    partition.validate_inverse_reference_integrity(inverse_sources)
    model._semantic_plan = _ModelSemanticPlan(
        backend=model._backend,
        module_order=model._module_order,
        namespace=model._namespace_declaration,
        input_binding=model._input,
        partition_schema=partition_schema,
        variable_groups=variable_groups,
        input_axes=input_axes,
        reference_targets=reference_targets,
        ensemble_forcing_fields=model.ensemble_forcing_fields,
        field_demand=model._field_demand,
        statistics=model._statistics_declaration,
        parameter_changes=(),
    )
    model._partition = PartitionCompiler(
        model,
        schema=partition_schema,
        variable_groups=variable_groups,
    )
    model._data = ModelDataCompiler(model)
    parameter_changes = ParameterSemanticCompiler(model, partition).compile(
        model.parameter_changes,
    )
    model._semantic_plan = replace(
        model._semantic_plan,
        parameter_changes=parameter_changes,
    )
    return model

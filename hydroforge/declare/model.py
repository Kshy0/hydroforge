"""Model declaration fields, class-level declarations and the frozen ``ModelSpec``."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self

import cftime
import torch
from pydantic import (
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from hydroforge.contracts.options import OptionsConfig
from hydroforge.contracts.parameters import ParameterChange
from hydroforge.contracts.runtime import BackendRequirement, ModuleRequirement
from hydroforge.contracts.schedule import SimulationSchedule
from hydroforge.contracts.step_fields import (
    StepFieldProvider,
    compile_step_field_expressions,
    validate_step_field_providers,
)
from hydroforge.contracts.windows import StatisticsPlan
from hydroforge.core.events import ConsoleEventSink, EventSink
from hydroforge.core.expr import parse_operation, parse_value_source
from hydroforge.core.naming import Identifier, validate_safe_path_component
from hydroforge.core.validation import FrozenMapping, HydroForgeModel
from hydroforge.data.input import InputProxy
from hydroforge.declare.kernel_field import _KernelField
from hydroforge.declare.module import (
    _SPEC,
    _STRICT,
    ModuleReference,
    _declared_attributes,
)
from hydroforge.declare.spec import ModelSpec, StepFieldPlan
from hydroforge.io.netcdf.options import (
    default_netcdf_options,
    normalize_netcdf_variable_options,
)
from hydroforge.parallel.mesh import EnsembleParallel

_BACKEND_REQUIREMENTS = TypeAdapter(
    FrozenMapping[Literal["torch", "cuda", "triton", "metal"], BackendRequirement],
    config=_STRICT,
)
_MODULE_REQUIREMENTS = TypeAdapter(
    FrozenMapping[str, ModuleRequirement],
    config=_STRICT,
)
_PARTITION_KEY = TypeAdapter(Identifier | None, config=_STRICT)
_PARTITION_GROUP = TypeAdapter(Identifier, config=_STRICT)


def _model_spec(cls: type[ModelDeclaration]) -> ModelSpec:
    """Validate and freeze one model class declaration."""

    for name, field in cls.model_fields.items():
        if isinstance(field.default, ModuleReference):
            raise ValueError(f"module reference {name!r} conflicts with a model field")
    references: dict[str, ModuleReference] = {}
    kernel_fields: list[str] = []
    for name, value in _declared_attributes(cls).items():
        if isinstance(value, ModuleReference):
            references[name] = value
        elif isinstance(value, _KernelField):
            kernel_fields.append(name)
    module_requirements = _MODULE_REQUIREMENTS.validate_python(cls.module_requirements)
    unknown_modules = set(module_requirements).difference(references)
    if unknown_modules:
        raise ValueError(
            f"module_requirements names unknown modules: {sorted(unknown_modules)}"
        )
    providers = validate_step_field_providers(cls.step_field_providers)
    modules = {
        name: reference.module_type.spec() for name, reference in references.items()
    }
    for module in modules.values():
        unknown = set(module.references).difference(modules)
        for field in module.tensor_fields.values():
            unknown.update(
                set((*field.tensor.depends_on, *field.tensor.required_by)).difference(
                    modules
                )
            )
        if unknown:
            raise ValueError(
                f"Module {module.name!r} references unknown modules: {sorted(unknown)}"
            )
    return ModelSpec(
        model_type=cls,
        modules=MappingProxyType(modules),
        required_modules=tuple(
            name for name, reference in references.items() if not reference.optional
        ),
        backend_requirements=_BACKEND_REQUIREMENTS.validate_python(
            cls.backend_requirements
        ),
        module_requirements=module_requirements,
        partition_key=_PARTITION_KEY.validate_python(cls.partition_key),
        partition_group=_PARTITION_GROUP.validate_python(cls.partition_group),
        step_fields=StepFieldPlan(
            providers=providers,
            expressions=compile_step_field_expressions(
                cls.step_field_expressions, host_sources=frozenset(providers)
            ),
        ),
        kernel_fields=tuple(kernel_fields),
        authored_fields=tuple(
            name
            for name in cls.model_fields
            if name not in ModelDeclaration.model_fields
        ),
    )


def include_option_required_modules(cls: type[ModelDeclaration], data: Any) -> Any:
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
    options = (
        raw_options
        if isinstance(raw_options, OptionsConfig)
        else TypeAdapter(options_field.annotation).validate_python(raw_options)
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
    module_specs = cls.spec().modules
    required = {
        module for modules in options.required_modules().values() for module in modules
    }
    pending = list(required)
    while pending:
        module_spec = module_specs.get(pending.pop())
        if module_spec is None:
            continue
        for dependency in module_spec.required_modules:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    selected = list(opened)
    for module in module_specs:
        if module in required and module not in selected:
            selected.append(module)
    for module in sorted(required.difference(module_specs)):
        if module not in selected:
            selected.append(module)
    values["opened_modules"] = tuple(selected)
    return values


class OutputConfig(HydroForgeModel):
    """What a model writes and where; ``dir=None`` writes no file at all."""

    dir: Path | None = Field(
        default=None,
        strict=False,
        description=(
            "Output root directory. None creates no directory and writes no "
            "manifest, log, NetCDF output or checkpoint."
        ),
    )
    experiment: str = Field(
        default="experiment",
        description="Experiment directory name beneath dir",
    )
    variables: FrozenMapping[str, tuple[str | FrozenMapping[str, str], ...]] = Field(
        default_factory=dict,
        description=(
            "Statistics outputs as {operation: [field or {alias: expression}]}."
        ),
    )
    statistics_plan: InstanceOf[StatisticsPlan] | None = Field(
        default=None,
        description=(
            "Optional temporal window policy for variables; omitted means "
            "every model step"
        ),
    )
    materialized: tuple[str, ...] = Field(
        default=(),
        description=(
            "Output-capable fields to keep resident for direct consumers "
            "without registering a statistics writer"
        ),
    )
    save_precision: Literal["float32", "float64"] | None = Field(
        default="float32",
        description=(
            "Floating-point precision used for persisted statistics; None "
            "preserves each statistics tensor's resolved precision."
        ),
    )
    workers: int = Field(
        default=2,
        ge=0,
        strict=True,
        description="Number of workers for writing output files",
    )
    split_by_year: bool = Field(
        default=False,
        strict=True,
        description="Whether to split output files by year",
    )
    sink: Literal["netcdf", "memory"] = Field(
        default="netcdf",
        description=(
            "Statistics destination: NetCDF files beneath dir, or results "
            "retained in memory for model.results"
        ),
    )
    result_device: torch.device = Field(
        default=torch.device("cpu"),
        description='Device of in-memory results; requires sink="memory"',
    )
    max_pending_steps: int = Field(
        default=200,
        ge=1,
        strict=True,
        description="Maximum number of pending time steps for output buffering",
    )
    netcdf: FrozenMapping[str, Any] = Field(
        default_factory=default_netcdf_options,
        description=(
            "Additional validated keyword options passed to netCDF4 "
            "Dataset.createVariable for dynamic output variables."
        ),
    )
    checkpoint_netcdf: FrozenMapping[str, Any] = Field(
        default_factory=default_netcdf_options,
        description=(
            "Validated netCDF4 Dataset.createVariable options for model "
            "checkpoint variables."
        ),
    )

    @field_validator("dir", mode="before")
    @classmethod
    def _reject_empty_dir(cls, value: Any) -> Any:
        """Keep lax path coercion without mapping ``""`` to the cwd."""

        if isinstance(value, str) and not value:
            raise ValueError("dir must not be an empty string")
        return value

    @field_validator("experiment", mode="before")
    @classmethod
    def _validate_experiment(cls, value: Any) -> str:
        """Require one directory component beneath ``dir``."""

        return validate_safe_path_component(value, label="experiment")

    @field_validator("netcdf", "checkpoint_netcdf", mode="before")
    @classmethod
    def _validate_netcdf_options(cls, value):
        return normalize_netcdf_variable_options(value)

    @field_validator("variables", mode="before")
    @classmethod
    def _validate_variables(cls, value: Any):
        """Canonicalize the original user-facing statistics declaration."""

        if type(value) is not dict:
            raise ValueError("variables must be an exact dict")
        normalized: dict[str, tuple[str | Mapping[str, str], ...]] = {}
        for operation, items in value.items():
            if type(operation) is not str or not operation:
                raise ValueError(
                    "variables operation names must be non-empty exact strings"
                )
            canonical = operation.lower()
            if canonical != "static":
                parse_operation(canonical)
            if canonical in normalized:
                raise ValueError(
                    f"variables contains duplicate normalized operation {canonical!r}"
                )
            if type(items) is not list:
                raise ValueError(f"variables[{operation!r}] must be an exact list")
            compiled_items: list[str | Mapping[str, str]] = []
            for item in items:
                if type(item) is str:
                    if not item:
                        raise ValueError("statistics field names must be non-empty")
                    compiled_items.append(item)
                    continue
                if type(item) is not dict or len(item) != 1:
                    raise ValueError(
                        "variables items must be field names or "
                        "one-item {alias: expression} dicts"
                    )
                alias, expression = next(iter(item.items()))
                if canonical == "static":
                    raise ValueError("static output does not accept expressions")
                if (
                    type(alias) is not str
                    or not alias
                    or type(expression) is not str
                    or not expression
                ):
                    raise ValueError(
                        "statistics aliases and expressions must be "
                        "non-empty exact strings"
                    )
                compiled_items.append(item)
                parse_value_source(expression)
            normalized[canonical] = tuple(compiled_items)
        pairs = set()
        expressions = {}
        for operation, items in normalized.items():
            for item in items:
                name, expression = (
                    (item, None) if isinstance(item, str) else next(iter(item.items()))
                )
                if (name, operation) in pairs:
                    raise ValueError("variables must not repeat a field/operation")
                pairs.add((name, operation))
                if expressions.setdefault(name, expression) != expression:
                    raise ValueError(
                        f"statistics output {name!r} has conflicting expressions"
                    )
        if normalized and not any(
            operation != "static" and items for operation, items in normalized.items()
        ):
            raise ValueError("variables requires at least one dynamic output")
        return normalized

    @model_validator(mode="after")
    def _validate_destination(self) -> Self:
        if self.statistics_plan is not None and not self.variables:
            raise ValueError("statistics_plan requires non-empty variables")
        if self.sink != "memory" and "result_device" in self.model_fields_set:
            raise ValueError(
                'result_device applies only to OutputConfig(sink="memory")'
            )
        if self.sink == "netcdf" and self.variables and self.dir is None:
            raise ValueError(
                "NetCDF statistics output requires OutputConfig(dir=...); "
                'use sink="memory" to retain results without files'
            )
        return self


class ModelDeclaration(HydroForgeModel):
    """Framework fields and class-level declarations shared by every model.

    Validators here only normalize one declared value; everything resolved
    from several values or from the environment lives in the compiled plan.
    """

    model_config = ConfigDict(
        ignored_types=(_KernelField, ModuleReference),
    )

    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = MappingProxyType(
        {}
    )
    module_requirements: ClassVar[Mapping[str, ModuleRequirement]] = MappingProxyType(
        {}
    )
    partition_key: ClassVar[str | None] = None
    partition_group: ClassVar[str] = "group_id"
    step_field_providers: ClassVar[Mapping[str, StepFieldProvider]] = MappingProxyType(
        {}
    )
    step_field_expressions: ClassVar[Mapping[str, str]] = MappingProxyType({})

    input_proxy: InstanceOf[InputProxy] = Field(
        description="InputProxy object containing model data",
    )
    opened_modules: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of active modules",
    )
    options: OptionsConfig = Field(
        default_factory=OptionsConfig,
        description="Immutable typed model options and constants",
    )
    device: torch.device = Field(
        default=torch.device("cpu"),
        description="Device for tensors (e.g., 'cuda:0', 'xpu:0', 'cpu')",
    )
    precision: Literal["float32", "float64"] = Field(
        default="float32",
        description="Base precision of the model",
    )
    metal_emulation: Literal["auto", "native", "float32x2"] = Field(
        default="auto",
        description="Metal hpfloat representation; auto selects float32x2 for Metal mixed precision",
    )
    mixed_precision: bool | None = Field(
        default=None,
        strict=True,
        description=(
            "Enable mixed precision for hpfloat (storage) tensors.\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  float32 → float64, float64 → float64 (no promotion).\n"
            "If omitted, uses the model's backend-specific default when declared; "
            "otherwise defaults to enabled for CUDA/ROCm and for XPU "
            "Triton devices that report FP64 support; it is disabled for "
            "other backends and XPU devices without FP64."
        ),
    )
    event_sink: EventSink = Field(
        default_factory=ConsoleEventSink,
        description="Structured lifecycle/progress event destination",
    )
    simulation_schedule: InstanceOf[SimulationSchedule] | None = Field(
        default=None,
        description="Runtime-owned model call schedule and calendar contract",
    )
    initial_time: datetime | cftime.datetime | None = Field(
        default=None,
        description=(
            "Initial runtime clock when no simulation schedule is supplied; "
            "a cftime date also selects the calendar"
        ),
    )
    parameter_changes: tuple[InstanceOf[ParameterChange], ...] = Field(
        default=(),
        description="Complete immutable scheduled parameter declarations",
    )
    ensemble_size: int | None = Field(
        default=None,
        ge=2,
        strict=True,
        description="Number of parallel simulations (ensemble members)",
    )
    ensemble_forcing_fields: FrozenMapping[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description=(
            "Construction-time member-batched forcing fields grouped by module; "
            "unlisted forcing fields remain shared"
        ),
    )
    execution_mode: Literal["auto", "eager"] = Field(
        default="auto",
        description=(
            "Execution scheduling policy. 'auto' selects the cached native "
            "capture supported by the active device and allowed by the "
            "model's backend requirement. CUDA conditional graphs require a "
            "native CUDA build with CUDA >= 12.4; ROCm/HIP uses the eager "
            "fallback. 'eager' keeps every launch directly observable for "
            "differentiation and debugging."
        ),
    )
    block_size: int | None = Field(
        default=None,
        ge=1,
        le=1024,
        strict=True,
        description=(
            "Global GPU block-size override. None selects the model's "
            "backend default or, without one, each kernel's backend default."
        ),
    )
    parallel: InstanceOf[EnsembleParallel] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Live ensemble process mesh; requires ensemble_size",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Statistics, checkpoint and output-file configuration",
    )

    # The compiled plan and the runtime that owns every materialized resource.
    _plan: Any = PrivateAttr()
    _runtime: Any = PrivateAttr()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        setattr(cls, _SPEC, _model_spec(cls))

    @classmethod
    def spec(cls) -> ModelSpec:
        """Return the frozen declaration of this model class."""

        spec = cls.__dict__.get(_SPEC)
        if spec is None:
            spec = _model_spec(cls)
            setattr(cls, _SPEC, spec)
        return spec

    @model_validator(mode="before")
    @classmethod
    def _include_option_required_modules(cls, data: Any) -> Any:
        return include_option_required_modules(cls, data)

    @field_validator("opened_modules", mode="before")
    @classmethod
    def _validate_modules(cls, value: Any) -> tuple[str, ...]:
        """Require an exact, complete and conflict-free module selection."""

        if type(value) is not tuple:
            raise ValueError("opened_modules must be an exact tuple")
        if any(type(module) is not str or not module for module in value):
            raise ValueError("opened_modules must contain non-empty exact strings")
        if not value:
            raise ValueError(
                "No modules opened. Please specify at least one module in "
                "opened_modules."
            )
        if len(value) != len(set(value)):
            raise ValueError("opened_modules must not contain duplicates")
        model_spec = cls.spec()
        modules = model_spec.modules
        for module in value:
            if module not in modules:
                raise ValueError(
                    f"Invalid module name: {module}. Available modules: {list(modules)}"
                )
        missing_model_modules = [
            name for name in model_spec.required_modules if name not in value
        ]
        if missing_model_modules:
            raise ValueError(
                "Missing required model modules in opened_modules: "
                f"{missing_model_modules}. Available modules: {value}"
            )
        for module in value:
            module_spec = modules[module]
            for name, reference in module_spec.references.items():
                if name in value:
                    reference.validate_type(modules[name].module_type)
            unknown_references = sorted(
                {
                    reference
                    for reference in module_spec.references
                    if reference not in modules
                }
            )
            if unknown_references:
                raise ValueError(
                    f"Module '{module}' declares references to unknown modules: "
                    f"{unknown_references}. Available modules: {list(modules)}"
                )
            required = module_spec.required_modules
            missing_deps = [dep for dep in required if dep not in value]
            if missing_deps:
                raise ValueError(
                    f"Module '{module}' has missing required modules in "
                    f"opened_modules: {missing_deps}. Required modules: "
                    f"{required}. Available modules: {value}"
                )
            present_conflicts = [
                conflict
                for conflict in module_spec.conflicts
                if conflict in value and conflict != module
            ]
            if present_conflicts:
                raise ValueError(
                    f"Module '{module}' conflicts with modules present in "
                    f"opened_modules: {present_conflicts}. These modules cannot "
                    "be enabled together."
                )
        return value

    @field_validator("ensemble_forcing_fields", mode="before")
    @classmethod
    def _validate_ensemble_forcing_declaration(cls, value: Any):
        if type(value) is not dict:
            raise ValueError("ensemble_forcing_fields must be an exact dict")
        normalized: dict[str, tuple[str, ...]] = {}
        for module_name, field_names in value.items():
            if type(module_name) is not str or not module_name:
                raise ValueError(
                    "ensemble_forcing_fields module names must be non-empty strings"
                )
            if type(field_names) is not tuple:
                raise ValueError(
                    f"ensemble_forcing_fields[{module_name!r}] must be an exact tuple"
                )
            if any(
                type(field_name) is not str or not field_name
                for field_name in field_names
            ):
                raise ValueError(
                    f"ensemble_forcing_fields[{module_name!r}] must contain "
                    "non-empty strings"
                )
            if len(field_names) != len(set(field_names)):
                raise ValueError(
                    f"ensemble_forcing_fields[{module_name!r}] contains duplicates"
                )
            normalized[module_name] = field_names
        return normalized

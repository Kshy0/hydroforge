# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from abc import ABC
from datetime import datetime, timedelta
from functools import cache, cached_property
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Mapping,
    Self,
    Union,
    cast,
)
from uuid import uuid4

import cftime
import numpy as np
import torch
import torch.distributed as dist
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.statistics.ir import (
    _StatisticsDeclaration,
    ExpressionSource,
    Reduction,
    ScatterSource,
    StatisticsProgram,
    TensorSource,
    build_variable_storage_plan,
    parse_operation,
    parse_value_source,
    validate_expression_constants,
)
from hydroforge.contracts.naming import RESERVED_CONTROL_STATE, sanitize_symbol
from hydroforge.compiler.namespace import NamespaceEntry
from hydroforge.data.distributed import ProcessTopology
from hydroforge.data.input import InputProxy
from hydroforge.contracts.kernel_field import _KernelField
from hydroforge.contracts.validation import HydroForgeModel, _immutable_dict
from hydroforge.contracts.fields import FieldDemandPlan, tensor_is_active
from hydroforge.contracts.temporal import (
    EveryStep,
    SimulationSchedule,
    StatisticsPlan,
    _StatisticsOutput,
    canonical_calendar,
    normalize_calendar_dates,
)
from hydroforge.contracts.events import ConsoleEventSink, EventSink, emit
from hydroforge.contracts.errors import (
    ResourceCleanupError,
    distributed_failure_error,
    failure_description,
)
from hydroforge.contracts.runtime import (
    BackendRequirement,
    DEFAULT_BACKEND_REQUIREMENT,
    DEFAULT_MODULE_REQUIREMENT,
    _effective_block_size,
    ModuleRequirement,
    RUNTIME_BACKEND_REQUIREMENTS,
)
from hydroforge.contracts.parameters import ParameterChange
from hydroforge.model.module import AbstractModule, ModuleReference
from hydroforge.serialization.netcdf import default_netcdf_options

if TYPE_CHECKING:
    from hydroforge.compiler.data import ModelDataCompiler
    from hydroforge.compiler.namespace import NamespaceCompiler
    from hydroforge.compiler.partition import (
        GroupRankLookup,
        PartitionCompiler,
    )
    from hydroforge.compiler.model import FieldOwner, _ModelSemanticPlan
    from hydroforge.compiler.statistics_binding import (
        DisabledStatisticsBinding,
        StatisticsBindingCompiler,
    )
    from hydroforge.data.model_input import ModelInput
    from hydroforge.execution.parameters import ParameterChangeEffect
    from hydroforge.execution.parameters import ParameterPlanRuntime
    from hydroforge.execution.progress import ProgressRuntime
    from hydroforge.execution.runtime import ModelExecution
    from hydroforge.output.checkpoint import CheckpointRuntime
    from hydroforge.contracts.fields import PartitionSchema


_STATISTICS_QUERY_CONTEXT = "hydroforge_statistics_model"
_MODEL_METHOD_CONTEXT = "hydroforge_model_method"


def _xpu_supports_fp64(device: torch.device) -> bool:
    """Return compiler-relevant XPU FP64 support or fail before lowering."""

    runtime = getattr(torch, "xpu", None)
    properties_getter = getattr(runtime, "get_device_properties", None)
    if properties_getter is None:
        raise RuntimeError(
            "this PyTorch XPU runtime cannot report FP64 capability; "
            "HydroForge cannot safely select float64 Triton storage"
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
            f"XPU device {str(device)!r} did not expose an exact has_fp64 "
            "capability; HydroForge cannot safely select float64 Triton storage"
        )
    return supported


def _default_mixed_precision(
    backend: str,
    device: torch.device,
    *,
    xpu_supports_fp64: bool | None = None,
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


def _qualified_type_name(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _distributed_date_signature(
    value: datetime | cftime.datetime | None,
) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (
        _qualified_type_name(type(value)),
        canonical_calendar(getattr(value, "calendar", "standard")),
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        getattr(value, "fold", None),
        getattr(value, "has_year_zero", None),
    )


def _distributed_array_signature(value: np.ndarray) -> tuple[Any, ...]:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(array)
    return (
        "numpy",
        canonical.dtype.str,
        tuple(array.shape),
        sha256(canonical.view(np.uint8).tobytes()).hexdigest(),
    )


def _distributed_tensor_signature(value: torch.Tensor) -> tuple[Any, ...]:
    canonical = value.detach().to(device="cpu").contiguous().reshape(-1)
    payload = canonical.view(torch.uint8).numpy().tobytes()
    return (
        "torch",
        str(value.dtype),
        tuple(value.shape),
        sha256(payload).hexdigest(),
    )


def _distributed_value_signature(value: Any) -> Any:
    """Encode one declaration as stable, equality-safe Python primitives."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return ("float", value.hex())
    if isinstance(value, (datetime, cftime.datetime)):
        return _distributed_date_signature(value)
    if type(value) is timedelta:
        return ("timedelta", value.days, value.seconds, value.microseconds)
    if isinstance(value, Path):
        return ("path", str(value.absolute()))
    if isinstance(value, torch.device):
        return ("device", value.type)
    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    if isinstance(value, torch.Tensor):
        return _distributed_tensor_signature(value)
    if isinstance(value, np.ndarray):
        return _distributed_array_signature(value)
    if isinstance(value, np.generic):
        return _distributed_array_signature(np.asarray(value))
    if isinstance(value, HydroForgeModel):
        fields = object.__getattribute__(value, "__dict__")
        return (
            "model",
            _qualified_type_name(type(value)),
            tuple(
                (name, _distributed_value_signature(fields[name]))
                for name in type(value).model_fields
            ),
        )
    if isinstance(value, Mapping):
        entries = tuple(
            (
                _distributed_value_signature(key),
                _distributed_value_signature(item),
            )
            for key, item in value.items()
        )
        return (
            "mapping",
            tuple(sorted(entries, key=lambda item: repr(item[0]))),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(map(_distributed_value_signature, value)))
    if isinstance(value, (set, frozenset)):
        items = tuple(map(_distributed_value_signature, value))
        return ("set", tuple(sorted(items, key=repr)))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    raise TypeError(
        "distributed runtime declarations cannot contain unsupported value "
        f"{type(value).__name__}"
    )


def _distributed_schedule_signature(
    schedule: SimulationSchedule | None,
) -> tuple[Any, ...] | None:
    """Digest explicit schedules without publishing every step object."""

    if schedule is None:
        return None
    if schedule._is_regular:
        return (
            "regular",
            schedule.calendar,
            _distributed_date_signature(schedule.regular_start),
            _distributed_date_signature(schedule.regular_end),
            _distributed_value_signature(schedule.regular_step),
            _distributed_value_signature(schedule.source_interval),
            _distributed_value_signature(schedule.spinup),
            schedule.num_spinup_steps,
            schedule.num_main_steps,
        )

    digest = sha256()
    for step in schedule.explicit_steps:
        encoded = json.dumps(
            _distributed_value_signature(step),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return (
        "explicit",
        schedule.calendar,
        len(schedule.explicit_steps),
        _distributed_date_signature(schedule.execution_start),
        _distributed_date_signature(schedule.execution_end),
        digest.hexdigest(),
    )


class _ModelClassDeclaration(HydroForgeModel):
    """Validated subclass-authoring declaration for ``AbstractModel``."""

    backend_requirements: Mapping[str, BackendRequirement]
    module_requirements: Mapping[str, ModuleRequirement]
    module_names: frozenset[str]
    partition_key: str | None
    partition_group: str
    cuda_extension_modules: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_declaration(self) -> Self:
        supported = {"torch", "cuda", "triton", "metal"}
        unknown_backends = set(self.backend_requirements).difference(supported)
        if unknown_backends:
            raise ValueError(
                f"backend_requirements has unknown backends: {sorted(unknown_backends)}"
            )
        unknown_modules = set(self.module_requirements).difference(
            self.module_names,
        )
        if unknown_modules:
            raise ValueError(
                f"module_requirements names unknown modules: {sorted(unknown_modules)}"
            )
        if self.partition_key is not None and not self.partition_key.isidentifier():
            raise ValueError("partition_key must be a Python identifier or None")
        if not self.partition_group.isidentifier():
            raise ValueError("partition_group must be a Python identifier")
        invalid_catalogs = tuple(
            name
            for name in self.cuda_extension_modules
            if not name
            or any(not component.isidentifier() for component in name.split("."))
        )
        if invalid_catalogs:
            raise ValueError(
                "cuda_extension_modules must contain dotted Python module "
                f"names; invalid={invalid_catalogs}"
            )
        if len(self.cuda_extension_modules) != len(set(self.cuda_extension_modules)):
            raise ValueError("cuda_extension_modules must not contain duplicates")
        object.__setattr__(
            self,
            "backend_requirements",
            MappingProxyType(dict(self.backend_requirements)),
        )
        object.__setattr__(
            self,
            "module_requirements",
            MappingProxyType(dict(self.module_requirements)),
        )
        return self


def _statistics_query_model(info: ValidationInfo) -> "AbstractModel":
    context = info.context
    if not isinstance(context, Mapping):
        raise ValueError("statistics query requires model context")
    model = context.get(_STATISTICS_QUERY_CONTEXT)
    if model is None:
        raise ValueError("statistics query requires model context")
    return model


class _StatisticsCollectionQuery(HydroForgeModel):
    """Validated request for one in-memory statistics collection view."""

    as_stacked: bool = True

    @model_validator(mode="after")
    def _validate_query(self, info: ValidationInfo) -> Self:
        model = _statistics_query_model(info)
        if model._statistics_plan is None:
            raise ValueError("model has no statistics declaration")
        if not model.in_memory_output:
            raise ValueError("statistics result access requires in_memory_output=True")
        return self


class _StatisticsItemQuery(HydroForgeModel):
    """Validated lookup of one output already declared by StatisticsPlan."""

    variable_name: str
    operation: str = "mean"
    as_stacked: bool = True
    access: Literal["result", "accumulator", "pop"]

    @model_validator(mode="after")
    def _validate_query(self, info: ValidationInfo) -> Self:
        model = _statistics_query_model(info)
        if model._statistics_plan is None:
            raise ValueError("model has no statistics declaration")
        declared = {
            (output.name, output.operation)
            for output in model._statistics_outputs
            if output.operation != "static"
        }
        key = (self.variable_name, self.operation)
        if key not in declared:
            raise ValueError(
                f"statistics output {self.variable_name!r}/"
                f"{self.operation!r} is not declared"
            )
        if self.access in {"result", "pop"} and not model.in_memory_output:
            raise ValueError("statistics result access requires in_memory_output=True")
        return self


class _SaveStateRequest(HydroForgeModel):
    @model_validator(mode="after")
    def _validate_checkpoint_support(self, info: ValidationInfo) -> Self:
        model = (
            info.context.get(_MODEL_METHOD_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if model is None:
            raise ValueError("save_state requires model context")
        if model.num_trials is not None:
            raise ValueError("checkpoint save currently requires a non-ensemble model")
        return self


class AbstractModel(HydroForgeModel, ABC):
    """
    Generic master controller for hydroforge models using the AbstractModule hierarchy.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
        ignored_types=(_KernelField, ModuleReference),
    )

    # Class variables
    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = MappingProxyType(
        {}
    )
    module_requirements: ClassVar[Mapping[str, ModuleRequirement]] = MappingProxyType(
        {}
    )
    partition_key: ClassVar[Optional[str]] = None
    partition_group: ClassVar[str] = "group_id"
    cuda_extension_modules: ClassVar[tuple[str, ...]] = ()
    # Instance fields
    experiment_name: str = Field(
        default="experiment",
        description="Name of the experiment",
    )
    input_proxy: InputProxy = Field(
        default=...,
        description="InputProxy object containing model data",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path("./out"),
        description="Path to the output directory",
    )
    opened_modules: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of active modules",
    )
    variables_to_save: Mapping[str, tuple[str | Mapping[str, str], ...]] = Field(
        default_factory=dict,
        description=(
            "Statistics outputs as {operation: [field or {alias: expression}]}."
        ),
    )
    materialized_outputs: tuple[str, ...] = Field(
        default=(),
        description=(
            "Output-capable fields to keep resident for direct consumers "
            "without registering a statistics writer"
        ),
    )
    precision: Literal["float32", "float64"] = Field(
        default="float32",
        description="Base precision of the model",
    )
    statistics_save_precision: Optional[Literal["float32", "float64"]] = Field(
        default="float32",
        description=(
            "Floating-point precision used for persisted statistics; None "
            "preserves each statistics tensor's resolved precision."
        ),
    )
    mixed_precision: Optional[bool] = Field(
        default=None,
        strict=True,
        description=(
            "Enable mixed precision for hpfloat (storage) tensors.\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  float32 → float64, float64 → float64 (no promotion).\n"
            "If omitted, defaults to enabled for CUDA/ROCm and for XPU "
            "Triton devices that report FP64 support; it is disabled for "
            "other backends and XPU devices without FP64."
        ),
    )
    execution_mode: Literal["auto", "eager"] = Field(
        default="auto",
        description=(
            "Execution scheduling policy. 'auto' selects the cached native "
            "capture supported by the active device; 'eager' keeps every "
            "launch directly observable for differentiation and debugging."
        ),
    )
    device: torch.device = Field(
        default=torch.device("cpu"),
        description="Device for tensors (e.g., 'cuda:0', 'xpu:0', 'cpu')",
    )
    event_sink: EventSink = Field(
        default_factory=ConsoleEventSink,
        description="Structured lifecycle/progress event destination",
    )
    BLOCK_SIZE: Optional[int] = Field(
        default=None,
        description=(
            "Global GPU block-size override. None lets each kernel select its "
            "backend default."
        ),
        ge=1,
        le=1024,
        strict=True,
    )
    output_workers: int = Field(
        default=2,
        ge=0,
        strict=True,
        description="Number of workers for writing output files",
    )
    output_split_by_year: bool = Field(
        default=False,
        strict=True,
        description="Whether to split output files by year",
    )
    num_trials: Optional[int] = Field(
        default=None,
        ge=2,
        strict=True,
        description="Number of parallel simulations (ensemble members)",
    )
    trial_forcing_fields: Mapping[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description=(
            "Construction-time trial-batched forcing fields grouped by module; "
            "unlisted forcing fields remain shared"
        ),
    )
    save_kernels: bool = Field(
        default=False,
        strict=True,
        description="Whether to save generated Triton kernels",
    )
    max_pending_steps: int = Field(
        default=200,
        ge=1,
        strict=True,
        description="Maximum number of pending time steps for output buffering",
    )
    max_pending_output_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=1,
        strict=True,
        description=(
            "Maximum aggregate bytes retained by streaming output buffers and "
            "submitted writer tasks"
        ),
    )
    initial_time: Optional[Union[datetime, cftime.datetime]] = Field(
        default=None,
        description=("Initial runtime clock when no simulation schedule is supplied"),
    )
    simulation_schedule: Optional[SimulationSchedule] = Field(
        default=None,
        description="Runtime-owned model call schedule and calendar contract",
    )
    statistics_plan: Optional[StatisticsPlan] = Field(
        default=None,
        description=(
            "Optional temporal window policy for variables_to_save; omitted "
            "means every model step"
        ),
    )
    parameter_changes: tuple[ParameterChange, ...] = Field(
        default=(),
        description="Complete immutable scheduled parameter declarations",
    )
    calendar: Optional[str] = Field(
        default=None,
        description=(
            "Calendar when no simulation schedule is configured. A schedule "
            "owns the calendar when present."
        ),
    )
    in_memory_output: bool = Field(
        default=False,
        strict=True,
        description="Store output in memory instead of writing to NC files",
    )
    result_device: Optional[torch.device] = Field(
        default=None,
        description="Device for in-memory results (default: CPU)",
    )
    output_netcdf_options: Mapping[str, Any] = Field(
        default_factory=default_netcdf_options,
        description=(
            "Additional validated keyword options passed to netCDF4 "
            "Dataset.createVariable for dynamic output variables."
        ),
    )
    checkpoint_netcdf_options: Mapping[str, Any] = Field(
        default_factory=default_netcdf_options,
        description=(
            "Validated netCDF4 Dataset.createVariable options for model "
            "checkpoint variables."
        ),
    )

    _modules: Dict[str, AbstractModule] = PrivateAttr(default_factory=dict)
    _module_links: Mapping[str, AbstractModule | None] | None = PrivateAttr(
        default=None,
    )
    _process_topology: ProcessTopology | None = PrivateAttr(default=None)
    _runtime_materialized: bool = PrivateAttr(default=False)
    _distributed_public_sequence: int = PrivateAttr(default=0)

    # Imports remain TYPE_CHECKING-only so the declarative layer does not gain
    # runtime dependencies on its compiler and execution consumers.
    _execution: ModelExecution = PrivateAttr()
    _namespace: NamespaceCompiler = PrivateAttr()
    _statistics: DisabledStatisticsBinding | StatisticsBindingCompiler = PrivateAttr()
    _checkpoint: CheckpointRuntime = PrivateAttr()
    _data: ModelDataCompiler = PrivateAttr()
    _input: ModelInput = PrivateAttr()
    _partition: PartitionCompiler = PrivateAttr()
    _field_namespace: Mapping[str, tuple[FieldOwner, ...]] = PrivateAttr()
    _parameters: ParameterPlanRuntime = PrivateAttr()
    _progress_service: ProgressRuntime = PrivateAttr()
    _current_time: Optional[Union[datetime, cftime.datetime]] = PrivateAttr(
        default=None,
    )
    _backend: str = PrivateAttr()
    _module_order: tuple[str, ...] = PrivateAttr()
    _namespace_declaration: Mapping[str, Any] = PrivateAttr()
    _statistics_declaration: _StatisticsDeclaration | None = PrivateAttr(
        default=None,
    )
    _statistics_plan: StatisticsPlan | None = PrivateAttr(default=None)
    _statistics_outputs: tuple[_StatisticsOutput, ...] = PrivateAttr(
        default=(),
    )
    _field_demand: FieldDemandPlan = PrivateAttr(
        default_factory=FieldDemandPlan.empty,
    )
    _semantic_plan: _ModelSemanticPlan = PrivateAttr()

    def _topology(self) -> ProcessTopology:
        topology = self._process_topology
        if topology is None:
            topology = ProcessTopology.capture()
            self._process_topology = topology
        return topology

    @property
    def rank(self) -> int:
        """Rank captured from the process group when this model was built."""

        return self._topology().rank

    @property
    def world_size(self) -> int:
        """World size captured from the process group when this model was built."""

        return self._topology().world_size

    @property
    def current_time(self) -> Optional[Union[datetime, cftime.datetime]]:
        """Return the private clock of the next managed model step."""

        self._ensure_runtime_materialized()
        return self._current_time

    def _set_runtime_current_time(
        self,
        value: Union[datetime, cftime.datetime],
    ) -> None:
        """Advance the private clock from the managed-step runtime."""

        self._current_time = value

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        module_types = cls._module_types()
        declaration = _ModelClassDeclaration(
            backend_requirements=cls.backend_requirements,
            module_requirements=cls.module_requirements,
            module_names=frozenset(module_types),
            partition_key=cls.partition_key,
            partition_group=cls.partition_group,
            cuda_extension_modules=cls.cuda_extension_modules,
        )
        cls.backend_requirements = declaration.backend_requirements
        cls.module_requirements = declaration.module_requirements
        cls.partition_key = declaration.partition_key
        cls.partition_group = declaration.partition_group
        cls.cuda_extension_modules = declaration.cuda_extension_modules

    @classmethod
    def _module_reference_fields(cls) -> Mapping[str, ModuleReference]:
        """Return the model's typed module declarations."""

        return ModuleReference.collect(cls)

    @classmethod
    @cache
    def _module_types(cls) -> Mapping[str, type[AbstractModule]]:
        """Return the immutable module catalog derived from declarations."""

        return MappingProxyType(
            {
                name: reference.module_type
                for name, reference in cls._module_reference_fields().items()
            }
        )

    @classmethod
    @cache
    def _compiled_schema(cls):
        """Return the immutable schema for every registered module class."""
        from hydroforge.contracts.fields import parse_module_schema

        return parse_module_schema(
            tuple(cls._module_types().values()),
            include_computed=True,
        )

    @field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, value: Any) -> Path:
        """Normalize the two explicitly supported path representations."""

        if isinstance(value, Path):
            return value
        if type(value) is str and value:
            return Path(value)
        raise ValueError("output_dir must be a non-empty exact string or Path")

    @field_validator("experiment_name", mode="before")
    @classmethod
    def _validate_experiment_name(cls, value: Any) -> str:
        """Require one directory component beneath ``output_dir``."""

        if type(value) is not str or not value:
            raise ValueError("experiment_name must be a non-empty exact string")
        if (
            value in {".", ".."}
            or Path(value).name != value
            or "/" in value
            or "\\" in value
        ):
            raise ValueError(
                "experiment_name must be one path component without separators"
            )
        return value

    @field_validator(
        "output_netcdf_options",
        "checkpoint_netcdf_options",
        mode="before",
    )
    @classmethod
    def _validate_output_netcdf_options(cls, value):
        from hydroforge.serialization.netcdf import (
            normalize_netcdf_variable_options,
        )

        return _immutable_dict(normalize_netcdf_variable_options(value))

    @field_validator("variables_to_save", mode="before")
    @classmethod
    def _validate_variables_to_save(cls, value: Any):
        """Canonicalize the original user-facing statistics declaration."""

        if type(value) is not dict:
            raise ValueError("variables_to_save must be an exact dict")
        normalized: dict[str, tuple[str | Mapping[str, str], ...]] = {}
        for operation, items in value.items():
            if type(operation) is not str or not operation:
                raise ValueError(
                    "variables_to_save operation names must be non-empty exact strings"
                )
            canonical = operation.lower()
            if canonical in normalized:
                raise ValueError(
                    "variables_to_save contains duplicate normalized "
                    f"operation {canonical!r}"
                )
            if type(items) is not list:
                raise ValueError(
                    f"variables_to_save[{operation!r}] must be an exact list"
                )
            compiled_items: list[str | Mapping[str, str]] = []
            for item in items:
                if type(item) is str:
                    if not item:
                        raise ValueError("statistics field names must be non-empty")
                    compiled_items.append(item)
                    continue
                if type(item) is not dict or len(item) != 1:
                    raise ValueError(
                        "variables_to_save items must be field names or "
                        "one-item {alias: expression} dicts"
                    )
                alias, expression = next(iter(item.items()))
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
                compiled_items.append(_immutable_dict(item))
            normalized[canonical] = tuple(compiled_items)
        return _immutable_dict(normalized)

    @field_validator("trial_forcing_fields", mode="before")
    @classmethod
    def _validate_trial_forcing_declaration(cls, value: Any):
        if type(value) is not dict:
            raise ValueError("trial_forcing_fields must be an exact dict")
        normalized: dict[str, tuple[str, ...]] = {}
        for module_name, field_names in value.items():
            if type(module_name) is not str or not module_name:
                raise ValueError(
                    "trial_forcing_fields module names must be non-empty strings"
                )
            if type(field_names) is not tuple:
                raise ValueError(
                    f"trial_forcing_fields[{module_name!r}] must be an exact tuple"
                )
            if any(
                type(field_name) is not str or not field_name
                for field_name in field_names
            ):
                raise ValueError(
                    f"trial_forcing_fields[{module_name!r}] must contain "
                    "non-empty strings"
                )
            if len(field_names) != len(set(field_names)):
                raise ValueError(
                    f"trial_forcing_fields[{module_name!r}] contains duplicates"
                )
            normalized[module_name] = field_names
        return _immutable_dict(normalized)

    @model_validator(mode="after")
    def _validate_trial_forcing_fields(self) -> Self:
        declaration = self.trial_forcing_fields
        if declaration and self.num_trials is None:
            raise ValueError("trial_forcing_fields require num_trials")
        module_types = self._module_types()
        opened = frozenset(self.opened_modules)
        for module_name, field_names in declaration.items():
            if module_name not in opened:
                raise ValueError(
                    f"trial forcing module {module_name!r} is not open"
                )
            module_type = module_types[module_name]
            for field_name in field_names:
                schema = module_type._get_tensor_schema(field_name)
                if schema is None or schema.tensor is None:
                    raise ValueError(
                        f"unknown trial forcing field "
                        f"{module_name}.{field_name}"
                    )
                if schema.tensor.category != "forcing":
                    raise ValueError(
                        f"trial forcing field {module_name}.{field_name} has "
                        f"category {schema.tensor.category!r}, expected 'forcing'"
                    )
                if not tensor_is_active(schema.tensor, self.opened_modules):
                    raise ValueError(
                        f"trial forcing field {module_name}.{field_name} is inactive"
                    )
        object.__setattr__(
            self,
            "trial_forcing_fields",
            _immutable_dict(declaration),
        )
        return self

    @model_validator(mode="after")
    def _validate_module_requirements(self) -> Self:
        for name in self.opened_modules:
            rule = self.module_requirements.get(
                name,
                DEFAULT_MODULE_REQUIREMENT,
            )
            if not rule.trials and self.num_trials is not None:
                raise ValueError(f"module {name!r} does not support ensemble trials")
        return self

    @model_validator(mode="after")
    def _validate_runtime_declaration(self) -> Self:
        """Canonicalize every model/runtime choice before initialization."""

        from hydroforge.kernels.registry import (
            _backend_device_types,
            _resolve_model_backend_trusted,
        )

        backend = _resolve_model_backend_trusted(self.device)
        self._backend = backend
        required_devices = _backend_device_types(backend)
        if (
            required_devices is not None
            and self.device.type not in required_devices
        ):
            required_label = (
                repr(required_devices[0])
                if len(required_devices) == 1
                else " or ".join(repr(item) for item in required_devices)
            )
            raise ValueError(
                f"HydroForge backend {backend!r} requires a "
                f"{required_label} model device, got {str(self.device)!r}"
            )

        mixed_precision = self.mixed_precision
        needs_xpu_fp64_capability = (
            self.device.type == "xpu"
            and (
                mixed_precision is None
                or mixed_precision
                or self.precision == "float64"
            )
        )
        xpu_supports_fp64 = (
            _xpu_supports_fp64(self.device)
            if needs_xpu_fp64_capability
            else None
        )
        if mixed_precision is None:
            mixed_precision = _default_mixed_precision(
                backend,
                self.device,
                xpu_supports_fp64=xpu_supports_fp64,
            )
            object.__setattr__(self, "mixed_precision", mixed_precision)
        if (
            self.device.type == "xpu"
            and (self.precision == "float64" or mixed_precision)
            and xpu_supports_fp64 is False
        ):
            raise ValueError(
                f"XPU device {str(self.device)!r} does not support FP64, but "
                "the model requests float64 storage through precision or "
                "mixed_precision"
            )

        if self.result_device is None:
            object.__setattr__(self, "result_device", torch.device("cpu"))

        if self.variables_to_save:
            plan = (
                StatisticsPlan()
                if self.statistics_plan is None
                else self.statistics_plan
            )
        else:
            if self.statistics_plan is not None:
                raise ValueError(
                    "statistics_plan requires a non-empty variables_to_save"
                )
            plan = None
        schedule = self.simulation_schedule
        if (
            plan is not None
            and schedule is None
            and not (
                isinstance(plan.inner, EveryStep)
                and isinstance(plan._effective_outer, EveryStep)
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
            if self.statistics_plan is not None:
                object.__setattr__(self, "statistics_plan", plan)
            validate_statistics_window_schedule(plan, schedule)
        self._statistics_plan = plan

        if schedule is not None:
            if self.initial_time is not None:
                raise ValueError(
                    "initial_time must not be configured together with "
                    "simulation_schedule"
                )
            if self.calendar is not None:
                configured = canonical_calendar(self.calendar)
                if configured != schedule.calendar:
                    raise ValueError(
                        f"model calendar {configured!r} differs from "
                        f"simulation schedule {schedule.calendar!r}"
                    )
            calendar = schedule.calendar
        else:
            calendar, normalized, _defaulted = normalize_calendar_dates(
                {"model initial_time": self.initial_time},
                calendar=self.calendar,
            )
            object.__setattr__(
                self,
                "initial_time",
                normalized["model initial_time"],
            )
        object.__setattr__(self, "calendar", calendar)

        runtime_rule = RUNTIME_BACKEND_REQUIREMENTS.get(
            backend,
            DEFAULT_BACKEND_REQUIREMENT,
        )
        model_rule = self.backend_requirements.get(
            backend,
            DEFAULT_BACKEND_REQUIREMENT,
        )
        runtime_rule._validate_precision(
            self.precision,
            mixed_precision,
            backend=backend,
        )
        model_rule._validate_precision(
            self.precision,
            mixed_precision,
            backend=backend,
        )
        if self.BLOCK_SIZE is not None or backend == "metal":
            block_size = _effective_block_size(
                self.BLOCK_SIZE,
                backend=backend,
            )
            if backend == "metal":
                object.__setattr__(self, "BLOCK_SIZE", block_size)
            model_rule._validate_block_size(block_size, backend=backend)
        if not model_rule.trials and self.num_trials is not None:
            raise ValueError(f"backend {backend!r} does not support ensemble trials")
        return self

    @model_validator(mode="after")
    def _compile_module_order(self) -> Self:
        """Validate and freeze the dependency order of opened modules."""

        self._module_order = self._resolved_module_order()
        return self

    def _resolved_module_order(self) -> tuple[str, ...]:
        """Return the deterministic dependency order for validated modules."""

        from graphlib import CycleError, TopologicalSorter

        module_types = self._module_types()
        opened = frozenset(self.opened_modules)
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for name in self.opened_modules:
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
                "opened module references must form an acyclic construction "
                f"graph: {error.args[1]}"
            ) from error

    @model_validator(mode="after")
    def _compile_output_tensor_activation(self) -> Self:
        """Resolve output requests to their concrete field dependencies."""

        if self._statistics_plan is None and not self.materialized_outputs:
            self._field_demand = FieldDemandPlan.empty()
            return self

        opened = frozenset(self.opened_modules)
        schema = type(self)._compiled_schema()
        module_types = self._module_types()
        qualified: dict[str, Any] = {}
        bare: dict[str, Any] = {}
        virtual: set[str] = set()
        ambiguous: set[str] = set()

        def install_bare(field: Any) -> None:
            name = field.name
            expression_virtual = bool(
                field.tensor.category == "virtual"
                and field.tensor.expression
            )
            if expression_virtual:
                if name not in virtual:
                    bare[name] = field
                    virtual.add(name)
                ambiguous.discard(name)
                return
            if name in virtual or name in ambiguous:
                return
            if name in bare:
                bare.pop(name)
                ambiguous.add(name)
                return
            bare[name] = field

        for module_name in self.opened_modules:
            excluded = set(module_types[module_name].nc_excluded_fields)
            for field in schema.fields(module_name):
                tensor = field.tensor
                if (
                    tensor is None
                    or field.excluded
                    or field.name in excluded
                    or not all(
                        dependency in opened
                        for dependency in tensor.depends_on
                    )
                ):
                    continue
                qualified[f"{module_name}.{field.name}"] = field
                install_bare(field)
            # Reference indices may appear in virtual expressions but not schema.fields.
            for field_name in module_types[module_name]._reference_index_fields():
                field = module_types[module_name]._get_tensor_schema(field_name)
                if field is None:
                    raise ValueError(
                        f"ReferenceIndexField {module_name}.{field_name} "
                        "has no tensor schema"
                    )
                qualified[f"{module_name}.{field.name}"] = field
                install_bare(field)

        # Propagate virtual-output demand before module construction.
        required: dict[str, set[str]] = {}
        observed: dict[str, set[str]] = {}
        known = set(qualified) | set(bare)
        visited: set[str] = set()

        def field_key(field: Any) -> str:
            return f"{field.module_name}.{field.name}"

        def source_dependencies(source: Any) -> tuple[str, ...]:
            if isinstance(source, TensorSource):
                return (source.name,)
            if isinstance(source, ExpressionSource):
                return source.expression.dependencies
            return (*source.value.dependencies, source.index)

        def resolve_field(name: str) -> Any | None:
            # Statistics use bare names; aliases may use qualified names.
            return bare.get(name) or qualified.get(name)

        def visit_field(field: Any) -> None:
            key = field_key(field)
            if key in visited:
                return
            visited.add(key)
            observed.setdefault(field.module_name, set()).add(field.name)
            required.setdefault(field.module_name, set()).add(field.name)
            tensor = field.tensor
            if (
                tensor is None
                or tensor.category != "virtual"
                or not tensor.expression
            ):
                return
            source = parse_value_source(tensor.expression, known)
            for dependency in source_dependencies(source):
                dependency_field = resolve_field(dependency)
                if dependency_field is not None:
                    visit_field(dependency_field)

        for name in self.materialized_outputs:
            field = qualified.get(name) if "." in name else bare.get(name)
            if field is None:
                raise ValueError(
                    f"materialized output {name!r} is unknown, ambiguous, "
                    "or inactive"
                )
            if field.tensor.output == "disabled":
                raise ValueError(
                    f"materialized output {name!r} is disabled for output"
                )
            visit_field(field)

        for items in self.variables_to_save.values():
            for item in items:
                direct = isinstance(item, str)
                name = item if direct else next(iter(item))
                field = qualified.get(name) if "." in name else bare.get(name)
                if direct:
                    if field is None:
                        continue
                    tensor = field.tensor
                    if tensor.output == "disabled":
                        raise ValueError(
                            f"statistics field {name!r} is disabled for output"
                        )
                    visit_field(field)
                    continue

                # Activate every concrete dependency of an alias expression.
                expression = next(iter(item.values()))
                if field is not None:
                    tensor = field.tensor
                    if (
                        tensor.depends_on
                        or tensor.required_by
                        or tensor.output_only
                    ):
                        observed.setdefault(field.module_name, set()).add(
                            field.name,
                        )
                        # Active conditional fields take precedence over aliases.
                        if tensor_is_active(
                            tensor,
                            self.opened_modules,
                            output_required=False,
                        ):
                            continue
                source = parse_value_source(expression, known)
                for dependency in source_dependencies(source):
                    dependency_field = resolve_field(dependency)
                    if dependency_field is not None:
                        visit_field(dependency_field)

        self._field_demand = FieldDemandPlan.from_sets(required, observed)
        return self

    def _is_tensor_field_active(
        self,
        module_name: str,
        field: Any,
    ) -> bool:
        """Resolve one field against the frozen model output specialization."""

        output_required = self._field_demand.is_required(
            module_name, field.name,
        )
        tensor = field.tensor
        return tensor_is_active(
            tensor,
            self.opened_modules,
            output_required=output_required,
        )

    @cached_property
    def dtype(self) -> torch.dtype:
        _dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
        }
        return _dtype_map[self.precision]

    @cached_property
    def output_full_dir(self) -> Path:
        output_full_dir = self.output_dir / self.experiment_name
        return output_full_dir

    @cached_property
    def log_path(self) -> Path:
        log_path = self.output_full_dir / "log.txt"
        return log_path

    @model_validator(mode="after")
    def _validate_namespace(self) -> Self:
        """
        Check for namespace conflicts across all opened modules.

        Virtual fields with an ``expr`` (scatter / plain aggregation outputs)
        are allowed to share a name with their source counterpart in another
        module — this is the standard subcell→cell aggregation pattern.
        """
        field_definitions = {}
        schema = self._compiled_schema()
        module_types = self._module_types()
        for module_name in self.opened_modules:
            excluded = set(module_types[module_name].nc_excluded_fields)
            for field in schema.fields(module_name):
                if field.tensor is not None:
                    unknown_dependencies = sorted(
                        set(
                            (*field.tensor.depends_on, *field.tensor.required_by),
                        ).difference(module_types)
                    )
                    if unknown_dependencies:
                        raise ValueError(
                            f"Tensor field {module_name}.{field.name} depends "
                            "on unknown modules: "
                            f"{unknown_dependencies}"
                        )
                if field.excluded or field.name in excluded:
                    continue
                if field.tensor is not None and not self._is_tensor_field_active(
                    module_name, field,
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
                    if new_virtual and not old_virtual:
                        field_definitions[field.name] = field
                    continue
                if (
                    field.tensor is not None
                    and previous.tensor is not None
                    and field.tensor.category == "init_state"
                    and previous.tensor.category == "init_state"
                ):
                    raise ValueError(
                        f"checkpoint state name {field.name!r} is declared by "
                        f"both {previous.module_name!r} and {module_name!r}; "
                        "state ownership must be unique"
                    )
                if (
                    field.annotation != previous.annotation
                    or field.tensor != previous.tensor
                ):
                    raise ValueError(
                        f"Namespace conflict for {field.name!r}: "
                        f"{previous.module_name} and {module_name} declare "
                        "different types or tensor metadata"
                    )
        self._namespace_declaration = MappingProxyType(field_definitions)
        return self

    def _materialize_runtime(self) -> None:
        """Build private runtime services from a validated model identity."""

        from hydroforge.compiler.initialization import ModelInitializer

        schedule = self.simulation_schedule
        self._current_time = (
            schedule.execution_start if schedule is not None else self.initial_time
        )

        self._prepare_output_directory()
        ModelInitializer(self).run()

    def _ensure_runtime_materialized(self) -> None:
        """Materialize runtime state as one rank-synchronous transaction."""

        if self._runtime_materialized:
            return
        if self.world_size > 1:
            self._coordinate_runtime_materialization_preflight()
        initialization_error: BaseException | None = None
        try:
            self._materialize_runtime()
        except BaseException as error:
            initialization_error = error

        if self.world_size > 1:
            self._coordinate_runtime_materialization(initialization_error)
            return

        if initialization_error is not None:
            self._discard_runtime_materialization()
            raise initialization_error
        self._runtime_materialized = True

    def _distributed_input_schema_signature(self) -> tuple[Any, ...]:
        """Describe external storage shape without hashing physical fields."""

        proxy = self.input_proxy
        return tuple(
            (
                name,
                self._input.get_var_shape(name),
                str(proxy._get_var_dtype(name)),
                self._semantic_plan.input_axes.get(name),
                self._semantic_plan.variable_groups.get(name),
            )
            for name in sorted(self._input.fields)
            if name in self._input
        )

    def _distributed_input_storage_signature(self) -> tuple[Any, ...]:
        """Identify active resident values and lazy source declarations."""

        proxy = self.input_proxy
        resident = dict(proxy._resident_items())
        fields: list[tuple[Any, ...]] = []
        for name in sorted(self._input.fields):
            if name not in self._input:
                continue
            if name in resident:
                fields.append((
                    name,
                    "resident",
                    _distributed_value_signature(resident[name]),
                ))
                continue
            source = proxy.sources[name]
            identity = source.file_identity
            fields.append((
                name,
                "netcdf",
                source.dimensions,
                source.shape,
                source.dtype,
                source.alignment_dim,
                _distributed_value_signature(source.alignment_indices),
                identity.size,
                identity.mtime_ns,
            ))
        return (
            tuple(fields),
            tuple(sorted(proxy.injected_vars)),
        )

    def _distributed_partition_identity_signature(self) -> tuple[Any, ...]:
        """Hash values that decide rank ownership and reference routing."""

        schema = self._semantic_plan.partition_schema
        names = set(schema.coordinates)
        if self.partition_group in self._input:
            names.add(self.partition_group)
        for name, metadata in schema.fields.items():
            if metadata.references or metadata.partition_by or metadata.selects:
                names.add(name)
        proxy = self.input_proxy
        return tuple(
            (
                name,
                _distributed_value_signature(proxy._get_value_trusted(name)),
            )
            for name in sorted(names)
            if name in proxy
        )

    def _distributed_runtime_declaration_signature(self) -> tuple[Any, ...]:
        """Return the complete rank-shared model control-plane identity."""

        module_types = self._module_types()
        return (
            ("model", _qualified_type_name(type(self))),
            (
                "modules",
                tuple(
                    (name, _qualified_type_name(module_types[name]))
                    for name in self.opened_modules
                ),
            ),
            ("module_order", self._module_order),
            ("partition", self.partition_key, self.partition_group),
            ("cuda_catalogs", self.cuda_extension_modules),
            ("backend", self._backend),
            ("device_type", self.device.type),
            ("precision", self.precision, self.mixed_precision),
            ("execution", self.execution_mode, self.BLOCK_SIZE),
            (
                "trials",
                self.num_trials,
                _distributed_value_signature(self.trial_forcing_fields),
            ),
            (
                "output",
                self.experiment_name,
                str(self.output_full_dir.absolute()),
                self.output_workers,
                self.output_split_by_year,
                self.max_pending_steps,
                self.max_pending_output_bytes,
                self.save_kernels,
                self.in_memory_output,
                cast(torch.device, self.result_device).type,
            ),
            ("calendar", self.calendar),
            ("initial_time", _distributed_date_signature(self.initial_time)),
            (
                "schedule",
                _distributed_schedule_signature(self.simulation_schedule),
            ),
            (
                "statistics",
                _distributed_value_signature(self.variables_to_save),
                _distributed_value_signature(self._statistics_plan),
                self.statistics_save_precision,
            ),
            (
                "netcdf",
                _distributed_value_signature(self.output_netcdf_options),
                _distributed_value_signature(self.checkpoint_netcdf_options),
            ),
            (
                "parameter_changes",
                _distributed_value_signature(self.parameter_changes),
            ),
            ("input_schema", self._distributed_input_schema_signature()),
            ("input_storage", self._distributed_input_storage_signature()),
            (
                "partition_identity",
                self._distributed_partition_identity_signature(),
            ),
        )

    def _coordinate_runtime_materialization_preflight(self) -> None:
        """Prove every rank will initialize the same runtime transaction."""

        signature: tuple[Any, ...] | None = None
        local_error: BaseException | None = None
        try:
            signature = self._distributed_runtime_declaration_signature()
        except BaseException as error:
            local_error = error
        failures = self._gather_distributed_failures(
            local_error,
            phase="runtime.materialization.preflight",
            signature=signature,
        )
        if not any(failure is not None for failure in failures):
            return
        if local_error is not None:
            raise local_error
        raise distributed_failure_error(
            "distributed model runtime declaration validation",
            failures,
        )

    def _distributed_compiled_runtime_signature(self) -> tuple[Any, ...]:
        """Describe rank-invariant services produced by initialization."""

        return (
            self._execution.backend,
            self._execution.capture_mode,
            self._checkpoint.plan.layout_signature,
            tuple(sorted(
                descriptor.protocol_name
                for descriptor in self._execution.step_policies
            )),
        )

    def _install_distributed_output_run_id(
        self, payloads: tuple[Any, ...],
    ) -> None:
        """Install the rank-zero output identity exchanged at runtime commit."""

        if len(payloads) != self.world_size:
            raise RuntimeError(
                "distributed runtime materialization returned an invalid "
                "output run-ID payload count"
            )
        run_id = payloads[0]
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError(
                "distributed runtime materialization did not publish a "
                "non-empty output run ID from rank zero"
            )
        if any(payload is not None for payload in payloads[1:]):
            raise RuntimeError(
                "distributed runtime materialization received output run IDs "
                "from nonzero ranks"
            )
        statistics = getattr(self, "_statistics", None)
        aggregator = getattr(statistics, "aggregator", None)
        if aggregator is not None:
            aggregator.run_id = run_id

    def _gather_distributed_failures(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
    ) -> tuple[dict[str, str] | None, ...]:
        """Publish one phase-tagged public transaction result to every rank."""

        failures, _payloads = self._exchange_distributed_public_transaction(
            error,
            phase=phase,
            signature=signature,
        )
        return failures

    def _exchange_distributed_public_transaction(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
        payload: Any = None,
    ) -> tuple[
        tuple[dict[str, str] | None, ...],
        tuple[Any, ...],
    ]:
        """Exchange one tagged transaction record and optional phase payload."""

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "multi-rank model transactions require an "
                "initialized torch.distributed process group"
            )
        if type(phase) is not str or not phase:
            raise RuntimeError(
                "distributed public transaction phase must be a non-empty string"
            )
        sequence = self._distributed_public_sequence
        self._distributed_public_sequence = sequence + 1
        local = (
            sequence,
            phase,
            signature,
            None if error is None else failure_description(error),
            payload,
        )
        observed: list[Any] = [None] * self.world_size
        dist.all_gather_object(
            observed,
            local,
        )
        if any(
            not isinstance(value, tuple)
            or len(value) != 5
            or type(value[0]) is not int
            or type(value[1]) is not str
            for value in observed
        ):
            raise RuntimeError(
                "distributed public transaction protocol received a malformed "
                f"record: {observed!r}"
            )
        identities = tuple((value[0], value[1]) for value in observed)
        if len(set(identities)) != 1:
            raise RuntimeError(
                "distributed public phase mismatch across ranks: "
                f"{identities!r}"
            )
        failures = tuple(value[3] for value in observed)
        if not any(failure is not None for failure in failures):
            signatures = tuple(value[2] for value in observed)
            if any(value != signatures[0] for value in signatures[1:]):
                raise RuntimeError(
                    "distributed public transaction inputs differ across "
                    f"ranks during {phase!r}: {signatures!r}"
                )
        return (
            cast(tuple[dict[str, str] | None, ...], failures),
            tuple(value[4] for value in observed),
        )

    def _release_runtime_materialization(self) -> None:
        """Release every service created by one materialization attempt."""

        failures: list[BaseException] = []
        statistics = getattr(self, "_statistics", None)
        execution = getattr(self, "_execution", None)
        if statistics is not None:
            try:
                statistics.close()
            except BaseException as error:
                failures.append(error)
        if execution is not None:
            try:
                execution.close()
            except BaseException as error:
                failures.append(error)
        self._discard_runtime_materialization()
        self._runtime_materialized = False
        if failures:
            error = ResourceCleanupError("model resources", failures)
            raise error from failures[0]

    def _coordinate_runtime_materialization(
        self,
        initialization_error: BaseException | None,
    ) -> None:
        """Commit initialization only after every rank reports success."""

        compiled_signature: tuple[Any, ...] | None = None
        if initialization_error is None:
            try:
                compiled_signature = self._distributed_compiled_runtime_signature()
            except BaseException as error:
                initialization_error = error
        run_id_candidate: str | None = None
        if initialization_error is None and self.rank == 0:
            try:
                run_id_candidate = str(uuid4())
            except BaseException as error:
                initialization_error = error
        try:
            initialization_failures, run_id_payloads = (
                self._exchange_distributed_public_transaction(
                    initialization_error,
                    phase="runtime.materialization",
                    signature=compiled_signature,
                    payload=run_id_candidate,
                )
            )
        except BaseException as coordination_error:
            cleanup_error: BaseException | None = None
            try:
                self._release_runtime_materialization()
            except BaseException as error:
                cleanup_error = error
            failures = tuple(
                error for error in (
                    initialization_error,
                    coordination_error,
                    cleanup_error,
                )
                if error is not None
            )
            if len(failures) == 1:
                raise failures[0]
            error = ResourceCleanupError(
                "distributed model initialization coordination",
                failures,
            )
            raise error from coordination_error

        if not any(failure is not None for failure in initialization_failures):
            try:
                self._install_distributed_output_run_id(run_id_payloads)
            except BaseException as error:
                initialization_error = error
                description = failure_description(error)
                initialization_failures = tuple(
                    description for _ in range(self.world_size)
                )
            else:
                self._runtime_materialized = True
                return

        cleanup_error: BaseException | None = None
        try:
            self._release_runtime_materialization()
        except BaseException as error:
            cleanup_error = error
        try:
            cleanup_failures = self._gather_distributed_failures(
                cleanup_error,
                phase="runtime.materialization.cleanup",
            )
        except BaseException as coordination_error:
            primary = (
                initialization_error
                if initialization_error is not None
                else distributed_failure_error(
                    "distributed model initialization",
                    initialization_failures,
                )
            )
            failures = [primary, coordination_error]
            if cleanup_error is not None:
                failures.append(cleanup_error)
            error = ResourceCleanupError(
                "distributed model initialization cleanup coordination",
                failures,
            )
            raise error from primary

        primary = (
            initialization_error
            if initialization_error is not None
            else distributed_failure_error(
                "distributed model initialization",
                initialization_failures,
            )
        )
        if any(failure is not None for failure in cleanup_failures):
            cleanup_failure = (
                cleanup_error
                if cleanup_error is not None
                else distributed_failure_error(
                    "distributed model initialization cleanup",
                    cleanup_failures,
                )
            )
            error = ResourceCleanupError(
                "distributed model initialization rollback",
                (primary, cleanup_failure),
            )
            raise error from primary
        raise primary

    def _discard_runtime_materialization(self) -> None:
        """Remove every object derived from one materialized runtime."""

        self._modules.clear()
        self._module_links = None
        for name in ("_variable_map", "group_id_to_rank"):
            self.__dict__.pop(name, None)

    def _ensure_healthy_runtime(self) -> None:
        """Enter trusted runtime only after a public request has validated."""

        self._ensure_runtime_materialized()
        failure = self._execution.failure
        if failure is not None:
            raise self._execution.poisoned_error(failure)

    def _prepare_output_directory(self) -> None:
        """Acquire the output directory only after validation completes."""

        if self.rank != 0:
            return
        if not self.output_full_dir.exists():
            self.output_full_dir.mkdir(parents=True, exist_ok=True)
            return
        emit(
            self,
            "warning",
            "output.directory_exists",
            "Output directory already exists; contents may be overwritten",
            directory=self.output_full_dir,
        )

    def initialize_model_state(self) -> None:
        """Initialize ordered model state inside HydroForge's transaction.

        HydroForge does not invoke module initialization hooks automatically;
        model authors explicitly call any module helpers here in physical order.
        """

    def _update_module_structures(self):
        """Call every module in declared order and commit one staged update."""

        from hydroforge.model.structure import StructuralUpdateContext

        context = StructuralUpdateContext(self)
        for module_name in self.opened_modules:
            self._modules[module_name].update_structure(context)
        return context.commit()

    def update_structure(self):
        """Run the ordered module structure pass between managed steps."""

        self._ensure_healthy_runtime()
        return self._update_module_structures()

    def commit_structural_update(self, replacements: Any):
        """Commit staged tensor storage and rebuild dimension consumers.

        The update is restricted to between-step cold paths. Symbolic
        dimension changes are inferred from declared tensor shapes, while
        tensor identities remain stable for model namespaces and observers.
        """

        from hydroforge.model.structure import commit_structural_update

        return commit_structural_update(self, replacements)

    def print_memory_summary(self) -> None:
        """
        Print a summary of memory usage by module.

        Each variable is attributed to the first module where it appears;
        duplicates (shared tensors) are skipped so total is never over-counted.
        """
        if self.rank != 0:
            return
        total_memory = 0
        global_seen_ptrs: set = set()
        module_memory: Dict[str, float] = {}

        for module_name in self.opened_modules:
            if module_name not in self._modules:
                continue
            module = self._modules[module_name]

            # Count only tensors not yet seen globally
            module_bytes = 0
            for field in module.tensor_schema():
                name = field.name
                # Skip computed fields that haven't been materialized yet
                # to avoid triggering @cached_property (lazy allocation).
                if field.computed and name not in module.__dict__:
                    continue
                if not hasattr(module, name):
                    continue
                value = getattr(module, name)
                if (
                    isinstance(value, torch.Tensor)
                    and value.device.type == module.device.type
                ):
                    ptr = value.data_ptr()
                    if ptr not in global_seen_ptrs:
                        global_seen_ptrs.add(ptr)
                        module_bytes += value.element_size() * value.nelement()

            total_memory += module_bytes
            module_memory[module_name] = module_bytes / (1024 * 1024)

        # Add StatisticsRuntime memory usage
        aggregator_mem = self._statistics.memory_usage()
        total_memory += aggregator_mem
        if aggregator_mem:
            module_memory["StatisticsAggregator"] = aggregator_mem / (1024 * 1024)

        emit(
            self,
            "info",
            "model.memory",
            "Model memory summary",
            rank=self.rank,
            modules=module_memory,
            total_mb=total_memory / (1024 * 1024),
        )

    @property
    def _partition_metadata(self) -> PartitionSchema:
        return self._semantic_plan.partition_schema

    @property
    def _variable_group_mapping(self) -> Mapping[str, str]:
        return self._semantic_plan.variable_groups

    @cached_property
    def _variable_map(self) -> Mapping[str, NamespaceEntry]:
        """
        Map variable names to immutable owner and coordinate metadata.
        This provides a unified way to lookup variables across all modules.

        Qualified forms (``module.field``) are always available. An
        expression-backed virtual field owns the unqualified name; otherwise,
        fields declared by multiple modules remain ambiguous and omit it.
        """
        return self._namespace.build()

    @cached_property
    def group_id_to_rank(self) -> GroupRankLookup:
        self._ensure_runtime_materialized()
        return self._partition.group_ranks

    def close(self) -> None:
        """Atomically release output workers and backend execution resources."""

        if not self._runtime_materialized:
            return
        self._release_runtime_materialization()

    def _execute_parameter_changes(
        self,
        current_time: Union[datetime, cftime.datetime],
    ) -> ParameterChangeEffect:
        return self._parameters.execute_parameter_change_plan(current_time)

    def _progress_start(self) -> None:
        self._progress_service.begin_step()

    def _progress_tick(self) -> bool:
        return self._progress_service.progress_tick()

    def _format_progress(self) -> str:
        return self._progress_service.format_progress()

    def get_output_results(
        self,
        as_stacked: bool = True,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """
        Get the in-memory output results (only available when in_memory_output=True).

        Args:
            as_stacked: If True (default), stack all time steps into a single tensor.
                       If False, return list of per-time-step tensors.

        Returns:
            Dictionary mapping output names to result tensors.
            Shape (when stacked): (time_steps, *actual_shape)

        Raises:
            RuntimeError: If not in in_memory_output mode or aggregator not initialized.
        """
        query = _StatisticsCollectionQuery.model_validate(
            {"as_stacked": as_stacked},
            context={_STATISTICS_QUERY_CONTEXT: self},
        )
        self._ensure_healthy_runtime()
        statistics = cast("StatisticsBindingCompiler", self._statistics)
        return statistics.results(stacked=query.as_stacked)

    def get_output_result(
        self,
        variable_name: str,
        op: str = "mean",
        as_stacked: bool = True,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Get a specific output result tensor by variable name and operation.

        Args:
            variable_name: Name of the variable
            op: Operation type (mean, max, min, last, etc.)
            as_stacked: If True (default), stack all time steps into a single tensor.

        Returns:
            Result tensor for the specified variable and operation.

        Raises:
            RuntimeError: If not in in_memory_output mode or aggregator not initialized.
            KeyError: If the specified variable/op combination doesn't exist.
        """
        query = _StatisticsItemQuery.model_validate(
            {
                "variable_name": variable_name,
                "operation": op,
                "as_stacked": as_stacked,
                "access": "result",
            },
            context={_STATISTICS_QUERY_CONTEXT: self},
        )
        self._ensure_healthy_runtime()
        statistics = cast("StatisticsBindingCompiler", self._statistics)
        return statistics.result(
            query.variable_name,
            query.operation,
            stacked=query.as_stacked,
        )

    def get_output_time_index(self) -> int:
        """Get the current output time index (number of finalized time steps)."""
        self._ensure_healthy_runtime()
        return self._statistics.time_index()

    def get_output_accumulator(
        self,
        variable_name: str,
        operation: str = "mean",
    ) -> torch.Tensor:
        """Return a differentiable snapshot without exposing captured storage."""

        query = _StatisticsItemQuery.model_validate(
            {
                "variable_name": variable_name,
                "operation": operation,
                "access": "accumulator",
            },
            context={_STATISTICS_QUERY_CONTEXT: self},
        )
        self._ensure_healthy_runtime()
        statistics = cast("StatisticsBindingCompiler", self._statistics)
        return statistics.accumulator(
            query.variable_name,
            query.operation,
        )

    def pop_output_result(
        self,
        variable_name: str,
        operation: str = "mean",
    ) -> torch.Tensor | None:
        """Pop the newest in-memory result without retaining its history."""

        query = _StatisticsItemQuery.model_validate(
            {
                "variable_name": variable_name,
                "operation": operation,
                "access": "pop",
            },
            context={_STATISTICS_QUERY_CONTEXT: self},
        )
        self._ensure_healthy_runtime()
        statistics = cast("StatisticsBindingCompiler", self._statistics)
        return statistics.pop_result(
            query.variable_name,
            query.operation,
        )

    def reset_output_time_index(self) -> None:
        """Reset the output time index to 0 for a new simulation run (in-memory mode only)."""
        self._ensure_healthy_runtime()
        self._statistics.reset_time_index()

    def shard_param(self) -> Dict[str, Any]:
        """Load and rank-slice parameters through the internal data service."""
        return self._data.shard()

    def save_state(self) -> InputProxy:
        """Persist a complete construction input at the committed clock."""
        validation_error: BaseException | None = None
        try:
            _SaveStateRequest.model_validate(
                {},
                context={_MODEL_METHOD_CONTEXT: self},
            )
        except BaseException as error:
            validation_error = error
        if self.world_size > 1:
            failures = self._gather_distributed_failures(
                validation_error,
                phase="checkpoint.save.api-validation",
            )
            if any(failure is not None for failure in failures):
                if validation_error is not None:
                    raise validation_error
                raise distributed_failure_error(
                    "distributed checkpoint save entry validation",
                    failures,
                )
        elif validation_error is not None:
            raise validation_error
        self._ensure_healthy_runtime()
        return self._checkpoint.save()

    @field_validator("opened_modules", mode="before")
    @classmethod
    def _validate_modules(cls, v: Any) -> tuple[str, ...]:
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
                "Missing required model modules in opened_modules: "
                f"{missing_model_modules}. Available modules: {v}"
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
                    f"Module '{module}' declares references to unknown modules: "
                    f"{unknown_references}. Available modules: "
                    f"{list(module_types)}"
                )
            required = module_class._required_modules()
            missing_deps = [dep for dep in required if dep not in v]
            if missing_deps:
                raise ValueError(
                    f"Module '{module}' has missing required modules in "
                    f"opened_modules: {missing_deps}. "
                    f"Required modules: {required}. "
                    f"Available modules: {v}"
                )
            present_conflicts = [
                conflict
                for conflict in module_class.conflicts
                if conflict in v and conflict != module
            ]
            if present_conflicts:
                raise ValueError(
                    f"Module '{module}' conflicts with modules present in opened_modules: "
                    f"{present_conflicts}. These modules cannot be enabled together."
                )
        return v

    @model_validator(mode="after")
    def _validate_statistics_outputs(
        self,
    ) -> Self:
        cls = type(self)
        plan = self._statistics_plan
        if plan is None:
            return self
        outputs: list[_StatisticsOutput] = []
        expressions: dict[str, str | None] = {}
        pairs: set[tuple[str, str]] = set()
        for operation, items in self.variables_to_save.items():
            for item in items:
                if isinstance(item, str):
                    name = item
                    expression = None
                else:
                    name, expression = next(iter(item.items()))
                output = _StatisticsOutput(
                    name=name,
                    operation=operation,
                    expression=expression,
                )
                pair = (output.name, output.operation)
                if pair in pairs:
                    raise ValueError(
                        "variables_to_save must not repeat a field/operation"
                    )
                pairs.add(pair)
                previous = expressions.setdefault(
                    output.name,
                    output.expression,
                )
                if previous != output.expression:
                    raise ValueError(
                        f"statistics output {output.name!r} has conflicting expressions"
                    )
                outputs.append(output)
        if not any(output.operation != "static" for output in outputs):
            raise ValueError("variables_to_save requires at least one dynamic output")
        self._statistics_outputs = tuple(outputs)
        opened_modules = self.opened_modules
        fields: dict[str, Any] = {}
        virtual_fields: set[str] = set()
        ambiguous: set[str] = set()
        schema = cls._compiled_schema()

        def install_field(module_name: str, field: Any) -> None:
            fields[f"{module_name}.{field.name}"] = field
            tensor = field.tensor
            expression_virtual = bool(
                tensor is not None
                and tensor.category == "virtual"
                and tensor.expression
            )
            if expression_virtual:
                if field.name not in virtual_fields:
                    fields[field.name] = field
                    virtual_fields.add(field.name)
                ambiguous.discard(field.name)
                return
            if field.name in virtual_fields or field.name in ambiguous:
                return
            if field.name in fields:
                fields.pop(field.name)
                ambiguous.add(field.name)
                return
            fields[field.name] = field

        module_types = cls._module_types()
        for module_name in opened_modules:
            for field in schema.fields(module_name):
                tensor = field.tensor
                if tensor is None:
                    continue
                # Keep inactive output-only metadata visible to expressions.
                if (
                    not self._is_tensor_field_active(module_name, field)
                    and not tensor.output_only
                ):
                    continue
                install_field(module_name, field)
            module_type = module_types[module_name]
            for field_name in module_type._reference_index_fields():
                field = module_type._get_tensor_schema(field_name)
                if field is None:
                    raise ValueError(
                        f"ReferenceIndexField {module_name}.{field_name} "
                        "has no tensor schema"
                    )
                install_field(module_name, field)
        known = set(fields)
        def active_declared_field(name: str) -> bool:
            field = fields.get(name)
            if field is None or field.tensor is None:
                return False
            return self._is_tensor_field_active(field.module_name, field)

        selection_targets = {
            field.tensor.selects.split(".")[-1]
            for field in fields.values()
            if field.tensor is not None and field.tensor.selects
        }

        def metadata(name: str) -> tuple[Any, ...]:
            tensor = fields[name].tensor
            coordinate = tensor.dim_coords
            if coordinate:
                coordinate = coordinate.split(".")[-1]
            return (
                tuple(
                    dimension.rsplit(".", 1)[-1]
                    if isinstance(dimension, str)
                    else dimension
                    for dimension in tensor.shape
                ),
                tensor.output,
                coordinate,
                tensor.dtype,
                tensor.category,
            )

        def dependencies(source: Any) -> tuple[str, ...]:
            if isinstance(source, TensorSource):
                return (source.name,)
            if isinstance(source, ExpressionSource):
                return source.expression.dependencies
            return (
                *source.value.dependencies,
                source.index,
            )

        def validate_expression(
            *,
            name: str,
            expression: str,
            target_metadata: tuple[Any, ...] | None,
            allow_scatter: bool,
        ) -> tuple[Any, ...]:
            source = parse_value_source(expression, known)
            if isinstance(source, ScatterSource) and not allow_scatter:
                raise ValueError(
                    "ad-hoc scatter statistics must be declared as a "
                    "computed tensor field"
                )
            names = (
                source.value.dependencies
                if isinstance(source, ScatterSource)
                else dependencies(source)
            )
            if not names:
                raise ValueError(
                    f"statistics expression {name!r} has no field dependency"
                )
            forcing_dependencies = tuple(
                dependency
                for dependency in names
                if metadata(dependency)[4] == "forcing"
            )
            if forcing_dependencies:
                raise ValueError(
                    f"statistics expression {name!r} depends on forcing fields "
                    f"{forcing_dependencies}; forcing layout is run-specific "
                    "and cannot define persistent output storage"
                )
            reference_name = next(
                (
                    dependency
                    for dependency in names
                    if metadata(dependency)[3] != "bool"
                ),
                names[0],
            )
            reference = metadata(reference_name)
            definite_trial_layouts: set[str] = set()
            for dependency in names:
                observed = metadata(dependency)
                incompatible = (
                    observed[0] != reference[0]
                    or observed[2] != reference[2]
                    or (
                        observed[3] != "bool"
                        and observed[3] != reference[3]
                    )
                )
                if incompatible:
                    raise ValueError(
                        f"statistics expression {name!r} mixes incompatible "
                        f"field metadata: {reference_name!r} has {reference}, but "
                        f"{dependency!r} has {observed}"
                    )
                if self.num_trials is not None:
                    if observed[4] in {"state", "init_state"}:
                        definite_trial_layouts.add("batched")
                    elif observed[4] in {"topology", "shared_state"}:
                        definite_trial_layouts.add("shared")
            if len(definite_trial_layouts) > 1:
                raise ValueError(
                    f"statistics expression {name!r} mixes shared and "
                    "trial-batched fields"
                )
            coordinates = {
                metadata(dependency)[2]
                for dependency in names
                if metadata(dependency)[2] is not None
            }
            if len(coordinates) > 1:
                raise ValueError(
                    f"statistics expression {name!r} mixes coordinate axes "
                    f"{sorted(coordinates)}"
                )
            target = None if target_metadata is None else target_metadata[2]
            if (
                not isinstance(source, ScatterSource)
                and coordinates
                and target is not None
                and target not in coordinates
            ):
                raise ValueError(
                    f"statistics field {name!r} declares dim_coords={target!r}, "
                    f"but its expression uses {next(iter(coordinates))!r}"
                )
            if len(reference[0]) < 1:
                raise ValueError(
                    f"statistics expression {name!r} must have at least one "
                    "logical dimension"
                )
            if len(reference[0]) > 2:
                raise ValueError(
                    f"statistics expression {name!r} has logical rank "
                    f"{len(reference[0])}; only rank <= 2 is supported"
                )
            shared_categories = {
                metadata(dependency)[4]
                for dependency in names
                if metadata(dependency)[4] in {"topology", "shared_state"}
            }
            if self.num_trials is not None and shared_categories:
                raise ValueError(
                    f"dynamic statistics expression {name!r} uses shared "
                    f"field categories {sorted(shared_categories)!r} in a "
                    "multi-trial model"
                )
            if isinstance(source, ScatterSource):
                index_tensor = fields[source.index].tensor
                index_coordinate = index_tensor.dim_coords
                if index_coordinate:
                    index_coordinate = index_coordinate.rsplit(".", 1)[-1]
                if coordinates and index_coordinate not in coordinates:
                    raise ValueError(
                        f"statistics scatter index {source.index!r} uses "
                        f"coordinate {index_coordinate!r}, but its values use "
                        f"{next(iter(coordinates))!r}"
                    )
                if (
                    len(index_tensor.shape) != 1
                    or index_tensor.dtype not in {"idx", "int"}
                    or index_tensor.category != "topology"
                ):
                    raise ValueError(
                        f"statistics scatter index {source.index!r} must be a "
                        "shared one-dimensional topology integer field"
                    )
                if len(reference[0]) != 1:
                    raise ValueError(
                        f"statistics scatter value {name!r} must be one-dimensional"
                    )
                target_field = fields.get(name)
                target_tensor = None if target_field is None else target_field.tensor
                target_coord = (
                    None if target_tensor is None else target_tensor.dim_coords
                )
                bare_target = (
                    None if target_coord is None else target_coord.split(".")[-1]
                )
                if (
                    target_tensor is None
                    or target_tensor.output != "auto"
                    or bare_target not in selection_targets
                ):
                    raise ValueError(
                        f"scatter statistics field {name!r} requires an "
                        "output selection on its declared coordinate"
                    )
            value_expression = (
                source.value
                if isinstance(source, ScatterSource)
                else source.expression
                if isinstance(source, ExpressionSource)
                else None
            )
            if value_expression is not None:
                from hydroforge.contracts.fields import concrete_tensor_dtype

                validate_expression_constants(
                    name,
                    value_expression,
                    concrete_tensor_dtype(
                        (
                            reference[3]
                            if target_metadata is None
                            else target_metadata[3]
                        ),
                        self.dtype,
                        self.mixed_precision,
                    ),
                )
            return reference if target_metadata is None else target_metadata

        def compile_operation(
            output_name: str,
            operation: str,
            field_metadata: tuple[Any, ...],
        ) -> Any:
            parsed = parse_operation(operation)
            shape, output, coordinate, dtype, _category = field_metadata
            if dtype not in {"float", "hpfloat"}:
                unsupported = (
                    parsed.inner is None
                    and parsed.outer
                    in {
                        Reduction.MEAN,
                        Reduction.SUM,
                    }
                ) or (
                    parsed.inner is not None
                    and (
                        parsed.inner
                        in {
                            Reduction.MEAN,
                            Reduction.SUM,
                            Reduction.MAX,
                            Reduction.MIN,
                        }
                        or parsed.outer
                        in {
                            Reduction.MEAN,
                            Reduction.SUM,
                        }
                        or parsed.k > 1
                    )
                )
                if unsupported:
                    raise ValueError(
                        f"statistics operation {operation!r} for non-floating "
                        f"field {output_name!r} is unsupported"
                    )

            selected = output == "auto" and coordinate in selection_targets
            if not selected and (parsed.k > 1 or parsed.stores_index):
                raise ValueError(
                    f"full-output statistics field {output_name!r} does not "
                    f"support top-k or arg operation {operation!r}"
                )
            if (
                selected
                and len(shape) == 2
                and (parsed.compound or parsed.k > 1 or parsed.stores_index)
            ):
                raise ValueError(
                    f"indexed-level statistics field {output_name!r} does not "
                    f"support compound, top-k, or arg operation {operation!r}"
                )
            return parsed

        def compile_output_options(
            output_name: str,
            field_metadata: tuple[Any, ...],
        ) -> Mapping[str, Any]:
            from hydroforge.contracts.fields import concrete_tensor_dtype
            from hydroforge.data.distributed import torch_to_numpy_dtype
            from hydroforge.serialization.netcdf import (
                _prepare_netcdf_variable_options_trusted,
                netcdf_dtype_encoding,
            )

            shape, _output, _coordinate, dtype, category = field_metadata
            batched = bool(
                self.num_trials is not None and category in {"state", "init_state"}
            )
            chunks = self.output_netcdf_options.get("chunksizes")
            if (
                chunks is not None
                and self.num_trials is not None
                and category in {"param", "derived_param", "virtual"}
            ):
                raise ValueError(
                    f"NetCDF chunksizes for statistics output {output_name!r} "
                    "cannot be fixed because its trial batching depends on "
                    "materialized parameter/expression storage"
                )
            dimensions = tuple(
                f"axis_{index}" for index in range(1 + len(shape) + int(batched))
            )
            tensor_dtype = concrete_tensor_dtype(
                dtype,
                self.dtype,
                self.mixed_precision,
            )
            saved_dtype = tensor_dtype
            if tensor_dtype.is_floating_point and self.statistics_save_precision:
                saved_dtype = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                }[self.statistics_save_precision]
            storage_dtype, logical_dtype = netcdf_dtype_encoding(
                torch_to_numpy_dtype(saved_dtype),
            )
            options = _prepare_netcdf_variable_options_trusted(
                self.output_netcdf_options,
                dtype=storage_dtype,
                dimensions=dimensions,
                name=output_name,
                logical_dtype=logical_dtype,
            )
            return MappingProxyType(dict(options))

        compiled_operations: dict[tuple[str, str], Any] = {}
        compiled_output_metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
        compiled_netcdf_options: dict[str, Mapping[str, Any]] = {}
        for output in outputs:
            if output.operation == "static":
                operation = None
            else:
                operation = output.operation
            if output.expression is not None and not active_declared_field(
                output.name,
            ):
                expression_metadata = validate_expression(
                    name=output.name,
                    expression=output.expression,
                    target_metadata=None,
                    allow_scatter=False,
                )
                compiled_operations[(output.name, operation)] = compile_operation(
                    output.name,
                    operation,
                    expression_metadata,
                )
                compiled_output_metadata[(output.name, operation)] = expression_metadata
                compiled_netcdf_options[f"{output.name}_{operation}"] = (
                    compile_output_options(
                        f"{output.name}_{operation}",
                        expression_metadata,
                    )
                )
                continue
            field = fields[output.name]
            tensor = field.tensor
            if output.expression is not None and not (
                tensor.depends_on
                or tensor.required_by
                or tensor.output_only
            ):
                raise ValueError(
                    f"statistics alias {output.name!r} shadows an "
                    "unconditional model field"
                )
            if tensor.output == "disabled" and output.expression is None:
                raise ValueError(
                    f"statistics field {output.name!r} is disabled for output"
                )
            if output.operation == "static" and (
                len(tensor.shape) != 1 or tensor.dim_coords is None
            ):
                raise ValueError(
                    f"static statistics field {output.name!r} must be "
                    "one-dimensional and declare dim_coords"
                )
            if output.operation != "static" and tensor.category not in {
                "state",
                "shared_state",
                "init_state",
                "param",
                "virtual",
            }:
                raise ValueError(
                    f"statistics field {output.name!r} has unsupported "
                    f"category {tensor.category!r}"
                )
            if output.operation != "static" and not tensor.shape:
                raise ValueError(
                    f"statistics field {output.name!r} must have at least one "
                    "logical dimension"
                )
            if output.operation != "static" and len(tensor.shape) > 2:
                raise ValueError(
                    f"statistics field {output.name!r} has logical rank "
                    f"{len(tensor.shape)}; only rank <= 2 is supported"
                )
            if (
                output.operation != "static"
                and self.num_trials is not None
                and tensor.category in {"topology", "shared_state"}
            ):
                raise ValueError(
                    f"dynamic statistics field {output.name!r} is shared in "
                    "a multi-trial model"
                )
            if tensor.category == "virtual" and tensor.expression:
                validate_expression(
                    name=output.name,
                    expression=tensor.expression,
                    target_metadata=metadata(output.name),
                    allow_scatter=True,
                )
            if operation is not None:
                field_metadata = metadata(output.name)
                compiled_operations[(output.name, operation)] = compile_operation(
                    output.name,
                    operation,
                    field_metadata,
                )
                compiled_output_metadata[(output.name, operation)] = field_metadata
                compiled_netcdf_options[f"{output.name}_{operation}"] = (
                    compile_output_options(
                        f"{output.name}_{operation}",
                        field_metadata,
                    )
                )

        virtual_graph: dict[str, tuple[str, ...]] = {}
        for name, field in fields.items():
            tensor = field.tensor
            if tensor.category != "virtual" or not tensor.expression:
                continue
            virtual_graph[name] = dependencies(
                parse_value_source(tensor.expression, known),
            )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited or name not in virtual_graph:
                return
            if name in visiting:
                raise ValueError(f"cyclic statistics dependency involving {name!r}")
            visiting.add(name)
            for dependency in virtual_graph[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in tuple(virtual_graph):
            visit(name)

        grouped_operations: dict[str, list[Any]] = {}
        compiled_sources: dict[str, Any] = {}
        kernel_inputs: set[str] = set()
        safe_outputs: dict[str, str] = {}
        generated_output_names: set[str] = set()

        def resolve_declared_source(name: str) -> Any:
            existing = compiled_sources.get(name)
            if existing is not None:
                return existing
            tensor = fields[name].tensor
            expression = (
                tensor.expression
                if tensor.category == "virtual" and tensor.expression
                else None
            )
            source = (
                TensorSource(name)
                if expression is None
                else parse_value_source(expression, known)
            )
            compiled_sources[name] = source
            source_dependencies = (
                source.value.dependencies
                if isinstance(source, ScatterSource)
                else source.expression.dependencies
                if isinstance(source, ExpressionSource)
                else ()
            )
            for dependency in source_dependencies:
                dependency_field = fields[dependency]
                dependency_tensor = dependency_field.tensor
                if (
                    dependency_tensor.category == "virtual"
                    and dependency_tensor.expression
                ):
                    resolve_declared_source(dependency)
            return source

        for output in outputs:
            if output.operation == "static":
                continue
            grouped_operations.setdefault(output.name, []).append(
                compiled_operations[(output.name, output.operation)],
            )
            if output.expression is None or active_declared_field(output.name):
                source = resolve_declared_source(output.name)
            else:
                source = parse_value_source(output.expression, known)
                compiled_sources[output.name] = source
                source_dependencies = (
                    source.value.dependencies
                    if isinstance(source, ScatterSource)
                    else source.expression.dependencies
                    if isinstance(source, ExpressionSource)
                    else ()
                )
                for dependency in source_dependencies:
                    dependency_tensor = fields[dependency].tensor
                    if (
                        dependency_tensor.category == "virtual"
                        and dependency_tensor.expression
                    ):
                        resolve_declared_source(dependency)
            kernel_inputs.update(dependencies(source))
            output_name = f"{output.name}_{output.operation}"
            safe_name = sanitize_symbol(output_name)
            if not safe_name:
                raise ValueError(
                    f"statistics output {output_name!r} has no valid NetCDF characters"
                )
            previous = safe_outputs.get(safe_name)
            if previous is not None and previous != output_name:
                raise ValueError(
                    f"statistics outputs {previous!r} and {output_name!r} "
                    f"both map to NetCDF variable {safe_name!r}"
                )
            safe_outputs[safe_name] = output_name
            operation = compiled_operations[(output.name, output.operation)]
            if operation.k > 1:
                output_storage_names = {
                    f"{safe_name}_{index}" for index in range(operation.k)
                }
            else:
                output_storage_names = {safe_name}
            generated_output_names.update(output_storage_names)

            output_metadata = compiled_output_metadata[(output.name, output.operation)]
            coordinate = output_metadata[2]
            if (
                coordinate is not None
                and sanitize_symbol(coordinate) in output_storage_names
            ):
                raise ValueError(
                    f"statistics output {output_name!r} conflicts with its "
                    f"NetCDF coordinate {coordinate!r}"
                )

        for static_name in (
            output.name for output in outputs if output.operation == "static"
        ):
            safe_static = sanitize_symbol(static_name)
            if safe_static == "time":
                raise ValueError(
                    f"static statistics field {static_name!r} conflicts with "
                    "the reserved NetCDF time variable"
                )
            if safe_static in generated_output_names:
                raise ValueError(
                    f"static statistics field {static_name!r} conflicts with "
                    "a generated NetCDF output variable"
                )

        storage_names: set[str] = set()
        for name, operations in grouped_operations.items():
            storage = build_variable_storage_plan(
                name,
                (),
                tuple(operations),
            )
            storage_names.update(slot.name for slot in storage.slots)

        reserved_collision = kernel_inputs.intersection(
            RESERVED_CONTROL_STATE,
        )
        if reserved_collision:
            raise ValueError(
                "statistics input names collide with reserved control state: "
                f"{sorted(reserved_collision)}"
            )
        storage_collision = kernel_inputs.intersection(storage_names)
        if storage_collision:
            raise ValueError(
                "statistics inputs collide with generated accumulator state: "
                f"{sorted(storage_collision)}"
            )
        symbols: dict[str, str] = {}
        for name in sorted(
            kernel_inputs | storage_names | set(grouped_operations),
        ):
            symbol = sanitize_symbol(name)
            previous = symbols.get(symbol)
            if previous is not None and previous != name:
                raise ValueError(
                    f"statistics names {previous!r} and {name!r} both map "
                    f"to generated symbol {symbol!r}"
                )
            symbols[symbol] = name
        self._statistics_declaration = _StatisticsDeclaration(
            program=StatisticsProgram(
                operations=MappingProxyType(
                    {
                        name: tuple(operations)
                        for name, operations in grouped_operations.items()
                    }
                ),
                sources=MappingProxyType(dict(compiled_sources)),
            ),
            static_names=tuple(
                output.name for output in outputs if output.operation == "static"
            ),
            netcdf_options=MappingProxyType(dict(compiled_netcdf_options)),
        )
        return self

    @field_validator("parameter_changes")
    @classmethod
    def _validate_parameter_changes(
        cls,
        changes: tuple[ParameterChange, ...],
        info: ValidationInfo,
    ) -> tuple[ParameterChange, ...]:
        if not changes:
            return changes
        schedule = info.data.get("simulation_schedule")
        if schedule is None:
            raise ValueError(
                "parameter_changes require simulation_schedule so every "
                "change can be resolved to an exact managed step"
            )
        normalized_changes: list[ParameterChange] = []
        for change in changes:
            _calendar, normalized, _defaulted = normalize_calendar_dates(
                {
                    f"parameter change {change.variable!r} start": (
                        change.start
                    ),
                },
                calendar=schedule.calendar,
            )
            start = normalized[
                f"parameter change {change.variable!r} start"
            ]
            bound = ParameterChange(
                variable=change.variable,
                start=start,
                active_steps=change.active_steps,
                delta=change._trusted_value("delta"),
                target_value=change._trusted_value("target_value"),
                target_ids=change._trusted_value("target_ids"),
                target_id_field=change.target_id_field,
            )
            try:
                main_index = schedule._main_index_at(bound.start)
            except KeyError:
                raise ValueError(
                    f"parameter change {bound.variable!r} start "
                    f"{bound.start!r} is not a main simulation step boundary"
                ) from None
            remaining_steps = schedule.num_main_steps - main_index
            if bound.active_steps > remaining_steps:
                raise ValueError(
                    f"parameter change {bound.variable!r} active_steps="
                    f"{bound.active_steps} exceeds the {remaining_steps} "
                    "main simulation step(s) remaining from its start"
                )
            normalized_changes.append(bound)

        return tuple(normalized_changes)

    @model_validator(mode="after")
    def _validate_input_contract(self) -> Self:
        """Bind external storage to the complete validated model schema.

        This is pure semantic validation.  It may inspect the already
        validated ``InputProxy`` identity, but it does not allocate model
        tensors, initialize a backend, open persistent resources or start the
        execution runtime.
        """

        from hydroforge.compiler.model import _ModelSemanticPlan
        from hydroforge.compiler.parameters import ParameterSemanticCompiler
        from hydroforge.compiler.partition import _PartitionSemanticCompiler
        from hydroforge.data.model_input import ModelInput

        self._input = ModelInput(self)
        partition = _PartitionSemanticCompiler(self)
        partition_schema = partition.schema
        variable_groups = partition.variable_groups
        input_axes = self._input.compile_partition_axes(partition)
        partition.validate_global_reference_integrity()
        reference_targets, inverse_sources = partition.compile_reference_targets()
        partition.validate_inverse_reference_integrity(inverse_sources)
        parameter_changes = ParameterSemanticCompiler(
            self,
            partition,
            input_axes=input_axes,
        ).compile(self.parameter_changes)
        self._semantic_plan = _ModelSemanticPlan(
            backend=self._backend,
            module_order=self._module_order,
            namespace=self._namespace_declaration,
            input_binding=self._input,
            partition_schema=partition_schema,
            variable_groups=variable_groups,
            input_axes=input_axes,
            reference_targets=reference_targets,
            trial_forcing_fields=self.trial_forcing_fields,
            field_demand=self._field_demand,
            statistics=self._statistics_declaration,
            parameter_changes=parameter_changes,
        )
        return self

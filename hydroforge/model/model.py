# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from datetime import datetime
from functools import cache, cached_property
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    cast,
)

import cftime
import torch
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.compiler.namespace import NamespaceEntry
from hydroforge.contracts.events import ConsoleEventSink, EventSink, emit
from hydroforge.contracts.fields import FieldDemandPlan, tensor_is_active
from hydroforge.contracts.kernel_field import _KernelField
from hydroforge.contracts.naming import DottedPath, Identifier
from hydroforge.contracts.options import OptionsConfig
from hydroforge.contracts.parameters import ParameterChange
from hydroforge.contracts.runtime import (
    BackendRequirement,
    ModuleRequirement,
)
from hydroforge.contracts.step_fields import StepFieldProvider
from hydroforge.contracts.temporal import (
    SimulationSchedule,
    StatisticsPlan,
    _StatisticsOutput,
    normalize_calendar_dates,
)
from hydroforge.contracts.validation import (
    FrozenMapping,
    HydroForgeModel,
    _immutable_dict,
)
from hydroforge.data.distributed import ProcessTopology
from hydroforge.data.input import InputProxy
from hydroforge.data.parallel import EnsembleParallel
from hydroforge.model.module import AbstractModule, ModuleReference
from hydroforge.serialization.netcdf import default_netcdf_options
from hydroforge.statistics.ir import (
    Expression,
    _StatisticsDeclaration,
)

if TYPE_CHECKING:
    from hydroforge.compiler.data import ModelDataCompiler
    from hydroforge.compiler.model import FieldOwner, _ModelSemanticPlan
    from hydroforge.compiler.namespace import NamespaceCompiler
    from hydroforge.compiler.partition import (
        GroupRankLookup,
        PartitionCompiler,
    )
    from hydroforge.compiler.statistics_binding import (
        DisabledStatisticsBinding,
        StatisticsBindingCompiler,
    )
    from hydroforge.contracts.fields import PartitionSchema
    from hydroforge.data.model_input import ModelInput
    from hydroforge.execution.lifecycle import RuntimeLifecycle
    from hydroforge.execution.parameters import (
        ParameterChangeEffect,
        ParameterPlanRuntime,
    )
    from hydroforge.execution.progress import ProgressRuntime
    from hydroforge.execution.runtime import ModelExecution
    from hydroforge.output.checkpoint import CheckpointRuntime


_STATISTICS_QUERY_CONTEXT = "hydroforge_statistics_model"
_MODEL_METHOD_CONTEXT = "hydroforge_model_method"
_EMPTY_STRUCTURE_HOOK = AbstractModule.update_structure


class _ModelClassDeclaration(HydroForgeModel):
    """Validated subclass-authoring declaration for ``AbstractModel``."""

    backend_requirements: FrozenMapping[
        Literal["torch", "cuda", "triton", "metal"], BackendRequirement
    ]
    module_requirements: FrozenMapping[str, ModuleRequirement]
    module_names: frozenset[str]
    partition_key: Identifier | None
    partition_group: Identifier
    cuda_extension_modules: tuple[DottedPath, ...]

    @model_validator(mode="after")
    def _validate_declaration(self) -> Self:
        unknown_modules = set(self.module_requirements).difference(
            self.module_names,
        )
        if unknown_modules:
            raise ValueError(
                f"module_requirements names unknown modules: {sorted(unknown_modules)}"
            )
        if len(self.cuda_extension_modules) != len(set(self.cuda_extension_modules)):
            raise ValueError("cuda_extension_modules must not contain duplicates")
        return self


def _statistics_query_model(info: ValidationInfo) -> AbstractModel:
    context = info.context
    if not isinstance(context, Mapping):
        raise ValueError("statistics query requires model context")
    model = context.get(_STATISTICS_QUERY_CONTEXT)
    if model is None:
        raise ValueError("statistics query requires model context")
    return model


class _StatisticsHistoryQuery(HydroForgeModel):
    """Require a declared, retained statistics history before materialization."""

    @model_validator(mode="after")
    def _validate_query(self, info: ValidationInfo) -> Self:
        model = _statistics_query_model(info)
        if model._statistics_plan is None:
            raise ValueError("model has no statistics declaration")
        if not model.in_memory_output:
            raise ValueError("statistics result access requires in_memory_output=True")
        return self


class _StatisticsCollectionQuery(_StatisticsHistoryQuery):
    """Validated request for one in-memory statistics collection view."""

    as_stacked: bool = True
    start: int | None = None
    stop: int | None = None


class _StatisticsDrainQuery(_StatisticsHistoryQuery):
    as_stacked: bool = True
    max_steps: int | None = Field(default=None, ge=0)


class _StatisticsBatchQuery(_StatisticsHistoryQuery):
    batch_size: int = Field(default=64, gt=0)

    @field_validator("batch_size", mode="before")
    @classmethod
    def _validate_exact_count(cls, value: Any) -> int:
        if type(value) is not int:
            raise ValueError("batch_size must be an exact integer")
        return value


class _StatisticsItemQuery(HydroForgeModel):
    """Validated lookup of one output already declared by StatisticsPlan."""

    variable_name: str
    operation: str = "mean"
    as_stacked: bool = True
    access: Literal["result", "accumulator", "pop"]
    start: int | None = Field(default=None, strict=True)
    stop: int | None = Field(default=None, strict=True)

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
        if model.ensemble_size is not None:
            raise ValueError("checkpoint save currently requires a non-ensemble model")
        return self


class AbstractModel(HydroForgeModel, ABC):
    """
    Generic master controller for hydroforge models using the AbstractModule hierarchy.
    """

    model_config = ConfigDict(
        ignored_types=(_KernelField, ModuleReference),
    )

    # Class variables
    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = MappingProxyType(
        {}
    )
    module_requirements: ClassVar[Mapping[str, ModuleRequirement]] = MappingProxyType(
        {}
    )
    partition_key: ClassVar[str | None] = None
    partition_group: ClassVar[str] = "group_id"
    cuda_extension_modules: ClassVar[tuple[str, ...]] = ()
    step_field_providers: ClassVar[Mapping[str, StepFieldProvider]] = MappingProxyType(
        {}
    )
    step_field_expressions: ClassVar[Mapping[str, str]] = MappingProxyType({})
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
    options: OptionsConfig = Field(
        default_factory=OptionsConfig,
        description="Immutable typed model options and constants",
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
    statistics_save_precision: Literal["float32", "float64"] | None = Field(
        default="float32",
        description=(
            "Floating-point precision used for persisted statistics; None "
            "preserves each statistics tensor's resolved precision."
        ),
    )
    mixed_precision: bool | None = Field(
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
            "capture supported by the active device. CUDA conditional graphs "
            "require a native CUDA build with CUDA >= 12.4; ROCm/HIP uses "
            "the eager fallback. 'eager' keeps every launch directly "
            "observable for differentiation and debugging."
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
    BLOCK_SIZE: int | None = Field(
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
    ensemble_size: int | None = Field(
        default=None,
        ge=2,
        strict=True,
        description="Number of parallel simulations (ensemble members)",
    )
    ensemble_forcing_fields: Mapping[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description=(
            "Construction-time member-batched forcing fields grouped by module; "
            "unlisted forcing fields remain shared"
        ),
    )
    parallel: EnsembleParallel | None = Field(default=None, exclude=True, repr=False)
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
    initial_time: datetime | cftime.datetime | None = Field(
        default=None,
        description=("Initial runtime clock when no simulation schedule is supplied"),
    )
    simulation_schedule: SimulationSchedule | None = Field(
        default=None,
        description="Runtime-owned model call schedule and calendar contract",
    )
    statistics_plan: StatisticsPlan | None = Field(
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
    calendar: str | None = Field(
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
    result_device: torch.device | None = Field(
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

    _modules: dict[str, AbstractModule] = PrivateAttr(default_factory=dict)
    _module_links: Mapping[str, AbstractModule | None] | None = PrivateAttr(
        default=None,
    )
    _process_topology: ProcessTopology | None = PrivateAttr(default=None)
    _runtime_materialized: bool = PrivateAttr(default=False)
    _lifecycle_service: RuntimeLifecycle | None = PrivateAttr(default=None)
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
    _current_time: datetime | cftime.datetime | None = PrivateAttr(
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
    _step_field_providers: Mapping[str, StepFieldProvider] = PrivateAttr()
    _step_field_expressions: Mapping[str, Expression] = PrivateAttr()

    @property
    def _runtime_lifecycle(self) -> RuntimeLifecycle:
        lifecycle = self.__pydantic_private__["_lifecycle_service"]
        if lifecycle is None:
            from hydroforge.execution.lifecycle import RuntimeLifecycle

            lifecycle = RuntimeLifecycle(self)
            self._lifecycle_service = lifecycle
        return lifecycle

    def _topology(self) -> ProcessTopology:
        topology = self.__pydantic_private__["_process_topology"]
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
    def local_ensemble_size(self) -> int | None:
        return (
            self.ensemble_size
            if self.parallel is None
            else self.parallel.local_ensemble_size
        )

    @property
    def spatial_rank(self) -> int:
        return self.rank if self.parallel is None else self.parallel.spatial_rank

    @property
    def spatial_world_size(self) -> int:
        return (
            self.world_size
            if self.parallel is None
            else self.parallel.spatial_partitions
        )

    @model_validator(mode="after")
    def _validate_parallel(self) -> Self:
        if self.parallel is not None:
            self.parallel.validate_live()
            if self.ensemble_size != self.parallel.ensemble_size:
                raise ValueError(
                    "model ensemble_size must match the ensemble process mesh"
                )
        return self

    @property
    def current_time(self) -> datetime | cftime.datetime | None:
        """Return the private clock of the next managed model step."""

        self._ensure_runtime_materialized()
        return self._current_time

    def _set_runtime_current_time(
        self,
        value: datetime | cftime.datetime,
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
        return _immutable_dict(normalized)

    @model_validator(mode="after")
    def _validate_ensemble_forcing_fields(self) -> Self:
        from hydroforge.compiler.declarations import validate_ensemble_forcing_fields

        return validate_ensemble_forcing_fields(self)

    @model_validator(mode="after")
    def _validate_module_requirements(self) -> Self:
        from hydroforge.compiler.declarations import validate_module_requirements

        return validate_module_requirements(self)

    @model_validator(mode="before")
    @classmethod
    def _include_option_required_modules(cls, data: Any) -> Any:
        from hydroforge.compiler.declarations import include_option_required_modules

        return include_option_required_modules(cls, data)

    @model_validator(mode="after")
    def _validate_option_module_requirements(self) -> Self:
        from hydroforge.compiler.declarations import validate_option_module_requirements

        return validate_option_module_requirements(self)

    @model_validator(mode="after")
    def _validate_runtime_declaration(self) -> Self:
        from hydroforge.compiler.declarations import validate_runtime_declaration

        return validate_runtime_declaration(self)

    @model_validator(mode="after")
    def _compile_module_order(self) -> Self:
        """Validate and freeze the dependency order of opened modules."""

        self._module_order = self._resolved_module_order()
        return self

    def _resolved_module_order(self) -> tuple[str, ...]:
        from hydroforge.compiler.declarations import resolved_module_order

        return resolved_module_order(self)

    @model_validator(mode="after")
    def _compile_output_tensor_activation(self) -> Self:
        from hydroforge.compiler.declarations import compile_output_tensor_activation

        return compile_output_tensor_activation(self)

    def _is_tensor_field_active(
        self,
        module_name: str,
        field: Any,
    ) -> bool:
        """Resolve one field against the frozen model output specialization."""

        output_required = self._field_demand.is_required(
            module_name,
            field.name,
        )
        tensor = field.tensor
        return tensor_is_active(
            tensor,
            self.opened_modules,
            output_required=output_required,
        )

    @cached_property
    def dtype(self) -> torch.dtype:
        return torch.float32 if self.precision == "float32" else torch.float64

    @cached_property
    def output_full_dir(self) -> Path:
        directory = self.output_dir / self.experiment_name
        if self.parallel is not None and self.parallel.ensemble_partitions > 1:
            directory /= f"ensemble_{self.parallel.ensemble_rank:04d}"
        return directory

    @cached_property
    def log_path(self) -> Path:
        return self.output_full_dir / "log.txt"

    @model_validator(mode="after")
    def _validate_namespace(self) -> Self:
        from hydroforge.compiler.declarations import validate_namespace

        return validate_namespace(self)

    def _materialize_runtime(self) -> None:
        return self._runtime_lifecycle.materialize_runtime()

    def _ensure_runtime_materialized(self) -> None:
        if self.__pydantic_private__["_runtime_materialized"]:
            return
        return self._runtime_lifecycle.ensure_runtime_materialized()

    def _distributed_input_schema_signature(self) -> tuple[Any, ...]:
        return self._runtime_lifecycle.distributed_input_schema_signature()

    def _distributed_input_storage_signature(self) -> tuple[Any, ...]:
        return self._runtime_lifecycle.distributed_input_storage_signature()

    def _distributed_partition_identity_signature(self) -> tuple[Any, ...]:
        return self._runtime_lifecycle.distributed_partition_identity_signature()

    def _distributed_runtime_declaration_signature(self) -> tuple[Any, ...]:
        return self._runtime_lifecycle.distributed_runtime_declaration_signature()

    def _coordinate_runtime_materialization_preflight(self) -> None:
        return self._runtime_lifecycle.coordinate_runtime_materialization_preflight()

    def _distributed_compiled_runtime_signature(self) -> tuple[Any, ...]:
        return self._runtime_lifecycle.distributed_compiled_runtime_signature()

    def _install_distributed_output_run_id(
        self,
        payloads: tuple[Any, ...],
    ) -> None:
        return self._runtime_lifecycle.install_distributed_output_run_id(payloads)

    def _gather_distributed_failures(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
    ) -> tuple[dict[str, str] | None, ...]:
        return self._runtime_lifecycle.gather_distributed_failures(
            error, phase=phase, signature=signature
        )

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
        return self._runtime_lifecycle.exchange_distributed_public_transaction(
            error, phase=phase, signature=signature, payload=payload
        )

    def _release_runtime_materialization(self) -> None:
        return self._runtime_lifecycle.release_runtime_materialization()

    def _coordinate_runtime_materialization(
        self,
        initialization_error: BaseException | None,
    ) -> None:
        return self._runtime_lifecycle.coordinate_runtime_materialization(
            initialization_error
        )

    def _discard_runtime_materialization(self) -> None:
        return self._runtime_lifecycle.discard_runtime_materialization()

    def _ensure_healthy_runtime(self) -> None:
        return self._runtime_lifecycle.ensure_healthy_runtime()

    def _prepare_output_directory(self) -> None:
        return self._runtime_lifecycle.prepare_output_directory()

    def initialize_model_state(self) -> None:
        """Initialize ordered model state inside HydroForge's transaction.

        Model subclasses own the complete initialization order.  This hook is
        called after all modules have been constructed and tensor modes applied,
        so a controller can explicitly sequence cross-module cold starts and
        workspace materialization inside HydroForge's transaction.
        """

    def _update_module_structures(self):
        """Call every module in declared order and commit one staged update."""

        from hydroforge.model.structure import StructuralUpdateContext

        hooks = tuple(
            hook
            for module_name in self.opened_modules
            if getattr(
                hook := self._modules[module_name].update_structure, "__func__", None
            )
            is not _EMPTY_STRUCTURE_HOOK
        )
        if not hooks:
            return None
        context = StructuralUpdateContext(self)
        for hook in hooks:
            hook(context)
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
        module_memory: dict[str, float] = {}

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
        return self._runtime_lifecycle.close()

    def _execute_parameter_changes(
        self,
        current_time: datetime | cftime.datetime,
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
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
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
            {"as_stacked": as_stacked, "start": start, "stop": stop},
            context={_STATISTICS_QUERY_CONTEXT: self},
        )
        self._ensure_healthy_runtime()
        statistics = cast("StatisticsBindingCompiler", self._statistics)
        return statistics.results(
            stacked=query.as_stacked, start=query.start, stop=query.stop
        )

    def get_output_result(
        self,
        variable_name: str,
        op: str = "mean",
        as_stacked: bool = True,
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
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
                "start": start,
                "stop": stop,
            },
            context={_STATISTICS_QUERY_CONTEXT: self},
        )
        self._ensure_healthy_runtime()
        statistics = cast("StatisticsBindingCompiler", self._statistics)
        return statistics.result(
            query.variable_name,
            query.operation,
            stacked=query.as_stacked,
            start=query.start,
            stop=query.stop,
        )

    def drain_output_results(
        self, max_steps: int | None = None, *, as_stacked: bool = True
    ):
        """Copy and release oldest retained samples; keep the simulation timeline."""

        query = _StatisticsDrainQuery.model_validate(
            {"as_stacked": as_stacked, "max_steps": max_steps},
            context={_STATISTICS_QUERY_CONTEXT: self},
        )
        self._ensure_healthy_runtime()
        return self._statistics.aggregator.drain_results(
            query.max_steps, as_stacked=query.as_stacked
        )

    def iter_output_results(self, batch_size: int = 64):
        """Read retained samples in bounded, ownership-isolated batches."""

        query = _StatisticsBatchQuery.model_validate(
            {"batch_size": batch_size}, context={_STATISTICS_QUERY_CONTEXT: self}
        )
        self._ensure_healthy_runtime()
        return self._statistics.aggregator.iter_results(query.batch_size)

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

    def shard_param(self) -> dict[str, Any]:
        """Load and rank-slice parameters through the internal data service."""
        return self._data.shard()

    def save_state(self) -> InputProxy:
        """Persist a complete construction input at the committed clock."""
        from hydroforge.execution.boundaries import coordinate_preflight

        validation_error: BaseException | None = None
        try:
            _SaveStateRequest.model_validate(
                {},
                context={_MODEL_METHOD_CONTEXT: self},
            )
        except BaseException as error:
            validation_error = error
        coordinate_preflight(
            self,
            validation_error,
            phase="checkpoint.save.api-validation",
            scope="distributed checkpoint save entry validation",
        )
        self._ensure_runtime_materialized()
        return self._checkpoint.save()

    @field_validator("opened_modules", mode="before")
    @classmethod
    def _validate_modules(cls, v: Any) -> tuple[str, ...]:
        from hydroforge.compiler.declarations import validate_modules

        return validate_modules(cls, v)

    @model_validator(mode="after")
    def _validate_statistics_outputs(
        self,
    ) -> Self:
        from hydroforge.compiler.statistics_declaration import (
            StatisticsDeclarationCompiler,
        )

        return StatisticsDeclarationCompiler(self).compile()

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
                    f"parameter change {change.variable!r} start": (change.start),
                },
                calendar=schedule.calendar,
            )
            start = normalized[f"parameter change {change.variable!r} start"]
            # Only the date changes. The source declaration owns its tensor
            # snapshots, and normalization above validated the new calendar date.
            bound = ParameterChange.model_construct(
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
        from hydroforge.compiler.declarations import validate_input_contract

        return validate_input_contract(self)

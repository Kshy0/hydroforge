# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
import inspect
from abc import ABC
from datetime import datetime, timedelta
from functools import cache, cached_property
from pathlib import Path
from types import MappingProxyType
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional,
                    Mapping, Self, Union)

import cftime
import torch
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      field_validator, model_validator)

from hydroforge.statistics.ir import parse_operation
from hydroforge.compiler.namespace import NamespaceEntry
from hydroforge.data.distributed import ProcessTopology
from hydroforge.data.input import InputProxy
from hydroforge.contracts.kernel_field import KernelField
from hydroforge.contracts.fields import tensor_is_active
from hydroforge.contracts.temporal import (
    SimulationSchedule,
    StatisticsPlan,
)
from hydroforge.contracts.events import ConsoleEventSink, EventSink, emit
from hydroforge.contracts.runtime import (
    BackendRequirement,
    DEFAULT_MODULE_REQUIREMENT,
    ModuleRequirement,
)
from hydroforge.model.module import AbstractModule, ModuleReference
from hydroforge.execution.boundaries import between_steps
from hydroforge.serialization.netcdf import default_netcdf_options

if TYPE_CHECKING:
    from hydroforge.compiler.data import ModelDataCompiler
    from hydroforge.compiler.namespace import NamespaceCompiler
    from hydroforge.compiler.partition import (
        GroupRankLookup,
        PartitionCompiler,
    )
    from hydroforge.compiler.model import FieldOwner
    from hydroforge.compiler.statistics_binding import StatisticsBindingCompiler
    from hydroforge.execution.outer import OuterRuntime
    from hydroforge.execution.parameters import ParameterChangeEffect
    from hydroforge.execution.parameters import ParameterPlanRuntime
    from hydroforge.execution.progress import ProgressRuntime
    from hydroforge.execution.runtime import ModelExecution
    from hydroforge.execution.substeps import SubstepRuntime
    from hydroforge.output.checkpoint import CheckpointRuntime
    from hydroforge.contracts.fields import PartitionSchema


class AbstractModel(BaseModel, ABC):
    """
    Generic master controller for hydroforge models using the AbstractModule hierarchy.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='forbid',
        ignored_types=(KernelField, ModuleReference),
    )

    # Class variables
    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = {}
    module_requirements: ClassVar[Mapping[str, ModuleRequirement]] = {}
    partition_key: ClassVar[Optional[str]] = None
    partition_group: ClassVar[str] = "group_id"
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
    opened_modules: List[str] = Field(
        default_factory=list,
        description="List of active modules",
    )
    # Canonical shape: dict[op -> list[str | {alias: expression}]];
    # op in {mean,sum,max,min,first,last};
    # one variable can appear under multiple ops.  Use the reserved key
    # ``"static"`` to register per-saved-point static variables — these
    # are materialised once at aggregator init and written into every
    # output NC alongside the dynamic results.
    variables_to_save: Dict[str, List[Union[str, Dict[str, str]]]] = Field(
        default_factory=dict,
        description=(
            "Statistics to save, in the form {op: [vars...]}. "
            "Supported ops: mean, sum, max, min, first, last. "
            "Use explicit compound operations such as argmax_mean when an "
            "extremum time index is required. "
            "Variables can be strings or {alias: expr} dicts.  The "
            "Output slicing is inferred from each variable's dim_coords and "
            "the coordinate's default SelectionField. "
            "reserved key ``\"static\"`` marks per-saved-point static "
            "metadata (e.g. shift_days) written once per output NC."
        ),
    )
    precision: Literal["float32", "float64"] = Field(
        default="float32",
        description="Base precision of the model",
    )
    mixed_precision: Optional[bool] = Field(
        default=None,
        description=(
            "Enable mixed precision for hpfloat (storage) tensors.\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  float32 → float64, float64 → float64 (no promotion).\n"
            "If omitted, defaults to enabled for cuda/triton backends and "
            "disabled for metal/other backends."
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
        description="Device for tensors (e.g., 'cuda:0', 'cpu')",
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
        description="Number of workers for writing output files",
    )
    output_split_by_year: bool = Field(
        default=False,
        description="Whether to split output files by year",
    )
    num_trials: Optional[int] = Field(
        default=None,
        description="Number of parallel simulations (ensemble members)",
    )
    save_kernels: bool = Field(
        default=False,
        description="Whether to save generated Triton kernels",
    )
    max_pending_steps: int = Field(
        default=200,
        description="Maximum number of pending time steps for output buffering",
    )
    current_time: Optional[Union[datetime, cftime.datetime]] = Field(
        default=None,
        description=(
            "Runtime-owned time of the next managed model step. A simulation "
            "schedule initializes and advances it automatically."
        ),
    )
    simulation_schedule: Optional[SimulationSchedule] = Field(
        default=None,
        description="Runtime-owned model call schedule and calendar contract",
    )
    statistics_plan: Optional[StatisticsPlan] = Field(
        default=None,
        description="Calendar-aware or explicit statistics window plan",
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
        description="Store output in memory instead of writing to NC files",
    )
    result_device: Optional[torch.device] = Field(
        default=None,
        description="Device for in-memory results (default: CPU)",
    )
    output_netcdf_options: Dict[str, Any] = Field(
        default_factory=default_netcdf_options,
        description=(
            "Additional validated keyword options passed to netCDF4 "
            "Dataset.createVariable for dynamic output variables."
        ),
    )
    checkpoint_netcdf_options: Dict[str, Any] = Field(
        default_factory=default_netcdf_options,
        description=(
            "Validated netCDF4 Dataset.createVariable options for model "
            "checkpoint variables."
        ),
    )

    _modules: Dict[str, AbstractModule] = PrivateAttr(default_factory=dict)
    _model_modules_bound: bool = PrivateAttr(default=False)
    _process_topology: ProcessTopology = PrivateAttr(
        default_factory=lambda: ProcessTopology.capture(),
    )

    # Imports remain TYPE_CHECKING-only so the declarative layer does not gain
    # runtime dependencies on its compiler and execution consumers.
    _execution: ModelExecution = PrivateAttr()
    _namespace: NamespaceCompiler = PrivateAttr()
    _statistics: StatisticsBindingCompiler = PrivateAttr()
    _checkpoint: CheckpointRuntime = PrivateAttr()
    _data: ModelDataCompiler = PrivateAttr()
    _partition: PartitionCompiler = PrivateAttr()
    _field_namespace: Mapping[str, tuple[FieldOwner, ...]] = PrivateAttr()
    _parameters: ParameterPlanRuntime = PrivateAttr()
    _progress_service: ProgressRuntime = PrivateAttr()
    _substeps: SubstepRuntime = PrivateAttr()
    _outer: OuterRuntime = PrivateAttr()

    _sealed_configuration: Optional[Dict[str, object]] = PrivateAttr(
        default=None,
    )

    @property
    def substeps(self) -> SubstepRuntime:
        """Typed compiled substep authoring interface."""

        return self._substeps

    @property
    def rank(self) -> int:
        """Rank captured from the process group when this model was built."""

        return self._process_topology.rank

    @property
    def world_size(self) -> int:
        """World size captured from the process group when this model was built."""

        return self._process_topology.world_size

    @property
    def outer(self) -> OuterRuntime:
        """Typed once-per-outer-step operator authoring interface."""

        return self._outer

    @property
    def requested_sub_steps(self) -> int:
        """Return the active request, defaulting to one fixed sub-step."""

        step = self._execution.active_step
        if step is None:
            raise RuntimeError(
                "requested_sub_steps is available only inside @managed_step"
            )
        raw = getattr(step, "requested_sub_steps", None)
        return 1 if raw is None else raw

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"rank", "world_size"}:
            raise AttributeError(
                f"model {name} is read-only process-group topology"
            )
        model_module = self.get_module_reference_fields().get(name)
        if model_module is not None:
            model_module.__set__(self, value)

        private = getattr(self, "__pydantic_private__", None)
        sealed = (
            None if private is None else private.get("_sealed_configuration")
        )
        if (
            sealed is not None
            and name == "current_time"
            and getattr(self, "simulation_schedule", None) is not None
        ):
            current = getattr(self, "current_time", None)
            if value != current:
                raise RuntimeError(
                    "current_time is runtime-owned when simulation_schedule "
                    "is configured"
                )
            return
        if sealed is not None and name in sealed and value is not sealed[name]:
            raise RuntimeError(
                f"model configuration field {name!r} is sealed after "
                "initialization"
            )
        super().__setattr__(name, value)

    def _set_runtime_current_time(
        self, value: Union[datetime, cftime.datetime],
    ) -> None:
        """Advance the sealed schedule clock from the managed-step runtime."""

        super().__setattr__("current_time", value)

    def __delattr__(self, name: str) -> None:
        if name in {"rank", "world_size"}:
            raise AttributeError(
                f"model {name} is read-only process-group topology"
            )
        private = getattr(self, "__pydantic_private__", None)
        sealed = (
            None if private is None else private.get("_sealed_configuration")
        )
        if sealed is not None and name in sealed:
            raise RuntimeError(
                f"model configuration field {name!r} is sealed and cannot "
                "be deleted"
            )
        super().__delattr__(name)

    def _seal_model_configuration(self) -> None:
        if self._sealed_configuration is not None:
            raise RuntimeError("model configuration is already sealed")
        # Plans, namespaces and statistics have already been compiled at this
        # point.  Freeze their collection-valued source declarations rather
        # than scanning them on every outer step for in-place mutations.
        self.opened_modules = tuple(self.opened_modules)
        self.variables_to_save = MappingProxyType({
            key: self._freeze_configuration_value(value)
            for key, value in self.variables_to_save.items()
        })
        self._sealed_configuration = {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name != "current_time"
        }

    @classmethod
    def _freeze_configuration_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return MappingProxyType({
                key: cls._freeze_configuration_value(item)
                for key, item in value.items()
            })
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_configuration_value(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return frozenset(
                cls._freeze_configuration_value(item) for item in value
            )
        return value

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        post_init = cls.__dict__.get("model_post_init")
        if (
            post_init is not None
            and inspect.unwrap(post_init)
            is not inspect.unwrap(AbstractModel.model_post_init)
        ):
            raise TypeError(
                f"{cls.__name__} may not override model_post_init(); put "
                "post-module initialization in initialize_model_state() so "
                "HydroForge can roll it back transactionally"
            )
        module_types = cls.module_types()
        unknown_backends = set(cls.backend_requirements).difference({
            "torch", "cuda", "triton", "metal",
        })
        if unknown_backends:
            raise ValueError(
                f"{cls.__name__}.backend_requirements has unknown backends: "
                f"{sorted(unknown_backends)}"
            )
        invalid_backends = {
            name: type(rule).__name__
            for name, rule in cls.backend_requirements.items()
            if not isinstance(rule, BackendRequirement)
        }
        if invalid_backends:
            raise TypeError(
                f"{cls.__name__}.backend_requirements must contain "
                f"BackendRequirement values: {invalid_backends}"
            )
        unknown_modules = set(cls.module_requirements).difference(module_types)
        if unknown_modules:
            raise ValueError(
                f"{cls.__name__}.module_requirements names unknown modules: "
                f"{sorted(unknown_modules)}"
            )
        invalid_modules = {
            name: type(rule).__name__
            for name, rule in cls.module_requirements.items()
            if not isinstance(rule, ModuleRequirement)
        }
        if invalid_modules:
            raise TypeError(
                f"{cls.__name__}.module_requirements must contain "
                f"ModuleRequirement values: {invalid_modules}"
            )
        cls.backend_requirements = MappingProxyType(
            dict(cls.backend_requirements),
        )
        cls.module_requirements = MappingProxyType(
            dict(cls.module_requirements),
        )

    @classmethod
    def get_module_reference_fields(cls) -> Dict[str, ModuleReference]:
        """Return the model's typed module declarations."""

        return ModuleReference.collect(cls)

    @classmethod
    @cache
    def module_types(cls) -> Mapping[str, type[AbstractModule]]:
        """Return the immutable module catalog derived from declarations."""

        return MappingProxyType({
            name: reference.module_type
            for name, reference in cls.get_module_reference_fields().items()
        })

    @classmethod
    @cache
    def compiled_schema(cls):
        """Return the immutable schema for every registered module class."""
        from hydroforge.contracts.fields import parse_module_schema

        return parse_module_schema(
            tuple(cls.module_types().values()), include_computed=True,
        )

    @field_validator(
        "output_netcdf_options", "checkpoint_netcdf_options", mode="before",
    )
    @classmethod
    def validate_output_netcdf_options(cls, value):
        from hydroforge.serialization.netcdf import (
            normalize_netcdf_variable_options,
        )

        return normalize_netcdf_variable_options(value)

    @field_validator("variables_to_save", mode="before")
    @classmethod
    def validate_variables_to_save_shape(cls, value):
        """Reject legacy output spellings before Pydantic can coerce them."""

        if type(value) is not dict:
            raise ValueError("variables_to_save must be an exact dict")
        normalized = {}
        for operation, items in value.items():
            if type(operation) is not str or not operation:
                raise ValueError(
                    "variables_to_save operation names must be non-empty strings"
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
            for item in items:
                if type(item) is str:
                    if not item:
                        raise ValueError("output field names must be non-empty")
                    continue
                if type(item) is not dict or len(item) != 1:
                    raise ValueError(
                        "output items must be field names or one-item "
                        "{alias: expression} dicts"
                    )
                alias, expression = next(iter(item.items()))
                if (
                    type(alias) is not str or not alias
                    or type(expression) is not str or not expression
                ):
                    raise ValueError(
                        "explicit output aliases and expressions must be "
                        "non-empty strings"
                    )
            normalized[canonical] = items
        return normalized

    @model_validator(mode="after")
    def validate_module_requirements(self) -> Self:
        for name in self.opened_modules:
            rule = self.module_requirements.get(
                name, DEFAULT_MODULE_REQUIREMENT,
            )
            if not rule.trials and self.num_trials is not None:
                raise ValueError(
                    f"module {name!r} does not support ensemble trials"
                )
        return self


    @field_validator('num_trials')
    @classmethod
    def validate_num_trials(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 1:
            raise ValueError("num_trials must be greater than 1 if specified. For single trial, use None.")
        return v

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

    def check_namespace_conflicts(self) -> None:
        """
        Check for namespace conflicts across all opened modules.

        Virtual fields with an ``expr`` (scatter / plain aggregation outputs)
        are allowed to share a name with their source counterpart in another
        module — this is the standard subcell→cell aggregation pattern.
        """
        field_definitions = {}
        schema = self.compiled_schema()
        module_types = self.module_types()
        for module_name in self.opened_modules:
            excluded = set(module_types[module_name].nc_excluded_fields)
            for field in schema.fields(module_name):
                if field.tensor is not None:
                    unknown_dependencies = sorted(
                        set(
                            (*field.tensor.depends_on,
                             *field.tensor.required_by),
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
                if (
                    field.tensor is not None
                    and not tensor_is_active(
                        field.tensor, self.opened_modules,
                    )
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
                    field.annotation != previous.annotation
                    or field.tensor != previous.tensor
                ):
                    raise ValueError(
                        f"Namespace conflict for {field.name!r}: "
                        f"{previous.module_name} and {module_name} declare "
                        "different types or tensor metadata"
                    )

    def model_post_init(self, __context: Any) -> None:
        from hydroforge.compiler.initialization import ModelInitializer

        ModelInitializer(self).run()

    def initialize_model_state(self) -> None:
        """Initialize ordered model state inside HydroForge's transaction.

        HydroForge does not invoke module initialization hooks automatically;
        model authors explicitly call any module helpers here in physical order.
        """

    def rebuild_runtime_state(self) -> None:
        """Rebuild non-checkpoint runtime state after checkpoint restoration.

        Implementations must preserve tensor identities used by compiled
        kernels and may only derive transient state from already restored
        model fields. HydroForge invokes this hook inside the checkpoint load
        transaction and invokes it again after rollback if the commit fails.
        """

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
                if isinstance(value, torch.Tensor) and value.device.type == module.device.type:
                    ptr = value.data_ptr()
                    if ptr not in global_seen_ptrs:
                        global_seen_ptrs.add(ptr)
                        module_bytes += value.element_size() * value.nelement()

            total_memory += module_bytes
            module_memory[module_name] = module_bytes / (1024 * 1024)

        # Add StatisticsRuntime memory usage
        aggregator = self._statistics.aggregator
        if aggregator is not None:
            aggregator_mem = aggregator.get_memory_usage()
            total_memory += aggregator_mem
            module_memory["StatisticsAggregator"] = aggregator_mem / (1024 * 1024)

        emit(
            self, "info", "model.memory", "Model memory summary",
            rank=self.rank, modules=module_memory,
            total_mb=total_memory / (1024 * 1024),
        )

    @property
    def partition_metadata(self) -> PartitionSchema:
        return self._partition.schema

    @property
    def variable_group_mapping(self) -> Mapping[str, str]:
        return self._partition.variable_groups

    @cached_property
    def variable_map(self) -> Mapping[str, NamespaceEntry]:
        """
        Map variable names to immutable owner and coordinate metadata.
        This provides a unified way to lookup variables across all modules.

        When a field name exists in multiple modules, the virtual field
        with an ``expr`` (scatter / plain aggregation output) takes
        priority for the unqualified name.  Both qualified forms
        (``module.field``) are always available.
        """
        return self._namespace.build()

    @cached_property
    def group_id_to_rank(self) -> GroupRankLookup:
        return self._partition.group_ranks

    def close(self) -> None:
        """Atomically release output workers and backend execution resources."""

        if self._execution.active_step is not None:
            raise RuntimeError(
                "model.close() is forbidden during an active managed step"
            )

        failures: list[BaseException] = []
        try:
            self._statistics.close()
        except BaseException as error:
            failures.append(error)
        try:
            self._execution.close()
        except BaseException as error:
            failures.append(error)
        if failures:
            from hydroforge.contracts import ResourceCleanupError

            error = ResourceCleanupError("model resources", failures)
            raise error from failures[0]

    def execute_parameter_change_plan(
        self, current_time: Union[datetime, cftime.datetime],
    ) -> ParameterChangeEffect:
        return self._parameters.execute_parameter_change_plan(current_time)

    @between_steps
    def add_parameter_change_plan(
        self,
        variable_name: str,
        start_time: Union[datetime, cftime.datetime],
        active_steps: int = 1,
        delta: Union[float, torch.Tensor] = 0.0,
        target_value: Optional[Union[float, torch.Tensor]] = None,
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
        target_id_field: Optional[str] = None,
    ) -> None:
        self._parameters.add_parameter_change_plan(
            variable_name=variable_name,
            start_time=start_time,
            active_steps=active_steps,
            delta=delta,
            target_value=target_value,
            target_ids=target_ids,
            target_id_field=target_id_field,
        )

    def get_variable(self, variable_name: str) -> torch.Tensor:
        value = self._parameters.get_variable(variable_name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"declared parameter {variable_name!r} is not a torch.Tensor"
            )
        return value.detach().clone(memory_format=torch.preserve_format)

    @between_steps
    def set_variable_value(
        self,
        variable_name: str,
        value: Union[float, torch.Tensor],
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        try:
            self._parameters.set_variable_value(
                variable_name, value, target_ids,
            )
        except BaseException as error:
            self._execution.poison(error, phase="direct parameter update")
            raise

    def summarize_plan(self) -> None:
        self._parameters.summarize_plan()

    @property
    def step_output_enabled(self) -> bool:
        """Return the effective output state of the active managed step."""

        context = self._execution.active_step
        if context is None:
            raise RuntimeError(
                "step_output_enabled is available only inside @managed_step"
            )
        return context.output_enabled

    @property
    def step_duration(self) -> timedelta:
        """Return the exact duration owned by the active managed step."""

        context = self._execution.active_step
        if context is None:
            raise RuntimeError(
                "step_duration is available only inside @managed_step"
            )
        return context.duration

    def progress_start(self) -> None:
        self._progress_service.begin_step()

    def progress_tick(self) -> None:
        self._progress_service.progress_tick()

    def format_progress(self) -> str:
        return self._progress_service.format_progress()

    @between_steps
    def get_output_results(
        self, as_stacked: bool = True,
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
        return self._statistics.results(stacked=as_stacked)

    @between_steps
    def get_output_result(
        self, variable_name: str, op: str = "mean",
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
        return self._statistics.result(variable_name, op, stacked=as_stacked)

    @between_steps
    def get_output_time_index(self) -> int:
        """Get the current output time index (number of finalized time steps)."""
        return self._statistics.time_index()

    @between_steps
    def get_output_accumulator(
        self, variable_name: str, operation: str = "mean",
    ) -> torch.Tensor:
        """Return a differentiable snapshot without exposing captured storage."""

        return self._statistics.accumulator(variable_name, operation)

    @between_steps
    def pop_output_result(
        self, variable_name: str, operation: str = "mean",
    ) -> torch.Tensor:
        """Pop the newest in-memory result without retaining its history."""

        return self._statistics.pop_result(variable_name, operation)

    @between_steps
    def reset_output_time_index(self) -> None:
        """Reset the output time index to 0 for a new simulation run (in-memory mode only)."""
        self._statistics.reset_time_index()

    def shard_param(self) -> Dict[str, Any]:
        """Load and rank-slice parameters through the internal data service."""
        return self._data.shard()

    @between_steps
    def save_state(self) -> InputProxy:
        """Persist physical state at the committed runtime clock."""
        return self._checkpoint.save()

    @between_steps
    def load_state(self, proxy: InputProxy) -> None:
        """Restore physical model state without changing runtime cursors."""
        self._checkpoint.load(proxy)

    @field_validator("opened_modules")
    @classmethod
    def validate_modules(cls, v: List[str]) -> List[str]:
        """Validate module names are valid"""
        if not v:
            raise ValueError("No modules opened. Please specify at least one module in opened_modules.")
        module_types = cls.module_types()
        for module in v:
            if module not in module_types:
                raise ValueError(f"Invalid module name: {module}. Available modules: {list(module_types)}")
        missing_model_modules = [
            name
            for name, reference in cls.get_module_reference_fields().items()
            if not reference.optional and name not in v
        ]
        if missing_model_modules:
            raise ValueError(
                "Missing required model modules in opened_modules: "
                f"{missing_model_modules}. Available modules: {v}"
            )
        for module in v:
            module_class = module_types[module]
            references = module_class.get_module_reference_fields().values()
            unknown_references = sorted({
                reference.module_name
                for reference in references
                if reference.module_name not in module_types
            })
            if unknown_references:
                raise ValueError(
                    f"Module '{module}' declares references to unknown modules: "
                    f"{unknown_references}. Available modules: "
                    f"{list(module_types)}"
                )
            required = module_class.required_modules()
            missing_deps = [dep for dep in required if dep not in v]
            if missing_deps:
                raise ValueError(
                    f"Module '{module}' has missing required modules in "
                    f"opened_modules: {missing_deps}. "
                    f"Required modules: {required}. "
                    f"Available modules: {v}"
                )
            present_conflicts = [
                conflict for conflict in module_class.conflicts
                if conflict in v and conflict != module
            ]
            if present_conflicts:
                raise ValueError(
                    f"Module '{module}' conflicts with modules present in opened_modules: "
                    f"{present_conflicts}. These modules cannot be enabled together."
                )
        return v

    @model_validator(mode="after")
    def validate_variables_to_save(self) -> Self:
        if not self.variables_to_save:
            return self
        pairs = []
        for op, variables in self.variables_to_save.items():
            operation = op
            # Static entries bypass operation-grammar and dynamic-field checks;
            # the runtime registers them once through register_static.
            if operation == "static":
                continue
            parse_operation(operation)
            for variable in variables:
                if isinstance(variable, dict):
                    alias, expression = next(iter(variable.items()))
                    pairs.append((alias, operation, True))
                else:
                    pairs.append((variable, operation, False))

        # Validate each variable exists. Output views are resolved later from
        # dim_coords and the coordinate's SelectionField.
        for var, _, is_explicit in pairs:
            if is_explicit:
                continue

            found = False
            for module in self.opened_modules:
                module_class = self.module_types()[module]
                fields = module_class.model_fields | module_class.model_computed_fields
                if var in fields:
                    found = True
                    break

            # If not found as direct field, check if it is a valid expression
            if not found:
                 # If the variable name contains characters other than alphanumeric and underscore,
                 # we assume it is a mathematical expression.
                 if re.search(r'[^a-zA-Z0-9_]', var):
                     found = True

            if not found:
                raise ValueError(f"Variable '{var}' not found in any opened module.")
        return self

    @model_validator(mode="after")
    def validate_output_full_dir(self) -> Self:
        if self.rank == 0:
            if not self.output_full_dir.exists():
                self.output_full_dir.mkdir(parents=True, exist_ok=True)
            else:
                emit(
                    self, "warning", "output.directory_exists",
                    "Output directory already exists; contents may be overwritten",
                    directory=self.output_full_dir,
                )
        return self

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
                    Mapping, Self, Tuple, Type, Union)

import cftime
import torch
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      field_validator, model_validator)

from hydroforge.statistics.ir import parse_operation
from hydroforge.data.input import InputProxy
from hydroforge.contracts.kernel_field import KernelField
from hydroforge.contracts.temporal import (
    SimulationSchedule,
    StatisticsPlan,
    timedelta_microseconds,
    timedelta_quotient,
)
from hydroforge.contracts.events import ConsoleEventSink, EventSink, emit
from hydroforge.contracts.runtime import (
    BackendRequirement,
    DEFAULT_MODULE_REQUIREMENT,
    ModuleRequirement,
)
from hydroforge.model.module import AbstractModule
from hydroforge.execution.boundaries import between_steps
from hydroforge.serialization.netcdf import default_netcdf_options

if TYPE_CHECKING:
    from hydroforge.execution.parameters import ParameterChangeEffect


class AbstractModel(BaseModel, ABC):
    """
    Generic master controller for hydroforge models using the AbstractModule hierarchy.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='forbid',
        ignored_types=(KernelField,),
    )

    # Class variables
    module_list: ClassVar[Dict[str, Type[AbstractModule]]] = {}
    feature_rules: ClassVar[Dict[str, Any]] = {}
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
    # Preferred shape: dict[op -> str | list[str]]; op in {mean,max,min,last};
    # one variable can appear under multiple ops.  Use the reserved key
    # ``"static"`` to register per-saved-point static variables — these
    # are materialised once at aggregator init and written into every
    # output NC alongside the dynamic results.
    variables_to_save: Optional[Dict[str, Union[str, List[Union[str, Dict[str, str]]]]]] = Field(
        default=None,
        description=(
            "Statistics to save, in the form {op: [vars...]}. "
            "Supported ops: mean, max, min, last, first, mid, sum. "
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
    mixed_precision: bool = Field(
        default=False,
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
    world_size: int = Field(
        default=1,
        description="Total number of distributed processes",
    )
    rank: int = Field(
        default=0,
        description="Current process rank in distributed setup",
    )
    device: torch.device = Field(
        default=torch.device("cpu"),
        description="Device for tensors (e.g., 'cuda:0', 'cpu')",
    )
    event_sink: EventSink = Field(
        default_factory=ConsoleEventSink,
        description="Structured lifecycle/progress event destination",
    )
    BLOCK_SIZE: int = Field(
        default=256,
        description="GPU block size for kernels",
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
    output_start_time: Optional[Union[datetime, cftime.datetime]] = Field(
        default=None,
        description="Time to start saving output",
    )
    statistics_interval: Optional[timedelta] = Field(
        default=None,
        description=(
            "Statistics window aligned to calendar time. None finalizes one "
            "output per step_advance call."
        ),
    )
    statistics_outer_interval: Optional[timedelta] = Field(
        default=None,
        description=(
            "Outer window for compound statistics. None uses one inner "
            "statistics window per outer window."
        ),
    )
    simulation_schedule: Optional[SimulationSchedule] = Field(
        default=None,
        description="Driver-owned model call schedule and calendar contract",
    )
    statistics_plan: Optional[StatisticsPlan] = Field(
        default=None,
        description="Calendar-aware or explicit statistics window plan",
    )
    calendar: str = Field(
        default="standard",
        description="Calendar type for time handling (e.g., standard, noleap)",
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

    _capabilities: frozenset[str] = PrivateAttr(default_factory=frozenset)
    # Concrete compiler/runtime objects are intentionally not imported here:
    # the declarative model layer must not depend on its consumers.
    _execution: Any = PrivateAttr()
    _namespace: Any = PrivateAttr()
    _statistics: Any = PrivateAttr()
    _checkpoint: Any = PrivateAttr()
    _data: Any = PrivateAttr()
    _partition: Any = PrivateAttr()
    _plan: Any = PrivateAttr()
    _parameters: Any = PrivateAttr()
    _progress_service: Any = PrivateAttr()

    # Progress Tracking
    _progress: Optional[Any] = PrivateAttr(default=None)
    _sealed_configuration: Optional[Dict[str, object]] = PrivateAttr(
        default=None,
    )

    def __setattr__(self, name: str, value: Any) -> None:
        private = getattr(self, "__pydantic_private__", None)
        sealed = (
            None if private is None else private.get("_sealed_configuration")
        )
        if sealed is not None and name in sealed and value is not sealed[name]:
            raise RuntimeError(
                f"model configuration field {name!r} is sealed after "
                "initialization"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
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
        if self.variables_to_save is not None:
            self.variables_to_save = MappingProxyType({
                key: self._freeze_configuration_value(value)
                for key, value in self.variables_to_save.items()
            })
        self._sealed_configuration = {
            name: getattr(self, name)
            for name in type(self).model_fields
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
        unknown_modules = set(cls.module_requirements).difference(cls.module_list)
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
        cls.module_list = MappingProxyType(dict(cls.module_list))
        cls.feature_rules = MappingProxyType(dict(cls.feature_rules))

    @classmethod
    @cache
    def compiled_schema(cls):
        """Return the immutable schema for every registered module class."""
        from hydroforge.contracts.fields import parse_module_schema

        return parse_module_schema(
            tuple(cls.module_list.values()), include_computed=True,
        )

    @field_validator("statistics_interval", "statistics_outer_interval")
    @classmethod
    def validate_statistics_interval(
        cls, value: Optional[timedelta],
    ) -> Optional[timedelta]:
        if value is not None and timedelta_microseconds(
            value, label="statistics interval",
        ) <= 0:
            raise ValueError("statistics intervals must be positive")
        return value

    @field_validator(
        "output_netcdf_options", "checkpoint_netcdf_options", mode="before",
    )
    @classmethod
    def validate_output_netcdf_options(cls, value):
        from hydroforge.serialization.netcdf import (
            normalize_netcdf_variable_options,
        )

        return normalize_netcdf_variable_options(value)

    @model_validator(mode="after")
    def validate_statistics_window_nesting(self) -> Self:
        inner = self.statistics_interval
        outer = self.statistics_outer_interval
        if inner is None:
            if outer is not None:
                raise ValueError(
                    "statistics_outer_interval requires statistics_interval"
                )
            return self
        if outer is None:
            return self
        try:
            ratio = timedelta_quotient(
                outer,
                inner,
                duration_label="statistics_outer_interval",
                interval_label="statistics_interval",
            )
        except ValueError as exc:
            raise ValueError(
                "statistics_outer_interval must be an integer multiple of "
                "statistics_interval"
            ) from exc
        if ratio < 1:
            raise ValueError(
                "statistics_outer_interval must not be shorter than "
                "statistics_interval"
            )
        return self

    @model_validator(mode='after')
    def align_output_start_time(self) -> Self:
        """
        Ensures output_start_time matches the specified calendar type.
        """
        if self.output_start_time is None or self.calendar == "standard":
            return self

        # If output_start_time is standard datetime but calendar is not standard (e.g. noleap)
        # we try to convert it to the appropriate cftime object.
        if isinstance(self.output_start_time, datetime) and not isinstance(self.output_start_time, cftime.datetime):
            try:
                # Create a dummy object to get the class type for this calendar
                dummy = cftime.num2date([0], units="days since 1900-01-01", calendar=self.calendar)[0]
                Cls = dummy.__class__

                kwargs = {}
                if hasattr(dummy, "has_year_zero"):
                    kwargs["has_year_zero"] = dummy.has_year_zero

                self.output_start_time = Cls(
                    self.output_start_time.year, self.output_start_time.month, self.output_start_time.day,
                    self.output_start_time.hour, self.output_start_time.minute, self.output_start_time.second,
                    self.output_start_time.microsecond, **kwargs
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"output_start_time {self.output_start_time!r} cannot be "
                    f"represented by calendar '{self.calendar}'"
                ) from exc
        return self

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
        for module_name in self.opened_modules:
            excluded = set(self.module_list[module_name].nc_excluded_fields)
            for field in schema.fields(module_name):
                if field.excluded or field.name in excluded:
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
        """Initialize cross-module state inside HydroForge's transaction."""

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

    def get_module(self, module_name: str) -> Optional[AbstractModule]:
        return self._modules.get(module_name) if module_name in self._capabilities else None

    def has_module(self, module_name: str) -> bool:
        """Return whether a registered module is open in this specialization."""
        return module_name in self._capabilities

    def has_feature(self, name: str) -> bool:
        """Evaluate a model capability once while a launch plan is built.

        Module names are capabilities automatically. Composite capabilities
        are declared in ``feature_rules`` as callables receiving the model.
        """
        if name not in self.module_list and name not in self.feature_rules:
            raise KeyError(f"unknown model feature {name!r}")
        return name in self._capabilities

    @property
    def partition_metadata(self):
        return self._partition.schema

    @property
    def variable_group_mapping(self) -> Dict[str, str]:
        return self._partition.variable_groups

    @cached_property
    def variable_map(self) -> Dict[str, Tuple[AbstractModule, str, Optional[str]]]:
        """
        Map variable names to (module_instance, field_name, id_attr).
        This provides a unified way to lookup variables across all modules.

        When a field name exists in multiple modules, the virtual field
        with an ``expr`` (scatter / plain aggregation output) takes
        priority for the unqualified name.  Both qualified forms
        (``module.field``) are always available.
        """
        return self._namespace.build()

    @cached_property
    def group_id_to_rank(self) -> Any:
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

    def get_variable(self, variable_name: str) -> Any:
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

    @between_steps
    def set_total_steps(self, total: int) -> None:
        self._progress_service.set_total_steps(total)

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
    def save_state(
        self, current_time: Optional[Union[datetime, cftime.datetime]],
    ) -> InputProxy:
        """Persist model state through the internal checkpoint service."""
        return self._checkpoint.save(current_time)

    @between_steps
    def load_state(self, proxy: InputProxy) -> None:
        """Restore model state through the internal checkpoint service."""
        self._checkpoint.load(proxy)

    @field_validator("opened_modules")
    @classmethod
    def validate_modules(cls, v: List[str]) -> List[str]:
        """Validate module names are valid"""
        if not v:
            raise ValueError("No modules opened. Please specify at least one module in opened_modules.")
        for module in v:
            if module not in cls.module_list:
                raise ValueError(f"Invalid module name: {module}. Available modules: {list(cls.module_list.keys())}")
        for module in v:
            module_class = cls.module_list[module]
            missing_deps = [dep for dep in module_class.dependencies if dep not in v]
            if missing_deps:
                raise ValueError(
                    f"Module '{module}' has missing dependencies in opened_modules: {missing_deps}. "
                    f"Required dependencies: {module_class.dependencies}. "
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
        if self.variables_to_save is None:
            return self
        # Validate shape: dict[op -> vars]
        if not isinstance(self.variables_to_save, dict):
            # Optional convenience: list[str] => mean
            names = list(self.variables_to_save) if isinstance(self.variables_to_save, list) else []
            pairs = [(n, "mean") for n in names]
        else:
            pairs = []
            for op, vs in self.variables_to_save.items():
                # Static (op=="static") entries bypass op-grammar checks;
                # the runtime registers them via register_static.
                if op == "static":
                    continue
                op_l = str(op).lower()
                parse_operation(op_l)

                if isinstance(vs, str):
                    vars_list = [vs]
                elif isinstance(vs, list):
                    vars_list = vs
                else:
                    raise ValueError(f"variables_to_save['{op}'] must be a string or list of strings/dicts")
                for var in vars_list:
                    if isinstance(var, dict):
                        var_name = next(iter(var.keys()))
                        # Explicit definition {alias: expr} -> Treat as valid virtual
                        pairs.append((var_name, op_l, True))
                    elif isinstance(var, (tuple, list)):
                        var_name = var[0]
                         # Explicit definition (alias, expr) -> Treat as valid virtual
                        pairs.append((var_name, op_l, True))
                    else:
                        pairs.append((var, op_l, False))

        # Validate each variable exists. Output views are resolved later from
        # dim_coords and the coordinate's SelectionField.
        for var, _, is_explicit in pairs:
            if is_explicit:
                continue

            found = False
            for module in self.opened_modules:
                module_class = self.module_list[module]
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
    def validate_rank(self) -> Self:
        """
        Validate that the current rank is within the world size.
        """
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid rank {self.rank} for world size {self.world_size}.")
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

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
from copy import copy
from abc import ABC
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import (Any, ClassVar, Dict, Iterator, List, Literal, Optional,
                    Self, Tuple, Type, Union)

import cftime
import numpy as np
import torch
import torch.distributed as dist
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      field_validator, model_validator)

from hydroforge.aggregator.aggregator import StatisticsAggregator
from hydroforge.modeling.distributed import find_indices_in, find_indices_in_torch
from hydroforge.modeling.input_proxy import InputProxy
from hydroforge.modeling.model_utils import (ActivePlan, ParameterPlanMixin,
                                             GroupRankLookup, PlanItem, ProgressMixin,
                                             ProgressTracker,
                                             compute_group_to_rank)
from hydroforge.modeling.module import AbstractModule
from hydroforge.runtime.backend import KERNEL_BACKEND


class AbstractModel(ParameterPlanMixin, ProgressMixin, BaseModel, ABC):
    """
    Generic master controller for hydroforge models using the AbstractModule hierarchy.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='forbid'
    )

    # Class variables
    module_list: ClassVar[Dict[str, Type[AbstractModule]]] = {}
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
            "For max/min, argmax/argmin are automatically computed. "
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
    BLOCK_SIZE: int = Field(
        default=256,
        description="GPU block size for kernels",
    )
    output_workers: int = Field(
        default=2,
        description="Number of workers for writing output files",
    )
    output_compression: str = Field(
        default="zlib",
        description="Compression algorithm for output NetCDF files",
    )
    output_complevel: int = Field(
        default=4,
        description="Compression level for output NetCDF files",
        ge=0,
        le=9,
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
    output_chunksizes: Optional[tuple] = Field(
        default=None,
        description="NetCDF chunk sizes for output data variables, e.g. (365, 1). "
                    "If None, uses netCDF4 default.",
    )

    _modules: Dict[str, AbstractModule] = PrivateAttr(default_factory=dict)

    _statistics_aggregator: Optional[StatisticsAggregator] = PrivateAttr(default=None)

    # Parameter Change Plan State
    _plans: List[PlanItem] = PrivateAttr(default_factory=list)
    _active_plans: List[ActivePlan] = PrivateAttr(default_factory=list)
    _next_plan_idx: int = PrivateAttr(default=0)
    _cached_grouped_plans: Optional[Dict[Tuple[int, str], List[ActivePlan]]] = PrivateAttr(default=None)

    # Progress Tracking
    _progress: Optional[ProgressTracker] = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def set_backend_mixed_precision_default(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "mixed_precision" in data:
            return data

        data = dict(data)
        device = torch.device(data.get("device", "cpu"))
        data["mixed_precision"] = (
            device.type == "cuda"
            and KERNEL_BACKEND in ("cuda", "triton")
        )
        return data

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


    @field_validator('num_trials')
    @classmethod
    def validate_num_trials(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 1:
            raise ValueError("num_trials must be greater than 1 if specified. For single trial, use None.")
        return v

    def _iter_all_fields(self, include_computed: bool = True) -> Iterator[Tuple[str, Type[AbstractModule], str, Any]]:
        """
        Iterate over all fields in all opened modules.
        Yields: (module_name, module_class, field_name, field_info)
        """
        for module_name in self.opened_modules:
            if module_name not in self.module_list:
                continue
            module_class = self.module_list[module_name]

            # Regular fields
            for name, info in module_class.get_model_fields().items():
                if name not in module_class.nc_excluded_fields:
                    yield module_name, module_class, name, info

            # Computed fields
            if include_computed:
                for name, info in module_class.get_model_computed_fields().items():
                    if name not in module_class.nc_excluded_fields:
                        yield module_name, module_class, name, info

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
        field_definitions: Dict[str, Tuple[str, Any]] = {}

        for module_name, _, field_name, field_info in self._iter_all_fields(include_computed=True):
            if field_name in field_definitions:
                existing_module, existing_info = field_definitions[field_name]

                new_extra = getattr(field_info, 'json_schema_extra', {}) or {}
                old_extra = getattr(existing_info, 'json_schema_extra', {}) or {}

                # Allow coexistence when one side is a virtual-with-expr
                # (scatter/plain aggregation output) — the pair is intentional.
                new_is_expr_virtual = (new_extra.get('category') == 'virtual'
                                       and new_extra.get('expr'))
                old_is_expr_virtual = (old_extra.get('category') == 'virtual'
                                       and old_extra.get('expr'))
                if new_is_expr_virtual or old_is_expr_virtual:
                    # Keep the expr-virtual as the primary definition
                    if new_is_expr_virtual and not old_is_expr_virtual:
                        field_definitions[field_name] = (module_name, field_info)
                    continue

                # Compare definitions
                new_type = getattr(field_info, 'annotation', getattr(field_info, 'return_type', None))
                old_type = getattr(existing_info, 'annotation', getattr(existing_info, 'return_type', None))

                if new_type != old_type or new_extra != old_extra:
                    raise ValueError(
                        f"Namespace conflict detected for field '{field_name}':\n"
                        f"  - Defined in '{existing_module}' with type={old_type}, extra={old_extra}\n"
                        f"  - Defined in '{module_name}' with type={new_type}, extra={new_extra}\n"
                        f"Please rename one of the fields to avoid ambiguity."
                    )
            else:
                field_definitions[field_name] = (module_name, field_info)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to validate opened modules and register them.
        """
        print(f"[rank {self.rank}]: Initializing ModelManager with opened modules:", self.opened_modules)

        self.check_namespace_conflicts()

        print(
            f"Using partition root: key={self.partition_key}, "
            f"group={self.partition_group}"
        )

        # Validate that all opened modules are registered
        module_data = self.shard_param()  # reads from NetCDF

        # Sort modules by dependency
        from graphlib import TopologicalSorter
        sorter = TopologicalSorter()
        for module_name in self.opened_modules:
            if module_name not in self.module_list:
                raise ValueError(f"Module {module_name} not found in module_list")
            deps = self.module_list[module_name].dependencies
            sorter.add(module_name, *deps)

        # Get sorted order
        sorted_modules = list(sorter.static_order())

        for module_name in sorted_modules:
            if module_name not in self.opened_modules:
                continue

            # Register the module instance with data
            module_class = self.module_list[module_name]
            module_instance = module_class(
                opened_modules=self.opened_modules,
                rank=self.rank,
                device=self.device,
                world_size=self.world_size,
                precision=self.dtype,
                mixed_precision=self.mixed_precision,
                num_trials=self.num_trials,
                **self._modules,
                **module_data,
            )
            self._modules[module_name] = module_instance

        self.initialize_statistics_aggregator()

        for module_name in self.opened_modules:
            mod = self.get_module(module_name)
            if mod:
                mod.handle_tensor_mode()

        self.print_memory_summary()
        print("All modules initialized successfully.")

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
        print(f"\n[rank {self.rank}] Memory Usage Summary:")
        print(f"{'Module':<30} | {'Memory (MB)':<15}")
        print(f"{'-' * 48}")

        for module_name in self.opened_modules:
            if module_name not in self._modules:
                continue
            module = self._modules[module_name]

            # Count only tensors not yet seen globally
            module_bytes = 0
            all_fields = module.get_model_fields().copy()
            all_fields.update(module.get_model_computed_fields())
            for name in all_fields:
                # Skip computed fields that haven't been materialized yet
                # to avoid triggering @cached_property (lazy allocation).
                if name in module.get_model_computed_fields() and name not in module.__dict__:
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
            print(f"{module_name:<30} | {module_bytes / (1024 * 1024):<15.2f}")

        # Add StatisticsAggregator memory usage
        if self._statistics_aggregator is not None:
            aggregator_mem = self._statistics_aggregator.get_memory_usage()
            total_memory += aggregator_mem
            print(f"{'StatisticsAggregator':<30} | {aggregator_mem / (1024 * 1024):<15.2f}")

        print(f"{'-' * 48}")
        print(f"{'Total':<30} | {total_memory / (1024 * 1024):<15.2f} MB\n")

    def get_module(self, module_name: str) -> Optional[AbstractModule]:
        return self._modules[module_name] if module_name in self.opened_modules else None

    @cached_property
    def partition_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Collect and validate the coordinate/foreign-key partition graph."""
        fields: Dict[str, Dict[str, Any]] = {}
        for _, _, name, info in self._iter_all_fields(include_computed=False):
            fields[name] = (getattr(info, "json_schema_extra", None) or {})

        coordinates = {
            name for name, extra in fields.items()
            if extra.get("is_coordinate", False)
        }
        selections: Dict[str, str] = {}
        if self.partition_key is None:
            raise ValueError("Model partition_key must be configured.")
        if self.partition_key not in coordinates:
            raise ValueError(
                f"partition_key '{self.partition_key}' must be a CoordinateField."
            )
        if self.partition_group not in fields:
            raise ValueError(
                f"partition_group '{self.partition_group}' is not declared."
            )

        for name, extra in fields.items():
            coord = extra.get("dim_coords")
            if coord:
                coord = coord.split(".")[-1]
                if coord not in coordinates:
                    raise ValueError(
                        f"Field '{name}' uses dim_coords='{coord}', but it is "
                        "not a CoordinateField."
                    )
            references = extra.get("references")
            if references:
                references = references.split(".")[-1]
            if references and references not in coordinates:
                raise ValueError(
                    f"Field '{name}' references unknown coordinate "
                    f"'{references}'."
                )
            selects = extra.get("selects")
            if selects:
                selects = selects.split(".")[-1]
            if selects:
                if name not in coordinates:
                    raise ValueError(
                        f"Selection '{name}' must be a CoordinateField."
                    )
                if references != selects:
                    raise ValueError(
                        f"Selection '{name}' must reference the coordinate it "
                        f"selects ('{selects}')."
                    )
                if selects in selections:
                    raise ValueError(
                        f"Coordinate '{selects}' has multiple default selections: "
                        f"'{selections[selects]}' and '{name}'."
                    )
                selections[selects] = name
            partition_by = extra.get("partition_by")
            if partition_by:
                partition_by = partition_by.split(".")[-1]
            replicated = extra.get("replicated", False)
            if replicated and name not in coordinates:
                raise ValueError(
                    f"replicated=True is only valid on CoordinateField, got '{name}'."
                )
            if replicated and (
                    name == self.partition_key or partition_by or references):
                raise ValueError(
                    f"Replicated coordinate '{name}' cannot define partition lineage."
                )
            if (
                name in coordinates
                and name != self.partition_key
                and not partition_by
                and not references
                and not replicated
            ):
                raise ValueError(
                    f"Coordinate '{name}' has no ownership lineage. Declare "
                    "partition_by/references or set replicated=True."
                )
            if partition_by:
                if name not in coordinates:
                    raise ValueError(
                        f"partition_by is only valid on CoordinateField, got '{name}'."
                    )
                if partition_by not in fields:
                    raise ValueError(
                        f"Coordinate '{name}' partitions by undeclared field "
                        f"'{partition_by}'."
                    )
                via_extra = fields[partition_by]
                via_coord = (via_extra.get("dim_coords") or "").split(".")[-1]
                if via_coord != name:
                    raise ValueError(
                        f"Partition field '{partition_by}' must be aligned to "
                        f"coordinate '{name}', got dim_coords={via_coord!r}."
                    )
                if not via_extra.get("references"):
                    raise ValueError(
                        f"Partition field '{partition_by}' must declare references."
                    )

        return {
            "fields": fields,
            "coordinates": coordinates,
            "selections": selections,
        }

    def _coordinate_is_partitioned(self, coordinate: str) -> bool:
        fields = self.partition_metadata["fields"]
        extra = fields[coordinate]
        if extra.get("replicated", False):
            return False
        return bool(
            coordinate == self.partition_key
            or extra.get("partition_by")
            or extra.get("references")
        )

    def _resolve_output_binding(
        self, field_info: Any,
    ) -> Tuple[Optional[str], Optional[torch.Tensor], Optional[str], Optional[torch.Tensor]]:
        """Resolve output indices and coordinates from a field's logical axis."""
        extra = getattr(field_info, "json_schema_extra", {}) or {}
        policy = extra.get("output", "auto")
        if policy == "disabled":
            return None, None, None, None

        dim_coords = extra.get("dim_coords")
        if not dim_coords:
            return None, None, None, None
        coordinate = dim_coords.split(".")[-1]
        if coordinate not in self.variable_map:
            raise ValueError(
                f"Output coordinate '{coordinate}' is not available in opened modules."
            )
        coord_module, coord_attr, _ = self.variable_map[coordinate]
        coord_tensor = getattr(coord_module, coord_attr)

        selection = None
        if policy == "auto":
            selection = self.partition_metadata["selections"].get(coordinate)
        if selection is None:
            return None, None, coordinate, coord_tensor

        selection_module, selection_attr, _ = self.variable_map[selection]
        selection_tensor = getattr(selection_module, selection_attr)
        if selection_tensor is None:
            return None, None, coordinate, coord_tensor

        if selection_tensor.numel() == 0:
            indices = torch.empty(
                0, dtype=torch.int32, device=self.device,
            )
        else:
            indices = find_indices_in_torch(selection_tensor, coord_tensor)
        if torch.any(indices < 0):
            missing = selection_tensor[indices < 0][:5].detach().cpu().tolist()
            raise ValueError(
                f"Selection '{selection}' contains values absent from coordinate "
                f"'{coordinate}'; examples: {missing}."
            )
        indices = indices.to(self.device)
        index_name = f"__selection_idx__{selection}"
        return index_name, indices, selection, selection_tensor

    def _bind_output_metadata(self, field_info: Any) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Return per-model field metadata with runtime output bindings attached."""
        bound = copy(field_info)
        extra = dict(getattr(field_info, "json_schema_extra", {}) or {})
        index_name, indices, coord_name, coord_tensor = self._resolve_output_binding(field_info)
        extra["output_index"] = index_name
        extra["output_coord"] = coord_name
        bound.json_schema_extra = extra
        tensors: Dict[str, torch.Tensor] = {}
        if index_name is not None and indices is not None:
            tensors[index_name] = indices
        if coord_name is not None and coord_tensor is not None:
            tensors[coord_name] = coord_tensor
        return bound, tensors

    @cached_property
    def variable_group_mapping(self) -> Dict[str, str]:
        """Map each distributed input field to its coordinate axis."""
        fields = self.partition_metadata["fields"]
        coordinates = self.partition_metadata["coordinates"]
        mapping: Dict[str, str] = {}
        for name, extra in fields.items():
            if name in coordinates:
                if self._coordinate_is_partitioned(name):
                    mapping[name] = name
                continue
            coord = extra.get("dim_coords")
            if coord:
                coord = coord.split(".")[-1]
                if self._coordinate_is_partitioned(coord):
                    mapping[name] = coord
        return mapping

    def _field_coordinate(self, field_info: Any) -> Optional[str]:
        extra = getattr(field_info, "json_schema_extra", {}) or {}
        coordinate = extra.get("dim_coords")
        if not coordinate:
            return None
        coordinate = coordinate.split(".")[-1]
        return coordinate if self._coordinate_is_partitioned(coordinate) else None

    def _coordinate_group_values(
        self, coordinate: str, resolving: Optional[set[str]] = None,
    ) -> np.ndarray:
        cache = self.__dict__.setdefault("_coordinate_group_cache", {})
        if coordinate in cache:
            return cache[coordinate]
        resolving = set() if resolving is None else resolving
        if coordinate in resolving:
            raise ValueError(f"Partition coordinate cycle detected at '{coordinate}'.")
        resolving.add(coordinate)

        fields = self.partition_metadata["fields"]
        extra = fields[coordinate]
        keys = np.asarray(self.input_proxy[coordinate])
        if keys.ndim != 1:
            raise ValueError(f"Coordinate '{coordinate}' must be 1-D.")
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"Coordinate '{coordinate}' must contain unique values.")

        if coordinate == self.partition_key:
            groups = np.asarray(self.input_proxy[self.partition_group])
            if groups.ndim != 1 or len(groups) != len(keys):
                raise ValueError(
                    f"partition_group '{self.partition_group}' must align with "
                    f"partition_key '{coordinate}'."
                )
        else:
            via = extra.get("partition_by")
            references = extra.get("references")
            if via:
                via = via.split(".")[-1]
                target = fields[via]["references"]
                target = target.split(".")[-1]
                idx = self._reference_index(via)
            elif references:
                target = references.split(".")[-1]
                idx = self._reference_index(coordinate)
            else:
                raise ValueError(
                    f"Coordinate '{coordinate}' has no partition lineage."
                )
            target_groups = self._coordinate_group_values(target, resolving)
            groups = target_groups[idx]

        resolving.remove(coordinate)
        cache[coordinate] = groups
        return groups

    def _reference_index(self, name: str) -> np.ndarray:
        """Validate one loaded reference and cache its global target indices."""
        cache = self.__dict__.setdefault("_reference_index_cache", {})
        if name in cache:
            return cache[name]
        fields = self.partition_metadata["fields"]
        extra = fields[name]
        target = (extra.get("references") or "").split(".")[-1]
        if not target:
            raise ValueError(f"Field '{name}' is not a reference field.")
        if name not in self.input_proxy or target not in self.input_proxy:
            raise ValueError(
                f"Reference field '{name}' requires loaded coordinate '{target}'."
            )
        values = np.asarray(self.input_proxy[name])
        target_values = np.asarray(self.input_proxy[target])
        if values.ndim != 1 or target_values.ndim != 1:
            raise ValueError(
                f"Reference field '{name}' and coordinate '{target}' must be 1-D."
            )
        idx = find_indices_in(values, target_values)
        missing = idx < 0
        if np.any(missing):
            examples = values[missing][:5].tolist()
            raise ValueError(
                f"Reference field '{name}' has {int(missing.sum())} value(s) "
                f"absent from coordinate '{target}'; examples: {examples}."
            )
        cache[name] = idx
        return idx

    def _group_indices_for_rank(self, coordinate: str) -> np.ndarray:
        groups = self._coordinate_group_values(coordinate)
        if not np.issubdtype(groups.dtype, np.integer):
            raise ValueError(
                f"Resolved groups for coordinate '{coordinate}' must be integer."
            )
        if np.any(groups < 0):
            raise ValueError(f"Resolved groups for coordinate '{coordinate}' are negative.")
        try:
            ranks = self.group_id_to_rank[groups]
        except KeyError as exc:
            raise ValueError(
                f"Resolved groups for coordinate '{coordinate}' are unknown."
            ) from exc
        return np.nonzero(ranks == self.rank)[0]

    def _logical_axis(self, field_name: str, field_info: Any, shape: Tuple[int, ...]) -> int:
        """Return the physical axis corresponding to tensor_shape[0]."""
        extra = getattr(field_info, "json_schema_extra", {}) or {}
        logical_ndim = len(extra.get("tensor_shape", ()))
        if len(shape) == logical_ndim:
            return 0
        if self.num_trials is not None and len(shape) == logical_ndim + 1:
            if shape[0] != self.num_trials:
                raise ValueError(
                    f"Batched field '{field_name}' has leading size {shape[0]}, "
                    f"expected num_trials={self.num_trials}."
                )
            return 1
        raise ValueError(
            f"Field '{field_name}' has rank {len(shape)}, but tensor_shape declares "
            f"{logical_ndim} logical dimension(s)."
        )

    def _validate_input_axes(self, fields_to_load: Dict[str, Any]) -> None:
        """Validate every loaded tensor's logical first axis against dim_coords."""
        for name, info in fields_to_load.items():
            if name not in self.input_proxy:
                continue
            extra = getattr(info, "json_schema_extra", {}) or {}
            coord = extra.get("dim_coords")
            if not coord:
                continue
            coord = coord.split(".")[-1]
            if coord not in self.input_proxy:
                raise ValueError(
                    f"Field '{name}' requires missing dim_coords '{coord}'."
                )
            shape = self.input_proxy.get_var_shape(name)
            coord_shape = self.input_proxy.get_var_shape(coord)
            if len(coord_shape) != 1:
                raise ValueError(f"Coordinate '{coord}' must be 1-D, got {coord_shape}.")
            axis = self._logical_axis(name, info, shape)
            if shape[axis] != coord_shape[0]:
                raise ValueError(
                    f"Field '{name}' logical axis length {shape[axis]} does not match "
                    f"dim_coords '{coord}' length {coord_shape[0]}."
                )

    def _validate_reference_integrity(self, module_data: Dict[str, Any]) -> None:
        """Validate loaded reference values against their global coordinates."""
        def as_numpy(value: Any) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        for name, extra in self.partition_metadata["fields"].items():
            target = extra.get("references")
            if not target:
                continue
            target = target.split(".")[-1]
            if name not in module_data or target not in self.input_proxy:
                continue
            values = as_numpy(module_data[name])
            targets = as_numpy(self.input_proxy[target])
            if values.ndim != 1 or targets.ndim != 1:
                raise ValueError(
                    f"Reference field '{name}' and coordinate '{target}' "
                    "must both be 1-D."
                )
            idx = find_indices_in(values, targets)
            missing = idx < 0
            if np.any(missing):
                examples = values[missing][:5].tolist()
                raise ValueError(
                    f"Reference field '{name}' has {int(missing.sum())} value(s) "
                    f"absent from global coordinate '{target}'; "
                    f"examples: {examples}."
                )

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
        mapping = {}
        # Track which unqualified names have a virtual-with-expr entry
        _has_expr_virtual: Dict[str, bool] = {}

        for module_name, _, field_name, field_info in self._iter_all_fields(include_computed=True):
            module = self.get_module(module_name)
            if module is None:
                continue

            # Determine ID attribute for coordinate lookup
            id_attr = None
            dim_coords = None
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                dim_coords = field_info.json_schema_extra.get("dim_coords")
            if dim_coords:
                id_attr = dim_coords

            entry = (module, field_name, id_attr)
            extra = getattr(field_info, 'json_schema_extra', {}) or {}
            is_expr_virtual = (extra.get('category') == 'virtual'
                               and extra.get('expr'))

            # Unqualified name: expr-virtual takes priority
            if field_name not in mapping:
                mapping[field_name] = entry
                _has_expr_virtual[field_name] = is_expr_virtual
            elif is_expr_virtual and not _has_expr_virtual.get(field_name):
                # New entry is an expr-virtual, existing is a source → overwrite
                mapping[field_name] = entry
                _has_expr_virtual[field_name] = True
            elif not is_expr_virtual and _has_expr_virtual.get(field_name):
                # Existing is expr-virtual, new is source → keep existing
                pass
            else:
                # Both same kind → last writer wins (original behaviour)
                mapping[field_name] = entry

            # Qualified name: always set
            mapping[f"{module_name}.{field_name}"] = entry

        for module_name in self.opened_modules:
            module = self.get_module(module_name)
            if module is None:
                continue
            for field_name in module.get_reference_index_fields():
                info = module.get_reference_index_field_info(field_name)
                extra = info.json_schema_extra or {}
                entry = (module, field_name, extra.get("dim_coords"))
                mapping.setdefault(field_name, entry)
                mapping[f"{module_name}.{field_name}"] = entry

        return mapping

    @cached_property
    def group_id_to_rank(self) -> GroupRankLookup:
        """
        Load primary group variable from InputProxy and compute
        a sparse ID->rank lookup using compute_group_to_rank.
        """
        if self.partition_group not in self.input_proxy:
            raise ValueError(
                f"Missing partition_group '{self.partition_group}' in InputProxy."
            )
        grp = self.input_proxy[self.partition_group]
        group_ids, ranks = compute_group_to_rank(self.world_size, np.asarray(grp))
        return GroupRankLookup(group_ids=group_ids, ranks=ranks)

    def initialize_statistics_aggregator(self) -> None:
        """
        Initialize the statistics aggregator for streaming NetCDF output.
        Registers variables together with output bindings inferred from coordinates.
        Avoids duplicate registration.
        """
        if not self.variables_to_save:
            return

        self._statistics_aggregator = StatisticsAggregator(
            device=self.device,
            output_dir=self.output_full_dir,
            rank=self.rank,
            num_workers=self.output_workers,
            compression=self.output_compression,
            complevel=self.output_complevel,
            output_split_by_year=self.output_split_by_year,
            num_trials=self.num_trials or 1,
            save_kernels=self.save_kernels,
            max_pending_steps=self.max_pending_steps,
            calendar=self.calendar,
            in_memory_mode=self.in_memory_output,
            result_device=self.result_device,
            save_precision=torch.float32,
            output_chunksizes=self.output_chunksizes,
        )

        registered_vars = set()

        # Normalize variables_to_save (op -> vars) into var -> set[ops]
        # Note: argmax/argmin can now be specified explicitly for timing information
        allowed_ops = {"mean", "max", "min", "last", "first", "mid", "sum", "argmax", "argmin"}
        topk_pattern = re.compile(r'^(max|min)(\d+)$')
        argtopk_pattern = re.compile(r'^arg(max|min)(\d*)$')  # argmax, argmin, argmax3, argmin3

        var_to_ops: Dict[str, List[str]] = {}
        # Stores explicit expressions provided by user: alias -> expression
        explicit_expressions: Dict[str, str] = {}

        for op, vars_val in self.variables_to_save.items():
            # ── Static per-saved-point metadata (op == "static") ──
            # Short-circuit: no inner/outer op parsing, no aggregation
            # buffers.  Each listed variable is gathered once by its
            # field's resolved coordinate selection and handed to the aggregator
            # via register_static, which will write it into every
            # output NC at file creation time.
            if op == "static":
                items = [vars_val] if isinstance(vars_val, str) else list(vars_val)
                for name in items:
                    if not isinstance(name, str):
                        raise ValueError(
                            "variables_to_save['static'] entries must be "
                            f"plain field names, got {name!r}")
                    if name not in self.variable_map:
                        raise ValueError(
                            f"Static variable '{name}' not found in any "
                            f"opened module")
                    module, attr, _ = self.variable_map[name]
                    tensor = getattr(module, attr)
                    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                        raise ValueError(
                            f"Static variable '{name}' must be a one-dimensional tensor"
                        )
                    info = module.get_tensor_field_info(attr)
                    if info is None:
                        raise ValueError(f"Static variable '{name}' has no field metadata")
                    if (getattr(info, "json_schema_extra", {}) or {}).get(
                            "output", "auto") == "disabled":
                        raise ValueError(f"Variable '{name}' is disabled for output")
                    bound_info, binding_tensors = self._bind_output_metadata(info)
                    static_coordinate = (
                        bound_info.json_schema_extra or {}
                    ).get("output_coord")
                    if static_coordinate is None:
                        raise ValueError(
                            f"Static variable '{name}' must declare dim_coords so "
                            "it can be scoped to compatible outputs."
                        )
                    if name == static_coordinate:
                        continue
                    output_index_name = (
                        bound_info.json_schema_extra or {}
                    ).get("output_index")
                    output_index = binding_tensors.get(output_index_name)
                    self._statistics_aggregator.register_static(
                        name,
                        tensor,
                        output_index=output_index,
                        coordinate=static_coordinate,
                    )
                continue

            op_l = str(op).lower()
            op_parts = op_l.split('_')

            # Validate op parts
            for p in op_parts:
                # Check against allowed patterns
                if p not in allowed_ops and not topk_pattern.match(p) and not argtopk_pattern.match(p):
                     raise ValueError(f"Invalid op '{op}'. Component '{p}' not in allowed ops: {sorted(allowed_ops)}, top-k pattern, or arg-top-k pattern.")

            # Single operation (no underscore) - this is the inner aggregation
            # topK and arg operations are NOT allowed standalone
            if len(op_parts) == 1:
                single_op = op_parts[0]
                if topk_pattern.match(single_op):
                    raise ValueError(f"Invalid op '{op}': standalone top-k ops are not allowed. Use a compound form like '{op}_last' or '{op}_max' instead.")
                if argtopk_pattern.match(single_op) or single_op in ('argmax', 'argmin'):
                    raise ValueError(f"Invalid op '{op}': arg operations (argmax, argmin, argmax3, etc.) cannot be used alone. They are only valid as outer operations in compound form like 'argmax_mean'.")

            if len(op_parts) > 1:
                outer, inner = op_parts[0], op_parts[1]
                # Check for outer restriction: mid cannot be an outer op
                if outer == 'mid':
                    raise ValueError(f"Invalid composite op '{op}': 'mid' cannot be used as an outer operation. It is only valid as an inner op (standalone 'mid').")
                # Check for inner restriction: top-k, arg-top-k, argmax/argmin cannot be inner ops
                if topk_pattern.match(inner) or argtopk_pattern.match(inner):
                    raise ValueError(f"Invalid composite op '{op}': '{inner}' (top-k/arg-top-k) cannot be used as an inner operation.")
                if inner in ('argmax', 'argmin'):
                    raise ValueError(f"Invalid composite op '{op}': '{inner}' cannot be used as an inner operation. arg operations are only valid as outer ops.")
                if len(op_parts) > 2:
                    raise ValueError(f"Invalid composite op '{op}': Only 2 levels of operations are supported.")

            # Standardize vars_val to a list of items
            if isinstance(vars_val, (str, tuple)):
                items = [vars_val]
            elif isinstance(vars_val, list):
                items = vars_val
            else:
                raise ValueError(f"variables_to_save['{op}'] must be a string, tuple (name, expr), or list of them.")

            for item in items:
                if isinstance(item, str):
                    name = item
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    name = item[0]
                    expr = item[1]
                    if name in explicit_expressions and explicit_expressions[name] != expr:
                         raise ValueError(f"Conflicting expressions for alias '{name}': '{explicit_expressions[name]}' vs '{expr}'")
                    explicit_expressions[name] = expr
                elif isinstance(item, dict):
                    if len(item) != 1:
                        raise ValueError(f"Dictionary item in variables_to_save['{op}'] must have exactly one key-value pair {{alias: expr}}. Got: {item}")
                    name, expr = next(iter(item.items()))
                    if name in explicit_expressions and explicit_expressions[name] != expr:
                         raise ValueError(f"Conflicting expressions for alias '{name}': '{explicit_expressions[name]}' vs '{expr}'")
                    explicit_expressions[name] = expr
                else:
                    raise ValueError(f"Invalid item in variables_to_save['{op}']: {item}. Must be string, dict {{name: expr}}, or (name, expr) tuple.")

                var_to_ops.setdefault(name, [])
                if op_l not in var_to_ops[name]:
                    var_to_ops[name].append(op_l)

        # Handle ad-hoc expressions in var_to_ops
        adhoc_virtuals: Dict[str, Any] = {}

        from hydroforge.aggregator.scatter_expr import parse_scatter_expr

        def get_field_meta(name):
            mod, attr, _ = self.variable_map[name]
            info = mod.get_tensor_field_info(attr)
            if info:
                extra = info.json_schema_extra or {}
                dim_coords = extra.get("dim_coords")
                if dim_coords:
                    dim_coords = dim_coords.split(".")[-1]
                return (extra.get("output", "auto"), dim_coords)
            return (None, None)

        current_vars = list(var_to_ops.keys())
        for var_name in current_vars:
            if var_name in self.variable_map:
                # Check if existing field is a virtual with scatter expr
                mod, attr, _ = self.variable_map[var_name]
                info = mod.get_tensor_field_info(attr)
                if info:
                    extra = getattr(info, 'json_schema_extra', {}) or {}
                    if extra.get('category') == 'virtual' and extra.get('expr'):
                        scatter = parse_scatter_expr(extra['expr'])
                        if scatter:
                            # Validate: index_var must exist in variable_map
                            if scatter.index_var not in self.variable_map:
                                raise ValueError(
                                    f"Scatter expression in '{var_name}': index variable "
                                    f"'{scatter.index_var}' not found in any module."
                                )
                            # Validate: all value tokens must exist
                            for tok in scatter.value_tokens:
                                if tok not in self.variable_map:
                                    raise ValueError(
                                        f"Scatter expression in '{var_name}': value token "
                                        f"'{tok}' not found in any module."
                                    )
                        else:
                            from hydroforge.aggregator.scatter_expr import extract_tokens
                            deps = [
                                token for token in extract_tokens(extra['expr'])
                                if token in self.variable_map
                            ]
                            target_coord = extra.get("dim_coords")
                            if target_coord:
                                target_coord = target_coord.split(".")[-1]
                            dep_coords = {
                                get_field_meta(dep)[1]
                                for dep in deps
                                if get_field_meta(dep)[1] is not None
                            }
                            if len(dep_coords) > 1:
                                raise ValueError(
                                    f"Virtual field '{var_name}' mixes coordinate axes "
                                    f"{sorted(dep_coords)}."
                                )
                            if dep_coords and target_coord not in dep_coords:
                                raise ValueError(
                                    f"Virtual field '{var_name}' declares dim_coords="
                                    f"'{target_coord}', but its expression uses "
                                    f"coordinate '{next(iter(dep_coords))}'."
                                )
                continue

            # If provided as tuple, usage is explicit. If string, assume string IS the expression.
            expression = explicit_expressions.get(var_name, var_name)

            # Check if it looks like an expression (simple heuristic)
            # and if we can resolve its tokens

            # Try to parse as scatter expression first
            scatter = parse_scatter_expr(expression)

            if scatter is not None:
                raise ValueError(
                    f"Scatter expression '{expression}' for ad-hoc variable '{var_name}' "
                    f"cannot be used inline. Scatter virtual fields must be defined in "
                    f"a module using computed_tensor_field(category='virtual', expr=...) "
                    f"with dim_coords pointing to the target dimension."
                )

            # ── Plain elementwise expression ──
            from hydroforge.aggregator.scatter_expr import extract_tokens
            tokens = extract_tokens(expression)

            valid_deps = []
            for t in tokens:
                 if t in self.variable_map:
                      valid_deps.append(t)

            # If no dependencies found, skip (likely invalid or constant)
            if not valid_deps:
                 continue

            # Consistency Check
            # Ensure all dependencies share the same output policy and logical axis.
            ref_meta = get_field_meta(valid_deps[0])
            for dep in valid_deps[1:]:
                 curr_meta = get_field_meta(dep)
                 if curr_meta != ref_meta:
                      raise ValueError(
                          f"Inconsistent metadata in virtual variable '{var_name}' (expression: '{expression}'). "
                          f"Dependency '{valid_deps[0]}' has {ref_meta}, but '{dep}' has {curr_meta}. "
                          "All dependencies in an expression must share the same output policy and dim_coords to ensure correct parallel iteration."
                      )

            # Create FieldInfo
            output, dim_coords = ref_meta

            new_info = Field(
                description=f"Ad-hoc expression: {expression}",
                json_schema_extra={
                    "category": "virtual",
                    "expr": expression,
                    "dim_coords": dim_coords,
                    "output": output,
                }
            )

            adhoc_virtuals[var_name] = new_info

        # No need to sanitize names or update var_to_ops keys further,
        # as var_to_ops already uses the keys we intend to register.

        if not var_to_ops:
            raise ValueError(
                "variables_to_save cannot contain only 'static' entries; "
                "at least one dynamic output is required to define the output file."
            )


        registered_vars_by_shape: Dict[str, List[str]] = {}

        # 1. Expand variables to include dependencies of virtual variables
        vars_to_process = list(var_to_ops.keys())
        vars_seen = set(vars_to_process)
        idx = 0
        while idx < len(vars_to_process):
            curr_var = vars_to_process[idx]
            idx += 1

            if curr_var not in self.variable_map:
                if curr_var in adhoc_virtuals:
                    field_info = adhoc_virtuals[curr_var]
                    # Check for deps in adhoc virtuals
                    expr = field_info.json_schema_extra.get("expr", "")
                    from hydroforge.aggregator.scatter_expr import \
                        extract_tokens as _et
                    deps = _et(expr)
                    for d in deps:
                        if d not in vars_seen:
                            vars_seen.add(d)
                            vars_to_process.append(d)
                continue

            module_instance, attr_name, _ = self.variable_map[curr_var]

            # Check normal fields
            field_info = module_instance.get_tensor_field_info(attr_name)

            if field_info:
                cat = field_info.json_schema_extra.get("category", "param")
                if cat == 'virtual':
                    expr = field_info.json_schema_extra.get("expr") or ""
                    # For scatter expressions, extract deps from value_tokens + index_var
                    scatter = parse_scatter_expr(expr)
                    if scatter:
                        scatter_deps = scatter.value_tokens | {scatter.index_var}
                        for d in scatter_deps:
                            if d not in vars_seen:
                                vars_seen.add(d)
                                vars_to_process.append(d)
                    else:
                        # Simple regex for potential tokens (supports dotted names)
                        from hydroforge.aggregator.scatter_expr import \
                            extract_tokens as _et2
                        deps = _et2(expr)
                        for d in deps:
                            if d not in vars_seen:
                                vars_seen.add(d)
                                vars_to_process.append(d)

        # 2. Register all variables (original + dependencies)
        registered_vars = set()
        for var_name in vars_to_process:
            if var_name not in self.variable_map:
                if var_name in adhoc_virtuals:
                     if var_name not in registered_vars:
                         self._statistics_aggregator.register_virtual_tensor(var_name, adhoc_virtuals[var_name])
                         registered_vars.add(var_name)
                     continue

                if self.rank == 0:
                    print(f"Warning: Variable '{var_name}' not found in variable_map or adhoc list. Skipping.")
                continue

            module_instance, attr_name, _ = self.variable_map[var_name]

            if not hasattr(module_instance, attr_name):
                continue

            tensor = getattr(module_instance, attr_name)
            field_info = module_instance.get_tensor_field_info(attr_name)

            if field_info is None:
                continue

            raw_extra = field_info.json_schema_extra or {}
            if var_name in var_to_ops and raw_extra.get("output", "auto") == "disabled":
                raise ValueError(f"Variable '{var_name}' is disabled for output")

            field_info, binding_tensors = self._bind_output_metadata(field_info)

            # Check category
            category = field_info.json_schema_extra.get("category", "param")
            allowed_cats = ("state", "shared_state", "init_state", "param", "virtual")
            # Topology tensors are allowed as dependencies (e.g. scatter index)
            # but not as direct outputs
            is_dependency_only = var_name not in var_to_ops
            if category == "topology" and is_dependency_only:
                pass  # allow topology as dependency
            elif category not in allowed_cats:
                    # Only warn if it's an explicitly requested output
                    if var_name in var_to_ops:
                        print(f"[rank {self.rank}] Warning: Variable '{var_name}' is category '{category}', skipping output (allowed: {allowed_cats}).")
                    continue

            # Check dimensionality and restrictions (only for requested outputs)
            if var_name in var_to_ops:
                ops = var_to_ops[var_name]
                if category != 'virtual':
                    output_index = field_info.json_schema_extra.get("output_index")
                    is_full_output = output_index is None
                    if is_full_output:
                        for op in ops:
                            op_base = op.split('_')[0]
                            if topk_pattern.match(op_base) or argtopk_pattern.match(op_base):
                                raise ValueError(
                                    f"Operation '{op}' is not supported for full-output "
                                    f"variable '{var_name}' yet."
                                )

                    # 1D is usually (N,), 2D is (N, Level).
                    # With trials: 1D is (T, N), 2D is (T, N, Level)
                    limit = 1
                    if self.num_trials and self.num_trials > 1:
                        limit = 2

                    is_real_2d = tensor.ndim > limit

                    if is_real_2d and not is_full_output:
                        for op in ops:
                            op_base = op.split('_')[0]
                            if topk_pattern.match(op_base) or op_base in ('max', 'min'):
                                raise ValueError(f"Operation '{op}' is not allowed for 2D variable '{var_name}' (ndim={tensor.ndim}). Only 'mean', 'sum', 'last', 'first', 'mid' are supported for 2D variables.")

            # Register the main tensor if not already done
            if var_name not in registered_vars:
                if category == 'virtual':
                    expr = field_info.json_schema_extra.get('expr', '')
                    if expr:
                        # Virtual with expression → metadata-only (scatter / plain)
                        self._statistics_aggregator.register_virtual_tensor(var_name, field_info)
                    elif tensor is not None:
                        # Virtual with no expr → source buffer; register as real
                        # tensor so scatter kernels can read the data.
                        self._statistics_aggregator.register_tensor(var_name, tensor, field_info)
                        shape_str = str(tuple(tensor.shape))
                        if shape_str not in registered_vars_by_shape:
                            registered_vars_by_shape[shape_str] = []
                        registered_vars_by_shape[shape_str].append(var_name)
                    else:
                        self._statistics_aggregator.register_virtual_tensor(var_name, field_info)
                else:
                    self._statistics_aggregator.register_tensor(var_name, tensor, field_info)
                    shape_str = str(tuple(tensor.shape))
                    if shape_str not in registered_vars_by_shape:
                        registered_vars_by_shape[shape_str] = []
                    registered_vars_by_shape[shape_str].append(var_name)

                registered_vars.add(var_name)

            # Register the runtime-generated output index and coordinate tensors.
            output_index = field_info.json_schema_extra.get("output_index")
            if output_index and output_index not in registered_vars:
                if output_index in binding_tensors:
                    save_tensor = binding_tensors[output_index]
                    self._statistics_aggregator.register_tensor(output_index, save_tensor, {})
                    registered_vars.add(output_index)
                    shape_str = str(tuple(save_tensor.shape))
                    if shape_str not in registered_vars_by_shape:
                        registered_vars_by_shape[shape_str] = []
                    registered_vars_by_shape[shape_str].append(output_index)
                else:
                    raise ValueError(
                        f"Runtime output index '{output_index}' was not resolved for '{var_name}'"
                    )

            output_coord = field_info.json_schema_extra.get("output_coord")
            if output_coord and output_coord not in registered_vars:
                if output_coord in binding_tensors:
                    coord_tensor = binding_tensors[output_coord]
                    self._statistics_aggregator.register_tensor(output_coord, coord_tensor, {})
                    registered_vars.add(output_coord)
                    shape_str = str(tuple(coord_tensor.shape))
                    if shape_str not in registered_vars_by_shape:
                        registered_vars_by_shape[shape_str] = []
                    registered_vars_by_shape[shape_str].append(output_coord)
                else:
                    raise ValueError(
                        f"Runtime output coordinate '{output_coord}' was not resolved for '{var_name}'"
                    )

        if registered_vars_by_shape:
            for shape_str, vars_list in registered_vars_by_shape.items():
                print(f"[rank {self.rank}]: Registered tensors for streaming: {', '.join(vars_list)} (shape: {shape_str})")

        self._statistics_aggregator.initialize_streaming_aggregation(
            variable_ops=var_to_ops
        )

    def update_statistics(self, sub_step: int, num_sub_steps: int, flags: int, weight: float, total_weight: float = 0.0, BLOCK_SIZE: int = 128) -> None:
        """
        Update streaming statistics with a time weight.
        Args:
            sub_step: Current sub-step index (0-based)
            num_sub_steps: Total number of sub-steps
            flags: Packed boolean flags (bit 0=is_first, 1=is_last, 2=is_outer_first, 3=is_outer_last)
            weight: dt in seconds for this sub-step (time-weighted accumulation)
            total_weight: Total elapsed time for the full inner window
            BLOCK_SIZE: GPU block size
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.update_statistics(sub_step, num_sub_steps, flags, weight, total_weight, BLOCK_SIZE=BLOCK_SIZE)

    def finalize_time_step(self, current_time: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize time step in aggregator (write current means to disk).
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.finalize_time_step(current_time)

    def close_statistics_aggregator(self) -> None:
        """Flush and close the statistics aggregator if it was initialized."""
        if self._statistics_aggregator is not None:
            self._statistics_aggregator._shutdown()

    def get_output_results(self, as_stacked: bool = True) -> Dict[str, torch.Tensor]:
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
        if self._statistics_aggregator is None:
            raise RuntimeError("Statistics aggregator not initialized")
        return self._statistics_aggregator.get_results(as_stacked=as_stacked)

    def get_output_result(self, variable_name: str, op: str = "mean", as_stacked: bool = True) -> torch.Tensor:
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
        if self._statistics_aggregator is None:
            raise RuntimeError("Statistics aggregator not initialized")
        return self._statistics_aggregator.get_result(variable_name, op, as_stacked=as_stacked)

    def get_output_time_index(self) -> int:
        """Get the current output time index (number of finalized time steps)."""
        if self._statistics_aggregator is None:
            return 0
        return self._statistics_aggregator.get_time_index()

    def reset_output_time_index(self) -> None:
        """Reset the output time index to 0 for a new simulation run (in-memory mode only)."""
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.reset_time_index()

    def shard_param(self) -> Dict[str, Any]:
        """
        Load fields by reading from InputProxy and slicing in-memory per rank.
        """
        module_data: Dict[str, Any] = {}

        # Collect unique fields to load across all opened modules
        fields_to_load: Dict[str, Any] = {}
        for _, _, field_name, field_info in self._iter_all_fields(include_computed=False):
            # Dependency references are injected from already constructed
            # modules according to the module DAG; they are not dataset fields.
            if getattr(field_info, "exclude", False):
                continue
            if field_name not in fields_to_load:
                fields_to_load[field_name] = field_info

        try:
            injected_vars = getattr(self.input_proxy, "injected_vars", set())
            unknown_injected = sorted(
                name for name in injected_vars
                if name not in fields_to_load
            )
            if unknown_injected:
                raise KeyError(
                    f"Injected InputProxy variables are not fields of opened modules: {unknown_injected}. "
                    f"Available module fields: {sorted(fields_to_load)}"
                )

            # Validate required fields exist
            missing_required = [
                name for name, info in fields_to_load.items()
                if info.is_required() and name not in self.input_proxy
            ]
            if missing_required:
                raise KeyError(
                    f"Required fields missing from InputProxy: {missing_required}. "
                    f"Available fields: {list(self.input_proxy.data.keys())}"
                )

            self._validate_input_axes(fields_to_load)

            # Pre-compute indices per group var for current rank
            group_vars_needed = set()
            for name in fields_to_load.keys():
                if name in self.variable_group_mapping:
                    # Only need group var if the variable itself is in the dataset
                    if name in self.input_proxy:
                        group_vars_needed.add(self.variable_group_mapping[name])
            group_indices_cache: Dict[str, np.ndarray] = {}
            for group_var in group_vars_needed:
                group_indices_cache[group_var] = self._group_indices_for_rank(group_var)
            # Full-domain foreign-key indices are needed only while resolving
            # partition lineage; do not retain them for the model lifetime.
            self.__dict__.pop("_reference_index_cache", None)

            print(f"[rank {self.rank}]: Loading data for modules {self.opened_modules}")

            def to_torch(arr: Any) -> torch.Tensor:
                # Use as_tensor to avoid unnecessary copy; unify float dtype only
                t = torch.as_tensor(arr)
                if t.is_floating_point() and t.dtype != self.dtype:
                    t = t.to(self.dtype)
                if not t.is_contiguous():
                    t = t.contiguous()
                return t

            def is_tensor_field(info: Any) -> bool:
                json_schema_extra = getattr(info, 'json_schema_extra', None)
                return isinstance(json_schema_extra, dict) and 'tensor_shape' in json_schema_extra

            def prepare_proxy_value(value: Any, info: Any) -> Any:
                if is_tensor_field(info):
                    return to_torch(value)
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        return value.detach().cpu().item()
                    return value.detach().cpu().numpy()
                if isinstance(value, np.ndarray):
                    if value.ndim == 0 or value.size == 1:
                        return value.item()
                if isinstance(value, np.generic):
                    return value.item()
                return value

            # Buckets for logging
            missing_fields = []
            no_local_fields: Dict[str, List[str]] = {}
            distributed_fields: Dict[Tuple[Tuple[int, ...], str], List[str]] = {}
            full_fields = []

            # Sort fields for deterministic processing order
            def sort_key(item):
                name, _ = item
                group = self.variable_group_mapping.get(name, "")
                if group is None:
                    group = ""
                return (str(group), name)

            sorted_fields = sorted(fields_to_load.items(), key=sort_key)

            for field_name, field_info in sorted_fields:
                if field_name not in self.input_proxy:
                    missing_fields.append(field_name)
                    continue

                group_var = self.variable_group_mapping.get(field_name, None)
                if group_var is not None:
                    idx = group_indices_cache[group_var]

                    # Use get_var_shape to check dimensions without loading full data
                    full_shape = self.input_proxy.get_var_shape(field_name)

                    logical_axis = self._logical_axis(
                        field_name, field_info, full_shape,
                    )
                    slicer = ((slice(None), idx)
                              if logical_axis == 1 else idx)

                    # Read only the subset
                    local_np = self.input_proxy.get_subset(field_name, slicer)
                    module_data[field_name] = prepare_proxy_value(local_np, field_info)

                    if idx.size == 0:
                        if group_var not in no_local_fields:
                            no_local_fields[group_var] = []
                        no_local_fields[group_var].append(field_name)
                    else:
                        shape = local_np.shape
                        key = (shape, group_var)
                        if key not in distributed_fields:
                            distributed_fields[key] = []
                        distributed_fields[key].append(field_name)
                else:
                    module_data[field_name] = prepare_proxy_value(self.input_proxy[field_name], field_info)
                    full_fields.append(field_name)

            # Flush logs
            for group_var, fields in no_local_fields.items():
                print(f"[rank {self.rank}]: No local data for distributed fields: {', '.join(fields)} (coordinate: {group_var})")

            for (shape, group_var), fields in distributed_fields.items():
                print(f"[rank {self.rank}]: Loaded distributed fields: {', '.join(fields)} (shape: {shape}, coordinate: {group_var})")

            if full_fields:
                print(f"[rank {self.rank}]: Loaded full fields: {', '.join(full_fields)} (no coordinate axis)")

            if missing_fields:
                print(f"[rank {self.rank}]: Optional fields not in InputProxy, using default: {', '.join(missing_fields)}")

            self._validate_reference_integrity(module_data)

        except Exception as e:
            raise RuntimeError(f"Error loading data from InputProxy: {e}")

        return module_data

    def save_state(self, current_time: Optional[Union[datetime, cftime.datetime]]) -> InputProxy:
        """
        Save model state to InputProxy and NetCDF files (.nc).
        """
        if self.num_trials is not None:
            print(f"[rank {self.rank}] Warning: save_state is not supported for multi-trial simulations.")
            return None

        timestamp = current_time.strftime("%Y%m%d_%H%M%S") if current_time else "latest"

        # Determine file path per-rank
        if self.world_size > 1:
            nc_path = self.output_full_dir / f"model_state_rank{self.rank}_{timestamp}.nc"
        else:
            nc_path = self.output_full_dir / f"model_state_{timestamp}.nc"

        # Collect data
        data = {}
        visited_fields = set()

        saved_distributed = []
        saved_global = []
        skipped_none = set()

        state_group_mapping: Dict[str, str] = {}
        for module_name in self.opened_modules:
            module = self._modules[module_name]
            field_items = [
                (name, info)
                for name, info in module.get_model_fields().items()
                if (getattr(info, "json_schema_extra", {}) or {}).get("category")
                in {"init_state", "state", "shared_state"}
            ]
            field_items.extend(
                (name, info)
                for name, info in module.get_model_computed_fields().items()
                if (getattr(info, "json_schema_extra", {}) or {}).get("category")
                in {"state", "shared_state"}
            )
            for field_name, field_info in field_items:
                if field_name in module.nc_excluded_fields or field_name in visited_fields:
                    continue

                if getattr(field_info, "exclude", False):
                    continue

                coordinate = self._field_coordinate(field_info)
                is_distributed = coordinate is not None
                if coordinate is not None:
                    state_group_mapping[field_name] = coordinate

                # Only rank 0 saves non-distributed variables
                if not is_distributed and self.rank != 0:
                    continue

                val = getattr(module, field_name)

                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()

                if val is None:
                    skipped_none.add(field_name)
                    continue

                data[field_name] = val
                visited_fields.add(field_name)
                skipped_none.discard(field_name)

                if is_distributed:
                    saved_distributed.append(field_name)
                else:
                    saved_global.append(field_name)

        # Coordinates are checkpoint metadata used to restore distributed
        # state by key after rank-wise merge. They are not loaded as state.
        for coordinate in sorted(set(state_group_mapping.values())):
            if coordinate in data:
                continue
            coord_module, coord_attr, _ = self.variable_map[coordinate]
            coord_value = getattr(coord_module, coord_attr)
            if isinstance(coord_value, torch.Tensor):
                coord_value = coord_value.detach().cpu().numpy()
            data[coordinate] = coord_value
            state_group_mapping[coordinate] = coordinate
            saved_distributed.append(coordinate)

        # Create InputProxy
        proxy = InputProxy(data, attrs={
            "title": "hydroforge Model State",
            "history": f"Created by hydroforge at {datetime.now().isoformat()}",
            "source": "hydroforge.modeling.model.AbstractModel.save_state"
        })

        # Write to file
        if nc_path.exists():
             print(f"[rank {self.rank}] Warning: Overwriting existing model state file: {nc_path}")

        proxy.to_nc(nc_path, output_complevel=self.output_complevel if self.world_size == 1 else 0)

        if saved_distributed:
            print(f"[rank {self.rank}] Saved distributed fields: {', '.join(saved_distributed)}")
        if saved_global:
            print(f"[rank {self.rank}] Saved global fields: {', '.join(saved_global)}")
        if skipped_none:
            print(f"[rank {self.rank}] Skipped None fields (not saved): {', '.join(sorted(list(skipped_none)))}")

        if self.world_size > 1:
            dist.barrier()

        # Merge step only done by rank 0
        if self.rank == 0 and self.world_size > 1:
            merged_path = self.output_full_dir / f"model_state_{timestamp}.nc"
            rank_paths = [self.output_full_dir / f"model_state_rank{r}_{timestamp}.nc" for r in range(self.world_size)]

            InputProxy.merge(
                merged_path,
                rank_paths,
                state_group_mapping,
                self.output_complevel,
            )

            # Remove rank files
            for p in rank_paths:
                try:
                    p.unlink()
                except Exception:
                    pass

            print(f"[rank 0] Model state merged to: {merged_path}")

        return proxy

    def load_state(self, proxy: InputProxy) -> None:
        """
        Restore model state from an InputProxy.
        Supports loading from both global (merged) and local (sharded) proxies.
        """
        print(f"[rank {self.rank}] Loading state from InputProxy...")

        loaded_count = 0

        # Cache checkpoint-coordinate indices for restoring distributed fields.
        coordinate_indices_cache: Dict[str, np.ndarray] = {}

        for module_name in self.opened_modules:
            module = self._modules[module_name]
            field_items = [
                (name, info)
                for name, info in module.get_model_fields().items()
                if (getattr(info, "json_schema_extra", {}) or {}).get("category")
                in {"init_state", "state", "shared_state"}
            ]
            field_items.extend(
                (name, info)
                for name, info in module.get_model_computed_fields().items()
                if (getattr(info, "json_schema_extra", {}) or {}).get("category")
                in {"state", "shared_state"}
            )

            for field_name, field_info in field_items:
                if field_name not in proxy:
                    continue

                # Skip excluded fields if they happen to be in proxy (unlikely but safe)
                if getattr(field_info, "exclude", False):
                    continue

                new_val = proxy[field_name]
                current_val = getattr(module, field_name)

                # Handle Tensor fields
                if isinstance(current_val, torch.Tensor):
                    # Convert new_val to numpy if it's a tensor (InputProxy might hold tensors)
                    if isinstance(new_val, torch.Tensor):
                        new_val = new_val.detach().cpu().numpy()

                    new_val = np.asarray(new_val)

                    group_var = self._field_coordinate(field_info)

                    # Distributed checkpoints are merged rank-by-rank and are
                    # therefore not guaranteed to preserve the original global
                    # ordering. Reindex by coordinate values, never by position.
                    if group_var is not None and group_var in proxy:
                        if group_var not in coordinate_indices_cache:
                            coord_module, coord_attr, _ = self.variable_map[group_var]
                            local_coord = getattr(coord_module, coord_attr)
                            if isinstance(local_coord, torch.Tensor):
                                local_coord = local_coord.detach().cpu().numpy()
                            checkpoint_coord = proxy[group_var]
                            if isinstance(checkpoint_coord, torch.Tensor):
                                checkpoint_coord = (
                                    checkpoint_coord.detach().cpu().numpy()
                                )
                            idx = find_indices_in(
                                np.asarray(local_coord),
                                np.asarray(checkpoint_coord),
                            )
                            if np.any(idx < 0):
                                missing = np.asarray(local_coord)[idx < 0][:5].tolist()
                                raise ValueError(
                                    f"Checkpoint coordinate '{group_var}' is missing "
                                    f"local IDs; examples: {missing}."
                                )
                            coordinate_indices_cache[group_var] = idx

                        idx = coordinate_indices_cache[group_var]
                        logical_axis = self._logical_axis(
                            field_name, field_info, tuple(new_val.shape),
                        )
                        slicer = [slice(None)] * new_val.ndim
                        slicer[logical_axis] = idx
                        local_val = new_val[tuple(slicer)]
                        if local_val.shape != tuple(current_val.shape):
                            raise ValueError(
                                f"Shape mismatch for '{field_name}' after coordinate "
                                f"restore: expected {tuple(current_val.shape)}, got "
                                f"{local_val.shape}."
                            )
                        current_val.copy_(
                            torch.as_tensor(local_val).to(current_val.device)
                        )
                        loaded_count += 1
                        continue

                    # Check 1: Direct shape match (Local file or scalar)
                    if new_val.shape == tuple(current_val.shape):
                        current_val.copy_(torch.as_tensor(new_val).to(current_val.device))
                        loaded_count += 1
                        continue

                    # Check 2: Distributed variable needing sharding (Global file)
                    group_var = self._field_coordinate(field_info)
                    if group_var is not None:
                        idx = self._group_indices_for_rank(group_var)

                        # Shard the global data
                        logical_axis = self._logical_axis(
                            field_name, field_info, tuple(new_val.shape),
                        )
                        slicer = [slice(None)] * new_val.ndim
                        slicer[logical_axis] = idx
                        try:
                            local_val = new_val[tuple(slicer)]
                        except IndexError as exc:
                            raise ValueError(
                                f"Cannot shard state field '{field_name}' with shape "
                                f"{new_val.shape} on logical axis {logical_axis}."
                            ) from exc

                        if local_val.shape == tuple(current_val.shape):
                            current_val.copy_(torch.as_tensor(local_val).to(current_val.device))
                            loaded_count += 1
                        else:
                            raise ValueError(
                                f"Shape mismatch for '{field_name}' after sharding: "
                                f"expected {tuple(current_val.shape)}, got {local_val.shape}."
                            )
                    else:
                        raise ValueError(
                            f"Shape mismatch for global state field '{field_name}': "
                            f"expected {tuple(current_val.shape)}, got {new_val.shape}."
                        )

                # Handle Scalar/Other fields
                else:
                    # For scalars, we just set the value
                    # If it's a numpy scalar, convert to python type if needed, or just set
                    if isinstance(new_val, (np.ndarray, np.generic)):
                        if new_val.ndim == 0:
                            new_val = new_val.item()

                    setattr(module, field_name, new_val)
                    loaded_count += 1

        print(f"[rank {self.rank}] Successfully loaded {loaded_count} variables from InputProxy.")

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
        # Note: argmax/argmin can now be specified explicitly for timing information
        allowed_ops = {"mean", "max", "min", "last", "first", "mid", "sum", "argmax", "argmin"}
        topk_pattern = re.compile(r'^(max|min)(\d+)$')
        argtopk_pattern = re.compile(r'^arg(max|min)(\d*)$')  # argmax, argmin, argmax3, argmin3

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
                op_parts = op_l.split('_')

                for p in op_parts:
                    if p not in allowed_ops and not topk_pattern.match(p) and not argtopk_pattern.match(p):
                        raise ValueError(f"Invalid statistics op '{op}'. Component '{p}' not in allowed ops: {sorted(allowed_ops)}, top-k pattern, or arg-top-k pattern.")

                # Single operation (no underscore) - this is the inner aggregation
                # arg operations are NOT allowed as inner operations
                if len(op_parts) == 1:
                    single_op = op_parts[0]
                    if argtopk_pattern.match(single_op) or single_op in ('argmax', 'argmin'):
                        raise ValueError(f"Invalid op '{op}': arg operations (argmax, argmin, argmax3, etc.) cannot be used alone. They are only valid as outer operations in compound form like 'argmax_mean'.")

                if len(op_parts) > 1:
                    outer, inner = op_parts[0], op_parts[1]
                    # Disallow mid as outer op
                    if outer == 'mid':
                        raise ValueError(f"Invalid composite op '{op}': 'mid' cannot be used as an outer operation. It is only valid as an inner op (standalone 'mid').")
                    # Disallow top-k, arg-top-k, argmax/argmin as inner ops
                    if topk_pattern.match(inner) or argtopk_pattern.match(inner):
                        raise ValueError(f"Invalid composite op '{op}': '{inner}' (top-k/arg-top-k) cannot be used as an inner operation.")
                    if inner in ('argmax', 'argmin'):
                        raise ValueError(f"Invalid composite op '{op}': '{inner}' cannot be used as an inner operation. arg operations are only valid as outer ops.")
                    # Disallow more than 2 levels for now
                    if len(op_parts) > 2:
                        raise ValueError(f"Invalid composite op '{op}': Only 2 levels of operations are supported.")

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
                print(f"Warning: Output directory {self.output_full_dir} already exists. Contents may be overwritten.")
        return self

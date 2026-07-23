# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import atexit
import hashlib
import linecache
import random
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import cftime
import numpy as np
import torch

from hydroforge.data.distributed import torch_to_numpy_dtype
from hydroforge.contracts import ResourceCleanupError
from hydroforge.statistics.ir import (
    ExpressionSource, ScatterSource, StorageDType, StorageInitialization,
    TensorSource, build_variable_storage_plan,
)
from hydroforge.statistics.compiler import StatisticsCompiler
from hydroforge.statistics.layout import StatisticsCompilation
from hydroforge.output.netcdf.writer import NetCDFWriter
from hydroforge.serialization.netcdf import (
    default_netcdf_options, normalize_netcdf_variable_options,
)
from hydroforge.serialization.files import atomic_write_text
from hydroforge.contracts.fields import RuntimeTensorMetadata
from hydroforge.contracts.naming import sanitize_symbol


@dataclass(frozen=True, slots=True)
class StatisticsConfig:
    """Complete immutable configuration for one statistics output sink."""

    device: torch.device
    backend: str
    output_dir: Path | None = None
    rank: int = 0
    world_size: int = 1
    num_workers: int = 4
    save_kernels: bool = False
    output_split_by_year: bool = False
    num_trials: int = 1
    max_pending_steps: int = 200
    calendar: str = "standard"
    time_unit: str = "days since 1900-01-01 00:00:00"
    in_memory: bool = False
    result_device: torch.device | None = None
    save_precision: torch.dtype | None = None
    output_netcdf_options: Mapping[str, Any] = field(
        default_factory=default_netcdf_options,
    )
    event_sink: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, torch.device):
            raise TypeError("statistics device must be a torch.device")
        if self.backend not in {"torch", "cuda", "triton", "metal"}:
            raise ValueError(f"unsupported statistics backend {self.backend!r}")
        for name in (
            "rank", "world_size", "num_workers", "num_trials",
            "max_pending_steps",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "rank" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"statistics {name} must be an exact int >= {minimum}"
                )
        if self.rank >= self.world_size:
            raise ValueError("statistics rank must be smaller than world_size")
        for name in ("save_kernels", "output_split_by_year", "in_memory"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"statistics {name} must be an exact bool")
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            raise TypeError("statistics output_dir must be a pathlib.Path or None")
        if not self.in_memory and self.output_dir is None:
            raise ValueError("statistics output_dir is required in streaming mode")
        if self.save_kernels and self.output_dir is None:
            raise ValueError("output_dir is required when save_kernels=True")
        if self.result_device is not None and not isinstance(
            self.result_device, torch.device,
        ):
            raise TypeError("statistics result_device must be a torch.device or None")
        if self.save_precision is not None and not isinstance(
            self.save_precision, torch.dtype,
        ):
            raise TypeError("statistics save_precision must be a torch.dtype or None")
        normalize_netcdf_variable_options(self.output_netcdf_options)
        if not isinstance(self.calendar, str) or not self.calendar:
            raise ValueError("statistics calendar must be a non-empty string")
        if not isinstance(self.time_unit, str) or not self.time_unit:
            raise ValueError("statistics time_unit must be a non-empty string")


class StatisticsRuntime:
    """
    Handles statistics aggregation with streaming NetCDF output to minimize memory usage.
    Each time step is immediately written to disk after accumulation.

    Supports two modes:
    1. Streaming mode (default): Write each time step to NetCDF files incrementally.
    2. In-memory mode: Store all time steps in memory (CPU by default) for small-scale analysis.
       Results are dynamically appended, no need to pre-specify total time steps.
    """

    def __init__(self, config: StatisticsConfig):
        """Initialize one statistics sink from its validated configuration.

        Per-saved-point static metadata (e.g. basin_id, per-catchment
        ``shift_days``) is registered after construction via
        :meth:`register_static`, mirroring the ``register_tensor`` /
        ``register_virtual_tensor`` pattern used for dynamic outputs.
        """
        self.device = config.device
        self.backend = config.backend
        from hydroforge.contracts.events import ConsoleEventSink

        self.event_sink = (
            ConsoleEventSink() if config.event_sink is None else config.event_sink
        )
        self.output_dir = config.output_dir
        self.rank = config.rank
        self.world_size = config.world_size
        self.num_workers = config.num_workers
        self.save_kernels = config.save_kernels
        self.output_split_by_year = config.output_split_by_year
        self.output_netcdf_options = normalize_netcdf_variable_options(
            config.output_netcdf_options,
        )
        self.num_trials = config.num_trials
        self._closed = False
        self.max_pending_steps = config.max_pending_steps
        self.calendar = config.calendar
        self.time_unit = config.time_unit
        self._current_year = None

        # In-memory mode settings
        self.in_memory_mode = config.in_memory
        self.result_device = (
            config.result_device
            if config.result_device is not None else torch.device("cpu")
        )
        self.save_precision = config.save_precision

        # Generated-kernel state (populated lazily by the codegen mixins).
        self._static_gather_function = None
        self.kernels_dir: Path | None = None

        # Create kernels directory if saving is enabled (must precede any
        # codegen step so the generated .py files have a destination).
        if self.save_kernels:
            self.kernels_dir = self.output_dir / "generated_kernels"
            self.kernels_dir.mkdir(parents=True, exist_ok=True)

        # Normalize static_vars.  Populated by :meth:`register_static`
        # on demand; the NetCDF writer consumes this dict at file
        # creation time.
        self.static_vars: Dict[str, Dict[str, Any]] = {}

        self._macro_step_index = 0  # Current macro step index (outer loop counter)

        # Time index tracking for argmax/argmin conversion
        # Maps macro step index -> datetime, populated during finalize_time_step
        self._macro_step_times: List[Union[datetime, cftime.datetime]] = []

        # Internal state
        # Generic stats state (for all ops)
        self._variables: Set[str] = set()  # original variable names
        self._variable_ops: Dict[str, List[str]] = {}  # var -> list[ops]
        self._storage: Dict[str, torch.Tensor] = {}  # out_name -> tensor
        self._output_keys: List[str] = [] # list of keys in storage that are outputs
        self._metadata: Dict[str, Dict[str, Any]] = {}  # out_name -> meta
        self._coord_cache: Dict[str, np.ndarray] = {}

        self._tensor_registry: Dict[str, torch.Tensor] = {}
        self._field_registry: Dict[str, RuntimeTensorMetadata] = {}

        # Cache for sanitized names
        self._safe_name_cache: Dict[str, str] = {}

        # Streaming mode support
        self._netcdf_files: Dict[str, Path] = {}  # out_name -> NetCDF file path

        self._all_created_files: Set[Path] = set()
        self._files_created: bool = False

        # Thread pool for background writing
        self._write_executors: List[ProcessPoolExecutor] = []
        self._pending_writes: List = []
        self._write_buffers: Dict[str, Dict[str, Any]] = {}

        # Kernel state (mean fast-path)
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states: Optional[Dict[str, torch.Tensor]] = None

        self._kernel_module = None
        self._generated_modules: list[tuple[str, str]] = []
        self._saved_kernel_file = None
        self._dirty_outputs: Set[str] = set()
        self._compiler = StatisticsCompiler(self)
        self._output = NetCDFWriter(self)

        # In-memory result tensors: out_name -> list of tensors (one per time step)
        # Only used when in_memory_mode=True
        self._result_tensors: Dict[str, List[torch.Tensor]] = {}
        self._current_time_index: int = 0  # Current time index for in-memory writing

        from hydroforge.contracts.events import emit

        emit(
            self, "info", "statistics.initialized",
            "Initialized streaming statistics",
            rank=self.rank, workers=self.num_workers,
        )
        if config.in_memory:
            emit(
                self, "info", "statistics.memory_mode",
                "Statistics results will be retained in memory",
                device=self.result_device,
            )
        if self.save_kernels:
            emit(
                self, "info", "statistics.kernel_output",
                "Generated statistics kernels will be saved",
                directory=self.kernels_dir,
            )
        atexit.register(self._shutdown)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("statistics runtime is closed")

    def _prepare_kernel_states(self) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        required_tensors: Dict[str, torch.Tensor] = {}
        ir = self._statistics_ir

        def tensor_dependencies(name: str) -> Set[str]:
            source = ir.sources.get(name, TensorSource(name))
            if isinstance(source, TensorSource):
                return {source.name} if source.name in self._tensor_registry else set()
            dependencies = (
                source.expression.dependencies
                if isinstance(source, ExpressionSource)
                else source.value.dependencies
            )
            result: Set[str] = set()
            if (
                isinstance(source, ScatterSource)
                and source.index in self._tensor_registry
            ):
                result.add(source.index)
            for dependency in dependencies:
                result.update(tensor_dependencies(dependency))
            return result

        # Add original variables and their output buffers
        for variable in ir.variables:
            var_name = variable.name
            for dependency in tensor_dependencies(var_name):
                required_tensors[dependency] = self._tensor_registry[dependency]

            for operation in variable.operations:
                op = operation.spelling
                out_name = f"{var_name}_{op}"
                required_tensors[out_name] = self._storage[out_name]

                # For explicit argmax/argmin operations, add their auxiliary storage
                if operation.stores_index:
                    suffix = "" if operation.k == 1 else str(operation.k)
                    aux_name = f"{var_name}_{operation.outer.value}{suffix}_aux"
                    if aux_name in self._storage:
                        required_tensors[aux_name] = self._storage[aux_name]

                # Add inner states for compound ops
                if operation.inner is not None:
                    inner = operation.inner.value
                    # 'last' inner op doesn't need cross-step state
                    if inner != 'last':
                        inner_name = f"{var_name}_{inner}_inner_state"
                        if inner_name in self._storage:
                            required_tensors[inner_name] = self._storage[inner_name]
                        if inner == 'mean':
                            w_name = f"{var_name}_{inner}_weight_state"
                            if w_name in self._storage:
                                required_tensors[w_name] = self._storage[w_name]

        # Collect required dimensions and output indices.
        required_dims: Set[str] = set()
        required_output_indices: Set[str] = set()
        for variable in ir.variables:
            if variable.output_group != "__full__":
                required_output_indices.add(variable.output_group)
            for dim_name in variable.tensor_shape:
                if isinstance(dim_name, str):
                    required_dims.add(dim_name)

        # Add scatter buffers and their source/index tensors
        for variable in ir.variables:
            scatter = variable.source
            if not isinstance(scatter, ScatterSource):
                continue
            var_name = variable.name
            buf_key = f"__scatter_buf_{var_name}"
            if buf_key in self._storage:
                required_tensors[buf_key] = self._storage[buf_key]
            if scatter.reduction.value == 'mean':
                cnt_key = f"__scatter_cnt_{var_name}"
                if cnt_key in self._storage:
                    required_tensors[cnt_key] = self._storage[cnt_key]
            # Ensure all scatter source tensors and index are in required_tensors
            if scatter.index in self._tensor_registry:
                required_tensors[scatter.index] = self._tensor_registry[scatter.index]
            for dependency in tensor_dependencies(var_name):
                required_tensors[dependency] = self._tensor_registry[dependency]

        # Add output_index tensors
        for output_index in required_output_indices:
            if output_index in self._tensor_registry:
                required_tensors[output_index] = self._tensor_registry[output_index]
            else:
                raise RuntimeError(f"Output index tensor '{output_index}' not registered")

        # Add dimension tensors/scalars
        for dim_name in required_dims:
            if dim_name in self._tensor_registry:
                tensor = self._tensor_registry[dim_name]
                if isinstance(tensor, (int, float)):
                    required_tensors[dim_name] = torch.tensor(tensor, device=self.device)
                else:
                    required_tensors[dim_name] = tensor

        from hydroforge.kernels.registry import devices_match
        mismatched = {
            name: str(tensor.device)
            for name, tensor in required_tensors.items()
            if not devices_match(tensor.device, self.device)
        }
        if mismatched:
            raise ValueError(
                f"Statistics kernel tensors must be on {self.device}: {mismatched}"
            )

        # Scalar parameters as 1-element device tensors for CUDA Graph compatibility.
        # Kernel code loads these via tl.load (Triton) or reads from states dict,
        # so CUDA Graphs can replay without recapture when values change.
        required_tensors['__weight'] = torch.zeros(
            1, device=self.device, dtype=torch.float32,
        )
        required_tensors['__total_weight'] = torch.zeros(
            1, device=self.device, dtype=torch.float32,
        )
        required_tensors['__num_macro_steps'] = torch.zeros(
            1, device=self.device, dtype=torch.float32,
        )
        required_tensors['__sub_step'] = torch.zeros(
            1, device=self.device, dtype=torch.int32,
        )
        required_tensors['__num_sub_steps'] = torch.zeros(
            1, device=self.device, dtype=torch.int32,
        )
        required_tensors['__flags'] = torch.zeros(
            1, device=self.device, dtype=torch.int32,
        )
        required_tensors['__macro_step_index'] = torch.zeros(
            1, device=self.device, dtype=torch.int32,
        )
        # Publish only after dependency resolution, device checks, and every
        # allocation succeeded.  Rebinding may never expose partial states.
        self._kernel_states = required_tensors

    def initialize_statistics(self, compilation: StatisticsCompilation) -> None:
        """Initialize aggregation tensors and metadata for provided variables and ops."""
        self._require_open()
        if not isinstance(compilation, StatisticsCompilation):
            raise TypeError(
                "statistics runtime requires a StatisticsCompilation; "
                "raw variable mappings must be compiled first"
            )
        # Reset generic state
        self._variables = set()
        self._variable_ops = {
            name: list(operations)
            for name, operations in compilation.variable_ops.items()
        }
        self._statistics_program = compilation.program
        self._statistics_layouts = compilation.layouts
        self._storage.clear()
        self._output_keys = []
        self._metadata.clear()
        self._output_is_outer: Dict[str, bool] = {}

        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states = None
        self._current_macro_step_count = 0.0
        self._outer_flags_ever_seen = False

        # Release a previous generated specialization before rebuilding.
        self._cleanup_generated_modules()

        # Validate and setup each variable
        for var_name, ops in self._variable_ops.items():
            operation_nodes = self._statistics_program.operations[var_name]
            source = self._statistics_program.sources.get(
                var_name, TensorSource(var_name),
            )

            field_info = self._field_registry[var_name]
            metadata = field_info.tensor
            layout = self._statistics_layouts[var_name]
            tensor_shape = metadata.shape
            output_index = field_info.output_index
            description = field_info.description
            output_coord = field_info.output_coord
            dim_coords = metadata.dim_coords
            target_dtype = layout.dtype
            full_output = output_index is None
            actual_shape = layout.actual_shape
            actual_ndim = layout.actual_ndim

            # Track
            self._variables.add(var_name)

            # Detect scatter virtual and allocate materialized buffer
            if isinstance(source, ScatterSource):
                full_target_size = layout.scatter_extent
                if full_target_size is None:
                    raise RuntimeError(
                        f"Scatter layout for {var_name!r} has no target extent"
                    )
                scatter_buf_key = f"__scatter_buf_{var_name}"
                buf_shape = (
                    (self.num_trials, full_target_size)
                    if self.num_trials > 1 else (full_target_size,)
                )
                self._storage[scatter_buf_key] = torch.zeros(
                    buf_shape, dtype=target_dtype, device=self.device
                )
                if source.reduction.value == 'mean':
                    scatter_cnt_key = f"__scatter_cnt_{var_name}"
                    self._storage[scatter_cnt_key] = torch.zeros(
                        buf_shape, dtype=target_dtype, device=self.device
                    )

            storage_plan = build_variable_storage_plan(
                var_name, tuple(actual_shape), operation_nodes,
            )
            for slot in storage_plan.slots:
                dtype = (
                    torch.int32 if slot.dtype is StorageDType.INDEX
                    else target_dtype
                )
                if slot.initialization is StorageInitialization.NEGATIVE_INFINITY:
                    tensor = torch.full(
                        slot.shape, -torch.inf, dtype=dtype, device=self.device,
                    )
                elif slot.initialization is StorageInitialization.POSITIVE_INFINITY:
                    tensor = torch.full(
                        slot.shape, torch.inf, dtype=dtype, device=self.device,
                    )
                else:
                    tensor = torch.zeros(
                        slot.shape, dtype=dtype, device=self.device,
                    )
                self._storage[slot.name] = tensor
                if slot.output:
                    self._output_keys.append(slot.name)

            for operation in operation_nodes:
                op = operation.spelling
                out_name = f"{var_name}_{op}"

                if output_coord and output_coord not in self._coord_cache:
                    coord_tensor = self._tensor_registry[output_coord]
                    self._coord_cache[output_coord] = coord_tensor.detach().cpu().numpy()

                # Downcast to save_precision if specified (e.g. float64 -> float32)
                save_dtype = target_dtype
                if self.save_precision is not None and target_dtype.is_floating_point:
                    save_dtype = self.save_precision
                out_dtype = torch_to_numpy_dtype(save_dtype)

                is_arg_op = operation.stores_index

                # Determine stride_input and scatter metadata
                scatter_info = None
                if isinstance(source, ScatterSource):
                    scatter_buf = self._storage[f"__scatter_buf_{var_name}"]
                    stride_input = (
                        scatter_buf.shape[-1] if self.num_trials > 1 else 0
                    )
                    scatter_info = {
                        'mode': source.reduction.value,
                        'value_expr': source.value.source,
                        'index_var': source.index,
                        'source_size': layout.scatter_source_size,
                    }
                else:
                    stride_input = layout.stride_input

                meta = {
                    'original_variable': var_name,
                    'op': op,
                    'output_index': output_index,
                    'full_output': full_output,
                    'tensor_shape': tensor_shape,
                    'dtype': 'i4' if is_arg_op else out_dtype,  # int32 for arg ops
                    'actual_shape': actual_shape,
                    'actual_ndim': actual_ndim,
                    'output_coord': output_coord,
                    'nc_coord_name': dim_coords.split('.')[-1] if dim_coords else None,
                    'description': f"{description} ({op})",
                    'stride_input': stride_input,
                    'k': operation.k,
                    'is_time_index': is_arg_op,  # argmax/argmin store integer indices
                    'scatter': scatter_info,  # None for non-scatter, dict for scatter virtuals
                }
                self._metadata[out_name] = meta

                # Classify as outer if it is a compound op (e.g. max_mean)
                self._output_is_outer[out_name] = operation.compound

        self._has_compound_ops = any(self._output_is_outer.values())

        from hydroforge.contracts.naming import sanitize_symbol
        safe_outputs: Dict[str, str] = {}
        for out_name in self._output_keys:
            safe_name = sanitize_symbol(out_name)
            if not safe_name:
                raise ValueError(
                    f"Output name '{out_name}' has no valid NetCDF characters."
                )
            previous = safe_outputs.get(safe_name)
            if previous is not None and previous != out_name:
                raise ValueError(
                    f"Output names '{previous}' and '{out_name}' both map to "
                    f"NetCDF variable '{safe_name}'."
                )
            safe_outputs[safe_name] = out_name

        # Generate kernels and prepare states for all requested variables/ops
        self._compiler.compile()
        self._prepare_kernel_states()

    def update_statistics(self, sub_step: int, num_sub_steps: int, flags: int,
                          weight: float, total_weight: float = 0.0) -> None:
        self._require_open()
        if not self._aggregator_generated:
            raise RuntimeError("Statistics aggregation has not been initialized")

        # Compute boolean flags from sub_step, num_sub_steps, flags
        # flags bits: 0=stat_is_first, 1=stat_is_last, 2=stat_is_outer_first, 3=stat_is_outer_last
        is_inner_last = bool(flags & 2) and (sub_step == num_sub_steps - 1)
        is_outer_first = bool(flags & 4) and is_inner_last
        is_outer_last = bool(flags & 8) and is_inner_last

        # Reset macro_step_index at the start of each outer statistics period
        # This ensures argmax/argmin indices are always relative to the start of the period
        if is_outer_first:
            self._macro_step_index = 0
            self._current_macro_step_count = 0.0
            self._outer_flags_ever_seen = True

        if is_inner_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We only trigger dirty for non-outer (Standard) ops when inner loop ends
                 if not is_outer:
                     self._dirty_outputs.add(out_name)

        if is_outer_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We trigger dirty for outer ops when outer loop ends
                 if is_outer:
                     self._dirty_outputs.add(out_name)

        if is_inner_last:
            self._current_macro_step_count += 1.0

        num_macro_steps = self._current_macro_step_count

        # Fill scalar tensors so kernels read updated values from fixed addresses
        states = self._kernel_states
        states['__weight'].fill_(weight)
        states['__total_weight'].fill_(total_weight)
        states['__num_macro_steps'].fill_(num_macro_steps)
        states['__sub_step'].fill_(sub_step)
        states['__num_sub_steps'].fill_(num_sub_steps)
        states['__flags'].fill_(flags)
        states['__macro_step_index'].fill_(self._macro_step_index)

        self._execute_statistics_kernel()

    def _execute_statistics_kernel(self) -> None:
        """Run the generated aggregator through its cached backend executor."""
        if self._execution is None:
            raise RuntimeError(
                "statistics execution runtime is not attached to a model"
            )
        self._execution.run_statistics(self, self._execution.model.BLOCK_SIZE)

    def _init_result_storage(self) -> None:
        if not self.in_memory_mode:
            return
        self._result_tensors.clear()
        self._current_time_index = 0
        for out_name in self._output_keys:
            self._result_tensors[out_name] = []

    def _result_dtype(self, out_name: str) -> torch.dtype:
        """Return the exact retained-output dtype for one storage slot."""

        try:
            dtype = self._storage[out_name].dtype
        except KeyError as exc:
            raise KeyError(f"No statistics storage exists for {out_name!r}") from exc
        if self.save_precision is not None and dtype.is_floating_point:
            return self.save_precision
        return dtype

    def _empty_result(self, out_name: str) -> torch.Tensor:
        storage = self._storage[out_name]
        return torch.empty(
            (0, *storage.shape),
            dtype=self._result_dtype(out_name),
            device=self.result_device,
        )

    def get_results(self, as_stacked: bool = True):
        self._require_open()
        if not self.in_memory_mode:
            raise RuntimeError("get_results() is only available in in_memory_mode")
        if not as_stacked:
            return {
                name: [value.clone(memory_format=torch.preserve_format) for value in values]
                for name, values in self._result_tensors.items()
            }
        return {
            name: (
                torch.stack(values, dim=0) if values
                else self._empty_result(name)
            )
            for name, values in self._result_tensors.items()
        }

    def get_result(
        self, variable_name: str, op: str = "mean", as_stacked: bool = True,
    ):
        self._require_open()
        if not self.in_memory_mode:
            raise RuntimeError("get_result() is only available in in_memory_mode")
        out_name = f"{variable_name}_{op}"
        if out_name not in self._result_tensors:
            raise KeyError(
                f"No result found for {out_name}. Available: "
                f"{list(self._result_tensors)}"
            )
        values = self._result_tensors[out_name]
        if not as_stacked:
            return [
                value.clone(memory_format=torch.preserve_format)
                for value in values
            ]
        return (
            torch.stack(values, dim=0) if values
            else self._empty_result(out_name)
        )

    def get_time_index(self) -> int:
        self._require_open()
        return self._current_time_index

    def reset_time_index(self) -> None:
        self._require_open()
        if not self.in_memory_mode:
            raise RuntimeError(
                "reset_time_index() is only available in in_memory_mode"
            )
        self._current_time_index = 0
        self._macro_step_times.clear()
        for out_name in self._result_tensors:
            self._result_tensors[out_name] = []

    # Output service
    def _create_netcdf_files(self, year: int | None = None) -> None:
        self._output._create_netcdf_files(year)

    def _flush_write_buffer(self, key: str) -> None:
        self._output._flush_write_buffer(key)

    def _flush_all_write_buffers(self) -> None:
        self._output._flush_all_write_buffers()

    def _buffer_and_maybe_flush(
        self, buf_key: str, var_name: str, data: np.ndarray, output_path,
        dt, batch_size: int,
    ) -> None:
        self._output._buffer_and_maybe_flush(
            buf_key, var_name, data, output_path, dt, batch_size,
        )

    def _finalize_time_step_in_memory(self, dt) -> None:
        self._output._finalize_time_step_in_memory(dt)

    def finalize_time_step(self, dt) -> None:
        self._require_open()
        self._output.finalize_time_step(dt)

    def check_background_failures(self, current_time=None) -> None:
        """Raise completed asynchronous output failures without waiting."""

        self._require_open()
        if not self.in_memory_mode:
            self._output.check_completed_writes(dt=current_time)

    def ensure_output_durable(self, current_time=None) -> None:
        """Flush and wait for every streaming row at a checkpoint boundary."""

        self._require_open()
        if not self.in_memory_mode:
            self._output.flush_and_wait(dt=current_time)

    def _cleanup_generated_modules(self) -> None:
        for module_name, filename in reversed(self._generated_modules):
            sys.modules.pop(module_name, None)
            linecache.cache.pop(filename, None)
        self._generated_modules.clear()
        self._kernel_module = None

    def _cleanup_lock_files(self) -> None:
        for output_path in self._all_created_files:
            lock_path = output_path.with_suffix(output_path.suffix + ".lock")
            lock_path.unlink(missing_ok=True)

    def _cleanup_executor(self) -> None:
        failures: list[BaseException] = []
        try:
            self._flush_all_write_buffers()
        except BaseException as error:
            failures.append(error)
        pending, self._pending_writes = self._pending_writes, []
        for item in pending:
            try:
                item.future.result()
            except BaseException as error:
                failures.append(error)
        executors, self._write_executors = self._write_executors, []
        for executor in executors:
            try:
                executor.shutdown(wait=True)
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("statistics output workers", failures)

    def _start_write_executors(self) -> None:
        """Create the background output process pools."""

        if self._write_executors:
            raise RuntimeError("statistics output workers are already started")
        created = []
        try:
            for _ in range(self.num_workers):
                executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=get_context("spawn"),
                )
                created.append(executor)
        except BaseException as primary:
            failures: list[BaseException] = [primary]
            for executor in reversed(created):
                try:
                    executor.shutdown(wait=True)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            if len(failures) > 1:
                error = ResourceCleanupError(
                    "statistics output worker startup", failures,
                )
                raise error from primary
            raise
        self._write_executors = created

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        for cleanup in (
            lambda: atexit.unregister(self._shutdown),
            self._cleanup_generated_modules,
            self._cleanup_executor,
            self._cleanup_lock_files,
        ):
            try:
                cleanup()
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("statistics runtime", failures)

    def get_memory_usage(self) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in self._storage.values():
            if isinstance(tensor, torch.Tensor) and tensor.data_ptr() not in seen:
                seen.add(tensor.data_ptr())
                total += tensor.element_size() * tensor.numel()
        return total

    def get_result_memory_usage(self) -> int:
        if not self.in_memory_mode:
            return 0
        return sum(
            tensor.element_size() * tensor.numel()
            for values in self._result_tensors.values()
            for tensor in values
            if isinstance(tensor, torch.Tensor)
        )

    def _get_safe_name(self, name: str) -> str:
        if name not in self._safe_name_cache:
            self._safe_name_cache[name] = sanitize_symbol(name)
        return self._safe_name_cache[name]

    def _generate_unique_name(self) -> str:
        timestamp = datetime.now().strftime("%H%M%S")
        seed = f"{self.rank}_{timestamp}_{random.randint(1000, 9999)}"
        digest = hashlib.md5(seed.encode()).hexdigest()[:6]
        return f"{timestamp}_r{self.rank}_{digest}"

    def _generate_static_gather_function(self) -> None:
        """Bind the one-shot static gather used only during initialization."""
        def gather_static_var(
            tensor: torch.Tensor,
            output_index: torch.Tensor | None,
        ) -> torch.Tensor:
            return tensor if output_index is None else tensor[output_index]

        self._static_gather_function = gather_static_var
        if self.save_kernels:
            path = self.kernels_dir / f"kern_static_{self._generate_unique_name()}.py"
            atomic_write_text(
                path,
                "def gather_static_var(tensor, output_index):\n"
                "    return tensor if output_index is None else tensor[output_index]\n",
            )

    def __del__(self) -> None:
        try:
            self._shutdown()
        except Exception:
            pass

    def register_tensor(
        self, name: str, tensor: torch.Tensor,
        field_info: RuntimeTensorMetadata,
    ) -> None:
        """
        Register a tensor with its metadata for potential aggregation.

        Args:
            name: Variable name
            tensor: PyTorch tensor (actual sampled data)
            field_info: Pydantic field information
        """
        self._register_tensor(
            name, tensor, field_info, require_execution_device=True,
        )

    def register_output_coordinate(
        self, name: str, tensor: torch.Tensor,
    ) -> None:
        """Register an output-axis value tensor outside the execution ABI.

        Coordinate and selection fields intentionally remain on the CPU. They
        are copied once into the NetCDF schema and never passed to a statistics
        kernel; requiring the model execution device would contradict their
        field contract.
        """

        self._register_tensor(
            name, tensor, {}, require_execution_device=False,
        )

    def _register_tensor(
        self, name: str, tensor: torch.Tensor,
        field_info: RuntimeTensorMetadata | dict[str, Any], *,
        require_execution_device: bool,
    ) -> None:
        self._require_open()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {name}, got {type(tensor)}")
        from hydroforge.kernels.registry import devices_match

        if (
            require_execution_device
            and not devices_match(tensor.device, self.device)
        ):
            raise ValueError(
                f"Statistics tensor {name!r} is on {tensor.device}, "
                f"expected {self.device}"
            )
        if tensor.layout is not torch.strided:
            raise ValueError(
                f"Statistics tensor {name!r} must use torch.strided layout, "
                f"got {tensor.layout}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"Statistics tensor {name!r} must be contiguous; generated "
                "backends use one canonical linear buffer ABI"
            )

        self._tensor_registry[name] = tensor
        self._field_registry[name] = field_info

        # Pre-cache safe name
        self._get_safe_name(name)

        # Invalidate pre-computed states when new tensors are registered
        self._kernel_states = None


    def register_virtual_tensor(
        self, name: str, field_info: RuntimeTensorMetadata,
    ) -> None:
        """
        Register a virtual tensor (no data, just metadata).

        Args:
            name: Variable name
            field_info: Pydantic field information (must contain expr)
        """
        self._require_open()
        self._field_registry[name] = field_info
        self._get_safe_name(name)
        # Do NOT add to _tensor_registry since it has no storage
        self._kernel_states = None


    def register_static(self, name: str, tensor: torch.Tensor,
                        output_index: Optional[torch.Tensor] = None,
                        dim: str = "saved_points",
                        coordinate: Optional[str] = None,
                        dtype: Optional[str] = None,
                        attrs: Optional[Dict[str, Any]] = None) -> None:
        """Register a per-saved-point static variable.

        The raw ``tensor`` (optionally gathered by ``output_index``) is
        materialised once via the generated gather kernel and stashed
        for the NetCDF writer, which emits it along ``dim`` at file
        creation time.  Mirrors :meth:`register_tensor` /
        :meth:`register_virtual_tensor` so all aggregator inputs flow
        through the same register_* surface.
        """
        self._require_open()
        if not isinstance(name, str) or not name:
            raise ValueError("static variable name must be a non-empty string")
        if name in self.static_vars:
            raise ValueError(f"static variable {name!r} is already registered")
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
            raise TypeError("static variable tensor must be a one-dimensional tensor")
        if not isinstance(dim, str) or not dim:
            raise ValueError("static variable dim must be a non-empty string")
        if not isinstance(coordinate, str) or not coordinate:
            raise ValueError(
                "static variable coordinate must be a non-empty string"
            )
        if attrs is not None and not isinstance(attrs, Mapping):
            raise TypeError("static variable attrs must be a mapping or None")
        if dtype is not None:
            try:
                normalized_dtype = np.dtype(dtype).str.lstrip("<>|")
            except TypeError as error:
                raise TypeError(
                    f"static variable dtype {dtype!r} is not a NumPy dtype"
                ) from error
        else:
            normalized_dtype = None
        if self._static_gather_function is None:
            self._generate_static_gather_function()
        if output_index is not None:
            if not isinstance(output_index, torch.Tensor):
                raise TypeError("static output_index must be a tensor or None")
            if output_index.device != tensor.device:
                raise ValueError(
                    "static output_index and tensor must be on the same device"
                )
            if output_index.ndim != 1 or output_index.dtype not in {
                torch.int32, torch.int64,
            }:
                raise TypeError(
                    "static output_index must be a one-dimensional int32/int64 tensor"
                )
            if output_index.numel() and bool((
                (output_index < 0) | (output_index >= tensor.numel())
            ).any()):
                raise IndexError("static output_index is outside the tensor extent")
        values = (
            self._static_gather_function(tensor, output_index)
            .detach().cpu().numpy()
        )
        self.static_vars[name] = {
            "values": values,
            "dim": dim,
            "coordinate": coordinate,
            "dtype": (
                normalized_dtype
                if normalized_dtype is not None
                else values.dtype.str.lstrip("<>|")
            ),
            "attrs": dict(attrs or {}),
        }


    def initialize(self, compilation: StatisticsCompilation) -> None:
        """
        Initialize streaming aggregation for specified variables.
        Creates NetCDF file structure but writes time steps incrementally.

        Args:
            compilation: Compiler-owned operations, expressions and layouts.
        """
        self._require_open()
        if not isinstance(compilation, StatisticsCompilation):
            raise TypeError(
                "statistics runtime requires a StatisticsCompilation; "
                "raw variable mappings must be compiled first"
            )
        from hydroforge.contracts.events import emit

        emit(
            self, "info", "statistics.variables",
            "Configured statistics variables",
            variables=dict(compilation.variable_ops),
        )

        # Enable streaming mode
        self._files_created = False

        # Initialize single time step aggregation (generic)
        self.initialize_statistics(compilation)

        # If in-memory mode, initialize result storage lists instead of starting file writers
        if self.in_memory_mode:
            self._init_result_storage()
            emit(
                self, "info", "statistics.memory_ready",
                "In-memory statistics aggregation initialized",
                outputs=len(self._result_tensors),
            )
        else:
            # Start the write executors (one per worker to guarantee serialization per variable)
            self._start_write_executors()
            self._pending_writes = []
            emit(
                self, "info", "statistics.streaming_ready",
                "Streaming statistics aggregation initialized",
                executors=len(self._write_executors),
            )

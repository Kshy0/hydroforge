# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import hashlib
import linecache
import math
import random
import sys
import weakref
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Set

import numpy as np
import torch
from hydroforge.contracts.runtime import DEFAULT_BLOCK_SIZE
from hydroforge.contracts.events import ConsoleEventSink, emit

from hydroforge.data.distributed import torch_to_numpy_dtype
from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.statistics.ir import (
    ExpressionSource, Reduction, ScatterSource, StorageDType,
    StatisticsProgram, StorageInitialization, TensorSource,
    build_variable_storage_plan,
)
from hydroforge.statistics.compiler import StatisticsCompiler
from hydroforge.statistics.layout import StatisticsCompilation, compile_statistics
from hydroforge.output.netcdf.writer import (
    _NetCDFOutputStream,
    _NetCDFWriteBuffer,
    _NetCDFWriter,
    _close_worker_netcdf_files,
    _initialize_netcdf_worker,
    compute_write_batch_size,
    constrain_write_batch_sizes,
)
from hydroforge.serialization.netcdf import (
    default_netcdf_options,
)
from hydroforge.serialization.files import atomic_write_text
from hydroforge.contracts.fields import RuntimeTensorMetadata
from hydroforge.contracts.naming import sanitize_symbol


def _weak_shutdown_callback(runtime: Any):
    """Return an atexit callback that does not keep ``runtime`` alive."""

    runtime_ref = weakref.ref(runtime)

    def shutdown() -> None:
        instance = runtime_ref()
        if instance is not None:
            instance._shutdown()

    return shutdown


@dataclass(frozen=True, slots=True)
class StatisticsStaticBinding:
    """One compiler-resolved static output owned by an installation."""

    name: str
    tensor: torch.Tensor
    output_index: torch.Tensor | None
    coordinate: str
    dim: str = "saved_points"


@dataclass(frozen=True, slots=True)
class StatisticsInstallation:
    """Complete trusted compiler output installed into a runtime once."""

    variable_ops: Mapping[str, tuple[str, ...]]
    program: StatisticsProgram
    tensors: Mapping[str, torch.Tensor]
    fields: Mapping[str, RuntimeTensorMetadata]
    statics: tuple[StatisticsStaticBinding, ...]
    netcdf_options: Mapping[str, Mapping[str, Any]]


@dataclass
class StatisticsRuntime:
    """Trusted execution state built from a validated model declaration."""

    device: torch.device
    backend: Literal["torch", "cuda", "triton", "metal"]
    installation: StatisticsInstallation = field(repr=False)
    execution: Any = field(repr=False)
    base_dtype: torch.dtype = torch.float32
    mixed_precision: bool = False
    output_dir: Path | None = None
    rank: int = 0
    world_size: int = 1
    num_workers: int = 4
    save_kernels: bool = False
    output_split_by_year: bool = False
    num_trials: int = 1
    max_pending_steps: int = 200
    max_pending_output_bytes: int = 512 * 1024 * 1024
    block_size: int = DEFAULT_BLOCK_SIZE
    calendar: str = "standard"
    time_unit: str = "days since 1900-01-01 00:00:00"
    in_memory: bool = False
    result_device: torch.device = field(
        default_factory=lambda: torch.device("cpu"),
    )
    save_precision: torch.dtype | None = None
    output_netcdf_options: Mapping[str, Any] = field(
        default_factory=default_netcdf_options,
    )
    event_sink: Any = field(default_factory=ConsoleEventSink)
    run_id: str | None = None

    _kernels_dir: Path | None = field(init=False, default=None, repr=False)
    _static_vars: Dict[str, Dict[str, Any]] = field(
        init=False, default_factory=dict, repr=False,
    )

    @property
    def in_memory_mode(self) -> bool:
        return self.in_memory

    @property
    def kernels_dir(self) -> Path | None:
        return self._kernels_dir

    @property
    def static_vars(self) -> Dict[str, Dict[str, Any]]:
        return self._static_vars

    def __post_init__(self) -> None:
        self._closed = False
        self._current_year = None

        # Create kernels directory if saving is enabled (must precede any
        # codegen step so the generated .py files have a destination).
        if self.save_kernels:
            self._kernels_dir = self.output_dir / "generated_kernels"
            self._kernels_dir.mkdir(parents=True, exist_ok=True)

        self._macro_step_index = 0  # Current macro step index (outer loop counter)
        self._macro_mean_count_limit: int | None = None

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
        self._structural_tensor_versions: dict[
            str, tuple[torch.Tensor, int]
        ] = {}

        # Cache for sanitized names
        self._safe_name_cache: Dict[str, str] = {}

        # Streaming mode support
        # Compatibility view for callers that inspect the created paths.
        self._netcdf_files: Dict[str, Path | list[Path]] = {}
        self._output_streams: dict[
            str, tuple[_NetCDFOutputStream, ...]
        ] = {}

        self._all_created_files: Set[Path] = set()
        self._files_created: bool = False

        # Thread pool for background writing
        self._write_executors: List[ProcessPoolExecutor] = []
        self._pending_writes: List = []
        self._write_buffers: Dict[str, _NetCDFWriteBuffer] = {}

        # Kernel state (mean fast-path)
        self._kernel_module = None
        self._generated_modules: list[tuple[str, str]] = []
        self._saved_kernel_file = None
        self._dirty_outputs: Set[str] = set()
        self._compiler = StatisticsCompiler(self)
        self._output = _NetCDFWriter(self)

        # In-memory result tensors: out_name -> list of tensors (one per time step)
        # Only used when in_memory_mode=True
        self._result_tensors: Dict[str, List[torch.Tensor]] = {}
        self._current_time_index: int = 0

        emit(
            self, "info", "statistics.initialized",
            "Initialized streaming statistics",
            rank=self.rank, workers=self.num_workers,
        )
        if self.in_memory:
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
        self._atexit_callback = _weak_shutdown_callback(self)
        atexit.register(self._atexit_callback)
        self._materialize_installation(self.installation)

    def _prepare_kernel_states(self) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        required_tensors: Dict[str, torch.Tensor] = {}
        ir = self._statistics_ir

        def tensor_dependencies(name: str) -> Set[str]:
            source = ir.sources.get(name, TensorSource(name))
            if isinstance(source, TensorSource):
                return {source.name}
            dependencies = (
                source.expression.dependencies
                if isinstance(source, ExpressionSource)
                else source.value.dependencies
            )
            result: Set[str] = set()
            if isinstance(source, ScatterSource):
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
                    aux_name = f"{var_name}_{operation.spelling}_aux"
                    required_tensors[aux_name] = self._storage[aux_name]

                if (
                    operation.inner is None
                    and operation.outer is Reduction.MEAN
                ):
                    weight_name = (
                        f"{var_name}_mean_sample_weight_state"
                    )
                    required_tensors[weight_name] = self._storage[weight_name]

                # Add inner states for compound ops
                if operation.inner is not None:
                    inner = operation.inner.value
                    # 'last' inner op doesn't need cross-step state
                    if inner != 'last':
                        inner_name = f"{var_name}_{inner}_inner_state"
                        required_tensors[inner_name] = self._storage[inner_name]
                        if inner == 'mean':
                            w_name = f"{var_name}_{inner}_weight_state"
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

        # Include scatter buffers from hidden virtual dependencies.
        for variable in ir.ordered_scatters():
            scatter = variable.source
            var_name = variable.name
            buf_key = f"__scatter_buf_{var_name}"
            required_tensors[buf_key] = self._storage[buf_key]
            if scatter.reduction.value == 'mean':
                cnt_key = f"__scatter_cnt_{var_name}"
                required_tensors[cnt_key] = self._storage[cnt_key]
            # Ensure all scatter source tensors and index are in required_tensors
            required_tensors[scatter.index] = self._tensor_registry[scatter.index]
            for dependency in tensor_dependencies(var_name):
                required_tensors[dependency] = self._tensor_registry[dependency]

        # Add output_index tensors
        for output_index in required_output_indices:
            required_tensors[output_index] = self._tensor_registry[output_index]

        # Add dimension tensors/scalars
        for dim_name in required_dims:
            if dim_name in self._tensor_registry:
                required_tensors[dim_name] = self._tensor_registry[dim_name]

        # Scalar parameters as 1-element device tensors for CUDA Graph compatibility.
        # Kernel code loads these via tl.load (Triton) or reads from states dict,
        # so CUDA Graphs can replay without recapture when values change.
        control_dtype = self._statistics_control_dtype()
        required_tensors['__weight'] = torch.zeros(
            1, device=self.device, dtype=control_dtype,
        )
        required_tensors['__total_weight'] = torch.zeros(
            1, device=self.device, dtype=control_dtype,
        )
        required_tensors['__num_macro_steps'] = torch.zeros(
            1, device=self.device, dtype=torch.int64,
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
            1, device=self.device, dtype=torch.int64,
        )
        # Publish only after dependency resolution, device checks, and every
        # allocation succeeded.  Rebinding may never expose partial states.
        self._kernel_states = required_tensors

    def _statistics_control_dtype(self) -> torch.dtype:
        """Return the precision shared by aggregation control scalars."""

        if self.device.type == "mps":
            return torch.float32
        if any(
            tensor.dtype == torch.float64
            for tensor in self._storage.values()
        ):
            return torch.float64
        return torch.float32

    def _materialize_compilation(
        self, compilation: StatisticsCompilation,
    ) -> None:
        """Materialize one trusted compiler-owned statistics program."""
        self._variable_ops = {
            name: list(operations)
            for name, operations in compilation.variable_ops.items()
        }
        self._statistics_program = compilation.program
        self._statistics_layouts = compilation.layouts
        self._output_is_outer: Dict[str, bool] = {}

        self._structural_tensor_versions = {}
        self._current_macro_step_count = 0
        self._macro_step_index = 0
        mean_count_limits: set[int] = set()
        for name, operations in compilation.program.operations.items():
            if not any(
                operation.compound and operation.outer is Reduction.MEAN
                for operation in operations
            ):
                continue
            dtype = compilation.layouts[name].dtype
            if dtype == torch.float32:
                mean_count_limits.add(2**24)
            elif dtype == torch.float64:
                mean_count_limits.add(2**53)
        self._macro_mean_count_limit = (
            min(mean_count_limits) if mean_count_limits else None
        )

        for var_name, source in self._statistics_program.sources.items():
            if (
                not isinstance(source, ScatterSource)
                or var_name in self._variable_ops
            ):
                continue
            layout = self._statistics_layouts[var_name]
            full_target_size = layout.scatter_extent
            shape = (
                (self.num_trials, full_target_size)
                if self.num_trials > 1 else (full_target_size,)
            )
            self._storage[f"__scatter_buf_{var_name}"] = torch.zeros(
                shape, dtype=layout.dtype, device=self.device,
            )
            if source.reduction is Reduction.MEAN:
                self._storage[f"__scatter_cnt_{var_name}"] = torch.zeros(
                    shape, dtype=torch.int32, device=self.device,
                )

        for var_name in self._variable_ops:
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
                        buf_shape, dtype=torch.int32, device=self.device
                    )

            storage_plan = build_variable_storage_plan(
                var_name, tuple(actual_shape), operation_nodes,
            )
            for slot in storage_plan.slots:
                dtype = (
                    torch.int64 if slot.dtype is StorageDType.INDEX
                    else target_dtype
                )
                if slot.initialization is StorageInitialization.NEGATIVE_INFINITY:
                    initial = (
                        -torch.inf if dtype.is_floating_point
                        else False if dtype is torch.bool
                        else torch.iinfo(dtype).min
                    )
                    tensor = torch.full(
                        slot.shape, initial, dtype=dtype, device=self.device,
                    )
                elif slot.initialization is StorageInitialization.POSITIVE_INFINITY:
                    initial = (
                        torch.inf if dtype.is_floating_point
                        else True if dtype is torch.bool
                        else torch.iinfo(dtype).max
                    )
                    tensor = torch.full(
                        slot.shape, initial, dtype=dtype, device=self.device,
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
                    coordinate = np.array(
                        coord_tensor.detach().cpu().numpy(),
                        dtype=np.int64,
                        order="C",
                        copy=True,
                    )
                    coordinate.setflags(write=False)
                    self._coord_cache[output_coord] = coordinate

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
                    'dtype': 'i8' if is_arg_op else out_dtype,
                    'actual_shape': actual_shape,
                    'actual_ndim': actual_ndim,
                    'batched': layout.batched,
                    'output_coord': output_coord,
                    'nc_coord_name': dim_coords.split('.')[-1] if dim_coords else None,
                    'description': f"{description} ({op})",
                    'stride_input': stride_input,
                    'k': operation.k,
                    'scatter': scatter_info,  # None for non-scatter, dict for scatter virtuals
                }
                self._metadata[out_name] = meta

                # Classify as outer if it is a compound op (e.g. max_mean)
                self._output_is_outer[out_name] = operation.compound

        from hydroforge.output.netcdf.schema import NetCDFSchema

        desired_batches: dict[str, int] = {}
        row_bytes: dict[str, int] = {}
        stream_counts: dict[str, int] = {}
        for name, metadata in self._metadata.items():
            order = metadata["k"]
            row_shape = (
                metadata["actual_shape"][:-1]
                if order > 1 else metadata["actual_shape"]
            )
            storage_dtype = np.dtype(metadata["dtype"])
            row_bytes[name] = max(
                1, math.prod(row_shape) * storage_dtype.itemsize,
            )
            stream_counts[name] = order
            desired_batches[name] = compute_write_batch_size(
                max(1, math.prod(row_shape)),
                storage_dtype.itemsize,
                max_batch=min(30, self.max_pending_steps),
            )
        write_batches = constrain_write_batch_sizes(
            desired_batches,
            row_bytes=row_bytes,
            stream_counts=stream_counts,
            max_pending_bytes=self.max_pending_output_bytes,
        )
        self._netcdf_schemas = {
            name: NetCDFSchema.compile(
                metadata,
                variable=name,
                num_trials=self.num_trials,
                netcdf_options=self.installation.netcdf_options[name],
                write_batch_size=write_batches[name],
            )
            for name, metadata in self._metadata.items()
        }

        # Generate kernels and prepare states for all requested variables/ops
        self._compiler.compile()
        self._prepare_kernel_states()

    def _claim_macro_step(
        self, *,
        is_inner_last: bool,
        is_outer_first: bool,
        is_outer_last: bool,
    ) -> tuple[int, int]:
        macro_step_index = 0 if is_outer_first else self._macro_step_index
        macro_step_count = (
            0 if is_outer_first else self._current_macro_step_count
        )
        next_count = macro_step_count + int(is_inner_last)
        limit = torch.iinfo(torch.int64).max
        if (
            macro_step_index > limit
            or next_count > limit
            or (is_inner_last and macro_step_index == limit)
        ):
            raise OverflowError(
                "statistics macro-step accounting exceeds int64 range"
            )
        mean_limit = self._macro_mean_count_limit
        if is_inner_last and mean_limit is not None and next_count > mean_limit:
            raise OverflowError(
                "statistics compound mean macro-step count exceeds the "
                f"largest consecutive integer exactly representable by its "
                f"accumulator dtype ({mean_limit})"
            )
        if is_outer_first:
            self._macro_step_index = 0
            self._current_macro_step_count = 0
        if is_inner_last:
            self._dirty_outputs.update(
                name for name, outer in self._output_is_outer.items()
                if not outer
            )
        if is_outer_last:
            self._dirty_outputs.update(
                name for name, outer in self._output_is_outer.items()
                if outer
            )
        if is_inner_last:
            self._current_macro_step_count = next_count
            self._macro_step_index = macro_step_index + 1
        return self._current_macro_step_count, macro_step_index

    def _convert_control_float(self, name: str, value: float) -> float:
        """Convert a trusted schedule weight while preserving overflow errors."""

        states = self._kernel_states
        dtype = states[f"__{name}"].dtype
        converted = float(torch.tensor(value, dtype=dtype).item())
        if not math.isfinite(converted):
            raise OverflowError(
                f"statistics {name} {value!r} exceeds {dtype} range"
            )
        if converted == 0.0:
            raise OverflowError(
                f"statistics {name} {value!r} underflows {dtype}"
            )
        return converted

    def update_statistics(
        self,
        sub_step: int,
        num_sub_steps: int,
        flags: int,
        weight: float,
        total_weight: float,
    ) -> None:
        converted_weight = self._convert_control_float("weight", weight)
        converted_total = self._convert_control_float(
            "total_weight", total_weight,
        )

        is_inner_last = bool(flags & 2) and (sub_step == num_sub_steps - 1)
        is_outer_first = bool(flags & 4) and is_inner_last
        is_outer_last = bool(flags & 8) and is_inner_last
        num_macro_steps, macro_step_index = self._claim_macro_step(
            is_inner_last=is_inner_last,
            is_outer_first=is_outer_first,
            is_outer_last=is_outer_last,
        )

        # Fill scalar tensors so kernels read updated values from fixed addresses
        states = self._kernel_states
        states['__weight'].fill_(converted_weight)
        states['__total_weight'].fill_(converted_total)
        states['__num_macro_steps'].fill_(num_macro_steps)
        states['__sub_step'].fill_(sub_step)
        states['__num_sub_steps'].fill_(num_sub_steps)
        states['__flags'].fill_(flags)
        states['__macro_step_index'].fill_(macro_step_index)

        self._execute_statistics_kernel()

    def _execute_statistics_kernel(self) -> None:
        """Run the generated aggregator through its cached backend executor."""
        self.execution.run_statistics(self, self.block_size)

    def _init_result_storage(self) -> None:
        self._current_time_index = 0
        for out_name in self._output_keys:
            self._result_tensors[out_name] = []

    def _result_dtype(self, out_name: str) -> torch.dtype:
        """Return the exact retained-output dtype for one storage slot."""

        dtype = self._storage[out_name].dtype
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
        out_name = f"{variable_name}_{op}"
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
        return self._current_time_index

    def reset_time_index(self) -> None:
        self._current_time_index = 0
        self._output.reset_timeline()
        for out_name in self._result_tensors:
            self._result_tensors[out_name] = []

    def finalize_time_step(self, dt: Any) -> None:
        self._output.finalize_time_step(dt)

    def check_background_failures(self, current_time: Any = None) -> None:
        """Raise completed asynchronous output failures without waiting."""

        if not self.in_memory_mode:
            self._output.check_completed_writes(dt=current_time)

    def require_output_coordinate_resize_safe(
        self,
        coordinate: str,
        *,
        old_extent: int,
        new_extent: int,
        stable_scatter_index: str | None = None,
        stable_output_coordinate: str | None = None,
    ) -> None:
        """Reject a resize unless affected outputs keep a stable domain."""

        if old_extent == new_extent:
            return
        unsafe: list[str] = []
        program = self.installation.program
        for variable in self._variable_ops:
            touches_coordinate = any(
                (
                    field := self._field_registry.get(leaf)
                ) is not None
                and field.tensor.dim_coords is not None
                and field.tensor.dim_coords.split(".")[-1] == coordinate
                for leaf in program.leaf_tensors(variable)
            )
            if not touches_coordinate:
                continue
            source = program.sources.get(variable, TensorSource(variable))
            scatter_is_stable = (
                stable_scatter_index is not None
                and stable_output_coordinate is not None
                and isinstance(source, ScatterSource)
                and source.index == stable_scatter_index
                and (
                    output_field := self._field_registry.get(variable)
                ) is not None
                and output_field.tensor.dim_coords is not None
                and output_field.tensor.dim_coords.split(".")[-1]
                == stable_output_coordinate
            )
            if not scatter_is_stable:
                unsafe.append(variable)

        if unsafe:
            raise RuntimeError(
                f"cannot resize output coordinate {coordinate!r} from "
                f"{old_extent} to {new_extent}; affected outputs do not preserve "
                f"a supported stable aggregation domain: {sorted(unsafe)}. "
                "Contributor growth is supported only through the declared "
                f"{stable_scatter_index!r} aggregation onto "
                f"{stable_output_coordinate!r}"
            )

    def recompile_resized_sources(self) -> None:
        """Recompile source addressing after model tensor shapes change.

        Output domains and accumulator layouts must remain unchanged.  This
        cold path is intended for topology growth where contributor arrays
        gain rows while the saved coordinate domain is stable.
        """

        compilation = compile_statistics(
            self,
            self.installation.variable_ops,
            self.installation.program,
        )
        for name in self._variable_ops:
            previous = self._statistics_layouts[name]
            updated = compilation.layouts[name]
            if (
                previous.actual_shape != updated.actual_shape
                or previous.dtype != updated.dtype
                or previous.batched != updated.batched
                or previous.scatter_extent != updated.scatter_extent
            ):
                raise RuntimeError(
                    "statistics topology growth changed the saved layout for "
                    f"{name!r}: {previous!r} -> {updated!r}"
                )

        self._cleanup_generated_modules()
        self._statistics_layouts = compilation.layouts
        for name, metadata in self._metadata.items():
            scatter = metadata.get("scatter")
            if scatter is None:
                continue
            variable = metadata["original_variable"]
            scatter["source_size"] = compilation.layouts[
                variable
            ].scatter_source_size
        self._compiler.compile()
        self._prepare_kernel_states()
        self._structural_tensor_versions = {}
        self.execution.statistics.invalidate()

    def refresh_address_stable_sources(self) -> None:
        """Refresh compiler-owned index copies without replacing graph storage."""

        variable_map = self.execution.model._namespace.build()
        with torch.inference_mode():
            for name, installed in self._tensor_registry.items():
                entry = variable_map.get(name)
                if entry is None:
                    continue
                live = getattr(entry.module, entry.field_name)
                if not isinstance(live, torch.Tensor) or installed is live:
                    continue
                schema = entry.module._get_tensor_schema(entry.field_name)
                if schema.tensor.is_coordinate:
                    continue
                if (
                    installed.shape != live.shape
                    or installed.dtype != live.dtype
                    or installed.device != live.device
                ):
                    raise RuntimeError(
                        f"address-stable statistics source {name!r} changed "
                        "shape, dtype, or device"
                    )
                installed.copy_(live)

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
            self._output._flush_all_write_buffers()
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
                executor.submit(_close_worker_netcdf_files).result()
            except BaseException as error:
                failures.append(error)
            try:
                executor.shutdown(wait=True)
            except BaseException as error:
                failures.append(error)
        self._write_buffers.clear()
        self._output_streams.clear()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("statistics output workers", failures)

    def _start_write_executors(self) -> None:
        """Create the background output process pools."""

        created = []
        try:
            for _ in range(self.num_workers):
                executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=get_context("spawn"),
                    initializer=_initialize_netcdf_worker,
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

    def _unregister_atexit(self) -> None:
        callback = getattr(self, "_atexit_callback", None)
        if callback is not None:
            atexit.unregister(callback)
            self._atexit_callback = None

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        for cleanup in (
            self._unregister_atexit,
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
            if tensor.data_ptr() not in seen:
                seen.add(tensor.data_ptr())
                total += tensor.element_size() * tensor.numel()
        return total

    def _get_safe_name(self, name: str) -> str:
        if name not in self._safe_name_cache:
            self._safe_name_cache[name] = sanitize_symbol(name)
        return self._safe_name_cache[name]

    def _generate_unique_name(self) -> str:
        timestamp = datetime.now().strftime("%H%M%S")
        seed = f"{self.rank}_{timestamp}_{random.randint(1000, 9999)}"
        digest = hashlib.md5(seed.encode()).hexdigest()[:6]
        return f"{timestamp}_r{self.rank}_{digest}"

    def __del__(self) -> None:
        try:
            self._shutdown()
        except Exception:
            pass

    def _materialize_static(self, binding: StatisticsStaticBinding) -> None:
        tensor = (
            binding.tensor
            if binding.output_index is None
            else binding.tensor[binding.output_index]
        )
        values = tensor.detach().cpu().numpy()
        values = np.array(values, order="C", copy=True)
        values.setflags(write=False)
        self.static_vars[binding.name] = {
            "values": values,
            "dim": binding.dim,
            "coordinate": binding.coordinate,
            "dtype": values.dtype.str.lstrip("<>|"),
            "attrs": {},
        }

    def _materialize_installation(
        self, installation: StatisticsInstallation,
    ) -> None:
        """Materialize the complete compiler-owned registry during construction."""

        self._tensor_registry = dict(installation.tensors)
        self._field_registry = dict(installation.fields)
        if self.save_kernels and installation.statics:
            path = self.kernels_dir / (
                f"kern_static_{self._generate_unique_name()}.py"
            )
            atomic_write_text(
                path,
                "def gather_static_var(tensor, output_index):\n"
                "    return tensor if output_index is None else "
                "tensor[output_index]\n",
            )
        for name in self._tensor_registry.keys() | self._field_registry.keys():
            self._get_safe_name(name)
        for binding in installation.statics:
            self._materialize_static(binding)
        compilation = compile_statistics(
            self,
            installation.variable_ops,
            installation.program,
        )
        self._activate_compilation(compilation)

    def _activate_compilation(
        self, compilation: StatisticsCompilation,
    ) -> None:
        """
        Initialize streaming aggregation for specified variables.
        Creates NetCDF file structure but writes time steps incrementally.

        Args:
            compilation: Compiler-owned operations, expressions and layouts.
        """
        from hydroforge.contracts.events import emit

        emit(
            self, "info", "statistics.variables",
            "Configured statistics variables",
            variables=dict(compilation.variable_ops),
        )

        # Enable streaming mode
        self._files_created = False
        self._current_year = None
        self._output.reset_timeline()

        # Initialize single time step aggregation (generic)
        self._materialize_compilation(compilation)

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

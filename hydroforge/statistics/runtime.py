# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import atexit
import math
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import cftime
import numpy as np
import torch

from hydroforge.compiler.generated import release_generated_module
from hydroforge.contracts.errors import ResourceCleanupError, cleanup_on_exit
from hydroforge.contracts.events import ConsoleEventSink, emit
from hydroforge.contracts.fields import RuntimeTensorMetadata
from hydroforge.contracts.naming import sanitize_symbol
from hydroforge.contracts.runtime import DEFAULT_BLOCK_SIZE
from hydroforge.kernels.backends.triton.dispatcher import TRITON_INT32_MAX_EXTENT
from hydroforge.output.conversion import _checked_output_tensor_copy
from hydroforge.output.netcdf.writer import _NetCDFWriter
from hydroforge.statistics.compiler import compile_statistics_program
from hydroforge.statistics.emitters.common import (
    HostSamplePhase,
    StatisticsCompileContext,
)
from hydroforge.statistics.ir import (
    Reduction,
    ScatterSource,
    StatisticsProgram,
    StorageDType,
    StorageInitialization,
    TensorSource,
    build_statistics_ir,
    build_variable_storage_plan,
)
from hydroforge.statistics.layout import (
    StatisticsCompilation,
    StatisticsVariableLayout,
    compile_statistics,
)


@lru_cache(maxsize=128)
def _control_float_value(dtype: torch.dtype, value: float) -> float:
    return float(torch.tensor(value, dtype=dtype, device="cpu").item())


def _weak_shutdown_callback(runtime: Any):
    """Return an atexit callback that does not keep ``runtime`` alive."""

    runtime_ref = weakref.ref(runtime)

    def shutdown() -> None:
        instance = runtime_ref()
        if instance is not None:
            instance.close()

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
    source_copies: tuple[tuple[str, torch.Tensor, torch.Tensor], ...] = ()


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
    ensemble_size: int = 1
    ensemble_member_ids: tuple[int, ...] | None = None
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
    event_sink: Any = field(default_factory=ConsoleEventSink)

    kernels_dir: Path | None = field(init=False, default=None, repr=False)
    static_vars: dict[str, dict[str, Any]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._closed = False

        # Create kernels directory if saving is enabled (must precede any
        # codegen step so the generated .py files have a destination).
        if self.save_kernels:
            self.kernels_dir = self.output_dir / "generated_kernels"
            self.kernels_dir.mkdir(parents=True, exist_ok=True)

        self._macro_step_index = 0  # Current macro step index (outer loop counter)
        self._macro_mean_count_limit: int | None = None

        # Internal state
        # Generic stats state (for all ops)
        self._variables: set[str] = set()  # original variable names
        self._variable_ops: dict[str, list[str]] = {}  # var -> list[ops]
        self._storage: dict[str, torch.Tensor] = {}  # out_name -> tensor
        self._output_keys: list[str] = []  # list of keys in storage that are outputs
        self._host_phase = HostSamplePhase()
        self._output_metadata: dict[str, dict[str, Any]] = {}  # out_name -> meta
        self._coord_cache: dict[str, np.ndarray] = {}

        self._tensor_registry: dict[str, torch.Tensor] = {}
        self._field_registry: dict[str, RuntimeTensorMetadata] = {}
        self._structural_tensor_versions: dict[str, tuple[torch.Tensor, int]] = {}

        # Cache for sanitized names
        self._safe_name_cache: dict[str, str] = {}

        # Kernel state (mean fast-path)
        self._kernel_module = None
        self._generated_modules: list[tuple[str, str]] = []
        self._saved_kernel_file = None
        self._dirty_outputs: set[str] = set()
        self._output: _NetCDFWriter | None = None
        self._last_time_number: float | None = None

        # In-memory result tensors: out_name -> list of tensors (one per time step)
        # Only used when in_memory=True
        self._result_tensors: dict[str, list[torch.Tensor]] = {}
        self._current_time_index: int = 0

        emit(
            self,
            "info",
            "statistics.initialized",
            "Initialized streaming statistics",
            rank=self.rank,
            workers=self.num_workers,
        )
        if self.in_memory:
            emit(
                self,
                "info",
                "statistics.memory_mode",
                "Statistics results will be retained in memory",
                device=self.result_device,
            )
        if self.save_kernels:
            emit(
                self,
                "info",
                "statistics.kernel_output",
                "Generated statistics kernels will be saved",
                directory=self.kernels_dir,
            )
        self._atexit_callback = _weak_shutdown_callback(self)
        atexit.register(self._atexit_callback)
        try:
            self._materialize_installation(self.installation)
        except BaseException:
            with cleanup_on_exit("statistics initialization", (self.close,)):
                raise

    def _prepare_kernel_states(self) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        required_tensors: dict[str, torch.Tensor] = {}
        ir = self._statistics_ir

        # Add original variables and their output buffers
        for variable in ir.variables:
            var_name = variable.name
            for dependency in self._statistics_program.leaf_tensors(var_name):
                required_tensors[dependency] = self._tensor_registry[dependency]

            for operation in variable.operations:
                op = operation.spelling
                out_name = f"{var_name}_{op}"
                required_tensors[out_name] = self._storage[out_name]

                # For explicit argmax/argmin operations, add their auxiliary storage
                if operation.stores_index:
                    aux_name = f"{var_name}_{operation.spelling}_aux"
                    required_tensors[aux_name] = self._storage[aux_name]

                if operation.inner is None and operation.outer is Reduction.MEAN:
                    weight_name = f"{var_name}_mean_sample_weight_state"
                    required_tensors[weight_name] = self._storage[weight_name]

                # Add inner states for compound ops
                if operation.inner is not None:
                    inner = operation.inner.value
                    # 'last' inner op doesn't need cross-step state
                    if inner != "last":
                        inner_name = f"{var_name}_{inner}_inner_state"
                        required_tensors[inner_name] = self._storage[inner_name]
                        if inner == "mean":
                            w_name = f"{var_name}_{inner}_weight_state"
                            required_tensors[w_name] = self._storage[w_name]

        # Collect required dimensions and output indices.
        required_dims: set[str] = set()
        required_output_indices: set[str] = set()
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
            if scatter.reduction.value == "mean":
                cnt_key = f"__scatter_cnt_{var_name}"
                required_tensors[cnt_key] = self._storage[cnt_key]
            # Ensure all scatter source tensors and index are in required_tensors
            required_tensors[scatter.index] = self._tensor_registry[scatter.index]
            for dependency in self._statistics_program.leaf_tensors(var_name):
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
        if self.backend == "triton" and self.device.type == "cuda":
            self._check_triton_offsets(required_tensors)
        control_dtype = self._statistics_control_dtype()
        layout = (
            ("__weight", control_dtype),
            ("__total_weight", control_dtype),
            ("__num_macro_steps", torch.int64),
            ("__macro_step_index", torch.int64),
            ("__sub_step", torch.int32),
            ("__num_sub_steps", torch.int32),
            ("__flags", torch.int32),
        )
        if self.device.type == "mps":
            # An unpinned host copy to MPS waits for the whole queue; queued
            # fills keep sampling asynchronous.
            required_tensors.update(
                {
                    name: torch.zeros(1, dtype=dtype, device=self.device)
                    for name, dtype in layout
                }
            )
            self._kernel_states = required_tensors
            self._control_host_slots = None
            return
        control_bytes = sum(dtype.itemsize for _name, dtype in layout)
        control_buffer = torch.zeros(
            control_bytes, dtype=torch.uint8, device=self.device
        )
        host_slots = []
        # Pinned slots let the host run ahead of queued statistics launches.
        for _slot in range(16 if self.device.type == "cuda" else 1):
            host = torch.zeros(
                control_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.device.type == "cuda",
            )
            host_slots.append(
                (host, {}, torch.cuda.Event() if self.device.type == "cuda" else None)
            )
        offset = 0
        for name, dtype in layout:
            stop = offset + dtype.itemsize
            required_tensors[name] = control_buffer[offset:stop].view(dtype)
            for host, views, _event in host_slots:
                views[name] = host[offset:stop].view(dtype).numpy()
            offset = stop
        # Publish only after dependency resolution, device checks, and every
        # allocation succeeded.  Rebinding may never expose partial states.
        self._kernel_states = required_tensors
        self._control_buffer = control_buffer
        self._control_host_slots = host_slots
        self._control_slot_index = 0
        self._control_host, self._control_host_values, _event = host_slots[0]

    def _check_triton_offsets(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Reject bindings whose int32 Triton element offsets would wrap."""

        for name, tensor in tensors.items():
            # Masked tail lanes of the last block address up to one block
            # of rows (times the trailing level/top-k axis) past the end.
            trailing = int(tensor.shape[-1]) if tensor.ndim > 1 else 1
            limit = TRITON_INT32_MAX_EXTENT - self.block_size * trailing
            if tensor.numel() > limit:
                raise OverflowError(
                    f"statistics buffer {name!r} has {tensor.numel()} elements "
                    "(including ensemble members); Triton statistics kernels "
                    f"use int32 offsets, so it must hold <= {limit} for "
                    f"BLOCK_SIZE={self.block_size}; use the cuda backend or "
                    "split the domain or ensemble"
                )

    def _statistics_control_dtype(self) -> torch.dtype:
        """Return the precision shared by aggregation control scalars."""

        if self.device.type == "mps":
            return torch.float32
        if any(tensor.dtype == torch.float64 for tensor in self._storage.values()):
            return torch.float64
        return torch.float32

    def _materialize_compilation(
        self,
        compilation: StatisticsCompilation,
    ) -> None:
        """Materialize one trusted compiler-owned statistics program."""
        self._variable_ops = {
            name: list(operations)
            for name, operations in compilation.variable_ops.items()
        }
        self._statistics_program = compilation.program
        self._statistics_layouts = compilation.layouts
        self._output_is_outer: dict[str, bool] = {}

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

        # Visible outputs and hidden dependencies use the same scatter storage.
        for var_name, source in self._statistics_program.sources.items():
            if not isinstance(source, ScatterSource):
                continue
            layout = self._statistics_layouts[var_name]
            full_target_size = layout.scatter_extent
            shape = (
                (self.ensemble_size, full_target_size)
                if layout.batched
                else (full_target_size,)
            )
            self._storage[f"__scatter_buf_{var_name}"] = torch.zeros(
                shape,
                dtype=layout.dtype,
                device=self.device,
            )
            if source.reduction is Reduction.MEAN:
                self._storage[f"__scatter_cnt_{var_name}"] = torch.zeros(
                    shape,
                    dtype=torch.int32,
                    device=self.device,
                )

        for var_name in self._variable_ops:
            operation_nodes = self._statistics_program.operations[var_name]
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

            # Track
            self._variables.add(var_name)

            storage_plan = build_variable_storage_plan(
                var_name,
                tuple(actual_shape),
                operation_nodes,
            )
            for slot in storage_plan.slots:
                dtype = (
                    torch.int64 if slot.dtype is StorageDType.INDEX else target_dtype
                )
                if slot.initialization is StorageInitialization.NEGATIVE_INFINITY:
                    initial = (
                        -torch.inf
                        if dtype.is_floating_point
                        else False
                        if dtype is torch.bool
                        else torch.iinfo(dtype).min
                    )
                    tensor = torch.full(
                        slot.shape,
                        initial,
                        dtype=dtype,
                        device=self.device,
                    )
                elif slot.initialization is StorageInitialization.POSITIVE_INFINITY:
                    initial = (
                        torch.inf
                        if dtype.is_floating_point
                        else True
                        if dtype is torch.bool
                        else torch.iinfo(dtype).max
                    )
                    tensor = torch.full(
                        slot.shape,
                        initial,
                        dtype=dtype,
                        device=self.device,
                    )
                else:
                    tensor = torch.zeros(
                        slot.shape,
                        dtype=dtype,
                        device=self.device,
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

                meta = {
                    "full_output": full_output,
                    "tensor_shape": tensor_shape,
                    "dtype": self._storage[out_name].dtype,
                    "actual_shape": tuple(self._storage[out_name].shape),
                    "batched": layout.batched,
                    "output_coord": output_coord,
                    "dim_coords": dim_coords,
                    "description": f"{description} ({op})",
                    "k": operation.k,
                }
                self._output_metadata[out_name] = meta

                # Classify as outer if it is a compound op (e.g. max_mean)
                self._output_is_outer[out_name] = operation.compound

        # Generate kernels and prepare states for all requested variables/ops
        self._compile_program(self._statistics_layouts)
        self._prepare_kernel_states()

    def _claim_macro_step(
        self,
        *,
        is_inner_last: bool,
        is_outer_first: bool,
        is_outer_last: bool,
    ) -> tuple[int, int]:
        macro_step_index = 0 if is_outer_first else self._macro_step_index
        macro_step_count = 0 if is_outer_first else self._current_macro_step_count
        next_count = macro_step_count + int(is_inner_last)
        limit = torch.iinfo(torch.int64).max
        if (
            macro_step_index > limit
            or next_count > limit
            or (is_inner_last and macro_step_index == limit)
        ):
            raise OverflowError("statistics macro-step accounting exceeds int64 range")
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
                name for name, outer in self._output_is_outer.items() if not outer
            )
        if is_outer_last:
            self._dirty_outputs.update(
                name for name, outer in self._output_is_outer.items() if outer
            )
        if is_inner_last:
            self._current_macro_step_count = next_count
            self._macro_step_index = macro_step_index + 1
        return self._current_macro_step_count, macro_step_index

    def _convert_control_float(self, name: str, value: float) -> float:
        """Convert a trusted schedule weight while preserving overflow errors."""

        states = self._kernel_states
        dtype = states[f"__{name}"].dtype
        converted = _control_float_value(dtype, value)
        if not math.isfinite(converted):
            raise OverflowError(f"statistics {name} {value!r} exceeds {dtype} range")
        if converted == 0.0:
            raise OverflowError(f"statistics {name} {value!r} underflows {dtype}")
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
            "total_weight",
            total_weight,
        )

        is_inner_first = bool(flags & 1) and sub_step == 0
        is_inner_last = bool(flags & 2) and (sub_step == num_sub_steps - 1)
        is_outer_first = bool(flags & 4) and is_inner_last
        is_outer_last = bool(flags & 8) and is_inner_last
        num_macro_steps, macro_step_index = self._claim_macro_step(
            is_inner_last=is_inner_last,
            is_outer_first=is_outer_first,
            is_outer_last=is_outer_last,
        )

        if self._control_host_slots is None:
            values = {
                "__weight": converted_weight,
                "__total_weight": converted_total,
                "__num_macro_steps": num_macro_steps,
                "__sub_step": sub_step,
                "__num_sub_steps": num_sub_steps,
                "__flags": flags,
                "__macro_step_index": macro_step_index,
            }
            for name, value in values.items():
                self._kernel_states[name].fill_(value)
            self._execute_statistics_kernel()
            return
        slots = self._control_host_slots
        host, host_values, event = slots[self._control_slot_index]
        if event is not None and not event.query():
            event.synchronize()
        self._control_host = host
        self._control_host_values = host_values
        host_values["__weight"][0] = converted_weight
        host_values["__total_weight"][0] = converted_total
        host_values["__num_macro_steps"][0] = num_macro_steps
        host_values["__sub_step"][0] = sub_step
        host_values["__num_sub_steps"][0] = num_sub_steps
        host_values["__flags"][0] = flags
        host_values["__macro_step_index"][0] = macro_step_index
        self._control_buffer.copy_(host, non_blocking=event is not None)
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
        self._control_slot_index = (self._control_slot_index + 1) % len(slots)

        # Captured launches must keep reading the device control state.
        if getattr(self.execution, "capture_mode", None) != "cuda_graph":
            self._host_phase.bits = (
                is_inner_first
                | is_inner_last << 1
                | is_outer_first << 2
                | is_outer_last << 3
            )
        try:
            self._execute_statistics_kernel()
        finally:
            self._host_phase.bits = -1

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

    def _result_snapshot(
        self, out_name: str, selection: slice, *, as_stacked: bool
    ) -> torch.Tensor | list[torch.Tensor]:
        values = self._result_tensors[out_name][selection]
        if not as_stacked:
            return [
                value.clone(memory_format=torch.preserve_format) for value in values
            ]
        return torch.stack(values, dim=0) if values else self._empty_result(out_name)

    def get_results(
        self,
        as_stacked: bool = True,
        *,
        start: int | None = None,
        stop: int | None = None,
    ):
        """Return isolated copies of an optional retained-output interval."""

        selection = slice(start, stop)
        return {
            name: self._result_snapshot(name, selection, as_stacked=as_stacked)
            for name in self._result_tensors
        }

    def get_result(
        self,
        variable_name: str,
        op: str = "mean",
        as_stacked: bool = True,
        *,
        start: int | None = None,
        stop: int | None = None,
    ):
        return self._result_snapshot(
            f"{variable_name}_{op}", slice(start, stop), as_stacked=as_stacked
        )

    def iter_results(self, batch_size: int = 64):
        """Yield isolated batches using the validated model query's count."""

        lengths = (len(values) for values in self._result_tensors.values())
        for start in range(0, max(lengths, default=0), batch_size):
            yield self.get_results(start=start, stop=start + batch_size)

    def drain_results(self, max_steps: int | None = None, *, as_stacked: bool = True):
        """Copy then release retained samples, without resetting simulation time."""

        result = self.get_results(as_stacked, stop=max_steps)
        for values in self._result_tensors.values():
            del values[:max_steps]
        return result

    def accumulator(self, variable: str, operation: str) -> torch.Tensor:
        """Return an ownership-isolated differentiable accumulator snapshot."""
        return self._storage[f"{variable}_{operation}"].clone(
            memory_format=torch.preserve_format,
        )

    def pop_result(self, variable: str, operation: str) -> torch.Tensor | None:
        """Remove and return the newest finalized in-memory result."""
        values = self._result_tensors[f"{variable}_{operation}"]
        return values.pop() if values else None

    def get_time_index(self) -> int:
        return self._current_time_index

    def reset_time_index(self) -> None:
        if self._output is not None:
            self._output.reset_staging()
        self._current_time_index = 0
        self._last_time_number = None
        for out_name in self._result_tensors:
            self._result_tensors[out_name] = []

    def _validate_next_time(
        self,
        dt: datetime | cftime.datetime,
    ) -> float:
        value = float(
            cftime.date2num(
                dt,
                units=self.time_unit,
                calendar=self.calendar,
            )
        )
        if not np.isfinite(value):
            raise ValueError("statistics output time must be finite")
        if self._last_time_number is not None and value <= self._last_time_number:
            raise ValueError("statistics output times must be strictly increasing")
        return value

    def finalize_time_step(self, dt: Any) -> None:
        time_number = self._validate_next_time(dt)
        keys = [key for key in self._output_keys if key in self._dirty_outputs]
        if self._output is None:
            copies = {
                key: _checked_output_tensor_copy(
                    self._storage[key],
                    target_device=self.result_device,
                    target_dtype=self._result_dtype(key),
                    name=key,
                )
                for key in keys
            }
            for key, value in copies.items():
                self._result_tensors[key].append(value)
        else:
            self._output.append(dt, {key: self._storage[key] for key in keys})
        self._dirty_outputs.difference_update(keys)
        self._current_time_index += 1
        self._last_time_number = time_number

    def check_background_failures(self, current_time: Any = None) -> None:
        """Raise completed asynchronous output failures without waiting."""

        if self._output is not None:
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
                (field := self._field_registry.get(leaf)) is not None
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
                and (output_field := self._field_registry.get(variable)) is not None
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
            self.installation.variable_ops,
            self.installation.program,
            tensors=self._tensor_registry,
            fields=self._field_registry,
            ensemble_size=self.ensemble_size,
            base_dtype=self.base_dtype,
            mixed_precision=self.mixed_precision,
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

        self._compile_program(compilation.layouts)
        self._prepare_kernel_states()
        self._structural_tensor_versions = {}
        self.execution.statistics.invalidate()

    def refresh_address_stable_sources(self) -> None:
        """Refresh compiler-owned index copies without replacing graph storage."""

        with torch.inference_mode():
            for name, live, installed in self.installation.source_copies:
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

    def _compile_program(
        self,
        layouts: Mapping[str, StatisticsVariableLayout],
    ) -> None:
        """Compile a candidate before replacing the installed statistics program."""
        ir = build_statistics_ir(
            self._statistics_program,
            fields=self._field_registry,
            layouts=layouts,
            symbol_names=self._safe_name_cache,
        )
        context = StatisticsCompileContext(
            device=self.device,
            rank=self.rank,
            ensemble_size=self.ensemble_size,
            save_kernels=self.save_kernels,
            kernels_dir=self.kernels_dir,
            variables=frozenset(self._variables),
            layouts=MappingProxyType(dict(layouts)),
            storage=MappingProxyType(dict(self._storage)),
            tensors=MappingProxyType(dict(self._tensor_registry)),
            symbol_names=MappingProxyType(dict(self._safe_name_cache)),
            control_dtype=self._statistics_control_dtype(),
            host_phase=self._host_phase,
        )
        backend = {"cpu": "torch", "mps": "metal"}.get(self.device.type, self.backend)
        result = compile_statistics_program(context, ir, backend=backend)
        try:
            self._cleanup_generated_modules()
        except BaseException:
            with cleanup_on_exit(
                "uninstalled statistics program",
                (
                    partial(release_generated_module, name, filename)
                    for name, filename in result.generated_modules
                ),
            ):
                raise
        self._statistics_layouts = layouts
        self._statistics_ir = result.lowering.ir
        self._statistics_lowering = result.lowering
        self._generated_modules = list(result.generated_modules)
        self._aggregator_function = result.function
        self._kernel_module = result.module
        self._saved_kernel_file = result.saved_kernel_file

    def _cleanup_generated_modules(self) -> None:
        modules, self._generated_modules = self._generated_modules, []
        self._kernel_module = None
        with cleanup_on_exit(
            "statistics generated modules",
            (
                partial(release_generated_module, name, filename)
                for name, filename in reversed(modules)
            ),
        ):
            pass

    def _unregister_atexit(self) -> None:
        callback = getattr(self, "_atexit_callback", None)
        if callback is not None:
            atexit.unregister(callback)
            self._atexit_callback = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        for cleanup in (
            self._unregister_atexit,
            self._cleanup_generated_modules,
            *((self._output.close,) if self._output is not None else ()),
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

    def __del__(self) -> None:
        try:
            self.close()
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
            "attrs": {},
        }

    def _materialize_installation(
        self,
        installation: StatisticsInstallation,
    ) -> None:
        """Materialize the complete compiler-owned registry during construction."""

        self._tensor_registry = dict(installation.tensors)
        self._field_registry = dict(installation.fields)
        for name in self._tensor_registry.keys() | self._field_registry.keys():
            self._get_safe_name(name)
        for binding in installation.statics:
            self._materialize_static(binding)
        compilation = compile_statistics(
            installation.variable_ops,
            installation.program,
            tensors=self._tensor_registry,
            fields=self._field_registry,
            ensemble_size=self.ensemble_size,
            base_dtype=self.base_dtype,
            mixed_precision=self.mixed_precision,
        )
        self._activate_compilation(compilation)

    def _activate_compilation(
        self,
        compilation: StatisticsCompilation,
    ) -> None:
        """
        Initialize streaming aggregation for specified variables.
        Creates NetCDF file structure but writes time steps incrementally.

        Args:
            compilation: Compiler-owned operations, expressions and layouts.
        """
        emit(
            self,
            "info",
            "statistics.variables",
            "Configured statistics variables",
            variables=dict(compilation.variable_ops),
        )

        # Enable streaming mode
        if self._output is not None:
            self._output.reset_staging()
        self._last_time_number = None

        # Initialize single time step aggregation (generic)
        self._materialize_compilation(compilation)

        # If in-memory mode, initialize result storage lists instead of starting file writers
        if self.in_memory:
            self._init_result_storage()
            emit(
                self,
                "info",
                "statistics.memory_ready",
                "In-memory statistics aggregation initialized",
                outputs=len(self._result_tensors),
            )
        else:
            self._output = _NetCDFWriter(
                metadata=self._output_metadata,
                coordinates=self._coord_cache,
                static_vars=self.static_vars,
                variable_options=self.installation.netcdf_options,
                output_dir=self.output_dir,
                rank=self.rank,
                world_size=self.world_size,
                ensemble_size=self.ensemble_size,
                ensemble_member_ids=self.ensemble_member_ids,
                calendar=self.calendar,
                time_unit=self.time_unit,
                num_workers=self.num_workers,
                output_split_by_year=self.output_split_by_year,
                max_pending_steps=self.max_pending_steps,
                max_pending_output_bytes=self.max_pending_output_bytes,
                save_precision=self.save_precision,
                event_sink=self.event_sink,
                on_failure=partial(
                    self.execution.poison, phase="statistics background write"
                ),
            )
            self._output.start()
            emit(
                self,
                "info",
                "statistics.streaming_ready",
                "Streaming statistics aggregation initialized",
                executors=self.num_workers,
            )

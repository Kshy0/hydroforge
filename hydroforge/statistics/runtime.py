# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import atexit
import math
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cftime
import numpy as np
import torch

from hydroforge.contracts.fields import RuntimeTensorMetadata
from hydroforge.contracts.schedule import SimulationSchedule, SimulationStep
from hydroforge.contracts.windows import StatisticsPlan
from hydroforge.core.arrays import find_indices_in_torch
from hydroforge.core.devices import devices_match
from hydroforge.core.errors import cleanup_on_exit
from hydroforge.core.events import ConsoleEventSink, emit
from hydroforge.core.expr import Reduction, ScatterSource, TensorSource
from hydroforge.core.naming import sanitize_symbol
from hydroforge.io.netcdf.encoding import saved_dtype
from hydroforge.io.rank_output.schema import TIME_UNITS
from hydroforge.io.rank_output.writer import RankOutputWriter
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain.python import release_generated_module
from hydroforge.platform.backend import Backend
from hydroforge.statistics.compiler import compile_statistics_program
from hydroforge.statistics.ir import StatisticsProgram, build_statistics_ir
from hydroforge.statistics.kernel_plan import StatisticsCompileContext
from hydroforge.statistics.layout import (
    StatisticsCompilation,
    StatisticsVariableLayout,
    compile_statistics,
)
from hydroforge.statistics.phases import (
    CONTROL_FLAGS,
    CONTROL_LAYOUT,
    CONTROL_MACRO_INDEX,
    CONTROL_MACRO_STEPS,
    CONTROL_PHASE,
    CONTROL_WEIGHT,
    SampleFlags,
    sample_phase,
)
from hydroforge.statistics.sinks import MemorySink, StatisticsSink
from hydroforge.statistics.storage import (
    StorageInitialization,
    StoragePlan,
    build_storage_plan,
)
from hydroforge.statistics.windows import (
    WHOLE_WINDOW,
    StatisticsWindowController,
    WindowState,
)

_INNER_LAST = int(SampleFlags.INNER_LAST)
_OUTER_FIRST = int(SampleFlags.OUTER_FIRST)


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
class StatisticsLaunch:
    """One installed statistics program and the storage it binds.

    ``function(states, block_size, phase)`` runs one sample; a negative
    ``phase`` gates every kernel on the device control instead, which is the
    form a captured or device-loop launch uses.  ``mutated`` is the storage
    the program writes, which capture rollback restores; model sources are
    only read and every launch path rewrites the control scalars first.
    """

    function: Callable[[Mapping[str, torch.Tensor], int, int], None]
    states: Mapping[str, torch.Tensor]
    block_size: int
    mutated: tuple[torch.Tensor, ...]

    def __call__(self, phase: int) -> None:
        self.function(self.states, self.block_size, phase)


@dataclass(frozen=True, slots=True)
class StatisticsInstallation:
    """Complete trusted compiler output installed into a runtime once."""

    variable_ops: Mapping[str, tuple[str, ...]]
    program: StatisticsProgram
    tensors: Mapping[str, torch.Tensor]
    fields: Mapping[str, RuntimeTensorMetadata]
    scatter_extents: Mapping[str, int]
    statics: tuple[StatisticsStaticBinding, ...]
    netcdf_options: Mapping[str, Mapping[str, Any]]
    selection_sources: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] = ()


@dataclass
class StatisticsRuntime:
    """Trusted execution state built from a validated model declaration."""

    device: torch.device
    backend: Backend
    installation: StatisticsInstallation = field(repr=False)
    windows: StatisticsPlan = field(repr=False)
    schedule: SimulationSchedule | None = field(repr=False)
    on_write_failure: Callable[[BaseException], None] = field(repr=False)
    block_size: int
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
    calendar: str = "standard"
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

        self._controller = StatisticsWindowController(self.windows, self.schedule)
        self._step_flags = 0

        # Internal state
        # Generic stats state (for all ops)
        self._variables: set[str] = set()  # original variable names
        self._variable_ops: dict[str, list[str]] = {}  # var -> list[ops]
        self._storage: dict[str, torch.Tensor] = {}  # out_name -> tensor
        self._output_keys: list[str] = []  # list of keys in storage that are outputs
        self._output_metadata: dict[str, dict[str, Any]] = {}  # out_name -> meta
        self._coord_cache: dict[str, np.ndarray] = {}

        self._tensor_registry: dict[str, torch.Tensor] = {}
        self._field_registry: dict[str, RuntimeTensorMetadata] = {}

        # Cache for sanitized names
        self._safe_name_cache: dict[str, str] = {}

        # Kernel state (mean fast-path)
        self._kernel_module = None
        self._generated_modules: list[tuple[str, str]] = []
        self._saved_kernel_file = None
        self._close_program = lambda: None
        # Destination of finalized samples, installed with the compilation.
        self.sink: StatisticsSink | None = None
        self._last_time_number: float | None = None
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

    def _prepare_kernel_states(self, ir=None) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        ir = self._statistics_ir if ir is None else ir
        scatters = ir.ordered_scatters()
        required_tensors: dict[str, torch.Tensor] = {
            name: self._storage[name]
            for name in self._storage_plan.owned_by(
                (
                    *(variable.name for variable in ir.variables),
                    *(scatter.name for scatter in scatters),
                )
            )
        }
        for name in (
            *(variable.name for variable in ir.variables),
            *(scatter.name for scatter in scatters),
        ):
            for dependency in self._statistics_program.leaf_tensors(name):
                required_tensors[dependency] = self._tensor_registry[dependency]
        for scatter in scatters:
            index = scatter.source.index
            required_tensors[index] = self._tensor_registry[index]
        for variable in ir.variables:
            if variable.output_group != "__full__":
                required_tensors[variable.output_group] = self._tensor_registry[
                    variable.output_group
                ]
            for dim_name in variable.tensor_shape:
                if isinstance(dim_name, str) and dim_name in self._tensor_registry:
                    required_tensors[dim_name] = self._tensor_registry[dim_name]

        for name, tensor in required_tensors.items():
            if not devices_match(tensor.device, self.device):
                raise ValueError(f"statistics buffer {name!r} must be on {self.device}")
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                raise ValueError(
                    f"statistics buffer {name!r} must be contiguous strided storage"
                )
        # Scalar parameters as 1-element device tensors for CUDA Graph compatibility.
        # Kernel code loads these via tl.load (Triton) or reads from states dict,
        # so CUDA Graphs can replay without recapture when values change.
        if self.backend.max_extent is not None:
            self._check_offsets(required_tensors)
        control_dtype = self._statistics_control_dtype()
        layout = tuple(
            (name, control_dtype if dtype is None else dtype)
            for name, dtype in CONTROL_LAYOUT
        )
        if self.device.type == "mps":
            # An unpinned host copy to MPS waits for the whole queue; queued
            # fills keep sampling asynchronous.
            required_tensors.update(
                {name: self._full_tensor((1,), 0, dtype) for name, dtype in layout}
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

    def _check_offsets(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Reject bindings whose kernel element offsets would wrap."""

        for name, tensor in tensors.items():
            # Masked tail lanes of the last block address up to one block of
            # rows past the end; only indexed storage rows span a trailing
            # level/top-k axis.  Every other binding is addressed flat.
            row = self._indexed_storage_rows.get(name, 1)
            self.backend.validate_extent(
                f"statistics buffer {name!r}", tensor.numel(), self.block_size * row
            )

    def _statistics_control_dtype(self) -> torch.dtype:
        """Return the precision shared by aggregation control scalars."""

        if self.backend.name == "metal" and any(
            layout.dtype == torch.float64
            for layout in self._statistics_layouts.values()
        ):
            return torch.float64
        if "float64" not in self.backend.precisions:
            return torch.float32
        if any(tensor.dtype == torch.float64 for tensor in self._storage.values()):
            return torch.float64
        return torch.float32

    def _full_tensor(self, shape, value, dtype):
        """Allocate statistics storage explicitly, including later recompiles."""
        if self.backend.name == "metal" and dtype == torch.float64:
            from hydroforge.kernels.emulated import EmulatedTensor

            return EmulatedTensor.encode(
                torch.full(shape, value, dtype=dtype, device="cpu"), self.device
            )
        return torch.full(shape, value, dtype=dtype, device=self.device)

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
        outer_outputs: dict[str, bool] = {}
        self._indexed_storage_rows: dict[str, int] = {}

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
                mean_count_limits.add(2**48 if self.backend.name == "metal" else 2**53)

        # Visible outputs and hidden dependencies use the same scatter storage.
        self._storage_plan = build_storage_plan(
            compilation.program,
            compilation.layouts,
            self._variable_ops,
            self.ensemble_size,
        )
        for slot in self._storage_plan.slots.values():
            row = 1
            if slot.owner in self._variable_ops:
                layout = self._statistics_layouts[slot.owner]
                if self._field_registry[slot.owner].output_index is not None:
                    row = math.prod(slot.shape[int(layout.batched) + 1 :])
            self.backend.validate_extent(
                f"statistics buffer {slot.name!r}",
                math.prod(slot.shape),
                self.block_size * row,
            )
        for slot in self._storage_plan.slots.values():
            dtype = slot.dtype
            if slot.initialization is StorageInitialization.NEGATIVE_INFINITY:
                initial = (
                    -torch.inf
                    if dtype.is_floating_point
                    else False
                    if dtype is torch.bool
                    else torch.iinfo(dtype).min
                )
            elif slot.initialization is StorageInitialization.POSITIVE_INFINITY:
                initial = (
                    torch.inf
                    if dtype.is_floating_point
                    else True
                    if dtype is torch.bool
                    else torch.iinfo(dtype).max
                )
            else:
                initial = 0
            self._storage[slot.name] = self._full_tensor(slot.shape, initial, dtype)
            if slot.owner in self._variable_ops:
                layout = self._statistics_layouts[slot.owner]
                if self._field_registry[slot.owner].output_index is not None:
                    # Saved points index the rows after any member axis.
                    self._indexed_storage_rows[slot.name] = math.prod(
                        slot.shape[int(layout.batched) + 1 :]
                    )
            if slot.output:
                self._output_keys.append(slot.name)

        for var_name in self._variable_ops:
            operation_nodes = self._statistics_program.operations[var_name]
            field_info = self._field_registry[var_name]
            metadata = field_info.tensor
            layout = self._statistics_layouts[var_name]
            tensor_shape = metadata.shape
            description = field_info.description
            output_coord = field_info.output_coord
            dim_coords = metadata.dim_coords
            full_output = field_info.output_index is None

            # Track
            self._variables.add(var_name)

            for operation in operation_nodes:
                op = operation.spelling
                out_name = StoragePlan.output(var_name, op)

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

                # Compound outputs (e.g. max_mean) publish at outer closes
                outer_outputs[out_name] = operation.compound

        self.window_state = WindowState(
            inner_outputs=tuple(
                name for name, outer in outer_outputs.items() if not outer
            ),
            outer_outputs=tuple(name for name, outer in outer_outputs.items() if outer),
            mean_count_limit=min(mean_count_limits) if mean_count_limits else None,
        )
        # Generate kernels and prepare states for all requested variables/ops
        self._compile_program(self._statistics_layouts)
        self._publish_launch()

    def _publish_launch(self) -> None:
        self.launch = StatisticsLaunch(
            self._aggregator_function,
            self._kernel_states,
            self.block_size,
            tuple(self._storage.values()),
        )
        self._requests = self._request_builder(self._kernel_states, self.block_size)

    def take_requests(self) -> tuple[CompileRequest, ...]:
        """The native compilation of the published program, once."""

        requests, self._requests = self._requests, ()
        return requests

    @property
    def per_substep(self) -> bool:
        """Whether device loops must fold a sample into every iteration."""
        return self._statistics_lowering.per_substep

    def begin_step(
        self,
        step: SimulationStep | None,
        *,
        enabled: bool,
        time: Any,
    ) -> bool:
        """Resolve the windows of one managed step; return whether it samples.

        ``step`` is ``None`` for an unscheduled call, which is a whole
        window.  A spin-up step belongs to no window.
        """

        if step is None:
            events = WHOLE_WINDOW
        elif step.is_spin_up:
            events = None
        else:
            events = self._controller.resolve(step)
        self._step_flags = self.window_state.begin(events, sampling=enabled, time=time)
        return self.window_state.sampling

    @property
    def step_flags(self) -> int:
        """Window bits of the begun step's samples (0 when it samples none)."""
        return self._step_flags

    @property
    def step_closes(self) -> bool:
        """Whether the begun step closes an inner window and may publish."""
        events = self.window_state.events
        return events is not None and events.inner_last

    def snapshot(self) -> tuple[Any, Any]:
        """Capture the window cursor that ``begin_step`` advances."""
        return self._controller.snapshot_state()

    def restore(self, snapshot: tuple[Any, Any]) -> None:
        self._controller.restore_snapshot_state(snapshot)
        self.window_state.events = None

    def sample(self, *, first: bool, last: bool, weight: float) -> int:
        """Publish the controls of one host-issued sample; return its phase."""

        phase = sample_phase(self._step_flags, first=first, last=last)
        state = self.window_state
        count, index = (
            state.claim(phase)
            if phase & _INNER_LAST
            else (state.macro_count, state.macro_index)
        )
        self._write_control(self._convert_weight(weight), phase, count, index)
        return phase

    def prelaunch(self) -> None:
        """Publish the controls of a device loop that folds every iteration.

        Loop controls write each iteration's weight and phase from the step
        bits written here; a fold at the loop's last iteration is claimed now.
        """

        flags = self._step_flags
        state = self.window_state
        count, index = (
            state.claim(flags)
            if flags & _INNER_LAST
            else (state.macro_count, state.macro_index)
        )
        self._write_control(0.0, 0, count, index)

    def finish_step(self) -> None:
        """Close the begun step: settle a window without a closing sample,
        then publish every output its closes completed."""

        state = self.window_state
        phase = state.settle_phase()
        if phase:
            count, index = state.claim(phase)
            self._write_control(0.0, phase, count, index)
            if self._settle_function is not None:
                self._settle_function(self._kernel_states, bool(phase & _OUTER_FIRST))
        label = state.close()
        if label is not None:
            self.finalize_time_step(label)

    def _write_control(self, weight: float, phase: int, count: int, index: int) -> None:
        values = {
            CONTROL_WEIGHT: weight,
            CONTROL_MACRO_STEPS: count,
            CONTROL_FLAGS: self._step_flags,
            CONTROL_PHASE: phase,
            CONTROL_MACRO_INDEX: index,
        }
        slots = self._control_host_slots
        if slots is None:
            for name, value in values.items():
                self._kernel_states[name].fill_(value)
            return
        host, host_values, event = slots[self._control_slot_index]
        if event is not None and not event.query():
            event.synchronize()
        for name, value in values.items():
            host_values[name][0] = value
        self._control_buffer.copy_(host, non_blocking=event is not None)
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
        self._control_slot_index = (self._control_slot_index + 1) % len(slots)

    def _convert_weight(self, weight: float) -> float:
        """Convert a trusted schedule weight while preserving overflow errors."""

        dtype = self._kernel_states[CONTROL_WEIGHT].dtype
        converted = _control_float_value(dtype, weight)
        if not math.isfinite(converted):
            raise OverflowError(f"statistics weight {weight!r} exceeds {dtype} range")
        if converted == 0.0:
            raise OverflowError(f"statistics weight {weight!r} underflows {dtype}")
        return converted

    def accumulator(self, variable: str, operation: str) -> torch.Tensor:
        """Return an ownership-isolated differentiable accumulator snapshot."""
        return self._storage[StoragePlan.output(variable, operation)].clone(
            memory_format=torch.preserve_format,
        )

    def get_time_index(self) -> int:
        return self._current_time_index

    def reset_time_index(self) -> None:
        self.sink.reset()
        self._current_time_index = 0
        self._last_time_number = None

    def _validate_next_time(
        self,
        dt: datetime | cftime.datetime,
    ) -> float:
        value = float(
            cftime.date2num(
                dt,
                units=TIME_UNITS,
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
        dirty = self.window_state.dirty
        keys = [key for key in self._output_keys if key in dirty]
        self.sink.append(dt, {key: self._storage[key] for key in keys})
        dirty.difference_update(keys)
        self._current_time_index += 1
        self._last_time_number = time_number

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
            scatter_extents=self.installation.scatter_extents,
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
        self._publish_launch()

    def content_requires_rebind(self, tensors) -> bool:
        """Address-stable index edits invalidate cold Metal CSR topology too."""
        if self.backend.name != "metal":
            return False
        sources = {
            self._tensor_registry[scatter.source.index].untyped_storage()._cdata
            for scatter in self._statistics_ir.ordered_scatters()
            if self._storage[StoragePlan.scatter_buffer(scatter.name)].dtype
            == torch.float64
        }
        return any(tensor.untyped_storage()._cdata in sources for tensor in tensors)

    def prepare_selection_update(
        self, replacements: Mapping[int, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Rebind selections against staged coordinates before mutation.

        Saved IDs and their order belong to the installed output schema. Only
        their position in the model's current coordinate may change.
        """

        updates = []
        for coordinate, selected, installed in self.installation.selection_sources:
            if id(coordinate) not in replacements and id(selected) not in replacements:
                continue
            candidate = replacements.get(id(selected), selected)
            if not torch.equal(candidate, selected):
                raise ValueError(
                    "structural updates cannot change installed output selection IDs"
                )
            fresh = find_indices_in_torch(
                candidate, replacements.get(id(coordinate), coordinate)
            ).to(installed.device)
            updates.append((installed, fresh))
        return tuple(updates)

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
        )
        result = compile_statistics_program(context, ir, dialect=self.backend.dialect)
        try:
            self._prepare_kernel_states(result.lowering.ir)
            self._cleanup_generated_modules()
        except BaseException:
            with cleanup_on_exit(
                "uninstalled statistics program",
                (
                    result.close,
                    *(
                        partial(release_generated_module, name, filename)
                        for name, filename in result.generated_modules
                    ),
                ),
            ):
                raise
        self._statistics_layouts = layouts
        self._statistics_ir = result.lowering.ir
        self._statistics_lowering = result.lowering
        self._generated_modules = list(result.generated_modules)
        self._aggregator_function = result.function
        self._request_builder = result.requests
        self._settle_function = result.settle
        self._kernel_module = result.module
        self._saved_kernel_file = result.saved_kernel_file
        self._close_program = result.close

    def _cleanup_generated_modules(self) -> None:
        modules, self._generated_modules = self._generated_modules, []
        self._kernel_module = None
        close, self._close_program = self._close_program, lambda: None
        with cleanup_on_exit(
            "statistics generated modules",
            (
                close,
                *(
                    partial(release_generated_module, name, filename)
                    for name, filename in reversed(modules)
                ),
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
        with cleanup_on_exit(
            "statistics runtime",
            (
                self._unregister_atexit,
                self._cleanup_generated_modules,
                *((self.sink.close,) if self.sink is not None else ()),
            ),
        ):
            pass

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
        values = tensor.detach().to(device="cpu", copy=True).numpy()
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
            scatter_extents=installation.scatter_extents,
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

        self._last_time_number = None

        # Initialize single time step aggregation (generic)
        self._materialize_compilation(compilation)

        if self.in_memory:
            self.sink = MemorySink(
                {
                    key: (
                        tuple(self._storage[key].shape),
                        saved_dtype(self._storage[key].dtype, self.save_precision),
                    )
                    for key in self._output_keys
                },
                device=self.result_device,
                encoded_outputs={
                    key
                    for key in self._output_keys
                    if getattr(self._storage[key], "encoding", None) == "float32x2"
                },
            )
            emit(
                self,
                "info",
                "statistics.memory_ready",
                "In-memory statistics aggregation initialized",
                outputs=len(self._output_keys),
            )
            return
        writer = RankOutputWriter(
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
            num_workers=self.num_workers,
            output_split_by_year=self.output_split_by_year,
            max_pending_steps=self.max_pending_steps,
            save_precision=self.save_precision,
            device=self.device,
            event_sink=self.event_sink,
            on_failure=self.on_write_failure,
        )
        self.sink = writer
        writer.start()
        emit(
            self,
            "info",
            "statistics.streaming_ready",
            "Streaming statistics aggregation initialized",
            executors=self.num_workers,
        )

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import atexit
import logging
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any
from uuid import uuid4

import cftime
import netCDF4 as nc
import numpy as np
import torch
from pydantic import validate_call

from hydroforge.contracts.errors import (
    ResourceCleanupError,
    cleanup_on_exit,
    failure_description,
)
from hydroforge.contracts.events import ModelEvent
from hydroforge.contracts.naming import sanitize_symbol
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.transfers import (
    _NonnegativeCount,
    _PendingCPUTransfer,
    _PositiveCount,
    _TensorCPUStager,
)
from hydroforge.output.conversion import _checked_narrowing, _raise_narrowing_failures
from hydroforge.output.netcdf.plan import (
    NetCDFCreateRequest,
    NetCDFWriteRequest,
    OutputFilePlan,
)
from hydroforge.output.netcdf.schema import NetCDFSchema
from hydroforge.serialization.files import fsync_directory, fsync_file
from hydroforge.serialization.netcdf import (
    BOOL_LOGICAL_DTYPE,
    COMMITTED_STEPS_ATTR,
    COORDINATE_ATTR,
    LOGICAL_DTYPE_ATTR,
    OUTPUT_FORMAT,
    OUTPUT_VERSION,
    RUN_ID_ATTR,
    _atomic_netcdf_dataset_trusted,
    _create_netcdf_variable_trusted,
    netcdf_dtype_encoding,
    start_blosc_zstd_probe,
)

logger = logging.getLogger(__name__)


_TORCH_NUMPY_DTYPES = {
    torch.bool: np.dtype(np.bool_),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
}


def _is_wsl() -> bool:
    import sys

    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/version", encoding="utf-8") as stream:
            version = stream.read().lower()
        return "microsoft" in version or "wsl" in version
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Default cap on per-submit IPC payload (bytes).  Each subprocess receives
# a pickled numpy array; keeping the payload bounded avoids excessive memory
# copies.  256 MB is a safe default for machines with ≥8 GB RAM.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_IPC_BYTES: int = 256 * 1024 * 1024
_DEFAULT_MAX_PENDING_OUTPUT_BYTES: int = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PendingNetCDFWrite:
    """One background task with exact timestep weights for every stream."""

    step_counts: tuple[tuple[str, int], ...]
    payload_bytes: int
    future: Any
    # Batch arrays still read by an in-flight pickling transport.
    buffers: tuple[_NetCDFWriteBuffer, ...] = ()


@dataclass(frozen=True, slots=True)
class _SharedNetCDFWrite:
    variable: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    output_path: Path
    times: tuple[Any, ...]


def _release_shared_output(memory: SharedMemory) -> None:
    def unlink() -> None:
        try:
            memory.unlink()
        except FileNotFoundError:
            pass

    with cleanup_on_exit("shared NetCDF output", (memory.close, unlink)):
        pass


def _release_output_when_done(future: Future, memory: SharedMemory) -> Future:
    """Publish completion only after shared-memory cleanup has also finished."""
    completed = Future()

    def finish(source: Future) -> None:
        try:
            with cleanup_on_exit(
                "shared NetCDF write", (lambda: _release_shared_output(memory),)
            ):
                result = source.result()
        except BaseException as error:
            completed.set_exception(error)
        else:
            completed.set_result(result)

    future.add_done_callback(finish)
    return completed


def _share_output_requests(requests: tuple[NetCDFWriteRequest, ...]):
    """Prepare bounded shared IPC, falling back to pickling before allocation.

    ``HYDROFORGE_SHARED_OUTPUT=0`` disables the shared-memory transport.
    """

    if os.environ.get("HYDROFORGE_SHARED_OUTPUT", "1") == "0":
        return None
    size = sum(request.data.nbytes for request in requests)
    if not 1024 * 1024 <= size <= _DEFAULT_MAX_IPC_BYTES:
        return None
    if os.path.isdir("/dev/shm"):
        status = os.statvfs("/dev/shm")
        if size * 2 > status.f_bavail * status.f_frsize:
            return None
    try:
        memory = SharedMemory(create=True, size=size)
    except OSError:
        return None
    descriptors = []
    offset = 0
    try:
        for request in requests:
            target = np.ndarray(
                request.data.shape,
                dtype=request.data.dtype,
                buffer=memory.buf,
                offset=offset,
            )
            np.copyto(target, request.data, casting="no")
            descriptors.append(
                _SharedNetCDFWrite(
                    request.variable,
                    request.data.shape,
                    request.data.dtype.str,
                    offset,
                    request.output_path,
                    request.times,
                )
            )
            offset += request.data.nbytes
            del target
    except BaseException:
        with cleanup_on_exit(
            "shared NetCDF preparation", (lambda: _release_shared_output(memory),)
        ):
            raise
    return memory, tuple(descriptors)


def _attach_shared_output(name: str) -> SharedMemory:
    """Attach without transferring unlink ownership from the submitting process."""

    if sys.version_info >= (3, 13):
        return SharedMemory(name=name, track=False)
    # Spawned writers share the submitter's tracker; unregistering here would
    # also remove the owner's registration before its unlink.
    return SharedMemory(name=name)


def _write_shared_netcdf_group(name: str, descriptors: tuple[_SharedNetCDFWrite, ...]):
    memory = _attach_shared_output(name)
    requests = ()
    with cleanup_on_exit("attached shared NetCDF output", (memory.close,)):
        try:
            requests = tuple(
                NetCDFWriteRequest(
                    item.variable,
                    np.ndarray(
                        item.shape,
                        dtype=np.dtype(item.dtype),
                        buffer=memory.buf,
                        offset=item.offset,
                    ),
                    item.output_path,
                    item.times,
                )
                for item in descriptors
            )
            return _write_netcdf_group_process(requests)
        finally:
            requests = ()


@dataclass(frozen=True, slots=True)
class _NetCDFOutputStream:
    """One compiled route from a statistics output to a NetCDF writer."""

    key: str
    path: Path
    batch_size: int
    component: int | None = None
    executor_index: int | None = None


@dataclass(slots=True)
class _NetCDFWriteBuffer:
    """One fixed-capacity contiguous batch owned by the submitting process."""

    stream: _NetCDFOutputStream
    data: np.ndarray
    count: int
    times: list[Any]

    @classmethod
    def allocate(
        cls,
        stream: _NetCDFOutputStream,
        row: np.ndarray,
        *,
        max_pending_steps: int,
        spare: np.ndarray | None = None,
    ) -> _NetCDFWriteBuffer:
        capacity = min(stream.batch_size, max_pending_steps)
        shape = (capacity, *row.shape)
        if spare is not None and spare.shape == shape and spare.dtype == row.dtype:
            data = spare
        else:
            data = np.empty(shape, dtype=row.dtype)
        return cls(stream=stream, data=data, count=0, times=[])

    @property
    def payload_bytes(self) -> int:
        return self.count * int(self.data[0].nbytes)

    @property
    def allocated_bytes(self) -> int:
        return int(self.data.nbytes)

    def append(self, row: np.ndarray, dt: Any) -> None:
        if self.count >= len(self.data):
            raise RuntimeError(
                f"NetCDF buffer for {self.stream.key!r} exceeded its capacity"
            )
        if row.dtype != self.data.dtype or row.shape != self.data.shape[1:]:
            raise TypeError(
                f"NetCDF buffer row for {self.stream.key!r} changed dtype or shape"
            )
        np.copyto(self.data[self.count], row, casting="no")
        self.count += 1
        self.times.append(dt)

    def request_data(self) -> np.ndarray:
        return self.data[: self.count]


_DEFAULT_MAX_BATCH: int = 30

# Stager key of the stacked device-side narrowing flags of one output step.
_NARROWING_FLAGS = "\0narrowing"


@dataclass(slots=True)
class _StagedOutputStep:
    """One output step whose device-to-host copy may still be in flight."""

    dt: Any
    keys: tuple[str, ...]
    transfer: _PendingCPUTransfer
    flags: tuple[tuple[torch.Tensor, str, str], ...]


@validate_call(config=HydroForgeModel.model_config)
def compute_write_batch_size(
    saved_points: _NonnegativeCount,
    dtype_bytes: _PositiveCount = 4,
    max_ipc_bytes: _PositiveCount = _DEFAULT_MAX_IPC_BYTES,
    max_batch: _PositiveCount = _DEFAULT_MAX_BATCH,
) -> int:
    """Return the number of time steps to batch per subprocess write.

    The batch is capped so that ``batch * saved_points * dtype_bytes``
    does not exceed *max_ipc_bytes*, reducing the batch to one step for very
    large grids (e.g. glb_01min).
    """
    per_step = saved_points * dtype_bytes
    batch = max_ipc_bytes // max(per_step, 1)
    return max(1, min(batch, max_batch))


@validate_call(config=HydroForgeModel.model_config)
def constrain_write_batch_sizes(
    desired: Mapping[str, _PositiveCount],
    *,
    row_bytes: Mapping[str, _NonnegativeCount],
    stream_counts: Mapping[str, _NonnegativeCount],
    max_pending_bytes: _PositiveCount,
) -> dict[str, int]:
    """Fit retained output memory into one process-wide byte budget.

    Every stream may hold a filling batch, one batch in flight and one
    reusable spare, and each output row is staged twice by the double-buffered
    device-to-host copy.
    """

    batches = dict(desired)
    total = sum(
        row_bytes[name] * stream_counts[name] * (2 + 3 * batch)
        for name, batch in batches.items()
    )
    while total > max_pending_bytes:
        candidates = [name for name, batch in batches.items() if batch > 1]
        if not candidates:
            break
        name = max(
            candidates,
            key=lambda item: (
                row_bytes[item] * stream_counts[item] * batches[item],
                item,
            ),
        )
        old = batches[name]
        new = max(1, old // 2)
        batches[name] = new
        total -= 3 * row_bytes[name] * stream_counts[name] * (old - new)
    return batches


_WORKER_FILE_CACHE_SIZE = 32
_WORKER_NETCDF_FILES: OrderedDict[
    Path,
    tuple[nc.Dataset, tuple[int, int]],
] = OrderedDict()


def _worker_file_identity(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino


def _close_worker_netcdf_files() -> None:
    """Close every Dataset cached inside one writer subprocess."""

    entries = tuple(_WORKER_NETCDF_FILES.values())
    _WORKER_NETCDF_FILES.clear()
    failures: list[BaseException] = []
    for dataset, _identity in entries:
        try:
            dataset.close()
        except BaseException as error:
            failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ResourceCleanupError("NetCDF worker file cache", failures)


def _initialize_netcdf_worker() -> None:
    """Install deterministic cleanup in one spawned output process."""

    atexit.register(_close_worker_netcdf_files)


def _evict_worker_netcdf_file(path: Path) -> None:
    entry = _WORKER_NETCDF_FILES.pop(path, None)
    if entry is not None:
        entry[0].close()


def _close_worker_netcdf_paths(paths: tuple[Path, ...]) -> None:
    """Release cached append handles of files that receive no further rows."""

    failures: list[BaseException] = []
    for path in paths:
        try:
            _evict_worker_netcdf_file(path.absolute())
        except BaseException as error:
            failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ResourceCleanupError("NetCDF worker file release", failures)


def _cached_worker_netcdf_file(path: Path) -> nc.Dataset:
    """Return an append handle, reopening if the path was externally replaced."""

    canonical = path.absolute()
    identity = _worker_file_identity(canonical)
    entry = _WORKER_NETCDF_FILES.pop(canonical, None)
    if entry is not None:
        dataset, cached_identity = entry
        if cached_identity == identity:
            _WORKER_NETCDF_FILES[canonical] = entry
            return dataset
        dataset.close()
    dataset = nc.Dataset(canonical, "a")
    _WORKER_NETCDF_FILES[canonical] = (dataset, identity)
    while len(_WORKER_NETCDF_FILES) > _WORKER_FILE_CACHE_SIZE:
        _old_path, (old_dataset, _old_identity) = _WORKER_NETCDF_FILES.popitem(
            last=False
        )
        old_dataset.close()
    return dataset


def _find_data_variable(ncfile, var_name: str):
    """Locate the target data variable inside an open NetCDF dataset."""
    safe = sanitize_symbol(var_name)
    if var_name in ncfile.variables:
        return var_name
    if safe in ncfile.variables:
        return safe
    raise KeyError(
        f"Could not find variable for '{var_name}' (safe: '{safe}') in {ncfile.filepath()}"
    )


def _wsl_drop_cache(output_path) -> None:
    """WSL optimisation: advise the kernel to drop page-cache for *output_path*."""
    if _is_wsl() and hasattr(os, "posix_fadvise"):
        try:
            with open(output_path, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _append_netcdf_request(
    ncfile: nc.Dataset,
    request: NetCDFWriteRequest,
) -> tuple[str, int]:
    """Append one batch through an already-open validated Dataset handle."""

    time_var = ncfile.variables["time"]
    target = _find_data_variable(ncfile, request.variable)
    variable = ncfile.variables[target]
    if time_var.dimensions != ("time",):
        raise ValueError(
            f"time variable in {request.output_path} must have dimensions ('time',)"
        )
    if not variable.dimensions or variable.dimensions[0] != "time":
        raise ValueError(
            f"NetCDF variable {target!r} in {request.output_path} must "
            "start with the time dimension"
        )
    logical_dtype = getattr(variable, LOGICAL_DTYPE_ATTR, None)
    expected_dtype = (
        np.dtype(np.bool_)
        if logical_dtype == BOOL_LOGICAL_DTYPE
        else np.dtype(variable.dtype)
    )
    if request.data.dtype != expected_dtype:
        raise TypeError(
            f"NetCDF write batch for {request.variable!r} has dtype "
            f"{request.data.dtype}, expected exact dtype {expected_dtype}"
        )
    expected_row_shape = tuple(variable.shape[1:])
    observed_row_shape = tuple(request.data.shape[1:])
    if observed_row_shape != expected_row_shape:
        raise ValueError(
            f"NetCDF write rows for {request.variable!r} have shape "
            f"{observed_row_shape}, expected {expected_row_shape}"
        )
    if len(request.data) != len(request.times) or not request.times:
        raise ValueError(
            "NetCDF write batch must contain matching non-empty data and times"
        )
    committed = ncfile.getncattr(COMMITTED_STEPS_ATTR)
    if isinstance(committed, (bool, np.bool_)) or not isinstance(
        committed,
        (int, np.integer),
    ):
        raise TypeError(
            f"{COMMITTED_STEPS_ATTR} must be an integer in {request.output_path}"
        )
    current_len = int(committed)
    physical_len = len(time_var)
    if current_len < 0 or len(variable) != physical_len:
        raise ValueError(
            f"invalid NetCDF append lengths in {request.output_path}: "
            f"committed={current_len}, time={physical_len}, "
            f"data={len(variable)}"
        )
    if physical_len != current_len:
        raise RuntimeError(
            f"NetCDF output {request.output_path} contains an "
            f"uncommitted append tail: committed={current_len}, "
            f"physical={physical_len}"
        )
    units = time_var.getncattr("units")
    calendar = time_var.getncattr("calendar")
    numeric_times = np.asarray(
        nc.date2num(
            request.times,
            units=units,
            calendar=calendar,
        )
    )
    if not np.isfinite(numeric_times).all():
        raise ValueError("NetCDF write timestamps must be finite datetimes")
    if len(numeric_times) > 1 and np.any(np.diff(numeric_times) <= 0):
        raise ValueError("NetCDF write timestamps must be strictly increasing")
    if current_len:
        previous = time_var[current_len - 1]
        if np.ma.is_masked(previous) or not np.isfinite(previous):
            raise ValueError(
                f"last committed timestamp in {request.output_path} is invalid"
            )
        if numeric_times[0] <= previous:
            raise ValueError(
                "NetCDF write timestamps must be strictly increasing across batches"
            )
    committed_len = current_len + len(request.times)
    variable[current_len:committed_len, ...] = request.data
    time_var[current_len:committed_len] = numeric_times
    ncfile.sync()
    ncfile.setncattr(COMMITTED_STEPS_ATTR, committed_len)
    ncfile.sync()
    _wsl_drop_cache(request.output_path)
    return request.variable, committed_len - 1


def _write_netcdf_process(request: NetCDFWriteRequest) -> tuple[str, int]:
    """Append one already-batched request in a single file transaction."""

    with nc.Dataset(request.output_path, "a") as ncfile:
        return _append_netcdf_request(ncfile, request)


def _write_netcdf_group_process(
    requests: tuple[NetCDFWriteRequest, ...],
) -> tuple[tuple[str, int], ...]:
    """Write several streams through worker-local persistent file handles."""

    results = []
    for request in requests:
        path = request.output_path.absolute()
        try:
            dataset = _cached_worker_netcdf_file(path)
            results.append(_append_netcdf_request(dataset, request))
        except BaseException:
            try:
                _evict_worker_netcdf_file(path)
            except BaseException:
                logger.exception(
                    "failed to evict NetCDF worker handle after append failure: %s",
                    path,
                )
            raise
    return tuple(results)


def _create_netcdf_file_process(
    request: NetCDFCreateRequest,
) -> Path | list[Path]:
    """Create one empty NetCDF file per declared output component."""
    mean_var_name = request.variable
    schema = request.schema
    coord_values = request.coordinate_values
    output_dir = request.output_dir
    rank = request.rank
    world_size = request.world_size
    year = request.year
    calendar = request.calendar
    time_unit = request.time_unit
    static_vars = request.static_variables
    run_id = str(uuid4()) if request.run_id is None else request.run_id

    safe_name = sanitize_symbol(mean_var_name)
    tensor_shape = schema.tensor_shape
    # nc_coord_name is derived from dim_coords (e.g. "catchment_id").
    coord_name = schema.coordinate_name
    k_val = schema.order
    dtype = schema.dtype
    file_actual_shape = schema.file_actual_shape

    # Helper to create a single NetCDF file
    def create_single_file(
        file_safe_name: str, file_var_name: str, description_suffix: str = ""
    ) -> Path:
        output_path = OutputFilePlan(
            directory=output_dir,
            variable=file_safe_name,
            rank=rank,
            year=year,
        ).path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with _atomic_netcdf_dataset_trusted(
            output_path,
            format="NETCDF4",
        ) as ncfile:
            # Write global attributes
            ncfile.setncattr("title", f"Time series for rank {rank}: {file_var_name}")
            ncfile.setncattr("original_variable_name", file_var_name)
            ncfile.setncattr("hydroforge_output_format", OUTPUT_FORMAT)
            ncfile.setncattr("hydroforge_output_version", OUTPUT_VERSION)
            ncfile.setncattr("hydroforge_rank", rank)
            ncfile.setncattr("hydroforge_world_size", world_size)
            ncfile.setncattr(RUN_ID_ATTR, run_id)
            ncfile.setncattr(COMMITTED_STEPS_ATTR, 0)
            has_coordinate = bool(coord_name) and coord_values is not None
            ncfile.setncattr(COORDINATE_ATTR, coord_name if has_coordinate else "")

            # Create time dimension (unlimited for streaming)
            ncfile.createDimension("time", None)

            for dimension, extent in schema.dimensions:
                ncfile.createDimension(dimension, extent)

            if schema.batched:
                member_ids = request.ensemble_member_ids
                if member_ids is None:
                    member_ids = tuple(range(dict(schema.dimensions)["ensemble"]))
                ensemble_coordinate = ncfile.createVariable(
                    "ensemble", "i8", ("ensemble",)
                )
                ensemble_coordinate[:] = member_ids

            if has_coordinate:
                coord_var = ncfile.createVariable(
                    coord_name,
                    coord_values.dtype,
                    ("saved_points",),
                )
                coord_var[:] = coord_values

            # Write user-supplied static per-point variables. Coordinate scope
            # decides applicability; once applicable, dimension mismatch is a
            # schema error rather than a silent omission.
            for sv_name, sv_spec in static_vars.items():
                sv_dim = sv_spec["dim"]
                if sv_spec["coordinate"] != schema.output_coordinate:
                    continue
                sv_values = sv_spec["values"]
                storage_dtype, logical_dtype = netcdf_dtype_encoding(
                    sv_values.dtype,
                )
                sv_var = ncfile.createVariable(
                    sv_name,
                    storage_dtype,
                    (sv_dim,),
                )
                if logical_dtype is not None:
                    sv_var.setncattr(LOGICAL_DTYPE_ATTR, logical_dtype)
                sv_var[:] = sv_values
                for ak, av in sv_spec["attrs"].items():
                    sv_var.setncattr(ak, av)

            time_var = ncfile.createVariable("time", "f8", ("time",))
            time_var.setncattr("units", time_unit)
            time_var.setncattr("calendar", calendar)

            # Create single data variable
            nc_var = _create_netcdf_variable_trusted(
                ncfile,
                file_safe_name,
                dtype,
                schema.data_dimensions,
                options=schema.create_options,
            )
            if schema.logical_dtype is not None:
                nc_var.setncattr(LOGICAL_DTYPE_ATTR, schema.logical_dtype)
            desc = schema.description + description_suffix
            nc_var.setncattr("description", desc)
            nc_var.setncattr("actual_shape", str(file_actual_shape))
            nc_var.setncattr("tensor_shape", str(tensor_shape))
            nc_var.setncattr("long_name", file_var_name)

        return output_path

    # For k > 1, create separate files for each k index
    if k_val > 1:
        paths = []
        for k_idx in range(k_val):
            file_safe_name = f"{safe_name}_{k_idx}"
            file_var_name = f"{mean_var_name}_{k_idx}"
            desc_suffix = f" [rank {k_idx}]"
            path = create_single_file(file_safe_name, file_var_name, desc_suffix)
            paths.append(path)
        return paths
    return create_single_file(safe_name, mean_var_name)


class _NetCDFWriter:
    """Own file output, bounded staging, background workers and their lifetime."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, Mapping[str, Any]],
        coordinates: Mapping[str, np.ndarray],
        static_vars: Mapping[str, Mapping[str, Any]],
        variable_options: Mapping[str, Mapping[str, Any]],
        output_dir: Path,
        rank: int,
        world_size: int,
        ensemble_size: int,
        ensemble_member_ids: tuple[int, ...] | None,
        calendar: str,
        time_unit: str,
        num_workers: int,
        output_split_by_year: bool,
        max_pending_steps: int,
        max_pending_output_bytes: int,
        save_precision: torch.dtype | None,
        event_sink: Any,
        run_id: str | None = None,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.ensemble_member_ids = ensemble_member_ids
        self.calendar = calendar
        self.time_unit = time_unit
        self.num_workers = num_workers
        self.output_split_by_year = output_split_by_year
        self.max_pending_steps = max_pending_steps
        self.max_pending_output_bytes = max_pending_output_bytes
        self.save_precision = save_precision
        self.event_sink = event_sink
        self.run_id = run_id
        self._on_failure = on_failure
        self.static_vars = static_vars
        self._coord_cache = coordinates
        self._closed = False
        self._current_year = None
        self._files_created = False
        self._netcdf_files: dict[str, Path | list[Path]] = {}
        self._all_created_files: set[Path] = set()
        self._output_streams: dict[str, tuple[_NetCDFOutputStream, ...]] = {}
        self._write_executors: list[ProcessPoolExecutor] = []
        self._pending_writes: list[PendingNetCDFWrite] = []
        self._write_buffers: dict[str, _NetCDFWriteBuffer] = {}
        self._background_failure: BaseException | None = None
        self._spare_buffers: dict[str, np.ndarray] = {}
        self._unsynced_paths: set[Path] = set()
        self._cpu_stager = _TensorCPUStager(max_pending_output_bytes // 2)
        self._staged_step: _StagedOutputStep | None = None
        desired_batches: dict[str, int] = {}
        row_bytes: dict[str, int] = {}
        stream_counts: dict[str, int] = {}
        for name, info in metadata.items():
            order = info["k"]
            row_shape = info["actual_shape"][:-1] if order > 1 else info["actual_shape"]
            dtype = info["dtype"]
            if save_precision is not None and dtype.is_floating_point:
                dtype = save_precision
            row_bytes[name] = max(1, math.prod(row_shape) * dtype.itemsize)
            stream_counts[name] = order
            desired_batches[name] = compute_write_batch_size(
                max(1, math.prod(row_shape)),
                dtype.itemsize,
                max_batch=min(30, max_pending_steps),
            )
        batches = constrain_write_batch_sizes(
            desired_batches,
            row_bytes=row_bytes,
            stream_counts=stream_counts,
            max_pending_bytes=max_pending_output_bytes,
        )
        self._netcdf_schemas = {
            name: NetCDFSchema.compile(
                info,
                variable=name,
                ensemble_size=ensemble_size,
                netcdf_options=variable_options[name],
                write_batch_size=batches[name],
                save_precision=save_precision,
            )
            for name, info in metadata.items()
        }
        try:
            start_blosc_zstd_probe()
        except (OSError, subprocess.SubprocessError):
            logger.warning("could not start the Blosc capability probe early")

    def reset_staging(self) -> None:
        self._drain_staged_step()
        self._cpu_stager.clear()
        self._spare_buffers.clear()

    def release_staging(self) -> None:
        """Wait for any in-flight copy, then release pinned staging memory."""

        staged, self._staged_step = self._staged_step, None
        try:
            if staged is not None:
                staged.transfer.wait()
        finally:
            self._cpu_stager.clear()

    def sync_appended_files(self) -> None:
        """Fsync every file appended since the last durability boundary."""

        failures: list[BaseException] = []
        for path in sorted(self._unsynced_paths):
            try:
                fsync_file(path)
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("NetCDF output fsync", failures)
        self._unsynced_paths.clear()

    def _emit_event(self, event: ModelEvent) -> None:
        """Keep observability failures outside the output transaction."""

        try:
            self.event_sink.emit(event)
        except Exception:
            logger.exception(
                "statistics event sink failed while emitting %s",
                event.name,
            )

    def _raise_if_background_failed(self) -> None:
        failure = self._background_failure
        if failure is not None:
            raise failure

    def _latch_failures(
        self,
        label: str,
        failures: list[BaseException],
    ) -> None:
        if not failures:
            return
        if len(failures) == 1:
            failure = failures[0]
        else:
            failure = ResourceCleanupError(label, failures)
        if self._background_failure is None:
            self._background_failure = failure
            if self._on_failure is not None:
                self._on_failure(failure)
        raise self._background_failure

    def _resolve_run_id(self) -> str:
        """Return one identity shared by every file in this output run."""

        if self.run_id is None:
            if self.world_size != 1:
                raise RuntimeError(
                    "multi-rank statistics output run ID was not installed by the "
                    "rank-synchronous runtime materialization transaction"
                )
            self.run_id = str(uuid4())
        return self.run_id

    def _create_netcdf_files(self, year: int | None = None) -> None:
        """Create empty NetCDF files with proper structure for streaming.

        Creation writes headers only (~5 ms per output), so it runs inline; a
        spawned pool cost ~1.6 s of interpreter start-up on the first step.
        """
        if not self.output_split_by_year and self._files_created:
            return

        self._raise_if_background_failed()
        self._emit_event(
            ModelEvent(
                level="info",
                name="output.create_start",
                message="Creating NetCDF file structure",
                fields={"year": year},
            )
        )

        # Resolve once and retain across every variable, rank, and split year.
        run_id = self._resolve_run_id()

        # Plan and validate the complete transaction before mutating any final
        # path.  Every file is first created in a sibling staging directory.
        requests: list[NetCDFCreateRequest] = []
        planned_by_variable: dict[str, tuple[Path, ...]] = {}
        planned_paths: list[Path] = []
        for out_name in self._netcdf_schemas:
            schema = self._netcdf_schemas[out_name]
            coord_name = schema.output_coordinate
            coord_values = None if coord_name is None else self._coord_cache[coord_name]
            request = NetCDFCreateRequest(
                variable=out_name,
                schema=schema,
                coordinate_values=coord_values,
                output_dir=self.output_dir,
                rank=self.rank,
                world_size=self.world_size,
                year=year,
                calendar=self.calendar,
                time_unit=self.time_unit,
                static_variables=self.static_vars,
                run_id=run_id,
                ensemble_member_ids=self.ensemble_member_ids,
            )
            requests.append(request)
            safe_name = sanitize_symbol(out_name)
            order = schema.order
            names = (
                tuple(f"{safe_name}_{index}" for index in range(order))
                if order > 1
                else (safe_name,)
            )
            output_paths = tuple(
                OutputFilePlan(
                    directory=self.output_dir,
                    variable=name,
                    rank=self.rank,
                    year=year,
                ).path
                for name in names
            )
            planned_by_variable[out_name] = output_paths
            planned_paths.extend(output_paths)
        # Compile each output route up front before mutating final paths.
        output_streams: dict[str, tuple[_NetCDFOutputStream, ...]] = {}
        batch_events: list[ModelEvent] = []
        executor_count = len(self._write_executors)
        stream_index = 0
        for out_name in self._netcdf_schemas:
            schema = self._netcdf_schemas[out_name]
            k_val = schema.order
            elements = max(1, math.prod(schema.file_actual_shape))
            element_size = np.dtype(schema.dtype).itemsize
            batch_size = schema.write_batch_size
            paths = planned_by_variable[out_name]
            streams: list[_NetCDFOutputStream] = []
            for component, path in enumerate(paths):
                key = f"{out_name}_{component}" if k_val > 1 else out_name
                streams.append(
                    _NetCDFOutputStream(
                        key=key,
                        path=path,
                        batch_size=batch_size,
                        component=component if k_val > 1 else None,
                        executor_index=(
                            stream_index % executor_count if executor_count else None
                        ),
                    )
                )
                stream_index += 1
            output_streams[out_name] = tuple(streams)
            batch_events.append(
                ModelEvent(
                    level="info",
                    name="output.batch_configured",
                    message="Configured NetCDF write batch",
                    fields={
                        "output": out_name,
                        "steps": batch_size,
                        "elements_per_step": elements,
                        "storage_bytes_per_element": element_size,
                    },
                )
            )

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stage_root = Path(
            tempfile.mkdtemp(
                prefix=".hydroforge-output-",
                dir=output_dir,
            )
        )
        stage_output = stage_root / "new"
        backup_dir = stage_root / "old"
        staged_results: dict[str, Path | list[Path]] = {}
        moves: list[tuple[Path, Path]] = []
        preserve_staging = False
        try:
            stage_output.mkdir()
            backup_dir.mkdir()
            for request in requests:
                expected = planned_by_variable[request.variable]
                _create_netcdf_file_process(
                    replace(
                        request,
                        output_dir=stage_output,
                    )
                )
                expected_staged = [
                    stage_output / final_path.name for final_path in expected
                ]
                if any(not path.is_file() for path in expected_staged):
                    raise RuntimeError(
                        f"NetCDF creation for {request.variable!r} did not "
                        "materialize every staged file"
                    )
                staged_results[request.variable] = (
                    list(expected) if len(expected) > 1 else expected[0]
                )
                moves.extend(zip(expected_staged, expected, strict=True))
            for _source, target in moves:
                if (
                    os.path.lexists(target)
                    and target.is_dir()
                    and not target.is_symlink()
                ):
                    raise IsADirectoryError(
                        f"NetCDF output path is a directory: {target}"
                    )
            for source, target in moves:
                if os.path.lexists(target) and not target.is_symlink():
                    source.chmod(stat.S_IMODE(target.stat().st_mode))

            backups: list[tuple[Path, Path]] = []
            installed: list[Path] = []
            try:
                for index, (source, target) in enumerate(moves):
                    if os.path.lexists(target):
                        backup = backup_dir / f"{index}.nc"
                        os.replace(target, backup)
                        backups.append((target, backup))
                    os.replace(source, target)
                    installed.append(target)
                fsync_directory(output_dir)
            except BaseException as primary:
                rollback_failures: list[BaseException] = []
                for target in reversed(installed):
                    try:
                        target.unlink(missing_ok=True)
                    except BaseException as error:
                        rollback_failures.append(error)
                for target, backup in reversed(backups):
                    try:
                        os.replace(backup, target)
                    except BaseException as error:
                        rollback_failures.append(error)
                if rollback_failures:
                    preserve_staging = True
                    error = ResourceCleanupError(
                        "NetCDF file creation rollback; recovery files "
                        f"retained at {stage_root}",
                        [primary, *rollback_failures],
                    )
                    raise error from primary
                raise
        finally:
            if not preserve_staging:
                try:
                    shutil.rmtree(stage_root)
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.exception(
                        "failed to remove NetCDF transaction staging directory %s",
                        stage_root,
                    )

        # The data transaction is now complete.  Publish owner state in one
        # non-I/O section, then report telemetry on a best-effort basis.
        self._netcdf_files.update(staged_results)
        self._all_created_files.update(planned_paths)
        self._write_buffers.clear()
        self._output_streams = output_streams
        self._files_created = True

        for event in batch_events:
            self._emit_event(event)
        for path in planned_paths:
            self._emit_event(
                ModelEvent(
                    level="info",
                    name="output.file_created",
                    message="Created NetCDF file",
                    fields={"path": str(path)},
                )
            )
        total_files = sum(map(len, output_streams.values()))
        self._emit_event(
            ModelEvent(
                level="info",
                name="output.create_complete",
                message="Created NetCDF files for streaming",
                fields={"files": total_files},
            )
        )

    def _partition_write_groups(
        self,
        keys: list[str],
    ) -> tuple[tuple[str, ...], ...]:
        """Group small buffers by executor without exceeding the IPC cap."""

        grouped: dict[int | None, list[list[Any]]] = {}
        for key in keys:
            buffer = self._write_buffers.get(key)
            if buffer is None or not buffer.count:
                continue
            executor_index = buffer.stream.executor_index
            payload_bytes = buffer.payload_bytes
            partitions = grouped.setdefault(executor_index, [])
            if not partitions or (
                partitions[-1][0]
                and partitions[-1][1] + payload_bytes > _DEFAULT_MAX_IPC_BYTES
            ):
                partitions.append([[], 0])
            partitions[-1][0].append(key)
            partitions[-1][1] += payload_bytes
        return tuple(
            tuple(partition[0])
            for partitions in grouped.values()
            for partition in partitions
        )

    def _submit_write_group(self, keys: tuple[str, ...]) -> None:
        """Submit multiple buffered streams as one worker task."""

        self._raise_if_background_failed()
        if not keys:
            return

        requests: list[NetCDFWriteRequest] = []
        step_counts: list[tuple[str, int]] = []
        buffers: list[_NetCDFWriteBuffer] = []
        buffer_keys: list[str] = []
        executor_index: int | None = None
        for key in keys:
            buffer = self._write_buffers.get(key)
            if buffer is None or not buffer.count:
                continue
            stream = buffer.stream
            if not buffers:
                executor_index = stream.executor_index
            elif stream.executor_index != executor_index:
                raise RuntimeError(
                    "one NetCDF write group cannot span writer executors"
                )
            times = tuple(buffer.times)
            requests.append(
                NetCDFWriteRequest(
                    variable=stream.key,
                    data=buffer.request_data(),
                    output_path=stream.path,
                    times=times,
                )
            )
            step_counts.append((key, len(times)))
            buffers.append(buffer)
            buffer_keys.append(key)
        if not requests:
            return

        request_group = tuple(requests)
        self._unsynced_paths.update(request.output_path for request in request_group)
        in_flight: tuple[_NetCDFWriteBuffer, ...] = ()
        if executor_index is None:
            for request in request_group:
                _write_netcdf_process(request)
        else:
            executor = self._write_executors[executor_index]
            shared = _share_output_requests(request_group)
            if shared is None:
                future = executor.submit(_write_netcdf_group_process, request_group)
                in_flight = tuple(buffers)
            else:
                memory, descriptors = shared
                try:
                    future = executor.submit(
                        _write_shared_netcdf_group, memory.name, descriptors
                    )
                except BaseException:
                    with cleanup_on_exit(
                        "shared NetCDF submission",
                        (lambda: _release_shared_output(memory),),
                    ):
                        raise
                future = _release_output_when_done(future, memory)
            self._pending_writes.append(
                PendingNetCDFWrite(
                    step_counts=tuple(step_counts),
                    payload_bytes=sum(buffer.payload_bytes for buffer in buffers),
                    future=future,
                    buffers=in_flight,
                )
            )

        for key, buffer in zip(buffer_keys, buffers, strict=True):
            if self._write_buffers.get(key) is buffer:
                self._write_buffers.pop(key)
        if not in_flight:
            self._recycle_buffers(buffers)

    def _recycle_buffers(self, buffers) -> None:
        """Keep one no-longer-referenced batch array per stream for reuse."""

        for buffer in buffers:
            self._spare_buffers[buffer.stream.key] = buffer.data

    def _release_worker_files(self, paths: tuple[Path, ...]) -> None:
        """Close completed files in every worker after their queued appends."""

        failures: list[BaseException] = []
        for executor in self._write_executors:
            try:
                future = executor.submit(_close_worker_netcdf_paths, paths)
            except BaseException as error:
                failures.append(error)
                continue
            self._pending_writes.append(
                PendingNetCDFWrite(step_counts=(), payload_bytes=0, future=future)
            )
        self._latch_failures("NetCDF worker file release", failures)

    def _flush_write_buffers(self, keys: list[str]) -> None:
        """Flush selected buffers, aggregating submissions where practical."""

        failures: list[BaseException] = []
        for group in self._partition_write_groups(keys):
            try:
                self._submit_write_group(group)
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("NetCDF write buffer groups", failures)

    def _flush_ready_write_buffers(self) -> None:
        """Flush every buffer that reached its effective time batch size."""

        ready = [
            key
            for key, buffer in self._write_buffers.items()
            if buffer.count
            >= min(
                buffer.stream.batch_size,
                self.max_pending_steps,
            )
        ]
        self._flush_write_buffers(ready)

    def _flush_all_write_buffers(self, *, latch: bool = True) -> None:
        """Flush every pending write buffer (called on year transition / shutdown)."""
        self._raise_if_background_failed()
        try:
            self._drain_staged_step()
            self._flush_write_buffers(list(self._write_buffers))
        except BaseException as error:
            if latch:
                self._latch_failures("NetCDF write buffers", [error])
            raise

    def _buffer_and_maybe_flush(
        self,
        stream: _NetCDFOutputStream,
        data: np.ndarray,
        dt: datetime | cftime.datetime,
    ) -> None:
        """Append one time step; ready buffers are flushed as one group."""
        self._raise_if_background_failed()
        if stream.key not in self._write_buffers:
            capacity = min(stream.batch_size, self.max_pending_steps)
            self._reserve_output_bytes(stream.key, capacity * data.nbytes, dt=dt)
            self._write_buffers[stream.key] = _NetCDFWriteBuffer.allocate(
                stream,
                data,
                max_pending_steps=self.max_pending_steps,
                spare=self._spare_buffers.pop(stream.key, None),
            )

        buf = self._write_buffers[stream.key]
        buf.append(data, dt)

    def _unfinished_payload_bytes(self) -> int:
        """Count retained output memory: staging, live, spare and in-flight."""

        return (
            self._cpu_stager.allocated_bytes
            + sum(buffer.allocated_bytes for buffer in self._write_buffers.values())
            + sum(int(data.nbytes) for data in self._spare_buffers.values())
            + sum(
                sum(buffer.allocated_bytes for buffer in pending.buffers)
                if pending.buffers
                else pending.payload_bytes
                for pending in self._pending_writes
            )
        )

    def _retire_oldest_output_memory(self, *, dt) -> bool:
        """Release one retained allocation; return False when none is left."""

        if self._pending_writes:
            pending = self._pending_writes.pop(0)
            try:
                self._wait_for(pending, dt=dt)
            except BaseException as error:
                self._latch_failures("byte-bounded NetCDF pending writes", [error])
            return True
        if self._spare_buffers:
            self._spare_buffers.pop(next(iter(self._spare_buffers)))
            return True
        return False

    def _reserve_output_bytes(self, key: str, nbytes: int, *, dt) -> None:
        """Retire memory until a new batch array for ``key`` fits the budget."""

        limit = self.max_pending_output_bytes
        while (
            key not in self._spare_buffers
            and self._unfinished_payload_bytes() + nbytes > limit
            and self._retire_oldest_output_memory(dt=dt)
        ):
            pass

    def _pending_step_counts(self) -> dict[str, int]:
        counts = {
            key: buffer.count
            for key, buffer in self._write_buffers.items()
            if buffer.count
        }
        for pending in self._pending_writes:
            for key, step_count in pending.step_counts:
                counts[key] = counts.get(key, 0) + step_count
        return counts

    def _wait_for(self, pending: PendingNetCDFWrite, *, dt) -> None:
        try:
            pending.future.result()
            self._recycle_buffers(pending.buffers)
        except Exception as exc:
            outputs = tuple(key for key, _count in pending.step_counts)
            self._emit_event(
                ModelEvent(
                    level="error",
                    name="output.write_failed",
                    message="Failed to write time step",
                    fields={
                        "output": outputs[0] if len(outputs) == 1 else outputs,
                        "outputs": outputs,
                        "steps": dict(pending.step_counts),
                        "time": str(dt),
                        "error": failure_description(exc)["message"],
                    },
                )
            )
            raise

    def check_completed_writes(self, *, dt) -> None:
        """Observe every completed background write without blocking."""

        self._raise_if_background_failed()
        remaining: list[PendingNetCDFWrite] = []
        failures: list[BaseException] = []
        for pending in self._pending_writes:
            if not pending.future.done():
                remaining.append(pending)
                continue
            try:
                self._wait_for(pending, dt=dt)
            except BaseException as error:
                failures.append(error)
        self._pending_writes = remaining
        if failures:
            self._latch_failures(
                "completed NetCDF background writes",
                failures,
            )

    def flush_and_wait(self, *, dt) -> None:
        """Make every buffered statistics row durable without closing workers."""

        self._raise_if_background_failed()
        failures: list[BaseException] = []
        try:
            self._flush_all_write_buffers(latch=False)
        except BaseException as error:
            failures.append(error)
        pending, self._pending_writes = self._pending_writes, []
        for item in pending:
            try:
                self._wait_for(item, dt=dt)
            except BaseException as error:
                failures.append(error)
        if not failures:
            # Workers have synced HDF5 buffers and committed-step markers to
            # the OS; fsync makes those committed rows survive a crash.
            try:
                self.sync_appended_files()
            except BaseException as error:
                failures.append(error)
        if failures:
            self._latch_failures(
                "NetCDF output durability boundary",
                failures,
            )

    def _limit_pending_output_bytes(self, *, dt) -> None:
        """Bound retained output memory by retiring the oldest allocations."""

        limit = self.max_pending_output_bytes
        while self._unfinished_payload_bytes() > limit:
            if not self._retire_oldest_output_memory(dt=dt):
                return

    def _limit_pending_steps(self, *, dt) -> None:
        """Bound each output stream by its exact unfinished timestep count."""

        self.check_completed_writes(dt=dt)

        if self.max_pending_steps == 1:
            pending, self._pending_writes = (self._pending_writes, [])
            failures: list[BaseException] = []
            for item in pending:
                try:
                    self._wait_for(item, dt=dt)
                except BaseException as error:
                    failures.append(error)
            if failures:
                self._latch_failures(
                    "single-step NetCDF pending writes",
                    failures,
                )
            return

        counts = self._pending_step_counts()
        while counts and max(counts.values()) > self.max_pending_steps:
            overfull = {
                key for key, count in counts.items() if count > self.max_pending_steps
            }
            index = next(
                (
                    index
                    for index, pending in enumerate(self._pending_writes)
                    if any(key in overfull for key, _count in pending.step_counts)
                )
            )
            pending = self._pending_writes.pop(index)
            try:
                self._wait_for(pending, dt=dt)
            except BaseException as error:
                self._latch_failures(
                    "bounded NetCDF pending writes",
                    [error],
                )
            for key, step_count in pending.step_counts:
                counts[key] -= step_count
                if counts[key] == 0:
                    counts.pop(key)

    def _prepare_output_files(self, dt: datetime | cftime.datetime) -> None:
        """Create the files that receive rows stamped ``dt``."""

        if self.output_split_by_year:
            if self._current_year is None:
                # First call - set up files
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
            elif self._current_year != dt.year:
                # Year transition – flush remaining buffers for the old year first
                self._flush_all_write_buffers()
                self._release_worker_files(
                    tuple(
                        stream.path
                        for streams in self._output_streams.values()
                        for stream in streams
                    )
                )
                # Year transition - create new files for new year
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
        elif not self._files_created:
            self._create_netcdf_files()

    def _stage_output_step(
        self,
        values: Mapping[str, torch.Tensor],
        dt: datetime | cftime.datetime,
    ) -> _StagedOutputStep:
        """Narrow and check on the storage device, then enqueue the copy."""

        flags: list[tuple[torch.Tensor, str, str]] = []
        snapshots: dict[str, torch.Tensor] = {}
        for name, storage in values.items():
            narrowed = _checked_narrowing(
                storage,
                self.save_precision
                if self.save_precision is not None and storage.is_floating_point()
                else storage.dtype,
                name=name,
                flags=flags,
            )
            if narrowed.dtype == storage.dtype and storage.device.type == "cuda":
                # A deferred copy must not observe the next step's updates.
                narrowed = narrowed.clone()
            snapshots[name] = narrowed
        if flags:
            device = flags[0][0].device
            snapshots[_NARROWING_FLAGS] = torch.stack(
                [flag.to(device=device) for flag, _name, _label in flags]
            )
        return _StagedOutputStep(
            dt=dt,
            keys=tuple(values),
            transfer=self._cpu_stager.stage_async(snapshots),
            flags=tuple(flags),
        )

    def _consume_staged_step(self, staged: _StagedOutputStep) -> None:
        """Validate a landed step and append its rows to the write buffers."""

        staged_outputs = staged.transfer.wait()
        if staged.flags:
            _raise_narrowing_failures(staged.flags, staged_outputs[_NARROWING_FLAGS])
        prepared: list[tuple[_NetCDFOutputStream, np.ndarray]] = []
        for out_name in staged.keys:
            time_step_data = staged_outputs[out_name].numpy()
            for stream in self._output_streams[out_name]:
                prepared.append(
                    (
                        stream,
                        time_step_data
                        if stream.component is None
                        else time_step_data[..., stream.component],
                    )
                )
        for stream, stream_data in prepared:
            self._buffer_and_maybe_flush(stream, stream_data, staged.dt)
        try:
            self._flush_ready_write_buffers()
        except BaseException as error:
            self._latch_failures("NetCDF write submission", [error])

    def _drain_staged_step(self) -> None:
        staged, self._staged_step = self._staged_step, None
        if staged is not None:
            self._consume_staged_step(staged)

    def append(
        self, dt: datetime | cftime.datetime, values: Mapping[str, torch.Tensor]
    ) -> None:
        """Snapshot one finalized sample; the caller may then reuse its tensors."""
        self._raise_if_background_failed()
        current = self._stage_output_step(values, dt)
        previous, self._staged_step = self._staged_step, None
        try:
            if previous is not None:
                self._consume_staged_step(previous)
        except BaseException as error:
            with cleanup_on_exit("statistics output staging", (current.transfer.wait,)):
                self._latch_failures("statistics output staging", [error])
        try:
            self._prepare_output_files(dt)
        except BaseException:
            with cleanup_on_exit(
                "statistics output preparation", (current.transfer.wait,)
            ):
                raise
        if current.transfer.asynchronous and self.max_pending_steps > 1:
            self._staged_step = current
        else:
            self._consume_staged_step(current)
        self._limit_pending_output_bytes(dt=dt)
        self._limit_pending_steps(dt=dt)

    def _cleanup_lock_files(self) -> None:
        with cleanup_on_exit(
            "NetCDF output locks",
            (
                lambda path=path: path.with_suffix(path.suffix + ".lock").unlink(
                    missing_ok=True
                )
                for path in self._all_created_files
            ),
        ):
            pass

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
                executor.submit(_close_worker_netcdf_files).result()
            except BaseException as error:
                failures.append(error)
            try:
                executor.shutdown(wait=True)
            except BaseException as error:
                failures.append(error)
        if not failures:
            # Workers have closed their handles; make the appended rows durable.
            try:
                self.sync_appended_files()
            except BaseException as error:
                failures.append(error)
        self._write_buffers.clear()
        self._output_streams.clear()
        try:
            self.release_staging()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("statistics output workers", failures)

    def start(self) -> None:
        """Start output workers while model initialization continues."""

        created = []
        ready = []
        try:
            for _ in range(self.num_workers):
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=get_context("spawn"),
                    initializer=_initialize_netcdf_worker,
                )
                created.append(executor)
                ready.append(
                    PendingNetCDFWrite(
                        step_counts=(),
                        payload_bytes=0,
                        future=executor.submit(os.getpid),
                    )
                )
        except BaseException as primary:
            failures: list[BaseException] = [primary]
            for executor in reversed(created):
                try:
                    executor.shutdown(wait=True)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            if len(failures) > 1:
                error = ResourceCleanupError(
                    "statistics output worker startup",
                    failures,
                )
                raise error from primary
            raise
        self._write_executors = created
        self._pending_writes.extend(ready)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        for cleanup in (self._cleanup_executor, self._cleanup_lock_files):
            try:
                cleanup()
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("NetCDF output", failures)

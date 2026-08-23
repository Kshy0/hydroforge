# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import atexit
from collections import OrderedDict
from collections.abc import Mapping
import os
import logging
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union
from uuid import uuid4

import cftime
import netCDF4 as nc
import numpy as np
import torch

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.contracts.naming import sanitize_symbol
from hydroforge.output.netcdf.plan import (
    NetCDFCreateRequest, NetCDFWriteRequest, OutputFilePlan,
)
from hydroforge.contracts.events import ModelEvent
from hydroforge.serialization.netcdf import (
    BOOL_LOGICAL_DTYPE, COMMITTED_STEPS_ATTR, LOGICAL_DTYPE_ATTR,
    OUTPUT_FORMAT, OUTPUT_VERSION, RUN_ID_ATTR,
    _atomic_netcdf_dataset_trusted, _create_netcdf_variable_trusted,
    netcdf_dtype_encoding,
)


logger = logging.getLogger(__name__)


_TORCH_NUMPY_DTYPES = {
    torch.bool: np.dtype(np.bool_),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
}


def _checked_output_array(
    tensor: torch.Tensor, target_dtype: torch.dtype, *, name: str,
) -> np.ndarray:
    """Materialize one output without allowing a finite value to overflow."""

    target = _TORCH_NUMPY_DTYPES[target_dtype]
    source = tensor.detach().cpu().numpy()
    if source.dtype == target:
        return source
    finite = np.isfinite(source)
    if target == np.dtype(np.float32) and source.size:
        if np.any(np.abs(source[finite]) > np.finfo(np.float32).max):
            raise OverflowError(
                f"statistics output {name!r} contains values outside "
                "float32 range"
            )
    converted = source.astype(target, copy=False)
    if source.size and np.any(finite & ~np.isfinite(converted)):
        raise OverflowError(
            f"statistics output {name!r} overflowed {target}"
        )
    if source.size and np.any(
        finite & (source != 0) & (converted == 0)
    ):
        raise OverflowError(
            f"statistics output {name!r} contains nonzero values that "
            f"underflow in {target}"
        )
    return converted


def _checked_output_tensor_copy(
    tensor: torch.Tensor,
    *,
    target_device: torch.device,
    target_dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Copy an in-memory result after validating a narrowing conversion."""

    if tensor.dtype != target_dtype:
        if target_dtype == torch.float32 and tensor.numel():
            finite = torch.isfinite(tensor)
            outside = finite & (
                torch.abs(tensor) > torch.finfo(torch.float32).max
            )
            if bool(outside.any().item()):
                raise OverflowError(
                    f"statistics output {name!r} contains values outside "
                    "float32 range"
                )
    if tensor.dtype != target_dtype and target_dtype == torch.float32:
        narrowed = tensor.to(dtype=target_dtype)
        if bool(
            (
                torch.isfinite(tensor)
                & (tensor != 0)
                & (narrowed == 0)
            ).any().item()
        ):
            raise OverflowError(
                f"statistics output {name!r} contains nonzero values that "
                "underflow in torch.float32"
            )
    return tensor.detach().to(
        device=target_device, dtype=target_dtype, copy=True,
    )


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
    ) -> _NetCDFWriteBuffer:
        capacity = min(stream.batch_size, max_pending_steps)
        data = np.empty((capacity, *row.shape), dtype=row.dtype)
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
        return self.data[:self.count]


def _static_variable_applies(
    name: str,
    specification: dict[str, Any],
    *,
    output_coordinate: str | None,
    dimensions,
) -> bool:
    """Resolve an explicitly coordinate-scoped static variable."""

    if specification["coordinate"] != output_coordinate:
        return False
    return True
_DEFAULT_MAX_BATCH: int = 30


def compute_write_batch_size(
    saved_points: int,
    dtype_bytes: int = 4,
    max_ipc_bytes: int = _DEFAULT_MAX_IPC_BYTES,
    max_batch: int = _DEFAULT_MAX_BATCH,
) -> int:
    """Return the number of time steps to batch per subprocess write.

    The batch is capped so that ``batch * saved_points * dtype_bytes``
    does not exceed *max_ipc_bytes*, reducing the batch to one step for very
    large grids (e.g. glb_01min).
    """
    per_step = saved_points * dtype_bytes
    batch = int(max_ipc_bytes / max(per_step, 1))
    return max(1, min(batch, max_batch))


def constrain_write_batch_sizes(
    desired: Mapping[str, int],
    *,
    row_bytes: Mapping[str, int],
    stream_counts: Mapping[str, int],
    max_pending_bytes: int,
) -> dict[str, int]:
    """Fit aggregate buffer capacities into one process-wide byte budget."""

    batches = dict(desired)
    total = sum(
        row_bytes[name] * stream_counts[name] * batch
        for name, batch in batches.items()
    )
    while total > max_pending_bytes:
        candidates = [name for name, batch in batches.items() if batch > 1]
        if not candidates:
            break
        name = max(
            candidates,
            key=lambda item: (
                row_bytes[item] * stream_counts[item] * batches[item], item,
            ),
        )
        old = batches[name]
        new = max(1, old // 2)
        batches[name] = new
        total -= row_bytes[name] * stream_counts[name] * (old - new)
    return batches


_WORKER_FILE_CACHE_SIZE = 32
_WORKER_NETCDF_FILES: OrderedDict[
    Path, tuple[nc.Dataset, tuple[int, int]],
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
        _old_path, (old_dataset, _old_identity) = (
            _WORKER_NETCDF_FILES.popitem(last=False)
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
    if _is_wsl() and hasattr(os, 'posix_fadvise'):
        try:
            with open(output_path, 'rb') as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _append_netcdf_request(
    ncfile: nc.Dataset, request: NetCDFWriteRequest,
) -> tuple[str, int]:
    """Append one batch through an already-open validated Dataset handle."""

    time_var = ncfile.variables["time"]
    target = _find_data_variable(ncfile, request.variable)
    variable = ncfile.variables[target]
    if time_var.dimensions != ("time",):
        raise ValueError(
            f"time variable in {request.output_path} must have dimensions "
            "('time',)"
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
        committed, (int, np.integer),
    ):
        raise TypeError(
            f"{COMMITTED_STEPS_ATTR} must be an integer in "
            f"{request.output_path}"
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
    numeric_times = np.asarray(nc.date2num(
        request.times, units=units, calendar=calendar,
    ))
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
                "NetCDF write timestamps must be strictly increasing "
                "across batches"
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
) -> Union[Path, List[Path]]:
    """
    Create one empty NetCDF file with the proper structure.

    Args:
        args: Tuple containing (mean_var_name, metadata, coord_values,
              output_dir, rank, year, calendar, time_unit, num_trials)

    Returns:
        Path or List[Path] to the created NetCDF file(s)
    """
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
    def create_single_file(file_safe_name: str, file_var_name: str, description_suffix: str = "") -> Path:
        output_path = OutputFilePlan(
            directory=output_dir,
            variable=file_safe_name,
            rank=rank,
            year=year,
        ).path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with _atomic_netcdf_dataset_trusted(
            output_path, format="NETCDF4",
        ) as ncfile:
            # Write global attributes
            ncfile.setncattr('title', f'Time series for rank {rank}: {file_var_name}')
            ncfile.setncattr('original_variable_name', file_var_name)
            ncfile.setncattr("hydroforge_output_format", OUTPUT_FORMAT)
            ncfile.setncattr("hydroforge_output_version", OUTPUT_VERSION)
            ncfile.setncattr("hydroforge_rank", rank)
            ncfile.setncattr("hydroforge_world_size", world_size)
            ncfile.setncattr(RUN_ID_ATTR, run_id)
            ncfile.setncattr(COMMITTED_STEPS_ATTR, 0)

            # Create time dimension (unlimited for streaming)
            ncfile.createDimension('time', None)

            dim_names = list(schema.data_dimensions)
            for dimension, extent in schema.dimensions:
                ncfile.createDimension(dimension, extent)

            if coord_name and coord_values is not None:
                coord_var = ncfile.createVariable(
                    coord_name,
                    coord_values.dtype,
                    ('saved_points',),
                )
                coord_var[:] = coord_values

            # Write user-supplied static per-point variables. Coordinate scope
            # decides applicability; once applicable, dimension mismatch is a
            # schema error rather than a silent omission.
            for sv_name, sv_spec in (static_vars or {}).items():
                sv_dim = sv_spec["dim"]
                output_coord = schema.output_coordinate
                if not _static_variable_applies(
                    sv_name, sv_spec,
                    output_coordinate=output_coord,
                    dimensions=ncfile.dimensions,
                ):
                    continue
                sv_values = sv_spec["values"]
                storage_dtype, logical_dtype = netcdf_dtype_encoding(
                    sv_values.dtype,
                )
                sv_var = ncfile.createVariable(
                    sv_name, storage_dtype, (sv_dim,),
                )
                if logical_dtype is not None:
                    sv_var.setncattr(LOGICAL_DTYPE_ATTR, logical_dtype)
                sv_var[:] = sv_values
                for ak, av in sv_spec["attrs"].items():
                    sv_var.setncattr(ak, av)

            time_var = ncfile.createVariable('time', 'f8', ('time',))
            time_var.setncattr('units', time_unit)
            time_var.setncattr('calendar', calendar)

            # Create single data variable
            nc_var = _create_netcdf_variable_trusted(
                ncfile,
                file_safe_name,
                dtype,
                dim_names,
                options=schema.create_options,
            )
            if schema.logical_dtype is not None:
                nc_var.setncattr(LOGICAL_DTYPE_ATTR, schema.logical_dtype)
            desc = schema.description + description_suffix
            nc_var.setncattr('description', desc)
            nc_var.setncattr('actual_shape', str(file_actual_shape))
            nc_var.setncattr('tensor_shape', str(tensor_shape))
            nc_var.setncattr('long_name', file_var_name)

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
    else:
        return create_single_file(safe_name, mean_var_name)



class _NetCDFWriter:
    """Own streaming and in-memory finalization for one aggregator."""

    def __init__(self, owner) -> None:
        self.owner = owner
        self._background_failure: BaseException | None = None
        self._last_time_number: float | None = None

    def reset_timeline(self) -> None:
        """Reset finalized-time ordering for a newly initialized output run."""

        self._last_time_number = None

    def _validate_next_time(
        self, dt: Union[datetime, cftime.datetime],
    ) -> float:
        value = float(nc.date2num(
            dt, units=self.owner.time_unit, calendar=self.owner.calendar,
        ))
        if not np.isfinite(value):
            raise ValueError("statistics output time must be finite")
        if (
            self._last_time_number is not None
            and value <= self._last_time_number
        ):
            raise ValueError(
                "statistics output times must be strictly increasing"
            )
        return value

    def _emit_event(self, event: ModelEvent) -> None:
        """Keep observability failures outside the output transaction."""

        try:
            self.owner.event_sink.emit(event)
        except Exception:
            logger.exception(
                "statistics event sink failed while emitting %s", event.name,
            )

    def _raise_if_background_failed(self) -> None:
        failure = self._background_failure
        if failure is not None:
            raise failure

    def _latch_failures(
        self, label: str, failures: list[BaseException],
    ) -> None:
        if not failures:
            return
        if len(failures) == 1:
            failure = failures[0]
        else:
            failure = ResourceCleanupError(label, failures)
        if self._background_failure is None:
            self._background_failure = failure
            execution = getattr(self.owner, "_execution", None)
            poison = getattr(execution, "poison", None)
            if callable(poison):
                poison(
                    failure, phase="statistics background write",
                )
        raise self._background_failure

    def _resolve_run_id(self) -> str:
        """Return one identity shared by every file in this output run."""

        existing = getattr(self.owner, "_output_run_id", None)
        if existing is not None:
            return existing

        explicit = getattr(self.owner, "run_id", None)
        if explicit is not None:
            run_id = explicit
        elif self.owner.world_size == 1:
            run_id = str(uuid4())
        else:
            raise RuntimeError(
                "multi-rank statistics output run ID was not installed by the "
                "rank-synchronous runtime materialization transaction"
            )

        self.owner._output_run_id = run_id
        return run_id

    def _create_netcdf_files(self, year: Optional[int] = None) -> None:
        """Create empty NetCDF files with proper structure for streaming.

        Creation writes headers only (~5 ms per output), so it runs inline; a
        spawned pool cost ~1.6 s of interpreter start-up on the first step.
        """
        if self.owner.in_memory_mode:
            # Skip file creation in in-memory mode
            return

        if not self.owner.output_split_by_year and self.owner._files_created:
            return

        self._raise_if_background_failed()
        self._emit_event(ModelEvent(
            level="info", name="output.create_start",
            message="Creating NetCDF file structure", fields={"year": year},
        ))

        # Resolve once and retain across every variable, rank, and split year.
        run_id = self._resolve_run_id()

        # Plan and validate the complete transaction before mutating any final
        # path.  Every file is first created in a sibling staging directory.
        requests: list[NetCDFCreateRequest] = []
        planned_by_variable: dict[str, tuple[Path, ...]] = {}
        planned_paths: list[Path] = []
        for out_name in self.owner._metadata:
            schema = self.owner._netcdf_schemas[out_name]
            coord_name = schema.output_coordinate
            coord_values = (
                None
                if coord_name is None
                else self.owner._coord_cache[coord_name]
            )
            request = NetCDFCreateRequest(
                variable=out_name,
                schema=schema,
                coordinate_values=coord_values,
                output_dir=self.owner.output_dir,
                rank=self.owner.rank,
                world_size=self.owner.world_size,
                year=year,
                calendar=self.owner.calendar,
                time_unit=self.owner.time_unit,
                static_variables=self.owner.static_vars,
                run_id=run_id,
            )
            requests.append(request)
            safe_name = sanitize_symbol(out_name)
            order = schema.order
            names = (
                tuple(f"{safe_name}_{index}" for index in range(order))
                if order > 1 else (safe_name,)
            )
            output_paths = tuple(
                OutputFilePlan(
                    directory=self.owner.output_dir,
                    variable=name,
                    rank=self.owner.rank,
                    year=year,
                ).path
                for name in names
            )
            planned_by_variable[out_name] = output_paths
            planned_paths.extend(output_paths)
        # Compile each output route up front before mutating final paths.
        output_streams: dict[str, tuple[_NetCDFOutputStream, ...]] = {}
        batch_events: list[ModelEvent] = []
        executor_count = len(self.owner._write_executors)
        stream_index = 0
        for out_name in self.owner._metadata:
            schema = self.owner._netcdf_schemas[out_name]
            k_val = schema.order
            elements = max(1, int(np.prod(schema.file_actual_shape)))
            element_size = np.dtype(schema.dtype).itemsize
            batch_size = schema.write_batch_size
            paths = planned_by_variable[out_name]
            streams: list[_NetCDFOutputStream] = []
            for component, path in enumerate(paths):
                key = f"{out_name}_{component}" if k_val > 1 else out_name
                streams.append(_NetCDFOutputStream(
                    key=key,
                    path=path,
                    batch_size=batch_size,
                    component=component if k_val > 1 else None,
                    executor_index=(
                        stream_index % executor_count if executor_count else None
                    ),
                ))
                stream_index += 1
            output_streams[out_name] = tuple(streams)
            batch_events.append(ModelEvent(
                level="info", name="output.batch_configured",
                message="Configured NetCDF write batch",
                fields={
                    "output": out_name,
                    "steps": batch_size,
                    "elements_per_step": elements,
                    "storage_bytes_per_element": element_size,
                },
            ))

        output_dir = Path(self.owner.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stage_root = Path(tempfile.mkdtemp(
            prefix=".hydroforge-output-", dir=output_dir,
        ))
        stage_output = stage_root / "new"
        backup_dir = stage_root / "old"
        stage_output.mkdir()
        backup_dir.mkdir()
        staged_results: dict[str, Path | list[Path]] = {}
        moves: list[tuple[Path, Path]] = []
        preserve_staging = False
        try:
            for request in requests:
                expected = planned_by_variable[request.variable]
                _create_netcdf_file_process(replace(
                    request, output_dir=stage_output,
                ))
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
        self.owner._netcdf_files.update(staged_results)
        self.owner._all_created_files.update(planned_paths)
        self.owner._write_buffers.clear()
        self.owner._output_streams = output_streams
        self.owner._files_created = True

        for event in batch_events:
            self._emit_event(event)
        for path in planned_paths:
            self._emit_event(ModelEvent(
                level="info", name="output.file_created",
                message="Created NetCDF file", fields={"path": str(path)},
            ))
        total_files = sum(map(len, output_streams.values()))
        self._emit_event(ModelEvent(
            level="info", name="output.create_complete",
            message="Created NetCDF files for streaming",
            fields={"files": total_files},
        ))

    @staticmethod
    def _buffer_payload_bytes(buffer: _NetCDFWriteBuffer) -> int:
        return buffer.payload_bytes

    def _partition_write_groups(
        self, keys: list[str],
    ) -> tuple[tuple[str, ...], ...]:
        """Group small buffers by executor without exceeding the IPC cap."""

        grouped: dict[int | None, list[list[Any]]] = {}
        for key in keys:
            buffer = self.owner._write_buffers.get(key)
            if buffer is None or not buffer.count:
                continue
            executor_index = buffer.stream.executor_index
            payload_bytes = self._buffer_payload_bytes(buffer)
            partitions = grouped.setdefault(executor_index, [])
            if (
                not partitions
                or (
                    partitions[-1][0]
                    and partitions[-1][1] + payload_bytes
                    > _DEFAULT_MAX_IPC_BYTES
                )
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
            buffer = self.owner._write_buffers.get(key)
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
            requests.append(NetCDFWriteRequest(
                variable=stream.key,
                data=buffer.request_data(),
                output_path=stream.path,
                times=times,
            ))
            step_counts.append((key, len(times)))
            buffers.append(buffer)
            buffer_keys.append(key)
        if not requests:
            return

        request_group = tuple(requests)
        if executor_index is None:
            for request in request_group:
                _write_netcdf_process(request)
        else:
            future = self.owner._write_executors[executor_index].submit(
                _write_netcdf_group_process, request_group,
            )
            self.owner._pending_writes.append(PendingNetCDFWrite(
                step_counts=tuple(step_counts),
                payload_bytes=sum(buffer.payload_bytes for buffer in buffers),
                future=future,
            ))

        for key, buffer in zip(buffer_keys, buffers, strict=True):
            if self.owner._write_buffers.get(key) is buffer:
                self.owner._write_buffers.pop(key)

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
            for key, buffer in self.owner._write_buffers.items()
            if buffer.count >= min(
                buffer.stream.batch_size,
                self.owner.max_pending_steps,
            )
        ]
        self._flush_write_buffers(ready)


    def _flush_all_write_buffers(self, *, latch: bool = True) -> None:
        """Flush every pending write buffer (called on year transition / shutdown)."""
        self._raise_if_background_failed()
        try:
            self._flush_write_buffers(list(self.owner._write_buffers))
        except BaseException as error:
            if latch:
                self._latch_failures("NetCDF write buffers", [error])
            raise

    def _streams_for_output(
        self, out_name: str,
    ) -> tuple[_NetCDFOutputStream, ...]:
        return self.owner._output_streams[out_name]


    def _buffer_and_maybe_flush(
        self,
        stream: _NetCDFOutputStream,
        data: np.ndarray,
        dt: Union[datetime, cftime.datetime],
    ) -> None:
        """Append one time step; ready buffers are flushed as one group."""
        self._raise_if_background_failed()
        if stream.key not in self.owner._write_buffers:
            self.owner._write_buffers[stream.key] = _NetCDFWriteBuffer.allocate(
                stream,
                data,
                max_pending_steps=self.owner.max_pending_steps,
            )

        buf = self.owner._write_buffers[stream.key]
        buf.append(data, dt)

    def _buffered_allocated_bytes(self) -> int:
        return sum(
            buffer.allocated_bytes
            for buffer in self.owner._write_buffers.values()
        )

    def _unfinished_payload_bytes(self) -> int:
        return self._buffered_allocated_bytes() + sum(
            pending.payload_bytes for pending in self.owner._pending_writes
        )

    def _pending_step_counts(self) -> dict[str, int]:
        counts = {
            key: buffer.count
            for key, buffer in self.owner._write_buffers.items()
            if buffer.count
        }
        for pending in self.owner._pending_writes:
            for key, step_count in pending.step_counts:
                counts[key] = counts.get(key, 0) + step_count
        return counts

    def _wait_for(self, pending: PendingNetCDFWrite, *, dt) -> None:
        try:
            pending.future.result()
        except Exception as exc:
            outputs = tuple(key for key, _count in pending.step_counts)
            self._emit_event(ModelEvent(
                level="error", name="output.write_failed",
                message="Failed to write time step",
                fields={
                    "output": outputs[0] if len(outputs) == 1 else outputs,
                    "outputs": outputs,
                    "steps": dict(pending.step_counts),
                    "time": str(dt), "error": str(exc),
                },
            ))
            raise

    def check_completed_writes(self, *, dt) -> None:
        """Observe every completed background write without blocking."""

        self._raise_if_background_failed()
        remaining: list[PendingNetCDFWrite] = []
        failures: list[BaseException] = []
        for pending in self.owner._pending_writes:
            if not pending.future.done():
                remaining.append(pending)
                continue
            try:
                self._wait_for(pending, dt=dt)
            except BaseException as error:
                failures.append(error)
        self.owner._pending_writes = remaining
        if failures:
            self._latch_failures(
                "completed NetCDF background writes", failures,
            )

    def flush_and_wait(self, *, dt) -> None:
        """Make every buffered statistics row durable without closing workers."""

        self._raise_if_background_failed()
        failures: list[BaseException] = []
        try:
            self._flush_all_write_buffers(latch=False)
        except BaseException as error:
            failures.append(error)
        pending, self.owner._pending_writes = self.owner._pending_writes, []
        for item in pending:
            try:
                self._wait_for(item, dt=dt)
            except BaseException as error:
                failures.append(error)
        if failures:
            self._latch_failures(
                "NetCDF output durability boundary", failures,
            )

    def _limit_pending_output_bytes(self, *, dt) -> None:
        """Bound aggregate buffered and submitted output memory."""

        limit = getattr(
            self.owner,
            "max_pending_output_bytes",
            _DEFAULT_MAX_PENDING_OUTPUT_BYTES,
        )
        if self._unfinished_payload_bytes() <= limit:
            return
        if self.owner._write_buffers:
            try:
                self._flush_all_write_buffers(latch=False)
            except BaseException as error:
                self._latch_failures(
                    "byte-bounded NetCDF write buffers", [error],
                )
        while (
            self.owner._pending_writes
            and self._unfinished_payload_bytes() > limit
        ):
            pending = self.owner._pending_writes.pop(0)
            try:
                self._wait_for(pending, dt=dt)
            except BaseException as error:
                self._latch_failures(
                    "byte-bounded NetCDF pending writes", [error],
                )

    def _limit_pending_steps(self, *, dt) -> None:
        """Bound each output stream by its exact unfinished timestep count."""

        self.check_completed_writes(dt=dt)

        if self.owner.max_pending_steps == 1:
            pending, self.owner._pending_writes = (
                self.owner._pending_writes, []
            )
            failures: list[BaseException] = []
            for item in pending:
                try:
                    self._wait_for(item, dt=dt)
                except BaseException as error:
                    failures.append(error)
            if failures:
                self._latch_failures(
                    "single-step NetCDF pending writes", failures,
                )
            return

        counts = self._pending_step_counts()
        while counts and max(counts.values()) > self.owner.max_pending_steps:
            overfull = {
                key for key, count in counts.items()
                if count > self.owner.max_pending_steps
            }
            index = next((
                index
                for index, pending in enumerate(self.owner._pending_writes)
                if any(
                    key in overfull for key, _count in pending.step_counts
                )
            ))
            pending = self.owner._pending_writes.pop(index)
            try:
                self._wait_for(pending, dt=dt)
            except BaseException as error:
                self._latch_failures(
                    "bounded NetCDF pending writes", [error],
                )
            for key, step_count in pending.step_counts:
                counts[key] -= step_count
                if counts[key] == 0:
                    counts.pop(key)


    def _finalize_time_step_in_memory(self, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize time step in in-memory mode by copying storage to result tensors.

        Args:
            dt: Time step to finalize
        """
        keys_to_write = [
            key for key in self.owner._output_keys
            if key in self.owner._dirty_outputs
        ]
        result_copies: dict[str, Any] = {}
        for out_name in keys_to_write:
            storage_tensor = self.owner._storage[out_name]
            result_copies[out_name] = _checked_output_tensor_copy(
                storage_tensor,
                target_device=self.owner.result_device,
                target_dtype=self.owner._result_dtype(out_name),
                name=out_name,
            )
        for out_name, result_copy in result_copies.items():
            self.owner._result_tensors[out_name].append(result_copy)
        self.owner._dirty_outputs.clear()
        self.owner._current_time_index += 1

        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True


    def finalize_time_step(self, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize the current time step by writing results to output.

        In streaming mode: writes to NetCDF files incrementally.
        In in-memory mode: copies current storage to result tensors.

        Args:
            dt: Time step to finalize (datetime or cftime.datetime)
        """
        time_number = self._validate_next_time(dt)

        # Handle in-memory mode
        if self.owner.in_memory_mode:
            self._finalize_time_step_in_memory(dt)
            self._last_time_number = time_number
            return

        self._raise_if_background_failed()
        if self.owner.output_split_by_year:
            if self.owner._current_year is None:
                # First call - set up files
                self._create_netcdf_files(year=dt.year)
                self.owner._current_year = dt.year
            elif self.owner._current_year != dt.year:
                # Year transition – flush remaining buffers for the old year first
                self._flush_all_write_buffers()
                # Year transition - create new files for new year
                self._create_netcdf_files(year=dt.year)
                self.owner._current_year = dt.year
        else:
            # Create NetCDF files if not already created
            if not self.owner._files_created:
                self._create_netcdf_files()

        # Resolve and materialize every output before mutating any buffer.  This
        # prevents a bad later output from leaving earlier outputs half-staged.
        keys_to_write = [
            key for key in self.owner._output_keys
            if key in self.owner._dirty_outputs
        ]
        prepared: list[tuple[_NetCDFOutputStream, np.ndarray]] = []
        for out_name in keys_to_write:
            tensor = self.owner._storage[out_name]
            streams = self._streams_for_output(out_name)

            # Convert tensor to numpy
            time_step_data = _checked_output_array(
                tensor, self.owner._result_dtype(out_name), name=out_name,
            )

            for stream in streams:
                stream_data = (
                    time_step_data
                    if stream.component is None
                    else time_step_data[..., stream.component]
                )
                prepared.append((stream, stream_data))

        for stream, stream_data in prepared:
            self._buffer_and_maybe_flush(stream, stream_data, dt)
        try:
            self._flush_ready_write_buffers()
        except BaseException as error:
            self._latch_failures("NetCDF write submission", [error])
        self.owner._dirty_outputs.difference_update(keys_to_write)

        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True

        self._limit_pending_output_bytes(dt=dt)
        self._limit_pending_steps(dt=dt)
        self.owner._current_time_index += 1
        self._last_time_number = time_number

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
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache, partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from uuid import uuid4
from weakref import WeakSet

import cftime
import netCDF4 as nc
import numpy as np
import torch

from hydroforge.core.arrays import torch_to_numpy_dtype
from hydroforge.core.errors import (
    ResourceCleanupError,
    cleanup_on_exit,
    failure_description,
)
from hydroforge.core.events import ModelEvent
from hydroforge.core.naming import sanitize_symbol
from hydroforge.io.files import fsync_directory, fsync_file
from hydroforge.io.netcdf.encoding import (
    BOOL_LOGICAL_DTYPE,
    COMPLETE_DATA_ATTR,
    LOGICAL_DTYPE_ATTR,
    decoded_output_tensor,
    narrowing_flag,
    netcdf_dtype_encoding,
    raise_narrowing_failures,
    saved_dtype,
)
from hydroforge.io.netcdf.options import (
    _probe_blosc_zstd_filter,
    create_netcdf_variable,
    ensure_hdf5_plugins,
    start_blosc_zstd_probe,
)
from hydroforge.io.netcdf.write import atomic_netcdf_dataset
from hydroforge.io.rank_output.ring import (
    OutputRing,
    RingSlots,
    attached_ring,
    detach_rings,
    page_lockable,
    place_slots,
    plan_output_batches,
)
from hydroforge.io.rank_output.schema import (
    COMMITTED_STEPS_ATTR,
    TIME_DIM,
    TIME_UNITS,
    NetCDFSchema,
    RankFileHeader,
    create_time_axis,
    rank_file_name,
    write_point_coordinate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NetCDFWriteRequest:
    """One batch of rows appended to one rank file."""

    variable: str
    data: np.ndarray
    output_path: Path
    times: tuple[Any, ...]
    chunk_cache: int | None = None


@dataclass(frozen=True, slots=True)
class _SlotWrite:
    """The few bytes an output worker needs to append one ring slot."""

    variable: str
    output_path: Path
    offset: int
    count: int
    row_shape: tuple[int, ...]
    dtype: str
    times: tuple[Any, ...]
    chunk_cache: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.count) is not int
            or self.count < 1
            or self.count != len(self.times)
        ):
            raise ValueError(
                "NetCDF write row count must be positive and match timestamps"
            )
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("NetCDF write ring offset must be nonnegative")

    def request(self, buffer: Any) -> NetCDFWriteRequest:
        data = np.ndarray(
            (self.count, *self.row_shape),
            dtype=np.dtype(self.dtype),
            buffer=buffer,
            offset=self.offset,
        )
        return NetCDFWriteRequest(
            self.variable, data, self.output_path, self.times, self.chunk_cache
        )


@cache
def _is_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/version", encoding="utf-8") as stream:
            version = stream.read().lower()
        return "microsoft" in version or "wsl" in version
    except OSError:
        return False


# Least number of append handles a process keeps open; a writer raises it to
# the number of files it routes to that process so appends never thrash.
_WORKER_FILE_CACHE_SIZE = 32


def _worker_file_identity(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino


class StreamAppendError(RuntimeError):
    """Appends of some streams of one write task failed; the others landed."""

    def __init__(self, streams: tuple[str, ...], message: str) -> None:
        super().__init__(streams, message)
        self.streams = streams
        self.message = message

    def __str__(self) -> str:
        return f"NetCDF append failed for {list(self.streams)}: {self.message}"


class _AppendHandles:
    """Bounded LRU of persistent append handles owned by one process.

    A handle is reopened when its path was externally replaced.  A path whose
    append failed is refused afterwards: appending later rows would close the
    gap silently instead of leaving the uncommitted batch detectable.
    """

    def __init__(self, capacity: int = _WORKER_FILE_CACHE_SIZE) -> None:
        self.capacity = max(1, capacity)
        self._handles: OrderedDict[Path, tuple[nc.Dataset, tuple[int, int]]] = (
            OrderedDict()
        )
        self._failed: set[Path] = set()

    def open(self, path: Path) -> nc.Dataset:
        """Return an append handle of ``path``, opening it when needed."""

        canonical = path.absolute()
        if canonical in self._failed:
            raise RuntimeError(
                f"an earlier append to NetCDF output {canonical} failed; "
                "later rows are refused"
            )
        identity = _worker_file_identity(canonical)
        entry = self._handles.pop(canonical, None)
        if entry is not None:
            dataset, cached_identity = entry
            if cached_identity == identity:
                self._handles[canonical] = entry
                return dataset
            dataset.close()
        ensure_hdf5_plugins()
        dataset = nc.Dataset(canonical, "a")
        self._handles[canonical] = (dataset, identity)
        while len(self._handles) > self.capacity:
            _old_path, (old_dataset, _old_identity) = self._handles.popitem(last=False)
            old_dataset.close()
        return dataset

    def evict(self, path: Path) -> None:
        entry = self._handles.pop(path.absolute(), None)
        if entry is not None:
            entry[0].close()

    def release(self, paths: Iterable[Path]) -> None:
        """Close the handles of files that receive no further rows."""

        with cleanup_on_exit(
            "NetCDF append handle release",
            tuple(partial(self.evict, path) for path in paths),
        ):
            pass

    def close(self) -> None:
        """Close every handle, attempting all of them."""

        entries = tuple(self._handles.values())
        self._handles.clear()
        with cleanup_on_exit(
            "NetCDF append handles", tuple(dataset.close for dataset, _ in entries)
        ):
            pass

    def append(
        self, requests: tuple[NetCDFWriteRequest, ...]
    ) -> tuple[tuple[str, int], ...]:
        """Append each request in order through the persistent handles."""

        results = []
        failed: list[tuple[str, BaseException]] = []
        for index, request in enumerate(requests):
            path = request.output_path.absolute()
            try:
                results.append(_append_netcdf_request(self.open(path), request))
            except BaseException as error:
                # An ``Exception`` costs only its own file; anything else
                # leaves this batch and the rest of the task unappended.
                fatal = not isinstance(error, Exception)
                self._failed.update(
                    item.output_path.absolute()
                    for item in (requests[index:] if fatal else (request,))
                )
                try:
                    self.evict(path)
                except BaseException:
                    logger.exception(
                        "failed to evict NetCDF append handle after append failure: %s",
                        path,
                    )
                if fatal:
                    raise
                failed.append((request.variable, error))
        if len(failed) == 1 and len(requests) == 1:
            raise failed[0][1]
        if failed:
            raise StreamAppendError(
                tuple(variable for variable, _error in failed),
                "; ".join(
                    f"{variable}: {failure_description(error)['message']}"
                    for variable, error in failed
                ),
            ) from failed[0][1]
        return tuple(results)


# Append handles of an output worker process, sized by its initializer.
_WORKER_HANDLES = _AppendHandles()
# In-process append handles of live writers, closed before any fork so no
# HDF5 handle open for writing is duplicated into a child process.
_OWNER_HANDLES: WeakSet[_AppendHandles] = WeakSet()


def _close_owner_handles_before_fork() -> None:
    for handles in tuple(_OWNER_HANDLES):
        try:
            handles.close()
        except BaseException:
            logger.exception("failed to close NetCDF append handles before fork")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_close_owner_handles_before_fork)


def _initialize_netcdf_worker(capacity: int = _WORKER_FILE_CACHE_SIZE) -> None:
    """Size the handle cache and install cleanup in one spawned output process."""

    global _WORKER_HANDLES
    _WORKER_HANDLES = _AppendHandles(capacity)
    atexit.register(_release_worker)


def _release_worker() -> None:
    """Close the cached files and ring mappings of one output worker."""

    with cleanup_on_exit("NetCDF output worker", (_WORKER_HANDLES.close, detach_rings)):
        pass


def _close_worker_netcdf_paths(paths: tuple[Path, ...]) -> None:
    """Release cached append handles of files that receive no further rows."""

    _WORKER_HANDLES.release(paths)


def _find_data_variable(ncfile: nc.Dataset, var_name: str) -> str:
    """Locate the target data variable inside an open NetCDF dataset."""
    safe = sanitize_symbol(var_name)
    if var_name in ncfile.variables:
        return var_name
    if safe in ncfile.variables:
        return safe
    raise KeyError(
        f"Could not find variable for '{var_name}' (safe: '{safe}') in {ncfile.filepath()}"
    )


def _wsl_drop_cache(output_path: Path) -> None:
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

    time_var = ncfile.variables[TIME_DIM]
    target = _find_data_variable(ncfile, request.variable)
    variable = ncfile.variables[target]
    if (
        request.chunk_cache is not None
        and variable.get_var_chunk_cache()[0] != request.chunk_cache
    ):
        # createVariable's cache size lasts only until that Dataset closes.
        # Apply the same budget to the handle that actually appends the rows.
        variable.set_var_chunk_cache(size=request.chunk_cache)
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
    if current_len < 0:
        raise ValueError(
            f"invalid NetCDF append length in {request.output_path}: "
            f"committed={current_len}"
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


def _write_ring_slots(
    ring: str, writes: tuple[_SlotWrite, ...]
) -> tuple[tuple[str, int], ...]:
    """Append slots of a shared ring through worker-local file handles."""

    buffer = attached_ring(ring)
    return _WORKER_HANDLES.append(tuple(write.request(buffer) for write in writes))


def _output_components(variable: str, order: int) -> tuple[tuple[str, str, str], ...]:
    """Return ``(file variable, long name, description suffix)`` per component.

    An order-``k`` output stores each of its ``k`` components in its own file.
    """

    safe_name = sanitize_symbol(variable)
    if order == 1:
        return ((safe_name, variable, ""),)
    return tuple(
        (f"{safe_name}_{index}", f"{variable}_{index}", f" [rank {index}]")
        for index in range(order)
    )


def _create_output_files(
    directory: Path,
    variable: str,
    schema: NetCDFSchema,
    *,
    header: RankFileHeader,
    coordinate_values: np.ndarray | None,
    year: int | None,
    calendar: str,
    static_variables: Mapping[str, Mapping[str, Any]],
    ensemble_member_ids: tuple[int, ...] | None,
) -> tuple[Path, ...]:
    """Create one empty rank file per declared output component."""

    paths = []
    for file_variable, long_name, description_suffix in _output_components(
        variable, schema.order
    ):
        output_path = directory / rank_file_name(file_variable, header.rank, year)
        paths.append(output_path)
        with atomic_netcdf_dataset(output_path, format="NETCDF4") as ncfile:
            ncfile.setncattr(
                "title", f"Time series for rank {header.rank}: {long_name}"
            )
            ncfile.setncattr("original_variable_name", long_name)
            header.write(ncfile)

            # The unlimited time axis is the first dimension of every file.
            ncfile.createDimension(TIME_DIM, None)
            for dimension, extent in schema.dimensions:
                ncfile.createDimension(dimension, extent)

            if schema.batched:
                member_ids = ensemble_member_ids
                if member_ids is None:
                    member_ids = tuple(range(dict(schema.dimensions)["ensemble"]))
                ensemble_coordinate = ncfile.createVariable(
                    "ensemble", "i8", ("ensemble",)
                )
                ensemble_coordinate.setncattr(COMPLETE_DATA_ATTR, "true")
                ensemble_coordinate[:] = member_ids

            write_point_coordinate(ncfile, schema.coordinate_name, coordinate_values)

            # Write user-supplied static per-point variables. Coordinate scope
            # decides applicability; once applicable, dimension mismatch is a
            # schema error rather than a silent omission.
            for sv_name, sv_spec in static_variables.items():
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
                sv_var.setncattr(COMPLETE_DATA_ATTR, "true")

            create_time_axis(ncfile, calendar=calendar)

            nc_var = create_netcdf_variable(
                ncfile,
                file_variable,
                schema.dtype,
                schema.data_dimensions,
                options=schema.create_options,
            )
            if schema.logical_dtype is not None:
                nc_var.setncattr(LOGICAL_DTYPE_ATTR, schema.logical_dtype)
            nc_var.setncattr(COMPLETE_DATA_ATTR, "true")
            nc_var.setncattr("description", schema.description + description_suffix)
            nc_var.setncattr("actual_shape", str(schema.file_actual_shape))
            nc_var.setncattr("tensor_shape", str(schema.tensor_shape))
            nc_var.setncattr("long_name", long_name)
    return tuple(paths)


@dataclass(frozen=True, slots=True)
class PendingNetCDFWrite:
    """One background task and the rows it appends per stream."""

    step_counts: tuple[tuple[str, int], ...]
    future: Future[Any]


@dataclass(eq=False, slots=True)
class _OutputStream:
    """One output component: its rank file, its ring slots and their fill."""

    key: str
    component: int | None
    worker: int | None
    slots: RingSlots
    chunk_cache: int | None = None
    path: Path | None = None
    # Slot receiving rows and the rows placed in it; a full slot is appended
    # once its last row landed, and the next row moves on to the next slot.
    slot: int = 0
    rows: int = 0
    # Per slot: rows already submitted by a durability flush, and the times
    # of the rows placed since.  A flushed slot keeps filling, so later
    # appends stay aligned with the file's time chunks.
    submitted: list[int] = field(default_factory=list)
    times: list[list[Any]] = field(default_factory=list)
    writes: list[PendingNetCDFWrite | None] = field(default_factory=list)
    buffers: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class _StagedStep:
    """One output step whose device-to-host copies may still be in flight."""

    rows: tuple[tuple[_OutputStream, int, int], ...]
    flags: tuple[tuple[torch.Tensor, str, str], ...]
    flag_row: int
    event: Any


class RankOutputWriter:
    """Own the rank files of one output run, their ring, workers and lifetime.

    Each finalized row is copied once into its ring slot; a full slot is
    appended to its file by a worker (in process with ``num_workers=0``) and
    reused after that append, so the ring bounds all retained output memory.
    From a CUDA device the copy runs beside compute and is checked one step
    later; other devices copy synchronously.
    """

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
        num_workers: int,
        output_split_by_year: bool,
        max_pending_steps: int,
        save_precision: torch.dtype | None,
        device: torch.device,
        event_sink: Any,
        run_id: str | None = None,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.ensemble_member_ids = ensemble_member_ids
        self.calendar = calendar
        self.num_workers = num_workers
        self.output_split_by_year = output_split_by_year
        self.max_pending_steps = max_pending_steps
        self.save_precision = save_precision
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.type == "cuda" and device.index is None
            else device
        )
        self.event_sink = event_sink
        self.run_id = run_id
        self._on_failure = on_failure
        self.static_vars = static_vars
        self._coord_cache = coordinates
        self._closed = False
        self._current_year = None
        self._files_created = False
        self._write_executors: list[ProcessPoolExecutor] = []
        self._pending_writes: list[PendingNetCDFWrite] = []
        self._background_failure: BaseException | None = None
        # Output streams whose buffered rows a failure invalidated; ``None``
        # once a failure invalidated the rows of every stream.
        self._poisoned: set[str] | None = set()
        self._unsynced_paths: set[Path] = set()
        self._ring: OutputRing | None = None
        self._asynchronous = False
        self._copy_stream: Any = None
        self._events: tuple[Any, ...] = ()
        self._flag_rows: tuple[torch.Tensor, ...] = ()
        self._flag_row = 0
        # Steps whose copies were enqueued, oldest first; a step stays here
        # until it landed, so an interrupted wait cannot lose its rows.
        self._staged: deque[_StagedStep] = deque()
        # Exact storage layout and saved dtype of every output.
        self._layouts: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._dtypes: dict[str, torch.dtype] = {}
        self._last_times: dict[str, float] = {}
        step_bytes: dict[str, int] = {}
        for name, info in metadata.items():
            shape = tuple(info["actual_shape"])
            self._layouts[name] = (shape, info["dtype"])
            dtype = self._dtypes[name] = saved_dtype(info["dtype"], save_precision)
            step_bytes[name] = math.prod(shape) * dtype.itemsize
        plans = plan_output_batches(
            step_bytes,
            max_pending_steps=max_pending_steps,
            background_writes=bool(num_workers),
        )
        self._page_lock = page_lockable(step_bytes, plans)
        self._netcdf_schemas = {
            name: NetCDFSchema.compile(
                info,
                variable=name,
                ensemble_size=ensemble_size,
                netcdf_options=variable_options[name],
                write_batch_size=plans[name][1],
                save_precision=save_precision,
            )
            for name, info in metadata.items()
        }
        file_variables: dict[str, str] = {}
        for name, schema in self._netcdf_schemas.items():
            for file_variable, _long_name, _suffix in _output_components(
                name, schema.order
            ):
                other = file_variables.setdefault(file_variable, name)
                if other != name:
                    raise ValueError(
                        f"statistics outputs {other!r} and {name!r} would both "
                        f"write the rank files of {file_variable!r}"
                    )
        rows = {}
        routes = []
        for name, schema in self._netcdf_schemas.items():
            dtype = np.dtype(torch_to_numpy_dtype(self._dtypes[name]))
            for component in range(schema.order):
                key = f"{name}_{component}" if schema.order > 1 else name
                rows[key] = (schema.file_actual_shape, dtype, *plans[name])
                routes.append((name, key, component if schema.order > 1 else None))
        placements, self._flags_offset = place_slots(rows)
        self._ring_bytes = self._flags_offset + 2 * len(metadata)
        self._streams: dict[str, list[_OutputStream]] = {name: [] for name in metadata}
        for name, key, component in routes:
            self._streams[name].append(
                _OutputStream(
                    key=key,
                    component=component,
                    worker=None,
                    slots=placements[key],
                    chunk_cache=self._netcdf_schemas[name].create_options.get(
                        "chunk_cache"
                    ),
                    submitted=[0] * placements[key].depth,
                    times=[[] for _slot in range(placements[key].depth)],
                    writes=[None] * placements[key].depth,
                )
            )
        if num_workers:
            loads = [0] * num_workers
            # Keep every file on one worker, balancing the bytes per output
            # step rather than sending large interleaved streams to one worker.
            for stream in sorted(
                self._all_streams(), key=lambda item: item.slots.row_bytes, reverse=True
            ):
                stream.worker = min(range(num_workers), key=loads.__getitem__)
                loads[stream.worker] += max(1, stream.slots.row_bytes)
        # Every file a process appends keeps its handle open.
        self._worker_files = [
            max(
                _WORKER_FILE_CACHE_SIZE,
                1 + sum(stream.worker == worker for stream in self._all_streams()),
            )
            for worker in range(num_workers)
        ]
        self._local_handles = _AppendHandles(
            max(_WORKER_FILE_CACHE_SIZE, 1 + len(self._all_streams()))
        )
        _OWNER_HANDLES.add(self._local_handles)
        try:
            start_blosc_zstd_probe(
                schema.create_options for schema in self._netcdf_schemas.values()
            )
        except (OSError, subprocess.SubprocessError):
            logger.warning("could not start the Blosc capability probe early")

    def sync_appended_files(self) -> None:
        """Fsync every file appended since the last durability boundary."""

        with cleanup_on_exit(
            "NetCDF output fsync",
            tuple(partial(fsync_file, path) for path in sorted(self._unsynced_paths)),
        ):
            pass
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

    def _poison(self, keys: Iterable[str]) -> None:
        """Mark streams whose buffered rows may no longer reach their files."""

        if self._poisoned is not None:
            self._poisoned.update(keys)

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

    def set_run_id(self, run_id: str) -> None:
        """Adopt the output identity shared by every rank of one run."""

        self.run_id = run_id

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
        header = RankFileHeader(
            rank=self.rank,
            world_size=self.world_size,
            run_id=run_id,
        )
        planned_by_variable: dict[str, tuple[Path, ...]] = {}
        batch_events: list[ModelEvent] = []
        for out_name, schema in self._netcdf_schemas.items():
            planned_by_variable[out_name] = tuple(
                self.output_dir / rank_file_name(file_variable, self.rank, year)
                for file_variable, _name, _suffix in _output_components(
                    out_name, schema.order
                )
            )
            batch_events.append(
                ModelEvent(
                    level="info",
                    name="output.batch_configured",
                    message="Configured NetCDF write batch",
                    fields={
                        "output": out_name,
                        "steps": schema.write_batch_size,
                        "elements_per_step": max(
                            1, math.prod(schema.file_actual_shape)
                        ),
                        "storage_bytes_per_element": np.dtype(schema.dtype).itemsize,
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
        moves: list[tuple[Path, Path]] = []
        preserve_staging = False
        try:
            stage_output.mkdir()
            backup_dir.mkdir()
            for out_name, schema in self._netcdf_schemas.items():
                expected = planned_by_variable[out_name]
                coordinate = schema.output_coordinate
                _create_output_files(
                    stage_output,
                    out_name,
                    schema,
                    header=header,
                    coordinate_values=(
                        None if coordinate is None else self._coord_cache[coordinate]
                    ),
                    year=year,
                    calendar=self.calendar,
                    static_variables=self.static_vars,
                    ensemble_member_ids=self.ensemble_member_ids,
                )
                expected_staged = [
                    stage_output / final_path.name for final_path in expected
                ]
                if any(not path.is_file() for path in expected_staged):
                    raise RuntimeError(
                        f"NetCDF creation for {out_name!r} did not "
                        "materialize every staged file"
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
                    # Keep the old permissions, but the run appends to the
                    # new file: a read-only predecessor must not lock it.
                    source.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)

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
        for out_name, streams in self._streams.items():
            for stream, path in zip(
                streams, planned_by_variable[out_name], strict=True
            ):
                stream.path = path
        self._files_created = True

        for event in batch_events:
            self._emit_event(event)
        for _source, path in moves:
            self._emit_event(
                ModelEvent(
                    level="info",
                    name="output.file_created",
                    message="Created NetCDF file",
                    fields={"path": str(path)},
                )
            )
        self._emit_event(
            ModelEvent(
                level="info",
                name="output.create_complete",
                message="Created NetCDF files for streaming",
                fields={"files": len(moves)},
            )
        )

    def _all_streams(self) -> tuple[_OutputStream, ...]:
        return tuple(stream for streams in self._streams.values() for stream in streams)

    def _submit(self, slots: list[tuple[_OutputStream, int, int]]) -> None:
        """Append each slot's unsubmitted rows up to ``stop``, one task per worker.

        ``slots`` holds ``(stream, slot, stop)``; the rows before the slot's
        ``submitted`` mark were already appended by a durability flush.
        """

        by_worker: dict[int | None, list[tuple[_OutputStream, int, int]]] = {}
        for item in slots:
            by_worker.setdefault(item[0].worker, []).append(item)
        failures: list[BaseException] = []
        for worker, items in by_worker.items():
            writes = []
            counts = []
            for stream, slot, stop in items:
                layout = stream.slots
                start = stream.submitted[slot]
                writes.append(
                    _SlotWrite(
                        stream.key,
                        stream.path,
                        layout.slot_offset(slot) + start * layout.row_bytes,
                        stop - start,
                        layout.row_shape,
                        layout.dtype.str,
                        tuple(stream.times[slot]),
                        stream.chunk_cache,
                    )
                )
                counts.append((stream.key, stop - start))
                stream.times[slot] = []
                stream.submitted[slot] = stop
            self._unsynced_paths.update(write.output_path for write in writes)
            if worker is None:
                # In process, one failed file does not hold back the others.
                for index, write in enumerate(writes):
                    try:
                        self._local_handles.append((write.request(self._ring.buffer),))
                    except Exception as error:
                        self._poison((counts[index][0],))
                        failures.append(error)
                    except BaseException:
                        self._poison(key for key, _count in counts[index:])
                        raise
                continue
            try:
                future = self._write_executors[worker].submit(
                    _write_ring_slots, self._ring.name, tuple(writes)
                )
            except BaseException as error:
                self._poison(key for key, _count in counts)
                failures.append(error)
                continue
            pending = PendingNetCDFWrite(step_counts=tuple(counts), future=future)
            self._pending_writes.append(pending)
            for stream, slot, _stop in items:
                stream.writes[slot] = pending
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("NetCDF write submission", failures)

    def _submit_full(self, rows: tuple[tuple[_OutputStream, int, int], ...]) -> None:
        """Append every slot whose last row just landed."""

        full = [
            (stream, slot, stream.slots.batch)
            for stream, slot, row in rows
            if row == stream.slots.batch - 1
        ]
        if not full:
            return
        try:
            self._submit(full)
        except BaseException as error:
            self._latch_failures("NetCDF write submission", [error])

    def _partial_slots(self) -> list[tuple[_OutputStream, int, int]]:
        """Return each partly filled slot holding rows not yet submitted."""

        return [
            (stream, stream.slot, stream.rows)
            for stream in self._all_streams()
            if stream.submitted[stream.slot] < stream.rows < stream.slots.batch
        ]

    def _next_row(
        self, stream: _OutputStream, *, dt: datetime | cftime.datetime
    ) -> tuple[int, int]:
        """Return where the stream's next row goes, once that slot is free."""

        slot, row = stream.slot, stream.rows
        if row == stream.slots.batch:
            slot, row = (slot + 1) % stream.slots.depth, 0
            if slot == stream.slot:
                # A single slot is appended once its last row, which may
                # still be in flight, landed.
                self._land_staged()
        pending = stream.writes[slot]
        if row == 0 and pending is not None:
            # The ring bounds retained rows: reuse waits for the slot's last
            # append (a stream's appends complete in submission order).  An
            # interrupted wait keeps the append tracked for a later wait.
            if pending in self._pending_writes:
                try:
                    self._wait_for(pending, dt=dt)
                except Exception as error:
                    self._pending_writes.remove(pending)
                    stream.writes[slot] = None
                    self._latch_failures("NetCDF slot reuse", [error])
                self._pending_writes.remove(pending)
            stream.writes[slot] = None
        return slot, row

    @staticmethod
    def _place(
        rows: list[tuple[_OutputStream, int, int]], dt: datetime | cftime.datetime
    ) -> None:
        for stream, slot, row in rows:
            stream.slot, stream.rows = slot, row + 1
            if row == 0:
                stream.times[slot] = []
                stream.submitted[slot] = 0
            stream.times[slot].append(dt)

    def _layout_of(self, name: str, storage: torch.Tensor) -> torch.dtype:
        if name not in self._layouts:
            raise ValueError(f"unknown statistics output {name!r}")
        if not isinstance(storage, torch.Tensor) or storage.layout != torch.strided:
            raise TypeError(f"statistics output {name!r} requires a strided tensor")
        if self._asynchronous and storage.device != self.device:
            raise ValueError(
                f"asynchronous statistics output {name!r} must be on {self.device}"
            )
        shape, dtype = self._layouts[name]
        if tuple(storage.shape) != shape or storage.dtype != dtype:
            raise TypeError(
                f"statistics output {name!r} changed layout: expected {shape} "
                f"{dtype}, got {tuple(storage.shape)} {storage.dtype}"
            )
        return self._dtypes[name]

    def _write_step(
        self, values: Mapping[str, torch.Tensor], dt: datetime | cftime.datetime
    ) -> None:
        """Copy one step into the ring, check it, and append full slots."""

        flags: list[tuple[torch.Tensor, str, str]] = []
        rows: list[tuple[_OutputStream, int, int]] = []
        for name, storage in values.items():
            dtype = self._dtypes[name]
            # Export logical values, never the integer hi/lo carrier. Take
            # one decoded snapshot for all components and narrowing checks.
            storage = decoded_output_tensor(storage)
            flag = narrowing_flag(storage, dtype, name=name)
            if flag is not None:
                flags.append(flag)
            for stream in self._streams[name]:
                slot, row = self._next_row(stream, dt=dt)
                source = (
                    storage
                    if stream.component is None
                    else storage[..., stream.component]
                ).detach()
                target = stream.buffers[slot][row]
                try:
                    if source.device.type == "mps" and (
                        target.data_ptr() % 4 or source.dtype != target.dtype
                    ):
                        # MPS copies into host memory need 4-byte aligned targets
                        # (rows of 1- or 2-byte values may start unaligned in a
                        # slot), and a float32 row widened into a float64 slot is
                        # left unwritten (torch 2.14): convert on the host.
                        source = source.cpu()
                    target.copy_(source)
                finally:
                    # A retained copy/narrowing failure must not keep a view
                    # alive after close drops the writer's ring buffers.
                    del target
                rows.append((stream, slot, row))
        # A rejected step leaves no row behind and the writer usable.
        raise_narrowing_failures(flags)
        self._place(rows, dt)
        self._submit_full(tuple(rows))

    def _stage(
        self, values: Mapping[str, torch.Tensor], dt: datetime | cftime.datetime
    ) -> None:
        """Snapshot on the device, enqueue the copies into the ring, queue the step.

        Ring views stay temporaries here and in :meth:`_land`, so a raised
        failure never holds one and the ring can still be released.
        """

        flags: list[tuple[torch.Tensor, str, str]] = []
        rows: list[tuple[_OutputStream, int, int]] = []
        snapshots: list[torch.Tensor] = []
        for name, storage in values.items():
            dtype = self._dtypes[name]
            flag = None
            for stream in self._streams[name]:
                rows.append((stream, *self._next_row(stream, dt=dt)))
                source = (
                    storage
                    if stream.component is None
                    else storage[..., stream.component]
                ).detach()
                # A deferred copy must not observe the next step's updates.
                snapshot = source.to(
                    dtype=dtype, memory_format=torch.contiguous_format, copy=True
                )
                snapshots.append(snapshot)
                # The converted snapshot reveals overflow without a temporary.
                entry = narrowing_flag(source, dtype, name=name, converted=snapshot)
                if entry is not None:
                    flag = entry if flag is None else (flag[0] | entry[0], *entry[1:])
            if flag is not None:
                flags.append(flag)
        if flags:
            snapshots.append(torch.stack([flag for flag, _name, _label in flags]))
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._events = (torch.cuda.Event(), torch.cuda.Event())
        copy_stream = self._copy_stream
        # Copy beside compute: the side stream starts after the queued
        # producers and never delays later kernels.
        copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        flag_row = self._flag_row
        with torch.cuda.stream(copy_stream):
            for (stream, slot, row), snapshot in zip(
                rows, snapshots[: len(rows)], strict=True
            ):
                stream.buffers[slot][row].copy_(snapshot, non_blocking=True)
            if flags:
                # The stacked flags follow the rows' snapshots.
                self._flag_rows[flag_row][: len(flags)].copy_(
                    snapshots[-1], non_blocking=True
                )
            for snapshot in snapshots:
                snapshot.record_stream(copy_stream)
            event = self._events[flag_row]
            event.record(copy_stream)
        self._flag_row = 1 - flag_row
        self._place(rows, dt)
        self._staged.append(_StagedStep(tuple(rows), tuple(flags), flag_row, event))

    def _land(self, staged: _StagedStep) -> None:
        """Wait for a staged step and check its flags."""

        staged.event.synchronize()
        if staged.flags:
            # Host values, so a raised failure holds no view of the ring.
            raise_narrowing_failures(
                staged.flags,
                self._flag_rows[staged.flag_row][: len(staged.flags)].tolist(),
            )

    def _land_staged(self, *, keep: int = 0) -> None:
        """Land queued steps, oldest first, until ``keep`` remain in flight.

        An interrupted wait leaves its step queued for a later landing.
        """

        while len(self._staged) > keep:
            staged = self._staged[0]
            try:
                self._land(staged)
            except Exception as error:
                # The rejected step's rows already sit in every stream's slot.
                self._poisoned = None
                # The newest step's copies still land before the failure.
                with cleanup_on_exit(
                    "statistics output staging", (self._staged[-1].event.synchronize,)
                ):
                    self._latch_failures("statistics output staging", [error])
            self._staged.popleft()
            self._submit_full(staged.rows)

    def _flush_all_write_buffers(
        self, *, latch: bool = True, next_slot: bool = False
    ) -> None:
        """Append every partly filled slot (year transition, durability, close).

        A flushed slot keeps receiving rows, so later appends stay aligned
        with the file's time chunks; ``next_slot`` instead starts each
        stream's next row in a fresh slot, aligned for a new file.
        """
        self._raise_if_background_failed()
        self._land_staged()
        try:
            partial = self._partial_slots()
            if partial:
                self._submit(partial)
        except BaseException as error:
            if latch:
                self._latch_failures("NetCDF write buffers", [error])
            raise
        if next_slot:
            for stream in self._all_streams():
                if 0 < stream.rows < stream.slots.batch:
                    stream.rows = stream.slots.batch

    def _flush_surviving_buffers(self) -> None:
        """After a failure, append the buffered rows the failure left intact."""

        poisoned = self._poisoned
        if poisoned is None or self._ring is None:
            return
        slots: list[tuple[_OutputStream, int, int]] = []
        while self._staged:
            staged = self._staged[0]
            # A step that failed its range check must not reach any file.
            self._land(staged)
            self._staged.popleft()
            slots.extend(
                (stream, slot, stream.slots.batch)
                for stream, slot, row in staged.rows
                if row == stream.slots.batch - 1
            )
        slots.extend(self._partial_slots())
        slots = [item for item in slots if item[0].key not in poisoned]
        if slots:
            self._submit(slots)

    def _release_worker_files(self, paths: tuple[Path, ...]) -> None:
        """Close completed files in every worker after their queued appends."""

        failures: list[BaseException] = []
        try:
            self._local_handles.release(paths)
        except BaseException as error:
            failures.append(error)
        for executor in self._write_executors:
            try:
                future = executor.submit(_close_worker_netcdf_paths, paths)
            except BaseException as error:
                failures.append(error)
                continue
            self._pending_writes.append(PendingNetCDFWrite((), future))
        self._latch_failures("NetCDF worker file release", failures)

    def _wait_for(
        self,
        pending: PendingNetCDFWrite,
        *,
        dt: datetime | cftime.datetime | None,
    ) -> None:
        try:
            pending.future.result()
        except Exception as exc:
            # A partial task failure names the streams that did not land.
            outputs = (
                exc.streams
                if isinstance(exc, StreamAppendError)
                else tuple(key for key, _count in pending.step_counts)
            )
            self._poison(outputs)
            self._emit_event(
                ModelEvent(
                    level="error",
                    name="output.write_failed",
                    message="Failed to write time step",
                    fields={
                        "output": outputs[0] if len(outputs) == 1 else outputs,
                        "outputs": outputs,
                        "steps": {
                            key: count
                            for key, count in pending.step_counts
                            if key in outputs
                        },
                        "time": str(dt),
                        "error": failure_description(exc)["message"],
                    },
                )
            )
            raise

    def poll(self, dt: datetime | cftime.datetime | None) -> None:
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
            except Exception as error:
                failures.append(error)
        self._pending_writes = remaining
        if failures:
            self._latch_failures(
                "completed NetCDF background writes",
                failures,
            )

    def flush(self, dt: datetime | cftime.datetime | None) -> None:
        """Make every accepted statistics row durable without closing workers."""

        self._raise_if_background_failed()
        failures: list[BaseException] = []
        try:
            self._flush_all_write_buffers(latch=False)
        except Exception as error:
            failures.append(error)
        pending, self._pending_writes = self._pending_writes, []
        for index, item in enumerate(pending):
            try:
                self._wait_for(item, dt=dt)
            except Exception as error:
                failures.append(error)
            except BaseException:
                # Interrupted: the unobserved appends stay tracked.
                self._pending_writes[:0] = pending[index:]
                raise
        if not failures:
            # Workers have synced HDF5 buffers and committed-step markers to
            # the OS; fsync makes those committed rows survive a crash.
            try:
                self.sync_appended_files()
            except Exception as error:
                self._poisoned = None
                failures.append(error)
        if failures:
            self._latch_failures(
                "NetCDF output durability boundary",
                failures,
            )

    def reset(self) -> None:
        """Land the in-flight step before the output timeline restarts."""

        self._land_staged()

    def _prepare_output_files(self, dt: datetime | cftime.datetime) -> None:
        """Create the files that receive rows stamped ``dt``."""

        if self.output_split_by_year:
            if self._current_year is None:
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
            elif self._current_year != dt.year:
                # Rows of the old year go to its files before the new ones,
                # and the new files start in fresh, chunk-aligned slots.
                self._flush_all_write_buffers(next_slot=True)
                self._release_worker_files(
                    tuple(stream.path for stream in self._all_streams())
                )
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
        elif not self._files_created:
            self._create_netcdf_files()

    def append(
        self, dt: datetime | cftime.datetime, values: Mapping[str, torch.Tensor]
    ) -> None:
        """Copy one finalized sample; the caller may then reuse its tensors."""
        self._raise_if_background_failed()
        if self._ring is None:
            raise RuntimeError("statistics output writer was not started")
        for name, storage in values.items():
            self._layout_of(name, storage)
        numeric_time = None
        if values:
            numeric_time = float(
                nc.date2num(dt, units=TIME_UNITS, calendar=self.calendar)
            )
            if not math.isfinite(numeric_time):
                raise ValueError("NetCDF write timestamps must be finite datetimes")
            if any(
                numeric_time <= self._last_times.get(name, -math.inf) for name in values
            ):
                raise ValueError("NetCDF write timestamps must be strictly increasing")
        self._prepare_output_files(dt)
        if not self._asynchronous:
            self._write_step(values, dt)
            self._last_times.update(dict.fromkeys(values, numeric_time))
            return
        # At most one step is in flight (an interrupted landing may leave
        # two, landed here first), so the two flag rows never overlap.
        self._land_staged(keep=1)
        self._stage(values, dt)
        # The previous step is checked while this one's copies run.
        self._land_staged(keep=1)
        self._last_times.update(dict.fromkeys(values, numeric_time))

    def _bind_ring(self, ring: OutputRing) -> None:
        """Map each stream's slots and the two flag rows onto ``ring``."""

        for stream in self._all_streams():
            layout = stream.slots
            stream.buffers = tuple(
                torch.from_numpy(
                    ring.array(
                        layout.slot_offset(slot),
                        (layout.batch, *layout.row_shape),
                        layout.dtype,
                    )
                )
                for slot in range(layout.depth)
            )
        count = len(self._streams)
        self._flag_rows = tuple(
            torch.from_numpy(
                ring.array(self._flags_offset + row * count, (count,), np.dtype(bool))
            )
            for row in range(2)
        )

    def _release_ring(self) -> None:
        """Wait for copies into the ring, drop its views, and release it."""

        ring, self._ring = self._ring, None
        self._staged.clear()
        for stream in self._all_streams():
            stream.buffers = ()
        self._flag_rows = ()
        if ring is None:
            return
        actions = [] if self._copy_stream is None else [self._copy_stream.synchronize]
        actions.append(ring.close)
        with cleanup_on_exit("statistics output ring", actions):
            pass

    def _shutdown(self) -> None:
        """Flush, drain and stop the workers, fsync, then release the ring.

        After a failure the rows of streams it left intact are still
        appended and every appended file is still made durable.
        """

        failures: list[BaseException] = []
        failure = self._background_failure
        try:
            if failure is None:
                self._flush_all_write_buffers()
            else:
                failures.append(failure)
                self._flush_surviving_buffers()
        except BaseException as error:
            if error is not failure:
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
                executor.submit(_release_worker).result()
            except BaseException as error:
                failures.append(error)
            try:
                executor.shutdown(wait=True)
            except BaseException as error:
                failures.append(error)
        try:
            self._local_handles.close()
        except BaseException as error:
            failures.append(error)
        # Every process has closed its handles; make the appended rows durable.
        try:
            self.sync_appended_files()
        except BaseException as error:
            failures.append(error)
        try:
            self._release_ring()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("statistics output workers", failures)

    def start(self) -> None:
        """Allocate the ring and start workers while initialization continues.

        A shared ring that ``/dev/shm`` cannot hold fails here; the error
        suggests ``workers=0``, which keeps a private ring in this process.
        """

        if self._ring is not None:
            raise RuntimeError("statistics output writer was already started")
        ring = OutputRing.create(
            self._ring_bytes,
            shared=self.num_workers > 0,
            device=self.device,
            page_lock=self._page_lock,
        )
        created = []
        ready = []
        try:
            self._ring = ring
            self._bind_ring(ring)
            for worker in range(self.num_workers):
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=get_context("spawn"),
                    initializer=_initialize_netcdf_worker,
                    initargs=(self._worker_files[worker],),
                )
                created.append(executor)
                ready.append(PendingNetCDFWrite((), executor.submit(os.getpid)))
        except BaseException as primary:
            failures: list[BaseException] = [primary]
            for executor in reversed(created):
                try:
                    executor.shutdown(wait=True)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            try:
                self._release_ring()
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
        self._asynchronous = ring.registered and self.max_pending_steps > 1
        if ring.registration_error is not None:
            self._emit_event(
                ModelEvent(
                    level="warning",
                    name="output.ring_pageable",
                    message=(
                        "The output ring is not page-locked; device copies "
                        "into it block"
                    ),
                    fields={"error": ring.registration_error},
                )
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with cleanup_on_exit(
            "statistics output",
            (lambda: _probe_blosc_zstd_filter(start_if_needed=False),),
        ):
            self._shutdown()

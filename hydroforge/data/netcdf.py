# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Shared NetCDF indexing and process-local read resources."""

import os
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, List, Optional, Tuple, Union
from weakref import WeakSet

import cftime
import numpy as np
from netCDF4 import Dataset

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.contracts.validation import HydroForgeModel


_READ_HANDLE_POOLS: WeakSet[Any] = WeakSet()
_READ_HANDLE_POOLS_LOCK = RLock()
_NETCDF_VARIABLE_CACHE_BYTES = 80 * 1024 * 1024
_NETCDF_LOGICAL_CHUNK_BYTES = 256 * 1024 * 1024
# A common two-worker/prefetch-two loader can retain four queued chunks plus
# the chunk being consumed.  Keep compact exported payloads near one variable
# cache budget across those five in-flight positions.
_EXPORTED_NETCDF_CHUNK_BYTES = _NETCDF_VARIABLE_CACHE_BYTES // 5


def _close_netcdf_read_handles_before_fork() -> None:
    """Prevent native NetCDF/HDF5 handles from crossing a process fork."""

    with _READ_HANDLE_POOLS_LOCK:
        pools = tuple(_READ_HANDLE_POOLS)
    for pool in pools:
        try:
            pool.close()
        except BaseException:
            # ``close`` clears ownership before reporting cleanup failures.
            # A fork hook cannot safely propagate such an exception.
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_close_netcdf_read_handles_before_fork)


class _NetCDFReadHandlePool:
    """Bounded lazy NetCDF handle cache owned by exactly one process.

    Handles are deliberately opened only on the first runtime read.  A pool
    copied into a DataLoader worker never reuses a handle created by its parent
    process, and spawn/forkserver pickling drops all live native resources.
    """

    def __init__(self, max_open_files: int = 8) -> None:
        if type(max_open_files) is not int or max_open_files < 1:
            raise ValueError("max_open_files must be a positive exact int")
        self.max_open_files = max_open_files
        self._pid = os.getpid()
        self._handles: OrderedDict[Path, Dataset] = OrderedDict()
        self._lock = RLock()
        with _READ_HANDLE_POOLS_LOCK:
            _READ_HANDLE_POOLS.add(self)

    def _reset_for_process(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        # Native HDF5 state inherited across fork must never be used in the
        # child.  Runtime pools are normally empty before workers start; clear
        # defensively if a caller did perform an earlier main-process read.
        self._handles = OrderedDict()
        self._lock = RLock()
        self._pid = pid

    def _open_locked(self, path: str | Path) -> Dataset:
        """Return one live handle while the process-local lock is held."""

        canonical = Path(path).absolute()
        dataset = self._handles.pop(canonical, None)
        if dataset is not None and dataset.isopen():
            self._handles[canonical] = dataset
            return dataset
        dataset = Dataset(canonical, "r")
        self._handles[canonical] = dataset
        while len(self._handles) > self.max_open_files:
            _old_path, old_dataset = self._handles.popitem(last=False)
            old_dataset.close()
        return dataset

    @contextmanager
    def acquire(self, path: str | Path) -> Iterator[Dataset]:
        """Serialize use of one persistent process-local read handle."""

        self._reset_for_process()
        with self._lock:
            yield self._open_locked(path)

    def close(self) -> None:
        """Close all handles owned by the current process, idempotently."""

        self._reset_for_process()
        failures: list[BaseException] = []
        with self._lock:
            handles = tuple(self._handles.values())
            self._handles.clear()
            for dataset in handles:
                try:
                    if dataset.isopen():
                        dataset.close()
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise ResourceCleanupError("NetCDF read handles", tuple(failures))

    def __getstate__(self) -> dict[str, int]:
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(max_open_files=state["max_open_files"])

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


def _planned_netcdf_chunk_len(
    path: str | Path,
    var_name: str,
    *,
    fallback: int = 24,
    max_steps: int = 256,
    max_bytes: int = _NETCDF_LOGICAL_CHUNK_BYTES,
    physical_chunk_max_bytes: int | None = None,
    physical_chunk_multiplier: int = 1,
    step_alignment: int = 1,
) -> int:
    """Choose a bounded logical batch from one variable's physical layout."""

    if physical_chunk_multiplier < 1:
        raise ValueError("physical_chunk_multiplier must be positive")
    if step_alignment < 1:
        raise ValueError("step_alignment must be positive")

    with Dataset(Path(path), "r") as dataset:
        variable = dataset.variables[var_name]
        chunking = variable.chunking()
        time_axes = [
            index for index, name in enumerate(variable.dimensions)
            if name in {"time", "valid_time"}
        ]
        if len(time_axes) != 1:
            return fallback
        time_axis = time_axes[0]
        element_bytes = np.dtype(variable.dtype).itemsize
        bytes_per_step = element_bytes * int(np.prod(
            [
                size for index, size in enumerate(variable.shape)
                if index != time_axis
            ],
            dtype=np.int64,
        ))
        memory_steps = max(1, max_bytes // max(1, bytes_per_step))
        if chunking == "contiguous" or not chunking:
            return max(1, min(fallback, max_steps, memory_steps))
        physical_steps = int(chunking[time_axis])
        if (
            physical_chunk_max_bytes is not None
            and physical_steps * bytes_per_step <= physical_chunk_max_bytes
        ):
            memory_steps = max(memory_steps, physical_steps)
        target_steps = max(
            fallback,
            physical_steps * physical_chunk_multiplier,
        )
        capacity_steps = min(max_steps, memory_steps)
        if step_alignment > 1:
            target_steps = (
                (target_steps + step_alignment - 1) // step_alignment
            ) * step_alignment
            aligned_capacity = (
                capacity_steps // step_alignment
            ) * step_alignment
            if aligned_capacity >= step_alignment:
                capacity_steps = aligned_capacity
        return max(1, min(target_steps, capacity_steps))


def _planned_exported_netcdf_chunk_len(
    path: str | Path,
    var_name: str,
) -> int:
    """Plan compact point runoff for the default prefetched read pipeline."""

    return _planned_netcdf_chunk_len(
        path,
        var_name,
        max_bytes=_EXPORTED_NETCDF_CHUNK_BYTES,
        physical_chunk_max_bytes=_NETCDF_LOGICAL_CHUNK_BYTES,
    )


def _configure_netcdf_variable_cache(
    variable: Any,
    selectors: tuple[Any, ...],
    *,
    time_axis: int,
    max_bytes: int = _NETCDF_VARIABLE_CACHE_BYTES,
) -> None:
    """Size a variable cache for one touched physical time slab."""

    chunking = variable.chunking()
    if chunking == "contiguous" or not chunking:
        return
    chunk_shape = tuple(int(value) for value in chunking)
    touched_chunks = 1
    for axis, (selector, chunk_size, axis_size) in enumerate(zip(
        selectors, chunk_shape, variable.shape, strict=True,
    )):
        if axis == time_axis:
            continue
        if isinstance(selector, slice):
            start, stop, step = selector.indices(axis_size)
            if stop <= start:
                return
            if step == 1:
                first = start // chunk_size
                last = (stop - 1) // chunk_size
                count = last - first + 1
            else:
                count = np.unique(
                    np.arange(start, stop, step, dtype=np.int64) // chunk_size,
                ).size
        elif isinstance(selector, np.ndarray):
            count = np.unique(selector // chunk_size).size
        else:
            count = 1
        touched_chunks *= max(1, int(count))
    chunk_bytes = int(np.prod(chunk_shape, dtype=np.int64)) * np.dtype(
        variable.dtype,
    ).itemsize
    desired_bytes = min(max_bytes, max(chunk_bytes, touched_chunks * chunk_bytes))
    current_bytes, current_elements, preemption = variable.get_var_chunk_cache()
    if desired_bytes <= current_bytes:
        return
    variable.set_var_chunk_cache(
        size=desired_bytes,
        nelems=max(current_elements, touched_chunks * 2 + 1),
        preemption=preemption,
    )


class _TimeKeyRequest(HydroForgeModel):
    value: datetime | cftime.datetime


def _single_file_key_trusted(dt: datetime | cftime.datetime) -> str:
    del dt
    return ""


def _daily_time_to_key_trusted(dt: datetime | cftime.datetime) -> str:
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"


def _yearly_time_to_key_trusted(dt: datetime | cftime.datetime) -> str:
    return f"{dt.year}"


def _monthly_time_to_key_trusted(dt: datetime | cftime.datetime) -> str:
    return f"{dt.year:04d}_{dt.month:02d}"


def single_file_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Constant key for single-file mode."""
    return _single_file_key_trusted(_TimeKeyRequest(value=dt).value)


def daily_time_to_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Default time-to-file key: one file per day (YYYYMMDD)."""
    return _daily_time_to_key_trusted(dt)


def yearly_time_to_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Default time-to-file key: one file per year."""
    return _yearly_time_to_key_trusted(_TimeKeyRequest(value=dt).value)


def monthly_time_to_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Default time-to-file key: one file per month (YYYY_MM)."""
    return _monthly_time_to_key_trusted(_TimeKeyRequest(value=dt).value)


def read_netcdf_var_sliced(var: Any, index: Any = None) -> np.ndarray:
    """Read a NetCDF variable using only slices for sequence indices.

    Integer or boolean sequence selectors are read as one or more contiguous
    slices, then reordered in memory to match the requested index order.
    """
    selectors = list(_normalize_netcdf_index(index, var.ndim))
    shape = tuple(var.shape)
    for axis, selector in enumerate(selectors):
        if np.ma.isMaskedArray(selector):
            raise TypeError("NetCDF selectors must not be masked arrays")
        if isinstance(selector, (bool, np.bool_)):
            raise TypeError("NetCDF scalar boolean selectors are invalid")
        if isinstance(selector, slice):
            selectors[axis] = _normalize_integer_slice(selector)
            continue
        integer_array = _as_integer_array(selector, shape[axis])
        if integer_array is not None:
            selectors[axis] = integer_array
        elif _is_scalar_integer(selector):
            integer = int(selector)
            if not -shape[axis] <= integer < shape[axis]:
                raise IndexError("Integer index exceeds dimension size")
            selectors[axis] = integer
        else:
            raise TypeError(
                "NetCDF selectors must be integer scalars, integer/boolean "
                "vectors, or slices"
            )
    return _read_netcdf_var_sliced_trusted(var, tuple(selectors))


def _read_netcdf_var_sliced_trusted(
    var: Any,
    selectors: tuple[Any, ...],
) -> np.ndarray:
    """Read with an already normalized and bounded orthogonal selector."""

    return _read_netcdf_var_sliced_recursive(var, list(selectors))


def _normalize_netcdf_index(index: Any, ndim: int) -> Tuple[Any, ...]:
    """Expand an index into one selector per dimension, resolving Ellipsis."""
    if ndim == 0:
        if index is None or index is Ellipsis:
            return ()
        if isinstance(index, tuple) and len(index) == 0:
            return ()
        if isinstance(index, tuple) and len(index) == 1 and index[0] is Ellipsis:
            return ()

    if index is None:
        return tuple(slice(None) for _ in range(ndim))
    if index is Ellipsis:
        return tuple(slice(None) for _ in range(ndim))
    if not isinstance(index, tuple):
        index = (index,)

    ellipsis_count = sum(1 for item in index if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("At most one ellipsis is allowed in a NetCDF index")
    if ellipsis_count == 1:
        fill_count = ndim - (len(index) - 1)
        if fill_count < 0:
            raise IndexError("NetCDF index has too many dimensions")
        expanded = []
        for item in index:
            if item is Ellipsis:
                expanded.extend(slice(None) for _ in range(fill_count))
            else:
                expanded.append(item)
        index = tuple(expanded)
    elif len(index) < ndim:
        index = index + tuple(slice(None) for _ in range(ndim - len(index)))

    if len(index) > ndim:
        raise IndexError("NetCDF index has too many dimensions")
    return tuple(index)


def _is_scalar_integer(value: Any) -> bool:
    """Return True if the selector is a single integer (Python or numpy)."""
    if np.ma.isMaskedArray(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 0 and arr.dtype.kind in "iu"


def _normalize_integer_slice(value: slice) -> slice:
    """Return a slice whose explicit components are genuine integer values."""

    normalized: list[int | None] = []
    for component in (value.start, value.stop, value.step):
        if component is None:
            normalized.append(None)
            continue
        if isinstance(component, (bool, np.bool_)) or not isinstance(
            component, (int, np.integer),
        ):
            raise TypeError("NetCDF slice bounds must be integer values")
        normalized.append(int(component))
    if normalized[2] == 0:
        raise ValueError("NetCDF slice step cannot be zero")
    return slice(*normalized)


def _as_integer_array(selector: Any, axis_length: int) -> Optional[np.ndarray]:
    """Convert a sequence/boolean selector to a 1-D int64 index, else None."""
    if np.ma.isMaskedArray(selector):
        raise TypeError("NetCDF selectors must not be masked arrays")
    if isinstance(selector, slice) or _is_scalar_integer(selector):
        return None
    try:
        arr = np.asarray(selector)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim != 1:
        raise IndexError("NetCDF sequence indices must be one-dimensional")
    if arr.dtype.kind == "b":
        if arr.size != axis_length:
            raise IndexError("Boolean index length must match the indexed axis")
        arr = np.flatnonzero(arr)
    elif arr.size == 0:
        arr = np.empty(0, dtype=np.int64)
    elif arr.dtype.kind in "iu":
        if arr.dtype.kind == "u":
            if np.any(arr >= axis_length):
                raise IndexError("Integer index exceeds dimension size")
        elif np.any((arr < -axis_length) | (arr >= axis_length)):
            raise IndexError("Integer index exceeds dimension size")
        arr = arr.astype(np.int64, copy=False)
    else:
        return None

    if arr.size == 0:
        return arr.astype(np.int64, copy=False)
    arr = np.where(arr < 0, arr + axis_length, arr).astype(np.int64, copy=False)
    if np.any((arr < 0) | (arr >= axis_length)):
        raise IndexError("Integer index exceeds dimension size")
    return arr


def _read_netcdf_var_sliced_recursive(var: Any, selectors: List[Any]) -> np.ndarray:
    """Read the variable, expanding the first array selector via slices."""
    for axis, selector in enumerate(selectors):
        if isinstance(selector, np.ndarray):
            return _read_sequence_axis(var, selectors, axis, selector)
    if not selectors:
        return var[...]
    return var[tuple(selectors)]


def _read_sequence_axis(
    var: Any,
    selectors: List[Any],
    axis: int,
    index: np.ndarray,
) -> np.ndarray:
    """Read one array-indexed axis as contiguous slices, then reorder."""
    axis_out = _output_axis(selectors, axis)
    if index.size == 0:
        empty_selectors = selectors.copy()
        empty_selectors[axis] = slice(0, 0)
        return _read_netcdf_var_sliced_recursive(var, empty_selectors)

    unique_index, inverse = np.unique(index, return_inverse=True)
    chunking = var.chunking()
    chunk_size = (
        None
        if chunking == "contiguous" or not chunking
        else int(chunking[axis])
    )
    chunks = []
    selected_positions = []
    output_offset = 0
    for start, stop, run_index in _coalesced_runs(unique_index, chunk_size):
        slice_selectors = selectors.copy()
        slice_selectors[axis] = slice(start, stop)
        chunks.append(_read_netcdf_var_sliced_recursive(var, slice_selectors))
        selected_positions.extend(output_offset + run_index - start)
        output_offset += stop - start

    if len(chunks) == 1:
        data = chunks[0]
    elif any(np.ma.isMaskedArray(chunk) for chunk in chunks):
        data = np.ma.concatenate(chunks, axis=axis_out)
    else:
        data = np.concatenate(chunks, axis=axis_out)

    selected_positions = np.asarray(selected_positions, dtype=np.int64)
    if not np.array_equal(
        selected_positions,
        np.arange(selected_positions.size, dtype=np.int64),
    ):
        if np.ma.isMaskedArray(data):
            data = np.ma.take(data, selected_positions, axis=axis_out)
        else:
            data = np.take(data, selected_positions, axis=axis_out)
    if index.shape == unique_index.shape and np.array_equal(index, unique_index):
        return data
    if np.ma.isMaskedArray(data):
        return np.ma.take(data, inverse, axis=axis_out)
    return np.take(data, inverse, axis=axis_out)


def _output_axis(selectors: List[Any], axis: int) -> int:
    """Map an input axis to its output axis after scalar dimensions collapse."""
    return sum(0 if _is_scalar_integer(selector) else 1 for selector in selectors[:axis])


def _contiguous_runs(index: np.ndarray) -> Iterator[Tuple[int, int]]:
    """Yield (start, stop) half-open ranges for each run of consecutive ints."""
    run_start = 0
    split_points = np.flatnonzero(np.diff(index) != 1) + 1
    for run_stop in np.concatenate((split_points, np.array([index.size]))):
        start = int(index[run_start])
        stop = int(index[run_stop - 1]) + 1
        yield start, stop
        run_start = int(run_stop)


def _coalesced_runs(
    index: np.ndarray,
    chunk_size: int | None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Coalesce fragmented selectors that occupy the same physical chunk."""

    if chunk_size is None:
        for start, stop in _contiguous_runs(index):
            mask = (index >= start) & (index < stop)
            yield start, stop, index[mask]
        return

    run_start = 0
    chunk_ids = index // chunk_size
    split_points = np.flatnonzero(
        (np.diff(index) != 1) & (np.diff(chunk_ids) != 0),
    ) + 1
    for run_stop in np.concatenate((split_points, np.array([index.size]))):
        run_index = index[run_start:run_stop]
        yield int(run_index[0]), int(run_index[-1]) + 1, run_index
        run_start = int(run_stop)

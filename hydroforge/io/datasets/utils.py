# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Shared dataset helpers used by all dataset classes."""

from datetime import datetime
from typing import Any, Iterator, List, Optional, Tuple, Union

import cftime
import numpy as np


def single_file_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Constant key for single-file mode."""
    return ""


def daily_time_to_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Default time-to-file key: one file per day (YYYYMMDD)."""
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"


def yearly_time_to_key(dt: Union[datetime, cftime.datetime]) -> str:
    """Default time-to-file key: one file per year."""
    return f"{dt.year}"


def monthly_time_to_key(dt: datetime) -> str:
    """Default time-to-file key: one file per month (YYYY_MM)."""
    return dt.strftime("%Y_%m")


def read_netcdf_var_sliced(var: Any, index: Any = None) -> np.ndarray:
    """Read a NetCDF variable using only slices for sequence indices.

    Integer or boolean sequence selectors are read as one or more contiguous
    slices, then reordered in memory to match the requested index order.
    """
    selectors = list(_normalize_netcdf_index(index, var.ndim))
    shape = tuple(var.shape)
    for axis, selector in enumerate(selectors):
        integer_array = _as_integer_array(selector, shape[axis])
        if integer_array is not None:
            selectors[axis] = integer_array
        elif _is_scalar_integer(selector):
            selectors[axis] = int(selector)
    return _read_netcdf_var_sliced_recursive(var, selectors)


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
    if isinstance(value, (int, np.integer)):
        return True
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 0 and arr.dtype.kind in "iu"


def _as_integer_array(selector: Any, axis_length: int) -> Optional[np.ndarray]:
    """Convert a sequence/boolean selector to a 1-D int64 index, else None."""
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
    elif arr.dtype.kind in "iu":
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
    chunks = []
    for start, stop in _contiguous_runs(unique_index):
        slice_selectors = selectors.copy()
        slice_selectors[axis] = slice(start, stop)
        chunks.append(_read_netcdf_var_sliced_recursive(var, slice_selectors))

    if len(chunks) == 1:
        data = chunks[0]
    elif any(np.ma.isMaskedArray(chunk) for chunk in chunks):
        data = np.ma.concatenate(chunks, axis=axis_out)
    else:
        data = np.concatenate(chunks, axis=axis_out)

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

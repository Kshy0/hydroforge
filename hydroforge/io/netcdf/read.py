# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Orthogonal NetCDF reads, read-cache sizing, and read-chunk planning."""

import math
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from netCDF4 import Dataset
from pydantic import Field, validate_call

from hydroforge.core.validation import HydroForgeModel
from hydroforge.io.netcdf.encoding import (
    BOOL_LOGICAL_DTYPE,
    BOOL_NETCDF_READ_DTYPES,
    LOGICAL_DTYPE_ATTR,
    read_netcdf_values,
)
from hydroforge.io.netcdf.options import ensure_hdf5_plugins

_NETCDF_VARIABLE_CACHE_BYTES = 80 * 1024 * 1024
_NETCDF_LOGICAL_CHUNK_BYTES = 256 * 1024 * 1024
# A common two-worker/prefetch-two loader can retain four queued chunks plus
# the chunk being consumed.  Keep compact point payloads near one variable
# cache budget across those five in-flight positions.
_POINT_NETCDF_CHUNK_BYTES = _NETCDF_VARIABLE_CACHE_BYTES // 5


def decodes_strings(variable: Any) -> bool:
    """Whether netCDF4 collapses the last character axis while reading."""

    return (
        np.dtype(variable.dtype) == np.dtype("S1")
        and getattr(variable, "_Encoding", None) is not None
        and variable.chartostring
        and bool(variable.shape)
    )


def decoded_dtype(variable: Any) -> np.dtype:
    """Dtype of netCDF4's automatic unsigned and packing decode.

    Apply the same scalar operations to empty storage: no file payload is read,
    and NumPy's promotion follows netCDF4 even for identity packing attributes.
    """

    dtype = np.dtype(variable.dtype)
    if getattr(variable, LOGICAL_DTYPE_ATTR, None) == BOOL_LOGICAL_DTYPE:
        if dtype not in BOOL_NETCDF_READ_DTYPES:
            raise TypeError("boolean NetCDF variables must use i1/u1 storage")
        return np.dtype(np.bool_)
    encoding = getattr(variable, "_Encoding", None)
    if decodes_strings(variable):
        kind = "S" if encoding in {"none", "None", "bytes"} else "U"
        return np.dtype(f"{kind}{variable.shape[-1]}")
    if dtype.kind == "i" and getattr(variable, "_Unsigned", None) in {"true", "True"}:
        dtype = np.dtype(dtype.str.replace("i", "u"))
    if not getattr(variable, "_isprimitive", True) or dtype.kind not in {"i", "u", "f"}:
        return dtype
    values = np.empty(0, dtype=dtype)
    scale = getattr(variable, "scale_factor", None)
    offset = getattr(variable, "add_offset", None)
    if scale is not None and offset is not None:
        values = (
            values * scale + offset
            if scale != 1 or offset != 0
            else values.astype(np.asarray(scale).dtype)
        )
    elif scale is not None and scale != 1:
        values = values * scale
    elif offset is not None and offset != 0:
        values = values + offset
    return values.dtype


def decoded_element_bytes(variable: Any) -> int:
    """Return a conservative in-memory element width for one variable read.

    netCDF4 applies packing attributes while reading, so a packed integer
    source materializes with the promoted attribute dtype.  Unpacked integer
    sources are promoted to float64 at the source payload boundary.
    """

    read_dtype = decoded_dtype(variable)
    if not np.issubdtype(read_dtype, np.floating):
        return max(8, read_dtype.itemsize)
    return read_dtype.itemsize


_PositiveCount = Annotated[int, Field(gt=0)]


@validate_call(config=HydroForgeModel.model_config)
def plan_read_chunk_len(
    path: str | Path,
    var_name: str,
    *,
    profile: Literal["grid", "points"] = "grid",
    fallback: _PositiveCount = 24,
    max_steps: _PositiveCount = 256,
    max_bytes: _PositiveCount | None = None,
    physical_chunk_multiplier: _PositiveCount = 1,
    step_alignment: _PositiveCount = 1,
) -> int:
    """Choose a bounded logical read batch from one variable's physical layout.

    ``profile="points"`` keeps compact point payloads within one variable cache
    across a prefetched loader while accepting a whole physical time chunk.
    """

    if profile == "points":
        default_bytes = _POINT_NETCDF_CHUNK_BYTES
        physical_chunk_max_bytes = _NETCDF_LOGICAL_CHUNK_BYTES
    else:
        default_bytes = _NETCDF_LOGICAL_CHUNK_BYTES
        physical_chunk_max_bytes = None
    if max_bytes is None:
        max_bytes = default_bytes

    ensure_hdf5_plugins()
    with Dataset(Path(path), "r") as dataset:
        variable = dataset.variables[var_name]
        chunking = variable.chunking()
        time_axes = [
            index
            for index, name in enumerate(variable.dimensions)
            if name in {"time", "valid_time"}
        ]
        if len(time_axes) != 1:
            return fallback
        time_axis = time_axes[0]
        element_bytes = decoded_element_bytes(variable)
        bytes_per_step = element_bytes * math.prod(
            size for index, size in enumerate(variable.shape) if index != time_axis
        )
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
            aligned_capacity = (capacity_steps // step_alignment) * step_alignment
            if aligned_capacity >= step_alignment:
                capacity_steps = aligned_capacity
        return max(1, min(target_steps, capacity_steps))


def configure_variable_cache(
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
    for axis, (selector, chunk_size, axis_size) in enumerate(
        zip(
            selectors,
            chunk_shape,
            variable.shape,
            strict=True,
        )
    ):
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
    chunk_bytes = math.prod(chunk_shape) * np.dtype(variable.dtype).itemsize
    desired_bytes = min(max_bytes, max(chunk_bytes, touched_chunks * chunk_bytes))
    current_bytes, current_elements, preemption = variable.get_var_chunk_cache()
    if desired_bytes <= current_bytes:
        return
    variable.set_var_chunk_cache(
        size=desired_bytes,
        nelems=max(current_elements, touched_chunks * 2 + 1),
        preemption=preemption,
    )


def normalize_selection(index: Any, shape: tuple[int, ...]) -> tuple[Any, ...]:
    """Bound one orthogonal index to integer, int64-vector, or slice selectors.

    Unsupported selectors raise ``TypeError`` and out-of-range positions raise
    ``IndexError``; each public entry point chooses its own error type.
    """

    selectors = list(_normalize_netcdf_index(index, len(shape)))
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
    return tuple(selectors)


def read_variable(variable: Any, selectors: tuple[Any, ...]) -> np.ndarray:
    """Read with :func:`normalize_selection` selectors using only slices.

    Each integer vector is read as one or more contiguous slices, then
    reordered in memory to match the requested index order.
    """

    return _read_sliced(variable, list(selectors))


def _normalize_netcdf_index(index: Any, ndim: int) -> tuple[Any, ...]:
    """Expand an index into one selector per dimension, resolving Ellipsis."""
    if ndim == 0:
        if index is None or index is Ellipsis:
            return ()
        if isinstance(index, tuple) and len(index) == 0:
            return ()
        if isinstance(index, tuple) and len(index) == 1 and index[0] is Ellipsis:
            return ()

    if index is None or index is Ellipsis:
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
    return index


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
            component,
            (int, np.integer),
        ):
            raise TypeError("NetCDF slice bounds must be integer values")
        normalized.append(int(component))
    if normalized[2] == 0:
        raise ValueError("NetCDF slice step cannot be zero")
    return slice(*normalized)


def _as_integer_array(selector: Any, axis_length: int) -> np.ndarray | None:
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

    return np.where(arr < 0, arr + axis_length, arr).astype(np.int64, copy=False)


def _read_sliced(var: Any, selectors: list[Any]) -> np.ndarray:
    """Read the variable, expanding the first array selector via slices."""
    for axis, selector in enumerate(selectors):
        if isinstance(selector, np.ndarray):
            return _read_sequence_axis(var, selectors, axis, selector)
    if not selectors:
        return read_netcdf_values(var)
    return read_netcdf_values(var, tuple(selectors))


def _compile_sequence_read_plan(index: np.ndarray, chunk_size: int | None):
    unique_index, inverse = np.unique(index, return_inverse=True)
    runs = []
    positions = []
    output_offset = 0
    for start, stop, run_index in _coalesced_runs(unique_index, chunk_size):
        runs.append((start, stop))
        positions.append(output_offset + run_index - start)
        output_offset += stop - start
    selected = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
    if np.array_equal(selected, np.arange(selected.size, dtype=np.int64)):
        selected = None
    else:
        selected.setflags(write=False)
    if np.array_equal(index, unique_index):
        inverse = None
    else:
        inverse.setflags(write=False)
    return tuple(runs), selected, inverse


@lru_cache(maxsize=16)
def _cached_sequence_read_plan(payload: bytes, chunk_size: int | None):
    return _compile_sequence_read_plan(
        np.frombuffer(payload, dtype=np.int64), chunk_size
    )


def prefer_sparse_axis(var: Any, axis: int, index: np.ndarray) -> bool:
    """Avoid whole rows only when selection and physical coverage are sparse."""

    extent = var.shape[axis]
    if index.size * 4 >= extent:
        return False
    chunking = var.chunking()
    if chunking == "contiguous" or not chunking:
        return index.size <= 128
    chunk_size = int(chunking[axis])
    touched = np.unique(index // chunk_size).size
    return touched <= 128 and touched * chunk_size * 2 < extent


def _read_sequence_axis(
    var: Any,
    selectors: list[Any],
    axis: int,
    index: np.ndarray,
) -> np.ndarray:
    """Read one array-indexed axis as contiguous slices, then reorder."""
    axis_out = output_axis(selectors, axis)
    if index.size == 0:
        empty_selectors = selectors.copy()
        empty_selectors[axis] = slice(0, 0)
        return _read_sliced(var, empty_selectors)

    chunking = var.chunking()
    chunk_size = (
        None if chunking == "contiguous" or not chunking else int(chunking[axis])
    )
    runs, selected_positions, inverse = (
        _cached_sequence_read_plan(
            index.astype(np.int64, copy=False).tobytes(), chunk_size
        )
        if index.nbytes <= 128 * 1024
        else _compile_sequence_read_plan(index, chunk_size)
    )
    chunks = []
    for start, stop in runs:
        slice_selectors = selectors.copy()
        slice_selectors[axis] = slice(start, stop)
        chunks.append(_read_sliced(var, slice_selectors))

    if len(chunks) == 1:
        data = chunks[0]
    elif any(np.ma.isMaskedArray(chunk) for chunk in chunks):
        data = np.ma.concatenate(chunks, axis=axis_out)
    else:
        data = np.concatenate(chunks, axis=axis_out)

    for positions in (selected_positions, inverse):
        if positions is not None:
            data = data.take(positions, axis=axis_out)
    return data


def output_axis(selectors: list[Any], axis: int) -> int:
    """Map an input axis to its output axis after scalar dimensions collapse."""
    return sum(
        0 if _is_scalar_integer(selector) else 1 for selector in selectors[:axis]
    )


def _coalesced_runs(
    index: np.ndarray,
    chunk_size: int | None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Coalesce fragmented selectors that occupy the same physical chunk."""

    if index.size == 0:
        return
    run_start = 0
    breaks = np.diff(index) != 1
    if chunk_size is not None:
        breaks &= np.diff(index // chunk_size) != 0
    split_points = np.flatnonzero(breaks) + 1
    for run_stop in np.concatenate((split_points, np.array([index.size]))):
        run_index = index[run_start:run_stop]
        yield int(run_index[0]), int(run_index[-1]) + 1, run_index
        run_start = int(run_stop)

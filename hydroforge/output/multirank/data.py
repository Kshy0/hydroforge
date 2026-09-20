"""Vector, grid, and time-series reads over a rank-output catalog."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from pydantic import validate_call

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.netcdf import _prefer_sparse_axis, _read_netcdf_var_sliced_trusted
from hydroforge.serialization.netcdf import (
    BOOL_LOGICAL_DTYPE,
    decode_netcdf_logical_array,
)

logger = logging.getLogger(__name__)

_XY_POINT_DTYPE = np.dtype([("x", np.int64), ("y", np.int64)])


@dataclass(frozen=True, slots=True)
class _OutputTimeRequest:
    """One normalized, reader-view-relative output time selection."""

    start: int
    stop: int
    member: int = 0
    level: int | None = None

    @property
    def time_slice(self) -> slice:
        return slice(self.start, self.stop, 1)

    @property
    def length(self) -> int:
        return self.stop - self.start

    @property
    def row_index(self) -> int:
        return self.start


def _validated_grid_fill_value(value: Any, dtype: np.dtype) -> Any:
    """Return one scalar fill value without bool/integer reinterpretation."""

    if dtype.kind == "b":
        if type(value) is bool or isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(
            "boolean reader fill_value must be an exact bool; pass False "
            "explicitly for unrepresented grid cells"
        )
    if dtype.kind == "f":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("real reader fill_value must be a real scalar")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            converted = dtype.type(value)
        if isinstance(value, Integral):
            if not np.isfinite(converted):
                raise OverflowError(
                    f"fill_value {value!r} is outside dtype {dtype} range"
                )
            if int(converted) != int(value):
                raise ValueError(
                    f"fill_value {value!r} is not exactly representable as {dtype}"
                )
            return converted
        try:
            source = value if isinstance(value, np.floating) else float(value)
        except (OverflowError, ValueError) as error:
            raise OverflowError(
                f"fill_value {value!r} is outside dtype {dtype} range"
            ) from error
        if np.isfinite(source) and not np.isfinite(converted):
            raise OverflowError(f"fill_value {value!r} is outside dtype {dtype} range")
        if source != 0.0 and converted == 0.0:
            raise OverflowError(f"fill_value {value!r} underflows dtype {dtype}")
        return converted
    if dtype.kind not in "iu":
        raise TypeError(f"unsupported reader fill dtype {dtype}")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("integer reader fill_value must not be boolean")
    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Real):
        if not np.isfinite(value) or float(value) != np.trunc(value):
            raise ValueError(f"fill_value {value!r} is not an exact finite integer")
        integer = int(value)
    else:
        raise TypeError("integer reader fill_value must be a real scalar")
    limits = np.iinfo(dtype)
    if integer < limits.min or integer > limits.max:
        raise OverflowError(
            f"fill_value {integer} is outside dtype {dtype} range "
            f"[{limits.min}, {limits.max}]"
        )
    return dtype.type(integer)


class MultiRankDataAccess:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self._series_indexes: dict[
            tuple[int, bool], tuple[np.ndarray, np.ndarray] | None
        ] = {}

    def _result_dtype(self, dtype: np.dtype | None) -> np.dtype:
        if dtype is not None:
            result = np.dtype(dtype)
            if result.kind not in "biuf":
                raise TypeError("reader dtype must be numeric or boolean")
            return result
        info = self.owner._rank_files[0]
        if info["logical_dtype"] == BOOL_LOGICAL_DTYPE:
            return np.dtype(np.bool_)
        return np.dtype(info["dtype"])

    @staticmethod
    def _array(value: Any, *, source: str) -> np.ndarray:
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"statistics data from {source} contains missing values")
        return np.asarray(value)

    @staticmethod
    def _cast_result(
        value: np.ndarray,
        dtype: np.dtype,
        *,
        label: str,
    ) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype == dtype:
            return array
        if dtype.kind == "b":
            if array.dtype.kind != "b":
                raise TypeError(f"{label} cannot be reinterpreted as boolean output")
            return array.astype(dtype, copy=False)
        if dtype.kind in "iu":
            if array.dtype.kind not in "iuf":
                raise TypeError(f"{label} cannot be converted to integer output")
            if array.dtype.kind == "f":
                if not np.isfinite(array).all() or np.any(array != np.trunc(array)):
                    raise ValueError(
                        f"{label} contains non-integral or non-finite values"
                    )
                # Compare against an exact power-of-two half-open interval.
                # Converting int64.max to float64 rounds it up to 2**63, so a
                # conventional ``value > limits.max`` check accepts 2**63 and
                # the subsequent cast wraps to int64.min.  uint64 has the same
                # alias at 2**64.
                bits = dtype.itemsize * 8
                signed = dtype.kind == "i"
                upper = math.ldexp(1.0, bits - int(signed))
                lower = -upper if signed else 0.0
                outside = (array < lower) | (array >= upper)
            else:
                limits = np.iinfo(dtype)
                outside = (array < limits.min) | (array > limits.max)
            if array.size and np.any(outside):
                raise OverflowError(f"{label} contains values outside {dtype} range")
            return array.astype(dtype, copy=False)
        if dtype.kind == "f":
            if array.dtype.kind not in "iuf":
                raise TypeError(f"{label} cannot be converted to real output")
            if dtype.itemsize < 8 and array.size:
                finite = array[np.isfinite(array)]
                if finite.size and np.any(np.abs(finite) > np.finfo(dtype).max):
                    raise OverflowError(
                        f"{label} contains values outside {dtype} range"
                    )
            converted = array.astype(dtype, copy=False)
            if np.any(np.isfinite(array) & ~np.isfinite(converted)):
                raise OverflowError(f"{label} contains values outside {dtype} range")
            if np.any(np.isfinite(array) & (array != 0) & (converted == 0)):
                raise OverflowError(
                    f"{label} contains nonzero values that underflow in {dtype}"
                )
            if array.dtype.kind in "iu" and not np.array_equal(
                array.astype(object),
                converted.astype(object),
            ):
                raise ValueError(
                    f"{label} contains integers that are not exactly "
                    f"representable as {dtype}"
                )
            return converted
        raise TypeError(f"unsupported reader result dtype {dtype}")

    def _validate_axes(
        self,
        info: dict,
        *,
        level: int | None,
        member: int,
    ) -> None:
        if info["has_ensemble"]:
            if not 0 <= member < info["member_count"]:
                raise IndexError(f"member out of range [0, {info['member_count'] - 1}]")
        elif member != 0:
            raise ValueError("member must be 0 for an output without a member axis")
        if info["has_levels"]:
            if level is None:
                raise TypeError(
                    "level must be an exact int for output dimension "
                    f"{info['level_dimension']!r}"
                )
            if not 0 <= level < info["n_levels"]:
                raise IndexError(
                    f"level for dimension {info['level_dimension']!r} is out "
                    f"of range [0, {info['n_levels'] - 1}]"
                )
        elif level is not None:
            raise ValueError(
                "level must be None for an output without a trailing value dimension"
            )

    def _validate_axes_request(
        self,
        *,
        level: int | None,
        member: int,
    ) -> None:
        self._validate_axes(self.owner._rank_files[0], level=level, member=member)

    def _make_row_request(
        self,
        *,
        time_index: int,
        level: int | None,
        member: int,
    ) -> _OutputTimeRequest:
        if not 0 <= time_index < self.owner._time_len:
            raise IndexError(f"t_index out of range [0, {self.owner._time_len - 1}]")
        request = _OutputTimeRequest(
            start=time_index,
            stop=time_index + 1,
            member=member,
            level=level,
        )
        self._validate_axes_request(level=request.level, member=request.member)
        return request

    def _make_series_request(
        self,
        *,
        time_slice: slice | None,
        level: int | None,
        member: int,
    ) -> _OutputTimeRequest:
        if time_slice is None:
            start, stop = 0, self.owner._time_len
        else:
            if not isinstance(time_slice, slice):
                raise TypeError("time_slice must be a slice or None")
            for name, value in (
                ("start", time_slice.start),
                ("stop", time_slice.stop),
                ("step", time_slice.step),
            ):
                if value is not None and type(value) is not int:
                    raise TypeError(f"time_slice {name} must be an exact int or None")
            if time_slice.step not in (None, 1):
                raise ValueError("time_slice step must be 1 or None")
            start, stop, _step = time_slice.indices(self.owner._time_len)
            if stop < start:
                stop = start
        request = _OutputTimeRequest(
            start=start,
            stop=stop,
            member=member,
            level=level,
        )
        self._validate_axes_request(
            level=request.level,
            member=request.member,
        )
        return request

    @staticmethod
    def _selection(
        info: dict, request: _OutputTimeRequest, time: Any, points: Any = slice(None)
    ) -> tuple:
        indices = [time]
        if info["has_ensemble"]:
            indices.append(request.member)
        indices.append(points)
        if info["has_levels"]:
            indices.append(request.level)
        return tuple(indices)

    @staticmethod
    def _read_cache_row(
        cache_arr: np.ndarray,
        info: dict,
        request: _OutputTimeRequest,
    ) -> np.ndarray:
        return cache_arr[
            MultiRankDataAccess._selection(info, request, request.row_index)
        ]

    def _read_netcdf_row(
        self,
        info: dict,
        request: _OutputTimeRequest,
    ) -> np.ndarray:
        """Read one requested row from the NetCDF shard that contains it."""
        orig_time = int(self.owner._t_indices[request.row_index])
        file_index, (start, _end) = next(
            (index, bounds)
            for index, bounds in enumerate(info["file_time_offsets"])
            if bounds[0] <= orig_time < bounds[1]
        )
        local_time = orig_time - start
        fp = self.owner._checked_source_path(info["paths"][file_index])
        with self.owner._read_handles.acquire(fp) as ds:
            var = ds.variables[self.owner.var_name]
            indices = self._selection(info, request, local_time)
            result = decode_netcdf_logical_array(
                var,
                var[indices],
                name=self.owner.var_name,
            )
        self.owner._verify_source_path(fp)
        return result

    def _read_rank_row(
        self,
        info: dict,
        request: _OutputTimeRequest,
    ) -> np.ndarray:
        cache = self.owner._rank_cache_for(info["rank_id"])
        if cache is not None:
            return self._read_cache_row(cache, info, request)
        return self._read_netcdf_row(info, request)

    @validate_call(config=HydroForgeModel.model_config)
    def get_vector(
        self,
        t_index: int,
        level: int | None = None,
        member: int = 0,
        dtype: Any = None,
    ) -> np.ndarray:
        request = self._make_row_request(
            time_index=t_index,
            level=level,
            member=member,
        )

        target_dtype = self._result_dtype(dtype)
        self.owner._ensure_cache_materialized()
        parts: list[np.ndarray] = []
        for info in self.owner._rank_files:
            if info["saved_points"] == 0:
                parts.append(np.empty((0,), dtype=target_dtype))
                continue

            data = self._read_rank_row(info, request)

            arr = self._array(data, source=info["paths"][0].name)
            arr = self._cast_result(
                arr,
                target_dtype,
                label="statistics vector",
            )
            parts.append(arr)
        return np.concatenate(parts, axis=0) if parts else np.array([])

    @validate_call(config=HydroForgeModel.model_config)
    def get_grid(
        self,
        t_index: int,
        level: int | None = None,
        member: int = 0,
        fill_value: Any = np.nan,
        dtype: Any = None,
    ) -> np.ndarray:
        request = self._make_row_request(
            time_index=t_index,
            level=level,
            member=member,
        )
        if self.owner.map_shape is None:
            raise RuntimeError("map_shape is not set; cannot project to grid.")

        nx_, ny_ = self.owner.map_shape
        target_dtype = self._result_dtype(dtype)
        validated_fill = _validated_grid_fill_value(
            fill_value,
            target_dtype,
        )
        self.owner._ensure_cache_materialized()
        grid = np.full(
            (nx_, ny_),
            validated_fill,
            dtype=target_dtype,
        )

        for info in self.owner._rank_files:
            if info["saved_points"] == 0:
                continue
            x = info.get("x")
            y = info.get("y")
            if x is None or y is None:
                raise RuntimeError(
                    f"rank {info['rank_id']} missing (x,y); set map_shape or "
                    "coord converter"
                )

            vals = self._read_rank_row(info, request)

            values = self._array(vals, source=info["paths"][0].name)
            grid[x, y] = self._cast_result(
                values,
                target_dtype,
                label="statistics grid",
            )
        return grid

    @staticmethod
    def _sorted_series_indices(
        pairs: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        out_cols = np.array([col for col, _ in pairs], dtype=np.int64)
        local_idx = np.array([li for _, li in pairs], dtype=np.int64)
        order = np.argsort(local_idx, kind="stable")
        return out_cols[order], local_idx[order]

    def _copy_series_from_cache(
        self,
        out: np.ndarray,
        cache: np.ndarray,
        info: dict,
        pairs: list[tuple[int, int]],
        request: _OutputTimeRequest,
        target_dtype: np.dtype,
    ) -> None:
        out_cols, local_idx = self._sorted_series_indices(pairs)
        indices = self._selection(info, request, request.time_slice, local_idx)
        chunk = self._array(
            cache[indices],
            source=info["paths"][0].name,
        )
        out[:, out_cols] = self._cast_result(
            chunk,
            target_dtype,
            label="statistics series",
        )

    def _copy_series_from_netcdf(
        self,
        out: np.ndarray,
        info: dict,
        pairs: list[tuple[int, int]],
        request: _OutputTimeRequest,
        target_dtype: np.dtype,
    ) -> None:
        out_cols, local_idx = self._sorted_series_indices(pairs)
        if request.length == 0:
            return

        global_start = self.owner._slice_start + request.start
        global_stop = self.owner._slice_start + request.stop
        for fp, (file_start, file_stop) in zip(
            info["paths"],
            info["file_time_offsets"],
            strict=True,
        ):
            requested_start = max(global_start, file_start)
            requested_stop = min(global_stop, file_stop)
            if requested_start >= requested_stop:
                continue

            local_start = requested_start - file_start
            local_stop = requested_stop - file_start
            output_start = requested_start - global_start
            checked_path = self.owner._checked_source_path(fp)
            with self.owner._read_handles.acquire(checked_path) as ds:
                var = ds.variables[self.owner.var_name]
                point_axis = 2 if info["has_ensemble"] else 1
                sparse = _prefer_sparse_axis(var, point_axis, local_idx)
                if self.owner.row_chunk_size is None:
                    # Bound the unfiltered NetCDF read even when the caller
                    # asks for one gauge column. The full saved-point row
                    # controls peak memory before selection.
                    bytes_per_row = max(
                        1,
                        math.prod(var.shape[1:]) * np.dtype(var.dtype).itemsize,
                    )
                    step = max(1, (256 * 1024 * 1024) // bytes_per_row)
                else:
                    step = self.owner.row_chunk_size
                for t0 in range(local_start, local_stop, step):
                    t1 = min(t0 + step, local_stop)
                    slices = self._selection(
                        info,
                        request,
                        slice(t0, t1),
                        local_idx if sparse else slice(None),
                    )
                    block = decode_netcdf_logical_array(
                        var,
                        _read_netcdf_var_sliced_trusted(var, slices)
                        if sparse
                        else var[slices],
                        name=self.owner.var_name,
                    )
                    block = self._array(block, source=fp.name)
                    selected = block if sparse else block[:, local_idx]
                    o0 = output_start + (t0 - local_start)
                    o1 = o0 + (t1 - t0)
                    out[o0:o1, out_cols] = self._cast_result(
                        selected,
                        target_dtype,
                        label="statistics series",
                    )
            self.owner._verify_source_path(checked_path)

    def _series_point_index(
        self,
        rank_index: int,
        *,
        use_xy: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Lazily index the reader's immutable coordinates with compact arrays."""

        key = (rank_index, use_xy)
        if key not in self._series_indexes:
            info = self.owner._rank_files[rank_index]
            if info["saved_points"] == 0:
                self._series_indexes[key] = None
            elif use_xy:
                if info.get("x") is None or info.get("y") is None:
                    self._series_indexes[key] = None
                else:
                    values = np.empty(info["saved_points"], dtype=_XY_POINT_DTYPE)
                    values["x"] = info["x"]
                    values["y"] = info["y"]
                    order = np.lexsort((values["y"], values["x"]))
                    self._series_indexes[key] = (values[order], order)
            elif info["coord_raw"] is None:
                self._series_indexes[key] = None
            else:
                values = np.asarray(info["coord_raw"]).ravel()
                order = np.argsort(values, kind="stable")
                self._series_indexes[key] = (values[order], order)
        return self._series_indexes[key]

    def resolve_series_points(
        self,
        queries: Sequence[int | tuple[int, int]],
        *,
        use_xy: bool,
    ) -> dict[int, list[tuple[int, int]]]:
        """Resolve a validated query to immutable rank-local column pairs."""

        query_values = (
            np.fromiter(queries, dtype=_XY_POINT_DTYPE, count=len(queries))
            if use_xy
            else np.asarray(queries, dtype=np.int64)
        )
        col_to_hits: list[tuple[int, int] | None] = [None] * len(queries)
        remaining = np.arange(len(queries))
        for rank_index in range(len(self.owner._rank_files)):
            if remaining.size == 0:
                break
            index = self._series_point_index(rank_index, use_xy=use_xy)
            if index is None:
                continue
            values, order = index
            wanted = query_values[remaining]
            positions = np.searchsorted(values, wanted)
            matched = positions < values.size
            matched[matched] = values[positions[matched]] == wanted[matched]
            for column, local_index in zip(
                remaining[matched],
                order[positions[matched]],
                strict=True,
            ):
                col_to_hits[column] = (rank_index, int(local_index))
            remaining = remaining[~matched]

        if remaining.size:
            raise ValueError("some points were not found in any rank")

        logger.debug("Resolved %d statistics points across ranks", len(queries))

        rank_to_cols: dict[int, list[tuple[int, int]]] = {}
        for column, (rank_index, local_index) in enumerate(col_to_hits):
            rank_to_cols.setdefault(rank_index, []).append((column, local_index))
        return rank_to_cols

    def get_series(self, query: Any) -> np.ndarray:
        """Execute one already validated and rank-resolved series query."""

        request = query.time_request
        target_dtype = query.target_dtype
        rank_to_cols = query.rank_to_columns
        column_count = sum(len(pairs) for pairs in rank_to_cols.values())
        out = np.empty(
            (request.length, column_count),
            dtype=target_dtype,
        )
        if request.length == 0 or column_count == 0:
            return out

        # Fast path for an already materialized in-memory cache.
        for r_idx, pairs in rank_to_cols.items():
            info = self.owner._rank_files[r_idx]
            cache_arr = self.owner._rank_cache_for(info["rank_id"])
            if cache_arr is not None:
                self._copy_series_from_cache(
                    out,
                    cache_arr,
                    info,
                    pairs,
                    request,
                    target_dtype,
                )
                continue

            # Choose sparse NetCDF indexing or bounded full-row reads according
            # to the physical chunk layout.
            self._copy_series_from_netcdf(
                out,
                info,
                pairs,
                request,
                target_dtype,
            )

        return out

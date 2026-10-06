# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Point time series read from the rank files of one statistics output."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any

import netCDF4 as nc
import numpy as np
from pydantic import (
    BeforeValidator,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
    validate_call,
)

from hydroforge.core.time import DateLike, normalize_calendar_dates
from hydroforge.core.validation import HydroForgeModel
from hydroforge.io.files import SourceFiles
from hydroforge.io.netcdf.encoding import (
    decode_netcdf_logical_array,
    read_netcdf_values,
)
from hydroforge.io.netcdf.read import prefer_sparse_axis, read_variable
from hydroforge.io.rank_output.catalog import RankCatalog, RankFiles, scan_rank_files

# Upper bound on persistent read handles, well below common descriptor limits.
_MAX_READ_HANDLES = 256
# Bound on one unfiltered NetCDF row block read before point selection.
_MAX_BLOCK_BYTES = 256 * 1024 * 1024


def _point_ids(value: Any) -> np.ndarray:
    array = np.ma.asarray(value)
    if np.ma.is_masked(array):
        raise ValueError("point IDs must not contain missing values")
    ids = np.asarray(array)
    if ids.ndim > 1:
        raise ValueError(f"point IDs must be one-dimensional; got shape {ids.shape}")
    if ids.size and ids.dtype.kind not in "iu":
        raise ValueError("point IDs must be integers")
    if ids.dtype.kind == "u" and ids.size and int(ids.max()) > np.iinfo(np.int64).max:
        raise ValueError("point IDs exceed int64 range")
    ids = ids.astype(np.int64).ravel()
    if np.unique(ids).size != ids.size:
        raise ValueError("duplicate points are not allowed")
    return ids


def _time_slice(value: Any) -> slice | None:
    if value is None:
        return None
    if not isinstance(value, slice):
        raise ValueError("time_slice must be a slice or None")
    for name in ("start", "stop", "step"):
        component = getattr(value, name)
        if component is not None and type(component) is not int:
            raise ValueError(f"time_slice {name} must be an exact int or None")
    if value.step not in (None, 1):
        raise ValueError("time_slice step must be 1 or None")
    return value


def _result_dtype(value: Any) -> np.dtype | None:
    if value is None:
        return None
    dtype = np.dtype(value)
    if dtype.kind not in "biuf":
        raise ValueError("reader dtype must be numeric or boolean")
    return dtype


_PointIds = Annotated[np.ndarray, BeforeValidator(_point_ids)]
_TimeSlice = Annotated[slice | None, BeforeValidator(_time_slice)]
_ResultDType = Annotated[Any, BeforeValidator(_result_dtype)]
_Index = Annotated[int, Field(ge=0)]


def _array(value: Any, *, source: str) -> np.ndarray:
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"statistics data from {source} contains missing values")
    return np.asarray(value)


def _cast_result(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert values to ``dtype`` without losing or reinterpreting any value."""

    label = "statistics series"
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
                raise ValueError(f"{label} contains non-integral or non-finite values")
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
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{label} cannot be converted to real output")
    if dtype.itemsize < 8 and array.size:
        finite = array[np.isfinite(array)]
        if finite.size and np.any(np.abs(finite) > np.finfo(dtype).max):
            raise OverflowError(f"{label} contains values outside {dtype} range")
    converted = array.astype(dtype, copy=False)
    # Tiny values may round to subnormals or zero; only overflow fails.
    if np.any(np.isfinite(array) & ~np.isfinite(converted)):
        raise OverflowError(f"{label} contains values outside {dtype} range")
    if array.dtype.kind in "iu" and array.size:
        # Round-trip in place of a per-element Python comparison.  A value
        # rounded up to 2**bits (or below the signed minimum) is out of the
        # source range, whose cast back would wrap or saturate.
        bits = array.dtype.itemsize * 8
        signed = array.dtype.kind == "i"
        upper = math.ldexp(1.0, bits - int(signed))
        lower = -upper if signed else 0.0
        in_range = (converted < upper) & (converted >= lower)
        back = np.where(in_range, converted, 0).astype(array.dtype)
        if not in_range.all() or np.any(back != array):
            raise ValueError(
                f"{label} contains integers that are not exactly representable "
                f"as {dtype}"
            )
    return converted


class MultiRankStatsReader(HydroForgeModel):
    """Read point series by ID from ``{var_name}_rank{rank}[_{year}].nc`` files.

    Variables must use the ``('time', ['ensemble'], 'saved_points',
    [value_axis])`` layout of the rank-output schema.  The saved-point
    coordinate is the file's ``hydroforge_coordinate``; files written without
    that attribute need an explicit ``coord_name``.  ``time_range`` is a
    closed interval selecting the reader's view of the common committed
    timeline.  The candidate file set and each file's identity are captured
    at construction; later reads reject files changed since.
    """

    base_dir: Annotated[Path, Field(strict=False)]
    var_name: Annotated[str, Field(min_length=1)]
    coord_name: Annotated[str, Field(min_length=1)] | None = None
    time_range: tuple[DateLike, DateLike] | None = None
    cache_enabled: bool = False
    split_by_year: bool = False
    row_chunk_size: Annotated[int, Field(ge=1)] | None = None

    _files: SourceFiles = PrivateAttr()
    _catalog: RankCatalog = PrivateAttr()
    _view: slice = PrivateAttr()
    _cache: dict[int, np.ndarray | None] | None = PrivateAttr(default=None)
    _indexes: dict[int, tuple[np.ndarray, np.ndarray] | None] = PrivateAttr(
        default_factory=dict
    )

    @field_validator("var_name")
    @classmethod
    def _validate_var_name(cls, value: str) -> str:
        if Path(value).name != value or "/" in value or "\\" in value:
            raise ValueError("var_name must not contain path separators")
        return value

    @field_validator("time_range")
    @classmethod
    def _validate_time_range(
        cls, value: tuple[DateLike, DateLike] | None
    ) -> tuple[DateLike, DateLike] | None:
        if value is None:
            return None
        _calendar, normalized, _defaulted = normalize_calendar_dates(
            {"time_range start": value[0], "time_range end": value[1]},
            calendar=None,
            preserve_cftime_declaration=True,
        )
        start, end = normalized["time_range start"], normalized["time_range end"]
        if start > end:
            raise ValueError("time_range start must be <= end (closed interval)")
        return start, end

    @model_validator(mode="after")
    def _scan(self):
        """Capture and validate the rank files once every field is valid."""

        try:
            paths = tuple(
                path.absolute()
                for path in sorted(self.base_dir.glob(f"{self.var_name}_rank*.nc"))
            )
            if not paths:
                raise FileNotFoundError(
                    f"No files found in {self.base_dir} matching: "
                    f"{self.var_name}_rank*.nc"
                )
            self._files = SourceFiles.capture(
                paths,
                label="Reader source file",
                max_open=min(_MAX_READ_HANDLES, max(8, len(paths))),
            )
            self._catalog = scan_rank_files(
                self._files,
                self.var_name,
                coord_name=self.coord_name,
                split_by_year=self.split_by_year,
            )
            self._view = self._time_view()
        except (
            KeyError,
            IndexError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise ValueError(
                str(error)
                if isinstance(error, ValueError)
                else f"{type(error).__name__}: {error}"
            ) from error
        return self

    def _time_view(self) -> slice:
        """Resolve ``time_range`` to rows of the catalog timeline."""

        catalog = self._catalog
        if self.time_range is None:
            return slice(0, len(catalog.times))
        _calendar, normalized, _defaulted = normalize_calendar_dates(
            {
                "time_range start": self.time_range[0],
                "time_range end": self.time_range[1],
            },
            calendar=catalog.calendar,
        )
        start, end = normalized["time_range start"], normalized["time_range end"]
        first, last = (
            nc.date2num(value, catalog.time_units, catalog.calendar)
            for value in (start, end)
        )
        values = catalog.time_values
        if first < values[0] or last > values[-1]:
            raise ValueError(
                "time_range outside available coverage. "
                f"Requested [{start} .. {end}] but coverage is "
                f"[{catalog.times[0]} .. {catalog.times[-1]}]."
            )
        begin = int(np.searchsorted(values, first, side="left"))
        stop = int(np.searchsorted(values, last, side="right"))
        if begin == stop:
            raise ValueError("No time steps found in the request range.")
        return slice(begin, stop)

    @property
    def time_len(self) -> int:
        """Return the number of time rows in this reader's view."""

        return self._view.stop - self._view.start

    @property
    def times(self) -> tuple[DateLike, ...]:
        """Return the timestamps of this reader's view."""

        return self._catalog.times[self._view]

    def get_all_cids(self) -> np.ndarray | None:
        """Return every rank's saved-point IDs in rank order, or ``None``."""

        cids = [
            rank.coordinate
            for rank in self._catalog.ranks
            if rank.saved_points and rank.coordinate is not None
        ]
        return np.concatenate(cids) if cids else None

    @validate_call(config=HydroForgeModel.model_config)
    def get_series(
        self,
        ids: _PointIds,
        *,
        level: _Index | None = None,
        member: _Index = 0,
        dtype: _ResultDType = None,
        time_slice: _TimeSlice = None,
    ) -> np.ndarray:
        """Return ``(time, len(ids))`` values over a half-open view slice."""

        layout = self._catalog.layout
        if layout.member_count is not None:
            if member >= layout.member_count:
                raise IndexError(f"member out of range [0, {layout.member_count - 1}]")
        elif member != 0:
            raise ValueError("member must be 0 for an output without a member axis")
        if layout.level_dimension is not None:
            if level is None:
                raise TypeError(
                    "level must be an exact int for output dimension "
                    f"{layout.level_dimension!r}"
                )
            if level >= layout.level_count:
                raise IndexError(
                    f"level for dimension {layout.level_dimension!r} is out "
                    f"of range [0, {layout.level_count - 1}]"
                )
        elif level is not None:
            raise ValueError(
                "level must be None for an output without a trailing value dimension"
            )
        source_dtype = layout.decoded_dtype
        if dtype is None:
            dtype = source_dtype
        else:
            _cast_result(np.empty(0, dtype=source_dtype), dtype)
        start, stop, _step = (time_slice or slice(None)).indices(self.time_len)
        stop = max(start, stop)
        columns = self._columns(ids)

        out = np.empty((stop - start, ids.size), dtype=dtype)
        if stop == start or ids.size == 0:
            return out
        if self.cache_enabled and self._cache is None:
            self._cache = self._load_cache()
        for rank_index, (out_columns, points) in columns.items():
            rank = self._catalog.ranks[rank_index]
            cache = None if self._cache is None else self._cache[rank.rank]
            if cache is not None:
                paths = tuple(
                    path
                    for path, (first, last) in zip(
                        rank.paths, rank.time_offsets, strict=True
                    )
                    if first < self._view.start + stop
                    and last > self._view.start + start
                )
                for path in paths:
                    self._files.verify(path)
                values = cache[
                    self._selection(slice(start, stop), member, points, level)
                ]
                out[:, out_columns] = _cast_result(
                    _array(values, source=rank.paths[0].name), dtype
                )
                for path in paths:
                    self._files.verify(path)
            else:
                self._read_series(
                    out,
                    rank,
                    out_columns,
                    points,
                    rows=(self._view.start + start, self._view.start + stop),
                    member=member,
                    level=level,
                    dtype=dtype,
                )
        return out

    def close(self) -> None:
        """Close this process's NetCDF handles; later reads reopen them."""

        self._files.close()

    def _selection(
        self, time: Any, member: int, points: Any, level: int | None
    ) -> tuple:
        layout = self._catalog.layout
        indices = [time]
        if layout.member_count is not None:
            indices.append(member)
        indices.append(points)
        if layout.level_dimension is not None:
            indices.append(level)
        return tuple(indices)

    def _columns(self, ids: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Map each ID to its rank and rank-local point, ordered by point."""

        hits: list[tuple[int, int] | None] = [None] * ids.size
        remaining = np.arange(ids.size)
        for rank_index in range(len(self._catalog.ranks)):
            if remaining.size == 0:
                break
            index = self._point_index(rank_index)
            if index is None:
                continue
            values, order = index
            wanted = ids[remaining]
            positions = np.searchsorted(values, wanted)
            matched = positions < values.size
            matched[matched] = values[positions[matched]] == wanted[matched]
            for column, point in zip(
                remaining[matched], order[positions[matched]], strict=True
            ):
                hits[column] = (rank_index, int(point))
            remaining = remaining[~matched]
        if remaining.size:
            raise ValueError("some points were not found in any rank")

        pairs: dict[int, list[tuple[int, int]]] = {}
        for column, (rank_index, point) in enumerate(hits):
            pairs.setdefault(rank_index, []).append((column, point))
        columns = {}
        for rank_index, rank_pairs in pairs.items():
            out_columns = np.array([column for column, _ in rank_pairs], dtype=np.int64)
            points = np.array([point for _, point in rank_pairs], dtype=np.int64)
            order = np.argsort(points, kind="stable")
            columns[rank_index] = (out_columns[order], points[order])
        return columns

    def _point_index(self, rank_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Lazily sort one rank's IDs for lookup."""

        if rank_index not in self._indexes:
            rank = self._catalog.ranks[rank_index]
            if rank.saved_points == 0 or rank.coordinate is None:
                self._indexes[rank_index] = None
            else:
                order = np.argsort(rank.coordinate, kind="stable")
                self._indexes[rank_index] = (rank.coordinate[order], order)
        return self._indexes[rank_index]

    def _load_cache(self) -> dict[int, np.ndarray | None]:
        """Read every rank's view rows, all points, into memory."""

        cache: dict[int, np.ndarray | None] = {}
        begin, end = self._view.start, self._view.stop
        for rank in self._catalog.ranks:
            cache[rank.rank] = None
            if rank.saved_points == 0:
                continue
            for path, (file_start, file_stop) in zip(
                rank.paths, rank.time_offsets, strict=True
            ):
                first, last = max(begin, file_start), min(end, file_stop)
                if first >= last:
                    continue
                with self._files.open_netcdf(path) as dataset:
                    variable = dataset.variables[self.var_name]
                    step = self.row_chunk_size or (last - first)
                    for row in range(first, last, step):
                        block = _array(
                            decode_netcdf_logical_array(
                                variable,
                                read_netcdf_values(
                                    variable,
                                    slice(
                                        row - file_start,
                                        min(row + step, last) - file_start,
                                    ),
                                ),
                                name=self.var_name,
                            ),
                            source=path.name,
                        )
                        if cache[rank.rank] is None:
                            cache[rank.rank] = np.empty(
                                (end - begin, *block.shape[1:]), dtype=block.dtype
                            )
                        cache[rank.rank][row - begin : row - begin + len(block)] = block
        return cache

    def _read_series(
        self,
        out: np.ndarray,
        rank: RankFiles,
        out_columns: np.ndarray,
        points: np.ndarray,
        *,
        rows: tuple[int, int],
        member: int,
        level: int | None,
        dtype: np.dtype,
    ) -> None:
        """Copy catalog rows ``[rows)`` of ``points`` into ``out``."""

        begin, end = rows
        for path, (file_start, file_stop) in zip(
            rank.paths, rank.time_offsets, strict=True
        ):
            first, last = max(begin, file_start), min(end, file_stop)
            if first >= last:
                continue
            with self._files.open_netcdf(path) as dataset:
                variable = dataset.variables[self.var_name]
                point_axis = 1 if self._catalog.layout.member_count is None else 2
                # A single contiguous column costs one NetCDF read even when
                # decompression touches full-width chunks; keep the other
                # columns out of the materialized host array.
                sparse = points.size == 1 or prefer_sparse_axis(
                    variable, point_axis, points
                )
                if self.row_chunk_size is None:
                    # Bound the unfiltered NetCDF read even when the caller
                    # asks for one gauge column: the full saved-point row
                    # controls peak memory before selection.
                    row_bytes = max(
                        1,
                        math.prod(variable.shape[1:])
                        * np.dtype(variable.dtype).itemsize,
                    )
                    step = max(1, _MAX_BLOCK_BYTES // row_bytes)
                else:
                    step = self.row_chunk_size
                for row in range(first, last, step):
                    stop = min(row + step, last)
                    selection = self._selection(
                        slice(row - file_start, stop - file_start),
                        member,
                        points if sparse else slice(None),
                        level,
                    )
                    block = _array(
                        decode_netcdf_logical_array(
                            variable,
                            read_variable(variable, selection)
                            if sparse
                            else read_netcdf_values(variable, selection),
                            name=self.var_name,
                        ),
                        source=path.name,
                    )
                    out[row - begin : stop - begin, out_columns] = _cast_result(
                        block if sparse else block[:, points], dtype
                    )

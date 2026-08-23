# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any, Callable, List, Optional, Sequence, Tuple, Union,
)

import cftime
import netCDF4 as nc
import numpy as np
from pydantic import Field, PrivateAttr, field_validator, model_validator

from hydroforge.output.multirank.catalog import RankOutputCatalog
from hydroforge.output.multirank.data import (
    MultiRankDataAccess, _OutputTimeRequest,
)
from hydroforge.output.multirank.plan import (
    _ReaderFileIdentity, _ReaderStoragePlan,
)
from hydroforge.serialization.netcdf import decode_netcdf_logical_array
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.contracts.temporal import (
    normalize_calendar_dates,
)

logger = logging.getLogger(__name__)


_SERIES_READER_CONTEXT = "hydroforge_multirank_reader"


class _ReaderSeriesQuery(HydroForgeModel):
    """One fully resolved public time-series query."""

    points: Any
    level: int | None = Field(default=None, ge=0, strict=True)
    trial: int = Field(default=0, ge=0, strict=True)
    dtype: Any = None
    time_slice: slice | None = None

    _time_request: _OutputTimeRequest = PrivateAttr()
    _target_dtype: np.dtype = PrivateAttr()
    _rank_to_columns: dict[int, list[tuple[int, int]]] = PrivateAttr()

    @model_validator(mode="after")
    def _resolve_query(self, info):
        context = info.context
        reader = (
            context.get(_SERIES_READER_CONTEXT)
            if isinstance(context, Mapping) else None
        )
        if reader is None:
            raise ValueError("reader series query requires reader context")

        request = reader._data_access._make_series_request(
            time_slice=self.time_slice,
            level=self.level,
            trial=self.trial,
        )
        target_dtype = reader._data_access._result_dtype(self.dtype)

        raw = self.points
        if isinstance(raw, (list, tuple)):
            if raw and all(
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(np.isscalar(value) for value in item)
                for item in raw
            ):
                arrays = [np.asarray(raw)]
            else:
                arrays = [np.asarray(value) for value in raw]
        else:
            arrays = [np.asarray(raw)]

        if not arrays:
            use_xy = False
            queries: tuple[int | tuple[int, int], ...] = ()
        else:
            kinds = set()
            for array in arrays:
                if array.ndim == 2 and array.shape[1] == 2:
                    kinds.add("xy")
                elif array.ndim in {0, 1}:
                    kinds.add("id")
                else:
                    raise ValueError(
                        f"unsupported points shape: {array.shape}"
                    )
                if array.dtype.kind not in "iu":
                    raise ValueError(
                        "point IDs and XY coordinates must be integers"
                    )
                if (
                    array.dtype.kind == "u"
                    and array.size
                    and int(array.max()) > np.iinfo(np.int64).max
                ):
                    raise ValueError(
                        "point IDs and XY coordinates exceed int64 range"
                    )
            if len(kinds) != 1:
                raise ValueError(
                    "provide either all XY (N,2) or all IDs (N,); do not mix"
                )
            use_xy = kinds.pop() == "xy"
            if use_xy:
                queries = tuple(
                    (int(x), int(y))
                    for array in arrays
                    for x, y in np.asarray(array)
                )
            else:
                queries = tuple(
                    int(value)
                    for array in arrays
                    for value in np.asarray(array).ravel()
                )
        if len(queries) != len(set(queries)):
            raise ValueError("duplicate points are not allowed")

        self._time_request = request
        self._target_dtype = target_dtype
        self._rank_to_columns = reader._data_access.resolve_series_points(
            queries, use_xy=use_xy,
        )
        return self

    @property
    def time_request(self) -> _OutputTimeRequest:
        return self._time_request

    @property
    def target_dtype(self) -> np.dtype:
        return self._target_dtype

    @property
    def rank_to_columns(self) -> dict[int, list[tuple[int, int]]]:
        return self._rank_to_columns


class MultiRankStatsReader(HydroForgeModel):
    """
    Manage per‑rank NetCDF outputs written by a StatisticsRuntime-like component.

    Major Features:
      - Auto-detect rank files: {var_name}_rank{rank}.nc
      - Derive (x, y) locations using validated construction fields:
          * map_shape=(nx, ny)                        -> linear indices
          * map_shape_nc=Path(...)                    -> validated NetCDF shape
          * coord_converter=callable                  -> custom conversion
      - Provide vector / grid / time series extraction APIs
      - Basic visualization (single time slice + animation)
      - Export time-sliced grids to CaMa-Flood-compatible Fortran-order binary
    """

    base_dir: Path
    var_name: str
    coord_name: str | None = None
    map_shape_input: tuple[int, int] | None = Field(
        default=None,
        validation_alias="map_shape",
        serialization_alias="map_shape",
        repr=False,
        description="Explicit immutable map shape supplied by the caller",
    )
    map_shape_nc: Path | None = None
    coord_converter: (
        Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None
    ) = None
    time_range: tuple[
        datetime | cftime.datetime,
        datetime | cftime.datetime,
    ] | None = None
    cache_enabled: bool = False
    split_by_year: bool = False
    row_chunk_size: int | None = Field(default=None, ge=1, strict=True)

    _storage_plan: _ReaderStoragePlan = PrivateAttr()
    _data_access: MultiRankDataAccess = PrivateAttr()
    _rank_cache: dict[int, np.ndarray | None] = PrivateAttr(
        default_factory=dict,
    )
    _cache_materialized: bool = PrivateAttr(default=False)

    @property
    def _rank_files(self):
        return self._storage_plan.rank_files

    @property
    def _time_units(self) -> str:
        return self._storage_plan.time_units

    @property
    def _time_calendar(self) -> str:
        return self._storage_plan.time_calendar

    @property
    def _time_values_num(self) -> np.ndarray:
        return self._storage_plan.time_values_num

    @property
    def _time_datetimes(self):
        return self._storage_plan.time_datetimes

    @property
    def _time_len(self) -> int:
        return self._storage_plan.time_len

    @property
    def _slice_start(self) -> int:
        return self._storage_plan.slice_start

    @property
    def _slice_end(self) -> int:
        return self._storage_plan.slice_end

    @property
    def _t_indices(self) -> np.ndarray:
        return self._storage_plan.time_indices

    @field_validator("map_shape_input")
    @classmethod
    def _validate_map_shape(
        cls, value: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if value is None:
            return None
        if type(value) is not tuple or len(value) != 2:
            raise ValueError("map_shape must be an exact (nx, ny) tuple")
        if any(type(extent) is not int or extent < 1 for extent in value):
            raise ValueError("map_shape values must be exact positive ints")
        return value

    @property
    def map_shape(self) -> tuple[int, int] | None:
        """Return the construction-time-resolved coordinate grid shape."""

        return self._storage_plan.map_shape

    @model_validator(mode="after")
    def _validate_reader_declaration(self):
        if not self.var_name:
            raise ValueError("var_name must be non-empty")
        if (
            Path(self.var_name).name != self.var_name
            or "/" in self.var_name
            or "\\" in self.var_name
        ):
            raise ValueError("var_name must not contain path separators")
        if self.coord_name is not None and not self.coord_name:
            raise ValueError("coord_name must be None or non-empty")
        coordinate_sources = sum(
            source is not None
            for source in (
                self.map_shape_input,
                self.map_shape_nc,
                self.coord_converter,
            )
        )
        if coordinate_sources > 1:
            raise ValueError(
                "map_shape, map_shape_nc and coord_converter are mutually "
                "exclusive coordinate identity sources"
            )
        if self.time_range is not None:
            start, end = self.time_range
            _calendar, normalized, _defaulted = normalize_calendar_dates(
                {
                    "time_range start": start,
                    "time_range end": end,
                },
                calendar=None,
                preserve_cftime_declaration=True,
            )
            start = normalized["time_range start"]
            end = normalized["time_range end"]
            if start > end:
                raise ValueError(
                    "time_range start must be <= end (closed interval)"
                )
            object.__setattr__(self, "time_range", (start, end))
        return self

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------
    def _safe_time_str(self, t_obj, fmt="%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime-like objects without assuming one implementation."""
        if hasattr(t_obj, "strftime"):
             try:
                 return t_obj.strftime(fmt)
             except (TypeError, ValueError, OverflowError):
                 pass

        if hasattr(t_obj, "isoformat"):
             try:
                 return t_obj.isoformat()
             except (TypeError, ValueError, OverflowError):
                 pass

        return str(t_obj)

    def _preload_cache(self) -> None:
        """Preload only the chosen inclusive slice [self._slice_start, self._slice_end]."""
        for info in self._rank_files:
            if info["saved_points"] == 0:
                self._rank_cache[info["rank_id"]] = None
                continue

            cache = None
            cache_rows = self._slice_end - self._slice_start + 1

            # Iterate through files and extract relevant parts
            for i, fp in enumerate(info["paths"]):
                file_start_global, file_end_global = info[
                    "file_time_offsets"
                ][i]

                # Check intersection with requested slice [self._slice_start, self._slice_end]
                # Intersection: max(start1, start2) to min(end1, end2)
                req_start = max(self._slice_start, file_start_global)
                req_end = min(self._slice_end + 1, file_end_global) # exclusive end

                if req_start < req_end:
                    # Calculate local indices
                    local_start = req_start - file_start_global
                    local_end = req_end - file_start_global

                    path = self._checked_source_path(fp)
                    with nc.Dataset(path, "r") as ds:
                        var = ds.variables[self.var_name]
                        # Slicing logic: always take all spatial/trial dims.
                        # Dimensions are
                        # (time, [trial], saved_points, [value_axis]).
                        if self.row_chunk_size is None:
                            data = var[local_start:local_end, ...]
                            chunks = ((req_start, data),)
                        else:
                            chunks = (
                                (
                                    file_start_global + t0,
                                    var[
                                        t0:min(
                                            t0 + self.row_chunk_size,
                                            local_end,
                                        ),
                                        ...,
                                    ],
                                )
                                for t0 in range(
                                    local_start,
                                    local_end,
                                    self.row_chunk_size,
                                )
                            )

                        for global_start, chunk in chunks:
                            chunk = decode_netcdf_logical_array(
                                var, chunk, name=self.var_name,
                            )
                            array = self._data_access._array(
                                chunk, source=fp.name,
                            )
                            if cache is None:
                                cache = np.empty(
                                    (cache_rows, *array.shape[1:]),
                                    dtype=array.dtype,
                                )
                            destination = global_start - self._slice_start
                            cache[destination:destination + array.shape[0]] = array
                    self._verify_source_path(path)

            self._rank_cache[info["rank_id"]] = cache

    def _ensure_cache_materialized(self) -> None:
        """Lazily acquire the optional resident cache after validation."""

        if not self.cache_enabled or self._cache_materialized:
            return
        self._preload_cache()
        self._cache_materialized = True

    # ----------------------------------------------------------------------------------
    # Validated source construction
    # ----------------------------------------------------------------------------------
    def _compile_storage_plan(
        self, resolved_map_shape: tuple[int, int] | None,
    ) -> _ReaderStoragePlan:
        """
        time_range: CLOSED interval (start_dt, end_dt), both inclusive.
        """
        candidate_paths = tuple(
            path.absolute()
            for path in sorted(
                self.base_dir.glob(f"{self.var_name}_rank*.nc")
            )
        )
        identities = {
            path: _ReaderFileIdentity.capture(path)
            for path in candidate_paths
        }
        state = SimpleNamespace(
            base_dir=self.base_dir,
            var_name=self.var_name,
            coord_name=self.coord_name,
            split_by_year=self.split_by_year,
            coord_converter=self.coord_converter,
            map_shape=resolved_map_shape,
            _rank_files=[],
            _time_units=None,
            _time_calendar=None,
            _time_values_num=None,
            _time_datetimes=[],
            _time_len=0,
        )
        catalog = RankOutputCatalog(state)
        state._rank_files = catalog.scan()
        if not state._rank_files:
            raise FileNotFoundError(
                f"No files found in {self.base_dir} matching: {self.var_name}_rank*.nc"
            )

        catalog.read_timeline()

        # Apply closed datetime slice with strict range checking (no clamping)
        if self.time_range is not None:
            # Strategy: Convert input range to numeric values using the NetCDF unit/calendar.
            start_in, end_in = self.time_range
            _calendar, normalized, _defaulted = normalize_calendar_dates(
                {
                    "time_range start": start_in,
                    "time_range end": end_in,
                },
                calendar=state._time_calendar,
            )
            start_in = normalized["time_range start"]
            end_in = normalized["time_range end"]
            object.__setattr__(self, "time_range", (start_in, end_in))

            t_start_val = nc.date2num(
                start_in, state._time_units, state._time_calendar,
            )
            t_end_val = nc.date2num(
                end_in, state._time_units, state._time_calendar,
            )
            file_min = state._time_values_num[0]
            file_max = state._time_values_num[-1]
            if t_start_val < file_min or t_end_val > file_max:
                raise ValueError(
                    "time_range outside available coverage. "
                    f"Requested [{self._safe_time_str(start_in)} .. "
                    f"{self._safe_time_str(end_in)}] but coverage is "
                    f"[{self._safe_time_str(state._time_datetimes[0])} .. "
                    f"{self._safe_time_str(state._time_datetimes[-1])}]."
                )

            valid_mask = (
                (state._time_values_num >= t_start_val)
                & (state._time_values_num <= t_end_val)
            )
            indices = np.flatnonzero(valid_mask)
            if indices.size == 0:
                raise ValueError("No time steps found in the request range.")
            left = int(indices[0])
            right = int(indices[-1])

            slice_start = left
            slice_end = right
            time_indices = np.arange(left, right + 1, dtype=np.int64)
            time_values = state._time_values_num[time_indices]
            time_datetimes = [
                state._time_datetimes[index] for index in time_indices
            ]
        else:
            slice_start = 0
            slice_end = state._time_len - 1
            time_indices = np.arange(state._time_len, dtype=np.int64)
            time_values = state._time_values_num
            time_datetimes = state._time_datetimes

        catalog.compute_coordinates()
        for identity_path, identity in identities.items():
            identity.verify(identity_path)
        return _ReaderStoragePlan.compile(
            rank_files=state._rank_files,
            time_units=state._time_units,
            time_calendar=state._time_calendar,
            time_values_num=time_values,
            time_datetimes=time_datetimes,
            slice_start=slice_start,
            slice_end=slice_end,
            time_indices=time_indices,
            map_shape=resolved_map_shape,
            file_identities=identities,
        )

    def _checked_source_path(self, path: str | Path) -> Path:
        return self._storage_plan.checked_path(path)

    def _verify_source_path(self, path: str | Path) -> None:
        self._storage_plan.verify_path(path)

    def _rank_cache_for(self, rank_id: int) -> np.ndarray | None:
        return self._rank_cache.get(rank_id)

    @model_validator(mode="after")
    def _inspect_reader_storage(self):
        """Resolve external sources after every semantic validator succeeds."""

        try:
            resolved_map_shape = (
                self._read_map_shape_from_nc(self.map_shape_nc)
                if self.map_shape_nc is not None
                else self.map_shape_input
            )
            self._storage_plan = self._compile_storage_plan(
                resolved_map_shape,
            )
            self._data_access = MultiRankDataAccess(self)
        except (
            KeyError, IndexError, OSError, RuntimeError, TypeError, ValueError,
            OverflowError,
        ) as error:
            raise ValueError(str(error)) from error
        return self

    # ----------------------------------------------------------------------------------
    # Data getters
    # ----------------------------------------------------------------------------------
    def _get_vector(
        self, t_index: int, level: Optional[int] = None, trial: int = 0,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        self._ensure_cache_materialized()
        return self._data_access.get_vector(t_index, level, trial, dtype)

    def _get_grid(
        self, t_index: int, level: Optional[int] = None, trial: int = 0,
        fill_value: float = np.nan, dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        self._ensure_cache_materialized()
        return self._data_access.get_grid(
            t_index, level, trial, fill_value, dtype,
        )

    def get_series(
        self, points: Union[np.ndarray, Sequence[np.ndarray]],
        level: Optional[int] = None, trial: int = 0,
        dtype: Optional[np.dtype] = None,
        *, time_slice: slice | None = None,
    ) -> np.ndarray:
        """Read point series over a half-open slice of this reader's view."""

        query = _ReaderSeriesQuery.model_validate(
            {
                "points": points,
                "level": level,
                "trial": trial,
                "dtype": dtype,
                "time_slice": time_slice,
            },
            context={_SERIES_READER_CONTEXT: self},
        )
        self._ensure_cache_materialized()
        return self._data_access.get_series(query)

    @property
    def time_len(self) -> int:
        """Return the validated reader view's number of time rows."""

        return self._time_len

    @property
    def times(self) -> tuple[datetime | cftime.datetime, ...]:
        """Return the immutable validated reader timeline."""

        return tuple(self._time_datetimes)

    @staticmethod
    def _map_extent(value, *, label: str) -> int:
        if np.ma.isMaskedArray(value) and np.any(
            np.ma.getmaskarray(value)
        ):
            raise ValueError(f"{label} contains missing values")
        array = np.asarray(value)
        if array.shape != ():
            raise ValueError(f"{label} must be a scalar positive integer")
        scalar = array.item()
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(
            scalar, (int, np.integer),
        ):
            raise ValueError(f"{label} must be an integer")
        result = int(scalar)
        if result < 1:
            raise ValueError(f"{label} must be positive")
        return result

    @classmethod
    def _read_map_shape_from_nc(
        cls,
        nc_path: Path,
    ) -> tuple[int, int]:
        p = Path(nc_path)
        with nc.Dataset(p, "r") as ds:
            attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
            candidates: list[tuple[str, tuple[int, int]]] = []
            if ("nx" in attrs) != ("ny" in attrs):
                raise ValueError("map-shape attributes must define both nx and ny")
            if "nx" in attrs:
                candidates.append(("nx/ny attributes", (
                    cls._map_extent(attrs["nx"], label="nx attribute"),
                    cls._map_extent(attrs["ny"], label="ny attribute"),
                )))
            if ("nx" in ds.variables) != ("ny" in ds.variables):
                raise ValueError("map-shape variables must define both nx and ny")
            if "nx" in ds.variables:
                candidates.append(("nx/ny variables", (
                    cls._map_extent(ds.variables["nx"][:], label="nx variable"),
                    cls._map_extent(ds.variables["ny"][:], label="ny variable"),
                )))
            if "map_shape" in ds.variables:
                raw_shape = ds.variables["map_shape"][:]
                if np.ma.isMaskedArray(raw_shape) and np.any(
                    np.ma.getmaskarray(raw_shape)
                ):
                    raise ValueError("map_shape variable contains missing values")
                arr = np.asarray(raw_shape)
                if arr.shape != (2,):
                    raise ValueError("map_shape variable must have shape (2,)")
                candidates.append(("map_shape variable", (
                    cls._map_extent(arr[0], label="map_shape[0]"),
                    cls._map_extent(arr[1], label="map_shape[1]"),
                )))
            if "map_shape" in attrs:
                arr = np.asarray(attrs["map_shape"])
                if arr.shape != (2,):
                    raise ValueError("map_shape attribute must have shape (2,)")
                candidates.append(("map_shape attribute", (
                    cls._map_extent(arr[0], label="map_shape[0]"),
                    cls._map_extent(arr[1], label="map_shape[1]"),
                )))
            for a, b in (("nx", "ny"), ("x", "y"), ("lon", "lat")):
                if a in ds.dimensions and b in ds.dimensions:
                    candidates.append((f"{a}/{b} dimensions", (
                        cls._map_extent(ds.dimensions[a].size, label=a),
                        cls._map_extent(ds.dimensions[b].size, label=b),
                    )))
        if not candidates:
            raise KeyError("Could not find nx/ny or map_shape (attrs/vars/dims).")
        shapes = {shape for _source, shape in candidates}
        if len(shapes) != 1:
            raise ValueError(
                "conflicting map-shape metadata: "
                + ", ".join(
                    f"{source}={shape}" for source, shape in candidates
                )
            )
        return candidates[0][1]

    # ----------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------
    def get_all_coords_xy(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for info in self._rank_files:
            if info["saved_points"] == 0:
                continue
            if info["x"] is None or info["y"] is None:
                return None, None
            xs.append(info["x"])
            ys.append(info["y"])
        if not xs:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(xs), np.concatenate(ys)

    def get_all_cids(self) -> Optional[np.ndarray]:
        cids: List[np.ndarray] = []
        for info in self._rank_files:
            if info["saved_points"] == 0 or info["coord_raw"] is None:
                continue
            cids.append(info["coord_raw"])
        if not cids:
            return None
        return np.concatenate(cids)

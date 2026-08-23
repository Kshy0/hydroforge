# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import cftime
import numpy as np
from netCDF4 import Dataset
from pydantic import Field, PrivateAttr, field_validator, model_validator

from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.base import (
    _TrustedSourceChunk,
    _trusted_source_chunk_payload,
    positive_finite_real,
)
from hydroforge.data.datasets.gridded import GriddedDataset
from hydroforge.data.datasets.timeline import DatasetTimeline, ReadOp
from hydroforge.data.netcdf import (
    _NetCDFReadHandlePool,
    _configure_netcdf_variable_cache,
    _planned_netcdf_chunk_len,
    _read_netcdf_var_sliced_trusted,
    yearly_time_to_key,
)
from hydroforge.data.numeric import canonical_float64, immutable_array
from hydroforge.contracts.temporal import DateLike
from hydroforge.contracts.validation import HydroForgeModel


class NetCDFDataset(GriddedDataset):
    """NetCDF-backed dataset with minimal I/O and a compact design.

    Key ideas:
    - Scan only time variables to build a global timeline and a dt->(file_key, local_index)
      map. No heavy data read during initialization.
    - Group requested timestamps into contiguous slices per file so each chunk is read with
      as few NetCDF reads as possible (often 1-2 reads per chunk).
    - Normalize variable dimensions to (T, Y, X) once per read; precompute a spatial mask
      and use a linear index list to quickly collapse (Y, X) -> N.
    """

    supports_time_aggregation: ClassVar[bool] = True

    base_dir: str | Path
    var_name: str
    prefix: str
    chunk_len: int | None = Field(default=None, strict=True, ge=1)
    unit_factor: float = 1.0
    suffix: str = ".nc"
    time_to_key: Callable[[DateLike], str] = yearly_time_to_key
    time_aggregation: str | Mapping[str, str] | None = None

    _bbox: tuple[int, int, int, int] | None = PrivateAttr(default=None)
    _bbox_local_indices: np.ndarray | None = PrivateAttr(default=None)
    _coordinates_cache: tuple[np.ndarray, np.ndarray] | None = PrivateAttr(
        default=None,
    )
    _grid_shape_cache: tuple[int, int] | None = PrivateAttr(default=None)
    _source_dtype: np.dtype | None = PrivateAttr(default=None)
    _coordinate_units_cache: tuple[str | None, str | None] | None = PrivateAttr(
        default=None,
    )
    _variable_axes_by_path: Mapping[Path, tuple[int, int, int]] = PrivateAttr(
        default_factory=dict
    )
    _timeline: DatasetTimeline = PrivateAttr()
    _read_handles: _NetCDFReadHandlePool = PrivateAttr(
        default_factory=_NetCDFReadHandlePool,
    )

    @field_validator("unit_factor")
    @classmethod
    def _validate_unit_factor(cls, value: float) -> float:
        return positive_finite_real(value, label="unit_factor")

    @field_validator("time_aggregation")
    @classmethod
    def _validate_aggregation(
        cls,
        value: str | Mapping[str, str] | None,
    ) -> str | Mapping[str, str] | None:
        return cls._normalize_time_aggregation(value)

    @model_validator(mode="after")
    def _inspect_netcdf_storage(self):
        if self.chunk_len is None:
            storage_start = self._storage_time(self.start_date)
            key = self.time_to_key(storage_start)
            if type(key) is not str:
                raise TypeError("time_to_key must return an exact string")
            path = Path(self.base_dir, f"{self.prefix}{key}{self.suffix}")
            object.__setattr__(
                self,
                "chunk_len",
                _planned_netcdf_chunk_len(path, self.var_name),
            )
            self._install_temporal_domain(self._temporal_domain)
        self._timeline = DatasetTimeline(
            self,
            base_dir=self.base_dir,
            prefix=self.prefix,
            suffix=self.suffix,
            time_to_key=self.time_to_key,
            time_aggregation=self.time_aggregation,
            data_variable=self.var_name,
        )
        axes_by_path: dict[Path, tuple[int, int, int]] = {}
        for key in sorted(self._timeline.file_times):
            path = Path(
                self.base_dir,
                f"{self.prefix}{key}{self.suffix}",
            )
            with Dataset(path, "r") as dataset:
                axes_by_path[self._canonical_source_path(path)] = (
                    self._validate_shard_coordinates(dataset, path)
                )
        self._variable_axes_by_path = axes_by_path
        self._validate_local_index_extent(
            self._grid_shape[0] * self._grid_shape[1],
            label="NetCDF grid",
        )
        self._compute_bbox_from_indices()
        self._record_source_files(
            Path(self.base_dir, f"{self.prefix}{key}{self.suffix}")
            for key in sorted(self._timeline.file_times)
        )
        return self

    @staticmethod
    def _storage_time(
        logical_time: Union[datetime, cftime.datetime],
    ) -> Union[datetime, cftime.datetime]:
        """Map public logical time to the timestamp stored on disk."""

        return logical_time

    # -------------------------
    # Variable shape helpers
    # -------------------------
    @staticmethod
    def _pick_dim(dim_names: Tuple[str, ...], *candidates: str) -> Optional[int]:
        m = {name: index for index, name in enumerate(dim_names)}
        matches = [(name, m[name]) for name in candidates if name in m]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous dimensions {[name for name, _ in matches]} in {dim_names}"
            )
        return None if not matches else matches[0][1]

    @classmethod
    def _tyx_axes(cls, dim_names: Tuple[str, ...]) -> tuple[int, int, int]:
        t_idx = cls._pick_dim(dim_names, "time", "valid_time")
        y_idx = cls._pick_dim(dim_names, "lat", "latitude", "y")
        x_idx = cls._pick_dim(dim_names, "lon", "longitude", "long", "x")
        if len(dim_names) != 3 or t_idx is None or y_idx is None or x_idx is None:
            raise ValueError(
                "NetCDF forcing variable must have exactly one time, one "
                f"latitude, and one longitude dimension; got {dim_names}"
            )
        return t_idx, y_idx, x_idx

    @staticmethod
    def _ensure_tyx(
        data: np.ndarray, t_idx: Optional[int], y_idx: int, x_idx: int
    ) -> np.ndarray:
        """Transpose one exact rank-three variable to ``(T, Y, X)``."""
        return np.transpose(data, axes=(cast(int, t_idx), y_idx, x_idx))

    @staticmethod
    def _coordinate_axis(
        dataset: Dataset,
        *,
        axis_dim: str,
        names: tuple[str, ...],
        label: str,
        path: Path,
    ) -> tuple[np.ndarray, str | None]:
        candidates = [
            dataset.variables[name]
            for name in names
            if (
                name in dataset.variables
                and dataset.variables[name].dimensions == (axis_dim,)
            )
        ]
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise ValueError(
                f"Unable to find a one-dimensional {label} coordinate for "
                f"dimension {axis_dim!r} in {path.name}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous {label} coordinates for dimension {axis_dim!r} "
                f"in {path.name}: {[item.name for item in candidates]}"
            )
        raw = candidates[0][:]
        if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
            raise ValueError(
                f"{label} coordinate {candidates[0].name!r} in {path.name} "
                "contains missing values"
            )
        values = np.asarray(raw)
        if values.ndim != 1:
            raise TypeError(
                f"{label} coordinate in {path.name} must be a "
                "one-dimensional real numeric array"
            )
        canonical = canonical_float64(
            raw,
            label=f"{label} coordinate in {path.name}",
        )
        if np.unique(canonical).size != canonical.size:
            raise ValueError(
                f"{label} coordinate in {path.name} contains duplicate values"
            )
        coordinate = candidates[0]
        if "units" not in coordinate.ncattrs():
            units = None
        else:
            units = coordinate.getncattr("units")
            if not isinstance(units, str) or not units.strip():
                raise ValueError(
                    f"{label} coordinate {coordinate.name!r} in {path.name} "
                    "must define units as a non-empty string when present"
                )
        return canonical, units

    def _validate_shard_coordinates(
        self,
        dataset: Dataset,
        path: Path,
    ) -> tuple[int, int, int]:
        """Require every time shard to use one exact spatial grid and order."""

        variable = dataset.variables[self.var_name]
        source_dtype = np.dtype(variable.dtype)
        if source_dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                f"NetCDF forcing variable {self.var_name!r} in {path.name} "
                f"must use a real numeric dtype; got {source_dtype}"
            )
        if self._source_dtype is None:
            self._source_dtype = source_dtype
        elif source_dtype != self._source_dtype:
            raise ValueError(
                f"NetCDF forcing dtype in shard {path.name} does not match "
                f"the canonical dtype {self._source_dtype}"
            )
        dimensions = tuple(variable.dimensions)
        _t_idx, y_idx, x_idx = self._tyx_axes(dimensions)
        grid_shape = (variable.shape[y_idx], variable.shape[x_idx])
        latitude, latitude_units = self._coordinate_axis(
            dataset,
            axis_dim=dimensions[y_idx],
            names=(dimensions[y_idx], "lat", "latitude", "y"),
            label="latitude",
            path=path,
        )
        longitude, longitude_units = self._coordinate_axis(
            dataset,
            axis_dim=dimensions[x_idx],
            names=(dimensions[x_idx], "lon", "longitude", "long", "x"),
            label="longitude",
            path=path,
        )
        observed_units = (longitude_units, latitude_units)
        if self._coordinates_cache is None:
            self._coordinates_cache = (
                immutable_array(longitude, order="C"),
                immutable_array(latitude, order="C"),
            )
            self._coordinate_units_cache = observed_units
            self._grid_shape_cache = grid_shape
        else:
            expected_longitude, expected_latitude = self._coordinates_cache
            if (
                longitude.shape != expected_longitude.shape
                or latitude.shape != expected_latitude.shape
                or not np.array_equal(longitude, expected_longitude)
                or not np.array_equal(latitude, expected_latitude)
            ):
                raise ValueError(
                    f"spatial coordinates in shard {path.name} do not match "
                    "the canonical shard in content and order"
                )
            if observed_units != self._coordinate_units_cache:
                raise ValueError(
                    f"spatial coordinate units in shard {path.name} do not "
                    "match the canonical shard"
                )
            if grid_shape != self._grid_shape_cache:
                raise ValueError(
                    f"spatial shape in shard {path.name} does not match "
                    "the canonical shard"
                )
        return _t_idx, y_idx, x_idx

    @property
    def _grid_shape(self) -> Tuple[int, int]:
        """Return the spatial shape validated for every source shard."""

        return cast(tuple[int, int], self._grid_shape_cache)

    def _compute_bbox_from_indices(self) -> None:
        """Compute 2D bounding box from local_indices for optimized reading.

        This method converts the 1D flattened indices to 2D (y, x) coordinates,
        finds the minimal bounding box, and creates a mapping from the original
        indices to indices relative to the bounding box.

        After calling this method:
        - self._bbox: (y_min, y_max, x_min, x_max) - inclusive bounds
        - self._bbox_local_indices: indices relative to the bounding box flatten
        """
        if self.local_indices is None:
            self._bbox = None
            self._bbox_local_indices = None
            return
        if self.local_indices.size == 0:
            self._bbox = None
            self._bbox_local_indices = None
            return

        ny, nx = self._grid_shape

        # Convert 1D indices to 2D coordinates
        # index = y * nx + x (C-order, row-major)
        y_coords = self.local_indices // nx
        x_coords = self.local_indices % nx

        # Compute bounding box
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        x_min, x_max = int(x_coords.min()), int(x_coords.max())

        self._bbox = (y_min, y_max, x_min, x_max)

        # Compute new width of the bounding box
        bbox_nx = x_max - x_min + 1

        # Convert global indices to bbox-local indices
        # new_index = (y - y_min) * bbox_nx + (x - x_min)
        local_y = y_coords - y_min
        local_x = x_coords - x_min
        self._bbox_local_indices = (local_y * bbox_nx + local_x).astype(np.int64)

    def _read_ops(self, ops: Sequence[ReadOp]) -> np.ndarray:
        """Execute per-file reads using absolute time indices.

        Each op is (file_key, abs_indices). Sequence indices are converted to
        contiguous NetCDF slices, then restored to the requested order in memory.

        When local_indices is set and a bounding box has been computed,
        this method reads only the bounding box region instead of the full grid,
        significantly reducing I/O for spatially concentrated catchments.

        Returns:
        - If local_indices is set: (T, N) compressed array
        - If local_indices is None: (T, Y, X) full grid array

        Spatial convention: (Y, X) = (lat, lon), C-order flatten (lon varies fastest)
        """
        ny, nx = self._grid_shape
        compressed = self.local_indices is not None

        if compressed and len(self.local_indices) == 0:
            total_len = sum(len(abs_indices) for _key, abs_indices in ops)
            return np.empty((total_len, 0), dtype=self.out_dtype)

        use_bbox = (
            compressed
            and self._bbox is not None
            and self._bbox_local_indices is not None
        )

        if not ops:
            if compressed:
                return np.empty((0, len(self.local_indices)), dtype=self.out_dtype)
            else:
                return np.empty((0, ny, nx), dtype=self.out_dtype)

        chunks: List[np.ndarray] = []

        for key, abs_indices in ops:
            path = self._checked_source_path(
                Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}",
            )
            with self._read_handles.acquire(path) as ds:
                var = ds.variables[self.var_name]
                t_idx, y_idx, x_idx = self._variable_axes_by_path[path]

                if not abs_indices:
                    continue

                abs_idx = np.asarray(abs_indices, dtype=np.int64)
                sel = [slice(None)] * var.ndim
                sel[t_idx] = abs_idx

                if use_bbox:
                    # Read only the bounding box region
                    y_min, y_max, x_min, x_max = self._bbox
                    sel[y_idx] = slice(y_min, y_max + 1)
                    sel[x_idx] = slice(x_min, x_max + 1)

                selectors = tuple(sel)
                _configure_netcdf_variable_cache(
                    var, selectors, time_axis=t_idx,
                )
                arr = _read_netcdf_var_sliced_trusted(var, selectors)

                # Normalize to (T, Y, X); Y/X may describe only the bbox.
                arr = self._ensure_tyx(arr, t_idx, y_idx, x_idx)

                if compressed:
                    # Flatten and extract active columns: (T, Y, X) -> (T, N)
                    T, Y, X = arr.shape
                    flat = arr.reshape(T, Y * X)
                    out = flat[:, self._bbox_local_indices]
                else:
                    out = arr

                out = self._as_nan_array(out)
                if np.issubdtype(out.dtype, np.floating):
                    missing = np.isnan(out)
                    if np.any(missing):
                        out = np.array(out, order="C", copy=True)
                        out[missing] = 0.0
                out = _trusted_source_chunk_payload(
                    out,
                    expected_rows=len(abs_indices),
                    clip_negative=self.clip_negative,
                )

            chunks.append(out)
            self._verify_source_path(path)

        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def _get_first_frame_nan_mask(self) -> Optional[np.ndarray]:
        """Read the first planned source frame and return a flat NaN/mask bitmap."""
        if not self._timeline.plan:
            return None

        first_op = None
        for entry in self._timeline.plan:
            for key, abs_indices in entry.operations:
                if abs_indices:
                    first_op = (key, int(abs_indices[0]))
                    break
            if first_op is not None:
                break

        if first_op is None:
            return None

        key, abs_index = first_op
        path = self._checked_source_path(
            Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}",
        )
        with self._read_handles.acquire(path) as ds:
            var = ds.variables[self.var_name]
            t_idx, y_idx, x_idx = self._variable_axes_by_path[path]

            sel = [slice(None)] * var.ndim
            sel[t_idx] = np.asarray([abs_index], dtype=np.int64)
            selectors = tuple(sel)
            _configure_netcdf_variable_cache(var, selectors, time_axis=t_idx)
            arr = _read_netcdf_var_sliced_trusted(var, selectors)
            arr = self._as_nan_array(arr)
            arr = self._ensure_tyx(arr, t_idx, y_idx, x_idx)
        self._verify_source_path(path)

        if not np.issubdtype(arr.dtype, np.floating):
            return np.zeros(arr.shape[1:], dtype=bool)
        return np.isnan(arr[0])

    def _finish_read(
        self, data: np.ndarray
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        calculation = self._canonical_calculation_data(
            data,
            label="NetCDF dataset input",
        )
        if self.time_aggregation is None:
            np.divide(calculation, self.unit_factor, out=calculation)
            converted = calculation
        else:
            aggregated = self._apply_time_aggregation(
                calculation,
                self._timeline.source_time_interval,
                self.time_aggregation,
            )
            if isinstance(aggregated, dict):
                for block in aggregated.values():
                    np.divide(block, self.unit_factor, out=block)
                converted = aggregated
            else:
                np.divide(aggregated, self.unit_factor, out=aggregated)
                converted = aggregated
        return self._finalize_output_data(
            converted,
            label="NetCDF dataset output",
        )

    def _read_chunk(
        self,
        chunk: SourceChunk,
    ) -> _TrustedSourceChunk:
        """Read one validated source request through the compiled I/O plan."""

        ops = self._timeline.read_for_chunk(chunk).operations
        data = self._read_ops(ops)
        return _TrustedSourceChunk(self._finish_read(data))

    def close(self) -> None:
        """Close this process's persistent NetCDF read handles."""

        self._read_handles.close()

    # -------------------------
    # Public API
    # -------------------------
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the canonical grid validated at Dataset construction."""

        return cast(tuple[np.ndarray, np.ndarray], self._coordinates_cache)


class _OpenMultivariableNetCDFRequest(HydroForgeModel):
    base_dir: str | Path
    var_specs: Any
    start_date: DateLike
    end_date: DateLike
    model_step: timedelta
    time_interval: timedelta = timedelta(days=1)
    calendar: str | None = None
    spin_up_cycles: int = Field(default=0, strict=True, ge=0)
    spin_up_start_date: DateLike | None = None
    spin_up_end_date: DateLike | None = None
    chunk_len: int | None = Field(default=None, ge=1, strict=True)
    unit_factor: float = 1.0
    suffix: str = ".nc"
    clip_negative: bool = Field(default=False, strict=True)
    time_to_key: Callable[[DateLike], str] = yearly_time_to_key

    _compiled_specs: tuple[tuple[str, dict[str, Any]], ...] = PrivateAttr()

    @model_validator(mode="after")
    def _validate_factory(self):
        from hydroforge.data.datasets.multivariable import (
            compile_variable_specs,
        )

        self._compiled_specs = compile_variable_specs(self.var_specs)
        return self

    @property
    def compiled_specs(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        return self._compiled_specs


def open_multivariable_netcdf(
    base_dir: str | Path,
    var_specs: Mapping[str, Mapping[str, Any]],
    *,
    start_date: DateLike,
    end_date: DateLike,
    model_step: timedelta,
    time_interval: timedelta = timedelta(days=1),
    calendar: str | None = None,
    spin_up_cycles: int = 0,
    spin_up_start_date: DateLike | None = None,
    spin_up_end_date: DateLike | None = None,
    chunk_len: int | None = None,
    unit_factor: float = 1.0,
    suffix: str = ".nc",
    clip_negative: bool = False,
    time_to_key: Callable[[Union[datetime, cftime.datetime]], str] = yearly_time_to_key,
):
    """Open aligned gridded variables as one generic composite."""
    from hydroforge.data.datasets.multivariable import (
        GriddedMultiVariableDataset,
    )

    request = _OpenMultivariableNetCDFRequest(
        base_dir=base_dir,
        var_specs=var_specs,
        start_date=start_date,
        end_date=end_date,
        model_step=model_step,
        time_interval=time_interval,
        calendar=calendar,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
        chunk_len=chunk_len,
        unit_factor=unit_factor,
        suffix=suffix,
        clip_negative=clip_negative,
        time_to_key=time_to_key,
    )
    shared = {
        "base_dir": request.base_dir,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "model_step": request.model_step,
        "time_interval": request.time_interval,
        "calendar": request.calendar,
        "spin_up_cycles": request.spin_up_cycles,
        "spin_up_start_date": request.spin_up_start_date,
        "spin_up_end_date": request.spin_up_end_date,
        "chunk_len": request.chunk_len,
        "unit_factor": request.unit_factor,
        "suffix": request.suffix,
        "clip_negative": request.clip_negative,
        "time_to_key": request.time_to_key,
    }
    if request.chunk_len is None:
        first_name, first_spec = request.compiled_specs[0]
        first_prefix = first_spec.get("prefix", f"{first_name}_")
        first_suffix = first_spec.get("suffix", request.suffix)
        first_key = request.time_to_key(request.start_date)
        first_path = Path(
            request.base_dir,
            f"{first_prefix}{first_key}{first_suffix}",
        )
        shared["chunk_len"] = _planned_netcdf_chunk_len(
            first_path,
            first_name,
        )
    datasets = {}
    for name, spec in request.compiled_specs:
        options = shared | spec
        options["var_name"] = name
        if "prefix" not in options:
            options["prefix"] = f"{name}_"
        datasets[name] = NetCDFDataset(**options)

    return GriddedMultiVariableDataset(datasets=datasets)

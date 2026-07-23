# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cftime
import numpy as np
from netCDF4 import Dataset

from hydroforge.data.datasets.gridded import GriddedDataset
from hydroforge.data.datasets.timeline import DatasetTimeline
from hydroforge.data.netcdf import read_netcdf_var_sliced, single_file_key, yearly_time_to_key


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
    supports_time_aggregation = True

    def __init__(
        self,
        base_dir: str,
        start_date: Union[datetime, cftime.datetime],
        end_date: Union[datetime, cftime.datetime],
        var_name: str,
        prefix: str,
        time_interval: timedelta = timedelta(days=1),
        chunk_len: int = 24,
        unit_factor: float = 1.0,
        suffix: str = ".nc",
        time_to_key: Optional[Callable[[Union[datetime, cftime.datetime]], str]] = yearly_time_to_key,
        time_aggregation: Optional[Union[str, Dict[str, str]]] = None,
        clip_negative: bool = False,
        *args,
        **kwargs,
    ):
        self.base_dir = base_dir
        self.unit_factor = unit_factor
        self.var_name = var_name
        self.prefix = prefix
        self.suffix = suffix
        self.time_to_key = time_to_key if time_to_key is not None else single_file_key
        self.time_aggregation = self._normalize_time_aggregation(time_aggregation)
        # Bounding box for optimized spatial reading (computed lazily)
        self._bbox: Optional[Tuple[int, int, int, int]] = None  # (y_min, y_max, x_min, x_max)
        self._bbox_local_indices: Optional[np.ndarray] = None  # indices relative to bbox

        # Build time metadata and per-chunk minimal-IO plans up-front (cheap).
        super().__init__(
            time_interval=time_interval,
            start_date=start_date,
            end_date=end_date,
            chunk_len=chunk_len,
            clip_negative=clip_negative,
            *args,
            **kwargs,
        )

        self._timeline = DatasetTimeline(
            self,
            base_dir=base_dir,
            prefix=prefix,
            suffix=suffix,
            time_to_key=self.time_to_key,
            time_aggregation=self.time_aggregation,
        )
        self._file_times = self._timeline.file_times
        self._global_times = self._timeline.global_times
        self._dt_to_loc = self._timeline.dt_to_loc
        self.source_time_interval = self._timeline.source_time_interval
        self._aggregation_factor = self._timeline.aggregation_factor
        self._plan = self._timeline.plan
        self._spin_up_chunks_template = self._timeline.spin_up_chunks_template
        self._aggregation_plan_enabled = self.time_aggregation is not None

    def _ops_from_times(self, times):
        return self._timeline.ops_from_times(times)

    def _build_plan_entry(self, times):
        return self._timeline.build_entry(times)

    def is_valid_time_index(self, idx: int) -> bool:
        """
        Checks if the given time index corresponds to a valid data step (not padding).
        """
        return self._timeline.is_valid_time_index(idx)

    # -------------------------
    # Variable shape helpers
    # -------------------------
    @staticmethod
    def _pick_dim(dim_names: Tuple[str, ...], *candidates: str) -> Optional[int]:
        m = {n.lower(): i for i, n in enumerate(dim_names)}
        for c in candidates:
            if c in m:
                return m[c]
        return None

    @staticmethod
    def _ensure_tyx(data: np.ndarray, t_idx: Optional[int], y_idx: int, x_idx: int) -> np.ndarray:
        """Transpose data so that axes become (T, Y, X). Extra dims must be size-1."""
        axes = list(range(data.ndim))
        if t_idx is None:
            raise ValueError("A time dimension is required in the variable.")
        front = [t_idx, y_idx, x_idx]
        back = [a for a in axes if a not in front]
        data = np.transpose(data, axes=front + back)
        if data.ndim > 3:
            # Collapse any trailing size-1 dims
            tail = data.shape[3:]
            if any(s != 1 for s in tail):
                raise ValueError(f"Unsupported extra non-size-1 dims after time/lat/lon: shape={data.shape}")
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2])
        return data

    @cached_property
    def _grid_shape(self) -> Tuple[int, int]:
        """Get (ny, nx) grid dimensions from the first file."""
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            var = ds.variables[self.var_name]
            dims = var.dimensions
            y_idx = self._pick_dim(dims, "lat", "latitude", "y")
            x_idx = self._pick_dim(dims, "lon", "longitude", "long", "x")
            if y_idx is None or x_idx is None:
                raise ValueError(f"Unable to recognize lat/lon dims in {dims}")
            shape = var.shape
            return (shape[y_idx], shape[x_idx])

    def _compute_bbox_from_indices(self) -> None:
        """Compute 2D bounding box from _local_indices for optimized reading.

        This method converts the 1D flattened indices to 2D (y, x) coordinates,
        finds the minimal bounding box, and creates a mapping from the original
        indices to indices relative to the bounding box.

        After calling this method:
        - self._bbox: (y_min, y_max, x_min, x_max) - inclusive bounds
        - self._bbox_local_indices: indices relative to the bounding box flatten
        """
        if self._local_indices is None:
            self._bbox = None
            self._bbox_local_indices = None
            return

        ny, nx = self._grid_shape

        # Convert 1D indices to 2D coordinates
        # index = y * nx + x (C-order, row-major)
        y_coords = self._local_indices // nx
        x_coords = self._local_indices % nx

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

    def _read_ops(self, ops: List[Tuple[str, List[int]]]) -> np.ndarray:
        """Execute per-file reads using absolute time indices.

        Each op is (file_key, abs_indices). Sequence indices are converted to
        contiguous NetCDF slices, then restored to the requested order in memory.

        When _local_indices is set and a bounding box has been computed,
        this method reads only the bounding box region instead of the full grid,
        significantly reducing I/O for spatially concentrated catchments.

        Returns:
        - If _local_indices is set: (T, N) compressed array
        - If _local_indices is None: (T, Y, X) full grid array

        Spatial convention: (Y, X) = (lat, lon), C-order flatten (lon varies fastest)
        """
        ny, nx = self._grid_shape
        compressed = self._local_indices is not None

        if compressed and len(self._local_indices) == 0:
            total_len = sum(len(abs_indices) for _key, abs_indices in ops)
            return np.empty((total_len, 0), dtype=self.out_dtype)

        # Lazily compute bounding box on first read if compressed mode is active
        if compressed and self._bbox is None:
            self._compute_bbox_from_indices()

        # Check if bounding box optimization is available
        use_bbox = (compressed and
                    self._bbox is not None and
                    self._bbox_local_indices is not None)
        if compressed and not use_bbox:
            raise RuntimeError(
                "compressed NetCDF reads require initialized bounding-box indices"
            )

        if not ops:
            if compressed:
                return np.empty((0, len(self._local_indices)), dtype=self.out_dtype)
            else:
                return np.empty((0, ny, nx), dtype=self.out_dtype)

        chunks: List[np.ndarray] = []

        for key, abs_indices in ops:
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                var = ds.variables[self.var_name]
                dims = var.dimensions
                t_idx = self._pick_dim(dims, "time", "valid_time")
                y_idx = self._pick_dim(dims, "lat", "latitude", "y")
                x_idx = self._pick_dim(dims, "lon", "longitude", "long", "x")
                if t_idx is None or y_idx is None or x_idx is None:
                    raise ValueError(f"Expect at least time/lat/lon dims, got: {dims}")

                if not abs_indices:
                    continue

                abs_idx = np.asarray(abs_indices, dtype=np.int32)
                sel = [slice(None)] * var.ndim
                sel[t_idx] = abs_idx

                if use_bbox:
                    # Read only the bounding box region
                    y_min, y_max, x_min, x_max = self._bbox
                    sel[y_idx] = slice(y_min, y_max + 1)
                    sel[x_idx] = slice(x_min, x_max + 1)

                arr = read_netcdf_var_sliced(var, tuple(sel))

                arr = self._apply_value_policy(arr)

                # Normalize to (T, Y, X) - note: Y, X may be bbox dimensions if use_bbox
                arr = self._ensure_tyx(arr, t_idx, y_idx, x_idx)

                if compressed:
                    # Flatten and extract active columns: (T, Y, X) -> (T, N)
                    T, Y, X = arr.shape
                    flat = arr.reshape(T, Y * X)

                    out = flat[:, self._bbox_local_indices]
                else:
                    # Keep as (T, Y, X)
                    out = arr

                out = out.astype(self.out_dtype, copy=False)
                chunks.append(out)

        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def _get_first_frame_nan_mask(self) -> Optional[np.ndarray]:
        """Read the first planned source frame and return a flat NaN/mask bitmap."""
        if not self._plan:
            return None

        first_op = None
        for entry in self._plan:
            for key, abs_indices in entry[1]:
                if abs_indices:
                    first_op = (key, int(abs_indices[0]))
                    break
            if first_op is not None:
                break

        if first_op is None:
            return None

        key, abs_index = first_op
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            var = ds.variables[self.var_name]
            dims = var.dimensions
            t_idx = self._pick_dim(dims, "time", "valid_time")
            y_idx = self._pick_dim(dims, "lat", "latitude", "y")
            x_idx = self._pick_dim(dims, "lon", "longitude", "long", "x")
            if t_idx is None or y_idx is None or x_idx is None:
                raise ValueError(f"Expect at least time/lat/lon dims, got: {dims}")

            sel = [slice(None)] * var.ndim
            sel[t_idx] = np.asarray([abs_index], dtype=np.int32)
            arr = read_netcdf_var_sliced(var, tuple(sel))
            arr = self._as_nan_array(arr)
            arr = self._ensure_tyx(arr, t_idx, y_idx, x_idx)

        if arr.shape[0] != 1:
            raise ValueError(f"Expected one frame for source NaN mask, got shape={arr.shape}")
        if not np.issubdtype(arr.dtype, np.floating):
            return np.zeros(arr.shape[1] * arr.shape[2], dtype=bool)
        return np.isnan(arr[0]).reshape(-1)

    def _finish_read(self, data: np.ndarray) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.time_aggregation is None:
            return data / self.unit_factor
        if not self._aggregation_plan_enabled:
            raise ValueError(
                "NetCDFDataset must be initialized with time_aggregation before "
                "time-aggregated chunks can be read."
            )
        data = self._apply_time_aggregation(
            data,
            self.source_time_interval,
            self.time_aggregation,
        )
        if isinstance(data, dict):
            return {name: block / self.unit_factor for name, block in data.items()}
        return data / self.unit_factor

    def read_chunk(self, idx: int) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reads the chunk at the specified index using the pre-computed plan.
        """
        if idx < 0 or idx >= len(self._plan):
            raise IndexError(f"Chunk index {idx} out of range (0-{len(self._plan)-1})")

        entry = self._plan[idx]
        ops = entry[1]
        data = self._read_ops(ops)
        return self._finish_read(data)

    def close(self) -> None:
        """No persistent open handles are kept; provided for interface completeness."""

    def get_data(self, current_time: Union[datetime, cftime.datetime], chunk_len: int) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Read a contiguous block starting at current_time with minimal NetCDF I/O.

        Returns:
        - If _local_indices is set: (T, N) compressed array
        - If _local_indices is None: (T, Y, X) full grid array
        """
        try:
            start_abs = self.get_index_by_time(current_time)
        except ValueError as e:
            raise ValueError(f"Start time {current_time} not found in global timeline") from e

        end_abs = min(start_abs + int(chunk_len), len(self._global_times))
        times = self._global_times[start_abs:end_abs]
        entry = self._build_plan_entry(times)
        ops = entry[1]
        data = self._read_ops(ops)
        return self._finish_read(data)

    # -------------------------
    # Public API
    # -------------------------
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) 1D arrays from the first file."""
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            lat = ds.variables.get("lat") or ds.variables.get("latitude")
            lon = ds.variables.get("lon") or ds.variables.get("longitude") or ds.variables.get("long")
            if lat is None or lon is None:
                raise ValueError("Unable to find lat/lon variables in the dataset.")
            return np.array(lon[:]), np.array(lat[:])


def open_multivariable_netcdf(
    base_dir: str,
    var_specs: Dict[str, dict],
    *,
    start_date: Union[datetime, cftime.datetime],
    end_date: Union[datetime, cftime.datetime],
    time_interval: timedelta = timedelta(days=1),
    chunk_len: int = 24,
    unit_factor: float = 1.0,
    suffix: str = ".nc",
    clip_negative: bool = False,
    time_to_key: Optional[Callable[[Union[datetime, cftime.datetime]], str]] = yearly_time_to_key,
    spin_up_cycles: int = 0,
    spin_up_start_date: Optional[Union[datetime, cftime.datetime]] = None,
    spin_up_end_date: Optional[Union[datetime, cftime.datetime]] = None,
):
    """Open aligned gridded variables as one generic composite."""
    if not var_specs:
        raise ValueError("var_specs must contain at least one variable")
    shared = {
        "base_dir": base_dir, "start_date": start_date,
        "end_date": end_date, "time_interval": time_interval,
        "chunk_len": chunk_len, "unit_factor": unit_factor,
        "suffix": suffix, "clip_negative": clip_negative,
        "time_to_key": time_to_key, "spin_up_cycles": spin_up_cycles,
        "spin_up_start_date": spin_up_start_date,
        "spin_up_end_date": spin_up_end_date,
    }
    datasets = {}
    for name, spec in var_specs.items():
        options = shared | dict(spec)
        options["var_name"] = name
        options.setdefault("prefix", f"{name}_")
        datasets[name] = NetCDFDataset(**options)
    from hydroforge.data.datasets.multivariable import MultiVariableDataset

    return MultiVariableDataset(datasets, loader_strategy="parallel")

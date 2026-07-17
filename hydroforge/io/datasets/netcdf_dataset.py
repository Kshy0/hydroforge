# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import cftime
import numpy as np
import torch
from netCDF4 import Dataset, num2date

from hydroforge.io.datasets.abstract_dataset import AbstractDataset
from hydroforge.io.datasets.utils import read_netcdf_var_sliced, single_file_key, yearly_time_to_key


class NetCDFDataset(AbstractDataset):
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

    def _validate_files_exist(self, keys: Set[str]) -> None:
        file_paths = []
        for key in sorted(keys):
            file_paths.append(Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}")

        self.validate_files_exist(file_paths)

    def _scan_time_metadata(self, start_dt: Union[datetime, cftime.datetime], end_dt: Union[datetime, cftime.datetime]) -> None:
        """Read only time vars to construct a global time index and lookup map."""
        # Build key -> first_dt map to help with date guessing
        key_to_first_dt: Dict[str, Union[datetime, cftime.datetime]] = {}
        aggregate = self.time_aggregation is not None
        scan_end = end_dt + self.time_interval if aggregate else end_dt
        step = self.time_interval
        t = start_dt
        while (t < scan_end if aggregate else t <= scan_end):
            k = self.time_to_key(t)
            if k not in key_to_first_dt:
                key_to_first_dt[k] = t
            t += step
        if aggregate:
            k_end = self.time_to_key(scan_end)
            if k_end not in key_to_first_dt:
                key_to_first_dt[k_end] = scan_end
        # Ensure end_date is covered in non-aggregation mode.
        if not aggregate:
            k_end = self.time_to_key(end_dt)
            if k_end not in key_to_first_dt:
                key_to_first_dt[k_end] = end_dt

        candidate_key_to_first_dt = key_to_first_dt
        if aggregate:
            key_to_first_dt = {
                key: first_dt
                for key, first_dt in key_to_first_dt.items()
                if (Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}").exists()
            }
            if not key_to_first_dt:
                self._validate_files_exist(set(candidate_key_to_first_dt.keys()))
        keys = set(key_to_first_dt.keys())
        self._validate_files_exist(keys)

        source_times = []
        for key in sorted(keys):
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                tvar = ds.variables.get("time") or ds.variables.get("valid_time")
                if tvar is None:
                    raise ValueError(f"Time variable not found in file: {path.name}")

                # Auto-detect calendar if not provided by user
                file_calendar = getattr(tvar, "calendar", "standard")
                if self.calendar != file_calendar:
                    self.update_calendar(file_calendar)
                    # Also update local loop bounds to match the new calendar
                    start_dt = self._convert_to_calendar(start_dt)
                    end_dt = self._convert_to_calendar(end_dt)

                try:
                    raw_dates = num2date(tvar[:], tvar.units, file_calendar)
                    dates = list(raw_dates)
                except (ValueError, TypeError):
                    # Fallback for "days since start" or similar non-standard units
                    base = None
                    if key in key_to_first_dt:
                        sample_dt = key_to_first_dt[key]
                        # Try to snap to year start
                        dt_year = datetime(sample_dt.year, 1, 1)
                        if self.time_to_key(dt_year) == key:
                            base = dt_year
                        else:
                            # Try to snap to month start
                            dt_month = datetime(sample_dt.year, sample_dt.month, 1)
                            if self.time_to_key(dt_month) == key:
                                base = dt_month
                            else:
                                # Fallback to the sample date itself
                                base = sample_dt

                    if base is None:
                        try:
                            year = int(key)
                            base = datetime(year, 1, 1)
                        except ValueError:
                            pass

                    if base is not None:
                        dates = [base + timedelta(days=float(x)) for x in tvar[:]]
                    else:
                        raise ValueError(f"Cannot parse time for key '{key}' with units '{getattr(tvar, 'units', '')}'")

                self._file_times[key] = []
                for i, dt in enumerate(dates):
                    self._file_times[key].append(dt)
                    in_range = (
                        start_dt <= dt < scan_end
                        if aggregate
                        else start_dt <= dt <= end_dt
                    )
                    if in_range:
                        self._dt_to_loc[dt] = (key, i)
                        if aggregate:
                            source_times.append(dt)
        expected_times: List[datetime] = []
        t = start_dt
        while t <= end_dt:
            expected_times.append(t)
            t += self.time_interval

        if aggregate:
            self.source_time_interval = self._infer_source_time_interval(source_times)
            self._aggregation_factor = self._get_time_aggregation_factor(self.source_time_interval)
            self._validate_source_times_for_aggregation(expected_times)
            self._global_times = expected_times
            return

        missing = [dt for dt in expected_times if dt not in self._dt_to_loc]
        if missing:
            preview = ", ".join(str(m) for m in missing[:10])
            raise ValueError(
                f"Missing required timestamps for the chosen time_interval. "
                f"First missing: {preview} (total {len(missing)}). "
                f"Check start_date alignment and dataset temporal resolution."
            )
        self._global_times = expected_times

    def _infer_source_time_interval(
        self,
        source_times: List[Union[datetime, cftime.datetime]],
    ) -> timedelta:
        source_times = sorted(source_times)
        duplicates = [
            source_times[i]
            for i in range(1, len(source_times))
            if source_times[i] == source_times[i - 1]
        ]
        if duplicates:
            preview = ", ".join(str(dt) for dt in duplicates[:10])
            raise ValueError(
                "Duplicate source timestamps found in NetCDF time axis. "
                f"First duplicates: {preview} (total {len(duplicates)})."
            )
        diffs = [source_times[i + 1] - source_times[i] for i in range(len(source_times) - 1)]
        if not diffs:
            raise ValueError("Unable to infer source_time_interval from NetCDF time axis")
        source_interval = diffs[0]
        source_seconds = source_interval.total_seconds()
        if source_seconds <= 0:
            raise ValueError("source_time_interval inferred from NetCDF time axis must be positive")
        irregular = [
            (source_times[i], source_times[i + 1], diff)
            for i, diff in enumerate(diffs)
            if not np.isclose(
                diff.total_seconds(),
                source_seconds,
                rtol=0.0,
                atol=1e-9,
            )
        ]
        if irregular:
            preview = ", ".join(
                f"{left}->{right} ({diff})"
                for left, right, diff in irregular[:5]
            )
            raise ValueError(
                "NetCDF source time axis must be uniformly spaced for time "
                f"aggregation; inferred first interval {source_interval}, "
                f"but found irregular intervals: {preview} "
                f"(total {len(irregular)})."
            )
        return source_interval

    def _source_times_for_output_times(
        self,
        output_times: List[Union[datetime, cftime.datetime]],
    ) -> List[Union[datetime, cftime.datetime]]:
        return [
            dt + self.source_time_interval * offset
            for dt in output_times
            for offset in range(self._aggregation_factor)
        ]

    def _validate_source_times_for_aggregation(
        self,
        output_times: List[Union[datetime, cftime.datetime]],
    ) -> None:
        required = self._source_times_for_output_times(output_times)
        missing = [dt for dt in required if dt not in self._dt_to_loc]
        if missing:
            preview = ", ".join(str(m) for m in missing[:10])
            raise ValueError(
                f"Missing required source timestamps for time aggregation. "
                f"First missing: {preview} (total {len(missing)})."
            )

    def _ops_from_times(self, times: List[Union[datetime, cftime.datetime]]) -> List[Tuple[str, List[int]]]:
        """Group requested datetimes into per-file absolute index ops.

        Output format: List of (file_key, abs_indices), where abs_indices are
        absolute time indices in that file to fetch, in the requested order.
        """
        if not times:
            return []

        # Preserve file order as first encountered in the times list
        file_order: List[str] = []
        file_to_indices: Dict[str, List[int]] = {}

        for dt in times:
            key, idx = self._dt_to_loc[dt]
            if key not in file_to_indices:
                file_to_indices[key] = []
                file_order.append(key)
            file_to_indices[key].append(idx)

        ops: List[Tuple[str, List[int]]] = []
        for key in file_order:
            idxs = file_to_indices[key]
            # Keep the order user requested; deduplicate while preserving order
            seen = set()
            uniq = [i for i in idxs if (i not in seen and not seen.add(i))]
            ops.append((key, uniq))

        return ops

    def _build_plan_entry(self, chunk_times: List[Union[datetime, cftime.datetime]]):
        if self.time_aggregation is None:
            return (chunk_times[0], self._ops_from_times(chunk_times))
        source_times = self._source_times_for_output_times(chunk_times)
        return (chunk_times[0], self._ops_from_times(source_times), len(chunk_times))

    def _build_simulation_plan(self) -> None:
        """
        Builds the sequence of chunks for the entire simulation, including spin-up.
        self._plan will be a list of (start_time, ops).
        """
        self._plan = []

        # Helper to build chunks for a time range
        def build_chunks_for_range(start_dt, end_dt):
            chunks = []
            times = []
            t = start_dt
            while t <= end_dt:
                times.append(t)
                t += self.time_interval

            total = len(times)
            if total == 0:
                return []

            n_chunks = (total + self.chunk_len - 1) // self.chunk_len
            for ci in range(n_chunks):
                a = ci * self.chunk_len
                b = min(a + self.chunk_len, total)
                chunk_times = times[a:b]
                # Store the start time of the chunk for reference
                chunks.append(self._build_plan_entry(chunk_times))
            return chunks

        # 1. Spin-up chunks
        self._spin_up_chunks_template = []
        if self.spin_up_cycles > 0:
            if self.spin_up_start_date is None or self.spin_up_end_date is None:
                raise ValueError("Spin-up dates must be provided if spin_up_cycles > 0")
            self._spin_up_chunks_template = build_chunks_for_range(self.spin_up_start_date, self.spin_up_end_date)

            for _ in range(self.spin_up_cycles):
                self._plan.extend(self._spin_up_chunks_template)

        # 2. Main simulation chunks
        main_chunks = build_chunks_for_range(self.start_date, self.end_date)
        self._plan.extend(main_chunks)

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
        self.source_time_interval = None
        self._aggregation_factor = None

        # Runtime metadata
        self._file_times = {}
        self._global_times = []
        self._dt_to_loc = {}
        # Each chunk plan is a list of (file_key, abs_time_indices) operations.
        # Sequence indices are read as NetCDF slices and reordered in memory.
        self._chunk_plan = []

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

        # Determine full data range needed
        scan_start = self.start_date
        scan_end = self.end_date
        if self.spin_up_cycles > 0:
            if self.spin_up_start_date is not None and self.spin_up_start_date < scan_start:
                scan_start = self.spin_up_start_date
            if self.spin_up_end_date is not None and self.spin_up_end_date > scan_end:
                scan_end = self.spin_up_end_date

        self._scan_time_metadata(scan_start, scan_end)
        self._build_simulation_plan()
        self._aggregation_plan_enabled = self.time_aggregation is not None

    def is_valid_time_index(self, idx: int) -> bool:
        """
        Checks if the given time index corresponds to a valid data step (not padding).
        """
        chunk_idx = idx // self.chunk_len
        offset = idx % self.chunk_len

        if chunk_idx >= len(self._plan):
             return False

        entry = self._plan[chunk_idx]
        if self.time_aggregation is None:
            _, ops = entry
            # Calculate real length of this chunk
            real_len = sum(len(x[1]) for x in ops)
        else:
            real_len = entry[2]

        return offset < real_len

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

                    if use_bbox:
                        # Use bbox-relative indices
                        out = flat[:, self._bbox_local_indices]
                    else:
                        # Use global indices (fallback, should not happen normally)
                        out = flat[:, self._local_indices]
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

    def _collect_required_keys(self) -> Set[str]:
        """Collect file keys covering [start_date, end_date] stepping by time_interval."""
        keys: Set[str] = set()
        t = self.start_date
        # + one extra step to ensure inclusive end coverage for non-divisible ranges
        while t <= self.end_date:
            keys.add(self.time_to_key(t))
            t += self.time_interval
        keys.add(self.time_to_key(self.end_date))
        return keys

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


class MultiVarNetCDFDataset(AbstractDataset):
    """Gridded multi-variable composite of :class:`NetCDFDataset`.

    Wraps one :class:`NetCDFDataset` per variable (one NetCDF file per
    variable) when every variable lives on the same grid. The spatial
    ``build_local_mapping`` is computed once from the first variable and the
    resulting compressed-grid selection is reused for the rest, so only the
    first file's mapping is ever read.

    ``__getitem__`` and :meth:`shard_forcing` emit a dict keyed by variable
    name; ``total_steps`` / ``time_iter`` are forwarded to the reference
    (first) dataset.

    Parameters
    ----------
    base_dir:
        Directory holding the per-variable NetCDF files.
    var_specs:
        Mapping ``{var_name: {"prefix": str, ...}}``. Per-variable kwargs
        override the shared kwargs; ``prefix`` defaults to ``f"{var_name}_"``.
    """

    def __init__(
        self,
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
    ) -> None:
        if not var_specs:
            raise ValueError("var_specs must contain at least one variable.")

        shared_kwargs = dict(
            base_dir=base_dir,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            chunk_len=chunk_len,
            unit_factor=unit_factor,
            suffix=suffix,
            clip_negative=clip_negative,
            time_to_key=time_to_key,
            spin_up_cycles=spin_up_cycles,
            spin_up_start_date=spin_up_start_date,
            spin_up_end_date=spin_up_end_date,
        )

        self._var_names: List[str] = list(var_specs.keys())
        self._datasets: List[NetCDFDataset] = []
        for var_name, spec in var_specs.items():
            kw = dict(shared_kwargs)
            kw.update(spec)
            kw["var_name"] = var_name
            kw.setdefault("prefix", f"{var_name}_")
            self._datasets.append(NetCDFDataset(**kw))

        ref = self._datasets[0]
        for ds, name in zip(
            self._datasets[1:], self._var_names[1:], strict=True,
        ):
            if len(ds) != len(ref):
                raise ValueError(
                    f"Length mismatch: {self._var_names[0]!r}={len(ref)} vs {name!r}={len(ds)}"
                )

        super().__init__(
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            chunk_len=int(ref.chunk_len),
            clip_negative=clip_negative,
            spin_up_cycles=spin_up_cycles,
            spin_up_start_date=spin_up_start_date,
            spin_up_end_date=spin_up_end_date,
        )

    @property
    def variables(self) -> List[str]:
        return list(self._var_names)

    @property
    def datasets(self) -> List[NetCDFDataset]:
        return list(self._datasets)

    @property
    def data_size(self) -> int:
        return self._datasets[0].data_size

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self._datasets[0].grid_shape

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._datasets[0].get_coordinates()

    def __len__(self) -> int:
        return len(self._datasets[0])

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            v: ds[idx]
            for v, ds in zip(self._var_names, self._datasets, strict=True)
        }

    def get_data(self, current_time, chunk_len):
        return {
            v: ds.get_data(current_time, chunk_len)
            for v, ds in zip(self._var_names, self._datasets, strict=True)
        }

    def build_local_mapping(
        self,
        mapping_file: str,
        desired_catchment_ids: Optional[np.ndarray] = None,
        device=None,
        precision: str = "float32",
    ):
        """Build the mapping once from the first variable and share it.

        All variables are assumed to lie on the same grid, so the compressed
        grid selection (``_local_indices``) resolved for the reference dataset
        is propagated to the rest. Returns the single shared local mapping; the
        caller keeps it as a local variable and passes it to
        :meth:`shard_forcing` (the dataset itself stays free of device tensors
        so it remains safe to fork across ``DataLoader`` workers).
        """
        ref = self._datasets[0]
        local_mapping = ref.build_local_mapping(
            mapping_file=mapping_file,
            desired_catchment_ids=desired_catchment_ids,
            device=device,
            precision=precision,
        )
        for ds in self._datasets[1:]:
            ds._local_indices = ref._local_indices
            ds._desired_catchment_ids = ref._desired_catchment_ids
        return local_mapping

    def shard_forcing(self, batch: Dict[str, torch.Tensor], local_mapping) -> Dict[str, torch.Tensor]:
        return {
            v: ds.shard_forcing(batch[v], local_mapping)
            for v, ds in zip(self._var_names, self._datasets, strict=True)
        }

    def iter_loaders(
        self,
        *,
        loader_workers: int = 1,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
    ):
        """Stream every variable through its own prefetching ``DataLoader``.

        Each per-variable NetCDF file gets a dedicated :class:`DataLoader` so
        the files are read concurrently. Following the CaMa-Flood-GPU idiom,
        ``batch_size`` equals ``loader_workers`` so each worker prefetches one
        chunk per batch. The single-variable loaders are advanced in lock-step
        with :func:`zip`, yielding ``{var: batch}`` where every batch shares the
        same simulation plan.
        """
        from torch.utils.data import DataLoader

        batch_size = max(1, loader_workers)
        loaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=loader_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if loader_workers > 0 else None,
            )
            for ds in self._datasets
        ]
        for batches in zip(*loaders, strict=True):
            yield {
                v: b
                for v, b in zip(self._var_names, batches, strict=True)
            }

    @property
    def total_steps(self) -> int:
        return self._datasets[0].total_steps

    def time_iter(self):
        return self._datasets[0].time_iter()

    def close(self) -> None:
        for ds in self._datasets:
            ds.close()

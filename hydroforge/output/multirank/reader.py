# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import cftime
import netCDF4 as nc
import numpy as np

from hydroforge.output.multirank.plotter import MultiRankPlotter
from hydroforge.output.multirank.catalog import RankOutputCatalog
from hydroforge.output.multirank.data import MultiRankDataAccess
from hydroforge.serialization.files import atomic_output_path

logger = logging.getLogger(__name__)


class MultiRankStatsReader:
    """
    Manage per‑rank NetCDF outputs written by a StatisticsRuntime-like component.

    Major Features:
      - Auto-detect rank files: {var_name}_rank{rank}.nc
      - Derive (x, y) locations for saved_points using one (mutually exclusive) method:
          * coord_source=(nx, ny) tuple               -> treat coord_raw as linear indices
          * coord_source=NetCDF file path             -> extract map shape (nx, ny)
          * coord_source=callable(coord_raw)->(x,y)   -> custom conversion
      - Provide vector / grid / time series extraction APIs
      - Basic visualization (single time slice + animation)
      - Export time-sliced grids to CaMa-Flood-compatible Fortran-order binary
    """

    @cached_property
    def _plotter(self) -> MultiRankPlotter:
        return MultiRankPlotter(self)

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------
    def _safe_time_str(self, t_obj, fmt="%Y-%m-%d %H:%M:%S") -> str:
        """Helper to safely format time objects (datetime, cftime, or others)."""
        # Try strftime first (works for datetime and modern cftime)
        if hasattr(t_obj, "strftime"):
             try:
                 return t_obj.strftime(fmt)
             except (TypeError, ValueError, OverflowError):
                 pass

        # Fallback to isoformat
        if hasattr(t_obj, "isoformat"):
             try:
                 return t_obj.isoformat()
             except (TypeError, ValueError, OverflowError):
                 pass

        # Fallback to string
        return str(t_obj)

    def _preload_cache(self) -> None:
        """Preload only the chosen inclusive slice [self._slice_start, self._slice_end]."""
        if self._slice_start is None or self._slice_end is None:
            raise RuntimeError("Slice indices not set.")

        # We need to map global slice to file-specific slices
        # self._file_time_offsets contains [(0, 366), (366, 731), ...]

        for info in self._rank_files:
            if info["saved_points"] == 0:
                info["cache"] = None
                continue

            rank_data_parts = []

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

                    try:
                        with nc.Dataset(fp, "r") as ds:
                            var = ds.variables[self.var_name]
                            # Slicing logic: always take all spatial/trial dims.
                            # Dimensions are (time, [trial], saved_points, [levels]).
                            if self.row_chunk_size is None:
                                data = var[local_start:local_end, ...]
                                rank_data_parts.append(
                                    self._data_access._array(
                                        data, source=fp.name,
                                    ).copy()
                                )
                            else:
                                for t0 in range(local_start, local_end, self.row_chunk_size):
                                    t1 = min(t0 + self.row_chunk_size, local_end)
                                    data = var[t0:t1, ...]
                                    rank_data_parts.append(
                                        self._data_access._array(
                                            data, source=fp.name,
                                        ).copy()
                                    )
                    except (OSError, KeyError, IndexError, ValueError) as exc:
                        raise RuntimeError(f"Failed to cache {fp}") from exc

            if rank_data_parts:
                info["cache"] = np.concatenate(rank_data_parts, axis=0)
            else:
                info["cache"] = None

    # ----------------------------------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        base_dir: Union[str, Path],
        var_name: str,
        coord_name: Optional[str] = None,
        coord_source: Optional[
            Union[
                Tuple[int, int],
                str,
                Path,
                Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
            ]
        ] = None,
        time_range: Optional[Tuple[Union[datetime, cftime.datetime], Union[datetime, cftime.datetime]]] = None,
        cache_enabled: bool = False,
        split_by_year: bool = False,
        row_chunk_size: Optional[int] = None,
    ):
        """
        time_range: CLOSED interval (start_dt, end_dt), both inclusive.
        """
        self.base_dir = Path(base_dir)
        self.var_name = var_name
        self.coord_name = coord_name

        self._map_shape: Optional[Tuple[int, int]] = None
        self._coord_converter: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None

        self._rank_files: List[dict] = []
        self._time_units: str | None = None
        self._time_calendar: str | None = None
        self._time_values_num: np.ndarray | None = None
        if type(cache_enabled) is not bool or type(split_by_year) is not bool:
            raise TypeError("cache_enabled and split_by_year must be exact bools")
        if row_chunk_size is not None and (
            type(row_chunk_size) is not int or row_chunk_size < 1
        ):
            raise ValueError("row_chunk_size must be an exact positive int")
        self.cache_enabled = cache_enabled
        self.split_by_year = split_by_year
        self.row_chunk_size = row_chunk_size

        self._slice_start: Optional[int] = None
        self._slice_end: Optional[int] = None
        self._t_indices: Optional[np.ndarray] = None

        # Interpret coord_source
        if coord_source is not None:
            if callable(coord_source):
                self._coord_converter = coord_source
            elif isinstance(coord_source, (str, Path)):
                self.load_map_shape_from_nc(coord_source)
            else:
                self.set_map_shape(coord_source)  # type: ignore[arg-type]

        self._catalog = RankOutputCatalog(self)
        self._data_access = MultiRankDataAccess(self)
        self._rank_files = self._catalog.scan()
        if not self._rank_files:
            raise FileNotFoundError(
                f"No files found in {self.base_dir} matching: {self.var_name}_rank*.nc"
            )

        self._catalog.read_timeline()

        # Apply closed datetime slice with strict range checking (no clamping)
        if time_range is not None:
            # Strategy: Convert input range to numeric values using the NetCDF unit/calendar.
            start_in, end_in = time_range

            t_start_val = nc.date2num(
                start_in, self._time_units, self._time_calendar,
            )
            t_end_val = nc.date2num(
                end_in, self._time_units, self._time_calendar,
            )
            if t_start_val > t_end_val:
                raise ValueError("time_range start must be <= end (closed interval).")

            file_min = self._time_values_num[0]
            file_max = self._time_values_num[-1]
            if t_start_val < file_min or t_end_val > file_max:
                raise ValueError(
                    "time_range outside available coverage. "
                    f"Requested [{self._safe_time_str(start_in)} .. "
                    f"{self._safe_time_str(end_in)}] but coverage is "
                    f"[{self._safe_time_str(self._time_datetimes[0])} .. "
                    f"{self._safe_time_str(self._time_datetimes[-1])}]."
                )

            valid_mask = (
                (self._time_values_num >= t_start_val)
                & (self._time_values_num <= t_end_val)
            )
            indices = np.flatnonzero(valid_mask)
            if indices.size == 0:
                raise ValueError("No time steps found in the request range.")
            left = int(indices[0])
            right = int(indices[-1])

            self._slice_start = left
            self._slice_end = right
            self._t_indices = np.arange(left, right + 1, dtype=np.int64)

            self._time_values_num = self._time_values_num[self._t_indices]
            self._time_datetimes = [self._time_datetimes[i] for i in self._t_indices]
            self._time_len = len(self._t_indices)
        else:
            self._slice_start = 0
            self._slice_end = self._time_len - 1
            self._t_indices = np.arange(self._time_len, dtype=np.int64)

        self._catalog.compute_coordinates(force=True)

        if self.cache_enabled:
            self._preload_cache()

    # ----------------------------------------------------------------------------------
    # Data getters
    # ----------------------------------------------------------------------------------
    def get_vector(
        self, t_index: int, level: Optional[int] = None, trial: int = 0,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        return self._data_access.get_vector(t_index, level, trial, dtype)

    def get_grid(
        self, t_index: int, level: Optional[int] = None, trial: int = 0,
        fill_value: float = np.nan, dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        return self._data_access.get_grid(
            t_index, level, trial, fill_value, dtype,
        )

    def get_series(
        self, points: Union[np.ndarray, Sequence[np.ndarray]],
        level: Optional[int] = None, trial: int = 0,
        fill_value: float = np.nan, dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        return self._data_access.get_series(
            points, level, trial, fill_value, dtype,
        )

    def plot_single_time(
        self,
        t_index: int = 0,
        level: Optional[int] = None,
        trial: int = 0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
        as_scatter_if_no_map: bool = True,
        s: float = 1.0,
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        self._plotter.plot_single_time(
            t_index=t_index, level=level, trial=trial, vmin=vmin, vmax=vmax,
            cmap=cmap, figsize=figsize,
            as_scatter_if_no_map=as_scatter_if_no_map, s=s,
            auto_crop=auto_crop, crop_pad=crop_pad,
        )

    def animate(
        self,
        out_path: Union[str, Path],
        level: Optional[int] = None,
        trial: int = 0,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
        fps: int = 10,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        self._plotter.animate(
            out_path=out_path, level=level, trial=trial,
            x_range=x_range, y_range=y_range, t_range=t_range, fps=fps,
            vmin=vmin, vmax=vmax, cmap=cmap, figsize=figsize,
            auto_crop=auto_crop, crop_pad=crop_pad,
        )

    def plot_series(
        self,
        points,
        level: Optional[int] = None,
        trial=0,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        ax=None,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        return self._plotter.plot_series(
            points=points, level=level, trial=trial, figsize=figsize,
            title=title, ax=ax, labels=labels, **kwargs,
        )

    @staticmethod
    def discover_k_variants(base_dir: Union[str, Path], base_var_name: str) -> List[str]:
        """
        Discover all k-indexed variants of a variable (e.g., for maxK operations).

        For a variable like 'river_depth_max3', this will find:
        - river_depth_max3_0
        - river_depth_max3_1
        - river_depth_max3_2

        Args:
            base_dir: Directory containing the NetCDF files
            base_var_name: Base variable name (e.g., 'river_depth_max3')

        Returns:
            List of variant names sorted by k index, or [base_var_name] if no k variants found
        """
        base_dir = Path(base_dir)
        # Pattern to match k-indexed files: {base_var_name}_{k}_rank*.nc
        pattern = f"{base_var_name}_*_rank*.nc"
        files = list(base_dir.glob(pattern))

        # Extract unique k indices
        k_pattern = re.compile(rf"^{re.escape(base_var_name)}_(\d+)_rank\d+.*\.nc$")
        k_indices = set()
        for f in files:
            m = k_pattern.match(f.name)
            if m:
                k_indices.add(int(m.group(1)))

        if k_indices:
            # Return sorted list of variant names
            return [f"{base_var_name}_{k}" for k in sorted(k_indices)]
        else:
            # No k variants found, check if base file exists
            base_pattern = f"{base_var_name}_rank*.nc"
            if list(base_dir.glob(base_pattern)):
                return [base_var_name]
            return []

    @staticmethod
    def list_available_variables(base_dir: Union[str, Path]) -> List[str]:
        """
        List all unique variable names available in the directory.

        This scans for files matching *_rank*.nc and extracts variable names.

        Args:
            base_dir: Directory containing the NetCDF files

        Returns:
            Sorted list of unique variable names
        """
        base_dir = Path(base_dir)
        files = list(base_dir.glob("*_rank*.nc"))

        # Pattern to extract variable name: {var_name}_rank{rank}[_{year}].nc
        var_pattern = re.compile(r"^(.+)_rank\d+(?:_\d{4})?\.nc$")
        var_names = set()
        for f in files:
            m = var_pattern.match(f.name)
            if m:
                var_names.add(m.group(1))

        return sorted(var_names)

    @property
    def num_ranks(self) -> int:
        return len(self._rank_files)

    @property
    def time_len(self) -> int:
        return self._time_len

    @property
    def times(self) -> List[datetime]:
        return self._time_datetimes

    @property
    def map_shape(self) -> Optional[Tuple[int, int]]:
        return self._map_shape

    def set_map_shape(self, map_shape: Tuple[int, int]) -> None:
        if type(map_shape) is not tuple or len(map_shape) != 2:
            raise ValueError("map_shape must be an exact (nx, ny) tuple")
        if any(type(value) is not int or value < 1 for value in map_shape):
            raise ValueError("map_shape values must be exact positive ints")
        self._map_shape = map_shape
        if self._rank_files:
            self._catalog.compute_coordinates(force=True)

    def load_map_shape_from_nc(
        self,
        nc_path: Union[str, Path],
    ) -> None:
        p = Path(nc_path)
        if not p.exists():
            raise FileNotFoundError(f"NetCDF file not found: {p}")

        nx = ny = None
        with nc.Dataset(p, "r") as ds:
            attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
            if "nx" in attrs and "ny" in attrs:
                nx = int(attrs["nx"])
                ny = int(attrs["ny"])
            if (nx is None or ny is None) and "nx" in ds.variables and "ny" in ds.variables:
                nx = int(np.array(ds.variables["nx"][:]).squeeze())
                ny = int(np.array(ds.variables["ny"][:]).squeeze())
            if (nx is None or ny is None) and "map_shape" in ds.variables:
                arr = np.array(ds.variables["map_shape"][:]).squeeze()
                if arr.size >= 2:
                    nx = int(arr[0])
                    ny = int(arr[1])
            if (nx is None or ny is None) and "map_shape" in attrs:
                arr = np.array(attrs["map_shape"]).squeeze()
                if np.size(arr) >= 2:
                    flat = np.ravel(arr)
                    nx = int(flat[0])
                    ny = int(flat[1])
            if nx is None or ny is None:
                dim_pairs = [("nx", "ny"), ("x", "y"), ("lon", "lat")]
                for a, b in dim_pairs:
                    if a in ds.dimensions and b in ds.dimensions:
                        nx = int(ds.dimensions[a].size)
                        ny = int(ds.dimensions[b].size)
                        break
        if nx is None or ny is None:
            raise KeyError("Could not find nx/ny or map_shape (attrs/vars/dims).")
        self.set_map_shape((nx, ny))

    def set_coord_converter(
        self,
        converter: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        self._coord_converter = converter
        self._catalog.compute_coordinates(force=True)

    # ----------------------------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------------------------
    def export_to_cama_bin(
        self,
        out_dir: Union[str, Path],
        out_var_name: str,
        t_range: Optional[Tuple[int, int]] = None,
        trial: int = 0,
        fill_value: float = 1e20,
        dtype: np.dtype = np.float32,
        progress: bool = True,
    ) -> None:
        if self._map_shape is None:
            raise RuntimeError("map_shape is required to export .bin files.")
        if any(info["has_levels"] for info in self._rank_files):
            raise ValueError("Variables with 'levels' not supported for export.")
        if not self.times or self._time_len == 0:
            raise RuntimeError("No time axis available for export.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t_start = 0 if t_range is None else max(0, int(t_range[0]))
        t_end = self._time_len if t_range is None else min(self._time_len, int(t_range[1]))
        if t_start >= t_end:
            raise ValueError("Invalid t_range: ensure t_start < t_end")

        year_to_indices: dict[int, List[int]] = {}
        for ti in range(t_start, t_end):
            year = int(self.times[ti].year)
            year_to_indices.setdefault(year, []).append(ti)

        for year in sorted(year_to_indices.keys()):
            year_path = out_dir / f"{out_var_name}{year}.bin"
            if progress:
                print(f"[BIN] writing year {year} -> {year_path.name} ({len(year_to_indices[year])} frames)")
            with atomic_output_path(year_path) as temporary:
                with temporary.open("wb") as fw:
                    for ti in year_to_indices[year]:
                        grid = self.get_grid(
                            ti, level=None, trial=trial,
                            fill_value=fill_value, dtype=dtype,
                        )
                        grid = np.where(
                            np.isfinite(grid), grid, fill_value,
                        ).astype(dtype, copy=False)
                        fw.write(np.asfortranarray(grid).tobytes(order="F"))

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

    def summary(self) -> str:
        slice_info = f"[{self._slice_start} .. {self._slice_end}] (inclusive)" if self._slice_start is not None else "N/A"
        lines = [
            f"Variable         : {self.var_name}",
            f"Base dir         : {self.base_dir}",
            f"Ranks            : {self.num_ranks}",
            f"Local time len   : {self.time_len}",
            f"Time slice idx   : {slice_info}",
            f"First time (loc) : {self._time_datetimes[0] if self._time_datetimes else 'N/A'}",
            f"Last  time (loc) : {self._time_datetimes[-1] if self._time_datetimes else 'N/A'}",
            f"Map shape        : {self._map_shape}",
            f"Coord converter  : {'yes' if self._coord_converter is not None else 'no'}",
        ]
        for i, info in enumerate(self._rank_files):
            lines.append(
                f"  - rank[{i}]: file={info['path'].name}, saved_points={info['saved_points']}, "
                f"trials={'yes (' + str(info['n_trials']) + ')' if info['has_trials'] else 'no'}, "
                f"levels={'yes (' + str(info['n_levels']) + ')' if info['has_levels'] else 'no'}, "
                f"coord={info['coord_name'] or 'N/A'}"
            )
        return "\n".join(lines)

"""Vector, grid, and time-series reads over a rank-output catalog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)


class MultiRankDataAccess:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def _result_dtype(self, dtype: Optional[np.dtype]) -> np.dtype:
        if dtype is not None:
            result = np.dtype(dtype)
            if result.kind not in "biufc":
                raise TypeError("reader dtype must be numeric or boolean")
            return result
        return np.dtype(self.owner._rank_files[0]["dtype"])

    @staticmethod
    def _array(value: Any, *, source: str) -> np.ndarray:
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"statistics data from {source} contains missing values")
        return np.asarray(value)

    def _validate_axes(
        self, info: dict, *, level: Optional[int], trial: int,
    ) -> None:
        if type(trial) is not int:
            raise TypeError("trial must be an exact int")
        if info["has_trials"]:
            if not 0 <= trial < info["n_trials"]:
                raise IndexError(
                    f"trial out of range [0, {info['n_trials'] - 1}]"
                )
        elif trial != 0:
            raise ValueError("trial must be 0 for an output without a trial axis")
        if info["has_levels"]:
            if type(level) is not int:
                raise TypeError(
                    "level must be an exact int for an output with levels"
                )
            if not 0 <= level < info["n_levels"]:
                raise IndexError(
                    f"level out of range [0, {info['n_levels'] - 1}]"
                )
        elif level is not None:
            raise ValueError("level must be None for an output without levels")

    def _validate_request(
        self, *, t_index: int | None, level: Optional[int], trial: int,
    ) -> None:
        if t_index is not None and type(t_index) is not int:
            raise TypeError("t_index must be an exact int")
        for info in self.owner._rank_files:
            self._validate_axes(info, level=level, trial=trial)

    def _get_data_from_files(self, info: dict, t_index: int, level: Optional[int] = None, trial: int = 0) -> np.ndarray:
        """Helper to fetch data for a single time step from the correct file."""
        orig_time = int(self.owner._t_indices[t_index])

        # Find which file contains orig_time
        for i, (start, end) in enumerate(info["file_time_offsets"]):
            if start <= orig_time < end:
                local_time = orig_time - start
                fp = info["paths"][i]
                with nc.Dataset(fp, "r") as ds:
                    var = ds.variables[self.owner.var_name]

                    # Build index tuple
                    # 1. time
                    indices = [local_time]

                    # 2. trial
                    if info["has_trials"]:
                        indices.append(trial)

                    # 3. saved_points (all)
                    indices.append(slice(None))

                    # 4. levels
                    if info["has_levels"]:
                        indices.append(level if level is not None else 0)

                    return var[tuple(indices)]

        # Should not happen if t_index is valid
        raise IndexError(f"Time index {orig_time} not found in any file.")

    def get_vector(
        self,
        t_index: int,
        level: Optional[int] = None,
        trial: int = 0,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        self._validate_request(t_index=t_index, level=level, trial=trial)
        if t_index < 0 or t_index >= self.owner._time_len:
            raise IndexError(f"t_index out of range [0, {self.owner._time_len - 1}]")

        target_dtype = self._result_dtype(dtype)
        parts: List[np.ndarray] = []
        for info in self.owner._rank_files:
            if info["saved_points"] == 0:
                parts.append(np.empty((0,), dtype=target_dtype))
                continue

            cache_arr = info.get("cache")
            if cache_arr is not None:
                # cache_arr shape: (time, [trial], saved_points, [levels])
                indices = [t_index]
                if info["has_trials"]:
                    indices.append(trial)
                indices.append(slice(None))
                if info["has_levels"]:
                    indices.append(level)

                data = cache_arr[tuple(indices)]
            else:
                data = self._get_data_from_files(info, t_index, level, trial)

            arr = self._array(data, source=info["path"].name)
            arr = arr.astype(target_dtype, copy=False)
            parts.append(arr)
        return np.concatenate(parts, axis=0) if parts else np.array([])

    def get_grid(
        self,
        t_index: int,
        level: Optional[int] = None,
        trial: int = 0,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        self._validate_request(t_index=t_index, level=level, trial=trial)
        if self.owner._map_shape is None:
            raise RuntimeError("map_shape is not set; cannot project to grid.")
        if t_index < 0 or t_index >= self.owner._time_len:
            raise IndexError(f"t_index out of range [0, {self.owner._time_len - 1}]")

        nx_, ny_ = self.owner._map_shape
        target_dtype = self._result_dtype(dtype)
        try:
            grid = np.full((nx_, ny_), fill_value, dtype=target_dtype)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"fill_value {fill_value!r} cannot be represented by reader "
                f"dtype {target_dtype}"
            ) from error

        for info in self.owner._rank_files:
            if info["saved_points"] == 0:
                continue
            x = info.get("x")
            y = info.get("y")
            if x is None or y is None:
                raise RuntimeError(f"{info['path'].name} missing (x,y); set map_shape or coord converter.")

            cache_arr = info.get("cache")
            if cache_arr is not None:
                # cache_arr shape: (time, [trial], saved_points, [levels])
                indices = [t_index]
                if info["has_trials"]:
                    indices.append(trial)
                indices.append(slice(None))
                if info["has_levels"]:
                    indices.append(level)

                vals = cache_arr[tuple(indices)]
            else:
                vals = self._get_data_from_files(info, t_index, level, trial)

            grid[x, y] = self._array(
                vals, source=info["path"].name,
            ).astype(target_dtype, copy=False)
        return grid

    @staticmethod
    def _sorted_series_indices(pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        out_cols = np.array([col for col, _ in pairs], dtype=np.int64)
        local_idx = np.array([li for _, li in pairs], dtype=np.int64)
        order = np.argsort(local_idx, kind="stable")
        return out_cols[order], local_idx[order]

    def _copy_series_from_cache(
        self,
        out: np.ndarray,
        cache_arr: np.ndarray,
        pairs: List[Tuple[int, int]],
        level: Optional[int],
        trial: int,
        target_dtype: np.dtype,
        info: dict,
    ) -> None:
        out_cols, local_idx = self._sorted_series_indices(pairs)
        indices = [slice(None)]
        if info["has_trials"]:
            indices.append(trial)
        indices.append(local_idx)
        if info["has_levels"]:
            indices.append(level)
        chunk = self._array(
            cache_arr[tuple(indices)], source=info["path"].name,
        )
        out[:, out_cols] = chunk.astype(target_dtype, copy=False)

    def _copy_series_from_row_chunks(
        self,
        out: np.ndarray,
        fp: Path,
        info: dict,
        pairs: List[Tuple[int, int]],
        local_start: int,
        local_end: int,
        out_start: int,
        level: Optional[int],
        trial: int,
        target_dtype: np.dtype,
    ) -> None:
        out_cols, local_idx = self._sorted_series_indices(pairs)
        with nc.Dataset(fp, "r") as ds:
            var = ds.variables[self.owner.var_name]
            step = (local_end - local_start
                    if self.owner.row_chunk_size is None else self.owner.row_chunk_size)
            for t0 in range(local_start, local_end, step):
                t1 = min(t0 + step, local_end)
                slices = [slice(t0, t1)]
                if info["has_trials"]:
                    slices.append(trial)
                slices.append(slice(None))
                if info["has_levels"]:
                    slices.append(level)
                block = self._array(var[tuple(slices)], source=fp.name)
                selected = block[:, local_idx]
                o0 = out_start + (t0 - local_start)
                o1 = o0 + (t1 - t0)
                out[o0:o1, out_cols] = selected.astype(target_dtype, copy=False)

    def get_series(
        self,
        points: Union[np.ndarray, Sequence[np.ndarray]],
        level: Optional[int] = None,
        trial: int = 0,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        self._validate_request(t_index=None, level=level, trial=trial)
        target_dtype = self._result_dtype(dtype)

        def _as_list(v):
            # Heuristic: if input is a list/tuple that looks like (N, 2) coordinates, treat as single array
            if isinstance(v, (list, tuple)):
                arr = np.asarray(v)
                # If it forms a valid (N, 2) array, wrap it as a single item.
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return [arr]
            return [np.asarray(a) for a in v] if isinstance(v, (list, tuple)) else [np.asarray(v)]
        arr_list = _as_list(points)
        if not arr_list:
            return np.empty((self.owner._time_len, 0), dtype=target_dtype)

        def _kind(a: np.ndarray) -> str:
            if a.ndim == 2 and a.shape[1] == 2:
                return "xy"
            if a.ndim == 1 or a.ndim == 0:
                return "id"
            raise ValueError(f"Unsupported points shape: {a.shape}")

        kinds = {_kind(a) for a in arr_list}
        if len(kinds) != 1:
            raise ValueError("Provide either all XY (N,2) or all IDs (N,). Do not mix.")
        use_xy = kinds.pop() == "xy"

        for array in arr_list:
            if array.dtype.kind not in "iu" or array.dtype.kind == "b":
                raise TypeError("point IDs and XY coordinates must be integers")

        if use_xy:
            queries = [
                (int(px), int(py))
                for array in arr_list for px, py in np.asarray(array)
            ]
        else:
            queries = [
                int(value) for array in arr_list
                for value in np.asarray(array).ravel()
            ]

        N = len(queries)
        if len(set(queries)) != N:
            raise ValueError("Duplicate points not allowed.")
        col_to_hits: List[Optional[Tuple[int, int]]] = [None] * N

        # Map queries to (rank_idx, local_index) and check all found
        if use_xy:
            for r_idx, info in enumerate(self.owner._rank_files):
                if info["saved_points"] == 0:
                    continue
                x, y = info.get("x"), info.get("y")
                if x is None or y is None:
                    continue

                # Build lookup map for this rank: (x, y) -> local_index
                if all(hit is not None for hit in col_to_hits):
                    break

                # Create a dictionary for O(1) lookup
                rank_lookup = {
                    (int(xi), int(yi)): i
                    for i, (xi, yi) in enumerate(zip(x, y, strict=True))
                }

                for c, (qx, qy) in enumerate(queries):
                    if col_to_hits[c] is not None:
                        continue

                    if (qx, qy) in rank_lookup:
                        col_to_hits[c] = (r_idx, rank_lookup[(qx, qy)])

        else:
            for r_idx, info in enumerate(self.owner._rank_files):
                if info["saved_points"] == 0 or info["coord_raw"] is None:
                    continue

                if all(hit is not None for hit in col_to_hits):
                    break

                raw = np.asarray(info["coord_raw"]).ravel()
                rank_lookup = { int(val): i for i, val in enumerate(raw) }

                for c, qid in enumerate(queries):
                    if col_to_hits[c] is not None:
                        continue

                    if qid in rank_lookup:
                        col_to_hits[c] = (r_idx, rank_lookup[qid])

        if any(hit is None for hit in col_to_hits):
            raise ValueError("Some points not found in any rank.")

        logger.debug("Resolved %d statistics points across ranks", len(queries))

        out = np.empty((self.owner._time_len, N), dtype=target_dtype)

        rank_to_cols: dict[int, List[Tuple[int, int]]] = {}
        for col, hit in enumerate(col_to_hits):
            r_idx, li = hit  # hit is guaranteed not None
            rank_to_cols.setdefault(r_idx, []).append((col, li))

        # Fast path for an already materialized in-memory cache.
        for r_idx, pairs in rank_to_cols.items():
            info = self.owner._rank_files[r_idx]
            cache_arr = info.get("cache")
            if cache_arr is not None:
                self._copy_series_from_cache(
                    out, cache_arr, pairs, level, trial, target_dtype, info)
                continue

            # Without a materialized cache, still avoid NetCDF advanced column
            # indexing: read row chunks with all saved_points, then select
            # sorted columns in NumPy.
            if self.owner._slice_start is None or self.owner._slice_end is None:
                raise RuntimeError("Internal error: time slice is not set.")

            # Iterate over files to fill data
            for i, fp in enumerate(info["paths"]):
                file_start_global, file_end_global = info[
                    "file_time_offsets"
                ][i]

                # Intersection with requested slice [self.owner._slice_start, self.owner._slice_end]
                req_start = max(self.owner._slice_start, file_start_global)
                req_end = min(self.owner._slice_end + 1, file_end_global)

                if req_start < req_end:
                    local_start = req_start - file_start_global
                    local_end = req_end - file_start_global

                    # out is indexed 0..self.owner._time_len-1 corresponding to self.owner._slice_start..self.owner._slice_end
                    out_start = req_start - self.owner._slice_start

                    self._copy_series_from_row_chunks(
                        out, fp, info, pairs, local_start, local_end,
                        out_start, level, trial, target_dtype,
                    )

        return out

    # ----------------------------------------------------------------------------------
    # Basic info
    # ----------------------------------------------------------------------------------

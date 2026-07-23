# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from netCDF4 import Dataset

from hydroforge.contracts.temporal import timedelta_quotient
from hydroforge.data.datasets.base import AbstractDataset
from hydroforge.data.datasets.timeline import DatasetTimeline
from hydroforge.data.netcdf import read_netcdf_var_sliced, single_file_key
from hydroforge.data.distributed import find_indices_in, is_rank_zero
from hydroforge.serialization.netcdf import (
    DEFAULT_NETCDF_OPTIONS,
    atomic_netcdf_dataset,
    normalize_netcdf_variable_options,
)

import numba as _numba


logger = logging.getLogger(__name__)

@_numba.njit(cache=True, parallel=True)
def _gather_nb_kernel(data, shift, base_t, length, oob_fill):
    T, C = data.shape
    out = np.full((length, C), oob_fill, dtype=data.dtype)
    for c in _numba.prange(C):
        s = int(shift[c])
        for t in range(length):
            src = base_t + t + s
            if 0 <= src < T:
                out[t, c] = data[src, c]
    return out

_NUMBA_C_THRESHOLD = 5000  # Use numba for C above this (≈8x faster for glb_15min)


class ExportedDataset(AbstractDataset):
    """Dataset for pre-aggregated catchment runoff (time, saved_points).

    This dataset reads runoff data that has already been aggregated to catchment level,
    typically exported from a grid-based dataset using export_catchment_data().

    File convention (by default): f"{var_name}_rank{rank}.nc"
    Variables expected:
      - time: numeric with units/calendar
      - catchment_id: (saved_points,) linear catchment ids
      - {var_name}: (time, saved_points) values

    Key differences from grid-based datasets:
      - Data is already at catchment level, no grid-to-catchment mapping needed
      - build_local_mapping only reorders columns to match desired catchment order
      - shard_forcing simply flattens (B, T, C) -> (B*T, C) without matrix multiplication
      - Each rank can read its own file independently
    """

    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        var_name: str,
        prefix: Optional[str],
        time_interval: timedelta = timedelta(days=1),
        suffix: str = "rank0.nc",
        time_to_key: Optional[Callable[[datetime], str]] = single_file_key,
        coord_name: str = "catchment_id",
        in_memory: bool = False,
        unit_factor: float = 1.0,
        time_aggregation: Optional[Union[str, Dict[str, str]]] = None,
        clip_negative: bool = False,
        *args,
        **kwargs,
    ):
        self.coord_name = coord_name
        self.base_dir = base_dir
        self.var_name = var_name
        self.prefix = prefix or ""
        self.suffix = suffix
        self.time_to_key = time_to_key if time_to_key is not None else single_file_key
        self.unit_factor = unit_factor
        self.time_aggregation = self._normalize_time_aggregation(time_aggregation)
        self._in_memory = in_memory
        self._memory_cache: Optional[np.ndarray] = None  # Shape: (total_time_steps, num_catchments)

        # Per-catchment integer day shift applied at read time (None = no shift).
        # Populated by :meth:`build_local_mapping`.
        self._shift_days: Optional[np.ndarray] = None  # (C,) int64

        # Window-sampling mode (populated by :meth:`enable_windows`).
        self._window_len: Optional[int] = None
        self._window_starts: Optional[np.ndarray] = None

        # Inflow overlay (populated by :meth:`attach_inflow_overlay`).
        # ``_inflow_valid_length_days[c]`` marks the per-column valid span
        # on the shifted read axis ``[0, valid_length[c])``.
        self._inflow_data: Optional[np.ndarray] = None   # (T_full, C_in) f32
        self._inflow_shift_days: Optional[np.ndarray] = None     # (C_in,)
        self._inflow_valid_length_days: Optional[np.ndarray] = None  # (C_in,)

        # Basin-level coordinated (shift, length), keyed by basin id.
        # Populated by :meth:`attach_inflow_overlay`.
        self._basin_shift: dict[int, int] = {}
        self._basin_length: dict[int, int] = {}

        # Loss overlay (populated by :meth:`attach_loss_overlay`); NaN preserved.
        self._loss_data: Optional[np.ndarray] = None     # (T_full, C_loss) f32
        self._loss_shift_days: Optional[np.ndarray] = None       # (C_loss,)

        # Precomputed shift groups for fast _gather dispatch (avoids np.unique per call).
        self._shift_day_groups: Optional[list] = None
        self._inflow_shift_groups: Optional[list] = None
        self._loss_shift_groups: Optional[list] = None

        self._column_bbox: Optional[Tuple[int, int]] = None
        self._column_bbox_local_indices: Optional[np.ndarray] = None

        # Auto-detect chunk_len from file's NetCDF time chunking if not provided
        if "chunk_len" not in kwargs:
            detected = self._detect_chunk_len(base_dir, prefix, suffix, var_name, start_date, time_to_key)
            if detected is not None:
                kwargs["chunk_len"] = detected

        super().__init__(
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            clip_negative=clip_negative,
            *args,
            **kwargs,
        )
        self._timeline = DatasetTimeline(
            self,
            base_dir=base_dir,
            prefix=self.prefix,
            suffix=suffix,
            time_to_key=self.time_to_key,
            time_aggregation=self.time_aggregation,
        )
        self._global_times = self._timeline.global_times
        self._plan = self._timeline.plan
        self.source_time_interval = self._timeline.source_time_interval
        self._aggregation_factor = self._timeline.aggregation_factor

    def _ops_from_times(self, times):
        return self._timeline.ops_from_times(times)

    def is_valid_time_index(self, idx: int) -> bool:
        return self._timeline.is_valid_time_index(idx)

    @staticmethod
    def _detect_chunk_len(base_dir, prefix, suffix, var_name, start_date, time_to_key):
        """Detect chunk_len from file's NetCDF time chunking."""
        key = time_to_key(start_date) if time_to_key else ""
        path = Path(base_dir) / f"{prefix}{key}{suffix}"
        if not path.exists():
            return None
        with Dataset(path, "r") as ds:
            if var_name not in ds.variables:
                raise KeyError(f"variable {var_name!r} is absent from {path}")
            var = ds.variables[var_name]
            chunking = var.chunking()
            if chunking == "contiguous" or not chunking:
                return None
            dims = tuple(d.lower() for d in var.dimensions)
            if "time" in dims:
                return int(chunking[dims.index("time")])
        return None

    @staticmethod
    def _compile_groups(shift: np.ndarray) -> list:
        """Precompute [(shift_val, col_indices), ...] for fast _gather dispatch."""
        unique_shifts, inv = np.unique(shift, return_inverse=True)
        return [(int(s), np.where(inv == i)[0]) for i, s in enumerate(unique_shifts)]

    # -------------------------
    # Coordinates (1D catchment IDs)
    # -------------------------
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return catchment coordinate arrays.

        Returns (output_coord, index) where:
          - output_coord: linear catchment id array of shape (C,)
          - index: simple 0..C-1 integer array of shape (C,)
        """
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            if self.coord_name not in ds.variables:
                raise ValueError(f"Coordinate variable '{self.coord_name}' not found in {path.name}. "
                               f"Available: {list(ds.variables.keys())}")
            arr = ds.variables[self.coord_name][:]
            sc = (arr.filled(0) if isinstance(arr, np.ma.MaskedArray) else np.asarray(arr)).astype(np.int64)
            return sc, np.arange(sc.shape[0], dtype=np.int64)

    @property
    def data_size(self) -> int:
        """Return number of catchments in the exported file."""
        if self._local_indices is not None:
            return len(self._local_indices)
        sc, _ = self.get_coordinates()
        return len(sc)

    # -------------------------
    # Reading helpers (T, C)
    # -------------------------
    @staticmethod
    def _ensure_tc(data: np.ndarray, t_idx: Optional[int], c_idx: Optional[int]) -> np.ndarray:
        """Transpose data to (T, C) format."""
        if t_idx is None:
            raise ValueError("A time dimension is required.")
        axes = list(range(data.ndim))
        if c_idx is None:
            rest = [a for a in axes if a != t_idx]
            if len(rest) != 1:
                raise ValueError(f"Expected one non-time axis, got shape={data.shape}")
            c_idx = rest[0]
        front = [t_idx, c_idx]
        back = [a for a in axes if a not in front]
        out = np.transpose(data, axes=front + back)
        if out.ndim > 2:
            tail = out.shape[2:]
            if any(s != 1 for s in tail):
                raise ValueError(f"Unsupported extra dims: shape={out.shape}")
            out = out.reshape(out.shape[0], out.shape[1])
        return out

    def _compute_column_bbox_from_indices(self) -> None:
        """Compute the minimal saved_points slice for mapped catchments."""
        if self._local_indices is None:
            self._column_bbox = None
            self._column_bbox_local_indices = None
            return
        if self._local_indices.size == 0:
            self._column_bbox = (0, -1)
            self._column_bbox_local_indices = np.empty((0,), dtype=np.int64)
            return

        col_min = int(self._local_indices.min())
        col_max = int(self._local_indices.max())
        self._column_bbox = (col_min, col_max)
        self._column_bbox_local_indices = (
            self._local_indices - col_min
        ).astype(np.int64, copy=False)

    def _read_ops(self, ops: List[Tuple[str, List[int]]]) -> np.ndarray:
        """Read time steps and reorder columns if _local_indices is set."""
        # Determine output size
        if self._local_indices is not None:
            if self._column_bbox is None:
                self._compute_column_bbox_from_indices()
            out_cols = len(self._local_indices)
        else:
            sc, _ = self.get_coordinates()
            out_cols = len(sc)

        use_column_bbox = (
            self._local_indices is not None
            and self._column_bbox is not None
            and self._column_bbox_local_indices is not None
        )

        if not ops:
            return np.empty((0, out_cols), dtype=self.out_dtype)

        chunks: List[np.ndarray] = []
        for key, abs_indices in ops:
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                var = ds.variables[self.var_name]
                dims = tuple(d.lower() for d in var.dimensions)
                if not (len(dims) == 2 and set(dims) == {"time", "saved_points"}):
                    raise ValueError(
                        f"Expected dims ('time','saved_points'), got {var.dimensions}"
                    )
                t_idx = dims.index("time")
                c_idx = dims.index("saved_points")
                if not abs_indices:
                    continue
                abs_idx = np.asarray(abs_indices, dtype=np.int32)
                sel = [slice(None)] * var.ndim
                sel[t_idx] = abs_idx
                if use_column_bbox:
                    col_min, col_max = self._column_bbox
                    sel[c_idx] = slice(col_min, col_max + 1)
                arr = read_netcdf_var_sliced(var, tuple(sel))
                arr = self._apply_value_policy(arr)
                arr = self._ensure_tc(arr, t_idx, c_idx)

                # Reorder columns if indices are set
                if self._local_indices is not None:
                    if use_column_bbox:
                        arr = arr[:, self._column_bbox_local_indices]
                    else:
                        arr = arr[:, self._local_indices]

                chunks.append(arr.astype(self.out_dtype, copy=False))

        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def _finish_read(self, data: np.ndarray):
        if self.time_aggregation is not None:
            data = self._apply_time_aggregation(
                data,
                self.source_time_interval,
                self.time_aggregation,
            )
            if isinstance(data, dict):
                return {name: block / self.unit_factor for name, block in data.items()}
        return data / self.unit_factor

    def read_chunk(self, idx: int):
        if idx < 0 or idx >= len(self._plan):
            raise IndexError(f"Chunk index {idx} out of range (0-{len(self._plan) - 1})")
        return self._finish_read(self._read_ops(self._plan[idx][1]))

    def get_data(self, current_time: datetime, chunk_len: int):
        start = self.get_index_by_time(current_time)
        times = self._global_times[start:min(start + int(chunk_len), len(self._global_times))]
        if not times:
            return np.empty((0, self.data_size), dtype=self.out_dtype)
        return self._finish_read(self._read_ops(self._timeline.build_entry(times)[1]))

    def close(self) -> None:
        """No persistent NetCDF handles are retained."""

    # -------------------------
    # Build local mapping (column reorder only)
    # -------------------------
    def build_local_mapping(
        self,
        desired_catchment_ids: np.ndarray,
        desired_basin_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Set up column reordering and per-catchment shift for runoff.

        Must be called **after** :meth:`attach_inflow_overlay` so that
        ``_basin_shift`` is populated.  The per-catchment runoff shift is
        auto-derived as ``shift[c] = _basin_shift.get(basin[c], 0)``.

        Parameters
        ----------
        desired_catchment_ids : np.ndarray, shape (C,)
            Catchment ids in the order consumers want.
        desired_basin_ids : np.ndarray, shape (C,), optional
            Basin id of each desired catchment (e.g.
            ``model.base.catchment_basin_id``).  When *None* (default) no
            per-catchment shift is derived, which is the correct behaviour
            when no inflow shift is needed.
        """
        # Load catchment IDs from file
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            if self.coord_name not in ds.variables:
                raise ValueError(f"Coordinate variable '{self.coord_name}' not found in {path}")
            arr = ds.variables[self.coord_name][:]
            file_catchment_ids = (arr.filled(0) if isinstance(arr, np.ma.MaskedArray)
                                  else np.asarray(arr)).astype(np.int64)

        col_pos = find_indices_in(desired_catchment_ids, file_catchment_ids)
        if np.any(col_pos == -1):
            missing = int(np.sum(col_pos == -1))
            raise ValueError(
                f"{missing} desired catchments not found in exported file {path.name}"
            )
        self._local_indices = col_pos.astype(np.int64)
        self._compute_column_bbox_from_indices()

        if is_rank_zero():
            logger.info(
                "Mapped %d catchments from %d in exported file",
                len(desired_catchment_ids), len(file_catchment_ids),
            )

        # Derive per-catchment shift from basin_shift (auto after attach_inflow_overlay).
        if desired_basin_ids is not None and self._basin_shift:
            bids = np.asarray(desired_basin_ids, dtype=np.int64).ravel()
            if bids.shape != (len(desired_catchment_ids),):
                raise ValueError(
                    f"desired_basin_ids must have shape "
                    f"({len(desired_catchment_ids)},); got {bids.shape}")
            sh = np.array(
                [int(self._basin_shift.get(int(b), 0)) for b in bids],
                dtype=np.int64,
            )
            if np.any(sh != 0):
                self._shift_days = sh
                self._shift_day_groups = self._compile_groups(sh)
                if is_rank_zero():
                    logger.info(
                        "Registered per-catchment shift: %d unique values, "
                        "range [%d, %d] days",
                        np.unique(sh).size, int(sh.min()), int(sh.max()),
                    )

        # Load to memory when inflow overlay is attached (enables large-window
        # reads for val/test) or when per-catchment shift is applied (requires
        # random-access _gather).
        if self._inflow_data is not None or self._shift_days is not None:
            self.load_to_memory()

        return None

    def load_to_memory(self) -> None:
        """Load all data into memory for faster repeated access.

        This method reads the entire dataset into a numpy array cached in memory,
        covering ALL files that span the [start_date, end_date] range.
        Subsequent __getitem__ calls will return slices from this cache instead
        of reading from disk.

        Note: build_local_mapping should be called first to set up
        the column reordering indices.
        """
        if self._memory_cache is not None:
            if is_rank_zero():
                logger.info("Exported data is already resident in memory")
            return

        # Read ALL time steps across all files using the multi-file infrastructure.
        ops = self._ops_from_times(self._global_times)
        all_data = self._read_ops(ops)

        # Store in cache with correct dtype and C-contiguous layout
        self._memory_cache = np.ascontiguousarray(all_data.astype(self.out_dtype, copy=False))

        if is_rank_zero():
            n_files = len(ops)
            mem_mb = self._memory_cache.nbytes / (1024 * 1024)
            logger.info(
                "Loaded exported data shape=%s from %d file(s) (%.1f MiB)",
                self._memory_cache.shape, n_files, mem_mb,
            )

    def export_quantiles(
        self,
        out_path: Union[str, Path],
        quantiles: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        var_name: Optional[str] = None,
        dtype: Literal["float32", "float64"] = "float32",
        netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
        max_buffer_mb: float = 4096.0,
    ) -> Path:
        """Compute per-catchment temporal quantiles and save to NetCDF.

        For each catchment, computes the specified quantile values across the
        time dimension and writes the result to a single NetCDF file.

        Output format (consistent with ExportedDataset conventions):
          - Dimensions: ``quantile`` (Q), ``saved_points`` (C)
          - Variables:
            * ``quantile``     (Q,)    - quantile levels (e.g. 0.0 … 1.0)
            * ``catchment_id`` (C,)    - catchment IDs (int64)
            * ``{var_name}``   (Q, C)  - quantile values

        If ``build_local_mapping`` has been called, the output follows the
        reordered catchment order; otherwise it uses the file's native order.

        Exact quantile computation requires the full time series per catchment.
        When the full (T, C) array exceeds ``max_buffer_mb``, catchments are
        processed in column-batches whose size is automatically computed so that
        each batch (T × batch_catchments) fits within the buffer limit.

        Args:
            out_path: Output NetCDF file path.
            quantiles: Sequence of quantile levels in [0, 1].
            var_name: Variable name in output file (default: ``self.var_name``).
            dtype: Output data type.
            netcdf_options: Validated NetCDF variable-creation options.
            max_buffer_mb: Maximum memory buffer in MB for reading data.
                When the full dataset exceeds this limit, catchments are
                processed in column-batches automatically. Default 4096 (4 GB).

        Returns:
            Path to the created NetCDF file.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        create_options = normalize_netcdf_variable_options(netcdf_options)
        var_name = var_name or self.var_name
        quantiles_arr = np.asarray(quantiles, dtype=np.float64)
        Q = len(quantiles_arr)
        elem_bytes = 4 if dtype == "float32" else 8

        # ---- catchment IDs (respecting column reorder) ----
        file_catchment_ids, _ = self.get_coordinates()
        if self._local_indices is not None:
            catchment_ids = file_catchment_ids[self._local_indices]
        else:
            catchment_ids = file_catchment_ids
        C_total = len(catchment_ids)
        T_total = self.num_main_steps

        # ---- determine whether full (T, C) fits in buffer ----
        max_buffer_bytes = max_buffer_mb * 1024 * 1024
        full_size = T_total * C_total * elem_bytes
        fits_in_memory = (self._memory_cache is not None or full_size <= max_buffer_bytes)

        if not fits_in_memory:
            # Column-batch mode: compute batch_size (num catchments per batch)
            # so that T_total × batch_size × elem_bytes <= max_buffer_bytes
            batch_size = max(1, int(max_buffer_bytes / (T_total * elem_bytes)))
            n_batches = (C_total + batch_size - 1) // batch_size
            if is_rank_zero():
                logger.info(
                    "Exported dataset %.1f GB exceeds %.0f MiB buffer; "
                    "processing %d catchments in %d batches of %d",
                    full_size / 1e9, max_buffer_mb, C_total, n_batches,
                    batch_size,
                )

        # ---- create output NetCDF ----
        dtype_nc = "f4" if dtype == "float32" else "f8"
        with atomic_netcdf_dataset(out_path, format="NETCDF4") as out_ds:
            out_ds.createDimension("quantile", Q)
            out_ds.createDimension("saved_points", C_total)

            q_var = out_ds.createVariable("quantile", "f8", ("quantile",))
            q_var[:] = quantiles_arr
            q_var.long_name = "quantile level"

            cid_var = out_ds.createVariable("catchment_id", "i8", ("saved_points",))
            cid_var[:] = catchment_ids

            data_var = out_ds.createVariable(
                var_name, dtype_nc, ("quantile", "saved_points"),
                **create_options,
            )
            data_var.long_name = f"{var_name} quantile values"

            if fits_in_memory:
                # ---- fits in memory: load full series once, compute quantiles ----
                if self._memory_cache is None:
                    self.load_to_memory()
                all_data = self._memory_cache[:T_total]
                q_values = np.quantile(all_data, quantiles_arr, axis=0)  # (Q, C)
                data_var[:] = q_values.astype(dtype)
            else:
                # ---- too large: batch by catchments (columns) ----
                # Exact quantile needs full time axis, so we read ALL time steps
                # for a subset of catchments per batch.
                ops = self._ops_from_times(self._global_times)
                for c_start in range(0, C_total, batch_size):
                    c_end = min(c_start + batch_size, C_total)
                    batch_cols = slice(c_start, c_end)

                    if self._local_indices is not None:
                        file_col_indices = self._local_indices[c_start:c_end]
                    else:
                        file_col_indices = np.arange(c_start, c_end, dtype=np.int64)

                    file_chunks: List[np.ndarray] = []
                    for key, abs_indices in ops:
                        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
                        with Dataset(path, "r") as ds_in:
                            var_in = ds_in.variables[self.var_name]
                            dims_in = tuple(d.lower() for d in var_in.dimensions)
                            t_idx = dims_in.index("time")
                            c_idx = dims_in.index("saved_points")

                            sel = [slice(None)] * var_in.ndim
                            sel[t_idx] = np.asarray(abs_indices, dtype=np.int64)
                            sel[c_idx] = file_col_indices
                            arr = read_netcdf_var_sliced(var_in, tuple(sel))
                            arr = self._apply_value_policy(arr)
                            batch_data = self._ensure_tc(arr, t_idx, c_idx)
                            file_chunks.append(batch_data)

                    all_batch = np.concatenate(file_chunks, axis=0) if len(file_chunks) > 1 else file_chunks[0]
                    q_batch = np.quantile(all_batch, quantiles_arr, axis=0)
                    data_var[:, batch_cols] = q_batch.astype(dtype)

        if is_rank_zero():
            logger.info(
                "Saved quantiles to %s: levels=%s, shape=(%d, %d)",
                out_path, quantiles_arr.tolist(), Q, C_total,
            )

        return out_path

    def shard_forcing(
        self,
        batch_data,
    ):
        """Flatten (B, T, C) -> (B*T, C).

        For ExportedDataset, data is already in the correct column order
        (set by build_local_mapping), so no matrix multiply is needed.

        When overlays are attached (:meth:`attach_inflow_overlay` and/or
        :meth:`attach_loss_overlay`), ``batch_data`` is a tuple of
        per-stream tensors; each is flattened independently and returned
        as a tuple in the same order.
        """
        if isinstance(batch_data, (tuple, list)):
            return tuple(self.shard_forcing(b) for b in batch_data)
        if batch_data.dim() == 3:
            B, T, C = batch_data.shape
            return batch_data.reshape(B * T, C).contiguous()
        elif batch_data.dim() == 4:
            B, T, K, C = batch_data.shape
            return batch_data.reshape(B * T, K, C).contiguous()
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {batch_data.dim()}D")

    # -------------------------
    # Override __getitem__ - no rank gating for exported data
    # -------------------------
    def __getitem__(self, idx):
        """Fetch a chunk or window (no rank gating for exported data).

        Dispatches on the active mode:

        * Window-sampling (``enable_windows``): ``base_t`` comes from
          ``_window_starts[idx]`` and length is ``_window_len``.
        * Chunked (default): ``base_t = idx * chunk_len``.

        Then returns ``_gather(memory_cache, _shift_days, base_t, length)``
        if data is cached, else a contiguous ``read_chunk(idx)`` slice.
        When :meth:`attach_inflow_overlay` is active, also gathers the
        inflow overlay and returns ``(runoff, inflow, inflow_valid)``;
        if :meth:`attach_loss_overlay` is additionally active, returns
        ``(runoff, inflow, inflow_valid, loss)``.
        """
        if idx < 0:
            idx += len(self)
        if self._window_starts is not None:
            base_t = int(self._window_starts[idx])
            length = int(self._window_len)
        else:
            base_t = self._chunk_base_t(idx)
            length = int(self.chunk_len)

        if self._memory_cache is not None:
            runoff = self._gather(self._memory_cache, self._shift_days, base_t, length,
                                  groups=self._shift_day_groups)
        else:
            if self._shift_days is not None:
                raise RuntimeError("per-catchment shift requires in-memory data; "
                                   "call load_to_memory() first")
            # Disk path: contiguous chunk read, zero-padded to `length`.
            data = self.read_chunk(idx)
            T = data.shape[0]
            if T < length:
                pad = np.zeros((length - T, self.data_size), dtype=self.out_dtype)
                data = np.vstack([data, pad]) if data.size else pad
            runoff = np.ascontiguousarray(data)

        if self._inflow_data is None:
            return runoff
        inflow = self._gather(self._inflow_data, self._inflow_shift_days, base_t, length,
                              groups=self._inflow_shift_groups)
        if self._loss_data is None:
            return runoff, inflow
        loss = self._gather(self._loss_data, self._loss_shift_days, base_t, length,
                            oob_fill=np.nan, groups=self._loss_shift_groups)
        return runoff, inflow, loss

    def _chunk_base_t(self, idx: int) -> int:
        """Return read-axis offset for a chunk, including spin-up cycling."""
        if self.spin_up_cycles <= 0:
            return idx * self.chunk_len
        if self.time_interval is None:
            raise ValueError("time_interval must be provided for spin-up reads")
        if self.spin_up_start_date is None or self.spin_up_end_date is None:
            raise ValueError(
                "spin_up_start_date and spin_up_end_date are required when "
                "spin_up_cycles > 0"
            )
        total_spin_up_chunks = self._spin_up_num_chunks * self.spin_up_cycles
        if idx < total_spin_up_chunks:
            cycle_idx = idx % self._spin_up_num_chunks
            spin_offset = self.spin_up_start_date - self.start_date
            spin_offset_steps = timedelta_quotient(
                spin_offset,
                self.time_interval,
                duration_label="spin-up read offset",
                interval_label="time_interval",
            )
            return spin_offset_steps + cycle_idx * self.chunk_len
        return (idx - total_spin_up_chunks) * self.chunk_len

    @staticmethod
    def _gather(data: np.ndarray, shift: Optional[np.ndarray],
                base_t: int, length: int,
                oob_fill: float = 0.0, *, groups: Optional[list] = None) -> np.ndarray:
        """Gather a ``(length, C)`` window from in-memory ``data``.

        Without ``shift``/``groups``: plain contiguous slice, zero-padded at
        boundaries.

        With shift, dispatches based on ``C``:
        - ``C >= _NUMBA_C_THRESHOLD`` and numba available → parallel per-column
          kernel (~8x faster for large C, e.g. glb_15min runoff).
        - Otherwise → precomputed ``groups`` slice-copy (fastest for small C,
          e.g. inflow/loss overlays).
        """
        T, C = data.shape
        if shift is None and groups is None:
            lo = max(base_t, 0)
            hi = min(base_t + length, T)
            if lo == base_t and hi == base_t + length:
                return data[lo:hi].copy()
            out = np.full((length, C), oob_fill, dtype=data.dtype)
            if lo < hi:
                out[lo - base_t: hi - base_t] = data[lo:hi]
            return out
        if C >= _NUMBA_C_THRESHOLD:
            return _gather_nb_kernel(data, shift, base_t, length, float(oob_fill))
        out = np.full((length, C), oob_fill, dtype=data.dtype)
        if groups is None:
            unique_shifts, inv = np.unique(shift, return_inverse=True)
            groups = [(int(s), np.where(inv == i)[0]) for i, s in enumerate(unique_shifts)]
        for s, cols in groups:
            src_lo = base_t + s
            clip_lo = max(src_lo, 0)
            clip_hi = min(src_lo + length, T)
            if clip_lo >= clip_hi:
                continue
            out[clip_lo - src_lo: clip_hi - src_lo, cols] = data[clip_lo:clip_hi, cols]
        return out

    def __len__(self) -> int:
        """Window mode length, or chunk-based length."""
        if self._window_starts is not None:
            return int(self._window_starts.size)
        return super().__len__()

    def enable_windows(self, window: int, stride: Optional[int] = None) -> None:
        """Switch ``__getitem__``/``__len__`` to shifted-window sampling.

        ``self[idx]`` returns ``(window, C)`` covering
        ``[starts[idx], starts[idx] + window)`` on the shifted time axis,
        where ``starts = np.arange(0, T - window + 1, stride)``.
        Combined with DataLoader ``shuffle=True`` this gives randomized
        training windows.  Compatible with per-catchment shift and with
        the inflow overlay.
        """
        window = int(window)
        stride = int(stride) if stride is not None else window
        if window <= 0 or stride <= 0:
            raise ValueError(f"window/stride must be positive; got {window}/{stride}")
        T = len(self._global_times)
        if T < window:
            raise ValueError(f"window={window} exceeds total time steps {T}")
        self._window_len = window
        self._window_starts = np.arange(0, T - window + 1, stride, dtype=np.int64)
        if is_rank_zero():
            logger.info(
                "Enabled window sampling: window=%d, stride=%d, windows=%d, "
                "time_steps=%d", window, stride, self._window_starts.size, T,
            )

    # -------------------------
    # Overlay helpers
    # -------------------------
    def _align_overlay_data(
        self,
        data: np.ndarray,
        data_start_date: datetime,
    ) -> np.ndarray:
        """Crop/pad a native-axis overlay to ``self._global_times``.

        ``data[0]`` corresponds to ``data_start_date``; output row 0
        corresponds to ``self.start_date`` and output has
        ``num_main_steps`` rows.  Missing rows are zero-filled; rows
        beyond the dataset window are dropped.  The daily time
        interval is assumed (the only case used by gauge overlays).
        """
        T_ds = int(self.num_main_steps)
        offset = (self.start_date - data_start_date).days
        T_src = int(data.shape[0])
        out = np.zeros((T_ds, data.shape[1]), dtype=np.float32)
        # Source row j aligns with output row (j - offset).
        src_lo = max(offset, 0)
        src_hi = min(offset + T_ds, T_src)
        if src_lo < src_hi:
            out[src_lo - offset: src_hi - offset] = data[src_lo:src_hi]
        return out

    @staticmethod
    def _longest_valid_run(valid: np.ndarray) -> tuple[int, int]:
        """Return ``(start, length)`` of the longest contiguous True run."""
        if not valid.any():
            return 0, 0
        padded = np.concatenate(([False], valid, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = ends - starts
        k = int(np.argmax(lengths))
        return int(starts[k]), int(lengths[k])

    # -------------------------
    # Inflow overlay
    # -------------------------
    def attach_inflow_overlay(
        self,
        data: np.ndarray,
        data_start_date: datetime,
        data_catchment_ids: np.ndarray,
        desired_catchment_ids: np.ndarray,
        desired_basin_ids: np.ndarray,
    ) -> None:
        """Attach an inflow overlay from raw gauge data.

        The overlay is supplied on its **native observation axis** with
        an explicit ``data_start_date``; this method handles time
        cropping to the dataset window, column reordering, per-column
        longest-valid-run detection (NaN in the raw data ⇒ invalid),
        per-basin ``(shift, length)`` coordination and NaN→0 filling.

        After this call:

        * ``self._basin_shift[b] / self._basin_length[b]`` expose the
          basin-coordinated offset/length keyed by basin id.  These are
          consumed by :meth:`build_local_mapping` (runoff per-catchment
          shift) and :meth:`attach_loss_overlay` (loss per-POI shift).
        * ``self._inflow_shift_days[c]`` equals
          ``self._basin_shift[desired_basin_ids[c]]``.
        * ``self._inflow_valid_length_days[c]`` equals
          ``self._basin_length[desired_basin_ids[c]]`` — the number of
          leading read-axis steps that are observed (the rest are 0).
          ``__getitem__`` returns ``(runoff, inflow[, loss])`` without
          a per-window validity mask; consumers read this attribute to
          mask partial windows.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(T_src, N_source)`` float.  Native-axis rows; NaN
            marks missing.
        data_start_date : datetime
            Calendar date of ``data[0]``.
        data_catchment_ids : np.ndarray
            Shape ``(N_source,)`` int64 — catchment IDs per column of
            ``data``.
        desired_catchment_ids : np.ndarray
            Shape ``(N_desired,)`` int64 — the column order the consumer
            expects (typically ``model.inflow.inflow_catchment_id``).
        desired_basin_ids : np.ndarray
            Shape ``(N_desired,)`` int64 — basin id of each desired
            column (typically ``model.base.catchment_basin_id["
            "model.inflow.inflow_catchment_idx]``). Used to
            coordinate per-basin (shift, length).
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"attach_inflow_overlay: data must be 2-D; got {data.shape}")

        src_cids = np.asarray(data_catchment_ids, dtype=np.int64)
        if src_cids.shape != (data.shape[1],):
            raise ValueError(
                f"attach_inflow_overlay: data_catchment_ids shape "
                f"{src_cids.shape} does not match data columns {data.shape[1]}")
        dst_cids = np.asarray(desired_catchment_ids, dtype=np.int64)
        if np.unique(dst_cids).size != dst_cids.size:
            raise ValueError(
                "attach_inflow_overlay: desired_catchment_ids must be unique; "
                "aggregate duplicate gauges on the dataset side"
            )
        dst_basin = np.asarray(desired_basin_ids, dtype=np.int64)
        if dst_basin.shape != dst_cids.shape:
            raise ValueError(
                f"attach_inflow_overlay: desired_basin_ids shape "
                f"{dst_basin.shape} != desired_catchment_ids shape "
                f"{dst_cids.shape}")

        source_groups = [np.flatnonzero(src_cids == cid) for cid in dst_cids]
        missing = sum(group.size == 0 for group in source_groups)
        if missing:
            raise ValueError(
                f"attach_inflow_overlay: {missing} desired catchment IDs "
                f"not found in data_catchment_ids")

        # Aggregate all source gauges mapped to the same injection catchment.
        # Preserve NaN when every contributing gauge is missing at a time step.
        data_reordered = np.empty((data.shape[0], dst_cids.size), dtype=np.float32)
        for c, cols in enumerate(source_groups):
            values = data[:, cols]
            all_missing = np.isnan(values).all(axis=1)
            total = np.nansum(values, axis=1, dtype=np.float32)
            total[all_missing] = np.nan
            data_reordered[:, c] = total
        data_reordered = np.ascontiguousarray(data_reordered)

        # Align time axis to the dataset window.
        aligned_raw = self._align_overlay_data(data_reordered, data_start_date)

        # 3. Per-column longest valid run on the aligned (dataset-window)
        #    axis.  NaN → invalid.
        C = dst_cids.size
        per_col_shift = np.zeros(C, dtype=np.int64)
        per_col_length = np.zeros(C, dtype=np.int64)
        for c in range(C):
            valid = ~np.isnan(aligned_raw[:, c])
            s, ln = self._longest_valid_run(valid)
            per_col_shift[c] = s
            per_col_length[c] = ln

        # 4. Per-basin coordination: shift = max leading offset,
        #    length = min end minus coord shift (clamped at 0).
        basin_shift: dict[int, int] = {}
        basin_end: dict[int, int] = {}
        for c in range(C):
            if per_col_length[c] == 0:
                continue
            b = int(dst_basin[c])
            s = int(per_col_shift[c])
            e = s + int(per_col_length[c])
            basin_shift[b] = max(basin_shift.get(b, 0), s)
            basin_end[b] = min(basin_end.get(b, e), e)
        basin_length: dict[int, int] = {
            b: max(0, basin_end[b] - basin_shift[b]) for b in basin_shift
        }

        # 5. Per desired column: (shift, length) = basin-coord.  Columns
        #    whose basin has no valid gauges retain (0, 0) and the
        #    overlay yields zero throughout for those columns.
        out_shift = np.zeros(C, dtype=np.int64)
        out_length = np.zeros(C, dtype=np.int64)
        for c in range(C):
            b = int(dst_basin[c])
            if b in basin_shift:
                out_shift[c] = basin_shift[b]
                out_length[c] = basin_length[b]

        # 6. Fill NaN with 0 on native axis AND zero-out positions
        #    outside ``[shift[c], shift[c] + length[c])`` so shifted
        #    reads beyond the valid span deterministically yield 0.
        inflow_data = np.where(np.isnan(aligned_raw), 0.0, aligned_raw)
        T_ds = inflow_data.shape[0]
        for c in range(C):
            s = int(out_shift[c])
            ln = int(out_length[c])
            if s > 0:
                inflow_data[:s, c] = 0.0
            if s + ln < T_ds:
                inflow_data[s + ln:, c] = 0.0
        self._inflow_data = np.ascontiguousarray(
            inflow_data.astype(np.float32))
        self._inflow_shift_days = out_shift
        self._inflow_shift_groups = self._compile_groups(out_shift)
        self._inflow_valid_length_days = out_length
        self._basin_shift = basin_shift
        self._basin_length = basin_length

        if is_rank_zero():
            n_with = int((out_length > 0).sum())
            max_shift = int(out_shift.max()) if C else 0
            logger.info(
                "Attached inflow overlay: gauges=%d, valid_spans=%d, "
                "basins=%d, max_shift_days=%d",
                C, n_with, len(basin_shift), max_shift,
            )

    @property
    def inflow_valid_length_days(self) -> Optional[np.ndarray]:
        """Per-column number of leading valid read-axis steps, or ``None``."""
        return self._inflow_valid_length_days

    @property
    def inflow_shift_days(self) -> Optional[np.ndarray]:
        """Per-column basin-coordinated read-axis shift (days), or ``None``."""
        return self._inflow_shift_days

    @property
    def basin_shift(self) -> dict[int, int]:
        return dict(self._basin_shift)

    @property
    def basin_length(self) -> dict[int, int]:
        return dict(self._basin_length)

    # -------------------------
    # Loss overlay
    # -------------------------
    def attach_loss_overlay(
        self,
        data: np.ndarray,
        data_start_date: datetime,
        data_catchment_ids: np.ndarray,
        desired_catchment_ids: np.ndarray,
        desired_basin_ids: np.ndarray,
    ) -> None:
        """Attach a loss-target overlay from raw gauge data.

        Symmetric to :meth:`attach_inflow_overlay` but preserves NaN
        (the loss function uses the NaN mask).  The per-column shift is
        read from ``self._basin_shift`` (populated by
        :meth:`attach_inflow_overlay`); columns whose basin has no
        inflow use shift 0.  No longest-run / length metadata is stored
        — ``__getitem__`` returns ``loss`` with NaN preserved outside
        the valid span via the gather's ``oob_fill=np.nan``.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"attach_loss_overlay: data must be 2-D; got {data.shape}")
        src_cids = np.asarray(data_catchment_ids, dtype=np.int64)
        if src_cids.shape != (data.shape[1],):
            raise ValueError(
                f"attach_loss_overlay: data_catchment_ids shape "
                f"{src_cids.shape} does not match data columns {data.shape[1]}")
        dst_cids = np.asarray(desired_catchment_ids, dtype=np.int64)
        dst_basin = np.asarray(desired_basin_ids, dtype=np.int64)
        if dst_basin.shape != dst_cids.shape:
            raise ValueError(
                f"attach_loss_overlay: desired_basin_ids shape "
                f"{dst_basin.shape} != desired_catchment_ids shape "
                f"{dst_cids.shape}")

        col_pos = find_indices_in(dst_cids, src_cids)
        if np.any(col_pos == -1):
            missing = int((col_pos == -1).sum())
            raise ValueError(
                f"attach_loss_overlay: {missing} desired catchment IDs "
                f"not found in data_catchment_ids")
        data_reordered = np.ascontiguousarray(data[:, col_pos])
        aligned = self._align_overlay_data(data_reordered, data_start_date)
        # _align_overlay_data zero-fills out-of-window rows; convert those
        # zeros back to NaN so the loss mask treats them as missing.
        # We detect out-of-window rows by re-computing the offset.
        T_ds = aligned.shape[0]
        offset = (self.start_date - data_start_date).days
        T_src = int(data.shape[0])
        oob = np.ones(T_ds, dtype=bool)
        lo = max(offset, 0)
        hi = min(offset + T_ds, T_src)
        if lo < hi:
            oob[lo - offset: hi - offset] = False
        aligned[oob, :] = np.nan

        shift = np.array(
            [int(self._basin_shift.get(int(b), 0)) for b in dst_basin],
            dtype=np.int64,
        )

        self._loss_data = aligned
        self._loss_shift_days = shift
        self._loss_shift_groups = self._compile_groups(shift)
        if is_rank_zero():
            nz = int((shift != 0).sum())
            logger.info(
                "Attached loss overlay: catchments=%d, nonzero_shifts=%d, "
                "max_shift=%d", dst_cids.size, nz,
                int(shift.max()) if shift.size else 0,
            )


# ---------------------------------------------------------------------------
# Composite multi-variable wrapper
# ---------------------------------------------------------------------------
def open_multivariable_exported(
    base_dir: str,
    var_specs,
    *,
    start_date: datetime,
    end_date: datetime,
    time_interval: timedelta = timedelta(days=1),
    chunk_len: Optional[int] = None,
    spin_up_cycles: int = 0,
    spin_up_start_date: Optional[datetime] = None,
    spin_up_end_date: Optional[datetime] = None,
    time_to_key: Optional[Callable[[datetime], str]] = None,
    coord_name: str = "catchment_id",
    in_memory: bool = False,
):
    """Open aligned catchment variables as one generic composite."""
    if not var_specs:
        raise ValueError("var_specs must contain at least one variable")
    shared = {
        "base_dir": base_dir, "start_date": start_date,
        "end_date": end_date, "time_interval": time_interval,
        "spin_up_cycles": spin_up_cycles,
        "spin_up_start_date": spin_up_start_date,
        "spin_up_end_date": spin_up_end_date,
        "coord_name": coord_name, "in_memory": in_memory,
    }
    if time_to_key is not None:
        shared["time_to_key"] = time_to_key
    if chunk_len is not None:
        shared["chunk_len"] = chunk_len
    datasets = {}
    for name, spec in var_specs.items():
        options = shared | dict(spec)
        options["var_name"] = name
        options.setdefault("prefix", f"{name}_")
        datasets[name] = ExportedDataset(**options)
    from hydroforge.data.datasets.multivariable import MultiVariableDataset

    return MultiVariableDataset(datasets, loader_strategy="combined")

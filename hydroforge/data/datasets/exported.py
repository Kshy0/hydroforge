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
      - shard_forcing validates concatenated (T, C) data without matrix multiplication
      - Each rank can read its own file independently
    """

    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        model_step: timedelta,
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
        self._memory_cache: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
        self._spin_up_memory_cache: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None
        # Absolute source-file column indices represented by both caches.
        # This lets ``in_memory=True`` preload in native order and still
        # honour a later ``build_local_mapping`` column reorder.
        self._memory_cache_file_indices: Optional[np.ndarray] = None

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
        self._coordinates_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # Name of the sparse axis in the input variable.  Exporters commonly
        # use ``saved_points``, but accepting a file's actual one-dimensional
        # catchment axis makes the reader interoperable with equivalent files
        # produced by other tools.
        self._point_dim: Optional[str] = None

        # Auto-detect chunk_len from file's NetCDF time chunking if not provided
        if "chunk_len" not in kwargs:
            detected = self._detect_chunk_len(
                base_dir,
                self.prefix,
                suffix,
                var_name,
                start_date,
                self.time_to_key,
            )
            if detected is not None:
                kwargs["chunk_len"] = detected

        super().__init__(
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            model_step=model_step,
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
        # ExportedDataset indexing, window sampling and the primary cache use
        # a main-only axis whose row zero is ``start_date``. The shared I/O
        # timeline separately follows the exact source chunks, including any
        # replayed spin-up rows.
        contract = self.temporal_contract
        self._global_times = [
            contract.support(index)[0] for index in range(contract.count)
        ]

        if self._in_memory:
            self.load_to_memory()

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
    def _validated_catchment_ids(
        self,
        value: Any,
        *,
        path: Path,
    ) -> np.ndarray:
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(
                f"Coordinate variable {self.coord_name!r} in {path.name} "
                "contains missing IDs"
            )
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError(
                f"Coordinate variable {self.coord_name!r} in {path.name} "
                "must be one-dimensional"
            )
        if array.dtype.kind not in "iu":
            raise TypeError(
                f"Coordinate variable {self.coord_name!r} in {path.name} "
                "must use an integer dtype"
            )
        if np.unique(array).size != array.size:
            raise ValueError(
                f"Coordinate variable {self.coord_name!r} in {path.name} "
                "contains duplicate IDs"
            )
        return array.astype(np.int64, copy=False)

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return catchment coordinate arrays.

        Returns (output_coord, index) where:
          - output_coord: linear catchment id array of shape (C,)
          - index: simple 0..C-1 integer array of shape (C,)
        """
        if self._coordinates_cache is None:
            key = self.time_to_key(self.start_date)
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                point_dim = self._infer_point_dim(ds, path)
                if self.coord_name not in ds.variables:
                    raise ValueError(
                        f"Coordinate variable '{self.coord_name}' not found in "
                        f"{path.name}. Available: {list(ds.variables.keys())}"
                    )
                coordinate = ds.variables[self.coord_name]
                if tuple(coordinate.dimensions) != (point_dim,):
                    raise ValueError(
                        f"Coordinate variable {self.coord_name!r} in {path.name} "
                        f"must have dimensions ({point_dim!r},), got "
                        f"{coordinate.dimensions}"
                    )
                coordinates = self._validated_catchment_ids(
                    coordinate[:], path=path
                )
            index = np.arange(coordinates.shape[0], dtype=np.int64)
            coordinates.setflags(write=False)
            index.setflags(write=False)
            self._coordinates_cache = (coordinates, index)
        return self._coordinates_cache

    @property
    def point_dim(self) -> str:
        """Name of the source variable's one-dimensional catchment axis."""

        if self._point_dim is None:
            self.get_coordinates()
        if self._point_dim is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("exported dataset point dimension was not resolved")
        return self._point_dim

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

    def _infer_point_dim(self, dataset: Dataset, path: Path) -> str:
        """Validate the exported variable and return its sparse point axis."""

        if self.var_name not in dataset.variables:
            raise KeyError(f"variable {self.var_name!r} is absent from {path}")
        dimensions = tuple(dataset.variables[self.var_name].dimensions)
        time_axes = [
            index for index, dimension in enumerate(dimensions)
            if dimension.lower() == "time"
        ]
        point_axes = [
            index for index in range(len(dimensions)) if index not in time_axes
        ]
        if len(time_axes) != 1 or len(point_axes) != 1:
            raise ValueError(
                f"Expected {self.var_name!r} in {path.name} to have one time "
                f"axis and one point axis, got {dimensions}"
            )
        point_dim = dimensions[point_axes[0]]
        if self._point_dim is None:
            self._point_dim = point_dim
        elif self._point_dim != point_dim:
            raise ValueError(
                f"Variable {self.var_name!r} uses point dimension "
                f"{point_dim!r} in {path.name}, expected {self._point_dim!r}"
            )
        return point_dim

    def _variable_axes(
        self,
        dataset: Dataset,
        variable: Any,
        path: Path,
    ) -> tuple[int, int]:
        """Return the time and sparse-point axes of one source variable."""

        point_dim = self._infer_point_dim(dataset, path)
        dimensions = tuple(variable.dimensions)
        return (
            next(
                index for index, dimension in enumerate(dimensions)
                if dimension.lower() == "time"
            ),
            dimensions.index(point_dim),
        )

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
                t_idx, c_idx = self._variable_axes(ds, var, path)
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
                self._timeline.source_time_interval,
                self.time_aggregation,
            )
        if self.unit_factor == 1.0:
            return data
        if isinstance(data, dict):
            return {name: block / self.unit_factor for name, block in data.items()}
        return data / self.unit_factor

    def _as_cache_data(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Normalize one processed cache while preserving aggregation maps."""
        if isinstance(data, dict):
            return {
                name: np.ascontiguousarray(
                    block.astype(self.out_dtype, copy=False)
                )
                for name, block in data.items()
            }
        return np.ascontiguousarray(data.astype(self.out_dtype, copy=False))

    @staticmethod
    def _cache_column_count(
        cache: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> int:
        if isinstance(cache, dict):
            first = next(iter(cache.values()))
            return int(first.shape[1])
        return int(cache.shape[1])

    @staticmethod
    def _cache_shape(cache: Union[np.ndarray, Dict[str, np.ndarray]]):
        if isinstance(cache, dict):
            return {name: block.shape for name, block in cache.items()}
        return cache.shape

    @staticmethod
    def _cache_nbytes(cache: Union[np.ndarray, Dict[str, np.ndarray]]) -> int:
        if isinstance(cache, dict):
            return sum(block.nbytes for block in cache.values())
        return cache.nbytes

    def _source_element_bytes(
        self, ops: Sequence[Tuple[str, Sequence[int]]],
    ) -> int:
        """Return a conservative element width for NetCDF source reads."""

        element_bytes = np.dtype(self.out_dtype).itemsize
        visited: set[str] = set()
        for key, _ in ops:
            if key in visited:
                continue
            visited.add(key)
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as dataset:
                variable = dataset.variables[self.var_name]
                read_dtype = np.dtype(variable.dtype)
                # netCDF4 applies packing attributes while reading. Include
                # their dtype in the estimate because a packed integer source
                # can therefore materialize as floating point in memory.
                for attribute in ("scale_factor", "add_offset"):
                    if hasattr(variable, attribute):
                        read_dtype = np.result_type(
                            read_dtype,
                            np.asarray(getattr(variable, attribute)).dtype,
                        )
                if not np.issubdtype(read_dtype, np.floating):
                    # A masked integer source is promoted to float64 when its
                    # missing values are converted to NaN by _apply_value_policy.
                    element_bytes = max(element_bytes, 8)
                element_bytes = max(element_bytes, read_dtype.itemsize)
        return element_bytes

    @staticmethod
    def _select_cache_columns(
        cache: Union[np.ndarray, Dict[str, np.ndarray]],
        positions: np.ndarray,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(cache, dict):
            return {
                name: np.ascontiguousarray(block[:, positions])
                for name, block in cache.items()
            }
        return np.ascontiguousarray(cache[:, positions])

    def read_chunk(self, idx: int):
        plan = self._timeline.plan
        if idx < 0 or idx >= len(plan):
            raise IndexError(f"Chunk index {idx} out of range (0-{len(plan) - 1})")
        return self._finish_read(self._read_ops(plan[idx][1]))

    def get_data(self, current_time: datetime, chunk_len: int):
        times = self._timeline.contiguous_times(current_time, chunk_len)
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
        *,
        time_shift_steps: Optional[np.ndarray] = None,
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
        time_shift_steps : np.ndarray, shape (C,), optional
            Explicit integer source-time offset for every mapped column.
            This is useful when another dataset has already coordinated the
            temporal spans. It is mutually exclusive with
            ``desired_basin_ids``.
        """
        if desired_basin_ids is not None and time_shift_steps is not None:
            raise ValueError(
                "desired_basin_ids and time_shift_steps are mutually exclusive"
            )
        # Reuse the validated coordinate cache populated on first access.
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        file_catchment_ids, _ = self.get_coordinates()

        col_pos = find_indices_in(desired_catchment_ids, file_catchment_ids)
        if np.any(col_pos == -1):
            missing = int(np.sum(col_pos == -1))
            raise ValueError(
                f"{missing} desired catchments not found in exported file {path.name}"
            )

        # ``in_memory=True`` loads before callers know the desired model
        # ordering.  Reindex an existing cache by its absolute file-column
        # provenance instead of treating already cached columns as if they
        # were still in native order.
        if self._memory_cache is not None:
            cached_file_indices = self._memory_cache_file_indices
            if cached_file_indices is None:
                raise RuntimeError(
                    "ExportedDataset memory cache is missing column provenance"
                )
            cache_pos = find_indices_in(col_pos, cached_file_indices)
            if np.any(cache_pos == -1):
                # A second mapping can request columns that a previous,
                # compressed cache no longer contains.  Reload below using
                # the new mapping rather than silently selecting wrong data.
                self._memory_cache = None
                self._spin_up_memory_cache = None
                self._memory_cache_file_indices = None
            else:
                self._memory_cache = self._select_cache_columns(
                    self._memory_cache, cache_pos
                )
                if self._spin_up_memory_cache is not None:
                    self._spin_up_memory_cache = self._select_cache_columns(
                        self._spin_up_memory_cache, cache_pos
                    )
                self._memory_cache_file_indices = col_pos.copy()

        self._local_indices = col_pos.astype(np.int64)
        self._compute_column_bbox_from_indices()

        if is_rank_zero():
            logger.info(
                "Mapped %d catchments from %d in exported file",
                len(desired_catchment_ids), len(file_catchment_ids),
            )

        # Derive per-catchment shift from basin_shift (auto after attach_inflow_overlay).
        self._shift_days = None
        self._shift_day_groups = None
        if time_shift_steps is not None:
            shifts = np.asarray(time_shift_steps)
            expected = (len(desired_catchment_ids),)
            if shifts.shape != expected:
                raise ValueError(
                    f"time_shift_steps must have shape {expected}; "
                    f"got {shifts.shape}"
                )
            if (
                not np.issubdtype(shifts.dtype, np.integer)
                or np.issubdtype(shifts.dtype, np.bool_)
            ):
                raise TypeError("time_shift_steps must contain integers")
            sh = shifts.astype(np.int64, copy=False)
            if np.any(sh != 0):
                self._shift_days = sh
                self._shift_day_groups = self._compile_groups(sh)
        elif desired_basin_ids is not None and self._basin_shift:
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
        if self._shift_days is not None and is_rank_zero():
            logger.info(
                "Registered per-catchment shift: %d unique values, "
                "range [%d, %d] steps",
                np.unique(self._shift_days).size,
                int(self._shift_days.min()),
                int(self._shift_days.max()),
            )

        # Load to memory when inflow overlay is attached (enables large-window
        # reads for val/test) or when per-catchment shift is applied (requires
        # random-access _gather).
        if (
            self._in_memory
            or self._inflow_data is not None
            or self._shift_days is not None
        ):
            self.load_to_memory()

        return None

    def load_to_memory(self) -> None:
        """Load all data into memory for faster repeated access.

        This method reads the entire dataset into a numpy array cached in memory,
        covering ALL files that span the [start_date, end_date] range.
        Subsequent __getitem__ calls will return slices from this cache instead
        of reading from disk.

        The cache records its absolute source columns, so callers may invoke
        this either before or after :meth:`build_local_mapping`.
        """
        if self._memory_cache is not None:
            if is_rank_zero():
                logger.info("Exported data is already resident in memory")
            return

        # Cache the same *output* axis returned by ``read_chunk``: value
        # policy, temporal aggregation and unit conversion must all happen
        # before data becomes resident.  Caching raw source frames makes
        # cache-backed indexing disagree with the disk path whenever either
        # aggregation or ``unit_factor`` is configured.
        main_entry = self._timeline.build_entry(self._global_times)
        all_data = self._finish_read(self._read_ops(main_entry[1]))

        # Spin-up can be disjoint from the main period by years.  Keep one
        # compact cache for the unique spin-up interval instead of indexing a
        # main-only cache with offsets relative to ``start_date`` (which used
        # to zero-fill valid spin-up data), or allocating the intervening gap.
        spin_data = None
        spinup = self.temporal_contract.spinup
        if spinup is not None:
            spin_count = self.chunk_plan.spinup_source_count_per_cycle
            spin_times = [
                spinup.source_start + self.time_interval * index
                for index in range(spin_count)
            ]
            spin_entry = self._timeline.build_entry(spin_times)
            spin_data = self._finish_read(self._read_ops(spin_entry[1]))

        # Store in cache with correct dtype and C-contiguous layout.
        self._memory_cache = self._as_cache_data(all_data)
        self._spin_up_memory_cache = (
            None if spin_data is None else self._as_cache_data(spin_data)
        )
        if self._local_indices is None:
            self._memory_cache_file_indices = np.arange(
                self._cache_column_count(self._memory_cache), dtype=np.int64
            )
        else:
            self._memory_cache_file_indices = self._local_indices.copy()

        if is_rank_zero():
            n_files = len(main_entry[1])
            mem_bytes = self._cache_nbytes(self._memory_cache)
            if self._spin_up_memory_cache is not None:
                mem_bytes += self._cache_nbytes(self._spin_up_memory_cache)
            logger.info(
                "Loaded exported data shape=%s, spin_up_shape=%s from %d "
                "file(s) (%.1f MiB)",
                self._cache_shape(self._memory_cache),
                None if self._spin_up_memory_cache is None else
                self._cache_shape(self._spin_up_memory_cache),
                n_files, mem_bytes / (1024 * 1024),
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
        if isinstance(self.time_aggregation, dict):
            raise ValueError(
                "export_quantiles requires one time-aggregation result; "
                "create a single-result ExportedDataset first"
            )
        create_options = normalize_netcdf_variable_options(netcdf_options)
        var_name = var_name or self.var_name
        quantiles_arr = np.asarray(quantiles, dtype=np.float64)
        Q = len(quantiles_arr)

        # ---- catchment IDs (respecting column reorder) ----
        file_catchment_ids, _ = self.get_coordinates()
        if self._local_indices is not None:
            catchment_ids = file_catchment_ids[self._local_indices]
        else:
            catchment_ids = file_catchment_ids
        C_total = len(catchment_ids)
        T_total = self.num_main_source_steps
        # ---- determine whether full (T, C) fits in buffer ----
        max_buffer_bytes = max_buffer_mb * 1024 * 1024
        main_ops = None
        source_rows = 0
        if self._memory_cache is not None:
            full_size = self._cache_nbytes(self._memory_cache)
            fits_in_memory = True
        else:
            # The peak read is on the expanded source axis, before temporal
            # aggregation.  Estimate it using the dataset's in-memory dtype,
            # not the requested NetCDF output dtype.
            main_ops = self._timeline.build_entry(self._global_times)[1]
            source_rows = sum(
                len(abs_indices) for _, abs_indices in main_ops
            )
            read_elem_bytes = self._source_element_bytes(main_ops)
            # Reading, concatenation/aggregation, and exact quantile
            # selection can briefly coexist. Three element-widths is a
            # conservative working-set estimate; using one array's size here
            # would advertise a buffer limit that NumPy immediately exceeds.
            working_elem_bytes = 3 * read_elem_bytes
            full_size = source_rows * C_total * working_elem_bytes
            fits_in_memory = full_size <= max_buffer_bytes

        if not fits_in_memory:
            # ``build_entry`` expands every requested output time into the
            # source frames required by temporal aggregation.  Reusing these
            # operations keeps this path identical to ``read_chunk`` and
            # ``load_to_memory``; ``ops_from_times`` alone would silently skip
            # the rest of each aggregation window.
            # Column-batch mode: compute batch_size (num catchments per batch)
            # from the expanded source axis, since aggregation may require
            # several source rows for every output row.
            rows_per_batch = max(1, source_rows)
            batch_size = max(
                1,
                int(
                    max_buffer_bytes
                    / (rows_per_batch * working_elem_bytes)
                ),
            )
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
                if main_ops is None:
                    raise RuntimeError("column-batch read plan was not built")
                for c_start in range(0, C_total, batch_size):
                    c_end = min(c_start + batch_size, C_total)
                    batch_cols = slice(c_start, c_end)

                    if self._local_indices is not None:
                        file_col_indices = self._local_indices[c_start:c_end]
                    else:
                        file_col_indices = np.arange(c_start, c_end, dtype=np.int64)

                    file_chunks: List[np.ndarray] = []
                    for key, abs_indices in main_ops:
                        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
                        with Dataset(path, "r") as ds_in:
                            var_in = ds_in.variables[self.var_name]
                            t_idx, c_idx = self._variable_axes(
                                ds_in, var_in, path,
                            )

                            sel = [slice(None)] * var_in.ndim
                            sel[t_idx] = np.asarray(abs_indices, dtype=np.int64)
                            sel[c_idx] = file_col_indices
                            arr = read_netcdf_var_sliced(var_in, tuple(sel))
                            arr = self._apply_value_policy(arr)
                            batch_data = self._ensure_tc(arr, t_idx, c_idx)
                            file_chunks.append(batch_data)

                    all_batch = (
                        np.concatenate(file_chunks, axis=0)
                        if len(file_chunks) > 1 else file_chunks[0]
                    )
                    if len(file_chunks) > 1:
                        # Drop references to the component reads before the
                        # aggregation/conversion allocation below.
                        file_chunks.clear()
                    else:
                        file_chunks.pop()
                    processed_batch = self._finish_read(all_batch)
                    q_batch = np.quantile(
                        processed_batch,
                        quantiles_arr,
                        axis=0,
                        overwrite_input=True,
                    )
                    data_var[:, batch_cols] = q_batch.astype(dtype)
                    # Python loop locals retain the previous batch unless
                    # explicitly released. Drop every large array before the
                    # next read so the three-array working-set estimate above
                    # remains conservative across batch boundaries.
                    del q_batch, processed_batch, all_batch, batch_data, arr

        if is_rank_zero():
            logger.info(
                "Saved quantiles to %s: levels=%s, shape=(%d, %d)",
                out_path, quantiles_arr.tolist(), Q, C_total,
            )

        return out_path

    def shard_forcing(
        self,
        chunk_data,
    ):
        """Validate already-concatenated ``(T, C)`` forcing.

        For ExportedDataset, data is already in the correct column order
        (set by build_local_mapping), so no matrix multiply is needed.

        When overlays are attached (:meth:`attach_inflow_overlay` and/or
        :meth:`attach_loss_overlay`), ``chunk_data`` is a tuple of
        per-stream tensors; each is flattened independently and returned
        as a tuple in the same order.
        """
        if isinstance(chunk_data, (tuple, list)):
            return tuple(
                self.shard_forcing(block) for block in chunk_data
            )
        if isinstance(chunk_data, Mapping):
            return {
                name: self.shard_forcing(block)
                for name, block in chunk_data.items()
            }
        if chunk_data.dim() in {2, 3}:
            out = chunk_data.contiguous()
        else:
            raise ValueError(
                "forcing must have shape (T, C) or (T, K, C); "
                f"got {tuple(chunk_data.shape)}"
            )
        return out

    # -------------------------
    # Override __getitem__ - no rank gating for exported data
    # -------------------------
    def _gather_cache(
        self,
        cache: Union[np.ndarray, Dict[str, np.ndarray]],
        shift: Optional[np.ndarray],
        base_t: int,
        length: int,
        *,
        groups: Optional[list],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(cache, dict):
            return {
                name: self._gather(
                    block, shift, base_t, length, groups=groups,
                )
                for name, block in cache.items()
            }
        return self._gather(cache, shift, base_t, length, groups=groups)

    def _validate_disk_data(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        length: int,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(data, dict):
            return {
                name: self._validate_disk_data(block, length)
                for name, block in data.items()
            }
        if data.ndim != 2 or data.shape[1] != self.data_size:
            raise ValueError(
                f"read_chunk returned shape {data.shape}, expected "
                f"(T, {self.data_size})"
            )
        if data.shape[0] != length:
            raise ValueError(
                f"read_chunk returned {data.shape[0]} rows, expected {length}"
            )
        return np.ascontiguousarray(data)

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
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"ExportedDataset index {idx} out of range for length {len(self)}"
            )
        if self._window_starts is not None:
            base_t = int(self._window_starts[idx])
            length = int(self._window_len)
            cache = self._memory_cache
        else:
            chunk = self.chunk_plan[idx]
            length = chunk.length
            base_t = chunk.phase_offset
            if chunk.phase == "spinup":
                cache = self._spin_up_memory_cache
            else:
                cache = self._memory_cache

        if self._memory_cache is not None:
            if cache is None:
                raise RuntimeError(
                    "spin-up memory cache is unavailable; call load_to_memory()"
                )
            runoff = self._gather_cache(
                cache, self._shift_days, base_t, length,
                groups=self._shift_day_groups,
            )
        else:
            if self._shift_days is not None:
                raise RuntimeError("per-catchment shift requires in-memory data; "
                                   "call load_to_memory() first")
            runoff = self._validate_disk_data(self.read_chunk(idx), length)

        runoff = self._apply_upsampling_policy(runoff)

        if self._inflow_data is None:
            return runoff
        # Overlays are aligned to the main dataset axis, not to the compact
        # spin-up cache.  Preserve the physical date-relative offset here.
        overlay_base_t = (
            int(self._window_starts[idx])
            if self._window_starts is not None
            else self._chunk_base_t(idx)
        )
        inflow = self._gather(self._inflow_data, self._inflow_shift_days,
                              overlay_base_t, length,
                              groups=self._inflow_shift_groups)
        if self._loss_data is None:
            return runoff, inflow
        loss = self._gather(self._loss_data, self._loss_shift_days,
                            overlay_base_t, length,
                            oob_fill=np.nan, groups=self._loss_shift_groups)
        return runoff, inflow, loss

    def _chunk_base_t(self, idx: int) -> int:
        """Return a chunk's logical offset from the main source origin."""

        return self.chunk_plan[idx].source_offset

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
        # Arbitrary/overlapping windows cannot be represented by the compiled
        # sequential chunk plan used by ``read_chunk``.  Make window mode
        # explicitly cache-backed instead of returning unrelated disk chunks.
        self.load_to_memory()
        if is_rank_zero():
            logger.info(
                "Enabled window sampling: window=%d, stride=%d, windows=%d, "
                "time_steps=%d", window, stride, self._window_starts.size, T,
            )

    def filter_windows(self, keep: np.ndarray) -> None:
        """Retain selected sampling windows without exposing private state."""

        if self._window_starts is None:
            raise RuntimeError("enable_windows must be called before filter_windows")
        mask = np.asarray(keep)
        if (
            not np.issubdtype(mask.dtype, np.bool_)
            or mask.shape != self._window_starts.shape
        ):
            raise ValueError(
                "window filter must be a boolean array with shape "
                f"{self._window_starts.shape}; got {mask.shape}"
            )
        self._window_starts = self._window_starts[mask]

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
        ``num_main_source_steps`` rows. Missing rows are NaN-filled so temporal
        padding cannot be mistaken for a real zero observation; rows beyond
        the dataset window are dropped.
        """
        T_ds = int(self.num_main_source_steps)
        T_src = int(data.shape[0])
        data_offset = timedelta_quotient(
            data_start_date - self.start_date,
            self.time_interval,
            duration_label="overlay start offset",
            interval_label="time_interval",
        )
        out = np.full((T_ds, data.shape[1]), np.nan, dtype=np.float32)
        src_lo = max(-data_offset, 0)
        dst_lo = max(data_offset, 0)
        count = min(T_src - src_lo, T_ds - dst_lo)
        if count > 0:
            out[dst_lo:dst_lo + count] = data[src_lo:src_lo + count]
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
        basin_shift_days: Optional[np.ndarray] = None,
        valid_length_days: Optional[np.ndarray] = None,
    ) -> None:
        """Attach an inflow overlay from raw gauge data.

        The overlay is supplied on its **native observation axis** with
        an explicit ``data_start_date``; this method handles time
        cropping to the dataset window, column reordering, per-column
        longest-valid-run detection (NaN in the raw data ⇒ invalid),
        per-basin ``(shift, length)`` coordination and NaN→0 filling.
        A producer that already qualified the native observation spans may
        pass both ``basin_shift_days`` and ``valid_length_days``; the pair is
        mapped by ``data_catchment_ids``, cropped to the dataset window and
        composed with the per-basin coordination instead of being inferred
        again from filled values.

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
        basin_shift_days, valid_length_days : np.ndarray, optional
            Native-axis valid spans corresponding one-for-one with
            ``data_catchment_ids``. They must be supplied together as
            non-negative integer arrays, and every span must fit inside
            ``data``. Despite the historical ``days`` name, values count
            dataset time steps.
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
        if (basin_shift_days is None) != (valid_length_days is None):
            raise ValueError(
                "attach_inflow_overlay: basin_shift_days and "
                "valid_length_days must be supplied together"
            )
        explicit_spans = basin_shift_days is not None
        source_shift = source_length = None
        if explicit_spans:
            raw_shift = np.asarray(basin_shift_days)
            raw_length = np.asarray(valid_length_days)
            expected_shape = (data.shape[1],)
            if raw_shift.shape != expected_shape or raw_length.shape != expected_shape:
                raise ValueError(
                    "attach_inflow_overlay: explicit shift/length arrays must "
                    f"both have shape {expected_shape}; got "
                    f"{raw_shift.shape} and {raw_length.shape}"
                )
            if not np.issubdtype(raw_shift.dtype, np.integer) or not np.issubdtype(
                raw_length.dtype, np.integer
            ):
                raise TypeError(
                    "attach_inflow_overlay: explicit shift/length arrays must "
                    "contain integers"
                )
            source_shift = raw_shift.astype(np.int64, copy=False)
            source_length = raw_length.astype(np.int64, copy=False)
            if np.any(source_shift < 0) or np.any(source_length < 0):
                raise ValueError(
                    "attach_inflow_overlay: explicit shift/length values must "
                    "be non-negative"
                )
            source_end = source_shift + source_length
            if np.any(source_end < source_shift) or np.any(source_end > data.shape[0]):
                raise ValueError(
                    "attach_inflow_overlay: every explicit valid span must fit "
                    "inside the native data time axis"
                )
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

        # 3. Determine per-column valid spans on the aligned dataset axis.
        #    Without explicit producer metadata, NaN (including alignment
        #    padding) is invalid. With metadata, duplicate source columns use
        #    the intersection because all contributors must be present for an
        #    aggregate to be fully observed.
        C = dst_cids.size
        per_col_shift = np.zeros(C, dtype=np.int64)
        per_col_length = np.zeros(C, dtype=np.int64)
        if explicit_spans:
            data_offset = timedelta_quotient(
                data_start_date - self.start_date,
                self.time_interval,
                duration_label="inflow valid-span offset",
                interval_label="time_interval",
            )
            T_ds = aligned_raw.shape[0]
            for c, cols in enumerate(source_groups):
                native_start = int(source_shift[cols].max())
                native_end = int(
                    (source_shift[cols] + source_length[cols]).min()
                )
                dataset_start = max(0, data_offset + native_start)
                dataset_end = min(T_ds, data_offset + native_end)
                if dataset_start < dataset_end:
                    per_col_shift[c] = dataset_start
                    per_col_length[c] = dataset_end - dataset_start
        else:
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
            if per_col_length[c] > 0 and b in basin_shift:
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
    def time_shift_steps(self) -> Optional[np.ndarray]:
        """Per-mapped-column source-time offset, or ``None`` for no shift."""

        return self._shift_days

    @property
    def inflow_valid_length_days(self) -> Optional[np.ndarray]:
        """Per-column number of leading valid read-axis steps, or ``None``."""

        return self._inflow_valid_length_days

    @property
    def inflow_shift_days(self) -> Optional[np.ndarray]:
        """Per-column basin-coordinated read-axis shift, or ``None``."""

        return self._inflow_shift_days

    @property
    def basin_shift(self) -> dict[int, int]:
        """Return the coordinated read-axis shift keyed by basin ID."""

        return dict(self._basin_shift)

    @property
    def basin_length(self) -> dict[int, int]:
        """Return the coordinated valid length keyed by basin ID."""

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
    model_step: timedelta,
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
        "model_step": model_step,
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

    return MultiVariableDataset(datasets)

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    cast,
)

import numba as _numba
import numpy as np
import torch
from netCDF4 import Dataset
from pydantic import (
    BeforeValidator,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.contracts.temporal import DateLike
from hydroforge.contracts.validation import FrozenMapping, HydroForgeModel
from hydroforge.data.datasets.base import (
    SourceDataset,
    _trusted_source_chunk_payload,
    _TrustedSourceChunk,
    _validated_dataset_index,
    _validated_forcing_shard,
    positive_finite_real,
)
from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.timeline import DatasetTimeline, ReadOp
from hydroforge.data.distributed import _find_indices_in_trusted, is_rank_zero
from hydroforge.data.netcdf import (
    _configure_netcdf_variable_cache,
    _NetCDFReadHandlePool,
    _planned_exported_netcdf_chunk_len,
    _read_netcdf_var_sliced_trusted,
    single_file_key,
)
from hydroforge.data.numeric import (
    canonical_float64,
    canonical_floating_array,
    canonical_ids,
    immutable_array,
)
from hydroforge.serialization.netcdf import (
    DEFAULT_NETCDF_OPTIONS,
    _atomic_netcdf_dataset_trusted,
    _create_netcdf_variable_trusted,
    _prepare_netcdf_variable_options_trusted,
    default_netcdf_options,
    normalize_netcdf_variable_options,
)

logger = logging.getLogger(__name__)


class _ExportedSelectionRequest(HydroForgeModel):
    """Validated point-column selection for an exported Dataset view."""

    desired_catchment_ids: np.ndarray
    time_shift_steps: np.ndarray | None = None

    @field_validator("desired_catchment_ids")
    @classmethod
    def _validate_ids(cls, value: np.ndarray) -> np.ndarray:
        result = canonical_ids(value, label="desired_catchment_ids")
        if np.unique(result).size != result.size:
            raise ValueError("desired_catchment_ids must be unique")
        return immutable_array(result, order="C")

    @field_validator("time_shift_steps")
    @classmethod
    def _validate_shift(cls, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        return _int64_vector(
            value, label="time_shift_steps", expected_shape=(value.size,)
        )

    @model_validator(mode="after")
    def _validate_shape(self):
        if (
            self.time_shift_steps is not None
            and self.time_shift_steps.shape != self.desired_catchment_ids.shape
        ):
            raise ValueError(
                f"time_shift_steps must have shape {self.desired_catchment_ids.shape}"
            )
        return self


class _ExportedSelectionBinding(HydroForgeModel):
    """Bind requested IDs to one validated exported coordinate axis."""

    desired_catchment_ids: np.ndarray
    file_catchment_ids: np.ndarray
    source_name: str

    _positions: np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def _bind(self):
        positions = _find_indices_in_trusted(
            self.desired_catchment_ids,
            self.file_catchment_ids,
        )
        if np.any(positions == -1):
            missing = int(np.sum(positions == -1))
            raise ValueError(
                f"{missing} desired catchments were not found in exported "
                f"file {self.source_name}"
            )
        self._positions = immutable_array(positions, dtype=np.int64, order="C")
        return self

    @property
    def positions(self) -> np.ndarray:
        return self._positions


_WINDOW_LENGTH_CONTEXT = "hydroforge_exported_total_steps"


class _ExportedWindowRequest(HydroForgeModel):
    """Validated immutable window request over one exported time axis."""

    window: int = Field(gt=0, strict=True)
    stride: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def _validate_extent(self, info: ValidationInfo):
        context = info.context
        total_steps = (
            context.get(_WINDOW_LENGTH_CONTEXT)
            if isinstance(context, Mapping)
            else None
        )
        if type(total_steps) is not int:
            raise ValueError("exported window request requires dataset context")
        if self.window > total_steps:
            raise ValueError(
                f"window={self.window} exceeds total time steps {total_steps}"
            )
        return self

    @property
    def resolved_stride(self) -> int:
        return self.window if self.stride is None else self.stride


_WINDOW_SHAPE_CONTEXT = "hydroforge_exported_window_shape"


class _ExportedFilterRequest(HydroForgeModel):
    """Validated boolean filter for an existing window Dataset identity."""

    keep: np.ndarray

    @field_validator("keep")
    @classmethod
    def _validate_keep(
        cls,
        value: np.ndarray,
        info: ValidationInfo,
    ) -> np.ndarray:
        if np.ma.isMaskedArray(value):
            raise ValueError("window filter must not be a masked array")
        mask = np.asarray(value)
        context = info.context
        expected_shape = (
            context.get(_WINDOW_SHAPE_CONTEXT) if isinstance(context, Mapping) else None
        )
        if expected_shape is None:
            raise ValueError("filtered() requires a windowed Dataset")
        if mask.dtype != np.dtype(np.bool_) or mask.shape != expected_shape:
            raise ValueError(
                "window filter must be a boolean array with shape "
                f"{expected_shape}; got {mask.shape}"
            )
        return immutable_array(mask, dtype=np.bool_, order="C")


def _int64_vector(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{label} must not be a masked array")
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}; got {array.shape}")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must contain integers")
    if (
        array.dtype.kind == "u"
        and array.size
        and np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{label} contains a value outside int64 range")
    return immutable_array(array, dtype=np.int64, order="C")


def _quantile_levels(value: Any) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise TypeError("quantiles must not be a masked array")
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("quantiles must be one-dimensional")
    if array.size == 0:
        raise ValueError("quantiles must not be empty")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("quantiles must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError("quantiles must contain only finite values")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("quantiles must lie within [0, 1]")
    result = canonical_float64(array, label="quantiles")
    if np.any(np.diff(result) <= 0):
        raise ValueError("quantiles must be strictly increasing")
    return result


class _QuantileExportRequest(HydroForgeModel):
    out_path: Annotated[Path, BeforeValidator(lambda value: Path(value))]
    quantiles: Annotated[np.ndarray, BeforeValidator(_quantile_levels)]
    var_name: str = Field(min_length=1)
    dtype: Literal["float32", "float64"] = "float32"
    netcdf_options: Annotated[
        FrozenMapping[str, Any], BeforeValidator(normalize_netcdf_variable_options)
    ] = Field(default_factory=default_netcdf_options)
    max_buffer_mb: Annotated[
        float, BeforeValidator(partial(positive_finite_real, label="max_buffer_mb"))
    ] = 4096.0

    _create_options: Mapping[str, Any] = PrivateAttr()

    @model_validator(mode="after")
    def _compile_options(self):
        self._create_options = _prepare_netcdf_variable_options_trusted(
            self.netcdf_options,
            dtype="f4" if self.dtype == "float32" else "f8",
            dimensions=("quantile", "saved_points"),
            name=self.var_name,
        )
        return self

    @property
    def create_options(self) -> Mapping[str, Any]:
        return self._create_options


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


_EXPORTED_READ_LENGTH_CONTEXT = "hydroforge_exported_read_length"


class _ExportedReadWindowQuery(HydroForgeModel):
    """One bounded main-axis read from an exported Dataset identity."""

    base_step: int = Field(ge=0)
    length: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _validate_extent(self, info: ValidationInfo):
        total = (
            info.context.get(_EXPORTED_READ_LENGTH_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if type(total) is not int or total < 1:
            raise ValueError("exported read query requires dataset context")
        if self.base_step + self.length > total:
            raise ValueError(
                "exported read window must satisfy "
                f"0 <= base_step < base_step + length <= {total}"
            )
        return self


class ExportedDataset(SourceDataset):
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
      - selected() returns a view in the desired catchment order
      - shard_forcing validates concatenated (T, C) data without matrix multiplication
      - Each rank can read its own file independently
    """

    supports_time_aggregation: ClassVar[bool] = True
    reusable_expression_reads: ClassVar[bool] = True
    _POINT_DIM: ClassVar[str] = "saved_points"

    base_dir: str | Path
    var_name: str
    prefix: str
    chunk_len: int | None = Field(default=None, strict=True, ge=1)
    suffix: str = "rank0.nc"
    time_to_key: Callable[[DateLike], str] = single_file_key
    coord_name: str = "catchment_id"
    in_memory: bool = False
    unit_factor: float = 1.0
    time_aggregation: str | Mapping[str, str] | None = None
    time_shift_steps: np.ndarray | None = Field(
        default=None,
        repr=False,
        description=("Immutable source-time offset for every selected output column"),
    )
    window_length: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Length of each immutable training window",
    )
    window_starts: np.ndarray | None = Field(
        default=None,
        repr=False,
        description="Immutable selected starts on the main source axis",
    )

    _memory_cache: np.ndarray | dict[str, np.ndarray] | None = PrivateAttr(
        default=None,
    )
    _spin_up_memory_cache: np.ndarray | dict[str, np.ndarray] | None = PrivateAttr(
        default=None,
    )
    _memory_cache_file_indices: np.ndarray | None = PrivateAttr(default=None)
    _shift_day_groups: list | None = PrivateAttr(default=None)
    _column_bbox: tuple[int, int] | None = PrivateAttr(default=None)
    _column_bbox_local_indices: np.ndarray | None = PrivateAttr(default=None)
    _coordinates_cache: tuple[np.ndarray, np.ndarray] | None = PrivateAttr(
        default=None,
    )
    _source_dtype: np.dtype | None = PrivateAttr(default=None)
    _timeline: DatasetTimeline = PrivateAttr()
    _global_times: list[DateLike] = PrivateAttr(default_factory=list)
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

    @field_validator("time_shift_steps")
    @classmethod
    def _validate_time_shift_steps(
        cls,
        value: np.ndarray | None,
        info: ValidationInfo,
    ) -> np.ndarray | None:
        selected = info.data.get("local_indices")
        if value is None:
            return None
        if selected is None:
            raise ValueError("time_shift_steps requires a selected exported Dataset")
        shift = _int64_vector(
            value,
            label="time_shift_steps",
            expected_shape=selected.shape,
        )
        if not np.any(shift):
            return None
        return shift

    @field_validator("window_starts")
    @classmethod
    def _validate_window_starts(
        cls,
        value: np.ndarray | None,
        info: ValidationInfo,
    ) -> np.ndarray | None:
        length = info.data.get("window_length")
        if (value is None) != (length is None):
            raise ValueError(
                "window_length and window_starts must be declared together"
            )
        if value is None:
            return None
        starts = _int64_vector(
            value, label="window_starts", expected_shape=(value.size,)
        )
        if starts.size == 0:
            raise ValueError("window_starts must contain at least one window")
        if np.any(starts < 0) or np.any(np.diff(starts) <= 0):
            raise ValueError(
                "window_starts must be nonnegative and strictly increasing"
            )
        return starts

    @model_validator(mode="after")
    def _inspect_exported_storage(self):
        if (
            self.window_starts is not None
            and int(self.window_starts[-1]) + int(self.window_length)
            > self._temporal_domain.count
        ):
            raise ValueError(
                "window_starts and window_length extend beyond the main "
                f"source axis of {self._temporal_domain.count} steps"
            )
        if self.chunk_len is None:
            key = self.time_to_key(self.start_date)
            if type(key) is not str:
                raise TypeError("time_to_key must return an exact string")
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with self._inspect_source_file(path):
                object.__setattr__(
                    self,
                    "chunk_len",
                    _planned_exported_netcdf_chunk_len(path, self.var_name),
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
        source_paths = tuple(
            Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            for key in sorted(self._timeline.file_times)
        )
        for path in source_paths:
            with self._inspect_source_file(path), Dataset(path, "r") as dataset:
                point_dim = self._infer_point_dim(dataset, path)
                self._validate_shard_coordinates(
                    dataset,
                    path,
                    point_dim,
                )
        self._validate_local_index_extent(
            len(cast(tuple[np.ndarray, np.ndarray], self._coordinates_cache)[0]),
            label="exported catchment axis",
        )
        self._compute_column_bbox_from_indices()
        self._record_source_files(source_paths)
        # ExportedDataset indexing, window sampling and the primary cache use
        # a main-only axis whose row zero is ``start_date``. The shared I/O
        # timeline separately follows the exact source chunks, including any
        # replayed spin-up rows.
        contract = self._temporal_domain
        self._global_times = [
            contract._support_trusted(index)[0] for index in range(contract.count)
        ]

        if self.time_shift_steps is not None:
            self._shift_day_groups = self._compile_groups(
                self.time_shift_steps,
            )

        return self

    def _rebuild(self, **updates: Any) -> ExportedDataset:
        """Derive a view from the already validated storage identity."""

        payload = {name: getattr(self, name) for name in type(self).model_fields}
        payload.update(updates)
        for name in (
            "local_indices",
            "desired_catchment_ids",
            "time_shift_steps",
            "window_starts",
        ):
            value = payload[name]
            if value is None:
                continue
            payload[name] = immutable_array(
                value,
                dtype=np.int64,
                order="C",
            )

        result = type(self).model_construct(**payload)
        result._temporal_domain = self._temporal_domain
        result._chunk_plan = self._chunk_plan
        result._simulation_schedule = self._simulation_schedule
        result._source_file_identities = dict(self._source_file_identities)
        result._coordinates_cache = self._coordinates_cache
        result._source_dtype = self._source_dtype
        result._timeline = self._timeline._rebind_trusted(result)
        result._global_times = list(self._global_times)
        result._read_handles = self._read_handles
        result._compute_column_bbox_from_indices()
        same_selection = self.local_indices is result.local_indices or (
            self.local_indices is not None
            and result.local_indices is not None
            and np.array_equal(self.local_indices, result.local_indices)
        )
        if same_selection:
            result._memory_cache = self._memory_cache
            result._spin_up_memory_cache = self._spin_up_memory_cache
            result._memory_cache_file_indices = self._memory_cache_file_indices
        if result.time_shift_steps is not None:
            result._shift_day_groups = result._compile_groups(
                result.time_shift_steps,
            )
        return result

    @staticmethod
    def _detect_chunk_len(base_dir, prefix, suffix, var_name, start_date, time_to_key):
        """Detect chunk_len from file's NetCDF time chunking."""
        key = time_to_key(start_date)
        if type(key) is not str:
            raise TypeError("time_to_key must return an exact string")
        path = Path(base_dir) / f"{prefix}{key}{suffix}"
        if not path.exists():
            return 24
        return _planned_exported_netcdf_chunk_len(path, var_name)

    @staticmethod
    def _compile_groups(shift: np.ndarray) -> list:
        """Precompute [(shift_val, col_indices), ...] for fast _gather dispatch."""
        if shift.size == 0:
            return []
        order = np.argsort(shift, kind="stable")
        sorted_shift = shift[order]
        boundaries = np.flatnonzero(sorted_shift[1:] != sorted_shift[:-1]) + 1
        return [
            (int(shift[columns[0]]), columns) for columns in np.split(order, boundaries)
        ]

    # -------------------------
    # Coordinates (1D catchment IDs)
    # -------------------------
    def _validated_catchment_ids(
        self,
        value: Any,
        *,
        path: Path,
    ) -> np.ndarray:
        array = canonical_ids(
            value, label=f"Coordinate variable {self.coord_name!r} in {path.name}"
        )
        if np.unique(array).size != array.size:
            raise ValueError(
                f"Coordinate variable {self.coord_name!r} in {path.name} contains duplicate IDs"
            )
        return array

    def _validate_shard_coordinates(
        self,
        dataset: Dataset,
        path: Path,
        point_dim: str | None = None,
    ) -> np.ndarray:
        """Validate one shard's catchment axis against the canonical shard.

        Exported shards are concatenated by time, so accepting a shard whose
        IDs are merely the same set in a different order silently attaches
        values to the wrong catchments.  The first opened shard establishes an
        immutable canonical order; every subsequent shard must match it
        exactly.
        """
        if point_dim is None:
            point_dim = self._infer_point_dim(dataset, path)
        coordinate = dataset.variables[self.coord_name]
        if tuple(coordinate.dimensions) != (point_dim,):
            raise ValueError(
                f"Coordinate variable {self.coord_name!r} in {path.name} "
                f"must have dimensions ({point_dim!r},), got "
                f"{coordinate.dimensions}"
            )
        values = self._validated_catchment_ids(coordinate[:], path=path)
        if self._coordinates_cache is None:
            index = np.arange(values.shape[0], dtype=np.int64)
            self._coordinates_cache = (
                immutable_array(values, order="C"),
                immutable_array(index, order="C"),
            )
        else:
            expected, _ = self._coordinates_cache
            if values.shape != expected.shape or not np.array_equal(values, expected):
                raise ValueError(
                    f"catchment_id coordinates in shard {path.name} do not "
                    "match the canonical shard in content and order"
                )
        return values

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return catchment coordinate arrays.

        Returns (output_coord, index) where:
          - output_coord: linear catchment id array of shape (C,)
          - index: simple 0..C-1 integer array of shape (C,)
        """
        return cast(tuple[np.ndarray, np.ndarray], self._coordinates_cache)

    @property
    def data_size(self) -> int:
        """Return number of catchments in the exported file."""
        if self.local_indices is not None:
            return len(self.local_indices)
        sc, _ = self.get_coordinates()
        return len(sc)

    # -------------------------
    # Reading helpers (T, C)
    # -------------------------
    def _infer_point_dim(self, dataset: Dataset, path: Path) -> str:
        """Validate the canonical exported variable dimensions."""

        variable = dataset.variables[self.var_name]
        source_dtype = np.dtype(variable.dtype)
        if source_dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                f"Variable {self.var_name!r} in {path.name} must use a real "
                f"numeric dtype; got {source_dtype}"
            )
        if self._source_dtype is None:
            self._source_dtype = source_dtype
        elif source_dtype != self._source_dtype:
            raise ValueError(
                f"Variable {self.var_name!r} dtype in {path.name} does not "
                f"match the canonical dtype {self._source_dtype}"
            )
        dimensions = tuple(variable.dimensions)
        expected = ("time", self._POINT_DIM)
        if dimensions != expected:
            raise ValueError(
                f"Variable {self.var_name!r} in {path.name} must have "
                f"dimensions {expected}; got {dimensions}"
            )
        return self._POINT_DIM

    def _compute_column_bbox_from_indices(self) -> None:
        """Compute the minimal saved_points slice for mapped catchments."""
        if self.local_indices is None:
            self._column_bbox = None
            self._column_bbox_local_indices = None
            return
        if self.local_indices.size == 0:
            self._column_bbox = (0, -1)
            self._column_bbox_local_indices = np.empty((0,), dtype=np.int64)
            return

        col_min = int(self.local_indices.min())
        col_max = int(self.local_indices.max())
        self._column_bbox = (col_min, col_max)
        self._column_bbox_local_indices = (self.local_indices - col_min).astype(
            np.int64, copy=False
        )

    def _read_source_columns(self, key, rows, columns, *, reorder=None) -> np.ndarray:
        """Read one canonical (time, point) block and verify its source identity."""
        path = self._checked_source_path(
            Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        )
        with self._read_handles.acquire(path) as dataset:
            variable = dataset.variables[self.var_name]
            selectors = (np.asarray(rows, dtype=np.int64), columns)
            _configure_netcdf_variable_cache(variable, selectors, time_axis=0)
            array = _read_netcdf_var_sliced_trusted(variable, selectors)
            if reorder is not None:
                array = array[:, reorder]
            result = _trusted_source_chunk_payload(
                array, expected_rows=len(rows), clip_negative=self.clip_negative
            )
        self._verify_source_path(path)
        return result

    def _read_ops(self, ops: Sequence[ReadOp]) -> np.ndarray:
        """Read time steps and reorder columns in the dataset's selected order."""
        if not ops:
            return np.empty((0, self.data_size), dtype=self.out_dtype)
        columns = slice(None)
        if self._column_bbox is not None:
            start, end = self._column_bbox
            columns = slice(start, end + 1)
        chunks = [
            self._read_source_columns(
                key, rows, columns, reorder=self._column_bbox_local_indices
            )
            for key, rows in ops
            if rows
        ]
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def _finish_read(self, data: np.ndarray):
        if self.time_aggregation is None and self.unit_factor == 1.0:
            return self._finalize_output_data(
                data,
                label="exported dataset output",
            )
        data = self._canonical_calculation_data(
            data,
            label="exported dataset input",
        )
        if self.time_aggregation is not None:
            data = self._apply_time_aggregation(
                data,
                self._timeline.source_time_interval,
                self.time_aggregation,
            )
        if self.unit_factor != 1.0:
            for block in data.values() if isinstance(data, dict) else (data,):
                np.divide(block, self.unit_factor, out=block)
        return self._finalize_output_data(data, label="exported dataset output")

    def _as_cache_data(
        self,
        data: np.ndarray | dict[str, np.ndarray],
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Normalize one processed cache while preserving aggregation maps."""
        if isinstance(data, dict):
            return {
                name: np.ascontiguousarray(block.astype(self.out_dtype, copy=False))
                for name, block in data.items()
            }
        return np.ascontiguousarray(data.astype(self.out_dtype, copy=False))

    @staticmethod
    def _cache_column_count(
        cache: np.ndarray | dict[str, np.ndarray],
    ) -> int:
        if isinstance(cache, dict):
            first = next(iter(cache.values()))
            return int(first.shape[1])
        return int(cache.shape[1])

    @staticmethod
    def _cache_shape(cache: np.ndarray | dict[str, np.ndarray]):
        if isinstance(cache, dict):
            return {name: block.shape for name, block in cache.items()}
        return cache.shape

    @staticmethod
    def _cache_nbytes(cache: np.ndarray | dict[str, np.ndarray]) -> int:
        if isinstance(cache, dict):
            return sum(block.nbytes for block in cache.values())
        return cache.nbytes

    def _source_element_bytes(
        self,
        ops: Sequence[ReadOp],
    ) -> int:
        """Return a conservative element width for NetCDF source reads."""

        element_bytes = np.dtype(self.out_dtype).itemsize
        for key in dict.fromkeys(key for key, _ in ops):
            path = self._checked_source_path(
                Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}",
            )
            with self._read_handles.acquire(path) as dataset:
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
                    # Missing integer values are promoted to float64 by the
                    # source payload boundary.
                    element_bytes = max(element_bytes, 8)
                element_bytes = max(element_bytes, read_dtype.itemsize)
            self._verify_source_path(path)
        return element_bytes

    def _read_chunk(self, chunk: SourceChunk) -> _TrustedSourceChunk:
        return _TrustedSourceChunk(
            self._finish_read(
                self._read_ops(self._timeline.read_for_chunk(chunk).operations)
            )
        )

    def close(self) -> None:
        """Close this process's persistent NetCDF read handles."""

        self._read_handles.close()

    def selected(
        self,
        desired_catchment_ids: np.ndarray,
        *,
        time_shift_steps: np.ndarray | None = None,
    ) -> ExportedDataset:
        """Return a validated immutable column/temporal selection.

        Parameters
        ----------
        desired_catchment_ids : np.ndarray, shape (C,)
            Catchment ids in the order consumers want.
        time_shift_steps : np.ndarray, shape (C,), optional
            Explicit integer source-time offset for every mapped column.
            This is useful when a composed Dataset has coordinated observation
            spans before constructing this view.
        """
        request = _ExportedSelectionRequest(
            desired_catchment_ids=desired_catchment_ids,
            time_shift_steps=time_shift_steps,
        )
        return self._selected_trusted(
            desired_catchment_ids=request.desired_catchment_ids,
            time_shift_steps=request.time_shift_steps,
        )

    def _selected_trusted(
        self,
        *,
        desired_catchment_ids: np.ndarray,
        time_shift_steps: np.ndarray | None,
    ) -> ExportedDataset:
        """Materialize one already validated exported-column selection."""

        # Reuse the validated coordinate cache populated on first access.
        key = sorted(self._timeline.file_times)[0]
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        file_catchment_ids, _ = self.get_coordinates()

        binding = _ExportedSelectionBinding(
            desired_catchment_ids=desired_catchment_ids,
            file_catchment_ids=file_catchment_ids,
            source_name=path.name,
        )
        local_indices = binding.positions
        if is_rank_zero():
            logger.info(
                "Mapped %d catchments from %d in exported file",
                len(desired_catchment_ids),
                len(file_catchment_ids),
            )

        return self._rebuild(
            local_indices=local_indices,
            desired_catchment_ids=desired_catchment_ids,
            time_shift_steps=time_shift_steps,
            window_length=None,
            window_starts=None,
        )

    def load_to_memory(self) -> None:
        """Load all data into memory for faster repeated access.

        This method reads the entire dataset into a numpy array cached in memory,
        covering ALL files that span the [start_date, end_date] range.
        Subsequent __getitem__ calls will return slices from this cache instead
        of reading from disk.

        Selection is part of this Dataset identity, so each selected Dataset
        owns an independent derived cache in its validated column order.
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
        main_ops = self._timeline.operations_for_times(self._global_times)
        all_data = self._finish_read(self._read_ops(main_ops))

        # Spin-up can be disjoint from the main period by years.  Keep one
        # compact cache for the unique spin-up interval instead of indexing a
        # main-only cache with offsets relative to ``start_date`` (which used
        # to zero-fill valid spin-up data), or allocating the intervening gap.
        spin_data = None
        spinup = self._temporal_domain.spinup
        if spinup is not None:
            spin_count = self.chunk_plan.spinup_source_count_per_cycle
            spin_times = [
                spinup.source_start + self.time_interval * index
                for index in range(spin_count)
            ]
            spin_ops = self._timeline.operations_for_times(spin_times)
            spin_data = self._finish_read(self._read_ops(spin_ops))

        # Store in cache with correct dtype and C-contiguous layout.
        memory_cache = self._as_cache_data(all_data)
        spin_up_memory_cache = (
            None if spin_data is None else self._as_cache_data(spin_data)
        )
        if self.local_indices is None:
            memory_cache_file_indices = np.arange(
                self._cache_column_count(memory_cache), dtype=np.int64
            )
        else:
            memory_cache_file_indices = self.local_indices.copy()

        self._spin_up_memory_cache = spin_up_memory_cache
        self._memory_cache_file_indices = memory_cache_file_indices
        self._memory_cache = memory_cache

        if is_rank_zero():
            n_files = len(main_ops)
            mem_bytes = self._cache_nbytes(self._memory_cache)
            if self._spin_up_memory_cache is not None:
                mem_bytes += self._cache_nbytes(self._spin_up_memory_cache)
            logger.info(
                "Loaded exported data shape=%s, spin_up_shape=%s from %d "
                "file(s) (%.1f MiB)",
                self._cache_shape(self._memory_cache),
                None
                if self._spin_up_memory_cache is None
                else self._cache_shape(self._spin_up_memory_cache),
                n_files,
                mem_bytes / (1024 * 1024),
            )

    def _export_quantiles(
        self,
        out_path: str | Path,
        quantiles: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        var_name: str | None = None,
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

        A selected Dataset writes its validated catchment order; an unselected
        Dataset writes the file's native order.

        Exact quantile computation requires the full time series per catchment.
        When the estimated source working set exceeds ``max_buffer_mb``,
        catchments are processed in column-batches. The estimate reserves three
        arrays on the expanded source time axis using the widest source/output
        element width. A budget smaller than one estimated column is rejected
        before reading data or creating output. Already-resident data is reused
        without new source reads. This is a source-buffer planning limit, not a
        bound on library caches, quantile-result arrays, or total process memory.

        Args:
            out_path: Output NetCDF file path.
            quantiles: Sequence of quantile levels in [0, 1].
            var_name: Variable name in output file (default: ``self.var_name``).
            dtype: Output data type.
            netcdf_options: Validated NetCDF variable-creation options.
            max_buffer_mb: Source working-set planning budget in MiB.
                Column batching must fit at least one estimated source column.
                Default 4096 (4 GiB).

        Returns:
            Path to the created NetCDF file.
        """
        if isinstance(self.time_aggregation, Mapping):
            raise ValueError(
                "export_quantiles requires one time-aggregation result; "
                "create a single-result ExportedDataset first"
            )
        request = _QuantileExportRequest(
            out_path=out_path,
            quantiles=quantiles,
            dtype=dtype,
            var_name=self.var_name if var_name is None else var_name,
            netcdf_options=netcdf_options,
            max_buffer_mb=max_buffer_mb,
        )
        out_path, dtype, resolved_var_name = (
            request.out_path,
            request.dtype,
            request.var_name,
        )
        max_buffer_mb, quantiles_arr = request.max_buffer_mb, request.quantiles
        Q = len(quantiles_arr)

        # ---- catchment IDs (respecting column reorder) ----
        file_catchment_ids, _ = self.get_coordinates()
        if self.local_indices is not None:
            catchment_ids = file_catchment_ids[self.local_indices]
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
            main_ops = self._timeline.operations_for_times(self._global_times)
            source_rows = sum(len(abs_indices) for _, abs_indices in main_ops)
            read_elem_bytes = self._source_element_bytes(main_ops)
            # Reading, concatenation/aggregation, and exact quantile
            # selection can briefly coexist. Three element-widths is a
            # conservative working-set estimate; using one array's size here
            # would advertise a buffer limit that NumPy immediately exceeds.
            working_elem_bytes = 3 * read_elem_bytes
            full_size = source_rows * C_total * working_elem_bytes
            fits_in_memory = full_size <= max_buffer_bytes

        if not fits_in_memory:
            # ``operations_for_times`` expands every requested output time into the
            # source frames required by temporal aggregation.  Reusing these
            # operations keeps this path identical to ``read_chunk`` and
            # ``load_to_memory``; direct timestamp lookup would silently skip
            # the rest of each aggregation window.
            # Column-batch mode: compute batch_size (num catchments per batch)
            # from the expanded source axis, since aggregation may require
            # several source rows for every output row.
            rows_per_batch = max(1, source_rows)
            minimum_column_bytes = rows_per_batch * working_elem_bytes
            if max_buffer_bytes < minimum_column_bytes:
                raise ValueError(
                    "max_buffer_mb is too small for one quantile source column; "
                    f"requires at least {minimum_column_bytes} bytes "
                    f"({minimum_column_bytes / 1024**2:g} MiB)"
                )
            batch_size = int(max_buffer_bytes / minimum_column_bytes)
            n_batches = (C_total + batch_size - 1) // batch_size
            if is_rank_zero():
                logger.info(
                    "Exported dataset %.1f GB exceeds %.0f MiB buffer; "
                    "processing %d catchments in %d batches of %d",
                    full_size / 1e9,
                    max_buffer_mb,
                    C_total,
                    n_batches,
                    batch_size,
                )

        # All arguments and the read plan are valid before touching the output
        # directory. Atomic NetCDF creation then protects against read or
        # numerical failures while computing individual batches.
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- create output NetCDF ----
        dtype_nc = "f4" if dtype == "float32" else "f8"
        create_options = request.create_options
        with _atomic_netcdf_dataset_trusted(
            out_path,
            format="NETCDF4",
        ) as out_ds:
            out_ds.createDimension("quantile", Q)
            out_ds.createDimension("saved_points", C_total)

            q_var = out_ds.createVariable("quantile", "f8", ("quantile",))
            q_var[:] = quantiles_arr
            q_var.long_name = "quantile level"

            cid_var = out_ds.createVariable("catchment_id", "i8", ("saved_points",))
            cid_var[:] = catchment_ids

            data_var = _create_netcdf_variable_trusted(
                out_ds,
                resolved_var_name,
                dtype_nc,
                ("quantile", "saved_points"),
                options=create_options,
            )
            data_var.long_name = f"{resolved_var_name} quantile values"

            if fits_in_memory:
                # ---- fits in memory: load full series once, compute quantiles ----
                if self._memory_cache is None:
                    self.load_to_memory()
                all_data = self._memory_cache[:T_total]
                if not np.isfinite(all_data).all():
                    raise ValueError(
                        "resident quantile source contains non-finite values"
                    )
                q_values = np.quantile(all_data, quantiles_arr, axis=0)  # (Q, C)
                data_var[:] = canonical_floating_array(
                    q_values,
                    dtype=dtype,
                    label="quantile result",
                )
            else:
                # ---- too large: batch by catchments (columns) ----
                # Exact quantile needs full time axis, so we read ALL time steps
                # for a subset of catchments per batch.
                if main_ops is None:
                    raise RuntimeError("column-batch read plan was not built")
                for c_start in range(0, C_total, batch_size):
                    c_end = min(c_start + batch_size, C_total)
                    batch_cols = slice(c_start, c_end)

                    if self.local_indices is not None:
                        file_col_indices = self.local_indices[c_start:c_end]
                    else:
                        file_col_indices = np.arange(c_start, c_end, dtype=np.int64)

                    file_chunks = [
                        self._read_source_columns(key, rows, file_col_indices)
                        for key, rows in main_ops
                    ]
                    all_batch = (
                        np.concatenate(file_chunks, axis=0)
                        if len(file_chunks) > 1
                        else file_chunks[0]
                    )
                    file_chunks.clear()
                    processed_batch = self._finish_read(all_batch)
                    if not np.isfinite(processed_batch).all():
                        raise ValueError(
                            "quantile source batch contains non-finite values"
                        )
                    q_batch = np.quantile(
                        processed_batch,
                        quantiles_arr,
                        axis=0,
                        overwrite_input=True,
                    )
                    data_var[:, batch_cols] = canonical_floating_array(
                        q_batch,
                        dtype=dtype,
                        label="quantile result",
                    )
                    # Python loop locals retain the previous batch unless
                    # explicitly released. Drop every large array before the
                    # next read so the three-array working-set estimate above
                    # remains conservative across batch boundaries.
                    del q_batch, processed_batch, all_batch

        if is_rank_zero():
            logger.info(
                "Saved quantiles to %s: levels=%s, shape=(%d, %d)",
                out_path,
                quantiles_arr.tolist(),
                Q,
                C_total,
            )

        return out_path

    def shard_forcing(
        self,
        chunk_data: Any,
    ) -> Any:
        """Validate already-concatenated ``(T, C)`` forcing.

        For ExportedDataset, data is already in the validated column order
        owned by this Dataset identity, so no matrix multiply is needed.
        """
        return self._shard_forcing_trusted(
            self._validate_forcing_shard(chunk_data),
        )

    def _validate_forcing_shard(self, chunk_data: Any) -> Any:
        """Canonicalize one new forcing batch at the public boundary."""

        dtype = torch.float32 if self.out_dtype == "float32" else torch.float64
        return _validated_forcing_shard(
            chunk_data,
            columns=self.data_size,
            dtype=dtype,
            device=None,
            allow_sequence=True,
        )

    @staticmethod
    def _shard_forcing_trusted(chunk_data: Any) -> Any:
        """Consume an already canonical exported forcing batch."""

        return chunk_data

    # -------------------------
    # Override __getitem__ - no rank gating for exported data
    # -------------------------
    def _gather_cache(
        self,
        cache: np.ndarray | dict[str, np.ndarray],
        shift: np.ndarray | None,
        base_t: int,
        length: int,
        *,
        groups: list | None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if isinstance(cache, dict):
            return {
                name: self._gather(
                    block,
                    shift,
                    base_t,
                    length,
                    groups=groups,
                )
                for name, block in cache.items()
            }
        return self._gather(cache, shift, base_t, length, groups=groups)

    def _cached_or_disk_chunk(
        self,
        chunk: SourceChunk,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Interpret one planned request through cache or NetCDF storage."""

        if self._memory_cache is None and (
            self.in_memory or self.time_shift_steps is not None
        ):
            self.load_to_memory()
        cache = (
            self._spin_up_memory_cache
            if chunk.phase == "spinup"
            else self._memory_cache
        )
        if self._memory_cache is not None:
            return self._gather_cache(
                cache,
                self.time_shift_steps,
                chunk.phase_offset,
                chunk.length,
                groups=self._shift_day_groups,
            )
        return self._read_chunk_trusted(chunk)

    def _get_chunk_trusted(self, chunk: SourceChunk):
        """Return one framework-produced request, including time shifts."""

        runoff = self._cached_or_disk_chunk(chunk)
        return self._apply_upsampling_policy(runoff)

    def read_window(self, base_step: int, length: int):
        """Read one validated main-axis window from this Dataset identity."""

        query = _ExportedReadWindowQuery.model_validate(
            {"base_step": base_step, "length": length},
            context={_EXPORTED_READ_LENGTH_CONTEXT: len(self._global_times)},
        )

        return self._read_window_trusted(query.base_step, query.length)

    def _read_window_trusted(self, base_step: int, length: int):
        """Read one compiler-produced main-axis window without revalidation."""

        self.load_to_memory()
        runoff = self._gather_cache(
            self._memory_cache,
            self.time_shift_steps,
            base_step,
            length,
            groups=self._shift_day_groups,
        )
        return self._apply_upsampling_policy(runoff)

    def __getitem__(self, idx: int):
        """Fetch one planned chunk or one explicit training window."""

        idx = _validated_dataset_index(self, idx)
        if self.window_starts is None:
            return self._get_chunk_trusted(self.chunk_plan._at_trusted(idx))

        return self._read_window_trusted(
            int(self.window_starts[idx]),
            int(self.window_length),
        )

    @staticmethod
    def _gather(
        data: np.ndarray,
        shift: np.ndarray | None,
        base_t: int,
        length: int,
        oob_fill: float = 0.0,
        *,
        groups: list | None = None,
    ) -> np.ndarray:
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
                out[lo - base_t : hi - base_t] = data[lo:hi]
            return out
        if C >= _NUMBA_C_THRESHOLD:
            return _gather_nb_kernel(data, shift, base_t, length, float(oob_fill))
        out = np.full((length, C), oob_fill, dtype=data.dtype)
        if groups is None:
            groups = ExportedDataset._compile_groups(shift)
        for s, cols in groups:
            src_lo = base_t + s
            clip_lo = max(src_lo, 0)
            clip_hi = min(src_lo + length, T)
            if clip_lo >= clip_hi:
                continue
            out[clip_lo - src_lo : clip_hi - src_lo, cols] = data[clip_lo:clip_hi, cols]
        return out

    def __len__(self) -> int:
        """Window mode length, or chunk-based length."""
        if self.window_starts is not None:
            return int(self.window_starts.size)
        return super().__len__()

    def windowed(
        self,
        window: int,
        stride: int | None = None,
    ) -> ExportedDataset:
        """Return an immutable shifted-window Dataset identity.

        ``self[idx]`` returns ``(window, C)`` covering
        ``[starts[idx], starts[idx] + window)`` on the shifted time axis,
        where ``starts = np.arange(0, T - window + 1, stride)``.
        Combined with DataLoader ``shuffle=True`` this gives randomized
        training windows.  Compatible with per-catchment shift and with
        the inflow overlay.
        """
        T = len(self._global_times)
        request = _ExportedWindowRequest.model_validate(
            {"window": window, "stride": stride},
            context={_WINDOW_LENGTH_CONTEXT: T},
        )
        resolved_stride = request.resolved_stride
        starts = np.arange(
            0,
            T - request.window + 1,
            resolved_stride,
            dtype=np.int64,
        )
        if is_rank_zero():
            logger.info(
                "Enabled window sampling: window=%d, stride=%d, windows=%d, "
                "time_steps=%d",
                request.window,
                resolved_stride,
                starts.size,
                T,
            )
        # Window reads are intentionally memory-backed.  Materialize once in
        # the creating process so fork-based DataLoader workers inherit the
        # cache copy-on-write instead of re-reading the full NetCDF source.
        self.load_to_memory()
        self.close()
        return self._rebuild(
            window_length=request.window,
            window_starts=starts,
        )

    def filtered(self, keep: np.ndarray) -> ExportedDataset:
        """Return a Dataset retaining the selected immutable windows."""

        request = _ExportedFilterRequest.model_validate(
            {"keep": keep},
            context={
                _WINDOW_SHAPE_CONTEXT: (
                    None if self.window_starts is None else self.window_starts.shape
                ),
            },
        )
        return self._rebuild(
            window_starts=self.window_starts[request.keep],
        )

    # -------------------------


# ---------------------------------------------------------------------------
# Composite multi-variable wrapper
# ---------------------------------------------------------------------------
class _OpenMultivariableExportedRequest(HydroForgeModel):
    base_dir: str | Path
    var_specs: Any
    start_date: DateLike
    end_date: DateLike
    time_interval: timedelta = timedelta(days=1)
    model_step: timedelta
    calendar: str | None = None
    spin_up_cycles: int = Field(default=0, strict=True, ge=0)
    spin_up_start_date: DateLike | None = None
    spin_up_end_date: DateLike | None = None
    chunk_len: int | None = Field(default=None, ge=1, strict=True)
    time_to_key: Callable[[datetime], str] | None = None
    coord_name: str = "catchment_id"
    in_memory: bool = Field(default=False, strict=True)

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


def open_multivariable_exported(
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
    time_to_key: Callable[[datetime], str] | None = None,
    coord_name: str = "catchment_id",
    in_memory: bool = False,
):
    """Open aligned catchment variables as one generic composite."""
    from hydroforge.data.datasets.multivariable import (
        ExportedMultiVariableDataset,
    )

    request = _OpenMultivariableExportedRequest(
        base_dir=base_dir,
        var_specs=var_specs,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval,
        model_step=model_step,
        calendar=calendar,
        spin_up_cycles=spin_up_cycles,
        spin_up_start_date=spin_up_start_date,
        spin_up_end_date=spin_up_end_date,
        chunk_len=chunk_len,
        time_to_key=time_to_key,
        coord_name=coord_name,
        in_memory=in_memory,
    )
    shared = {
        "base_dir": request.base_dir,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "time_interval": request.time_interval,
        "model_step": request.model_step,
        "calendar": request.calendar,
        "spin_up_cycles": request.spin_up_cycles,
        "spin_up_start_date": request.spin_up_start_date,
        "spin_up_end_date": request.spin_up_end_date,
        "coord_name": request.coord_name,
        "in_memory": request.in_memory,
    }
    if request.time_to_key is not None:
        shared["time_to_key"] = request.time_to_key
    if request.chunk_len is not None:
        shared["chunk_len"] = request.chunk_len
    else:
        first_name, first_spec = request.compiled_specs[0]
        first_prefix = first_spec.get("prefix", f"{first_name}_")
        first_suffix = first_spec.get("suffix", "rank0.nc")
        first_time_to_key = (
            request.time_to_key if request.time_to_key is not None else single_file_key
        )
        shared["chunk_len"] = ExportedDataset._detect_chunk_len(
            request.base_dir,
            first_prefix,
            first_suffix,
            first_name,
            request.start_date,
            first_time_to_key,
        )
    datasets = {}
    for name, spec in request.compiled_specs:
        options = shared | spec
        options["var_name"] = name
        if "prefix" not in options:
            options["prefix"] = f"{name}_"
        datasets[name] = ExportedDataset(**options)

    return ExportedMultiVariableDataset(datasets=datasets)

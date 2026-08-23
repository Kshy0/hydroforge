# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, Literal, Optional, Self, Tuple

import cftime
import numpy as np
from pydantic import PrivateAttr, field_validator, model_validator

from hydroforge.data.datasets.base import (
    _TrustedSourceChunk,
    _trusted_source_chunk_payload,
    positive_finite_real,
)
from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.gridded import GriddedDataset
from hydroforge.data.netcdf import daily_time_to_key, single_file_key
from hydroforge.contracts.temporal import (
    DateLike, _timedelta_quotient_trusted,
)
from hydroforge.contracts.validation import _immutable_dict


FileStartDate = (
    DateLike
    | Mapping[str, DateLike]
    | Callable[[str], DateLike]
)


class DailyBinDataset(GriddedDataset):
    """
    Dataset class that reads daily binary files.

    By default each binary file contains one day's data, with filenames
    ``{prefix}{YYYYMMDD}{suffix}``.  The ``time_to_key`` callable controls
    the mapping from date to filename key:

    * **One file per day** (default): ``time_to_key = daily_time_to_key``
      → every date gets a unique key, each file has one frame.
    * **Grouped/single file**: provide ``file_start_date`` (or a key→date
    mapping/callback) to identify frame zero in each file.  Requested dates
    are mapped to their absolute offset from that origin; they are never
    renumbered from zero merely because a run requests a subset of dates.
    """

    base_dir: str | Path
    shape: tuple[int, int]
    prefix: str
    unit_factor: float = 1.0
    bin_dtype: str = "float32"
    suffix: str = ".one"
    out_dtype: Literal["float32", "float64"] = "float32"
    lat_south_to_north: bool = False
    lon_0_to_360: bool = False
    time_to_key: Callable[[DateLike], str] | None = daily_time_to_key
    file_start_date: FileStartDate | None = None

    _storage_dtype: np.dtype = PrivateAttr()
    _key_cache: dict[DateLike, str] = PrivateAttr(default_factory=dict)
    _daily_layout: bool = PrivateAttr(default=False)
    _dt_to_loc: dict[DateLike, tuple[str, int]] = PrivateAttr(
        default_factory=dict,
    )

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, shape: tuple[int, int]) -> tuple[int, int]:
        if len(shape) != 2:
            raise ValueError("shape must contain exactly two dimensions")
        if any(type(extent) is not int or extent < 1 for extent in shape):
            raise ValueError("shape values must be exact positive ints")
        return shape

    @field_validator("unit_factor")
    @classmethod
    def _validate_unit_factor(cls, value: float) -> float:
        return positive_finite_real(value, label="unit_factor")

    @field_validator("time_to_key", mode="before")
    @classmethod
    def _normalize_time_to_key(cls, value):
        return single_file_key if value is None else value

    @field_validator("bin_dtype")
    @classmethod
    def _validate_bin_dtype(cls, value: str) -> str:
        storage_dtype = np.dtype(value)
        if storage_dtype.kind not in {"i", "u", "f"}:
            raise ValueError("bin_dtype must describe a real numeric dtype")
        return value

    @model_validator(mode="after")
    def _validate_binary_layout(self) -> Self:
        if self.time_interval != timedelta(days=1):
            raise ValueError("DailyBinDataset time_interval must be one day")
        if self.chunk_len != 1:
            raise ValueError("DailyBinDataset chunk_len must be 1")
        configured = self.file_start_date
        if configured is not None and not (
            isinstance(configured, (datetime, cftime.datetime, Mapping))
            or callable(configured)
        ):
            raise ValueError(
                "file_start_date must be a datetime, mapping, callable, or None"
            )
        if isinstance(configured, Mapping):
            invalid = {
                key: type(value).__name__
                for key, value in configured.items()
                if (
                    type(key) is not str
                    or not key
                    or not isinstance(value, (datetime, cftime.datetime))
                )
            }
            if invalid:
                raise ValueError(
                    "file_start_date mappings require non-empty exact string "
                    f"keys and datetime values: {invalid}"
                )
            object.__setattr__(
                self,
                "file_start_date",
                _immutable_dict(configured),
            )
        return self

    @model_validator(mode="after")
    def _inspect_binary_storage(self):
        self._storage_dtype = np.dtype(self.bin_dtype)
        self._validate_local_index_extent(
            self.shape[0] * self.shape[1], label="binary grid",
        )
        self._build_file_mapping()
        self._inspect_required_files()
        return self

    def _build_file_mapping(self):
        """Map each simulation date to ``(file_key, absolute frame index)``."""
        dates = {
            timestamp
            for chunk in self.chunk_plan
            for timestamp in chunk._source_times()
        }

        by_key: dict[str, list] = {}
        for dt in dates:
            by_key.setdefault(self._storage_key(dt), []).append(dt)
        daily_layout = (
            all(
                len(key_dates) == 1
                and key == daily_time_to_key(key_dates[0])
                for key, key_dates in by_key.items()
            )
            and len(by_key) == len(dates)
        )
        self._daily_layout = daily_layout
        # Only the canonical one-file-per-day layout has an implicit frame
        # zero. Any grouped/custom layout (including a one-date subset of a
        # constant file) needs an explicit origin.
        if daily_layout and self.file_start_date is not None:
            raise ValueError(
                "file_start_date must be None for one-file-per-day binary "
                "layouts"
            )
        if not daily_layout and self.file_start_date is None:
            raise ValueError(
                "file_start_date is required for grouped or custom binary "
                "file layouts"
            )

        locations: dict[DateLike, tuple[str, int]] = {}
        for key, key_dates in by_key.items():
            ordered = sorted(key_dates)
            if daily_layout:
                for dt in ordered:
                    locations[dt] = (key, 0)
                continue
            origin = self._file_origin(key)
            for dt in ordered:
                frame_idx = _timedelta_quotient_trusted(
                    dt - origin,
                    self.time_interval,
                    duration_label=(
                        f"file frame offset for key {key!r}"
                    ),
                    interval_label="daily binary time_interval",
                )
                if frame_idx < 0:
                    raise ValueError(
                        f"date {dt!s} precedes file_start_date {origin!s} "
                        f"for binary file key {key!r}"
                    )
                locations[dt] = (key, frame_idx)
        self._dt_to_loc = locations

    def _storage_key(self, timestamp: DateLike) -> str:
        key = self.time_to_key(timestamp)
        if type(key) is not str:
            raise ValueError(
                "DailyBinDataset time_to_key must return an exact string"
            )
        previous = self._key_cache.setdefault(timestamp, key)
        if previous != key:
            raise ValueError(
                "DailyBinDataset time_to_key must be deterministic; "
                f"{timestamp} mapped to both {previous!r} and {key!r}"
            )
        return key

    def _file_origin(self, key: str) -> DateLike:
        """Resolve and validate the explicit origin for one storage key."""
        configured = self.file_start_date
        if isinstance(configured, Mapping):
            try:
                origin = configured[key]
            except KeyError as error:
                raise ValueError(
                    f"file_start_date has no origin for binary file key {key!r}"
                ) from error
        elif callable(configured):
            origin = configured(key)
        else:
            origin = configured
        if not isinstance(origin, (datetime, cftime.datetime)):
            raise ValueError(
                "file_start_date values must be datetime or cftime datetime"
            )
        return self._require_calendar_datetime(
            origin,
            label=f"file_start_date for key {key!r}",
        )

    def _inspect_required_files(self):
        """Validate that all required files exist and match expected size."""
        required_paths = set()
        for key, _frame_idx in self._dt_to_loc.values():
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            required_paths.add(path)
        # Validate file sizes are consistent with shape
        ny, nx = self.shape
        frame_bytes = ny * nx * self._storage_dtype.itemsize
        for fp in required_paths:
            file_bytes = Path(fp).stat().st_size
            if file_bytes % frame_bytes != 0:
                raise ValueError(
                    f"File size mismatch: {fp} is {file_bytes} bytes, "
                    f"but shape={self.shape} dtype={self.bin_dtype} expects "
                    f"multiples of {frame_bytes} bytes "
                    f"(got {file_bytes / frame_bytes:.4f} frames). "
                    f"Check the 'shape' parameter."
                )
            observed_frames = file_bytes // frame_bytes
            if self._daily_layout and observed_frames != 1:
                raise ValueError(
                    f"Daily binary file {fp} must contain exactly one frame; "
                    f"found {observed_frames}"
                )
        self._record_source_files(required_paths)

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) coordinate arrays.

        Note: shape is (ny, nx) = (lat, lon), so shape[0] is lat size, shape[1] is lon size.
        Coordinates are cell centers, computed from shape assuming global coverage.

        If lat_south_to_north is True, latitude goes from -90 to 90 (south to north).
        Otherwise, latitude goes from 90 to -90 (north to south, default).

        If lon_0_to_360 is True, longitude goes from 0 to 360 (e.g. ERA5-Land binary).
        Otherwise, longitude goes from -180 to 180 (default).
        """
        ny, nx = self.shape
        # Resolution in degrees
        res_lat = 180.0 / ny
        res_lon = 360.0 / nx
        # Cell centers
        if self.lat_south_to_north:
            lat = np.linspace(-90 + res_lat / 2, 90 - res_lat / 2, ny)
        else:
            lat = np.linspace(90 - res_lat / 2, -90 + res_lat / 2, ny)
        if self.lon_0_to_360:
            lon = np.linspace(res_lon / 2, 360 - res_lon / 2, nx)
        else:
            lon = np.linspace(-180 + res_lon / 2, 180 - res_lon / 2, nx)
        return lon, lat

    def _read_chunk(self, chunk: SourceChunk) -> _TrustedSourceChunk:
        """Read one day's data from binary file.

        Returns:
        - If local_indices is set: (1, N) compressed array
        - If local_indices is None: (1, Y, X) full grid array

        Spatial convention: (Y, X) = (lat, lon), C-order flatten (lon varies fastest)
        """
        key, frame_idx = self._dt_to_loc[chunk.source_start]
        filename = f"{self.prefix}{key}{self.suffix}"
        file_path = self._checked_source_path(
            Path(self.base_dir) / filename,
        )

        ny, nx = self.shape
        frame_size = ny * nx
        element_size = self._storage_dtype.itemsize

        data = np.fromfile(
            file_path, dtype=self._storage_dtype,
            count=frame_size, offset=frame_idx * frame_size * element_size,
        )
        self._verify_source_path(file_path)
        data = _trusted_source_chunk_payload(
            data.reshape(1, ny, nx),
            expected_rows=1,
            clip_negative=self.clip_negative,
        )
        data = self._canonical_calculation_data(
            data, label="daily binary dataset input",
        )
        np.divide(data, self.unit_factor, out=data)
        data = self._finalize_output_data(
            data, label="daily binary dataset output",
        )

        if self.local_indices is not None:
            result = data.reshape(1, frame_size)[:, self.local_indices]
        else:
            result = data
        return _TrustedSourceChunk(result)

    def _get_first_frame_nan_mask(self) -> Optional[np.ndarray]:
        key, frame_idx = self._dt_to_loc[self.start_date]
        file_path = self._checked_source_path(
            Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}",
        )

        ny, nx = self.shape
        frame_size = ny * nx
        element_size = self._storage_dtype.itemsize
        data = np.fromfile(
            file_path,
            dtype=self._storage_dtype,
            count=frame_size,
            offset=frame_idx * frame_size * element_size,
        )
        self._verify_source_path(file_path)
        data = data.reshape(ny, nx)
        if not np.issubdtype(data.dtype, np.floating):
            return np.zeros((ny, nx), dtype=bool)
        return np.isnan(data)

    def close(self):
        pass

    def __len__(self):
        """
        Returns the total number of samples in the dataset based on the time range.
        """
        return super().__len__()

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import cftime
import numpy as np
from pydantic import Field, PrivateAttr, field_validator

from hydroforge.core.arrays import immutable_array
from hydroforge.core.time import (
    DateLike,
    require_calendar,
    require_date,
    timedelta_quotient,
)
from hydroforge.core.validation import FrozenMapping
from hydroforge.data.datasets.base import ForcingDataset, SourceDirectory
from hydroforge.data.datasets.keys import daily_time_to_key, single_file_key
from hydroforge.data.datasets.plan import DatasetPlan, TemporalDomain
from hydroforge.data.datasets.space import GridSpace
from hydroforge.data.datasets.storage import SOURCE_FILE_LABEL, UnitFactor
from hydroforge.data.datasets.timeline import StorageLayout
from hydroforge.data.datasets.values import MissingPolicy, convert
from hydroforge.io.files import SourceFiles

FileStartDate = (
    DateLike | FrozenMapping[Annotated[str, Field(min_length=1)], DateLike] | Callable
)
_FrameRun = tuple[str, int, int]


class DailyBinDataset(ForcingDataset):
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

    Consecutive frames of one file are read by a single ``np.fromfile``; a
    mapped view keeps only its source cells before any value check, so cells
    outside it (for example ocean NaN) never affect the result.  Missing (NaN)
    values become zero unless ``missing="error"``.
    """

    base_dir: SourceDirectory
    shape: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]]
    prefix: str
    unit_factor: UnitFactor = 1.0
    bin_dtype: str = "float32"
    suffix: str = ".one"
    lat_south_to_north: bool = False
    lon_0_to_360: bool = False
    time_to_key: Callable[[DateLike], str] = daily_time_to_key
    file_start_date: FileStartDate | None = None
    missing: MissingPolicy = "zero"

    _files: SourceFiles = PrivateAttr()
    _layout: StorageLayout = PrivateAttr()
    _runs: tuple[tuple[_FrameRun, ...], ...] = PrivateAttr()
    _space: GridSpace = PrivateAttr()

    @field_validator("time_to_key", mode="before")
    @classmethod
    def _single_file_default(cls, value: Callable | None) -> Callable:
        return single_file_key if value is None else value

    @field_validator("bin_dtype")
    @classmethod
    def _validate_bin_dtype(cls, value: str) -> str:
        if np.dtype(value).kind not in {"i", "u", "f"}:
            raise ValueError("bin_dtype must describe a real numeric dtype")
        return value

    def _compile_plan(self, domain: TemporalDomain) -> DatasetPlan:
        if self.time_interval != timedelta(days=1):
            raise ValueError("DailyBinDataset time_interval must be one day")
        plan = self._planned(domain, self.chunk_len)
        layout = StorageLayout(
            base_dir=Path(self.base_dir),
            prefix=self.prefix,
            suffix=self.suffix,
            time_to_key=self.time_to_key,
        )
        locations, daily = self._frame_locations(plan, layout)
        self._files = self._inspect_frames(locations, layout, daily=daily)
        self._layout = layout
        self._runs = tuple(
            self._frame_runs(chunk.source_times(), locations)
            for chunk in plan.chunk_plan
        )
        self._space = self._grid()
        return plan

    def _frame_locations(
        self, plan: DatasetPlan, layout: StorageLayout
    ) -> tuple[dict[DateLike, tuple[str, int]], bool]:
        """Map each simulated date to ``(file key, absolute frame index)``.

        Also returns whether the files follow the one-file-per-day layout.
        """

        key = layout.keys()
        by_key: dict[str, list[DateLike]] = {}
        for time in dict.fromkeys(
            time for chunk in plan.chunk_plan for time in chunk.source_times()
        ):
            by_key.setdefault(key(time), []).append(time)
        daily = all(
            len(times) == 1 and name == daily_time_to_key(times[0])
            for name, times in by_key.items()
        )
        # Only the canonical one-file-per-day layout has an implicit frame
        # zero. Any grouped/custom layout (including a one-date subset of a
        # constant file) needs an explicit origin.
        if daily and self.file_start_date is not None:
            raise ValueError(
                "file_start_date must be None for one-file-per-day binary layouts"
            )
        if not daily and self.file_start_date is None:
            raise ValueError(
                "file_start_date is required for grouped or custom binary file layouts"
            )
        locations: dict[DateLike, tuple[str, int]] = {}
        for name, times in by_key.items():
            if daily:
                locations[times[0]] = (name, 0)
                continue
            origin = self._file_origin(name, plan.domain)
            for time in sorted(times):
                frame = timedelta_quotient(
                    time - origin,
                    self.time_interval,
                    duration_label=f"file frame offset for key {name!r}",
                    interval_label="daily binary time_interval",
                )
                if frame < 0:
                    raise ValueError(
                        f"date {time!s} precedes file_start_date {origin!s} "
                        f"for binary file key {name!r}"
                    )
                locations[time] = (name, frame)
        return locations, daily

    def _file_origin(self, key: str, domain: TemporalDomain) -> DateLike:
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
        label = f"file_start_date for key {key!r}"
        require_date(origin, label=label)
        require_calendar(origin, domain.calendar, label=label)
        if type(origin) is not type(domain.start):
            raise ValueError(
                f"{label} must use the same datetime representation as "
                "dataset start_date"
            )
        return origin

    def _inspect_frames(
        self,
        locations: Mapping[DateLike, tuple[str, int]],
        layout: StorageLayout,
        *,
        daily: bool,
    ) -> SourceFiles:
        """Validate that every required file exists and holds its frames."""

        required: dict[Path, int] = {}
        for key, frame in locations.values():
            path = layout.path(key)
            required[path] = max(required.get(path, 0), frame + 1)
        frame_bytes = self.shape[0] * self.shape[1] * np.dtype(self.bin_dtype).itemsize
        inspection = SourceFiles.inspect(label=SOURCE_FILE_LABEL)
        for path, minimum_frames in required.items():
            with inspection.open(path):
                file_bytes = path.stat().st_size
                if file_bytes % frame_bytes != 0:
                    raise ValueError(
                        f"File size mismatch: {path} is {file_bytes} bytes, "
                        f"but shape={self.shape} dtype={self.bin_dtype} expects "
                        f"multiples of {frame_bytes} bytes "
                        f"(got {file_bytes / frame_bytes:.4f} frames). "
                        f"Check the 'shape' parameter."
                    )
                frames = file_bytes // frame_bytes
                if daily and frames != 1:
                    raise ValueError(
                        f"Daily binary file {path} must contain exactly one frame; "
                        f"found {frames}"
                    )
                if frames < minimum_frames:
                    raise ValueError(
                        f"Binary file {path} contains {frames} frames, "
                        f"but the requested absolute frame offsets require "
                        f"at least {minimum_frames} frames"
                    )
        return inspection.files()

    @staticmethod
    def _frame_runs(
        times: tuple[DateLike, ...],
        locations: Mapping[DateLike, tuple[str, int]],
    ) -> tuple[_FrameRun, ...]:
        """Group consecutive frames of one file into ``(key, first, count)`` runs."""

        runs: list[_FrameRun] = []
        for time in times:
            key, frame = locations[time]
            if runs and runs[-1][0] == key and runs[-1][1] + runs[-1][2] == frame:
                runs[-1] = (key, runs[-1][1], runs[-1][2] + 1)
            else:
                runs.append((key, frame, 1))
        return tuple(runs)

    def _grid(self) -> GridSpace:
        """Cell centres of a global grid in the declared axis orientation.

        ``shape`` is ``(ny, nx)``: latitude runs 90→-90 (or -90→90 when
        ``lat_south_to_north``), longitude -180→180 (or 0→360 when
        ``lon_0_to_360``).
        """

        ny, nx = self.shape
        res_lat = 180.0 / ny
        res_lon = 360.0 / nx
        if self.lat_south_to_north:
            lat = np.linspace(-90 + res_lat / 2, 90 - res_lat / 2, ny)
        else:
            lat = np.linspace(90 - res_lat / 2, -90 + res_lat / 2, ny)
        if self.lon_0_to_360:
            lon = np.linspace(res_lon / 2, 360 - res_lon / 2, nx)
        else:
            lon = np.linspace(-180 + res_lon / 2, 180 - res_lon / 2, nx)
        return GridSpace(longitude=immutable_array(lon), latitude=immutable_array(lat))

    @property
    def space(self) -> GridSpace:
        return self._space

    def read_storage(self, chunk) -> np.ndarray:
        """Read ``(T, Y, X)`` frames, or ``(T, N)`` for a mapped view."""

        data = self._read_frames(self._runs[chunk.index], self._space.selection)
        if self._space.selection is None:
            return data.reshape(chunk.length, *self.shape)
        return data

    def _read_frames(
        self, runs: tuple[_FrameRun, ...], selection: np.ndarray | None
    ) -> np.ndarray:
        frame_size = self.shape[0] * self.shape[1]
        storage_dtype = np.dtype(self.bin_dtype)
        blocks = []
        for key, first_frame, count in runs:
            path = self._files.checked(self._layout.path(key))
            data = np.fromfile(
                path,
                dtype=storage_dtype,
                count=count * frame_size,
                offset=first_frame * frame_size * storage_dtype.itemsize,
            )
            self._files.verify(path)
            data = data.reshape(count, frame_size)
            blocks.append(data if selection is None else data[:, selection])
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)

    def _convert(self, values: np.ndarray) -> np.ndarray:
        return convert(
            values,
            out_dtype=self.out_dtype,
            unit_factor=self.unit_factor,
            label="daily binary dataset",
        )

    def _first_frame_missing(self) -> np.ndarray:
        """``(Y, X)`` NaN mask of the frame at ``start_date``."""

        key, frame, _count = self._runs[self.chunk_plan.num_spinup_chunks][0]
        data = self._read_frames(((key, frame, 1),), None)
        return np.isnan(data.reshape(self.shape))

    def close(self) -> None:
        """Close this process's file handles (binary reads keep none)."""

        self._files.close()

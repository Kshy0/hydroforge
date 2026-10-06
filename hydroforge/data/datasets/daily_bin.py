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
from hydroforge.data.datasets.plan import DatasetPlan, SourceChunk, TemporalDomain
from hydroforge.data.datasets.space import GridSpace
from hydroforge.data.datasets.storage import (
    SOURCE_FILE_LABEL,
    UnitFactor,
    UnitsName,
    resolve_units,
)
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

    Consecutive frames of one file are read together; a mapped view reads
    only the latitude rows spanning its source cells and keeps only those
    cells before any value check, so cells outside it (for example ocean NaN)
    never affect the result.  Missing values (NaN, or ``fill_value`` when set)
    become zero unless ``missing="error"``.

    Binary files record no units: ``target_units`` converts from the declared
    ``source_units`` with :func:`~hydroforge.core.units.check_units`.

    The grid is global unless ``extent=(west, east, south, north)`` gives the
    outer cell edges of a regional grid.
    """

    base_dir: SourceDirectory
    shape: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]]
    prefix: str
    unit_factor: UnitFactor = 1.0
    source_units: UnitsName | None = None
    target_units: UnitsName | None = None
    bin_dtype: str = "float32"
    suffix: str = ".one"
    lat_south_to_north: bool = False
    lon_0_to_360: bool = False
    extent: tuple[float, float, float, float] | None = None
    fill_value: float | None = None
    time_to_key: Callable[[DateLike], str] = daily_time_to_key
    file_start_date: FileStartDate | None = None
    missing: MissingPolicy = "zero"

    _files: SourceFiles = PrivateAttr()
    _layout: StorageLayout = PrivateAttr()
    _runs: tuple[tuple[_FrameRun, ...], ...] = PrivateAttr()
    _space: GridSpace = PrivateAttr()
    _units: tuple[float, float, float] = PrivateAttr(default=(1.0, 1.0, 0.0))

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

    @field_validator("extent")
    @classmethod
    def _validate_extent(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is not None:
            west, east, south, north = value
            if not all(np.isfinite(value)) or west >= east or south >= north:
                raise ValueError(
                    "extent must be finite (west, east, south, north) with "
                    "west < east and south < north"
                )
        return value

    def _compile_plan(self, domain: TemporalDomain) -> DatasetPlan:
        if self.extent is not None and self.lon_0_to_360:
            raise ValueError("lon_0_to_360 applies only to the default global extent")
        if self.time_interval != timedelta(days=1):
            raise ValueError("DailyBinDataset time_interval must be one day")
        self._units = resolve_units(self)
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
        """Cell centres of the grid in the declared axis orientation.

        ``shape`` is ``(ny, nx)``: latitude runs north→south (or south→north
        when ``lat_south_to_north``) and longitude west→east over ``extent``,
        by default 90→-90 and -180→180 (or 0→360 when ``lon_0_to_360``).
        """

        ny, nx = self.shape
        if self.extent is not None:
            west, east, south, north = self.extent
        else:
            west, east = (0.0, 360.0) if self.lon_0_to_360 else (-180.0, 180.0)
            south, north = -90.0, 90.0
        res_lat = (north - south) / ny
        res_lon = (east - west) / nx
        if self.lat_south_to_north:
            lat = np.linspace(south + res_lat / 2, north - res_lat / 2, ny)
        else:
            lat = np.linspace(north - res_lat / 2, south + res_lat / 2, ny)
        lon = np.linspace(west + res_lon / 2, east - res_lon / 2, nx)
        return GridSpace(longitude=immutable_array(lon), latitude=immutable_array(lat))

    @property
    def space(self) -> GridSpace:
        return self._space

    def read_storage(self, chunk: SourceChunk) -> np.ndarray:
        """Read ``(T, Y, X)`` frames, or ``(T, N)`` for a mapped view."""

        data = self._read_frames(self._runs[chunk.index], self._space.selection)
        if self._space.selection is None:
            data = data.reshape(chunk.length, *self.shape)
        return self._masked(data)

    def _masked(self, data: np.ndarray) -> np.ndarray:
        """Mask ``fill_value`` sentinels so the missing policy applies."""

        if self.fill_value is None:
            return data
        fill = np.asarray(self.fill_value).astype(data.dtype)
        missing = data == fill
        return np.ma.MaskedArray(data, missing) if missing.any() else data

    def _read_frames(
        self, runs: tuple[_FrameRun, ...], selection: np.ndarray | None
    ) -> np.ndarray:
        ny, nx = self.shape
        frame_size = ny * nx
        storage_dtype = np.dtype(self.bin_dtype)
        rows = columns = None
        if selection is not None and selection.size:
            # Frames are row-major, so the latitude rows spanning the
            # selection are one contiguous band of each frame.
            first_row = int(selection.min()) // nx
            rows = slice(first_row, int(selection.max()) // nx + 1)
            columns = selection - first_row * nx
        blocks = []
        for key, first_frame, count in runs:
            path = self._files.checked(self._layout.path(key))
            if selection is None:
                data = np.fromfile(
                    path,
                    dtype=storage_dtype,
                    count=count * frame_size,
                    offset=first_frame * frame_size * storage_dtype.itemsize,
                ).reshape(count, frame_size)
            elif columns is None:
                data = np.empty((count, 0), dtype=storage_dtype)
            else:
                frames = np.memmap(
                    path,
                    dtype=storage_dtype,
                    mode="r",
                    offset=first_frame * frame_size * storage_dtype.itemsize,
                    shape=(count, ny, nx),
                )
                band = np.array(frames[:, rows, :]).reshape(count, -1)
                del frames
                data = band[:, columns]
            self._files.verify(path)
            blocks.append(data)
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)

    def _convert(self, values: np.ndarray) -> np.ndarray:
        unit_factor, unit_scale, unit_offset = self._units
        return convert(
            values,
            out_dtype=self.out_dtype,
            unit_factor=unit_factor,
            label="daily binary dataset",
            unit_scale=unit_scale,
            unit_offset=unit_offset,
        )

    def _first_frame_missing(self) -> np.ndarray:
        """``(Y, X)`` missing-value mask of the frame at ``start_date``."""

        key, frame, _count = self._runs[self.chunk_plan.num_spinup_chunks][0]
        data = self._read_frames(((key, frame, 1),), None).reshape(self.shape)
        missing = np.ma.getmaskarray(self._masked(data))
        if data.dtype.kind == "f":
            missing = missing | np.isnan(data)
        return missing

    def close(self) -> None:
        """Close this process's file handles (binary reads keep none)."""

        self._files.close()

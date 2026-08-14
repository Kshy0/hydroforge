# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import cftime
from netCDF4 import Dataset, num2date

from hydroforge.contracts.temporal import (
    canonical_calendar,
    timedelta_microseconds,
)

if TYPE_CHECKING:
    from hydroforge.data.datasets.base import AbstractDataset
    from hydroforge.data.datasets.chunking import SourceChunk


DateTime = datetime | cftime.datetime
ReadOp = tuple[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class TimelineRead:
    """One immutable NetCDF storage plan for a logical time request."""

    storage_start: DateTime
    operations: tuple[ReadOp, ...]
    output_length: int


class DatasetTimeline:
    """Compile NetCDF timestamps into immutable-style chunk read plans.

    Spatial layout is deliberately absent. Both gridded and pre-aggregated
    catchment datasets can therefore share temporal planning without sharing
    unsupported spatial capabilities through inheritance.
    """

    def __init__(
        self,
        owner: AbstractDataset,
        *,
        base_dir: str,
        prefix: str,
        suffix: str,
        time_to_key,
        time_aggregation,
    ) -> None:
        self.owner = owner
        self.base_dir = base_dir
        self.prefix = prefix
        self.suffix = suffix
        self.time_to_key = time_to_key
        self.time_aggregation = time_aggregation
        self.file_times: dict[str, list[DateTime]] = {}
        self.dt_to_loc: dict[DateTime, tuple[str, int]] = {}
        self.source_time_interval: timedelta | None = None
        self.aggregation_factor: int | None = None
        self.plan: tuple[TimelineRead, ...] = ()

        self._scan(self._required_output_times())
        self._build_plan()

    def _path(self, key: str) -> Path:
        return Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"

    def _validate_files(self, keys: set[str]) -> None:
        self.owner.validate_files_exist([self._path(key) for key in sorted(keys)])

    def _file_calendar(self, key: str) -> str:
        """Read one shard's CF calendar without decoding its heavy variable."""

        path = self._path(key)
        with Dataset(path, "r") as dataset:
            time_var = self._time_variable(dataset, path)
            return canonical_calendar(getattr(time_var, "calendar", "standard"))

    def _discover_keys(self) -> set[str]:
        """Discover every flat source shard matching this prefix/suffix."""

        directory = Path(self.base_dir)
        if not directory.is_dir():
            return set()
        keys: set[str] = set()
        for path in directory.iterdir():
            if not path.is_file():
                continue
            name = path.name
            if not name.startswith(self.prefix) or not name.endswith(self.suffix):
                continue
            stop = len(name) - len(self.suffix) if self.suffix else len(name)
            if stop < len(self.prefix):
                continue
            key = name[len(self.prefix):stop]
            if self._path(key) == path:
                keys.add(key)
        return keys

    def _required_output_times(self) -> list[DateTime]:
        """Return distinct I/O timestamps required by the shared chunk plan."""

        required: list[DateTime] = []
        seen: set[DateTime] = set()
        for chunk in self.owner.chunk_plan:
            for timestamp in self.storage_times_for_chunk(chunk):
                if timestamp not in seen:
                    seen.add(timestamp)
                    required.append(timestamp)
        return required

    @staticmethod
    def _support_ranges(
        starts: list[DateTime], width: timedelta,
    ) -> tuple[tuple[DateTime, DateTime], ...]:
        """Merge the half-open source windows supporting output timestamps."""

        ranges: list[tuple[DateTime, DateTime]] = []
        for start in sorted(set(starts)):
            end = start + width
            if ranges and start <= ranges[-1][1]:
                previous_start, previous_end = ranges[-1]
                ranges[-1] = (previous_start, max(previous_end, end))
            else:
                ranges.append((start, end))
        return tuple(ranges)

    @staticmethod
    def _inside_supports(
        timestamp: DateTime,
        supports: tuple[tuple[DateTime, DateTime], ...],
    ) -> bool:
        return any(start <= timestamp < end for start, end in supports)

    def _scan(self, required_times: list[DateTime]) -> None:
        owner = self.owner
        aggregate = self.time_aggregation is not None
        required_set = set(required_times)
        supports = self._support_ranges(required_times, owner.time_interval)
        candidates = {self.time_to_key(timestamp) for timestamp in required_times}

        # The configured calendar is often a default.  Probe an existing shard
        # before validating keys so leap-day filenames and cftime conversion
        # follow the source calendar rather than a guessed standard calendar.
        probe = sorted(key for key in candidates if self._path(key).exists())
        if probe:
            source_calendar = self._file_calendar(probe[0])
            if owner.calendar != source_calendar:
                owner._adopt_source_calendar(source_calendar)
                required_times = self._required_output_times()
                required_set = set(required_times)
                supports = self._support_ranges(
                    required_times, owner.time_interval,
                )
                candidates = {
                    self.time_to_key(timestamp) for timestamp in required_times
                }

        if aggregate:
            # An output interval can span multiple file partitions. Deriving
            # keys only at output boundaries would skip every interior shard
            # (for example a monthly file inside a 70-day aggregation step).
            keys = self._discover_keys()
            if not keys:
                self._validate_files(candidates)
        else:
            keys = candidates
        self._validate_files(keys)

        source_times: list[DateTime] = []
        source_calendar: str | None = None
        seen_times: dict[DateTime, Path] = {}
        for key in sorted(keys):
            path = self._path(key)
            with Dataset(path, "r") as dataset:
                time_var = self._time_variable(dataset, path)
                file_calendar = canonical_calendar(
                    getattr(time_var, "calendar", "standard"),
                )
                if source_calendar is None:
                    source_calendar = file_calendar
                elif file_calendar != source_calendar:
                    raise ValueError(
                        "forcing files use inconsistent calendars: "
                        f"{source_calendar!r} and {file_calendar!r} in {path.name}"
                    )
                if owner.calendar != source_calendar:
                    owner._adopt_source_calendar(file_calendar)
                    required_times = self._required_output_times()
                    required_set = set(required_times)
                    supports = self._support_ranges(
                        required_times, owner.time_interval,
                    )
                dates = self._validated_dates(time_var, path)
                self._require_unique(dates, seen_times, path)
                seen_times.update((dt, path) for dt in dates)
                self.file_times[key] = list(dates)
                for index, dt in enumerate(dates):
                    in_range = (
                        self._inside_supports(dt, supports)
                        if aggregate else dt in required_set
                    )
                    if in_range:
                        self.dt_to_loc[dt] = (key, index)
                        if aggregate:
                            source_times.append(dt)

        if aggregate:
            self.source_time_interval = self._infer_source_interval(source_times)
            self.aggregation_factor = owner._get_time_aggregation_factor(self.source_time_interval)
            self._validate_aggregation_times(required_times)
        else:
            missing = [dt for dt in required_times if dt not in self.dt_to_loc]
            if missing:
                preview = ", ".join(str(dt) for dt in missing[:10])
                raise ValueError(
                    "Missing required timestamps for the chosen time_interval. "
                    f"First missing: {preview} (total {len(missing)}). "
                    "Check start_date alignment and dataset temporal resolution."
                )

    @staticmethod
    def _time_variable(dataset: Dataset, path: Path):
        time_var = (
            dataset.variables.get("time")
            or dataset.variables.get("valid_time")
        )
        if time_var is None:
            raise ValueError(f"Time variable not found in file: {path.name}")
        return time_var

    @classmethod
    def _validated_dates(cls, time_var, path: Path) -> list[DateTime]:
        dates = cls._decode_dates(time_var, path)
        if not dates:
            raise ValueError(f"Time axis is empty in {path.name}")
        non_increasing = [
            right for left, right in zip(dates, dates[1:])
            if right <= left
        ]
        if non_increasing:
            raise ValueError(
                f"Time axis in {path.name} must be strictly increasing; "
                f"first invalid timestamp is {non_increasing[0]}"
            )
        return dates

    @staticmethod
    def _require_unique(
        dates: list[DateTime], existing: dict[DateTime, Path], path: Path,
    ) -> None:
        duplicate = next((date for date in dates if date in existing), None)
        if duplicate is not None:
            raise ValueError(
                f"Timestamp {duplicate} occurs in both "
                f"{existing[duplicate].name} and {path.name}"
            )

    @staticmethod
    def _decode_dates(time_var, path: Path) -> list[DateTime]:
        calendar = getattr(time_var, "calendar", "standard")
        units = getattr(time_var, "units", None)
        if not isinstance(units, str) or not units.strip():
            raise ValueError(
                f"Time variable in {path.name} must define non-empty CF units"
            )
        try:
            return list(num2date(time_var[:], units, calendar))
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(
                f"Cannot decode CF time axis in {path.name}: "
                f"units={units!r}, calendar={calendar!r}"
            ) from error

    def ensure_support_time(self, timestamp: DateTime) -> None:
        """Index one extra support timestamp needed by a storage adapter.

        Normal timeline compilation indexes only planned output support. Some
        formats, such as cumulative ERA5 records, also need a predecessor.
        File discovery and time-axis validation remain the timeline's concern.
        """

        if timestamp in self.dt_to_loc:
            return

        key = self.time_to_key(timestamp)
        dates = self.file_times.get(key)
        path = self._path(key)
        if dates is None:
            self._validate_files({key})
            with Dataset(path, "r") as dataset:
                time_var = self._time_variable(dataset, path)
                file_calendar = canonical_calendar(
                    getattr(time_var, "calendar", "standard"),
                )
                if file_calendar != canonical_calendar(self.owner.calendar):
                    raise ValueError(
                        "forcing files use inconsistent calendars: "
                        f"{self.owner.calendar!r} and {file_calendar!r} in "
                        f"{path.name}"
                    )
                dates = self._validated_dates(time_var, path)
            existing_times = {
                date: self._path(existing_key)
                for existing_key, existing_dates in self.file_times.items()
                if existing_key != key
                for date in existing_dates
            }
            self._require_unique(dates, existing_times, path)
            self.file_times[key] = dates

        try:
            index = dates.index(timestamp)
        except ValueError:
            raise ValueError(
                f"Missing support timestamp {timestamp} in {path.name}"
            ) from None
        self.dt_to_loc[timestamp] = (key, index)

    @staticmethod
    def _infer_source_interval(source_times: list[DateTime]) -> timedelta:
        source_times = sorted(source_times)
        duplicates = [right for left, right in zip(source_times, source_times[1:]) if left == right]
        if duplicates:
            preview = ", ".join(str(dt) for dt in duplicates[:10])
            raise ValueError(f"Duplicate source timestamps found. First duplicates: {preview}")
        diffs = [right - left for left, right in zip(source_times, source_times[1:])]
        if not diffs:
            raise ValueError("Unable to infer source_time_interval from NetCDF time axis")
        widths = [
            timedelta_microseconds(diff, label="source_time_interval")
            for diff in diffs
        ]
        interval_width = min(widths)
        if interval_width <= 0:
            raise ValueError("source_time_interval must be positive")
        # Requested forcing can consist of disjoint segments (for example a
        # spin-up year and a much later main run).  Segment gaps are valid
        # multiples of the physical source interval; missing timestamps inside
        # an actual aggregation window are rejected separately by
        # _validate_aggregation_times().
        irregular = [
            diff for diff, width in zip(diffs, widths, strict=True)
            if width % interval_width
        ]
        if irregular:
            raise ValueError(
                "NetCDF source time axis must lie on one uniformly spaced "
                f"grid; smallest interval is {timedelta(microseconds=interval_width)}"
            )
        return timedelta(microseconds=interval_width)

    def source_times(self, output_times: list[DateTime]) -> list[DateTime]:
        if self.source_time_interval is None or self.aggregation_factor is None:
            raise RuntimeError(
                "source_times() requires a compiled time-aggregation plan"
            )
        return [
            dt + self.source_time_interval * offset
            for dt in output_times
            for offset in range(self.aggregation_factor)
        ]

    def _validate_aggregation_times(self, output_times: list[DateTime]) -> None:
        missing = [dt for dt in self.source_times(output_times) if dt not in self.dt_to_loc]
        if missing:
            preview = ", ".join(str(dt) for dt in missing[:10])
            raise ValueError(
                "Missing required source timestamps for time aggregation. "
                f"First missing: {preview} (total {len(missing)})."
            )

    def _ops_from_times(self, times: list[DateTime]) -> tuple[ReadOp, ...]:
        # Preserve the logical time order.  Grouping by file globally is
        # incorrect when a custom key function revisits a shard (A, B, A):
        # concatenating all A reads before B silently permutes the timeline.
        operations: list[ReadOp] = []
        current_key: str | None = None
        current_indices: list[int] = []

        def flush() -> None:
            nonlocal current_key, current_indices
            if current_key is not None and current_indices:
                operations.append((current_key, tuple(current_indices)))
            current_key = None
            current_indices = []

        for dt in times:
            key, index = self.dt_to_loc[dt]
            if key != current_key:
                flush()
                current_key = key
            current_indices.append(index)
        flush()
        return tuple(operations)

    def _build_read(self, times: list[DateTime]) -> TimelineRead:
        if not times:
            raise ValueError("timeline read requires at least one timestamp")
        if self.time_aggregation is None:
            operations = self._ops_from_times(times)
        else:
            operations = self._ops_from_times(self.source_times(times))
        return TimelineRead(
            storage_start=times[0],
            operations=operations,
            output_length=len(times),
        )

    def operations_for_times(
        self, times: list[DateTime],
    ) -> tuple[ReadOp, ...]:
        """Compile storage operations for an arbitrary logical time window."""

        return self._build_read(times).operations

    def read_for_chunk(self, chunk: SourceChunk) -> TimelineRead:
        """Return the storage plan owned by one exact source request."""

        self.owner.chunk_plan.validate_chunk(chunk)
        return self.plan[chunk.index]

    def storage_times_for_chunk(self, chunk: SourceChunk) -> list[DateTime]:
        """Expand one logical source request into exact storage timestamps."""

        mapper = getattr(self.owner, "_storage_time", None)
        logical_times = chunk.source_times(self.owner.time_interval)
        if mapper is None:
            return list(logical_times)
        return [mapper(timestamp) for timestamp in logical_times]

    def _build_plan(self) -> None:
        """Compile I/O operations from the owner's shared source chunks."""

        self.plan = tuple(
            self._build_read(self.storage_times_for_chunk(chunk))
            for chunk in self.owner.chunk_plan
        )

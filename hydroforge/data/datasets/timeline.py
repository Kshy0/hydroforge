# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Time-keyed file layouts and NetCDF timelines compiled into read plans.

A timeline maps each logical time to ``(file key, index on the file's time
axis)``.  Scanning inspects only the files that the requested times (or, for
time aggregation, their support windows) need; nothing here holds or changes a
dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import num2date

from hydroforge.core.time import (
    DateLike,
    canonical_calendar,
    timedelta_microseconds,
    timedelta_quotient,
)
from hydroforge.core.validation import frozen_dict
from hydroforge.data.datasets.plan import DatasetPlan, SourceChunk
from hydroforge.io.files import FileInspection
from hydroforge.io.rank_output.schema import (
    COMMITTED_STEPS_ATTR,
    FORMAT_ATTR,
    RANK_ATTR,
    RUN_ID_ATTR,
    TIME_DIM,
    VERSION_ATTR,
    WORLD_SIZE_ATTR,
    RankFileHeader,
)

ReadOp = tuple[str, tuple[int, ...]]
_UNSET = object()


@dataclass(frozen=True, slots=True)
class StorageLayout:
    """Files ``base_dir / f"{prefix}{time_to_key(time)}{suffix}"``."""

    base_dir: Path
    prefix: str
    suffix: str
    time_to_key: Callable[[DateLike], str]

    def path(self, key: str) -> Path:
        return self.base_dir / f"{self.prefix}{key}{self.suffix}"

    def keys(self) -> Callable[[DateLike], str]:
        """Return a key function that also enforces ``time_to_key`` determinism."""

        seen: dict[DateLike, str] = {}

        def key(timestamp: DateLike) -> str:
            value = self.time_to_key(timestamp)
            if type(value) is not str:
                raise TypeError(
                    "dataset time_to_key must return an exact string; got "
                    f"{type(value).__name__} for {timestamp}"
                )
            previous = seen.setdefault(timestamp, value)
            if previous != value:
                raise ValueError(
                    "dataset time_to_key must be deterministic; "
                    f"{timestamp} mapped to both {previous!r} and {value!r}"
                )
            return value

        return key


@dataclass(frozen=True, slots=True)
class NetCDFTimeline:
    """Scanned file times and the compiled storage reads of every plan chunk."""

    locations: Mapping[DateLike, tuple[str, int]]
    keys: tuple[str, ...]
    source_interval: timedelta | None
    aggregation_factor: int
    reads: tuple[tuple[ReadOp, ...], ...]

    def operations(self, times: Iterable[DateLike]) -> tuple[ReadOp, ...]:
        """Storage reads of arbitrary output times (aggregation expanded)."""

        return _operations(
            self.locations,
            _source_times(times, self.source_interval, self.aggregation_factor),
        )


def _source_times(
    times: Iterable[DateLike],
    source_interval: timedelta | None,
    factor: int,
) -> list[DateLike]:
    """The source records of output times, ``factor`` per aggregated time."""

    if source_interval is None:
        return list(times)
    return [
        time + source_interval * offset for time in times for offset in range(factor)
    ]


def _operations(
    locations: Mapping[DateLike, tuple[str, int]],
    times: Iterable[DateLike],
) -> tuple[ReadOp, ...]:
    # Preserve the logical time order.  Grouping by file globally is
    # incorrect when a custom key function revisits a shard (A, B, A).
    return tuple(
        (key, tuple(index for _key, index in group))
        for key, group in groupby(
            (locations[time] for time in times), key=lambda location: location[0]
        )
    )


def _support_ranges(
    starts: Sequence[DateLike],
    width: timedelta,
) -> tuple[tuple[DateLike, DateLike], ...]:
    """Merge the half-open source windows supporting output timestamps."""

    ranges: list[tuple[DateLike, DateLike]] = []
    for start in sorted(set(starts)):
        end = start + width
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return tuple(ranges)


def _support_keys(
    key: Callable[[DateLike], str],
    supports: tuple[tuple[DateLike, DateLike], ...],
    anchors: Sequence[DateLike],
) -> set[str]:
    """Return the storage keys of every partition meeting the supports.

    ``time_to_key`` is sampled at each anchor and window edge; any interval
    whose end keys differ is bisected down to one microsecond, so every
    contiguous key partition inside a window is found.  A key that reappears
    without showing at a sample fails the later completeness check instead of
    being guessed.
    """

    resolution = timedelta(microseconds=1)
    ordered = sorted(set(anchors))
    keys: set[str] = set()
    pending = []
    for start, end in supports:
        points = sorted(
            {start, end - resolution}.union(
                anchor for anchor in ordered if start <= anchor < end
            )
        )
        sampled = [(point, key(point)) for point in points]
        keys.update(value for _point, value in sampled)
        pending.extend(
            (left, right)
            for left, right in zip(sampled, sampled[1:])
            if left[1] != right[1]
        )
    while pending:
        (left, left_key), (right, right_key) = pending.pop()
        if right - left <= resolution:
            continue
        middle = left + (right - left) // 2
        middle_key = key(middle)
        keys.add(middle_key)
        if middle_key != left_key:
            pending.append(((left, left_key), (middle, middle_key)))
        if middle_key != right_key:
            pending.append(((middle, middle_key), (right, right_key)))
    return keys


def _time_variable(dataset: Any, path: Path, variable: str) -> Any:
    data_dimensions = set(dataset.variables[variable].dimensions)
    candidates = [
        dataset.variables[name]
        for name in ("time", "valid_time")
        if name in dataset.variables
        and len(dataset.variables[name].dimensions) == 1
        and dataset.variables[name].dimensions[0] in data_dimensions
    ]
    if not candidates:
        raise ValueError(f"Time variable not found in file: {path.name}")
    if len(candidates) > 1:
        names = [candidate.name for candidate in candidates]
        raise ValueError(f"Ambiguous time variables in {path.name}: {names}")
    return candidates[0]


def _file_calendar(time_variable: Any) -> str:
    return canonical_calendar(getattr(time_variable, "calendar", "standard"))


def probe_calendar(
    inspection: FileInspection,
    layout: StorageLayout,
    variable: str,
    required: Sequence[DateLike],
    *,
    support_width: timedelta | None,
) -> str | None:
    """Return the CF calendar of the first existing file the times need."""

    key = layout.keys()
    candidates = {key(time) for time in required}
    probe = sorted(name for name in candidates if layout.path(name).exists())
    if not probe and support_width is not None:
        supports = _support_ranges(required, support_width)
        probe = sorted(
            name
            for name in _support_keys(key, supports, required)
            if layout.path(name).exists()
        )
    if not probe:
        return None
    path = layout.path(probe[0])
    with inspection.open_netcdf(path) as dataset:
        return _file_calendar(_time_variable(dataset, path, variable))


class TimelineScan:
    """Accumulate validated file times while one dataset is being compiled."""

    def __init__(
        self,
        inspection: FileInspection,
        layout: StorageLayout,
        *,
        variable: str,
        calendar: str,
        template: type,
        inspect_variable: Callable[[Any, Path], None],
    ) -> None:
        self._inspection = inspection
        self._layout = layout
        self._variable = variable
        self._calendar = calendar
        self._template = template
        self._inspect_variable = inspect_variable
        self._key = layout.keys()
        self._units: object = _UNSET
        self._file_calendar: str | None = None
        self._seen: dict[DateLike, str] = {}
        self.file_times: dict[str, list[DateLike]] = {}
        self.locations: dict[DateLike, tuple[str, int]] = {}
        self.source_interval: timedelta | None = None
        self.aggregation_factor = 1

    def scan(
        self,
        required: Sequence[DateLike],
        *,
        support_width: timedelta | None,
    ) -> None:
        """Locate every required time, or every source time of its window.

        ``support_width`` (the output interval) enables time aggregation:
        each required time then needs the source records in
        ``[time, time + support_width)``.
        """

        required_set = set(required)
        candidates = {self._key(time) for time in required}
        if support_width is None:
            self._scan_files(candidates)
            for key in sorted(candidates):
                for index, time in enumerate(self.file_times[key]):
                    if time in required_set:
                        self.locations[time] = (key, index)
            missing = [time for time in required if time not in self.locations]
            if missing:
                preview = ", ".join(str(time) for time in missing[:10])
                raise ValueError(
                    "Missing required timestamps for the chosen time_interval. "
                    f"First missing: {preview} (total {len(missing)}). "
                    "Check start_date alignment and dataset temporal resolution."
                )
            return

        # An output interval can span multiple file partitions, so keys at
        # output boundaries alone would skip interior shards (for example a
        # monthly file inside a 70-day aggregation step).  Files outside the
        # needed range are never listed or opened.
        supports = _support_ranges(required, support_width)
        keys = {
            key
            for key in _support_keys(self._key, supports, required)
            if self._layout.path(key).exists()
        } or candidates
        self._scan_files(keys)
        source_times = []
        for key in sorted(keys):
            for index, time in enumerate(self.file_times[key]):
                if any(start <= time < end for start, end in supports):
                    self.locations[time] = (key, index)
                    source_times.append(time)
        # Retain adjacent timestamps from the inspected files when the
        # requested support contains only one record. Coverage is checked
        # independently below; a missing requested frame is never inferred.
        cadence_times = source_times
        if len(cadence_times) < 2:
            cadence_times = [
                time for key in sorted(keys) for time in self.file_times[key]
            ]
        self.source_interval = _source_interval(cadence_times)
        self.aggregation_factor = _aggregation_factor(
            support_width, self.source_interval
        )
        missing = [
            time
            for time in _source_times(
                required, self.source_interval, self.aggregation_factor
            )
            if time not in self.locations
        ]
        if missing:
            preview = ", ".join(str(time) for time in missing[:10])
            raise ValueError(
                "Missing required source timestamps for time aggregation. "
                f"First missing: {preview} (total {len(missing)})."
            )

    def locate(self, time: DateLike) -> None:
        """Locate one extra support time, scanning its file if needed.

        Raises ``LookupError`` when that file lacks the time.
        """

        if time in self.locations:
            return
        key = self._key(time)
        self._scan_files((key,))
        try:
            index = self.file_times[key].index(time)
        except ValueError:
            raise LookupError(
                f"Missing support timestamp {time} in {self._layout.path(key).name}"
            ) from None
        self.locations[time] = (key, index)

    def freeze(
        self,
        plan: DatasetPlan,
        read_times: Callable[[SourceChunk], Sequence[DateLike]],
    ) -> NetCDFTimeline:
        """Compile the read of every chunk once."""

        reads = tuple(
            _operations(
                self.locations,
                _source_times(
                    read_times(chunk), self.source_interval, self.aggregation_factor
                ),
            )
            for chunk in plan.chunk_plan
        )
        return NetCDFTimeline(
            locations=frozen_dict(self.locations),
            keys=tuple(sorted(self.file_times)),
            source_interval=self.source_interval,
            aggregation_factor=self.aggregation_factor,
            reads=reads,
        )

    def _scan_files(self, keys: Iterable[str]) -> None:
        for key in sorted(set(keys).difference(self.file_times)):
            path = self._layout.path(key)
            with self._inspection.open_netcdf(path) as dataset:
                time_variable = _time_variable(dataset, path, self._variable)
                self._check_units(dataset, path)
                self._check_calendar(_file_calendar(time_variable), path)
                committed = self._committed_steps(dataset, time_variable, path)
                dates = self._dates(time_variable, path, key, committed)
                self._inspect_variable(dataset, path)
            duplicate = next((date for date in dates if date in self._seen), None)
            if duplicate is not None:
                raise ValueError(
                    f"Timestamp {duplicate} occurs in both "
                    f"{self._layout.path(self._seen[duplicate]).name} and {path.name}"
                )
            self._seen.update(dict.fromkeys(dates, key))
            self.file_times[key] = dates

    def _check_calendar(self, calendar: str, path: Path) -> None:
        if self._file_calendar is None:
            self._file_calendar = calendar
        elif calendar != self._file_calendar:
            raise ValueError(
                "forcing files use inconsistent calendars: "
                f"{self._file_calendar!r} and {calendar!r} in {path.name}"
            )
        if calendar != self._calendar:
            raise ValueError(
                f"forcing files use calendar {calendar!r}, but the dataset "
                f"declares or implies calendar {self._calendar!r}"
            )

    def _check_units(self, dataset: Any, path: Path) -> None:
        """Require time-concatenated shards to describe one physical unit."""

        variable = dataset.variables[self._variable]
        if "units" not in variable.ncattrs():
            units = None
        else:
            units = variable.getncattr("units")
            if not isinstance(units, str) or not units.strip():
                raise ValueError(
                    f"Data variable {self._variable!r} in {path.name} "
                    "must define units as a non-empty string when present"
                )
        if self._units is _UNSET:
            self._units = units
        elif units != self._units:
            raise ValueError(
                f"Data variable {self._variable!r} uses inconsistent "
                f"units across forcing shards: {self._units!r} and "
                f"{units!r} in {path.name}"
            )

    def _committed_steps(
        self, dataset: Any, time_variable: Any, path: Path
    ) -> int | None:
        """Restrict framework output to published rows; leave external files alone."""

        protocol = {
            FORMAT_ATTR,
            VERSION_ATTR,
            RANK_ATTR,
            WORLD_SIZE_ATTR,
            RUN_ID_ATTR,
            COMMITTED_STEPS_ATTR,
        }
        if not protocol.intersection(dataset.ncattrs()):
            return None
        header = RankFileHeader.read(dataset, path=path)
        variable = dataset.variables[self._variable]
        if time_variable.dimensions != (TIME_DIM,) or (
            not variable.dimensions or variable.dimensions[0] != TIME_DIM
        ):
            raise ValueError(f"Rank output in {path.name} must share the time axis")
        if header.committed_steps > len(time_variable):
            raise ValueError(
                f"Rank output in {path.name} has inconsistent committed steps: "
                f"committed={header.committed_steps}, time={len(time_variable)}"
            )
        return header.committed_steps

    def _dates(
        self, time_variable: Any, path: Path, key: str, committed: int | None
    ) -> list[DateLike]:
        calendar = getattr(time_variable, "calendar", "standard")
        units = getattr(time_variable, "units", None)
        if not isinstance(units, str) or not units.strip():
            raise ValueError(
                f"Time variable in {path.name} must define non-empty CF units"
            )
        if time_variable.ndim != 1 or np.dtype(time_variable.dtype).kind not in "iuf":
            raise ValueError(
                f"Time variable in {path.name} must be one-dimensional and numeric"
            )
        raw = time_variable[:committed]
        if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
            raise ValueError(f"Time variable in {path.name} contains missing values")
        values = np.asarray(raw)
        if values.ndim != 1:
            raise ValueError(f"Time variable in {path.name} must be one-dimensional")
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError(
                f"Time variable in {path.name} must contain finite numeric values"
            )
        try:
            dates = list(num2date(values, units, calendar))
        except (ValueError, TypeError, OverflowError) as error:
            dates = (
                self._year_relative_days(values, key)
                if units.strip().casefold() == "days since start"
                else None
            )
            if dates is None:
                raise ValueError(
                    f"Cannot decode CF time axis in {path.name}: "
                    f"units={units!r}, calendar={calendar!r}"
                ) from error
        if not dates:
            raise ValueError(f"Time axis is empty in {path.name}")
        non_increasing = next(
            (right for left, right in zip(dates, dates[1:]) if right <= left),
            None,
        )
        if non_increasing is not None:
            raise ValueError(
                f"Time axis in {path.name} must be strictly increasing; "
                f"first invalid timestamp is {non_increasing}"
            )
        return dates

    def _year_relative_days(
        self,
        values: np.ndarray,
        key: str,
    ) -> list[DateLike] | None:
        """Decode the legacy yearly ``days since start`` convention."""

        if len(key) != 4 or not key.isascii() or not key.isdigit():
            return None
        origin = (
            datetime(int(key), 1, 1)
            if self._template is datetime
            else self._template(int(key), 1, 1)
        )
        dates: list[DateLike] = []
        for value in values:
            microseconds = float(value) * 86_400_000_000
            rounded = round(microseconds)
            if not np.isclose(microseconds, rounded, rtol=0.0, atol=1.0e-6):
                return None
            try:
                date = origin + timedelta(microseconds=rounded)
            except OverflowError:
                return None
            if self._key(date) != key:
                return None
            dates.append(date)
        return dates


def _aggregation_factor(interval: timedelta, source_interval: timedelta) -> int:
    try:
        factor = timedelta_quotient(
            interval,
            source_interval,
            duration_label="time_interval",
            interval_label="source_time_interval",
        )
    except ValueError as error:
        raise ValueError(
            "time_interval must be an exact integer multiple of "
            "source_time_interval for time aggregation"
        ) from error
    if factor <= 0:
        raise ValueError(
            "time_interval must not be shorter than source_time_interval "
            "for time aggregation"
        )
    return factor


def _source_interval(source_times: list[DateLike]) -> timedelta:
    source_times = sorted(source_times)
    diffs = [right - left for left, right in zip(source_times, source_times[1:])]
    if not diffs:
        raise ValueError("Unable to infer source_time_interval from NetCDF time axis")
    widths = [timedelta_microseconds(diff) for diff in diffs]
    interval_width = min(widths)
    # Requested forcing can consist of disjoint segments (for example a
    # spin-up year and a much later main run).  Segment gaps are valid
    # multiples of the physical source interval; missing timestamps inside an
    # aggregation window are rejected separately.
    if any(width % interval_width for width in widths):
        raise ValueError(
            "NetCDF source time axis must lie on one uniformly spaced "
            f"grid; smallest interval is {timedelta(microseconds=interval_width)}"
        )
    return timedelta(microseconds=interval_width)

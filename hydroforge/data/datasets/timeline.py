# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

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


DateTime = datetime | cftime.datetime
ReadOp = tuple[str, list[int]]


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
        self.global_times: list[DateTime] = []
        self.dt_to_loc: dict[DateTime, tuple[str, int]] = {}
        self.source_time_interval: timedelta | None = None
        self.aggregation_factor: int | None = None
        self.plan: list[tuple] = []
        self.spin_up_chunks_template: list[tuple] = []

        scan_start = owner.start_date
        scan_end = owner.end_date
        if owner.spin_up_cycles > 0:
            if owner.spin_up_start_date is not None:
                scan_start = min(scan_start, owner.spin_up_start_date)
            if owner.spin_up_end_date is not None:
                scan_end = max(scan_end, owner.spin_up_end_date)
        self._scan(scan_start, scan_end)
        self._build_plan()

    def _path(self, key: str) -> Path:
        return Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"

    def _validate_files(self, keys: set[str]) -> None:
        self.owner.validate_files_exist([self._path(key) for key in sorted(keys)])

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

    def _scan(self, start_dt: DateTime, end_dt: DateTime) -> None:
        owner = self.owner
        aggregate = self.time_aggregation is not None
        scan_end = end_dt + owner.time_interval if aggregate else end_dt
        key_to_first: dict[str, DateTime] = {}
        current = start_dt
        while current < scan_end if aggregate else current <= scan_end:
            key_to_first.setdefault(self.time_to_key(current), current)
            current += owner.time_interval
        key_to_first.setdefault(self.time_to_key(scan_end if aggregate else end_dt), scan_end if aggregate else end_dt)

        candidates = set(key_to_first)
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
                time_var = dataset.variables.get("time") or dataset.variables.get("valid_time")
                if time_var is None:
                    raise ValueError(f"Time variable not found in file: {path.name}")
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
                    owner.update_calendar(file_calendar)
                    start_dt = owner._convert_to_calendar(start_dt)
                    end_dt = owner._convert_to_calendar(end_dt)
                    scan_end = owner._convert_to_calendar(scan_end)
                dates = self._decode_dates(time_var, path)
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
                duplicate = next(
                    (dt for dt in dates if dt in seen_times), None,
                )
                if duplicate is not None:
                    raise ValueError(
                        f"Timestamp {duplicate} occurs in both "
                        f"{seen_times[duplicate].name} and {path.name}"
                    )
                seen_times.update((dt, path) for dt in dates)
                self.file_times[key] = list(dates)
                for index, dt in enumerate(dates):
                    in_range = start_dt <= dt < scan_end if aggregate else start_dt <= dt <= end_dt
                    if in_range:
                        self.dt_to_loc[dt] = (key, index)
                        if aggregate:
                            source_times.append(dt)

        expected: list[DateTime] = []
        current = start_dt
        while current <= end_dt:
            expected.append(current)
            current += owner.time_interval
        if aggregate:
            self.source_time_interval = self._infer_source_interval(source_times)
            self.aggregation_factor = owner._get_time_aggregation_factor(self.source_time_interval)
            self._validate_aggregation_times(expected)
        else:
            missing = [dt for dt in expected if dt not in self.dt_to_loc]
            if missing:
                preview = ", ".join(str(dt) for dt in missing[:10])
                raise ValueError(
                    "Missing required timestamps for the chosen time_interval. "
                    f"First missing: {preview} (total {len(missing)}). "
                    "Check start_date alignment and dataset temporal resolution."
                )
        self.global_times = expected

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

    def ops_from_times(self, times: list[DateTime]) -> list[ReadOp]:
        order: list[str] = []
        by_file: dict[str, list[int]] = {}
        for dt in times:
            key, index = self.dt_to_loc[dt]
            if key not in by_file:
                order.append(key)
                by_file[key] = []
            if index not in by_file[key]:
                by_file[key].append(index)
        return [(key, by_file[key]) for key in order]

    def build_entry(self, times: list[DateTime]) -> tuple:
        if self.time_aggregation is None:
            return times[0], self.ops_from_times(times)
        return times[0], self.ops_from_times(self.source_times(times)), len(times)

    def _chunks(self, start_dt: DateTime, end_dt: DateTime) -> list[tuple]:
        times: list[DateTime] = []
        current = start_dt
        while current <= end_dt:
            times.append(current)
            current += self.owner.time_interval
        size = self.owner.chunk_len
        return [self.build_entry(times[start:start + size]) for start in range(0, len(times), size)]

    def _build_plan(self) -> None:
        owner = self.owner
        if owner.spin_up_cycles > 0:
            if owner.spin_up_start_date is None or owner.spin_up_end_date is None:
                raise ValueError("Spin-up dates must be provided if spin_up_cycles > 0")
            self.spin_up_chunks_template = self._chunks(owner.spin_up_start_date, owner.spin_up_end_date)
            for _ in range(owner.spin_up_cycles):
                self.plan.extend(self.spin_up_chunks_template)
        self.plan.extend(self._chunks(owner.start_date, owner.end_date))

    def is_valid_time_index(self, index: int) -> bool:
        if type(index) is not int or index < 0:
            return False
        chunk_index, offset = divmod(index, self.owner.chunk_len)
        if chunk_index >= len(self.plan):
            return False
        entry = self.plan[chunk_index]
        real_length = sum(len(op[1]) for op in entry[1]) if self.time_aggregation is None else entry[2]
        return offset < real_length

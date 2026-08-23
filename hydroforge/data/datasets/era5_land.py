# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, ClassVar, Optional, Self, Union

import cftime
import numpy as np
from netCDF4 import Dataset
from pydantic import Field, model_validator

from hydroforge.contracts.temporal import _timedelta_quotient_trusted
from hydroforge.data.datasets.base import _TrustedSourceChunk
from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.netcdf import NetCDFDataset
from hydroforge.data.netcdf import (
    _planned_netcdf_chunk_len,
    monthly_time_to_key,
)


_ERA5_LOGICAL_CHUNK_BYTES = 4 * 1024**3
_ERA5_PHYSICAL_CHUNK_MULTIPLIER = 2


class ERA5LandAccumDataset(NetCDFDataset):
    """
    ERA5-Land dataset for accumulated (cumulative) variables such as hourly runoff.

    The caller supplies physical start_date and end_date.  Each time point t
    represents the runoff for the interval [t, t + Δt).  For example:
        start_date = datetime(2000, 1, 1)            # first interval: [00:00, 01:00)
        end_date   = datetime(2000, 12, 31, 23, 0)   # last  interval: [23:00, 00:00 next day)

    SourceChunk.source_start and SimulationStep.source_start always return
    these physical (unshifted) times. SimulationStep.start follows model
    execution, including replayed spin-up cycles.

    Why we shift internally by +time_interval:
    ERA5-Land accumulated variables (e.g., hourly runoff `ro`) are time-stamped
    at the END of the accumulation period. Many preprocessed hourly files also
    store values as "cumulative since 00:00 UTC of the same day," with an
    important caveat:
      - At 00:00, the record stores the previous day's total (24h) accumulation.
      - The value at 01:00 represents the accumulation over [00:00, 01:00) of the new day.
      - The value at 02:00 represents the accumulation over [00:00, 02:00), and so on.

    When we want per-interval (hourly) increments aligned to [t, t+Δt), we need
    the cumulative value at (t+Δt). Therefore, internally the data-reading
    window is shifted forward by one time_interval. This shift is transparent
    to the caller.

    Note: because of the +Δt shift, reading the last physical time step may
    require a data file one interval beyond end_date. For example, with hourly
    data and end_date = datetime(2000, 12, 31, 23), the file runoff_2001_01.nc
    must exist and contain at least the 00:00 record.

    Example (Δt = 1 hour, units in mm):
      Cumulative (00:00 holds the previous day's 24h total):
        23:00 -> 10.0   (covers [00:00, 23:00) of the same day)
        00:00 -> 12.0   (yesterday's 24h total)
        01:00 -> 1.0    (new day: covers [00:00, 01:00))
      Desired hourly increments:
        [23:00, 00:00) -> 12.0 - 10.0 = 2.0
        [00:00, 01:00) -> 1.0

    Implementation outline (_transform_cumulative_to_incremental):
      1) Use the actual physical timestamp of every output row to identify
         midnight, when the first cumulative record of a day is used as-is.
      2) At every other timestamp, subtract the preceding cumulative record.
         If a read starts away from midnight, that predecessor is loaded as a
         support frame even when it lives in the preceding monthly file.
      3) Optionally clip negative finite differences to zero.

    This keeps the output aligned with the physical interval [t, t+Δt) and avoids
    off-by-one mistakes caused by end-of-period time stamps and the 00:00 daily total.
    """
    supports_time_aggregation: ClassVar[bool] = False

    base_dir: str | Path
    chunk_len: int | None = Field(default=None, strict=True, ge=1)
    var_name: str = "ro"
    prefix: str = "runoff_"
    suffix: str = ".nc"
    time_to_key: Callable[[datetime], str] = monthly_time_to_key
    clip_incremental_negative: bool = True
    time_aggregation: str | Mapping[str, str] | None = None

    def _planned_storage_chunk_len(self, path: Path) -> int:
        """Batch physical slabs while retaining midnight chunk boundaries."""

        daily_steps = _timedelta_quotient_trusted(
            timedelta(days=1),
            self.time_interval,
            duration_label="one day",
            interval_label="ERA5 time_interval",
        )
        return _planned_netcdf_chunk_len(
            path,
            self.var_name,
            fallback=daily_steps,
            max_bytes=_ERA5_LOGICAL_CHUNK_BYTES,
            physical_chunk_multiplier=_ERA5_PHYSICAL_CHUNK_MULTIPLIER,
            step_alignment=daily_steps,
        )

    @model_validator(mode="after")
    def _validate_era5_domain(self) -> Self:
        if self.time_aggregation is not None:
            raise ValueError(
                "ERA5LandAccumDataset does not support time_aggregation: "
                "cumulative records must be differenced before aggregation"
            )

        # Configure time resolution first.  Daily cumulative resets can only be
        # represented exactly when the requested interval divides one day.
        _timedelta_quotient_trusted(
            timedelta(days=1),
            self.time_interval,
            duration_label="one day",
            interval_label="ERA5 time_interval",
        )
        self._validate_daily_grid_alignment(
            self.start_date, self.time_interval, "start_date",
        )
        if self.spin_up_start_date is not None:
            self._validate_daily_grid_alignment(
                self.spin_up_start_date,
                self.time_interval,
                "spin_up_start_date",
            )
        return self

    @model_validator(mode="after")
    def _compile_cumulative_predecessors(self) -> Self:
        """Freeze every predecessor required by non-midnight chunk reads."""

        for chunk in self.chunk_plan:
            physical_times = tuple(chunk._source_times())
            if self._is_day_start(physical_times[0]):
                continue
            source_times = self._timeline.storage_times_for_chunk(chunk)
            predecessor = source_times[0] - self.time_interval
            try:
                self._timeline.ensure_support_time(predecessor)
            except ValueError as error:
                if str(error).startswith("Missing support timestamp"):
                    raise ValueError(
                        "Missing cumulative predecessor timestamp "
                        f"{predecessor}; it is required for a non-midnight "
                        "ERA5 interval"
                    ) from error
                raise

        source_paths = tuple(
            Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            for key in sorted(self._timeline.file_times)
        )
        axes_by_path: dict[Path, tuple[int, int, int]] = {}
        for path in source_paths:
            with Dataset(path, "r") as dataset:
                axes_by_path[
                    self._canonical_source_path(path)
                ] = self._validate_shard_coordinates(dataset, path)
        self._variable_axes_by_path = axes_by_path
        self._record_source_files(source_paths)
        return self

    def _storage_time(
        self, logical_time: Union[datetime, cftime.datetime],
    ) -> Union[datetime, cftime.datetime]:
        """Map interval-start time to ERA5's interval-end timestamp."""

        return logical_time + self.time_interval

    @staticmethod
    def _validate_daily_grid_alignment(
        dt: Union[datetime, cftime.datetime],
        interval: timedelta,
        label: str,
    ) -> None:
        """Reject a time grid that skips over the known midnight reset."""
        since_midnight = timedelta(
            hours=dt.hour,
            minutes=dt.minute,
            seconds=dt.second,
            microseconds=dt.microsecond,
        )
        try:
            _timedelta_quotient_trusted(
                since_midnight,
                interval,
                duration_label=f"{label} time-of-day",
                interval_label="ERA5 time_interval",
            )
        except ValueError as error:
            raise ValueError(
                f"{label} must lie on a time grid anchored at midnight so "
                "daily ERA5 cumulative resets are observable"
            ) from error

    @staticmethod
    def _is_day_start(dt: Union[datetime, cftime.datetime]) -> bool:
        return (
            dt.hour == 0
            and dt.minute == 0
            and dt.second == 0
            and dt.microsecond == 0
        )

    def _transform_cumulative_to_incremental(
        self,
        arr: np.ndarray,
        physical_times: list[Union[datetime, cftime.datetime]],
        previous: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert daily cumulative records using their physical interval times."""
        reset = np.fromiter(
            (self._is_day_start(dt) for dt in physical_times),
            dtype=bool,
            count=len(physical_times),
        )
        increments = np.empty_like(arr)
        if reset[0]:
            increments[0] = arr[0]
        else:
            increments[0] = arr[0] - previous

        if arr.shape[0] > 1:
            diff = arr[1:] - arr[:-1]
            increments[1:] = diff
            increments[reset] = arr[reset]
        if self.clip_incremental_negative:
            np.maximum(increments, 0, out=increments)
        return increments

    def _read_chunk(self, chunk: SourceChunk) -> _TrustedSourceChunk:
        physical_times = list(chunk._source_times())
        source_times = self._timeline.storage_times_for_chunk(chunk)

        needs_previous = not self._is_day_start(physical_times[0])
        read_times = source_times
        if needs_previous:
            predecessor = source_times[0] - self.time_interval
            read_times = [predecessor, *source_times]

        ops = self._timeline.operations_for_times(read_times)
        data = self._canonical_calculation_data(
            self._read_ops(ops), label="ERA5 cumulative input",
        ) / self.unit_factor

        previous = data[0] if needs_previous else None
        arr = data[1:] if needs_previous else data
        increments = self._transform_cumulative_to_incremental(
            arr, physical_times, previous,
        )
        return _TrustedSourceChunk(self._finalize_output_data(
            increments, label="ERA5 cumulative increment output",
        ))

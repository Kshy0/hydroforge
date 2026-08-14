# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from typing import Callable, Optional, Union

import cftime
import numpy as np

from hydroforge.contracts.temporal import timedelta_quotient
from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.netcdf import NetCDFDataset
from hydroforge.data.netcdf import monthly_time_to_key


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
    supports_time_aggregation = False

    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        model_step: timedelta,
        time_interval: timedelta = timedelta(hours=1),
        chunk_len: int = 24,
        var_name: str = "ro",
        prefix: str = "runoff_",
        suffix: str = ".nc",
        time_to_key: Optional[Callable[[datetime], str]] = monthly_time_to_key,
        spin_up_start_date: Optional[datetime] = None,
        spin_up_end_date: Optional[datetime] = None,
        clip_incremental_negative: bool = True,
        time_aggregation: Optional[Union[str, dict[str, str]]] = None,
        *args,
        **kwargs,
    ):
        if time_aggregation is not None:
            raise ValueError(
                "ERA5LandAccumDataset does not support time_aggregation: "
                "cumulative records must be differenced before aggregation"
            )

        # Whether to clip negative increments to zero during cumulative-to-incremental
        # conversion.  True (default) is correct for non-negative fluxes like runoff.
        # Set to False for fluxes that can legitimately be negative.
        self.clip_incremental_negative = clip_incremental_negative

        # Configure time resolution first.  Daily cumulative resets can only be
        # represented exactly when the requested interval divides one day.
        timedelta_quotient(
            timedelta(days=1),
            time_interval,
            duration_label="one day",
            interval_label="ERA5 time_interval",
        )
        self._validate_daily_grid_alignment(start_date, time_interval, "start_date")
        if spin_up_start_date is not None:
            self._validate_daily_grid_alignment(
                spin_up_start_date, time_interval, "spin_up_start_date",
            )

        # Keep the public contract on physical interval starts. NetCDF storage
        # timestamps are mapped to interval ends by _storage_time().
        self._era5_time_shift = time_interval

        super().__init__(
            base_dir=base_dir,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            model_step=model_step,
            chunk_len=chunk_len,
            var_name=var_name,
            prefix=prefix,
            suffix=suffix,
            time_to_key=time_to_key,
            time_aggregation=None,
            spin_up_start_date=spin_up_start_date,
            spin_up_end_date=spin_up_end_date,
            *args,
            **kwargs,
        )

    def _storage_time(
        self, logical_time: Union[datetime, cftime.datetime],
    ) -> Union[datetime, cftime.datetime]:
        """Map interval-start time to ERA5's interval-end timestamp."""

        return logical_time + self._era5_time_shift

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
            timedelta_quotient(
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

    def _ensure_source_time_available(
        self, source_time: Union[datetime, cftime.datetime],
    ) -> None:
        """Index the cumulative predecessor needed by a non-midnight read."""

        try:
            self._timeline.ensure_support_time(source_time)
        except ValueError as error:
            if str(error).startswith("Missing support timestamp"):
                raise ValueError(
                    f"Missing cumulative predecessor timestamp {source_time}; "
                    "it is required for a non-midnight ERA5 interval"
                ) from error
            raise

    def _transform_cumulative_to_incremental(
        self,
        arr: np.ndarray,
        physical_times: list[Union[datetime, cftime.datetime]],
        previous: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert daily cumulative records using their physical interval times."""
        if arr.shape[0] != len(physical_times):
            raise ValueError(
                f"Data length {arr.shape[0]} does not match timestamp count "
                f"{len(physical_times)}"
            )
        if arr.shape[0] == 0:
            return np.empty_like(arr)

        reset = np.fromiter(
            (self._is_day_start(dt) for dt in physical_times),
            dtype=bool,
            count=len(physical_times),
        )
        if not reset[0] and previous is None:
            raise ValueError(
                "A preceding cumulative frame is required when an ERA5 read "
                "starts away from midnight"
            )

        increments = np.empty_like(arr)
        if reset[0]:
            increments[0] = arr[0]
        else:
            increments[0] = arr[0] - previous
            if self.clip_incremental_negative:
                np.maximum(increments[0], 0, out=increments[0])

        if arr.shape[0] > 1:
            diff = arr[1:] - arr[:-1]
            if self.clip_incremental_negative:
                np.maximum(diff, 0, out=diff)
            increments[1:] = diff
            increments[reset] = arr[reset]
        return increments

    def _read_chunk(self, chunk: SourceChunk) -> np.ndarray:
        physical_times = list(chunk.source_times(self.time_interval))
        source_times = self._timeline.storage_times_for_chunk(chunk)

        needs_previous = not self._is_day_start(physical_times[0])
        read_times = source_times
        if needs_previous:
            predecessor = source_times[0] - self.time_interval
            self._ensure_source_time_available(predecessor)
            read_times = [predecessor, *source_times]

        ops = self._timeline.operations_for_times(read_times)
        data = self._finish_read(self._read_ops(ops))
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "ERA5 cumulative conversion requires a single array result"
            )

        previous = data[0] if needs_previous else None
        arr = data[1:] if needs_previous else data
        return self._transform_cumulative_to_incremental(
            arr, physical_times, previous,
        )

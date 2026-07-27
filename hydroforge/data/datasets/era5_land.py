# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Union

import cftime
import numpy as np
from netCDF4 import Dataset

from hydroforge.contracts.temporal import canonical_calendar, timedelta_quotient
from hydroforge.data.datasets.netcdf import NetCDFDataset
from hydroforge.data.netcdf import monthly_time_to_key


class ERA5LandAccumDataset(NetCDFDataset):
    """
    ERA5-Land dataset for accumulated (cumulative) variables such as hourly runoff.

    The caller supplies physical start_date and end_date.  Each time point t
    represents the runoff for the interval [t, t + Δt).  For example:
        start_date = datetime(2000, 1, 1)            # first interval: [00:00, 01:00)
        end_date   = datetime(2000, 12, 31, 23, 0)   # last  interval: [23:00, 00:00 next day)

    get_time_by_index() and DatasetStep.source_time always return these physical
    (unshifted) times. DatasetStep.model_time follows the same physical time,
    including replayed spin-up cycles.

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
        self.num_daily_steps = timedelta_quotient(
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

        # ---- Store the original physical dates for time reporting ----
        self._physical_start_date = start_date
        self._physical_end_date = end_date
        self._physical_spin_up_start_date = spin_up_start_date
        self._physical_spin_up_end_date = spin_up_end_date
        self._era5_time_shift = time_interval   # the +Δt shift applied to data reading

        # Shift spin-up dates if provided, similar to main simulation dates
        if spin_up_start_date is not None:
            spin_up_start_date += time_interval
        if spin_up_end_date is not None:
            spin_up_end_date += time_interval

        super().__init__(
            base_dir=base_dir,
            start_date=start_date + time_interval,
            end_date=end_date + time_interval,
            time_interval=time_interval,
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

    # ------------------------------------------------------------------
    # Time reporting: return physical (unshifted) times
    # ------------------------------------------------------------------
    @property
    def main_start_time(self):
        return self._convert_to_calendar(self._physical_start_date)

    def get_time_by_index(self, idx: int) -> Union[datetime, cftime.datetime]:
        """Return the physical simulation time for step `idx`.

        The base class stores shifted dates (start_date + Δt) for internal
        data-reading purposes.  We subtract the shift here so that callers
        always see the original physical time.
        """
        shifted_time = super().get_time_by_index(idx)
        return shifted_time - self._era5_time_shift

    def get_index_by_time(self, dt: Union[datetime, cftime.datetime]) -> int:
        """Return the step index for a physical datetime."""
        return super().get_index_by_time(dt + self._era5_time_shift)

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
        """Add one support timestamp to the lookup used by the read planner.

        The normal NetCDF timeline only indexes the requested shifted window.
        A non-midnight first interval also needs the cumulative frame immediately
        before that window, which can be in a preceding file partition.
        """
        if source_time in self._dt_to_loc:
            return

        key = self.time_to_key(source_time)
        dates = self._file_times.get(key)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        if dates is None:
            self.validate_files_exist([path])
            with Dataset(path, "r") as dataset:
                time_var = dataset.variables.get("time") or dataset.variables.get("valid_time")
                if time_var is None:
                    raise ValueError(f"Time variable not found in file: {path.name}")
                file_calendar = canonical_calendar(
                    getattr(time_var, "calendar", "standard"),
                )
                if file_calendar != canonical_calendar(self.calendar):
                    raise ValueError(
                        "forcing files use inconsistent calendars: "
                        f"{self.calendar!r} and {file_calendar!r} in {path.name}"
                    )
                dates = self._timeline._decode_dates(time_var, path)
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
            existing_times = {
                date: existing_key
                for existing_key, existing_dates in self._file_times.items()
                if existing_key != key
                for date in existing_dates
            }
            duplicate = next(
                (date for date in dates if date in existing_times), None,
            )
            if duplicate is not None:
                raise ValueError(
                    f"Timestamp {duplicate} occurs in both "
                    f"{self.prefix}{existing_times[duplicate]}{self.suffix} "
                    f"and {path.name}"
                )
            self._file_times[key] = dates

        index = next(
            (index for index, date in enumerate(dates) if date == source_time),
            None,
        )
        if index is None:
            raise ValueError(
                f"Missing cumulative predecessor timestamp {source_time} in "
                f"{path.name}; it is required for a non-midnight ERA5 interval"
            )
        self._dt_to_loc[source_time] = (key, index)

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

    def get_data(
        self,
        current_time: Union[datetime, cftime.datetime],
        chunk_len: int,
    ) -> np.ndarray:
        source_times = self._timeline.contiguous_times(current_time, chunk_len)
        physical_times = [dt - self._era5_time_shift for dt in source_times]

        needs_previous = not self._is_day_start(physical_times[0])
        read_times = source_times
        if needs_previous:
            predecessor = source_times[0] - self.time_interval
            self._ensure_source_time_available(predecessor)
            read_times = [predecessor, *source_times]

        ops = self._ops_from_times(read_times)
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

    def read_chunk(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self._plan):
            raise IndexError(f"Chunk index {idx} out of range (0-{len(self._plan)-1})")
        entry = self._plan[idx]
        count = entry[2] if len(entry) > 2 else sum(
            len(indices) for _key, indices in entry[1]
        )
        physical_start = entry[0] - self._era5_time_shift
        return self.get_data(physical_start, count)

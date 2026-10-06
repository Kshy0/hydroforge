# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path

import numpy as np

from hydroforge.core.time import DateLike, timedelta_quotient
from hydroforge.data.datasets.keys import monthly_time_to_key
from hydroforge.data.datasets.netcdf import NetCDFDataset
from hydroforge.data.datasets.plan import DatasetPlan, SourceChunk, TemporalDomain
from hydroforge.data.datasets.timeline import TimelineScan
from hydroforge.data.datasets.values import as_float64, finalize, ingest
from hydroforge.io.netcdf.read import plan_read_chunk_len

_ERA5_LOGICAL_CHUNK_BYTES = 4 * 1024**3
_ERA5_PHYSICAL_CHUNK_MULTIPLIER = 2


def _is_day_start(time: DateLike) -> bool:
    return (
        time.hour == 0
        and time.minute == 0
        and time.second == 0
        and time.microsecond == 0
    )


def _missing_mask(raw: np.ndarray) -> np.ndarray | None:
    """Masked or NaN positions of one read, or ``None`` when there are none."""

    mask = np.ma.getmask(raw)
    missing = None if mask is np.ma.nomask or not mask.any() else mask
    data = np.ma.getdata(raw)
    if (
        data.dtype.kind == "f"
        and data.size
        and not (np.isfinite(data.min()) and np.isfinite(data.max()))
    ):
        nan = np.isnan(data)
        if nan.any():
            missing = nan if missing is None else missing | nan
    return missing


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
      3) Clip negative increments to zero when ``clip_incremental_negative``
         (the default) or ``clip_negative`` is set.  Cumulative records are
         never clipped before differencing, so both options bound the
         reported increments rather than the stored accumulations.

    This keeps the output aligned with the physical interval [t, t+Δt) and avoids
    off-by-one mistakes caused by end-of-period time stamps and the 00:00 daily total.
    """

    var_name: str = "ro"
    prefix: str = "runoff_"
    time_to_key: Callable[[DateLike], str] = monthly_time_to_key
    clip_incremental_negative: bool = True

    def _compile_plan(self, domain: TemporalDomain) -> DatasetPlan:
        if self.time_aggregation is not None:
            raise ValueError(
                "ERA5LandAccumDataset does not support time_aggregation: "
                "cumulative records must be differenced before aggregation"
            )
        # Daily cumulative resets are only representable when the interval
        # divides one day and the time grid is anchored at midnight.
        self._daily_steps()
        for label, start in (
            ("start_date", self.start_date),
            ("spin_up_start_date", self.spin_up_start_date),
        ):
            if start is not None:
                self._require_midnight_grid(start, label)
        plan = super()._compile_plan(domain)
        if self._store.unit_offset != 0.0:
            raise ValueError(
                "ERA5LandAccumDataset cannot apply an offset unit conversion "
                "to cumulative records"
            )
        return plan

    def _daily_steps(self) -> int:
        return timedelta_quotient(
            timedelta(days=1),
            self.time_interval,
            duration_label="one day",
            interval_label="ERA5 time_interval",
        )

    def _require_midnight_grid(self, time: DateLike, label: str) -> None:
        """Reject a time grid that skips over the known midnight reset."""

        since_midnight = timedelta(
            hours=time.hour,
            minutes=time.minute,
            seconds=time.second,
            microseconds=time.microsecond,
        )
        try:
            timedelta_quotient(
                since_midnight,
                self.time_interval,
                duration_label=f"{label} time-of-day",
                interval_label="ERA5 time_interval",
            )
        except ValueError as error:
            raise ValueError(
                f"{label} must lie on a time grid anchored at midnight so "
                "daily ERA5 cumulative resets are observable"
            ) from error

    def _storage_offset(self) -> timedelta:
        """Records are stamped at the end of their interval."""

        return self.time_interval

    def _planned_chunk_len(self, path: Path) -> int:
        """Batch physical slabs while retaining midnight chunk boundaries."""

        daily_steps = self._daily_steps()
        return plan_read_chunk_len(
            path,
            self.var_name,
            fallback=daily_steps,
            max_bytes=_ERA5_LOGICAL_CHUNK_BYTES,
            physical_chunk_multiplier=_ERA5_PHYSICAL_CHUNK_MULTIPLIER,
            step_alignment=daily_steps,
        )

    def _read_times(self, chunk: SourceChunk) -> Sequence[DateLike]:
        """A non-midnight chunk also reads the cumulative record before it."""

        storage = [time + self.time_interval for time in chunk.source_times()]
        if _is_day_start(chunk.source_start):
            return storage
        return [chunk.source_start, *storage]

    def _locate_support(self, scan: TimelineScan, plan: DatasetPlan) -> None:
        for chunk in plan.chunk_plan:
            if _is_day_start(chunk.source_start):
                continue
            try:
                scan.locate(chunk.source_start)
            except LookupError as error:
                raise ValueError(
                    "Missing cumulative predecessor timestamp "
                    f"{chunk.source_start}; it is required for a non-midnight "
                    "ERA5 interval"
                ) from error

    def _transform_cumulative_to_incremental(
        self,
        arr: np.ndarray,
        physical_times: Sequence[DateLike],
        previous: np.ndarray | None = None,
        missing: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert daily cumulative records using their physical interval times."""
        reset = np.fromiter(
            (_is_day_start(time) for time in physical_times),
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
        if missing is not None:
            offset = int(previous is not None)
            invalid = missing[offset:].copy()
            if offset:
                invalid[0] |= missing[0]
            invalid[1:] |= missing[offset:-1]
            invalid[reset] = missing[offset:][reset]
            increments[invalid] = 0
        if self.clip_incremental_negative or self.clip_negative:
            np.maximum(increments, 0, out=increments)
        return increments

    def _read_source(self, index: int) -> np.ndarray:
        chunk = self.chunk_plan.chunks[index]
        needs_previous = not _is_day_start(chunk.source_start)
        raw = self._read_operations(self._store.timeline.reads[index])
        # Keep observation validity until after differencing. A missing
        # predecessor makes the following increment missing too; midnight
        # resets depend only on the current observation.
        missing = _missing_mask(raw)
        # Accumulations are never clipped: clipping bounds the increments.
        values = ingest(
            raw,
            rows=chunk.length + needs_previous,
            missing=self.missing,
            clip_negative=False,
            label="source chunk",
        )
        # The fresh read (and so its float64 promotion) is owned here.
        data = as_float64(values, label="ERA5 cumulative input")
        store = self._store
        if store.unit_factor != 1.0:
            np.divide(data, store.unit_factor, out=data)
        if store.unit_scale != 1.0:
            np.multiply(data, store.unit_scale, out=data)
        increments = self._transform_cumulative_to_incremental(
            data[1:] if needs_previous else data,
            chunk.source_times(),
            data[0] if needs_previous else None,
            missing=missing,
        )
        return finalize(
            increments,
            out_dtype=self.out_dtype,
            checked=True,
            label="ERA5 cumulative increment output",
        )

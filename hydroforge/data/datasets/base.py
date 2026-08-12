# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional, Union

import cftime
import numpy as np
import torch

from hydroforge.data.distributed import is_rank_zero
from hydroforge.contracts.temporal import (
    UpsamplingMethod,
    DatasetTemporalContract,
    SimulationSchedule,
    SpinupSchedule,
    canonical_calendar,
    timedelta_microseconds,
    timedelta_quotient,
)
from hydroforge.data.datasets.chunking import SourceChunkPlan


logger = logging.getLogger(__name__)


def _close_dataset_tree(root: object, *, scope: str) -> None:
    """Close each unique leaf in one composite dataset ownership tree."""

    pending = [root]
    visited: set[int] = set()
    leaves: list[object] = []
    failures: list[BaseException] = []
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in visited:
            continue
        visited.add(identity)
        try:
            children = getattr(item, "_close_children", None)
        except BaseException as error:
            failures.append(error)
            continue
        if children is None:
            leaves.append(item)
            continue
        if not callable(children):
            failures.append(TypeError(
                f"dataset {type(item).__name__}._close_children must be callable"
            ))
            continue
        try:
            owned = children()
        except BaseException as error:
            failures.append(error)
            continue
        if not isinstance(owned, tuple):
            failures.append(TypeError(
                f"dataset {type(item).__name__}._close_children() must return "
                "a tuple"
            ))
            continue
        if owned:
            pending.extend(reversed(owned))
        else:
            leaves.append(item)

    for leaf in leaves:
        close = getattr(leaf, "close", None)
        if not callable(close):
            failures.append(TypeError(
                f"dataset leaf {type(leaf).__name__} has no close() method"
            ))
            continue
        try:
            close()
        except BaseException as error:
            failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        from hydroforge.contracts import ResourceCleanupError

        raise ResourceCleanupError(scope, failures)

class AbstractDataset(torch.utils.data.Dataset, ABC):
    """
    Custom abstract class that inherits from PyTorch Dataset.
    Defines a common interface for accessing data with distributed support.
    """
    supports_time_aggregation = False

    @property
    def main_start_time(self):
        """Physical start of the main source support exposed to drivers."""
        return self.start_date

    @property
    def main_end_time(self):
        """Physical inclusive end of the main source support."""
        return self.end_date

    @property
    def spin_up_start_time(self):
        """Physical start of the source interval replayed for spin-up."""
        return self.spin_up_start_date

    @property
    def spin_up_end_time(self):
        """Physical inclusive end of the source interval replayed for spin-up."""
        return self.spin_up_end_date

    @property
    def temporal_contract(self) -> DatasetTemporalContract:
        """Return the immutable source contract compiled during initialization."""

        return self._temporal_contract

    @property
    def chunk_plan(self) -> SourceChunkPlan:
        """Return the immutable real-length source-chunk plan."""

        return self._chunk_plan

    @property
    def simulation_schedule(self) -> SimulationSchedule:
        """Return the immutable schedule compiled during initialization."""

        return self._simulation_schedule

    @property
    def reuse_count(self) -> int:
        """Return the schedule-owned number of model calls per source row."""

        return self._simulation_schedule.reuse_count

    @staticmethod
    def _validate_time_aggregation(method: Optional[str]) -> Optional[str]:
        if method is None:
            return None
        if method not in ("mean", "max", "min", "sum"):
            raise ValueError(
                f"Unsupported time_aggregation={method!r}; "
                "expected one of: mean, max, min, sum"
            )
        return method

    @classmethod
    def _normalize_time_aggregation(
        cls,
        time_aggregation: Optional[Union[str, Dict[str, str]]],
    ) -> Optional[Union[str, Dict[str, str]]]:
        if time_aggregation is None:
            return None
        if isinstance(time_aggregation, str):
            return cls._validate_time_aggregation(time_aggregation)
        if isinstance(time_aggregation, dict):
            if not time_aggregation:
                raise ValueError("time_aggregation mapping must not be empty")
            return {
                str(name): cls._validate_time_aggregation(method)
                for name, method in time_aggregation.items()
            }
        raise TypeError("time_aggregation must be None, a string, or a dict")

    def _get_time_aggregation_factor(self, source_time_interval: timedelta) -> int:
        if timedelta_microseconds(
            source_time_interval, label="source_time_interval",
        ) <= 0:
            raise ValueError("source_time_interval must be positive")
        try:
            factor = timedelta_quotient(
                self.time_interval,
                source_time_interval,
                duration_label="time_interval",
                interval_label="source_time_interval",
            )
        except ValueError as exc:
            raise ValueError(
                "time_interval must be an exact integer multiple of "
                "source_time_interval for time aggregation"
            ) from exc
        if factor <= 0:
            raise ValueError(
                "time_interval must not be shorter than source_time_interval "
                "for time aggregation"
            )
        return factor

    def _aggregate_time_axis(
        self,
        data: np.ndarray,
        source_time_interval: timedelta,
        method: str,
    ) -> np.ndarray:
        method = self._validate_time_aggregation(method)
        factor = self._get_time_aggregation_factor(source_time_interval)
        if data.shape[0] % factor != 0:
            raise ValueError(
                f"Cannot aggregate {data.shape[0]} source frames into "
                f"windows of {factor} frames"
            )
        grouped = data.reshape((data.shape[0] // factor, factor) + data.shape[1:])
        if method == "mean":
            out = grouped.mean(axis=1)
        elif method == "max":
            out = grouped.max(axis=1)
        elif method == "min":
            out = grouped.min(axis=1)
        elif method == "sum":
            out = grouped.sum(axis=1)
        else:
            raise ValueError(f"Unsupported time_aggregation={method!r}")
        return out.astype(self.out_dtype, copy=False)

    def _apply_time_aggregation(
        self,
        data: np.ndarray,
        source_time_interval: timedelta,
        time_aggregation: Union[str, Dict[str, str]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        time_aggregation = self._normalize_time_aggregation(time_aggregation)
        if isinstance(time_aggregation, str):
            return self._aggregate_time_axis(data, source_time_interval, time_aggregation)
        return {
            name: self._aggregate_time_axis(data, source_time_interval, method)
            for name, method in time_aggregation.items()
        }

    def _convert_to_calendar(self, dt: Union[datetime, cftime.datetime]) -> Union[datetime, cftime.datetime]:
        if dt is None:
            return None
        if self.calendar == "standard":
            if isinstance(dt, cftime.datetime):
                return datetime(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
                    dt.microsecond,
                )
            return dt
        else:
            from hydroforge.contracts.temporal import convert_calendar_date

            return convert_calendar_date(dt, self.calendar)
    def __init__(
        self,
        start_date: Union[datetime, cftime.datetime],
        end_date: Union[datetime, cftime.datetime],
        time_interval: timedelta,
        model_step: timedelta,
        out_dtype: str = "float32",
        chunk_len: int = 1,
        spin_up_cycles: int = 0,
        spin_up_start_date: Optional[Union[datetime, cftime.datetime]] = None,
        spin_up_end_date: Optional[Union[datetime, cftime.datetime]] = None,
        calendar: str = "standard",
        clip_negative: bool = False,
        upsampling: UpsamplingMethod | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if type(chunk_len) is not int or chunk_len < 1:
            raise ValueError("chunk_len must be an exact positive int")
        if type(spin_up_cycles) is not int or spin_up_cycles < 0:
            raise ValueError("spin_up_cycles must be an exact non-negative int")
        if type(clip_negative) is not bool:
            raise TypeError("clip_negative must be an exact bool")
        if timedelta_microseconds(
            time_interval, label="dataset time_interval",
        ) <= 0:
            raise ValueError("dataset time_interval must be positive")
        model_step_microseconds = timedelta_microseconds(
            model_step, label="model_step",
        )
        if model_step_microseconds <= 0:
            raise ValueError("model_step must be positive")
        if model_step_microseconds > timedelta_microseconds(
            time_interval, label="dataset time_interval",
        ):
            raise ValueError("model_step must not exceed dataset time_interval")
        try:
            normalized_dtype = np.dtype(out_dtype)
        except TypeError as error:
            raise TypeError(f"invalid dataset out_dtype {out_dtype!r}") from error
        if (spin_up_start_date is None) != (spin_up_end_date is None):
            raise ValueError(
                "spin_up_start_date and spin_up_end_date must be provided together"
            )
        if spin_up_cycles > 0 and spin_up_start_date is None:
            raise ValueError(
                "spin_up_start_date and spin_up_end_date are required when "
                "spin_up_cycles is positive"
            )

        self.out_dtype = normalized_dtype.name
        self.chunk_len = chunk_len
        self.start_date = start_date
        self.end_date = end_date
        self.spin_up_cycles = spin_up_cycles
        self.spin_up_start_date = spin_up_start_date
        self.spin_up_end_date = spin_up_end_date
        self.time_interval = time_interval
        self.model_step = model_step
        self.upsampling = upsampling
        self.calendar = canonical_calendar(calendar)
        self.clip_negative = clip_negative

        # Local grid indices for spatial compression (set by build_local_mapping)
        self._local_indices: Optional[np.ndarray] = None
        self._desired_catchment_ids: Optional[np.ndarray] = None

        # Convert dates to the specified calendar immediately
        self.start_date = self._convert_to_calendar(start_date)
        self.end_date = self._convert_to_calendar(end_date)
        self.spin_up_start_date = self._convert_to_calendar(spin_up_start_date)
        self.spin_up_end_date = self._convert_to_calendar(spin_up_end_date)
        if self.start_date is None or self.end_date is None:
            raise ValueError("dataset start_date and end_date are required")
        if self.end_date < self.start_date:
            raise ValueError("dataset end_date must not precede start_date")
        timedelta_quotient(
            self.end_date - self.start_date,
            self.time_interval,
            duration_label="dataset endpoint span",
            interval_label="time_interval",
        )
        if self.spin_up_start_date is not None:
            if self.spin_up_end_date < self.spin_up_start_date:
                raise ValueError(
                    "spin_up_end_date must not precede spin_up_start_date"
                )
            timedelta_quotient(
                self.spin_up_end_date - self.spin_up_start_date,
                self.time_interval,
                duration_label="spin-up endpoint span",
                interval_label="time_interval",
            )
        self._compile_temporal_state()
        if self.reuse_count > 1 and upsampling not in {"repeat", "distribute"}:
            raise ValueError(
                "upsampling must be explicitly 'repeat' or 'distribute' "
                "when model_step is shorter than dataset time_interval"
            )
        if self.reuse_count == 1 and upsampling is not None:
            raise ValueError(
                "upsampling must be None when model_step equals time_interval"
            )

    def _compile_temporal_state(self) -> None:
        """Compile all source counts, chunks and model calls from one contract."""

        start = self.main_start_time
        end = self.main_end_time
        count = timedelta_quotient(
            end - start,
            self.time_interval,
            duration_label="main dataset endpoint span",
            interval_label="time_interval",
        ) + 1
        spinup = None
        if self.spin_up_cycles > 0:
            spinup_start = self.spin_up_start_time
            spinup_end = self.spin_up_end_time
            if spinup_start is None or spinup_end is None:
                raise ValueError(
                    "spin_up_start_date and spin_up_end_date are required "
                    "when spin_up_cycles is positive"
                )
            spinup = SpinupSchedule(
                source_start=spinup_start,
                source_end=spinup_end + self.time_interval,
                cycles=self.spin_up_cycles,
            )
        contract = DatasetTemporalContract(
            calendar=self.calendar,
            start=start,
            interval=self.time_interval,
            count=count,
            spinup=spinup,
        )
        chunk_plan = SourceChunkPlan(contract, self.chunk_len)
        schedule = SimulationSchedule.from_contract(
            contract, step=self.model_step,
        )
        self._temporal_contract = contract
        self._chunk_plan = chunk_plan
        self._simulation_schedule = schedule

    @staticmethod
    def _as_nan_array(data: np.ndarray) -> np.ndarray:
        """Convert NetCDF masked values to NaN while preserving normal values."""
        if isinstance(data, np.ma.MaskedArray):
            mask = np.ma.getmaskarray(data)
            if np.any(mask):
                if np.issubdtype(data.dtype, np.floating):
                    return np.asarray(data.filled(np.nan))
                return np.asarray(data.astype(np.float64).filled(np.nan))
            return np.asarray(data.data)
        return np.asarray(data)

    def _apply_value_policy(self, data: np.ndarray) -> np.ndarray:
        """Convert masked values to zero and optionally clip negatives."""
        arr = self._as_nan_array(data)

        if np.issubdtype(arr.dtype, np.floating):
            nan_mask = np.isnan(arr)
            if np.any(nan_mask):
                if not arr.flags.writeable:
                    arr = arr.copy()
                arr[nan_mask] = 0.0

        if self.clip_negative:
            if not arr.flags.writeable:
                arr = arr.copy()
            np.maximum(arr, 0, out=arr)
        return arr

    def _apply_upsampling_policy(self, data: Any) -> Any:
        if self.upsampling != "distribute":
            return data
        if isinstance(data, Mapping):
            return {
                name: self._apply_upsampling_policy(value)
                for name, value in data.items()
            }
        return data / self.reuse_count

    def update_calendar(self, calendar: str):
        """
        Updates the calendar and converts all date attributes to the new calendar.
        """
        self.calendar = canonical_calendar(calendar)
        self.start_date = self._convert_to_calendar(self.start_date)
        self.end_date = self._convert_to_calendar(self.end_date)
        self.spin_up_start_date = self._convert_to_calendar(self.spin_up_start_date)
        self.spin_up_end_date = self._convert_to_calendar(self.spin_up_end_date)
        self._compile_temporal_state()

    def validate_files_exist(self, file_paths: list[Union[str, Path]]) -> None:
        """
        Validates that all files in the provided list exist.
        Raises FileNotFoundError if any are missing.
        """
        missing_files = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                missing_files.append(str(path))

        if missing_files:
            raise FileNotFoundError(
                "The following required data files are missing:\n" +
                "\n".join(missing_files)
            )

        if is_rank_zero() and self.spin_up_cycles > 0:
            logger.info("Spin-up enabled: %d cycles", self.spin_up_cycles)

    def forcing_blocks(self, chunk: Any) -> Iterator[tuple[Any, int]]:
        """Yield each source item once with its exact model-call reuse count."""

        if isinstance(chunk, Mapping):
            lengths = {name: len(value) for name, value in chunk.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"forcing variables have different lengths: {lengths}")
            length = next(iter(lengths.values()))
            for index in range(length):
                yield (
                    {name: value[index] for name, value in chunk.items()},
                    self.reuse_count,
                )
            return
        for item in chunk:
            yield item, self.reuse_count

    @property
    def num_spin_up_chunks(self) -> int:
        return self._chunk_plan.num_spinup_chunks

    def read_chunk(self, idx: int) -> np.ndarray:
        """Read one planned real-length chunk, including spin-up replays."""

        chunk = self._chunk_plan[idx]
        return self.get_data(chunk.source_start, chunk.length)

    def source_chunk_length(self, idx: int) -> int:
        """Return the real source-row count for one storage chunk."""

        return self._chunk_plan[idx].length

    def source_time_at(
        self, chunk_index: int, offset: int,
    ) -> Union[datetime, cftime.datetime]:
        """Return one logical source timestamp from the shared chunk plan."""

        return self._chunk_plan.source_time(chunk_index, offset)

    @abstractmethod
    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """
        Read a contiguous time block starting at current_time.

        Inputs:
        - current_time: start datetime aligned to the dataset time grid
        - chunk_len: positive integer upper bound of steps to read

        Returns:
        - If _local_indices is None (before build_local_mapping):
            3D numpy array with shape (T, Y, X) where Y=lat, X=lon
        - If _local_indices is set (after build_local_mapping):
            2D numpy array with shape (T, N) where N = number of active grids
        - T ∈ [1, chunk_len]. The final block near the end may have T < chunk_len.

        Spatial convention:
        - Dimension order: (lat, lon) i.e. (Y, X)
        - When flattening: C-order (row-major), lon varies fastest
        - Coordinate arrays from get_coordinates(): (lon, lat)

        Implementation notes:
        - Do not read beyond the available time range; truncate instead.
        - Do not pad to chunk_len.
        - Preserve chronological order for the returned timesteps.
        """
        ...

    def get_time_by_index(self, idx: int) -> Union[datetime, cftime.datetime]:
        """Return a source time on the historical chunk-major flat axis.

        Short-chunk padding positions are rejected instead of being assigned a
        fabricated timestamp. New code should prefer :meth:`source_time_at`.
        """

        if type(idx) is not int:
            raise TypeError("dataset time index must be an exact int")
        if idx < 0:
            raise IndexError(idx)
        chunk_index, offset = divmod(idx, self.chunk_len)
        return self.source_time_at(chunk_index, offset)

    def get_index_by_time(self, dt: Union[datetime, cftime.datetime]) -> int:
        """Returns the index in the main simulation timeline for a given datetime."""
        offset = dt - self._temporal_contract.start
        return timedelta_quotient(
            offset,
            self._temporal_contract.interval,
            duration_label="requested time offset",
            interval_label="dataset sample interval",
        )

    @property
    def num_main_source_steps(self) -> int:
        return self._temporal_contract.count

    @property
    def num_spin_up_source_steps_per_cycle(self) -> int:
        return self._chunk_plan.spinup_source_count_per_cycle

    @abstractmethod
    def close(self) -> None:
        """
        Close any open resources or files. Implementations must be idempotent.
        """

    def _close_children(self) -> tuple[object, ...]:
        """Return directly owned datasets; leaves own no child datasets."""

        return ()

    def _combine(self, other, operation, reverse=False):
        from hydroforge.data.datasets.expression import DatasetExpression

        is_dataset = isinstance(other, (AbstractDataset, DatasetExpression))
        is_scalar = isinstance(other, (int, float, np.number))

        if not (is_dataset or is_scalar):
            return NotImplemented

        left, right = (other, self) if reverse else (self, other)
        return DatasetExpression(left, operation, right)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Fetch one chunk (T <= chunk_len) starting at chunk index `idx`.

        The final chunk retains its real time length instead of being padded.
        """
        if idx < 0:
            idx += len(self)

        compressed = self._local_indices is not None
        expected_length = self.source_chunk_length(idx)
        data = self.read_chunk(idx)

        if isinstance(data, dict):
            return {
                name: self._prepare_chunk_array(
                    block, compressed, expected_length=expected_length,
                )
                for name, block in data.items()
            }

        return self._prepare_chunk_array(
            data, compressed, expected_length=expected_length,
        )

    def _prepare_chunk_array(
        self,
        data: np.ndarray,
        compressed: bool,
        *,
        expected_length: int,
    ) -> np.ndarray:

        if data.shape[0] != expected_length:
            raise ValueError(
                f"read_chunk returned {data.shape[0]} rows, expected "
                f"{expected_length} from the dataset chunk plan"
            )

        if compressed:
            # Expect (T, N)
            N = self.data_size
            if data.ndim != 2 or data.shape[1] != N:
                raise ValueError(
                    f"read_chunk returned shape {tuple(data.shape)}, expected (T, {N})"
                )
        else:
            # Expect (T, Y, X)
            ny, nx = self.grid_shape
            if data.ndim != 3 or data.shape[1] != ny or data.shape[2] != nx:
                raise ValueError(
                    f"read_chunk returned shape {tuple(data.shape)}, expected (T, {ny}, {nx})"
                )
        data = self._apply_value_policy(data)
        data = self._apply_upsampling_policy(data)
        return np.ascontiguousarray(data)

    def __len__(self) -> int:
        return len(self._chunk_plan)

    def __add__(self, other):
        return self._combine(other, "add")

    def __radd__(self, other):
        return self._combine(other, "add", reverse=True)

    def __sub__(self, other):
        return self._combine(other, "sub")

    def __rsub__(self, other):
        return self._combine(other, "sub", reverse=True)

    def __mul__(self, other):
        return self._combine(other, "mul")

    def __rmul__(self, other):
        return self._combine(other, "mul", reverse=True)

    def __truediv__(self, other):
        return self._combine(other, "div")

    def __rtruediv__(self, other):
        return self._combine(other, "div", reverse=True)

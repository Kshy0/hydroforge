# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import cftime
import numpy as np
import torch

from hydroforge.data.distributed import is_rank_zero
from hydroforge.contracts.temporal import (
    DatasetTemporalContract,
    timedelta_microseconds,
    timedelta_quotient,
)


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

    def temporal_contract(self) -> DatasetTemporalContract:
        """Describe the main forcing support without choosing model steps."""
        start = self.main_start_time
        if start is None or self.time_interval is None:
            raise ValueError(
                "dataset start_date and time_interval are required for a "
                "temporal contract"
            )
        return DatasetTemporalContract(
            calendar=self.calendar,
            start=start,
            interval=self.time_interval,
            count=self.num_main_steps,
        )

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
        out_dtype: str = "float32",
        chunk_len: int = 1,
        spin_up_cycles: int = 0,
        spin_up_start_date: Optional[Union[datetime, cftime.datetime]] = None,
        spin_up_end_date: Optional[Union[datetime, cftime.datetime]] = None,
        calendar: str = "standard",
        clip_negative: bool = False,
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
        try:
            normalized_dtype = np.dtype(out_dtype)
        except TypeError as error:
            raise TypeError(f"invalid dataset out_dtype {out_dtype!r}") from error
        if (spin_up_start_date is None) != (spin_up_end_date is None):
            raise ValueError(
                "spin_up_start_date and spin_up_end_date must be provided together"
            )

        self.out_dtype = normalized_dtype.name
        self.chunk_len = chunk_len
        self.start_date = start_date
        self.end_date = end_date
        self.spin_up_cycles = spin_up_cycles
        self.spin_up_start_date = spin_up_start_date
        self.spin_up_end_date = spin_up_end_date
        self.time_interval = time_interval
        self.calendar = calendar
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
        self._spin_up_num_chunks = 0
        if self.spin_up_cycles > 0:
            self._calc_spin_up_params()

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

    def update_calendar(self, calendar: str):
        """
        Updates the calendar and converts all date attributes to the new calendar.
        """
        self.calendar = calendar
        self.start_date = self._convert_to_calendar(self.start_date)
        self.end_date = self._convert_to_calendar(self.end_date)
        self.spin_up_start_date = self._convert_to_calendar(self.spin_up_start_date)
        self.spin_up_end_date = self._convert_to_calendar(self.spin_up_end_date)

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

    def time_iter(self) -> Iterator[Tuple[datetime, bool, bool]]:
        """Returns an iterator that yields (time, is_valid, is_spin_up) tuples step-by-step."""
        valid_steps_count = 0

        # Calculate spin-up steps
        spin_up_steps = 0
        if self.spin_up_cycles > 0 and self.time_interval is not None:
             duration = self.get_spin_up_duration()
             spin_up_steps = timedelta_quotient(
                 duration,
                 self.time_interval,
                 duration_label="spin-up duration",
                 interval_label="time_interval",
             )

        # Iterate exactly as many times as the DataLoader will produce data points
        # This ensures we handle padding steps at the end of the last chunk correctly
        total_chunks = len(self)
        total_items = total_chunks * self.chunk_len

        for idx in range(total_items):
            try:
                dt = self.get_time_by_index(idx)
                valid = self.is_valid_time_index(idx)
            except IndexError:
                # Padding steps (out of bounds)
                dt = datetime.min
                valid = False

            is_spin_up = valid_steps_count < spin_up_steps
            yield dt, valid, is_spin_up

            if valid:
                valid_steps_count += 1

    def get_spin_up_duration(self) -> timedelta:
        """Calculates the total duration of the spin-up period."""
        if self.spin_up_cycles > 0:
            if self.time_interval is None:
                 raise ValueError("time_interval must be provided for spin-up calculation")

            if self.spin_up_start_date is None or self.spin_up_end_date is None:
                raise ValueError("spin_up_start_date and spin_up_end_date must be provided if spin_up_cycles > 0")

            # Calculate duration of one cycle
            # Assuming spin_up_end_date is inclusive, so we add one time_interval
            cycle_duration = self.spin_up_end_date - self.spin_up_start_date + self.time_interval

            return cycle_duration * self.spin_up_cycles
        return timedelta(0)

    def get_virtual_start_time(self, verbose: bool = False) -> datetime:
        """Calculates the virtual start time including spin-up."""
        if self.start_date is None:
            raise ValueError("start_date is required to calculate virtual start time")

        duration = self.get_spin_up_duration()
        virtual_start = self.start_date - duration

        if verbose and is_rank_zero() and self.spin_up_cycles > 0:
             logger.info("Spin-up duration: %s", duration)
             logger.info("Virtual start time: %s", virtual_start)

        return virtual_start

    def _calc_spin_up_params(self):
        if self.spin_up_cycles > 0:
            if self.time_interval is None:
                 raise ValueError("time_interval must be provided for spin-up calculation")
            if self.spin_up_start_date is None or self.spin_up_end_date is None:
                raise ValueError(
                    "spin_up_start_date and spin_up_end_date are required "
                    "when spin_up_cycles is positive"
                )
            # Calculate number of chunks in spin-up period
            total_duration = self.spin_up_end_date - self.spin_up_start_date
            total_steps = timedelta_quotient(
                total_duration,
                self.time_interval,
                duration_label="spin-up endpoint span",
                interval_label="time_interval",
            ) + 1
            self._spin_up_num_chunks = (total_steps + self.chunk_len - 1) // self.chunk_len
        else:
            self._spin_up_num_chunks = 0

    @property
    def num_spin_up_chunks(self) -> int:
        if self.spin_up_cycles > 0:
             return self._spin_up_num_chunks * self.spin_up_cycles
        return 0

    def read_chunk(self, idx: int) -> np.ndarray:
        """
        Default implementation of read_chunk that handles spin-up logic.
        Requires time_interval to be set.
        """
        if self.time_interval is None:
             raise ValueError("time_interval is required for chunked temporal access")

        if self.spin_up_cycles > 0:
             total_spin_up_chunks = self._spin_up_num_chunks * self.spin_up_cycles

             if idx < total_spin_up_chunks:
                 # In spin-up
                 cycle_idx = idx % self._spin_up_num_chunks
                 # Time relative to spin_up_start_date
                 steps_offset = cycle_idx * self.chunk_len
                 current_time = self.spin_up_start_date + self.time_interval * steps_offset
                 return self.get_data(current_time, self.chunk_len)

             # Main simulation
             idx -= total_spin_up_chunks

        # Main simulation time
        steps_offset = idx * self.chunk_len

        if self.start_date is None:
            raise ValueError("start_date is required to read the main simulation")

        current_time = self.start_date + self.time_interval * steps_offset
        return self.get_data(current_time, self.chunk_len)

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
        - Do not pad to chunk_len here; AbstractDataset.__getitem__ will pad.
        - Preserve chronological order for the returned timesteps.
        """
        ...

    def get_time_by_index(self, idx: int) -> Union[datetime, cftime.datetime]:
        """
        Returns the datetime corresponding to the given index.
        Default implementation handles spin-up and linear time stepping.
        """
        if self.time_interval is None:
             raise ValueError("time_interval is required for indexed temporal access")

        if self.spin_up_cycles > 0:
            if self.spin_up_start_date is None or self.spin_up_end_date is None:
                 raise ValueError("Spin-up dates must be provided")

            # Calculate items (including padding) in one spin-up cycle
            chunks_per_cycle = self.num_spin_up_chunks // self.spin_up_cycles
            items_per_cycle = chunks_per_cycle * self.chunk_len

            total_spin_up_items = items_per_cycle * self.spin_up_cycles

            if idx < total_spin_up_items:
                # In spin-up
                # cycle_idx is which repetition of spin-up we are in
                # idx_in_cycle is the index within that repetition (including padding)
                idx_in_cycle = idx % items_per_cycle

                return self.spin_up_start_date + self.time_interval * idx_in_cycle

            # Main simulation
            idx -= total_spin_up_items

        if self.start_date is None:
             raise AttributeError("Dataset must have 'start_date'")

        return self.start_date + self.time_interval * idx

    def get_index_by_time(self, dt: Union[datetime, cftime.datetime]) -> int:
        """Returns the index in the main simulation timeline for a given datetime."""
        if self.start_date is None or self.time_interval is None:
             raise ValueError("start_date and time_interval required")

        offset = dt - self.start_date
        return timedelta_quotient(
            offset,
            self.time_interval,
            duration_label="requested time offset",
            interval_label="time_interval",
        )

    @property
    def num_main_steps(self) -> int:
        if self.start_date is None or self.end_date is None or self.time_interval is None:
            return 0
        duration = self.end_date - self.start_date
        return timedelta_quotient(
            duration,
            self.time_interval,
            duration_label="main dataset endpoint span",
            interval_label="time_interval",
        ) + 1

    @property
    def num_spin_up_steps(self) -> int:
        if self.spin_up_cycles <= 0:
            return 0
        cycle_duration = self.spin_up_end_date - self.spin_up_start_date
        steps_per_cycle = timedelta_quotient(
            cycle_duration,
            self.time_interval,
            duration_label="spin-up endpoint span",
            interval_label="time_interval",
        ) + 1
        return steps_per_cycle * self.spin_up_cycles

    @property
    def total_steps(self) -> int:
        return self.num_spin_up_steps + self.num_main_steps

    def is_valid_time_index(self, idx: int) -> bool:
        """
        Checks if the given time index corresponds to a valid data step.
        Handles padding gaps in spin-up and main simulation.
        """
        if idx < 0:
            return False

        if self.spin_up_cycles > 0:
            chunks_per_cycle = self._spin_up_num_chunks
            items_per_cycle = chunks_per_cycle * self.chunk_len
            total_spin_up_items = items_per_cycle * self.spin_up_cycles

            if idx < total_spin_up_items:
                # In spin-up region
                idx_in_cycle = idx % items_per_cycle

                # Calculate valid steps per cycle
                cycle_duration = self.spin_up_end_date - self.spin_up_start_date
                steps_per_cycle = timedelta_quotient(
                    cycle_duration,
                    self.time_interval,
                    duration_label="spin-up endpoint span",
                    interval_label="time_interval",
                ) + 1

                return idx_in_cycle < steps_per_cycle

            # Move to main simulation region
            idx -= total_spin_up_items

        # Main simulation region
        return idx < self.num_main_steps

    def _real_len(self) -> int:
        """Number of chunks in main simulation."""
        total = self.num_main_steps
        return (total + self.chunk_len - 1) // self.chunk_len

    def validate_files_in_range(self, file_path_gen: Callable[[datetime], Path]) -> None:
        """
        Validates that files exist for all time steps in the simulation, including spin-up.
        file_path_gen: function that takes a datetime and returns a Path to the file.
        """
        if self.time_interval is None:
             raise ValueError("time_interval must be provided for validation")

        file_paths = set()

        # Main simulation
        if self.start_date and self.end_date:
            curr = self.start_date
            while curr <= self.end_date:
                file_paths.add(file_path_gen(curr))
                curr += self.time_interval

        # Spin-up
        if self.spin_up_cycles > 0:
            if self.spin_up_start_date and self.spin_up_end_date:
                curr = self.spin_up_start_date
                while curr <= self.spin_up_end_date:
                    file_paths.add(file_path_gen(curr))
                    curr += self.time_interval

        self.validate_files_exist(list(file_paths))

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

        Returns:
        - If build_local_mapping has been called: (chunk_len, N) compressed data
        - Otherwise: (chunk_len, Y, X) full grid data

        Pads with zeros if the actual data is shorter than chunk_len.
        """
        if idx < 0:
            idx += len(self)

        compressed = self._local_indices is not None
        data = self.read_chunk(idx)

        if isinstance(data, dict):
            return {name: self._pad_chunk_array(block, compressed) for name, block in data.items()}

        return self._pad_chunk_array(data, compressed)

    def _pad_chunk_array(self, data: np.ndarray, compressed: bool) -> np.ndarray:

        if compressed:
            # Expect (T, N)
            N = self.data_size
            if data.ndim != 2 or data.shape[1] != N:
                raise ValueError(
                    f"read_chunk returned shape {tuple(data.shape)}, expected (T, {N})"
                )
            T = data.shape[0]
            if T < self.chunk_len:
                pad = np.zeros((self.chunk_len - T, N), dtype=self.out_dtype)
                data = np.vstack([data, pad]) if data.size else pad
        else:
            # Expect (T, Y, X)
            ny, nx = self.grid_shape
            if data.ndim != 3 or data.shape[1] != ny or data.shape[2] != nx:
                raise ValueError(
                    f"read_chunk returned shape {tuple(data.shape)}, expected (T, {ny}, {nx})"
                )
            T = data.shape[0]
            if T < self.chunk_len:
                pad = np.zeros((self.chunk_len - T, ny, nx), dtype=self.out_dtype)
                data = np.vstack([data, pad]) if data.size else pad

        data = self._apply_value_policy(data)
        return np.ascontiguousarray(data)

    def __len__(self) -> int:
        return self._real_len() + self.num_spin_up_chunks

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

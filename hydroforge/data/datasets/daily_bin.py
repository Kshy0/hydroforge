# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cftime
import numpy as np

from hydroforge.data.datasets.gridded import GriddedDataset
from hydroforge.data.netcdf import daily_time_to_key, single_file_key


class DailyBinDataset(GriddedDataset):
    """
    Dataset class that reads daily binary files.

    By default each binary file contains one day's data, with filenames
    ``{prefix}{YYYYMMDD}{suffix}``.  The ``time_to_key`` callable controls
    the mapping from date to filename key:

    * **One file per day** (default): ``time_to_key = daily_time_to_key``
      → every date gets a unique key, each file has one frame.
    * **Single file** (e.g. 365-day climatology): use a constant key
      ``time_to_key = lambda dt: ""``  (or any constant string).
      Consecutive frames are read from the file by offset.
    """

    def _build_file_mapping(self):
        """Map each simulation date to ``(file_key, frame_index)``."""
        dates = {
            chunk.source_time(offset, self.time_interval)
            for chunk in self.chunk_plan
            for offset in range(chunk.length)
        }

        by_key: dict[str, list] = {}
        for dt in dates:
            by_key.setdefault(self.time_to_key(dt), []).append(dt)
        self._dt_to_loc = {
            dt: (key, frame_idx)
            for key, key_dates in by_key.items()
            for frame_idx, dt in enumerate(sorted(key_dates))
        }

    def _validate_files_exist(self):
        """Validate that all required files exist and match expected size."""
        required_frames = {}
        for key, frame_idx in self._dt_to_loc.values():
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            required_frames[path] = max(
                required_frames.get(path, 0), frame_idx + 1,
            )
        self.validate_files_exist(list(required_frames))

        # Validate file sizes are consistent with shape
        ny, nx = self.shape
        frame_bytes = ny * nx * np.dtype(self.bin_dtype).itemsize
        for fp, minimum_frames in required_frames.items():
            file_bytes = Path(fp).stat().st_size
            if file_bytes % frame_bytes != 0:
                raise ValueError(
                    f"File size mismatch: {fp} is {file_bytes} bytes, "
                    f"but shape={self.shape} dtype={self.bin_dtype} expects "
                    f"multiples of {frame_bytes} bytes "
                    f"(got {file_bytes / frame_bytes:.4f} frames). "
                    f"Check the 'shape' parameter."
                )
            observed_frames = file_bytes // frame_bytes
            if observed_frames < minimum_frames:
                raise ValueError(
                    f"File {fp} contains {observed_frames} frame(s), but the "
                    f"configured timeline reads at least {minimum_frames}."
                )

    def __init__(self,
                 base_dir: str,
                 shape: List[int],
                 start_date: datetime,
                 end_date: datetime,
                 model_step: timedelta,
                 prefix: str,
                 unit_factor: float = 1.0, # mm/day divided by unit_factor to get m/s
                 bin_dtype: str = "float32",
                 suffix: str = ".one",
                 out_dtype: str = "float32",
                 calendar: str = "standard",
                 lat_south_to_north: bool = False,  # If True, latitude goes from south to north
                 lon_0_to_360: bool = False,  # If True, longitude goes from 0 to 360 (e.g. ERA5-Land binary)
                 time_to_key: Optional[Callable[[Union[datetime, cftime.datetime]], str]] = daily_time_to_key,
                 *args, **kwargs):

        self.base_dir = base_dir
        self.shape = tuple(shape)
        self.unit_factor = unit_factor
        self.bin_dtype = bin_dtype
        self.prefix = prefix
        self.suffix = suffix
        self.lat_south_to_north = lat_south_to_north
        self.lon_0_to_360 = lon_0_to_360
        self.time_to_key = time_to_key if time_to_key is not None else single_file_key
        super().__init__(
            out_dtype=out_dtype,
            chunk_len=1,
            time_interval=timedelta(days=1),
            model_step=model_step,
            start_date=start_date,
            end_date=end_date,
            calendar=calendar,
            *args,
            **kwargs,
        )
        self._build_file_mapping()
        self._validate_files_exist()

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) coordinate arrays.

        Note: shape is (ny, nx) = (lat, lon), so shape[0] is lat size, shape[1] is lon size.
        Coordinates are cell centers, computed from shape assuming global coverage.

        If lat_south_to_north is True, latitude goes from -90 to 90 (south to north).
        Otherwise, latitude goes from 90 to -90 (north to south, default).

        If lon_0_to_360 is True, longitude goes from 0 to 360 (e.g. ERA5-Land binary).
        Otherwise, longitude goes from -180 to 180 (default).
        """
        ny, nx = self.shape
        # Resolution in degrees
        res_lat = 180.0 / ny
        res_lon = 360.0 / nx
        # Cell centers
        if self.lat_south_to_north:
            lat = np.linspace(-90 + res_lat / 2, 90 - res_lat / 2, ny)
        else:
            lat = np.linspace(90 - res_lat / 2, -90 + res_lat / 2, ny)
        if self.lon_0_to_360:
            lon = np.linspace(res_lon / 2, 360 - res_lon / 2, nx)
        else:
            lon = np.linspace(-180 + res_lon / 2, 180 - res_lon / 2, nx)
        return lon, lat

    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """Read one day's data from binary file.

        Returns:
        - If _local_indices is set: (1, N) compressed array
        - If _local_indices is None: (1, Y, X) full grid array

        Spatial convention: (Y, X) = (lat, lon), C-order flatten (lon varies fastest)
        """
        if chunk_len != 1:
            raise ValueError("DailyBinDataset only supports chunk_len=1 (one day per file)")

        key, frame_idx = self._dt_to_loc[current_time]
        filename = f"{self.prefix}{key}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        ny, nx = self.shape
        frame_size = ny * nx
        element_size = np.dtype(self.bin_dtype).itemsize

        data = np.fromfile(
            file_path, dtype=self.bin_dtype,
            count=frame_size, offset=frame_idx * frame_size * element_size,
        )
        if data.size != frame_size:
            raise ValueError(
                f"Could not read frame {frame_idx} from {file_path}; expected "
                f"{frame_size} values, got {data.size}"
            )
        data = data.astype(self.out_dtype) / self.unit_factor
        data = self._apply_value_policy(data)

        if self._local_indices is not None:
            return data[self._local_indices][None, :]
        else:
            return data.reshape(1, ny, nx)

    def _get_first_frame_nan_mask(self) -> Optional[np.ndarray]:
        key, frame_idx = self._dt_to_loc[self.start_date]
        file_path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"

        ny, nx = self.shape
        frame_size = ny * nx
        element_size = np.dtype(self.bin_dtype).itemsize
        data = np.fromfile(
            file_path,
            dtype=self.bin_dtype,
            count=frame_size,
            offset=frame_idx * frame_size * element_size,
        )
        if data.size != frame_size:
            raise ValueError(
                f"Could not read first frame from {file_path}; expected "
                f"{frame_size} values, got {data.size}"
            )
        if not np.issubdtype(data.dtype, np.floating):
            return np.zeros(frame_size, dtype=bool)
        return np.isnan(data).reshape(-1)

    def close(self):
        pass

    def __len__(self):
        """
        Returns the total number of samples in the dataset based on the time range.
        """
        return super().__len__()

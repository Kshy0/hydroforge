# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

import cftime
import netCDF4 as nc
import numpy as np
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

from hydroforge.modeling.distributed import find_indices_in, is_rank_zero
from hydroforge.io.aggregate import build_cama_mapping

class AbstractDataset(torch.utils.data.Dataset, ABC):
    """
    Custom abstract class that inherits from PyTorch Dataset.
    Defines a common interface for accessing data with distributed support.
    """
    supports_time_aggregation = False

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
        source_seconds = source_time_interval.total_seconds()
        if source_seconds <= 0:
            raise ValueError("source_time_interval must be positive")
        ratio = self.time_interval.total_seconds() / source_seconds
        factor = int(round(ratio))
        if factor <= 0 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-9):
            raise ValueError(
                "time_interval must be an integer multiple of "
                f"source_time_interval for time aggregation; got "
                f"time_interval={self.time_interval}, "
                f"source_time_interval={source_time_interval}"
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
                return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            return dt
        else:
            # Convert to cftime with self.calendar
            # If it's already cftime, we recreate it to ensure the calendar attribute matches
            return cftime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, calendar=self.calendar)
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
        skip_nan: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.out_dtype = out_dtype
        self.chunk_len = chunk_len
        self.start_date = start_date
        self.end_date = end_date
        self.spin_up_cycles = spin_up_cycles
        self.spin_up_start_date = spin_up_start_date
        self.spin_up_end_date = spin_up_end_date
        self.time_interval = time_interval
        self.calendar = calendar
        self.clip_negative = clip_negative
        self.skip_nan = skip_nan

        # Local grid indices for spatial compression (set by build_local_mapping)
        self._local_indices: Optional[np.ndarray] = None

        # Convert dates to the specified calendar immediately
        self.start_date = self._convert_to_calendar(start_date)
        self.end_date = self._convert_to_calendar(end_date)
        self.spin_up_start_date = self._convert_to_calendar(spin_up_start_date)
        self.spin_up_end_date = self._convert_to_calendar(spin_up_end_date)

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
        """Convert masked values, optionally zero NaNs, and optionally clip negatives."""
        arr = self._as_nan_array(data)

        if self.skip_nan and np.issubdtype(arr.dtype, np.floating):
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

    def _get_first_frame_nan_mask(self) -> Optional[np.ndarray]:
        """Return a flat full-grid NaN mask for skip_nan mapping, if supported."""
        return None

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
            print(f"[AbstractDataset] Spin-up enabled: {self.spin_up_cycles} cycles.")

    def time_iter(self) -> Iterator[Tuple[datetime, bool, bool]]:
        """Returns an iterator that yields (time, is_valid, is_spin_up) tuples step-by-step."""
        valid_steps_count = 0

        # Calculate spin-up steps
        spin_up_steps = 0
        if self.spin_up_cycles > 0 and self.time_interval is not None:
             duration = self.get_spin_up_duration()
             spin_up_steps = int(duration.total_seconds() / self.time_interval.total_seconds())

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
        if not hasattr(self, 'start_date'):
             raise AttributeError("Dataset must have 'start_date' to calculate virtual start time")

        duration = self.get_spin_up_duration()
        virtual_start = self.start_date - duration

        if verbose and is_rank_zero() and self.spin_up_cycles > 0:
             print(f"[AbstractDataset] Spin-up duration: {duration}")
             print(f"[AbstractDataset] Virtual start time: {virtual_start}")

        return virtual_start

    def _calc_spin_up_params(self):
        if self.spin_up_cycles > 0:
            if self.time_interval is None:
                 raise ValueError("time_interval must be provided for spin-up calculation")
            # Calculate number of chunks in spin-up period
            total_duration = self.spin_up_end_date - self.spin_up_start_date
            total_steps = int((total_duration.total_seconds() / self.time_interval.total_seconds())) + 1
            self._spin_up_num_chunks = (total_steps + self.chunk_len - 1) // self.chunk_len
        else:
            self._spin_up_num_chunks = 0

    @property
    def num_spin_up_chunks(self) -> int:
        if self.spin_up_cycles > 0:
             if not hasattr(self, '_spin_up_num_chunks'):
                 self._calc_spin_up_params()
             return self._spin_up_num_chunks * self.spin_up_cycles
        return 0

    def read_chunk(self, idx: int) -> np.ndarray:
        """
        Default implementation of read_chunk that handles spin-up logic.
        Requires time_interval to be set.
        """
        if self.time_interval is None:
             raise NotImplementedError("time_interval must be provided for default read_chunk")

        if self.spin_up_cycles > 0:
             if not hasattr(self, '_spin_up_num_chunks'):
                 self._calc_spin_up_params()

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

        if not hasattr(self, 'start_date'):
             raise AttributeError("Dataset must have 'start_date' attribute to use default read_chunk")

        current_time = self.start_date + self.time_interval * steps_offset
        return self.get_data(current_time, self.chunk_len)

    def shard_forcing(
        self,
        batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        local_mapping: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Map grid data to catchments and handle distributed sync.

        Expected input shape:
          - (B, T, N) for single trial
          - (B, T, K, N) for K trials

        N should match data_size (compressed source grids after build_local_mapping).
        Output shape: (M, C) where M is the product of non-spatial dims, C = number of catchments.
        """
        if isinstance(batch_data, dict):
            return {
                name: self.shard_forcing(block, local_mapping)
                for name, block in batch_data.items()
            }

        if batch_data.dim() == 3:
            B, T, N = batch_data.shape
            flat = batch_data.reshape(B * T, N)
        elif batch_data.dim() == 4:
            B, T, K, N = batch_data.shape
            flat = batch_data.reshape(B * T * K, N)
        else:
            raise ValueError(f"batch_data must be 3D or 4D, got shape {tuple(batch_data.shape)}")

        if self.skip_nan and flat.is_floating_point():
            flat = torch.where(torch.isnan(flat), torch.zeros_like(flat), flat)
        if self.clip_negative:
            flat = torch.clamp_min(flat, 0)

        out = (flat @ local_mapping).contiguous()

        # If input was 4D (B, T, K, N), reshape output to (B*T, K, C)
        # This makes it ready for step-by-step slicing in the main loop
        if batch_data.dim() == 4:
            B, T, K, N = batch_data.shape
            # out is currently (B*T*K, C)
            # Reshape to (B*T, K, C) so that out[step] gives (K, C) for all trials
            out = out.view(B * T, K, -1)

        return out

    def build_local_mapping(self,
                                  mapping_file: str,
                                  desired_catchment_ids: Optional[np.ndarray] = None,
                                  device: Optional[torch.device] = None,
                                  precision: Literal["float32", "float64"]="float32") -> torch.Tensor:
        """
        Build PyTorch sparse matrix for mapping grid data to specified catchments.

        This method:
        1. Loads the sparse mapping matrix from npz file
        2. Extracts submatrix for desired catchments (or all if desired_catchment_ids is None)
        3. Identifies non-zero columns (active source grids)
        4. Sets self._local_indices for use in __getitem__
        5. Returns the compressed mapping matrix

        After calling this method, data_size will return the compressed size,
        and __getitem__ will automatically extract only the needed columns.

        Args:
            mapping_file: Path to the npz file containing the mapping matrix
            desired_catchment_ids: Array of catchment IDs to include. If None, uses all
                                   catchments from the mapping file in their original order.
            device: PyTorch device for the output tensor
            precision: Data precision ("float32" or "float64")

        Returns:
            local_mapping: Stensor of shape (n_active_grids, n_catchments)
        """
        mapping_file = Path(mapping_file)
        torch_precision = torch.float32 if precision == "float32" else torch.float64
        if mapping_file is None or not os.path.exists(mapping_file):
            raise ValueError("Mapping file not found. Cannot build local matrix.")

        # Store the mapping file path for later use
        self._mapping_file = mapping_file

        # Load the scipy compressed sparsparse e matrix data
        mapping_data = np.load(mapping_file)

        all_catchment_ids = mapping_data['catchment_ids']
        sparse_data = mapping_data['sparse_data']
        sparse_indices = mapping_data['sparse_indices']
        sparse_indptr = mapping_data['sparse_indptr']
        matrix_shape = tuple(mapping_data['matrix_shape'])

        full_sparse_matrix = csr_matrix(
            (sparse_data, sparse_indices, sparse_indptr),
            shape=matrix_shape
        )

        # If desired_catchment_ids is None, use all catchments from the mapping file
        if desired_catchment_ids is None:
            desired_catchment_ids = all_catchment_ids
            desired_row_indices = np.arange(len(all_catchment_ids))
        else:
            # Use find_indices_in to get row indices for desired catchments
            desired_row_indices = find_indices_in(desired_catchment_ids, all_catchment_ids)

            # Check which catchments were found
            valid_idx = desired_row_indices != -1

            if np.any(~valid_idx):
                raise ValueError(
                    f"Some desired catchments ({np.sum(~valid_idx)}) were not found in the mapping file. "
                    "Please check your input data or mapping file."
                )

        # Verify mapping matrix columns match full grid size
        lon, lat = self.get_coordinates()
        full_grid_size = len(lon) * len(lat)
        if matrix_shape[1] != full_grid_size:
            raise ValueError(
                f"Mapping matrix columns ({matrix_shape[1]}) != full grid size ({full_grid_size}). "
                "Please regenerate the mapping file."
            )

        # Validate coordinate metadata if present in npz
        if 'coord_lon' in mapping_data and 'coord_lat' in mapping_data:
            saved_lon = mapping_data['coord_lon']
            saved_lat = mapping_data['coord_lat']

            lon_match = (len(lon) == len(saved_lon) and
                        np.allclose(lon, saved_lon, rtol=1e-5, atol=1e-8))
            lat_match = (len(lat) == len(saved_lat) and
                        np.allclose(lat, saved_lat, rtol=1e-5, atol=1e-8))

            if not lon_match or not lat_match:
                raise ValueError(
                    f"Coordinate mismatch between dataset and mapping file.\n"
                    f"Dataset: lon[{len(lon)}] range [{lon[0]:.4f}, {lon[-1]:.4f}], "
                    f"lat[{len(lat)}] range [{lat[0]:.4f}, {lat[-1]:.4f}]\n"
                    f"Mapping: lon[{len(saved_lon)}] range [{saved_lon[0]:.4f}, {saved_lon[-1]:.4f}], "
                    f"lat[{len(saved_lat)}] range [{saved_lat[0]:.4f}, {saved_lat[-1]:.4f}]\n"
                    "Please regenerate the mapping file with matching coordinates."
                )
        else:
            if is_rank_zero():
                print("[AbstractDataset] Warning: Mapping file has no coordinate metadata. "
                      "Consider regenerating for better validation.")

        # Extract submatrix for desired catchments only
        submatrix = full_sparse_matrix[desired_row_indices, :]

        initial_col_sums = np.array(submatrix.sum(axis=0)).flatten()
        initial_non_zero_cols = np.where(initial_col_sums != 0)[0].astype(np.int64)

        if len(initial_non_zero_cols) == 0:
            raise ValueError("No non-zero grid data found for the desired catchments.")

        if self.skip_nan:
            nan_mask = self._get_first_frame_nan_mask()
            if nan_mask is None:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support skip_nan mapping."
                )
            nan_mask = np.asarray(nan_mask, dtype=bool).reshape(-1)
            if nan_mask.size != full_grid_size:
                raise ValueError(
                    f"skip_nan mask size ({nan_mask.size}) != full grid size "
                    f"({full_grid_size})"
                )

            active_nan_cols = initial_non_zero_cols[nan_mask[initial_non_zero_cols]]
            if active_nan_cols.size:
                original_row_sums = np.asarray(submatrix.sum(axis=1)).ravel()
                coo = submatrix.tocoo()
                keep = ~nan_mask[coo.col]
                submatrix = csr_matrix(
                    (coo.data[keep], (coo.row[keep], coo.col[keep])),
                    shape=submatrix.shape,
                )
                valid_row_sums = np.asarray(submatrix.sum(axis=1)).ravel()

                scale = np.ones_like(original_row_sums, dtype=np.float64)
                can_scale = (original_row_sums > 0) & (valid_row_sums > 0)
                scale[can_scale] = original_row_sums[can_scale] / valid_row_sums[can_scale]
                submatrix = submatrix.multiply(scale[:, None]).tocsr()

                empty_rows = np.where((original_row_sums > 0) & (valid_row_sums <= 0))[0]
                if is_rank_zero():
                    print(
                        f"[AbstractDataset] skip_nan removed {active_nan_cols.size} "
                        "active source grids with NaN in the first frame."
                    )
                    if empty_rows.size:
                        print(
                            f"[AbstractDataset] Warning: {empty_rows.size} catchments "
                            "have no valid source grids after skip_nan masking; "
                            "their mapped values will be zero."
                        )

        # Remove columns that are all zeros to optimize memory
        col_sums = np.array(submatrix.sum(axis=0)).flatten()
        non_zero_cols = np.where(col_sums != 0)[0].astype(np.int64)

        if len(non_zero_cols) == 0:
            if not self.skip_nan:
                raise ValueError("No non-zero grid data found for the desired catchments.")
            if is_rank_zero():
                print(
                    "[AbstractDataset] Warning: no valid source grids remain "
                    "after skip_nan masking; mapped values will be zero."
                )

        # Store the local grid indices for use in __getitem__
        self._local_indices = non_zero_cols

        # Store the desired catchment IDs for use in export_catchment_data
        self._desired_catchment_ids = np.asarray(desired_catchment_ids)

        # Extract final submatrix with only non-zero columns and transpose
        # Shape: (num_source_grids, num_catchments)
        final_submatrix = submatrix[:, non_zero_cols].T.tocoo()

        # Convert to PyTorch sparse COO tensor
        row_tensor = torch.from_numpy(final_submatrix.row.astype(np.int64)).to(device)
        col_tensor = torch.from_numpy(final_submatrix.col.astype(np.int64)).to(device)
        data_tensor = torch.from_numpy(final_submatrix.data.astype(np.float32)).to(device).to(torch_precision)

        indices = torch.stack([row_tensor, col_tensor])
        local_mapping = torch.sparse_coo_tensor(
            indices, data_tensor,
            size=(len(non_zero_cols), len(desired_catchment_ids)),
            dtype=torch_precision,
            device=device
        ).coalesce()

        if is_rank_zero():
            print(f"[AbstractDataset] Built local mapping matrix: {len(non_zero_cols)} active grids "
                  f"out of {full_grid_size} total")
            print(f"[AbstractDataset] Mapping {len(non_zero_cols)} source grids -> {len(desired_catchment_ids)} catchments")

        return local_mapping

    def export_climatology(
        self,
        out_path: Union[str, Path],
        local_mapping: torch.Tensor,
        var_name: str,
        dtype: Literal["float32", "float64"] = "float32",
        complevel: int = 4,
        device: Union[str, torch.device] = "cpu",
        units: str = "m3/s",
        description: Optional[str] = None,
    ) -> Path:
        """
        Compute the temporal-mean (climatological average) and export to NetCDF.

        This mirrors the logic of Fortran-based routing models: iterate over
        every timestep in the dataset, accumulate the sum, and divide by the number
        of steps to obtain the daily-mean climatology mapped to catchments.

        Requires ``build_local_mapping()`` to be called first.
        The compressed grid data is mapped to catchments via sparse matmul and then
        time-averaged.  Output NetCDF has dimension ``(saved_points,)`` with a
        ``catchment_id`` coordinate variable.

        Args:
            out_path: Full path (including filename) for the output NetCDF file.
            local_mapping: Sparse tensor returned by ``build_local_mapping()``,
                shape ``(n_grids, n_catchments)``.
            var_name: Variable name written into the NetCDF file.
            dtype: Output data type (``"float32"`` or ``"float64"``).
            complevel: zlib compression level (0–9).
            device: Device for computation (``"cpu"`` or ``"cuda:X"``).
            units: Units attribute written to the output variable.
            description: Optional long description attribute.

        Returns:
            Path to the created NetCDF file.
        """
        if not hasattr(self, '_desired_catchment_ids') or self._desired_catchment_ids is None:
            raise ValueError(
                "build_local_mapping() must be called before "
                "export_climatology()."
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        catchment_ids = self._desired_catchment_ids
        n_catch = len(catchment_ids)

        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("[export_climatology] CUDA not available; falling back to CPU.")
            dev = torch.device("cpu")

        # Prepare transposed mapping matrix: (n_catch, n_grids)
        t_mapping = local_mapping.to(dev).to(torch_dtype)
        t_mapping_T = t_mapping.t().coalesce()

        # ----- Accumulate mean over all chunks -----
        n_chunks = len(self)
        total_steps = 0
        accumulator = torch.zeros(n_catch, dtype=torch.float64, device=dev)

        pbar = tqdm(range(n_chunks), desc="Computing climatology", unit="chunk")
        for ci in pbar:
            block = self.read_chunk(ci)  # (T, n_grids)

            # Determine how many valid steps in this block
            base_idx = ci * self.chunk_len
            T = block.shape[0]
            valid_T = sum(
                1 for k in range(T) if self.is_valid_time_index(base_idx + k)
            )
            if valid_T == 0:
                continue

            block = block[:valid_T]

            # block: (T, n_grids)
            block_t = torch.as_tensor(block, dtype=torch_dtype, device=dev)
            # (n_catch, n_grids) @ (n_grids, T) -> (n_catch, T)
            agg = torch.sparse.mm(t_mapping_T, block_t.T)
            accumulator += agg.sum(dim=1).to(torch.float64)

            total_steps += valid_T

        pbar.close()

        if total_steps == 0:
            raise RuntimeError("No valid timesteps found — cannot compute climatology.")

        mean_data = (accumulator / total_steps).cpu().numpy()
        print(f"[export_climatology] Averaged over {total_steps} timesteps")

        # ----- Write NetCDF -----
        dtype_nc = "f4" if dtype == "float32" else "f8"
        desc = description if description else f"Time-averaged {var_name} over {total_steps} steps"

        with nc.Dataset(str(out_path), "w", format="NETCDF4") as ds:
            ds.setncattr("title", f"Climatology ({var_name})")
            ds.setncattr("total_timesteps", total_steps)

            ds.createDimension("saved_points", n_catch)

            cid_var = ds.createVariable("catchment_id", "i8", ("saved_points",))
            cid_var[:] = catchment_ids

            out_var = ds.createVariable(
                var_name, dtype_nc, ("saved_points",),
                zlib=True, complevel=int(complevel),
            )
            out_var[:] = mean_data.astype(
                np.float32 if dtype == "float32" else np.float64
            )
            out_var.setncattr("description", desc)
            out_var.setncattr("units", units)

        print(f"[export_climatology] Saved to {out_path}")
        return out_path

    def export_catchment_data(
        self,
        out_dir: str | Path,
        local_mapping: torch.Tensor,
        var_name: str = "var",
        dtype: Literal["float32", "float64"] = "float32",
        complevel: int = 4,
        normalized: bool = False,
        device: str | torch.device = "cpu",
        split_by_year: bool = False,
        units: Union[str, Dict[str, str]] = "m3/s",
        description: Optional[Union[str, Dict[str, str]]] = None,
        filename: Optional[Union[str, Dict[str, str]]] = None,
        chunksizes: Optional[tuple] = None,
    ) -> Union[Path, List[Path], Dict[str, Path], Dict[str, List[Path]]]:
        """
        Export catchment-aggregated data to a NetCDF file readable by MultiRankStatsReader.

        Requires build_local_mapping() to be called first.

        - Output filename: {filename}_rank0.nc or {var_name}_rank0.nc if filename not specified
          (or with _{year} suffix if split_by_year)
        - Dimensions: time (unlimited), saved_points
        - Variables:
            * time: numeric with units and calendar
            * catchment_id: (saved_points,) catchment IDs
            * {var_name}: (time, saved_points) aggregated data (area-weighted mean)

        GPU acceleration:
        - Set `device="cuda:0"` (or any CUDA device) to enable GPU-accelerated sparse matmul.
        - Falls back to CPU if CUDA is not available.

        Args:
            out_dir: Output directory for NetCDF files
            local_mapping: Sparse tensor from build_local_mapping(), shape (n_grids, n_catchments)
            var_name: Variable name in output NetCDF
            dtype: Output data type
            complevel: Compression level (0-9)
            normalized: If True, normalize mapping weights to sum to 1 per catchment
            device: Device for computation ("cpu" or "cuda:X")
            split_by_year: If True, create separate files per year
            units: Units string for the output variable
            description: Optional description for the output variable
            filename: Optional custom filename prefix (default: var_name)
            chunksizes: Optional NetCDF chunk sizes for the data variable, e.g. (365, 1).
                If None, uses netCDF4 default. Setting (T, 1) optimizes per-station
                time series reads at the cost of slower writes.
        """
        if not isinstance(var_name, str):
            raise TypeError(
                "var_name must be a string; define time_aggregation when "
                "constructing the dataset."
            )

        active_aggregation = getattr(self, "time_aggregation", None)

        if isinstance(active_aggregation, dict):
            if not self.supports_time_aggregation:
                raise ValueError(
                    f"{type(self).__name__} does not support time aggregation."
                )
            output_methods = active_aggregation
            returns_mapping = True
        else:
            if active_aggregation is not None and not self.supports_time_aggregation:
                raise ValueError(
                    f"{type(self).__name__} does not support time aggregation."
                )
            output_methods = {str(var_name): active_aggregation}
            returns_mapping = False

        # Require build_local_mapping() to be called first
        if not hasattr(self, '_desired_catchment_ids') or self._desired_catchment_ids is None:
            raise ValueError(
                "build_local_mapping() must be called before export_catchment_data(). "
                "This sets the catchment IDs and grid mapping."
            )

        catchment_ids = self._desired_catchment_ids
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_catch = len(catchment_ids)
        n_cols = self.data_size  # This is the compressed size after build_local_mapping

        # Prepare device and torch types
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        numpy_dtype = np.float32 if dtype == "float32" else np.float64
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU for export_catchment_data.")
            dev = torch.device("cpu")

        # Use the provided local mapping matrix
        # Shape: (n_cols, n_catch) - maps compressed source grids to catchments
        t_mapping = local_mapping.to(dev).to(torch_dtype)

        if normalized:
            # Normalize by row sums (each catchment's total area)
            # t_mapping shape: (n_cols, n_catch)
            # We need to normalize columns (each catchment)
            col_sums = torch.sparse.sum(t_mapping, dim=0).to_dense()  # (n_catch,)
            # Create a diagonal scaling matrix or normalize in-place
            # For COO tensor, we need to work with the values
            t_mapping = t_mapping.coalesce()
            indices = t_mapping.indices()  # (2, nnz)
            values = t_mapping.values()    # (nnz,)
            col_indices = indices[1]       # column index for each value
            col_sums_expanded = col_sums[col_indices]
            nz_mask = col_sums_expanded > 0
            new_values = torch.zeros_like(values)
            new_values[nz_mask] = values[nz_mask] / col_sums_expanded[nz_mask]
            t_mapping = torch.sparse_coo_tensor(
                indices, new_values, t_mapping.size(), dtype=torch_dtype, device=dev
            ).coalesce()

        # Pre-compute transposed mapping matrix for efficient batch multiplication
        # t_mapping shape: (n_cols, n_catch)
        # t_mapping_T shape: (n_catch, n_cols) for sparse.mm(sparse, dense)
        t_mapping_T = t_mapping.t().coalesce()

        dtype_nc = "f4" if dtype == "float32" else "f8"

        def _metadata(metadata, name, default):
            if metadata is None:
                return default
            if isinstance(metadata, str):
                return metadata
            return metadata.get(name, default)

        def _file_prefix(name):
            if filename is None:
                return name
            if isinstance(filename, str):
                if len(output_methods) == 1:
                    return filename
                return f"{filename}_{name}"
            return filename.get(name, name)

        def _init_nc(path, name, method):
            ds = nc.Dataset(path, "w", format="NETCDF4")
            ds.setncattr("title", f"Aggregated catchment data ({name})")
            if method is not None:
                ds.setncattr("time_aggregation", method)
            ds.createDimension("time", None)
            ds.createDimension("saved_points", n_catch)

            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.setncattr("units", "seconds since 1900-01-01 00:00:00")
            time_var.setncattr("calendar", "standard")

            save_coord = ds.createVariable("catchment_id", "i8", ("saved_points",))
            save_coord[:] = catchment_ids

            out_var = ds.createVariable(
                name,
                dtype_nc,
                ("time", "saved_points"),
                zlib=True,
                complevel=int(complevel),
                chunksizes=chunksizes,
            )
            default_desc = (
                f"Catchment-aggregated {name} ({method})"
                if method is not None
                else f"Catchment-aggregated {name} (area-weighted mean)"
            )
            desc = _metadata(description, name, default_desc)
            out_var.setncattr("description", desc)
            out_var.setncattr("units", _metadata(units, name, ""))
            return ds, time_var, out_var

        writers = {}
        created_files = {name: [] for name in output_methods}
        current_year = None
        write_idx = 0
        total_steps = self.num_main_steps + self.num_spin_up_steps

        def _close_writers():
            nonlocal writers
            for ds, _time_var, _out_var in writers.values():
                ds.close()
            writers = {}

        def _open_writers(year=None):
            nonlocal write_idx
            _close_writers()
            for name, method in output_methods.items():
                if year is None:
                    nc_path = out_dir / f"{_file_prefix(name)}_rank0.nc"
                else:
                    nc_path = out_dir / f"{_file_prefix(name)}_rank0_{year}.nc"
                writers[name] = _init_nc(nc_path, name, method)
                created_files[name].append(nc_path)
            write_idx = 0

        pbar = None
        try:
            if not split_by_year:
                _open_writers()

            n_chunks = len(self)
            pbar = tqdm(total=total_steps, desc="Exporting", unit="step")
            for ci in range(n_chunks):
                base_idx = ci * self.chunk_len
                read_data = self.read_chunk(ci)
                if isinstance(read_data, dict):
                    blocks = read_data
                    if set(blocks) != set(output_methods):
                        raise ValueError(
                            f"read_chunk returned variables {sorted(blocks)}, "
                            f"but export expected {sorted(output_methods)}"
                        )
                else:
                    if len(output_methods) != 1:
                        raise ValueError(
                            "Multiple time aggregations require read_chunk() "
                            "to return a dict"
                        )
                    name = next(iter(output_methods))
                    blocks = {name: read_data}

                first_block = next(iter(blocks.values()))
                T = int(first_block.shape[0])
                mapped_blocks = {}
                for name, block in blocks.items():
                    if block.ndim != 2 or block.shape[1] != n_cols:
                        raise ValueError(
                            f"Data block shape {tuple(block.shape)} incompatible with "
                            f"mapping columns {n_cols} at chunk {ci}. "
                            "Please call build_local_mapping() before "
                            "export_catchment_data()."
                        )
                    if int(block.shape[0]) != T:
                        raise ValueError("All exported variables must share chunk length")

                    # t_mapping_T @ block.T = (n_catch, n_cols) @ (n_cols, T)
                    block_tensor = torch.as_tensor(block, dtype=torch_dtype, device=dev)
                    agg_block = torch.sparse.mm(t_mapping_T, block_tensor.T)
                    mapped_blocks[name] = agg_block.T.contiguous().to("cpu").numpy()

                # Write each timestep in the block
                for k in range(T):
                    dt_k = self.get_time_by_index(base_idx + k)

                    if split_by_year:
                        year = dt_k.year
                        if year != current_year:
                            current_year = year
                            _open_writers(current_year)

                    for name in output_methods:
                        _ds, time_var, out_var = writers[name]
                        out_var[write_idx, :] = mapped_blocks[name][k, :].astype(
                            numpy_dtype, copy=False
                        )
                        time_val = nc.date2num(
                            dt_k,
                            units=time_var.getncattr("units"),
                            calendar=time_var.getncattr("calendar"),
                        )
                        time_var[write_idx] = time_val
                    write_idx += 1
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()
            _close_writers()

        if returns_mapping:
            if split_by_year:
                return created_files
            return {name: paths[0] for name, paths in created_files.items()}

        only_name = next(iter(output_methods))
        return created_files[only_name] if split_by_year else created_files[only_name][0]

    def generate_mapping_table(
        self,
        map_dir: str,
        out_dir: str,
        npz_file: str = "grid_mapping.npz",
        mapinfo_txt: str = "location.txt",
        hires_tag: str = "1min",
        lowres_idx_precision: str = "<i4",
        hires_idx_precision: str = "<i2",
        map_precision: str = "<f4",
        parameter_nc: str | Path | None = None,
        allow_oob_zero: bool = False,
    ):
        """Generate the CaMa grid mapping table and save it as an npz file.

        Thin convenience wrapper: delegates the orchestration to
        :func:`hydroforge.io.aggregate.build_cama_mapping` using this dataset's
        source coordinates, then saves the resulting :class:`MappingTable`.  When
        ``parameter_nc`` is given, rows are aligned/subset to its ``catchment_id``
        order.
        """
        ro_lon, ro_lat = self.get_coordinates()
        mapping = build_cama_mapping(
            ro_lon,
            ro_lat,
            map_dir,
            hires_tag=hires_tag,
            mapinfo_txt=mapinfo_txt,
            lowres_idx_precision=lowres_idx_precision,
            hires_idx_precision=hires_idx_precision,
            map_precision=map_precision,
            parameter_nc=parameter_nc,
            allow_oob_zero=allow_oob_zero,
            producer=f"{type(self).__name__}.generate_mapping_table",
        )
        output_path = Path(out_dir) / npz_file
        mapping.save(output_path)
        print(f"Saved grid mapping to {output_path}")
        print(
            f"Mapping contains {mapping.matrix.shape[0]} catchments "
            f"and {mapping.matrix.nnz} non-zero source grid mappings"
        )
        print(f"Matrix shape: {mapping.matrix.shape[0]} x {mapping.matrix.shape[1]}")
        print(f"Coordinate metadata: lon[{len(ro_lon)}] x lat[{len(ro_lat)}]")

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
        raise NotImplementedError

    @abstractmethod
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        To be implemented by subclasses, returns the coordinates of the dataset.
        """
        raise NotImplementedError

    def get_time_by_index(self, idx: int) -> Union[datetime, cftime.datetime]:
        """
        Returns the datetime corresponding to the given index.
        Default implementation handles spin-up and linear time stepping.
        """
        if self.time_interval is None:
             raise NotImplementedError("time_interval must be provided for default get_time_by_index")

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
        return int(offset.total_seconds() / self.time_interval.total_seconds())

    @property
    def num_main_steps(self) -> int:
        if self.start_date is None or self.end_date is None or self.time_interval is None:
            return 0
        duration = self.end_date - self.start_date
        return int(duration.total_seconds() / self.time_interval.total_seconds()) + 1

    @property
    def num_spin_up_steps(self) -> int:
        if self.spin_up_cycles <= 0:
            return 0
        cycle_duration = self.spin_up_end_date - self.spin_up_start_date
        steps_per_cycle = int(cycle_duration.total_seconds() / self.time_interval.total_seconds()) + 1
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
            if not hasattr(self, '_spin_up_num_chunks'):
                 self._calc_spin_up_params()

            chunks_per_cycle = self._spin_up_num_chunks
            items_per_cycle = chunks_per_cycle * self.chunk_len
            total_spin_up_items = items_per_cycle * self.spin_up_cycles

            if idx < total_spin_up_items:
                # In spin-up region
                idx_in_cycle = idx % items_per_cycle

                # Calculate valid steps per cycle
                cycle_duration = self.spin_up_end_date - self.spin_up_start_date
                steps_per_cycle = int(cycle_duration.total_seconds() / self.time_interval.total_seconds()) + 1

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
        Closes any open resources or files.
        """

    @property
    def data_size(self) -> int:
        """
        Returns the number of source grid points to be loaded per timestep.

        Before build_local_mapping is called: returns full grid size.
        After build_local_mapping is called: returns compressed size (active grids only).
        """
        if self._local_indices is not None:
            return len(self._local_indices)
        # Full grid size from coordinates
        lon, lat = self.get_coordinates()
        return len(lon) * len(lat)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """
        Returns (ny, nx) = (lat_size, lon_size) grid dimensions.

        Spatial convention: (Y, X) = (lat, lon)
        """
        lon, lat = self.get_coordinates()
        return (len(lat), len(lon))

    def _combine(self, other, operation, reverse=False):
        is_dataset = isinstance(other, AbstractDataset)
        is_scalar = isinstance(other, (int, float, np.number))

        if not (is_dataset or is_scalar):
            return NotImplemented

        operands = [self, other] if not reverse else [other, self]
        return MixedDataset(operands, operation=operation)

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


class MixedDataset(AbstractDataset):
    """
    A dataset that combines multiple datasets (or scalars) by applying an operation.
    """
    def __init__(self, operands: List[Union[AbstractDataset, float, int]], operation: str = "add"):
        if not operands:
            raise ValueError("operands list cannot be empty")

        base = None
        for op in operands:
            if isinstance(op, AbstractDataset):
                base = op
                break

        if base is None:
            raise ValueError("MixedDataset requires at least one AbstractDataset operand")

        self.base_dataset = base
        self.operands = []

        can_flatten = operation in ["add", "mul"]

        for op in operands:
            if can_flatten and isinstance(op, MixedDataset) and op.operation == operation:
                self.operands.extend(op.operands)
            else:
                self.operands.append(op)

        for i, op in enumerate(self.operands):
            if isinstance(op, AbstractDataset) and op is not base:
                if op.start_date != base.start_date:
                    raise ValueError(f"Operand {i} has different start_date")
                if op.end_date != base.end_date:
                    raise ValueError(f"Operand {i} has different end_date")
                if op.time_interval != base.time_interval:
                    raise ValueError(f"Operand {i} has different time_interval")
                if op.chunk_len != base.chunk_len:
                    raise ValueError(f"Operand {i} has different chunk_len")
                if op.data_size != base.data_size:
                    raise ValueError(f"Operand {i} has different data_size")

        # Initialize AbstractDataset using the base dataset's attributes
        super().__init__(
            start_date=base.start_date,
            end_date=base.end_date,
            time_interval=base.time_interval,
            out_dtype=base.out_dtype,
            chunk_len=base.chunk_len,
            spin_up_cycles=base.spin_up_cycles,
            spin_up_start_date=base.spin_up_start_date,
            spin_up_end_date=base.spin_up_end_date,
            calendar=base.calendar,
            clip_negative=base.clip_negative,
            skip_nan=base.skip_nan,
        )
        self.operation = operation

    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        def _fetch(op):
            if isinstance(op, AbstractDataset):
                return op.get_data(current_time, chunk_len)
            return op

        data = _fetch(self.operands[0])

        for op in self.operands[1:]:
            val = _fetch(op)
            if self.operation == "add":
                data = data + val
            elif self.operation == "sub":
                data = data - val
            elif self.operation == "mul":
                data = data * val
            elif self.operation == "div":
                data = data / val
            else:
                raise NotImplementedError(f"Operation {self.operation} not implemented")

        return data

    @property
    def data_mask(self) -> np.ndarray:
        return self.base_dataset.data_mask

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.base_dataset.get_coordinates()

    def close(self) -> None:
        for op in self.operands:
            if hasattr(op, 'close'):
                op.close()

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cftime
import netCDF4 as nc
import numpy as np

from hydroforge.aggregator.utils import is_wsl, sanitize_symbol

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator


# ---------------------------------------------------------------------------
# Default cap on per-submit IPC payload (bytes).  Each subprocess receives
# a pickled numpy array; keeping the payload bounded avoids excessive memory
# copies.  256 MB is a safe default for machines with ≥8 GB RAM.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_IPC_BYTES: int = 256 * 1024 * 1024
_DEFAULT_MAX_BATCH: int = 30


def compute_write_batch_size(
    saved_points: int,
    dtype_bytes: int = 4,
    max_ipc_bytes: int = _DEFAULT_MAX_IPC_BYTES,
    max_batch: int = _DEFAULT_MAX_BATCH,
) -> int:
    """Return the number of time steps to batch per subprocess write.

    The batch is capped so that ``batch * saved_points * dtype_bytes``
    does not exceed *max_ipc_bytes*, giving an automatic fallback to
    single-step writes for very large grids (e.g. glb_01min).
    """
    per_step = saved_points * dtype_bytes
    batch = int(max_ipc_bytes / max(per_step, 1))
    return max(1, min(batch, max_batch))


def _find_data_variable(ncfile, var_name: str):
    """Locate the target data variable inside an open NetCDF dataset."""
    safe = sanitize_symbol(var_name)
    if var_name in ncfile.variables:
        return var_name
    if safe in ncfile.variables:
        return safe
    _skip = {'time', 'trial', 'saved_points', 'levels'}
    for v in ncfile.variables:
        vobj = ncfile.variables[v]
        if v in _skip:
            continue
        if vobj.dimensions == ('saved_points',) and vobj.dtype.kind in ('i', 'u'):
            continue
        return v
    raise KeyError(
        f"Could not find variable for '{var_name}' (safe: '{safe}') in {ncfile.filepath()}"
    )


def _wsl_drop_cache(output_path) -> None:
    """WSL optimisation: advise the kernel to drop page-cache for *output_path*."""
    if is_wsl() and hasattr(os, 'posix_fadvise'):
        try:
            with open(output_path, 'rb') as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass


def _write_time_step_netcdf_process(args: Tuple[Any, ...]) -> Tuple[str, int]:
    """Write a single time step to a NetCDF file. Each file contains a single variable."""
    (var_name, time_step_data, output_path, time_datetime) = args

    with nc.Dataset(output_path, 'a') as ncfile:
        time_var = ncfile.variables['time']
        target_var = _find_data_variable(ncfile, var_name)
        nc_var = ncfile.variables[target_var]
        current_len = len(nc_var)

        # Append data
        if time_step_data.ndim == 1:
            nc_var[current_len, :] = time_step_data
        elif time_step_data.ndim == 2:
            nc_var[current_len, :, :] = time_step_data
        elif time_step_data.ndim == 3:
            nc_var[current_len, :, :, :] = time_step_data

        # Append datetime
        time_unit = time_var.getncattr("units")
        calendar = time_var.getncattr("calendar")
        time_val = nc.date2num(time_datetime, units=time_unit, calendar=calendar)
        time_var[current_len] = time_val

    _wsl_drop_cache(output_path)
    return (var_name, current_len)


def _write_batch_netcdf_process(args: Tuple[Any, ...]) -> Tuple[str, int]:
    """Write multiple time steps in one file open/close cycle.

    Expects *args* = ``(var_name, data_batch, output_path, dt_list)`` where
    *data_batch* has shape ``(N, ...)``, and *dt_list* is a list of N
    datetime objects.
    """
    (var_name, data_batch, output_path, dt_list) = args
    n_steps = data_batch.shape[0]

    with nc.Dataset(output_path, 'a') as ncfile:
        time_var = ncfile.variables['time']
        target_var = _find_data_variable(ncfile, var_name)
        nc_var = ncfile.variables[target_var]
        current_len = len(nc_var)

        time_unit = time_var.getncattr("units")
        calendar = time_var.getncattr("calendar")

        for i in range(n_steps):
            row = data_batch[i]
            t = current_len + i
            if row.ndim == 1:
                nc_var[t, :] = row
            elif row.ndim == 2:
                nc_var[t, :, :] = row
            elif row.ndim == 3:
                nc_var[t, :, :, :] = row

            time_var[t] = nc.date2num(dt_list[i], units=time_unit, calendar=calendar)

    _wsl_drop_cache(output_path)
    return (var_name, current_len + n_steps - 1)



def _create_netcdf_file_process(args: Tuple[Any, ...]) -> Union[Path, List[Path]]:
    """
    Process function for creating empty NetCDF files with proper structure.
    This function runs in a separate process.

    Args:
        args: Tuple containing (mean_var_name, metadata, coord_values,
              output_dir, complevel, rank, year, calendar, time_unit, num_trials)

    Returns:
        Path or List[Path] to the created NetCDF file(s)
    """
    (mean_var_name, metadata, coord_values, output_dir, complevel, rank, year, calendar, time_unit, num_trials, *extra) = args
    chunksizes = extra[0] if extra else None

    safe_name = sanitize_symbol(mean_var_name)

    actual_shape = metadata.get('actual_shape', ())  # Spatial shape
    tensor_shape = metadata.get('tensor_shape', ())  # Logical grid shape
    # nc_coord_name is derived from dim_coords (e.g. "catchment_id").
    coord_name = metadata.get('nc_coord_name')
    dtype = metadata.get('dtype', 'f8')
    k_val = metadata.get('k', 1)

    # Helper to create a single NetCDF file
    def create_single_file(file_safe_name: str, file_var_name: str, description_suffix: str = "") -> Path:
        if year is not None:
            filename = f"{file_safe_name}_rank{rank}_{year}.nc"
        else:
            filename = f"{file_safe_name}_rank{rank}.nc"
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
            # Write global attributes
            ncfile.setncattr('title', f'Time series for rank {rank}: {file_var_name}')
            ncfile.setncattr('original_variable_name', file_var_name)

            # Create time dimension (unlimited for streaming)
            ncfile.createDimension('time', None)

            # Create spatial/vertical dimensions based on actual shape
            dim_names = ['time']  # Always start with time

            if num_trials > 1:
                dim_names.append('trial')
                ncfile.createDimension('trial', num_trials)

                dim_names.append('saved_points')
                ncfile.createDimension('saved_points', actual_shape[1])

                if len(actual_shape) > 2:
                    dim_names.append('levels')
                    ncfile.createDimension('levels', actual_shape[2])
            else:
                if len(actual_shape) == 1:
                    # 1D spatial: time + saved points
                    dim_names.append('saved_points')
                    ncfile.createDimension('saved_points', actual_shape[0])
                elif len(actual_shape) == 2:
                    # 2D: time + saved points + levels
                    dim_names.extend(['saved_points', 'levels'])
                    ncfile.createDimension('saved_points', actual_shape[0])
                    ncfile.createDimension('levels', actual_shape[1])

            if coord_name and coord_values is not None:
                coord_var = ncfile.createVariable(
                    coord_name,
                    coord_values.dtype,
                    ('saved_points',),
                )
                coord_var[:] = coord_values

            time_var = ncfile.createVariable('time', 'f8', ('time',))
            time_var.setncattr('units', time_unit)
            time_var.setncattr('calendar', calendar)

            # Create single data variable
            nc_var = ncfile.createVariable(
                file_safe_name,
                dtype,
                dim_names,
                zlib=True,
                complevel=complevel,
                chunksizes=chunksizes)
            desc = metadata.get("description", "") + description_suffix
            nc_var.setncattr('description', desc)
            nc_var.setncattr('actual_shape', str(actual_shape))
            nc_var.setncattr('tensor_shape', str(tensor_shape))
            nc_var.setncattr('long_name', file_var_name)

        return output_path

    # For k > 1, create separate files for each k index
    if k_val > 1:
        paths = []
        for k_idx in range(k_val):
            file_safe_name = f"{safe_name}_{k_idx}"
            file_var_name = f"{mean_var_name}_{k_idx}"
            desc_suffix = f" [rank {k_idx}]"
            path = create_single_file(file_safe_name, file_var_name, desc_suffix)
            paths.append(path)
        return paths
    else:
        return create_single_file(safe_name, mean_var_name)



class NetCDFIOMixin:
    """Mixin providing NetCDF file creation and streaming write functionality."""

    def _create_netcdf_files(self: StatisticsAggregator, year: Optional[int] = None) -> None:
        """Create empty NetCDF files with proper structure for streaming."""
        if self.in_memory_mode:
            # Skip file creation in in-memory mode
            return

        if not self.output_split_by_year and self._files_created:
            return

        print(f"Creating NetCDF file structure...{' (Year: ' + str(year) + ')' if year else ''}")

        # Prepare file creation tasks
        creation_futures = {}
        # Use number of outputs instead of variables (supports multiple ops)
        n_outputs = len(self._metadata)
        actual_workers = max(1, min(self.num_workers, n_outputs))

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            items = list(self._metadata.items())
            for out_name, metadata in items:
                coord_name = metadata.get('save_coord')
                coord_values = self._coord_cache.get(coord_name, None)
                args = (out_name, metadata, coord_values, self.output_dir, self.complevel, self.rank, year, self.calendar, self.time_unit, self.num_trials, self.output_chunksizes)
                future = executor.submit(_create_netcdf_file_process, args)
                creation_futures[future] = (out_name, metadata.get('k', 1))

            # Collect results
            for future in as_completed(creation_futures):
                out_name, k_val = creation_futures[future]
                try:
                    result = future.result()
                    # Handle both single path and list of paths (for k > 1)
                    if isinstance(result, list):
                        # Multiple files for k > 1
                        self._netcdf_files[out_name] = result  # Store as list
                        for p in result:
                            self._all_created_files.add(p)
                            print(f"  Created {p.name}")
                    else:
                        self._netcdf_files[out_name] = result
                        self._all_created_files.add(result)
                        print(f"  Created {result.name}")
                except Exception as exc:
                    print(f"  Failed to create file for {out_name}: {exc}")
                    raise exc

        self._files_created = True
        total_files = sum(len(v) if isinstance(v, list) else 1 for v in self._netcdf_files.values())
        print(f"Created {total_files} NetCDF files for streaming")

        # --- Initialise write-batch buffers ---
        # Each buffer key mirrors a submit key (out_name or out_name_kN).
        # Value: dict(data=[], dt=[], path=Path)
        self._write_buffers: Dict[str, Dict[str, Any]] = {}
        # Compute the adaptive batch size from the first registered output
        first_meta = next(iter(self._metadata.values()), {})
        actual_shape = first_meta.get('actual_shape', (1,))
        # saved_points is the first spatial dim (or second if trials present)
        sp = actual_shape[1] if self.num_trials > 1 and len(actual_shape) > 1 else actual_shape[0] if actual_shape else 1
        dtype_bytes = np.dtype(first_meta.get('dtype', 'f4')).itemsize
        self._write_batch_size = compute_write_batch_size(sp, dtype_bytes)
        print(f"  Write batch size: {self._write_batch_size} steps (saved_points={sp:,})")


    def _flush_write_buffer(self: StatisticsAggregator, key: str) -> None:
        """Submit a buffered batch to the executor and clear the buffer.

        Falls back to synchronous write if the executor is already shut down
        (e.g. during interpreter shutdown).
        """
        buf = self._write_buffers.get(key)
        if not buf or len(buf['data']) == 0:
            return

        data_batch = np.stack(buf['data'], axis=0)
        dt_list = list(buf['dt'])
        output_path = buf['path']
        var_name = buf['var_name']

        if data_batch.shape[0] == 1:
            args = (var_name, data_batch[0], output_path, dt_list[0])
            writer = _write_time_step_netcdf_process
        else:
            args = (var_name, data_batch, output_path, dt_list)
            writer = _write_batch_netcdf_process

        try:
            idx = abs(hash(key)) % len(self._write_executors)
            future = self._write_executors[idx].submit(writer, args)
            self._write_futures.append(future)
        except RuntimeError:
            # Executor already shut down — write synchronously
            writer(args)

        buf['data'].clear()
        buf['dt'].clear()


    def _flush_all_write_buffers(self: StatisticsAggregator) -> None:
        """Flush every pending write buffer (called on year transition / shutdown)."""
        for key in list(self._write_buffers):
            self._flush_write_buffer(key)


    def _buffer_and_maybe_flush(
        self: StatisticsAggregator,
        buf_key: str,
        var_name: str,
        data: np.ndarray,
        output_path,
        dt: Union[datetime, cftime.datetime],
        batch_size: int,
    ) -> None:
        """Append one time step to a write buffer; flush when full."""
        if buf_key not in self._write_buffers:
            self._write_buffers[buf_key] = dict(data=[], dt=[], path=output_path, var_name=var_name)

        buf = self._write_buffers[buf_key]
        # Update path in case of year-split (new file for new year)
        buf['path'] = output_path
        buf['data'].append(data.copy())
        buf['dt'].append(dt)

        if len(buf['data']) >= batch_size:
            self._flush_write_buffer(buf_key)


    def _finalize_time_step_in_memory(self: StatisticsAggregator, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize time step in in-memory mode by copying storage to result tensors.

        Args:
            dt: Time step to finalize
        """
        # Increment macro step index for next iteration
        self._macro_step_index += 1

        # Get dirty outputs to write
        keys_to_write = [k for k in self._output_keys if k in self._dirty_outputs]
        self._dirty_outputs.clear()

        # Append storage tensors to result lists
        for out_name in keys_to_write:
            if out_name not in self._result_tensors:
                continue

            storage_tensor = self._storage[out_name]

            # Clone and move to result device (default CPU)
            # This frees GPU memory and allows dynamic growth
            result_copy = storage_tensor.detach().clone().to(self.result_device)
            self._result_tensors[out_name].append(result_copy)

        # Advance time index
        self._current_time_index += 1

        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True


    def finalize_time_step(self: StatisticsAggregator, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize the current time step by writing results to output.

        In streaming mode: writes to NetCDF files incrementally.
        In in-memory mode: copies current storage to result tensors.

        Args:
            dt: Time step to finalize (datetime or cftime.datetime)
        """
        # Error if compound ops are configured but outer flags have never been set
        if (self._has_compound_ops
                and not self._outer_flags_ever_seen):
            compound_ops = [
                f"{meta['original_variable']}.{meta['op']}"
                for name, meta in self._metadata.items()
                if self._output_is_outer.get(name, False)
            ]
            raise RuntimeError(
                f"Compound statistics {compound_ops} are configured but "
                f"stat_is_outer_first / stat_is_outer_last flags have never "
                f"been set to True. These outputs will never be written. "
                f"Set stat_is_outer_first=True at the start of each outer "
                f"window (e.g. Jan 1) and stat_is_outer_last=True at the "
                f"end (e.g. Dec 31) in step_advance()."
            )

        # Record this time step for argmax/argmin index-to-time conversion
        # This is called at the end of each outer loop iteration
        self._macro_step_times.append(dt)

        # Handle in-memory mode
        if self.in_memory_mode:
            self._finalize_time_step_in_memory(dt)
            return

        if self.output_split_by_year:
            if self._current_year is None:
                # First call - set up files
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
            elif self._current_year != dt.year:
                # Year transition – flush remaining buffers for the old year first
                self._flush_all_write_buffers()
                # Year transition - create new files for new year
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
                self._macro_step_times = [dt]  # Reset time mapping for new year (keep current dt)
        else:
            # Create NetCDF files if not already created
            if not self._files_created:
                self._create_netcdf_files()

        # Increment macro step index for next iteration
        # (Note: index is reset to 0 in update_statistics when is_outer_first=True)
        self._macro_step_index += 1

        # Write all outputs that are marked dirty
        # Use explicit list of output keys to maintain order/determinism
        keys_to_write = [k for k in self._output_keys if k in self._dirty_outputs]

        # Clear dirty set for next step
        self._dirty_outputs.clear()

        batch_size = getattr(self, '_write_batch_size', 1)

        for out_name in keys_to_write:
            tensor = self._storage[out_name]

            if out_name not in self._netcdf_files:
                continue
            output_paths = self._netcdf_files[out_name]

            # Check if this is a time index variable (argmax/argmin)
            metadata = self._metadata.get(out_name, {})
            is_time_index = metadata.get('is_time_index', False)
            k_val = metadata.get('k', 1)

            # Convert tensor to numpy
            raw_data = tensor.detach().cpu().numpy()

            # For time index variables, keep as integer indices (no conversion to time)
            # They will be stored as int32 in the NC file
            if is_time_index:
                time_step_data = raw_data.astype(np.int32)
            else:
                time_step_data = raw_data

            # Handle k > 1 case: write to separate files
            if k_val > 1 and isinstance(output_paths, list):
                for k_idx, output_path in enumerate(output_paths):
                    # Extract k-th slice (last dimension is k)
                    if time_step_data.ndim == 2:
                        k_data = time_step_data[:, k_idx]
                    elif time_step_data.ndim == 3:
                        k_data = time_step_data[:, :, k_idx]
                    elif time_step_data.ndim == 4:
                        k_data = time_step_data[:, :, :, k_idx]
                    else:
                        k_data = time_step_data[..., k_idx]

                    buf_key = f"{out_name}_{k_idx}"
                    file_var_name = f"{out_name}_{k_idx}"
                    self._buffer_and_maybe_flush(buf_key, file_var_name, k_data, output_path, dt, batch_size)
            else:
                # Single file case (k=1 or legacy)
                output_path = output_paths if not isinstance(output_paths, list) else output_paths[0]
                self._buffer_and_maybe_flush(out_name, out_name, time_step_data, output_path, dt, batch_size)

        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True

        # Manage backlog: Wait if too many steps are pending
        batch_n = len(self._storage)
        max_futures = self.max_pending_steps * batch_n

        while len(self._write_futures) > max_futures:
            # Pop the oldest future and wait for it
            future = self._write_futures.pop(0)
            try:
                future.result()
            except Exception as exc:
                print(f"  Failed to write time step (backlog): {exc}")
                raise exc

        # If we are strictly synchronous (max_pending_steps=1), we can clear the list
        # to keep it perfectly clean, although the loop above handles it too.
        if self.max_pending_steps == 1 and len(self._write_futures) >= batch_n:
             # Wait for the current batch completely (old behavior)
             for future in self._write_futures:
                 try:
                     future.result()
                 except Exception as exc:
                     print(f"  Failed to write time step {dt}: {exc}")
                     raise exc
             self._write_futures.clear()

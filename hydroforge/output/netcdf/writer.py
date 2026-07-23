# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, List, Optional, Union

import cftime
import netCDF4 as nc
import numpy as np

from hydroforge.contracts import ResourceCleanupError
from hydroforge.contracts.naming import sanitize_symbol
from hydroforge.output.netcdf.plan import (
    COMMITTED_STEPS_ATTR, OUTPUT_FORMAT, OUTPUT_VERSION,
    NetCDFCreateRequest, NetCDFWriteRequest, OutputFilePlan,
)
from hydroforge.output.netcdf.schema import NetCDFSchema
from hydroforge.contracts.events import ModelEvent
from hydroforge.serialization.netcdf import (
    atomic_netcdf_dataset, normalize_netcdf_variable_options,
)


def _is_wsl() -> bool:
    import sys

    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/version", encoding="utf-8") as stream:
            version = stream.read().lower()
        return "microsoft" in version or "wsl" in version
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Default cap on per-submit IPC payload (bytes).  Each subprocess receives
# a pickled numpy array; keeping the payload bounded avoids excessive memory
# copies.  256 MB is a safe default for machines with ≥8 GB RAM.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_IPC_BYTES: int = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PendingNetCDFWrite:
    """One background write with an exact per-output timestep weight."""

    key: str
    step_count: int
    future: Any

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("pending NetCDF write key must be non-empty")
        if type(self.step_count) is not int or self.step_count < 1:
            raise ValueError("pending NetCDF write step_count must be positive")
        if not callable(getattr(self.future, "result", None)):
            raise TypeError("pending NetCDF write future must define result()")
        if not callable(getattr(self.future, "done", None)):
            raise TypeError("pending NetCDF write future must define done()")


def _validated_netcdf_options(
    variable: str,
    options: dict[str, Any],
    dimensions: list[str],
) -> dict[str, Any]:
    create_options = normalize_netcdf_variable_options(options)
    chunksizes = create_options.get("chunksizes")
    if chunksizes is not None and len(chunksizes) != len(dimensions):
        raise ValueError(
            f"NetCDF chunksizes for {variable!r} have rank {len(chunksizes)}, "
            f"but dimensions {dimensions} have rank {len(dimensions)}"
        )
    return create_options


def _static_variable_applies(
    name: str,
    specification: dict[str, Any],
    *,
    output_coordinate: str | None,
    dimensions,
) -> bool:
    """Resolve an explicitly coordinate-scoped static variable."""

    if specification["coordinate"] != output_coordinate:
        return False
    dimension = specification["dim"]
    if dimension not in dimensions:
        raise ValueError(
            f"static variable {name!r} targets coordinate "
            f"{output_coordinate!r} and dimension {dimension!r}, but that "
            "dimension is absent from the output schema"
        )
    return True
_DEFAULT_MAX_BATCH: int = 30


def compute_write_batch_size(
    saved_points: int,
    dtype_bytes: int = 4,
    max_ipc_bytes: int = _DEFAULT_MAX_IPC_BYTES,
    max_batch: int = _DEFAULT_MAX_BATCH,
) -> int:
    """Return the number of time steps to batch per subprocess write.

    The batch is capped so that ``batch * saved_points * dtype_bytes``
    does not exceed *max_ipc_bytes*, reducing the batch to one step for very
    large grids (e.g. glb_01min).
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
    raise KeyError(
        f"Could not find variable for '{var_name}' (safe: '{safe}') in {ncfile.filepath()}"
    )


def _wsl_drop_cache(output_path) -> None:
    """WSL optimisation: advise the kernel to drop page-cache for *output_path*."""
    if _is_wsl() and hasattr(os, 'posix_fadvise'):
        try:
            with open(output_path, 'rb') as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _write_netcdf_process(request: NetCDFWriteRequest) -> tuple[str, int]:
    """Append one already-batched request in a single file transaction."""
    if request.data.ndim < 1 or request.data.shape[0] != len(request.times):
        raise ValueError(
            "NetCDF write batch data/time lengths must be identical"
        )
    if not request.times:
        raise ValueError("NetCDF write batch must contain at least one row")
    with nc.Dataset(request.output_path, "a") as ncfile:
        time_var = ncfile.variables["time"]
        target = _find_data_variable(ncfile, request.variable)
        variable = ncfile.variables[target]
        committed = ncfile.getncattr(COMMITTED_STEPS_ATTR)
        if isinstance(committed, (bool, np.bool_)) or not isinstance(
            committed, (int, np.integer),
        ):
            raise TypeError(
                f"{COMMITTED_STEPS_ATTR} must be an integer in "
                f"{request.output_path}"
            )
        current_len = int(committed)
        physical_len = len(time_var)
        if current_len < 0 or len(variable) != physical_len:
            raise ValueError(
                f"invalid NetCDF append lengths in {request.output_path}: "
                f"committed={current_len}, time={physical_len}, "
                f"data={len(variable)}"
            )
        if physical_len != current_len:
            raise RuntimeError(
                f"NetCDF output {request.output_path} contains an "
                f"uncommitted append tail: committed={current_len}, "
                f"physical={physical_len}"
            )
        units = time_var.getncattr("units")
        calendar = time_var.getncattr("calendar")
        for offset, (row, timestamp) in enumerate(
            zip(request.data, request.times, strict=True),
        ):
            index = current_len + offset
            variable[index, ...] = row
            time_var[index] = nc.date2num(
                timestamp, units=units, calendar=calendar,
            )
        ncfile.sync()
        committed_len = current_len + len(request.times)
        ncfile.setncattr(COMMITTED_STEPS_ATTR, committed_len)
        ncfile.sync()
    _wsl_drop_cache(request.output_path)
    return request.variable, committed_len - 1


def _create_netcdf_file_process(
    request: NetCDFCreateRequest,
) -> Union[Path, List[Path]]:
    """
    Process function for creating empty NetCDF files with proper structure.
    This function runs in a separate process.

    Args:
        args: Tuple containing (mean_var_name, metadata, coord_values,
              output_dir, rank, year, calendar, time_unit, num_trials)

    Returns:
        Path or List[Path] to the created NetCDF file(s)
    """
    mean_var_name = request.variable
    metadata = request.metadata
    coord_values = request.coordinate_values
    output_dir = request.output_dir
    rank = request.rank
    world_size = request.world_size
    year = request.year
    calendar = request.calendar
    time_unit = request.time_unit
    num_trials = request.num_trials
    static_vars = request.static_variables
    netcdf_options = request.netcdf_options

    safe_name = sanitize_symbol(mean_var_name)

    schema = NetCDFSchema.compile(metadata)
    actual_shape = schema.actual_shape
    tensor_shape = schema.tensor_shape
    # nc_coord_name is derived from dim_coords (e.g. "catchment_id").
    coord_name = schema.coordinate_name
    dtype = schema.dtype
    k_val = schema.order

    # Helper to create a single NetCDF file
    def create_single_file(file_safe_name: str, file_var_name: str, description_suffix: str = "") -> Path:
        output_path = OutputFilePlan(
            directory=output_dir,
            variable=file_safe_name,
            rank=rank,
            year=year,
        ).path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        reserved = {coord_name} if coord_name else set()
        reserved.update((static_vars or {}).keys())
        if file_safe_name in reserved:
            raise ValueError(
                f"Output variable '{file_safe_name}' conflicts with coordinate "
                "or static metadata of the same name."
            )

        with atomic_netcdf_dataset(output_path, format="NETCDF4") as ncfile:
            # Write global attributes
            ncfile.setncattr('title', f'Time series for rank {rank}: {file_var_name}')
            ncfile.setncattr('original_variable_name', file_var_name)
            ncfile.setncattr("hydroforge_output_format", OUTPUT_FORMAT)
            ncfile.setncattr("hydroforge_output_version", OUTPUT_VERSION)
            ncfile.setncattr("hydroforge_rank", rank)
            ncfile.setncattr("hydroforge_world_size", world_size)
            ncfile.setncattr(COMMITTED_STEPS_ATTR, 0)

            # Create time dimension (unlimited for streaming)
            ncfile.createDimension('time', None)

            # Create spatial/vertical dimensions based on actual shape
            dim_names = ['time']  # Always start with time

            if metadata.get('full_output'):
                actual_shape_tuple = tuple(int(v) for v in actual_shape)
                if num_trials > 1 and actual_shape_tuple and actual_shape_tuple[0] == num_trials:
                    dim_names.append('trial')
                    ncfile.createDimension('trial', num_trials)
                    data_shape = actual_shape_tuple[1:]
                else:
                    data_shape = actual_shape_tuple

                logical_dims = list(tensor_shape or ())
                if len(logical_dims) == len(actual_shape_tuple):
                    logical_dims = logical_dims[-len(data_shape):]
                elif len(logical_dims) != len(data_shape):
                    logical_dims = [f"dim_{i}" for i in range(len(data_shape))]

                # Full coordinate output uses the same axis as selected output.
                if coord_name and data_shape:
                    logical_dims[0] = 'saved_points'

                used_dims = set(dim_names)
                for i, (dim_name, dim_size) in enumerate(
                    zip(logical_dims, data_shape, strict=True)
                ):
                    nc_dim = sanitize_symbol(str(dim_name)) or f"dim_{i}"
                    if nc_dim in used_dims:
                        nc_dim = f"{nc_dim}_{i}"
                    used_dims.add(nc_dim)
                    dim_names.append(nc_dim)
                    ncfile.createDimension(nc_dim, int(dim_size))

            elif num_trials > 1:
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
                if 'saved_points' not in ncfile.dimensions:
                    raise ValueError(
                        f"Output coordinate '{coord_name}' requires saved_points."
                    )
                expected_coord_size = ncfile.dimensions['saved_points'].size
                if len(coord_values) != expected_coord_size:
                    raise ValueError(
                        f"Output coordinate '{coord_name}' length "
                        f"{len(coord_values)} != saved_points {expected_coord_size}."
                    )
                coord_var = ncfile.createVariable(
                    coord_name,
                    coord_values.dtype,
                    ('saved_points',),
                )
                coord_var[:] = coord_values

            # Write user-supplied static per-point variables. Coordinate scope
            # decides applicability; once applicable, dimension mismatch is a
            # schema error rather than a silent omission.
            for sv_name, sv_spec in (static_vars or {}).items():
                sv_dim = sv_spec.get("dim", "saved_points")
                output_coord = metadata.get("output_coord")
                if not _static_variable_applies(
                    sv_name, sv_spec,
                    output_coordinate=output_coord,
                    dimensions=ncfile.dimensions,
                ):
                    continue
                sv_values = np.asarray(sv_spec["values"])
                expected = ncfile.dimensions[sv_dim].size
                if sv_values.shape[0] != expected:
                    raise ValueError(
                        f"static_vars['{sv_name}'] length {sv_values.shape[0]} "
                        f"!= dim '{sv_dim}' size {expected} in {output_path.name}"
                    )
                sv_var = ncfile.createVariable(
                    sv_name, sv_spec.get("dtype", sv_values.dtype.str.lstrip("<>|")),
                    (sv_dim,),
                )
                sv_var[:] = sv_values
                for ak, av in sv_spec.get("attrs", {}).items():
                    sv_var.setncattr(ak, av)

            time_var = ncfile.createVariable('time', 'f8', ('time',))
            time_var.setncattr('units', time_unit)
            time_var.setncattr('calendar', calendar)

            # Create single data variable
            create_options = _validated_netcdf_options(
                file_var_name, netcdf_options, dim_names,
            )
            nc_var = ncfile.createVariable(
                file_safe_name, dtype, dim_names, **create_options,
            )
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



class NetCDFWriter:
    """Own streaming and in-memory finalization for one aggregator."""

    def __init__(self, owner) -> None:
        self.owner = owner

    def _create_netcdf_files(self, year: Optional[int] = None) -> None:
        """Create empty NetCDF files with proper structure for streaming."""
        if self.owner.in_memory_mode:
            # Skip file creation in in-memory mode
            return

        if not self.owner.output_split_by_year and self.owner._files_created:
            return

        self.owner.event_sink.emit(ModelEvent(
            "info", "output.create_start", "Creating NetCDF file structure",
            {"year": year},
        ))

        # Plan the complete transaction before starting workers. Each worker
        # creates its NetCDF file in write mode, matching the long-standing
        # model behavior of replacing output from an earlier run.
        requests: list[NetCDFCreateRequest] = []
        planned_paths: list[Path] = []
        for out_name, metadata in self.owner._metadata.items():
            coord_name = metadata.get('output_coord')
            coord_values = self.owner._coord_cache.get(coord_name, None)
            request = NetCDFCreateRequest(
                variable=out_name,
                metadata=metadata,
                coordinate_values=coord_values,
                output_dir=self.owner.output_dir,
                rank=self.owner.rank,
                world_size=self.owner.world_size,
                year=year,
                calendar=self.owner.calendar,
                time_unit=self.owner.time_unit,
                num_trials=self.owner.num_trials,
                static_variables=self.owner.static_vars,
                netcdf_options=self.owner.output_netcdf_options,
            )
            requests.append(request)
            safe_name = sanitize_symbol(out_name)
            order = NetCDFSchema.compile(metadata).order
            names = (
                tuple(f"{safe_name}_{index}" for index in range(order))
                if order > 1 else (safe_name,)
            )
            planned_paths.extend(
                OutputFilePlan(
                    directory=self.owner.output_dir,
                    variable=name,
                    rank=self.owner.rank,
                    year=year,
                ).path
                for name in names
            )
        if not requests:
            raise RuntimeError("cannot create statistics output without variables")
        duplicate_paths = {
            path for path in planned_paths if planned_paths.count(path) > 1
        }
        if duplicate_paths:
            raise ValueError(
                "statistics outputs resolve to duplicate file paths: "
                f"{sorted(map(str, duplicate_paths))}"
            )
        # Prepare file creation tasks.
        creation_futures = {}
        n_outputs = len(requests)
        actual_workers = max(1, min(self.owner.num_workers, n_outputs))
        staged: dict[str, Path | list[Path]] = {}
        try:
            with ProcessPoolExecutor(
                max_workers=actual_workers, mp_context=get_context("spawn"),
            ) as executor:
                for request in requests:
                    future = executor.submit(_create_netcdf_file_process, request)
                    creation_futures[future] = request.variable
                for future in as_completed(creation_futures):
                    out_name = creation_futures[future]
                    staged[out_name] = future.result()
        except BaseException:
            for path in planned_paths:
                path.unlink(missing_ok=True)
            raise

        if set(staged) != {request.variable for request in requests}:
            raise RuntimeError("NetCDF file creation returned an incomplete result set")
        self.owner._netcdf_files.update(staged)
        for result in staged.values():
            paths = result if isinstance(result, list) else [result]
            for path in paths:
                self.owner._all_created_files.add(path)
                self.owner.event_sink.emit(ModelEvent(
                    "info", "output.file_created", "Created NetCDF file",
                    {"path": str(path)},
                ))

        self.owner._files_created = True
        total_files = sum(len(v) if isinstance(v, list) else 1 for v in self.owner._netcdf_files.values())
        self.owner.event_sink.emit(ModelEvent(
            "info", "output.create_complete", "Created NetCDF files for streaming",
            {"files": total_files},
        ))

        # --- Initialise write-batch buffers ---
        # Each buffer key mirrors a submit key (out_name or out_name_kN).
        # Value: dict(data=[], dt=[], path=Path)
        self.owner._write_buffers.clear()
        # Compute the adaptive batch size from the first registered output
        first_meta = next(iter(self.owner._metadata.values()), {})
        actual_shape = first_meta.get('actual_shape', (1,))
        if first_meta.get('full_output'):
            sp = int(np.prod(actual_shape)) if actual_shape else 1
            size_label = "elements"
        else:
            # saved_points is the first spatial dim (or second if trials present)
            sp = actual_shape[1] if self.owner.num_trials > 1 and len(actual_shape) > 1 else actual_shape[0] if actual_shape else 1
            size_label = "saved_points"
        dtype_bytes = np.dtype(first_meta.get('dtype', 'f4')).itemsize
        self.owner._write_batch_size = compute_write_batch_size(sp, dtype_bytes)
        self.owner.event_sink.emit(ModelEvent(
            "info", "output.batch_configured", "Configured NetCDF write batch",
            {"steps": self.owner._write_batch_size, size_label: sp},
        ))


    def _flush_write_buffer(self, key: str) -> None:
        """Submit a buffered batch to its configured writer executor."""
        buf = self.owner._write_buffers.get(key)
        if not buf or len(buf['data']) == 0:
            return

        data_batch = np.stack(buf['data'], axis=0)
        dt_list = list(buf['dt'])
        output_path = buf['path']
        var_name = buf['var_name']

        request = NetCDFWriteRequest(
            variable=var_name,
            data=data_batch,
            output_path=output_path,
            times=tuple(dt_list),
        )

        if not self.owner._write_executors:
            raise RuntimeError(
                "NetCDF write buffer cannot flush without an active executor"
            )
        idx = abs(hash(key)) % len(self.owner._write_executors)
        future = self.owner._write_executors[idx].submit(
            _write_netcdf_process, request,
        )
        self.owner._pending_writes.append(PendingNetCDFWrite(
            key=key, step_count=len(dt_list), future=future,
        ))

        buf['data'].clear()
        buf['dt'].clear()


    def _flush_all_write_buffers(self) -> None:
        """Flush every pending write buffer (called on year transition / shutdown)."""
        failures: list[BaseException] = []
        for key in list(self.owner._write_buffers):
            try:
                self.owner._flush_write_buffer(key)
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError("NetCDF write buffers", failures)

    def _output_paths(self, out_name: str):
        try:
            return self.owner._netcdf_files[out_name]
        except KeyError as error:
            raise RuntimeError(
                f"statistics output {out_name!r} is dirty but has no "
                "created NetCDF file"
            ) from error


    def _buffer_and_maybe_flush(
        self,
        buf_key: str,
        var_name: str,
        data: np.ndarray,
        output_path,
        dt: Union[datetime, cftime.datetime],
        batch_size: int,
    ) -> None:
        """Append one time step to a write buffer; flush when full."""
        if buf_key not in self.owner._write_buffers:
            self.owner._write_buffers[buf_key] = dict(data=[], dt=[], path=output_path, var_name=var_name)

        buf = self.owner._write_buffers[buf_key]
        # Update path in case of year-split (new file for new year)
        buf['path'] = output_path
        buf['data'].append(data.copy())
        buf['dt'].append(dt)

        effective_batch_size = min(batch_size, self.owner.max_pending_steps)
        if len(buf['data']) >= effective_batch_size:
            self.owner._flush_write_buffer(buf_key)

    def _pending_step_counts(self) -> dict[str, int]:
        counts = {
            key: len(buffer["data"])
            for key, buffer in self.owner._write_buffers.items()
            if buffer["data"]
        }
        for pending in self.owner._pending_writes:
            counts[pending.key] = counts.get(pending.key, 0) + pending.step_count
        return counts

    def _wait_for(self, pending: PendingNetCDFWrite, *, dt) -> None:
        try:
            pending.future.result()
        except Exception as exc:
            self.owner.event_sink.emit(ModelEvent(
                "error", "output.write_failed", "Failed to write time step",
                {
                    "output": pending.key, "steps": pending.step_count,
                    "time": str(dt), "error": str(exc),
                },
            ))
            raise

    def check_completed_writes(self, *, dt) -> None:
        """Observe every completed background write without blocking."""

        remaining: list[PendingNetCDFWrite] = []
        failures: list[BaseException] = []
        for pending in self.owner._pending_writes:
            if not pending.future.done():
                remaining.append(pending)
                continue
            try:
                self._wait_for(pending, dt=dt)
            except BaseException as error:
                failures.append(error)
        self.owner._pending_writes = remaining
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError(
                "completed NetCDF background writes", failures,
            )

    def flush_and_wait(self, *, dt) -> None:
        """Make every buffered statistics row durable without closing workers."""

        failures: list[BaseException] = []
        try:
            self._flush_all_write_buffers()
        except BaseException as error:
            failures.append(error)
        pending, self.owner._pending_writes = self.owner._pending_writes, []
        for item in pending:
            try:
                self._wait_for(item, dt=dt)
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ResourceCleanupError(
                "NetCDF output durability boundary", failures,
            )

    def _limit_pending_steps(self, *, dt) -> None:
        """Bound each output stream by its exact unfinished timestep count."""

        self.check_completed_writes(dt=dt)

        if self.owner.max_pending_steps == 1:
            pending, self.owner._pending_writes = (
                self.owner._pending_writes, []
            )
            failures: list[BaseException] = []
            for item in pending:
                try:
                    self._wait_for(item, dt=dt)
                except BaseException as error:
                    failures.append(error)
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise ResourceCleanupError(
                    "single-step NetCDF pending writes", failures,
                )
            return

        counts = self._pending_step_counts()
        while counts and max(counts.values()) > self.owner.max_pending_steps:
            overfull = {
                key for key, count in counts.items()
                if count > self.owner.max_pending_steps
            }
            index = next((
                index
                for index, pending in enumerate(self.owner._pending_writes)
                if pending.key in overfull
            ), None)
            if index is None:
                raise RuntimeError(
                    "NetCDF pending-step accounting exceeded its limit "
                    "without a submitted write to drain"
                )
            pending = self.owner._pending_writes.pop(index)
            self._wait_for(pending, dt=dt)
            counts[pending.key] -= pending.step_count
            if counts[pending.key] == 0:
                counts.pop(pending.key)


    def _finalize_time_step_in_memory(self, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize time step in in-memory mode by copying storage to result tensors.

        Args:
            dt: Time step to finalize
        """
        # Increment macro step index for next iteration
        self.owner._macro_step_index += 1

        # Get dirty outputs to write
        keys_to_write = [k for k in self.owner._output_keys if k in self.owner._dirty_outputs]
        self.owner._dirty_outputs.clear()

        # Append storage tensors to result lists
        for out_name in keys_to_write:
            if out_name not in self.owner._result_tensors:
                continue

            storage_tensor = self.owner._storage[out_name]

            # Transfer an ownership-isolated snapshot in the exact declared
            # output precision. ``copy=True`` is required even when the output
            # device/dtype already match the accumulator.
            result_copy = storage_tensor.detach().to(
                device=self.owner.result_device,
                dtype=self.owner._result_dtype(out_name),
                copy=True,
            )
            self.owner._result_tensors[out_name].append(result_copy)

        # Advance time index
        self.owner._current_time_index += 1

        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True


    def finalize_time_step(self, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize the current time step by writing results to output.

        In streaming mode: writes to NetCDF files incrementally.
        In in-memory mode: copies current storage to result tensors.

        Args:
            dt: Time step to finalize (datetime or cftime.datetime)
        """
        # Error if compound ops are configured but outer flags have never been set
        if (self.owner._has_compound_ops
                and not self.owner._outer_flags_ever_seen):
            compound_ops = [
                f"{meta['original_variable']}.{meta['op']}"
                for name, meta in self.owner._metadata.items()
                if self.owner._output_is_outer.get(name, False)
            ]
            raise RuntimeError(
                f"Compound statistics {compound_ops} are configured but "
                "the managed step policy has not observed an outer statistics "
                "window. Configure statistics_interval/statistics_outer_interval "
                "on the model and advance through @managed_step."
            )

        # Record this time step for argmax/argmin index-to-time conversion
        # This is called at the end of each outer loop iteration
        self.owner._macro_step_times.append(dt)

        # Handle in-memory mode
        if self.owner.in_memory_mode:
            self.owner._finalize_time_step_in_memory(dt)
            return

        if self.owner.output_split_by_year:
            if self.owner._current_year is None:
                # First call - set up files
                self.owner._create_netcdf_files(year=dt.year)
                self.owner._current_year = dt.year
            elif self.owner._current_year != dt.year:
                # Year transition – flush remaining buffers for the old year first
                self.owner._flush_all_write_buffers()
                # Year transition - create new files for new year
                self.owner._create_netcdf_files(year=dt.year)
                self.owner._current_year = dt.year
                self.owner._macro_step_times = [dt]  # Reset time mapping for new year (keep current dt)
        else:
            # Create NetCDF files if not already created
            if not self.owner._files_created:
                self.owner._create_netcdf_files()

        # Increment macro step index for next iteration
        # (Note: index is reset to 0 in update_statistics when is_outer_first=True)
        self.owner._macro_step_index += 1

        # Write all outputs that are marked dirty
        # Use explicit list of output keys to maintain order/determinism
        keys_to_write = [k for k in self.owner._output_keys if k in self.owner._dirty_outputs]

        # Clear dirty set for next step
        self.owner._dirty_outputs.clear()

        batch_size = getattr(self.owner, '_write_batch_size', 1)

        for out_name in keys_to_write:
            tensor = self.owner._storage[out_name]

            output_paths = self._output_paths(out_name)

            # Check if this is a time index variable (argmax/argmin)
            metadata = self.owner._metadata.get(out_name, {})
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
                    self.owner._buffer_and_maybe_flush(buf_key, file_var_name, k_data, output_path, dt, batch_size)
            else:
                # Single file case (k=1 or legacy)
                output_path = output_paths if not isinstance(output_paths, list) else output_paths[0]
                self.owner._buffer_and_maybe_flush(out_name, out_name, time_step_data, output_path, dt, batch_size)

        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True

        self._limit_pending_steps(dt=dt)

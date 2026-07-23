"""NetCDF export workflows shared by forcing datasets."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import netCDF4 as nc
import numpy as np
import torch
from tqdm import tqdm

from hydroforge.serialization.netcdf import (
    DEFAULT_NETCDF_OPTIONS,
    atomic_netcdf_dataset,
    normalize_netcdf_variable_options,
)


logger = logging.getLogger(__name__)


class DatasetExporter:
    """Explicit NetCDF export service for one gridded dataset."""

    def __init__(self, owner) -> None:
        self.owner = owner

    def __len__(self) -> int:
        return len(self.owner)

    @property
    def _desired_catchment_ids(self):
        return self.owner._desired_catchment_ids

    @property
    def chunk_len(self):
        return self.owner.chunk_len

    @property
    def data_size(self):
        return self.owner.data_size

    @property
    def num_main_steps(self):
        return self.owner.num_main_steps

    @property
    def num_spin_up_steps(self):
        return self.owner.num_spin_up_steps

    def get_time_by_index(self, index):
        return self.owner.get_time_by_index(index)

    def is_valid_time_index(self, index):
        return self.owner.is_valid_time_index(index)

    def read_chunk(self, index: int):
        return self.owner.read_chunk(index)

    @property
    def supports_time_aggregation(self) -> bool:
        return bool(self.owner.supports_time_aggregation)

    def export_climatology(
        self,
        out_path: Union[str, Path],
        local_mapping: torch.Tensor,
        var_name: str,
        dtype: Literal["float32", "float64"] = "float32",
        netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
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
            netcdf_options: Validated NetCDF variable-creation options.
            device: Device for computation (``"cpu"`` or ``"cuda:X"``).
            units: Units attribute written to the output variable.
            description: Optional long description attribute.

        Returns:
            Path to the created NetCDF file.
        """
        if self._desired_catchment_ids is None:
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
            raise RuntimeError(
                "CUDA was requested for export_climatology but is unavailable"
            )

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
        logger.info("Climatology averaged over %d timesteps", total_steps)

        # ----- Write NetCDF -----
        dtype_nc = "f4" if dtype == "float32" else "f8"
        desc = description if description else f"Time-averaged {var_name} over {total_steps} steps"

        create_options = normalize_netcdf_variable_options(netcdf_options)
        with atomic_netcdf_dataset(out_path, format="NETCDF4") as ds:
            ds.setncattr("title", f"Climatology ({var_name})")
            ds.setncattr("total_timesteps", total_steps)

            ds.createDimension("saved_points", n_catch)

            cid_var = ds.createVariable("catchment_id", "i8", ("saved_points",))
            cid_var[:] = catchment_ids

            out_var = ds.createVariable(
                var_name, dtype_nc, ("saved_points",),
                **create_options,
            )
            out_var[:] = mean_data.astype(
                np.float32 if dtype == "float32" else np.float64
            )
            out_var.setncattr("description", desc)
            out_var.setncattr("units", units)

        logger.info("Saved climatology to %s", out_path)
        return out_path

    def export_catchment_data(
        self,
        out_dir: str | Path,
        local_mapping: torch.Tensor,
        var_name: str = "var",
        dtype: Literal["float32", "float64"] = "float32",
        netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
        normalized: bool = False,
        device: str | torch.device = "cpu",
        split_by_year: bool = False,
        units: Union[str, Dict[str, str]] = "m3/s",
        description: Optional[Union[str, Dict[str, str]]] = None,
        filename: Optional[Union[str, Dict[str, str]]] = None,
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
        - A requested CUDA device must be available; device selection is explicit.

        Args:
            out_dir: Output directory for NetCDF files
            local_mapping: Sparse tensor from build_local_mapping(), shape (n_grids, n_catchments)
            var_name: Variable name in output NetCDF
            dtype: Output data type
            netcdf_options: Validated NetCDF variable-creation options.
            normalized: If True, normalize mapping weights to sum to 1 per catchment
            device: Device for computation ("cpu" or "cuda:X")
            split_by_year: If True, create separate files per year
            units: Units string for the output variable
            description: Optional description for the output variable
            filename: Optional custom filename prefix (default: var_name)
        """
        if not isinstance(var_name, str):
            raise TypeError(
                "var_name must be a string; define time_aggregation when "
                "constructing the dataset."
            )
        create_options = normalize_netcdf_variable_options(netcdf_options)

        active_aggregation = getattr(self.owner, "time_aggregation", None)

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
        if self._desired_catchment_ids is None:
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
            raise RuntimeError(
                "CUDA was requested for export_catchment_data but is unavailable"
            )

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

        def _init_nc(stack, path, name, method):
            ds = stack.enter_context(
                atomic_netcdf_dataset(path, format="NETCDF4"),
            )
            ds.setncattr("title", f"Aggregated catchment data ({name})")
            if method is not None:
                ds.setncattr("time_aggregation", method)
            ds.createDimension("time", None)
            ds.createDimension("saved_points", n_catch)

            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.setncattr("units", "seconds since 1900-01-01 00:00:00")
            time_var.setncattr("calendar", "standard")

            output_coord = ds.createVariable("catchment_id", "i8", ("saved_points",))
            output_coord[:] = catchment_ids

            out_var = ds.createVariable(
                name,
                dtype_nc,
                ("time", "saved_points"),
                **create_options,
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
        writer_stack = None
        created_files = {name: [] for name in output_methods}
        current_year = None
        write_idx = 0
        total_steps = self.num_main_steps + self.num_spin_up_steps

        def _close_writers(error=None):
            nonlocal writers, writer_stack
            if writer_stack is not None:
                if error is None:
                    writer_stack.close()
                else:
                    writer_stack.__exit__(
                        type(error), error, error.__traceback__,
                    )
            writers = {}
            writer_stack = None

        def _open_writers(year=None):
            nonlocal write_idx, writer_stack
            _close_writers()
            writer_stack = ExitStack()
            try:
                for name, method in output_methods.items():
                    if year is None:
                        nc_path = out_dir / f"{_file_prefix(name)}_rank0.nc"
                    else:
                        nc_path = out_dir / f"{_file_prefix(name)}_rank0_{year}.nc"
                    writers[name] = _init_nc(
                        writer_stack, nc_path, name, method,
                    )
                    created_files[name].append(nc_path)
            except BaseException as error:
                _close_writers(error)
                raise
            write_idx = 0

        pbar = None
        failure = None
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
        except BaseException as error:
            failure = error
            raise
        finally:
            if pbar is not None:
                pbar.close()
            _close_writers(failure)

        if returns_mapping:
            if split_by_year:
                return created_files
            return {name: paths[0] for name, paths in created_files.items()}

        only_name = next(iter(output_methods))
        return created_files[only_name] if split_by_year else created_files[only_name][0]

"""NetCDF export workflows shared by forcing datasets."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Union
from uuid import uuid4

import netCDF4 as nc
import numpy as np
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator
from tqdm import tqdm

from hydroforge.serialization.netcdf import (
    COMMITTED_STEPS_ATTR,
    DEFAULT_NETCDF_OPTIONS,
    OUTPUT_FORMAT,
    OUTPUT_VERSION,
    RUN_ID_ATTR,
    _atomic_netcdf_dataset_trusted,
    _create_netcdf_variable_trusted,
    _prepare_netcdf_variable_options_trusted,
    normalize_netcdf_variable_options,
)
from hydroforge.contracts.naming import sanitize_symbol
from hydroforge.contracts.validation import HydroForgeModel, _immutable_dict
from hydroforge.data.numeric import canonical_floating_array


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hydroforge.data.datasets.gridded import GriddedDataset


def _output_name(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty exact string")
    if (
        Path(value).name != value
        or value in {".", ".."}
        or sanitize_symbol(value) != value
    ):
        raise ValueError(f"{label} must be one safe NetCDF/file component")
    return value


def _output_path(value: Any, *, label: str) -> Path:
    del label
    return Path(value)


def _output_device(value: Any) -> torch.device:
    return torch.device(value)


def _metadata_values(
    value: str | Mapping[str, str] | None,
    *,
    label: str,
    names: tuple[str, ...],
    default: Callable[[str], str],
) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({name: default(name) for name in names})
    if type(value) is str:
        return MappingProxyType(dict.fromkeys(names, value))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a string, mapping, or None")
    if set(value) != set(names):
        raise ValueError(f"{label} mapping keys must be exactly {list(names)}")
    if any(type(item) is not str for item in value.values()):
        raise ValueError(f"{label} mapping values must be exact strings")
    return MappingProxyType({name: value[name] for name in names})


class _ClimatologyExportRequest(HydroForgeModel):
    owner: Any = Field(exclude=True)
    local_mapping: torch.Tensor = Field(exclude=True, repr=False)
    out_path: Path
    var_name: str
    dtype: Literal["float32", "float64"] = "float32"
    netcdf_options: Mapping[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_NETCDF_OPTIONS),
    )
    device: torch.device = torch.device("cpu")
    units: str = "m3/s"
    description: str | None = None

    _create_options: Mapping[str, Any] = PrivateAttr()

    @field_validator("out_path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> Path:
        return _output_path(value, label="out_path")

    @field_validator("var_name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _output_name(value, label="var_name")

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, value: Any) -> torch.device:
        return _output_device(value)

    @field_validator("netcdf_options", mode="before")
    @classmethod
    def _validate_options(cls, value: Any) -> Mapping[str, Any]:
        return normalize_netcdf_variable_options(value)

    @model_validator(mode="after")
    def _compile(self):
        dtype_nc = "f4" if self.dtype == "float32" else "f8"
        options = _prepare_netcdf_variable_options_trusted(
            self.netcdf_options,
            dtype=dtype_nc,
            dimensions=("saved_points",),
            name=self.var_name,
        )
        object.__setattr__(
            self, "netcdf_options", _immutable_dict(self.netcdf_options),
        )
        self._create_options = _immutable_dict(options)
        return self

    @property
    def create_options(self) -> Mapping[str, Any]:
        return self._create_options


class _CatchmentExportRequest(HydroForgeModel):
    owner: Any = Field(exclude=True)
    local_mapping: torch.Tensor = Field(exclude=True, repr=False)
    out_dir: Path
    var_name: str = "var"
    filename: str | Mapping[str, str] | None = None
    dtype: Literal["float32", "float64"] = "float32"
    netcdf_options: Mapping[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_NETCDF_OPTIONS),
    )
    normalized: bool = Field(default=False, strict=True)
    device: torch.device = torch.device("cpu")
    split_by_year: bool = Field(default=False, strict=True)
    units: str | Mapping[str, str] = "m3/s"
    description: str | Mapping[str, str] | None = None

    _output_methods: Mapping[str, str | None] = PrivateAttr()
    _returns_mapping: bool = PrivateAttr()
    _filenames: Mapping[str, str] = PrivateAttr()
    _units: Mapping[str, str] = PrivateAttr()
    _descriptions: Mapping[str, str] = PrivateAttr()
    _create_options: Mapping[str, Mapping[str, Any]] = PrivateAttr()

    @field_validator("out_dir", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> Path:
        return _output_path(value, label="out_dir")

    @field_validator("var_name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _output_name(value, label="var_name")

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, value: Any) -> torch.device:
        return _output_device(value)

    @field_validator("netcdf_options", mode="before")
    @classmethod
    def _validate_options(cls, value: Any) -> Mapping[str, Any]:
        return normalize_netcdf_variable_options(value)

    @model_validator(mode="after")
    def _compile(self):
        active = self.owner.time_aggregation
        if isinstance(active, Mapping):
            output_methods = dict(active)
            returns_mapping = True
        else:
            output_methods = {self.var_name: active}
            returns_mapping = False
        output_names = tuple(output_methods)
        for name in output_names:
            _output_name(name, label="time_aggregation output name")

        filenames = _metadata_values(
            self.filename,
            label="filename",
            names=output_names,
            default=lambda name: name,
        )
        filenames = MappingProxyType({
            name: _output_name(value, label=f"filename[{name!r}]")
            for name, value in filenames.items()
        })
        descriptions = _metadata_values(
            self.description,
            label="description",
            names=output_names,
            default=lambda name: (
                f"Catchment-aggregated {name} ({output_methods[name]})"
                if output_methods[name] is not None
                else f"Catchment-aggregated {name} (area-weighted mean)"
            ),
        )
        units = _metadata_values(
            self.units,
            label="units",
            names=output_names,
            default=lambda _name: "",
        )
        dtype_nc = "f4" if self.dtype == "float32" else "f8"
        create_options = {
            name: _immutable_dict(_prepare_netcdf_variable_options_trusted(
                self.netcdf_options,
                dtype=dtype_nc,
                dimensions=("time", "saved_points"),
                name=name,
            ))
            for name in output_names
        }

        object.__setattr__(
            self, "netcdf_options", _immutable_dict(self.netcdf_options),
        )
        self._output_methods = MappingProxyType(output_methods)
        self._returns_mapping = returns_mapping
        self._filenames = filenames
        self._descriptions = descriptions
        self._units = units
        self._create_options = MappingProxyType(create_options)
        return self

    @property
    def output_methods(self) -> Mapping[str, str | None]:
        return self._output_methods

    @property
    def returns_mapping(self) -> bool:
        return self._returns_mapping

    @property
    def filenames(self) -> Mapping[str, str]:
        return self._filenames

    @property
    def units_by_name(self) -> Mapping[str, str]:
        return self._units

    @property
    def descriptions(self) -> Mapping[str, str]:
        return self._descriptions

    @property
    def create_options(self) -> Mapping[str, Mapping[str, Any]]:
        return self._create_options


class DatasetExporter:
    """Explicit NetCDF export service for one gridded dataset."""

    def __init__(self, owner: GriddedDataset) -> None:
        self.owner = owner

    @staticmethod
    def _output_array(
        value: Any, *, dtype: str, label: str,
    ) -> np.ndarray:
        return canonical_floating_array(value, dtype=dtype, label=label)

    @staticmethod
    def _prepare_mapping(
        local_mapping: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Move the caller-owned sparse mapping for this export call."""

        return local_mapping.to(device=device, dtype=dtype).coalesce()

    def export_climatology(
        self,
        out_path: str | Path,
        local_mapping: torch.Tensor,
        var_name: str,
        dtype: Literal["float32", "float64"] = "float32",
        netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
        device: str | torch.device = "cpu",
        units: str = "m3/s",
        description: str | None = None,
    ) -> Path:
        """Validate one export request before reading or creating files."""

        request = _ClimatologyExportRequest.model_validate({
            "owner": self.owner,
            "out_path": out_path,
            "local_mapping": local_mapping,
            "var_name": var_name,
            "dtype": dtype,
            "netcdf_options": netcdf_options,
            "device": device,
            "units": units,
            "description": description,
        })
        return self._export_climatology_trusted(request)

    def _export_climatology_trusted(
        self, request: _ClimatologyExportRequest,
    ) -> Path:
        """
        Compute the temporal-mean (climatological average) and export to NetCDF.

        This mirrors the logic of Fortran-based routing models: iterate over
        every timestep in the dataset, accumulate the sum, and divide by the number
        of steps to obtain the daily-mean climatology mapped to catchments.

        The mapped Dataset's active grid data is aggregated via sparse matmul and
        time-averaged.  Output NetCDF has dimension ``(saved_points,)`` with a
        ``catchment_id`` coordinate variable.

        Args:
            out_path: Full path (including filename) for the output NetCDF file.
            local_mapping: Sparse tensor returned by ``build_local_mapping()``.
            var_name: Variable name written into the NetCDF file.
            dtype: Output data type (``"float32"`` or ``"float64"``).
            netcdf_options: Validated NetCDF variable-creation options.
            device: Device for computation (``"cpu"`` or ``"cuda:X"``).
            units: Units attribute written to the output variable.
            description: Optional long description attribute.

        Returns:
            Path to the created NetCDF file.
        """
        out_path = request.out_path
        var_name = request.var_name
        dtype = request.dtype
        dev = request.device
        units = request.units
        description = request.description
        catchment_ids = self.owner.desired_catchment_ids
        n_catch = len(catchment_ids)

        # Prepare transposed mapping matrix: (n_catch, n_grids)
        t_mapping = self._prepare_mapping(
            request.local_mapping,
            device=dev,
            dtype=torch.float64,
        )
        t_mapping_T = t_mapping.t().coalesce()

        # ----- Accumulate mean over all chunks -----
        first_chunk = self.owner._num_spin_up_chunks
        n_chunks = len(self.owner)
        total_steps = 0
        accumulator = torch.zeros(n_catch, dtype=torch.float64, device=dev)

        pbar = tqdm(
            range(first_chunk, n_chunks),
            desc="Computing climatology", unit="chunk",
        )
        for ci in pbar:
            chunk = self.owner.chunk_plan._at_trusted(ci)
            block = self.owner._read_chunk_trusted(chunk)  # (T, n_grids)
            valid_T = chunk.length
            block = np.ascontiguousarray(
                block, dtype=np.float64,
            )
            block_t = torch.as_tensor(block, dtype=torch.float64, device=dev)
            # (n_catch, n_grids) @ (n_grids, T) -> (n_catch, T)
            agg = torch.sparse.mm(t_mapping_T, block_t.T)
            accumulator += agg.sum(dim=1).to(torch.float64)

            total_steps += valid_T

        pbar.close()

        if total_steps == 0:
            raise RuntimeError("No valid timesteps found — cannot compute climatology.")

        mean_data = self._output_array(
            (accumulator / total_steps).cpu().numpy(),
            dtype=dtype,
            label="climatology result",
        )
        logger.info("Climatology averaged over %d timesteps", total_steps)

        # ----- Write NetCDF -----
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dtype_nc = "f4" if dtype == "float32" else "f8"
        create_options = request.create_options
        desc = (
            f"Time-averaged {var_name} over {total_steps} steps"
            if description is None else description
        )

        with _atomic_netcdf_dataset_trusted(
            out_path, format="NETCDF4",
        ) as ds:
            ds.setncattr("title", f"Climatology ({var_name})")
            ds.setncattr("total_timesteps", total_steps)

            ds.createDimension("saved_points", n_catch)

            cid_var = ds.createVariable("catchment_id", "i8", ("saved_points",))
            cid_var[:] = catchment_ids

            out_var = _create_netcdf_variable_trusted(
                ds,
                var_name,
                dtype_nc,
                ("saved_points",),
                options=create_options,
            )
            out_var[:] = mean_data
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
        units: str | Mapping[str, str] = "m3/s",
        description: str | Mapping[str, str] | None = None,
        filename: str | Mapping[str, str] | None = None,
    ) -> Union[Path, List[Path], Dict[str, Path], Dict[str, List[Path]]]:
        """Validate one export request before reading or creating files."""

        request = _CatchmentExportRequest.model_validate({
            "owner": self.owner,
            "out_dir": out_dir,
            "local_mapping": local_mapping,
            "var_name": var_name,
            "dtype": dtype,
            "netcdf_options": netcdf_options,
            "normalized": normalized,
            "device": device,
            "split_by_year": split_by_year,
            "units": units,
            "description": description,
            "filename": filename,
        })
        return self._export_catchment_data_trusted(request)

    def _export_catchment_data_trusted(
        self, request: _CatchmentExportRequest,
    ) -> Union[Path, List[Path], Dict[str, Path], Dict[str, List[Path]]]:
        """
        Export catchment-aggregated data to a NetCDF file readable by MultiRankStatsReader.

        Requires ``build_local_mapping()`` to have been called on the dataset.

        - Output filename: {var_name}_rank0.nc
          (or {var_name}_rank0_{year}.nc if split_by_year)
        - Dimensions: time (unlimited), saved_points
        - Variables:
            * time: numeric with units and calendar
            * catchment_id: (saved_points,) catchment IDs
            * {var_name}: (time, saved_points) aggregated data

        GPU acceleration:
        - Set `device="cuda:0"` (or any CUDA device) to enable GPU-accelerated sparse matmul.
        - A requested CUDA device must be available; device selection is explicit.

        Args:
            out_dir: Output directory for NetCDF files
            local_mapping: Sparse tensor returned by ``build_local_mapping()``.
            var_name: Variable name in output NetCDF
            dtype: Output data type
            netcdf_options: Validated NetCDF variable-creation options.
            normalized: If True, normalize mapping weights to sum to 1 per catchment
            device: Device for computation ("cpu" or "cuda:X")
            split_by_year: If True, create separate files per year
            units: Units string for the output variable
            description: Optional description for the output variable
        """
        out_dir = request.out_dir
        dtype = request.dtype
        normalized = request.normalized
        dev = request.device
        split_by_year = request.split_by_year
        output_methods = request.output_methods
        returns_mapping = request.returns_mapping
        filename_values = request.filenames
        description_values = request.descriptions
        units_values = request.units_by_name
        catchment_ids = self.owner.desired_catchment_ids

        n_catch = len(catchment_ids)

        # Use the provided local mapping matrix
        # Shape: (n_cols, n_catch) - maps compressed source grids to catchments
        t_mapping = self._prepare_mapping(
            request.local_mapping,
            device=dev,
            dtype=torch.float64,
        )

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
                indices, new_values, t_mapping.size(),
                dtype=torch.float64, device=dev,
            ).coalesce()

        # Pre-compute transposed mapping matrix for efficient batch multiplication
        # t_mapping shape: (n_cols, n_catch)
        # t_mapping_T shape: (n_catch, n_cols) for sparse.mm(sparse, dense)
        t_mapping_T = t_mapping.t().coalesce()

        out_dir.mkdir(parents=True, exist_ok=True)

        dtype_nc = "f4" if dtype == "float32" else "f8"
        run_id = str(uuid4())

        def _init_nc(stack, path, name, method):
            create_options = request.create_options[name]
            ds = stack.enter_context(
                _atomic_netcdf_dataset_trusted(path, format="NETCDF4"),
            )
            ds.setncattr("title", f"Aggregated catchment data ({name})")
            ds.setncattr("hydroforge_output_format", OUTPUT_FORMAT)
            ds.setncattr("hydroforge_output_version", OUTPUT_VERSION)
            ds.setncattr("hydroforge_rank", 0)
            ds.setncattr("hydroforge_world_size", 1)
            ds.setncattr(RUN_ID_ATTR, run_id)
            ds.setncattr(COMMITTED_STEPS_ATTR, 0)
            if method is not None:
                ds.setncattr("time_aggregation", method)
            ds.createDimension("time", None)
            ds.createDimension("saved_points", n_catch)

            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.setncattr("units", "seconds since 1900-01-01 00:00:00")
            time_var.setncattr(
                "calendar", getattr(self.owner, "calendar", "standard"),
            )

            output_coord = ds.createVariable("catchment_id", "i8", ("saved_points",))
            output_coord[:] = catchment_ids

            out_var = _create_netcdf_variable_trusted(
                ds,
                name,
                dtype_nc,
                ("time", "saved_points"),
                options=create_options,
            )
            out_var.setncattr("description", description_values[name])
            out_var.setncattr("units", units_values[name])
            return ds, time_var, out_var

        writers = {}
        writer_stack = None
        created_files = {name: [] for name in output_methods}
        current_year = None
        write_idx = 0
        total_steps = self.owner.num_main_source_steps

        def _close_writers(error=None):
            nonlocal writers, writer_stack
            if writer_stack is not None:
                if error is None:
                    try:
                        for dataset, time_variable, _variable in writers.values():
                            dataset.setncattr(
                                COMMITTED_STEPS_ATTR, len(time_variable),
                            )
                            dataset.sync()
                    except BaseException as commit_error:
                        writer_stack.__exit__(
                            type(commit_error), commit_error,
                            commit_error.__traceback__,
                        )
                        writers = {}
                        writer_stack = None
                        raise
                    else:
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
                    filename = filename_values[name]
                    if year is None:
                        nc_path = out_dir / f"{filename}_rank0.nc"
                    else:
                        nc_path = (
                            out_dir
                            / f"{filename}_rank0_{year}.nc"
                        )
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

            first_chunk = self.owner._num_spin_up_chunks
            n_chunks = len(self.owner)
            pbar = tqdm(total=total_steps, desc="Exporting", unit="step")
            for ci in range(first_chunk, n_chunks):
                chunk = self.owner.chunk_plan._at_trusted(ci)
                read_data = self.owner._read_chunk_trusted(chunk)
                if isinstance(read_data, dict):
                    blocks = read_data
                else:
                    name = next(iter(output_methods))
                    blocks = {name: read_data}

                T = chunk.length
                mapped_blocks = {}
                for name, block in blocks.items():
                    # t_mapping_T @ block.T = (n_catch, n_cols) @ (n_cols, T)
                    block = np.ascontiguousarray(
                        block, dtype=np.float64,
                    )
                    block_tensor = torch.as_tensor(
                        block, dtype=torch.float64, device=dev,
                    )
                    agg_block = torch.sparse.mm(t_mapping_T, block_tensor.T)
                    mapped_blocks[name] = self._output_array(
                        agg_block.T.contiguous().to("cpu").numpy(),
                        dtype=dtype,
                        label=(
                            f"aggregated variable {name!r} at chunk {ci}"
                        ),
                    )

                # Write maximal same-file runs as blocks.  Chunk data is
                # already resident, so row-at-a-time writes only add HDF5
                # extension, chunk lookup and compression overhead.
                chunk_times = chunk._source_times()
                run_start = 0
                while run_start < T:
                    if split_by_year:
                        year = chunk_times[run_start].year
                        if year != current_year:
                            current_year = year
                            _open_writers(current_year)
                        run_end = run_start + 1
                        while (
                            run_end < T
                            and chunk_times[run_end].year == current_year
                        ):
                            run_end += 1
                    else:
                        run_end = T

                    _ds, first_time_var, _out_var = next(
                        iter(writers.values())
                    )
                    time_values = nc.date2num(
                        chunk_times[run_start:run_end],
                        units=first_time_var.getncattr("units"),
                        calendar=first_time_var.getncattr("calendar"),
                    )
                    write_end = write_idx + run_end - run_start
                    for name in output_methods:
                        _ds, time_var, out_var = writers[name]
                        out_var[write_idx:write_end, :] = mapped_blocks[name][
                            run_start:run_end, :
                        ]
                        time_var[write_idx:write_end] = time_values
                    pbar.update(run_end - run_start)
                    write_idx = write_end
                    run_start = run_end
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

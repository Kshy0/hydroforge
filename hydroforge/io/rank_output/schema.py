"""The rank-output file schema: names, header, time axis, point coordinate.

Every rank file of one output run is ``{variable}_rank{rank}[_{year}].nc``
with the format header :class:`RankFileHeader`, an unlimited ``time`` axis in
:data:`TIME_UNITS`, and optionally a ``saved_points`` integer coordinate named
by the global attribute :data:`COORDINATE_ATTR` (empty when absent).
Per-variable NetCDF layouts are compiled up front by :class:`NetCDFSchema`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import numpy as np
import torch

from hydroforge.core.arrays import torch_to_numpy_dtype
from hydroforge.core.naming import sanitize_symbol
from hydroforge.io.netcdf.encoding import (
    COMPLETE_DATA_ATTR,
    netcdf_dtype_encoding,
    read_netcdf_values,
    saved_dtype,
)
from hydroforge.io.netcdf.options import (
    MIN_BLOSC_CHUNK_BYTES,
    plan_streaming_netcdf_chunks,
    prepare_netcdf_variable_options,
)
from hydroforge.io.netcdf.read import decoded_dtype

OUTPUT_FORMAT = "hydroforge.statistics"
OUTPUT_VERSION = 3
FORMAT_ATTR = "hydroforge_output_format"
VERSION_ATTR = "hydroforge_output_version"
RANK_ATTR = "hydroforge_rank"
WORLD_SIZE_ATTR = "hydroforge_world_size"
RUN_ID_ATTR = "hydroforge_run_id"
COMMITTED_STEPS_ATTR = "hydroforge_committed_steps"
# Name of the ('saved_points',) coordinate variable; empty when none is written.
COORDINATE_ATTR = "hydroforge_coordinate"
TIME_DIM = "time"
POINT_DIM = "saved_points"
TIME_UNITS = "days since 1900-01-01 00:00:00"

_NATIVE_INTEGERS = (int, np.int32, np.int64)


def rank_file_name(variable: str, rank: int, year: int | None = None) -> str:
    """Return the file name of one rank (and year) of an output variable."""

    return f"{variable}_rank{rank}" + ("" if year is None else f"_{year}") + ".nc"


def parse_rank_file_name(
    name: str,
    variable: str,
    *,
    split_by_year: bool,
) -> tuple[int, int | None] | None:
    """Return ``(rank, year)`` of a :func:`rank_file_name`, else ``None``."""

    pattern = rf"{re.escape(variable)}_rank(\d+)" + (
        r"_(-?\d+)\.nc" if split_by_year else r"\.nc"
    )
    match = re.fullmatch(pattern, name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)) if split_by_year else None


@dataclass(frozen=True, slots=True)
class RankFileHeader:
    """Format identity and commit state of one rank file."""

    rank: int
    world_size: int
    run_id: str
    committed_steps: int = 0

    def write(self, dataset: Any) -> None:
        dataset.setncattr(FORMAT_ATTR, OUTPUT_FORMAT)
        dataset.setncattr(VERSION_ATTR, OUTPUT_VERSION)
        dataset.setncattr(RANK_ATTR, self.rank)
        dataset.setncattr(WORLD_SIZE_ATTR, self.world_size)
        dataset.setncattr(RUN_ID_ATTR, self.run_id)
        dataset.setncattr(COMMITTED_STEPS_ATTR, self.committed_steps)

    @classmethod
    def read(cls, dataset: Any, *, path: Path) -> Self:
        """Read and validate the header written by :meth:`write`."""

        present = set(dataset.ncattrs())
        missing = [
            name
            for name in (
                FORMAT_ATTR,
                VERSION_ATTR,
                RANK_ATTR,
                WORLD_SIZE_ATTR,
                RUN_ID_ATTR,
                COMMITTED_STEPS_ATTR,
            )
            if name not in present
        ]
        if missing:
            raise ValueError(
                f"{path.name} is not a version {OUTPUT_VERSION} rank output: "
                f"missing global attributes {missing}"
            )
        output_format = dataset.getncattr(FORMAT_ATTR)
        if output_format != OUTPUT_FORMAT:
            raise ValueError(f"unsupported output format {output_format!r}")
        version, rank, world_size = (
            dataset.getncattr(name)
            for name in (VERSION_ATTR, RANK_ATTR, WORLD_SIZE_ATTR)
        )
        for name, value in (
            (VERSION_ATTR, version),
            (RANK_ATTR, rank),
            (WORLD_SIZE_ATTR, world_size),
        ):
            if type(value) not in _NATIVE_INTEGERS:
                raise ValueError(f"{name} must be a Python, int32 or int64 integer")
        if version != OUTPUT_VERSION:
            raise ValueError(
                f"unsupported output version {version}; version "
                f"{OUTPUT_VERSION} with a shared run identity is required"
            )
        if not 0 <= rank < world_size:
            raise ValueError("invalid hydroforge_world_size/rank contract")
        run_id = dataset.getncattr(RUN_ID_ATTR)
        if type(run_id) is not str or not run_id or run_id != run_id.strip():
            raise ValueError(
                f"{RUN_ID_ATTR} must be a non-empty string without surrounding "
                "whitespace"
            )
        committed = dataset.getncattr(COMMITTED_STEPS_ATTR)
        if (
            isinstance(committed, (bool, np.bool_))
            or not isinstance(committed, (int, np.integer))
            or committed < 0
        ):
            raise ValueError(f"{COMMITTED_STEPS_ATTR} must be a nonnegative integer")
        return cls(
            rank=int(rank),
            world_size=int(world_size),
            run_id=run_id,
            committed_steps=int(committed),
        )


def create_time_axis(dataset: Any, *, calendar: str) -> Any:
    """Create the f8 ``time`` coordinate in :data:`TIME_UNITS`.

    The unlimited ``time`` dimension is created unless the caller declared it
    first to fix the dimension order.
    """

    if TIME_DIM not in dataset.dimensions:
        dataset.createDimension(TIME_DIM, None)
    variable = dataset.createVariable(TIME_DIM, "f8", (TIME_DIM,))
    variable.setncattr("units", TIME_UNITS)
    variable.setncattr("calendar", calendar)
    return variable


def write_point_coordinate(
    dataset: Any,
    name: str | None,
    values: np.ndarray | None,
) -> None:
    """Declare the saved-point coordinate and write it, or declare none.

    The ``saved_points`` dimension is created unless the caller declared it
    first to fix the dimension order.
    """

    declared = bool(name) and values is not None
    dataset.setncattr(COORDINATE_ATTR, name if declared else "")
    if not declared:
        return
    if POINT_DIM not in dataset.dimensions:
        dataset.createDimension(POINT_DIM, values.size)
    variable = dataset.createVariable(name, values.dtype, (POINT_DIM,))
    variable.setncattr(COMPLETE_DATA_ATTR, "true")
    variable[:] = values


def read_point_coordinate(
    dataset: Any,
    *,
    name: str | None,
    path: Path,
) -> tuple[str | None, np.ndarray | None]:
    """Return the saved-point coordinate as unique int64 IDs.

    ``name`` overrides the file's :data:`COORDINATE_ATTR`; a file without that
    attribute requires it.
    """

    if name is None:
        if COORDINATE_ATTR not in dataset.ncattrs():
            raise ValueError(
                f"{path.name} does not declare {COORDINATE_ATTR}; pass the "
                "saved_points coordinate name explicitly (coord_name=...)"
            )
        name = dataset.getncattr(COORDINATE_ATTR)
        if type(name) is not str:
            raise TypeError(f"{COORDINATE_ATTR} must be a string")
        if not name:
            return None, None
    variable = dataset.variables[name]
    if variable.dimensions != (POINT_DIM,):
        raise ValueError(f"coordinate {name!r} must have dimensions ('{POINT_DIM}',)")
    if decoded_dtype(variable).kind not in "iu":
        raise TypeError(f"coordinate {name!r} must contain integers")
    raw = read_netcdf_values(variable)
    if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
        raise ValueError(f"coordinate {name!r} contains missing values")
    values = np.asarray(raw)
    if values.dtype.kind not in "iu":
        raise TypeError(f"coordinate {name!r} must contain integers")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise OverflowError(f"coordinate {name!r} contains values outside int64 range")
    coordinate = np.array(values, dtype=np.int64, order="C", copy=True)
    if np.unique(coordinate).size != coordinate.size:
        raise ValueError(f"coordinate {name!r} contains duplicate IDs")
    return name, coordinate


@dataclass(frozen=True, slots=True)
class NetCDFSchema:
    actual_shape: tuple[int, ...]
    tensor_shape: tuple[int | str, ...]
    coordinate_name: str | None
    dtype: str
    logical_dtype: str | None
    order: int
    write_batch_size: int
    full_output: bool
    batched: bool
    description: str
    output_coordinate: str | None
    file_actual_shape: tuple[int, ...]
    logical_actual_shape: tuple[int, ...]
    dimensions: tuple[tuple[str, int], ...]
    data_dimensions: tuple[str, ...]
    create_options: Mapping[str, Any]

    @classmethod
    def compile(
        cls,
        metadata: Mapping[str, Any],
        *,
        variable: str,
        ensemble_size: int,
        netcdf_options: Mapping[str, Any],
        write_batch_size: int = 1,
        save_precision: torch.dtype | None = None,
    ) -> Self:
        storage_dtype, logical_dtype = netcdf_dtype_encoding(
            torch_to_numpy_dtype(saved_dtype(metadata["dtype"], save_precision))
        )
        actual_shape = metadata["actual_shape"]
        tensor_shape = metadata["tensor_shape"]
        dim_coords = metadata.get("dim_coords")
        coordinate_name = dim_coords.rsplit(".", 1)[-1] if dim_coords else None
        order = metadata["k"]
        batched = metadata["batched"]
        full_output = metadata["full_output"]
        file_actual_shape = actual_shape[:-1] if order > 1 else actual_shape
        logical_actual_shape = file_actual_shape[1:] if batched else file_actual_shape
        data_dimensions = [TIME_DIM]
        used_dimensions = {TIME_DIM}
        dimensions: list[tuple[str, int]] = []

        def add_dimension(name: str, extent: int, *, axis: int) -> None:
            """Append one deterministic NetCDF dimension with a unique name."""

            base = sanitize_symbol(name) or f"dim_{axis}"
            candidate = base
            if candidate in used_dimensions:
                candidate = f"{base}_{axis}"
            suffix = 1
            while candidate in used_dimensions:
                candidate = f"{base}_{axis}_{suffix}"
                suffix += 1
            used_dimensions.add(candidate)
            data_dimensions.append(candidate)
            dimensions.append((candidate, extent))

        if batched:
            data_dimensions.append("ensemble")
            used_dimensions.add("ensemble")
            dimensions.append(("ensemble", ensemble_size))
        if full_output:
            logical_dimensions = list(tensor_shape)
            if coordinate_name and logical_actual_shape:
                logical_dimensions[0] = POINT_DIM
            for axis, (name, extent) in enumerate(
                zip(
                    logical_dimensions,
                    logical_actual_shape,
                    strict=True,
                )
            ):
                logical_name = f"dim_{axis}" if type(name) is int else name
                add_dimension(logical_name, extent, axis=axis)
        else:
            data_dimensions.append(POINT_DIM)
            used_dimensions.add(POINT_DIM)
            dimensions.append((POINT_DIM, logical_actual_shape[0]))
            if len(logical_actual_shape) == 2:
                data_dimensions.append("levels")
                used_dimensions.add("levels")
                dimensions.append(("levels", logical_actual_shape[1]))
        if netcdf_options.get("contiguous") is True:
            raise ValueError(
                f"streaming NetCDF output {variable!r} has an unlimited time "
                "dimension and cannot use contiguous=True"
            )
        row_shape = tuple(extent for _name, extent in dimensions)
        create_options = plan_streaming_netcdf_chunks(
            netcdf_options,
            dtype=storage_dtype,
            row_shape=row_shape,
            write_batch_size=write_batch_size,
        )
        create_options = prepare_netcdf_variable_options(
            create_options,
            dtype=storage_dtype,
            dimensions=data_dimensions,
            name=variable,
            logical_dtype=logical_dtype,
        )
        chunks = create_options.get("chunksizes")
        if chunks is not None:
            for axis, (chunk, (_name, extent)) in enumerate(
                zip(chunks[1:], dimensions, strict=True),
                start=1,
            ):
                if extent > 0 and chunk > extent:
                    raise ValueError(
                        f"NetCDF chunksizes for {variable!r} axis {axis} "
                        f"exceed dimension extent {extent}: {chunk}"
                    )
            compression = create_options.get("compression")
            if (
                type(compression) is str
                and compression.startswith("blosc_")
                and math.prod(chunks) * storage_dtype.itemsize < MIN_BLOSC_CHUNK_BYTES
            ):
                raise ValueError(
                    f"Blosc chunksizes for {variable!r} must encode at least "
                    f"{MIN_BLOSC_CHUNK_BYTES} bytes per chunk"
                )
        return cls(
            actual_shape=actual_shape,
            tensor_shape=tensor_shape,
            coordinate_name=coordinate_name,
            dtype=storage_dtype.str.lstrip("<>|"),
            logical_dtype=logical_dtype,
            order=order,
            write_batch_size=write_batch_size,
            full_output=full_output,
            batched=batched,
            description=metadata.get("description", ""),
            output_coordinate=metadata.get("output_coord"),
            file_actual_shape=file_actual_shape,
            logical_actual_shape=logical_actual_shape,
            dimensions=tuple(dimensions),
            data_dimensions=tuple(data_dimensions),
            create_options=MappingProxyType(create_options),
        )

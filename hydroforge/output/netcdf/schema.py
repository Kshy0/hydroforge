"""Immutable NetCDF variable schema compiled before file creation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Self

from hydroforge.serialization.netcdf import (
    _prepare_netcdf_variable_options_trusted,
    MIN_BLOSC_CHUNK_BYTES,
    netcdf_dtype_encoding,
    plan_streaming_netcdf_chunks,
)
from hydroforge.contracts.naming import sanitize_symbol


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
    metadata: Mapping[str, Any]

    @classmethod
    def compile(
        cls,
        metadata: Mapping[str, Any],
        *,
        variable: str,
        num_trials: int,
        netcdf_options: Mapping[str, Any],
        write_batch_size: int = 1,
    ) -> Self:
        storage_dtype, logical_dtype = netcdf_dtype_encoding(metadata["dtype"])
        actual_shape = metadata["actual_shape"]
        tensor_shape = metadata["tensor_shape"]
        coordinate_name = metadata.get("nc_coord_name")
        order = metadata["k"]
        batched = metadata["batched"]
        full_output = metadata["full_output"]
        file_actual_shape = (
            actual_shape[:-1] if order > 1 else actual_shape
        )
        logical_actual_shape = (
            file_actual_shape[1:] if batched else file_actual_shape
        )
        data_dimensions = ["time"]
        used_dimensions = {"time"}
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

        if full_output:
            if batched:
                data_dimensions.append("trial")
                used_dimensions.add("trial")
                dimensions.append(("trial", num_trials))
            logical_dimensions = list(tensor_shape)
            if coordinate_name and logical_actual_shape:
                logical_dimensions[0] = "saved_points"
            for axis, (name, extent) in enumerate(zip(
                logical_dimensions, logical_actual_shape, strict=True,
            )):
                logical_name = f"dim_{axis}" if type(name) is int else name
                add_dimension(logical_name, extent, axis=axis)
        else:
            if batched:
                data_dimensions.append("trial")
                used_dimensions.add("trial")
                dimensions.append(("trial", num_trials))
            data_dimensions.append("saved_points")
            used_dimensions.add("saved_points")
            dimensions.append(("saved_points", logical_actual_shape[0]))
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
        create_options = _prepare_netcdf_variable_options_trusted(
            create_options,
            dtype=storage_dtype,
            dimensions=data_dimensions,
            name=variable,
            logical_dtype=logical_dtype,
        )
        chunks = create_options.get("chunksizes")
        if chunks is not None:
            for axis, (chunk, (_name, extent)) in enumerate(
                zip(chunks[1:], dimensions, strict=True), start=1,
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
                and math.prod(chunks) * storage_dtype.itemsize
                < MIN_BLOSC_CHUNK_BYTES
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
            create_options=MappingProxyType(dict(create_options)),
            metadata=MappingProxyType(dict(metadata)),
        )

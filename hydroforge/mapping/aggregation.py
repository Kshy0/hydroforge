# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
"""Offline spatial aggregation: build mapping tables and aggregate fields.

These functions own the *generation* and *offline export* responsibilities that
used to be fused onto the dataset classes.  They operate on plain source
coordinates plus a target spec (a CaMa map directory or a regular point
``parameters.nc``), so they never need a dataset instance.  Each validates its
declaration once at the call boundary.

Public functions
----------------
- :func:`build_cama_mapping` — source grid -> CaMa catchments via MERIT hires pixels.
- :func:`build_point_mapping` — source grid -> a regular 1D cell list (e.g. VIC).
- :func:`aggregate_field_to_nc` — apply a saved mapping to a static/climatology field.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import netCDF4 as nc
import numpy as np
from pydantic import Field, validate_call

from hydroforge.core.arrays import (
    canonical_float64,
    canonical_floating_array,
    canonical_ids,
    find_indices_in,
    positive_finite_float64,
)
from hydroforge.core.naming import validate_netcdf_name, validate_safe_path_component
from hydroforge.core.time import canonical_calendar
from hydroforge.core.validation import HydroForgeModel
from hydroforge.io.netcdf.options import (
    NetCDFOptions,
    create_netcdf_variable,
    default_netcdf_options,
    ensure_hdf5_plugins,
    prepare_netcdf_variable_options,
)
from hydroforge.io.netcdf.write import atomic_netcdf_dataset
from hydroforge.io.rank_output.schema import (
    POINT_DIM,
    TIME_DIM,
    RankFileHeader,
    rank_file_name,
    write_point_coordinate,
)
from hydroforge.mapping.build import (
    _cama_hires_mapping,
    build_regular_grid_mapping,
)
from hydroforge.mapping.cama import (
    FloatPrecision,
    IndexPrecision,
    _read_cama_catchments,
)
from hydroforge.mapping.grid import RegularGrid
from hydroforge.mapping.table import MappingTable
from hydroforge.mapping.target import TargetSupport

logger = logging.getLogger(__name__)

_FilePath = Annotated[Path, Field(strict=False)]
_Name = Annotated[str, Field(min_length=1)]

# The output time axis stores decoded values, so only these semantic
# attributes carry over; packing, masking and bounds belong to the source.
_TIME_SEMANTIC_ATTRIBUTES = ("units", "calendar", "long_name", "standard_name", "axis")


@dataclass(frozen=True, slots=True)
class _AggregateFieldPlan:
    mapping: MappingTable
    field: np.ndarray
    has_time: bool
    time_values: np.ndarray | None
    time_attributes: Mapping[str, Any]
    output_options: Mapping[str, Any]


def _compile_aggregate_field_plan(
    field_nc: Path,
    var_name: str,
    mapping_npz: Path,
    *,
    normalized: bool,
    out_name: str,
    dtype: str,
    netcdf_options: Mapping[str, Any],
) -> _AggregateFieldPlan:
    """Validate and own all external arrays before aggregation begins."""

    mapping = MappingTable.load(mapping_npz)
    if normalized:
        mapping = mapping.row_normalized()

    time_values = None
    time_attributes: dict[str, Any] = {}
    ensure_hdf5_plugins()
    with nc.Dataset(field_nc, "r") as dataset:
        variable = dataset.variables[var_name]
        if variable.ndim not in {2, 3}:
            raise ValueError(
                f"field variable {var_name!r} must be 2-D or 3-D; "
                f"got shape {variable.shape}"
            )
        if tuple(variable.shape[-2:]) != mapping._source_shape:
            raise ValueError(
                f"field variable {var_name!r} has spatial shape "
                f"{tuple(variable.shape[-2:])}, expected "
                f"{mapping._source_shape}"
            )
        spatial_dimensions = variable.dimensions[-2:]
        for dimension, expected, label in (
            (spatial_dimensions[0], mapping.source_y, "y"),
            (spatial_dimensions[1], mapping.source_x, "x"),
        ):
            coordinate = dataset.variables[dimension]
            if coordinate.dimensions != (dimension,):
                raise ValueError(
                    f"spatial coordinate {dimension!r} must be one-dimensional"
                )
            raw_coordinate = coordinate[:]
            if np.ma.isMaskedArray(raw_coordinate) and np.any(
                np.ma.getmaskarray(raw_coordinate)
            ):
                raise ValueError(
                    f"spatial coordinate {dimension!r} contains missing values"
                )
            observed = canonical_float64(
                raw_coordinate,
                label=f"spatial coordinate {dimension!r}",
            )
            if observed.shape != expected.shape or not np.array_equal(
                observed, expected
            ):
                raise ValueError(
                    f"spatial coordinate {dimension!r} does not match the "
                    f"mapping source {label}-axis"
                )
        if np.dtype(variable.dtype).kind not in {"f", "i", "u"}:
            raise ValueError(
                f"field variable {var_name!r} must contain real numeric values"
            )
        has_time = variable.ndim == 3
        ntime = int(variable.shape[0]) if has_time else 1
        if out_name == "catchment_id" or (has_time and out_name == TIME_DIM):
            raise ValueError(
                f"aggregate output {out_name!r} conflicts with a coordinate variable"
            )
        output_options = prepare_netcdf_variable_options(
            netcdf_options,
            dtype=dtype,
            dimensions=(TIME_DIM, POINT_DIM) if has_time else (POINT_DIM,),
            name=out_name,
        )
        if has_time:
            time_dimension = variable.dimensions[0]
            time_variable = dataset.variables[time_dimension]
            if time_variable.dimensions != (time_dimension,):
                raise ValueError(
                    f"time coordinate {time_dimension!r} must be one-dimensional"
                )
            units = getattr(time_variable, "units", None)
            if not isinstance(units, str) or not units.strip():
                raise ValueError("time coordinate must declare nonempty CF units")
            calendar = canonical_calendar(
                getattr(time_variable, "calendar", "standard")
            )
            raw_time = time_variable[:]
            if np.ma.isMaskedArray(raw_time) and np.any(np.ma.getmaskarray(raw_time)):
                raise ValueError("time coordinate contains missing values")
            time_values = np.array(raw_time, order="C", copy=True)
            if (
                time_values.shape != (ntime,)
                or time_values.dtype.kind not in {"f", "i", "u"}
                or not np.isfinite(time_values).all()
            ):
                raise ValueError(
                    "time coordinate must contain one finite numeric value "
                    "per field row"
                )
            decoded = list(nc.num2date(time_values, units=units, calendar=calendar))
            if any(right <= left for left, right in zip(decoded, decoded[1:])):
                raise ValueError("time coordinate must be strictly increasing")
            time_attributes = {
                name: time_variable.getncattr(name)
                for name in _TIME_SEMANTIC_ATTRIBUTES
                if name in time_variable.ncattrs()
            }

        field = variable[:]

    if np.ma.isMaskedArray(field):
        mask = np.ma.getmaskarray(field)
        raw_field = np.asarray(field.data)
        valid_values = canonical_floating_array(
            raw_field[~mask],
            dtype="float64",
            label=f"field variable {var_name!r}",
        )
        canonical_field = np.empty(raw_field.shape, dtype=np.float64)
        canonical_field[~mask] = valid_values
        canonical_field[mask] = np.nan
    else:
        canonical_field = canonical_floating_array(
            field,
            dtype="float64",
            label=f"field variable {var_name!r}",
        )
    if not has_time:
        canonical_field = canonical_field[None, ...]
    canonical_field.setflags(write=False)
    if time_values is not None:
        time_values.setflags(write=False)
    return _AggregateFieldPlan(
        mapping=mapping,
        field=canonical_field,
        has_time=has_time,
        time_values=time_values,
        time_attributes=time_attributes,
        output_options=output_options,
    )


@validate_call(config=HydroForgeModel.model_config)
def build_cama_mapping(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    map_dir: _FilePath,
    *,
    source_lon_bounds: np.ndarray | None = None,
    source_lat_bounds: np.ndarray | None = None,
    hires_tag: str | None = "1min",
    mapinfo_txt: str = "location.txt",
    lowres_idx_precision: IndexPrecision = "<i4",
    hires_idx_precision: IndexPrecision = "<i2",
    map_precision: FloatPrecision = "<f4",
    parameter_nc: _FilePath | None = None,
    allow_oob_zero: bool = False,
    producer: _Name = "build_cama_mapping",
) -> MappingTable:
    """Build an area-weighted ``catchment x source`` mapping from MERIT hires pixels.

    Rows follow the ``parameter_nc`` catchment order when given, otherwise the
    CaMa map order; weights are raw hires pixel areas (no per-row
    normalization).
    """

    source = RegularGrid.from_coordinates(
        source_lon,
        source_lat,
        x_bounds=source_lon_bounds,
        y_bounds=source_lat_bounds,
    )
    catchment_id, nx, ny, nextxy_data = _read_cama_catchments(
        map_dir, lowres_idx_precision=lowres_idx_precision
    )
    if parameter_nc is None:
        target_ids = canonical_ids(catchment_id, label="CaMa catchment IDs")
    else:
        ensure_hdf5_plugins()
        with nc.Dataset(parameter_nc, "r") as dataset:
            target_ids = canonical_ids(
                dataset.variables["catchment_id"][...],
                label="parameter catchment_id",
            )
        if np.unique(target_ids).size != target_ids.size:
            raise ValueError("parameter catchment_id must be unique")
        present = find_indices_in(target_ids, catchment_id) >= 0
        if not np.all(present):
            missing = target_ids[~present]
            raise ValueError(
                f"{missing.size} parameter catchment id(s) are absent "
                f"from the map; examples={missing[:5].tolist()}"
            )
    mapping = _cama_hires_mapping(
        source,
        target_ids,
        map_dir,
        nx,
        ny,
        nextxy_data,
        hires_tag=hires_tag,
        mapinfo_txt=mapinfo_txt,
        hires_idx_precision=hires_idx_precision,
        map_precision=map_precision,
        allow_oob_zero=allow_oob_zero,
        metadata={"producer": producer},
    )
    empty_rows = int(np.sum(np.diff(mapping.matrix.indptr) == 0))
    if empty_rows > 0:
        logger.warning(
            "%d catchments were not mapped to source grids; their grid input "
            "will always be zero.",
            empty_rows,
        )
    return mapping


@validate_call(config=HydroForgeModel.model_config)
def build_point_mapping(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    parameter_nc: _FilePath,
    *,
    method: Literal["nearest", "overlap"] = "overlap",
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    id_name: str = "catchment_id",
    gsize: Any = None,
    producer: _Name = "build_point_mapping",
) -> MappingTable:
    """Build a mapping from a source grid onto a regular 1D point-cell list.

    Targets a model stored as a sparse 1D list of regular grid cells with
    per-cell ``longitude`` / ``latitude`` (e.g. VIC) — no MERIT basemap.  The
    mapping ``method`` (``"overlap"`` or ``"nearest"``) is chosen by the caller;
    ``overlap`` needs a cell size from the ``gsize`` argument or the file's
    ``gsize`` attribute.
    """

    if gsize is not None:
        gsize = positive_finite_float64(gsize, label="gsize")
    source = RegularGrid.from_coordinates(source_lon, source_lat)
    ensure_hdf5_plugins()
    with nc.Dataset(parameter_nc, "r") as dataset:
        lon = dataset.variables[lon_name][:]
        lat = dataset.variables[lat_name][:]
        raw_ids = dataset.variables[id_name][:]
        for label, value in ((lon_name, lon), (lat_name, lat)):
            if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
                raise ValueError(
                    f"parameter variable {label!r} contains missing values"
                )
        target_ids = canonical_ids(raw_ids, label=id_name)
        if gsize is None and "gsize" in dataset.ncattrs():
            gsize = positive_finite_float64(dataset.getncattr("gsize"), label="gsize")
    if method == "overlap" and gsize is None:
        raise ValueError(
            f"{parameter_nc.name} requests overlap mapping "
            "but has no 'gsize' attribute (and none was passed) to "
            "build cell bounds; add 'gsize' or use method='nearest'."
        )
    target = TargetSupport.from_points(
        np.asarray(lon),
        np.asarray(lat),
        target_ids=target_ids,
        cell_size=gsize if method == "overlap" else None,
    )
    return build_regular_grid_mapping(
        source,
        target,
        method=method,
        normalization="mean",
        metadata={"producer": producer},
    )


def _aggregate_masked_field(
    mapping: MappingTable,
    field: np.ndarray,
    *,
    normalized: bool,
) -> np.ndarray:
    """Aggregate a field whose masked source cells are NaN.

    Masked cells never poison a target that has valid sources.  Normalized
    output is the weighted mean over valid sources, ``M @ x / M @ valid``;
    unnormalized sums treat masked cells as zero.  A target is NaN only when
    it has weights but none of its sources is valid; empty rows stay zero.
    """

    missing = np.isnan(field)
    if not missing.any():
        return mapping.apply(field, layout="grid")
    totals = mapping.apply(np.where(missing, 0.0, field), layout="grid")
    valid_weight = mapping.apply((~missing).astype(np.float64), layout="grid")
    row_weight = np.asarray(mapping.matrix.sum(axis=1), dtype=np.float64).ravel()
    if normalized:
        with np.errstate(divide="ignore", invalid="ignore"):
            totals = totals / valid_weight
        totals[..., row_weight == 0] = 0.0
    else:
        totals[(valid_weight == 0) & (row_weight != 0)] = np.nan
    return totals


@validate_call(config=HydroForgeModel.model_config)
def aggregate_field_to_nc(
    field_nc: _FilePath,
    var_name: str,
    mapping_npz: _FilePath,
    out_dir: _FilePath,
    *,
    out_name: str | None = None,
    dtype: Literal["float32", "float64"] = "float32",
    netcdf_options: NetCDFOptions = Field(default_factory=default_netcdf_options),
    units: _Name = "mm",
    description: str | None = None,
    normalized: bool = False,
) -> Path:
    """Apply a saved mapping to a static or climatology field, writing a NetCDF.

    ``field_nc`` must hold ``var_name`` as ``(lat, lon)`` or ``(time, lat, lon)``
    on the same source grid the mapping was built from.  The output variable is
    named ``out_name`` (default ``var_name``) in ``{out_name}_rank0.nc``.
    Output dims are ``(saved_points,)`` or ``(time, saved_points)`` with a
    ``catchment_id`` coordinate.  A time series carries the rank-output header
    and the source time axis unchanged, so ``MultiRankStatsReader`` reads it.

    Masked source cells are excluded: with ``normalized=True`` each target is
    the weighted mean of its valid sources; unnormalized sums treat masked
    cells as zero.  A target is NaN only when all of its sources are masked.
    """

    out_name = validate_safe_path_component(
        var_name if out_name is None else out_name, label="out_name"
    )
    validate_netcdf_name(out_name)
    plan = _compile_aggregate_field_plan(
        field_nc,
        var_name,
        mapping_npz,
        normalized=normalized,
        out_name=out_name,
        dtype=dtype,
        netcdf_options=netcdf_options,
    )
    mapping = plan.mapping
    field = plan.field
    has_time = plan.has_time
    time_values = plan.time_values
    time_attributes = plan.time_attributes

    aggregated = canonical_floating_array(
        _aggregate_masked_field(mapping, field, normalized=normalized),
        dtype=dtype,
        label="aggregated values",
        allow_nan=True,
    )
    n_catch = mapping.matrix.shape[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / rank_file_name(out_name, 0)
    dtype_nc = "f4" if dtype == "float32" else "f8"

    with atomic_netcdf_dataset(nc_path, format="NETCDF4") as ds:
        ds.setncattr("title", f"Aggregated catchment parameter ({out_name})")
        if has_time:
            RankFileHeader(
                rank=0,
                world_size=1,
                run_id=str(uuid4()),
                committed_steps=len(time_values),
            ).write(ds)
            ds.createDimension(TIME_DIM, None)
        ds.createDimension(POINT_DIM, n_catch)

        if has_time:
            time_var = ds.createVariable(TIME_DIM, time_values.dtype, (TIME_DIM,))
            time_var.setncatts(time_attributes)
            time_var[:] = time_values

        write_point_coordinate(ds, "catchment_id", mapping.target_ids)

        dims = (TIME_DIM, POINT_DIM) if has_time else (POINT_DIM,)
        out_var = create_netcdf_variable(
            ds,
            out_name,
            dtype_nc,
            dims,
            options=plan.output_options,
        )
        resolved_description = (
            f"Catchment-aggregated {out_name}" if description is None else description
        )
        out_var.setncattr("description", resolved_description)
        out_var.setncattr("units", units)

        if has_time:
            out_var[:, :] = aggregated
        else:
            out_var[:] = aggregated[0]

    return nc_path

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
"""Offline spatial aggregation: build mapping tables and aggregate fields.

These functions own the *generation* and *offline export* responsibilities that
used to be fused onto the dataset classes.  They operate on plain source
coordinates plus a target spec (a CaMa map directory or a regular point
``parameters.nc``), so they never need a :class:`AbstractDataset` instance.

Public functions
----------------
- :func:`build_cama_mapping` — source grid -> CaMa catchments via MERIT hires pixels.
- :func:`build_point_mapping` — source grid -> a regular 1D cell list (e.g. VIC).
- :func:`aggregate_field_to_nc` — apply a saved mapping to a static/climatology field.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any, Optional, Union

import netCDF4 as nc
import numpy as np

from hydroforge.data.mapping import (MappingTable, RegularGrid,
                                           TargetSupport,
                                           build_hires_aggregate_mapping,
                                           build_regular_grid_mapping,
                                           read_cama_catchments,
                                           read_cama_hires_pixels)
from hydroforge.data.distributed import find_indices_in
from hydroforge.serialization.netcdf import (
    DEFAULT_NETCDF_OPTIONS,
    atomic_netcdf_dataset,
    normalize_netcdf_variable_options,
)


def build_cama_mapping(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    map_dir: Union[str, Path],
    *,
    hires_tag: Optional[str] = "1min",
    mapinfo_txt: str = "location.txt",
    lowres_idx_precision: str = "<i4",
    hires_idx_precision: str = "<i2",
    map_precision: str = "<f4",
    parameter_nc: Union[str, Path, None] = None,
    allow_oob_zero: bool = False,
    producer: str = "build_cama_mapping",
) -> MappingTable:
    """Build an area-weighted ``catchment x source`` mapping from MERIT hires pixels.

    Rows follow the ``parameter_nc`` catchment order when given, otherwise the
    CaMa map order; weights are raw hires pixel areas (no per-row
    normalization).
    """
    map_dir = Path(map_dir)
    catchment_id, nx, ny, nextxy_data = read_cama_catchments(
        map_dir, lowres_idx_precision=lowres_idx_precision
    )

    desired_ids = None
    if parameter_nc is not None:
        with nc.Dataset(Path(parameter_nc), "r") as ds:
            if "catchment_id" not in ds.variables:
                raise KeyError("'catchment_id' not found in parameter_nc")
            desired_ids = np.asarray(
                ds.variables["catchment_id"][...], dtype=np.int64,
            )

    catchment_id_hires, valid_areas, valid_lon, valid_lat = read_cama_hires_pixels(
        map_dir,
        nx,
        ny,
        nextxy_data,
        hires_tag=hires_tag,
        mapinfo_txt=mapinfo_txt,
        hires_idx_precision=hires_idx_precision,
        map_precision=map_precision,
    )

    source = RegularGrid.from_coordinates(np.asarray(source_lon), np.asarray(source_lat))
    if desired_ids is not None:
        present = find_indices_in(desired_ids, catchment_id) >= 0
        if not np.all(present):
            missing = desired_ids[~present]
            raise ValueError(
                f"{missing.size} parameter catchment id(s) are absent from the "
                f"map; examples={missing[:5].tolist()}"
            )
        target_ids = desired_ids
    else:
        target_ids = catchment_id.astype(np.int64)

    mapping = build_hires_aggregate_mapping(
        source,
        target_ids,
        catchment_id_hires,
        valid_areas,
        valid_lon,
        valid_lat,
        allow_oob_zero=allow_oob_zero,
        metadata={"producer": producer},
    )

    empty_rows = int(np.sum(np.diff(mapping.matrix.indptr) == 0))
    if empty_rows > 0:
        print(
            f"Warning: {empty_rows} catchments were not mapped to source grids. "
            "Their grid input will always be zero."
        )
    return mapping


def build_point_mapping(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    parameter_nc: Union[str, Path],
    *,
    method: str = "overlap",
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    id_name: str = "catchment_id",
    gsize: Optional[float] = None,
    producer: str = "build_point_mapping",
) -> MappingTable:
    """Build a mapping from a source grid onto a regular 1D point-cell list.

    Targets a model stored as a sparse 1D list of regular grid cells with
    per-cell ``longitude`` / ``latitude`` (e.g. VIC) — no MERIT basemap.  The
    mapping ``method`` (``"overlap"`` or ``"nearest"``) is chosen by the caller;
    ``overlap`` needs a cell size from the ``gsize`` argument or the file's
    ``gsize`` attribute.
    """
    with nc.Dataset(Path(parameter_nc), "r") as pnc:
        lon = np.asarray(pnc.variables[lon_name][:], dtype=np.float64)
        lat = np.asarray(pnc.variables[lat_name][:], dtype=np.float64)
        target_ids = np.asarray(pnc.variables[id_name][:], dtype=np.int64)
        if gsize is None and "gsize" in pnc.ncattrs():
            gsize = float(pnc.getncattr("gsize"))

    if method == "overlap" and gsize is None:
        raise ValueError(
            f"{Path(parameter_nc).name} requests overlap mapping but has no "
            "'gsize' attribute (and none was passed) to build cell bounds; add "
            "'gsize' or use method='nearest'."
        )

    source = RegularGrid.from_coordinates(np.asarray(source_lon), np.asarray(source_lat))
    target = TargetSupport.from_points(
        lon,
        lat,
        target_ids=target_ids,
        cell_size=gsize if method == "overlap" else None,
    )
    return build_regular_grid_mapping(
        source,
        target,
        method=method,
        metadata={"producer": producer},
    )


def aggregate_field_to_nc(
    field_nc: Union[str, Path],
    var_name: str,
    mapping_npz: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    out_name: Optional[str] = None,
    dtype: str = "float32",
    netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
    units: str = "mm",
    description: Optional[str] = None,
    normalized: bool = False,
) -> Path:
    """Apply a saved mapping to a static or climatology field, writing a NetCDF.

    ``field_nc`` must hold ``var_name`` as ``(lat, lon)`` or ``(time, lat, lon)``
    on the same source grid the mapping was built from.  The output variable is
    named ``out_name`` (default ``var_name``).  Output dims are
    ``(saved_points,)`` or ``(time, saved_points)`` with a ``catchment_id``
    coordinate, readable by ``MultiRankStatsReader``.
    """
    out_name = out_name or var_name
    mapping = MappingTable.load(mapping_npz)
    if normalized:
        mapping = mapping.row_normalized()

    np_dtype = np.float32 if dtype == "float32" else np.float64
    field_nc = Path(field_nc)
    with nc.Dataset(field_nc, "r") as ds:
        var = ds.variables[var_name]
        has_time = var.ndim == 3
        ntime = int(var.shape[0]) if has_time else 1
        field = np.asarray(var[:])
    if isinstance(field, np.ma.MaskedArray):
        field = field.filled(np.nan)
    field = field.astype(np_dtype, copy=False)
    if not has_time:
        field = field[None, ...]

    aggregated = mapping.apply(field).astype(np_dtype)  # (ntime, num_targets)
    n_catch = mapping.matrix.shape[0]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / f"{out_name}_rank0.nc"
    dtype_nc = "f4" if dtype == "float32" else "f8"

    create_options = normalize_netcdf_variable_options(netcdf_options)
    with atomic_netcdf_dataset(nc_path, format="NETCDF4") as ds:
        ds.setncattr("title", f"Aggregated catchment parameter ({out_name})")
        if has_time:
            ds.createDimension("time", None)
        ds.createDimension("saved_points", n_catch)

        if has_time:
            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.setncattr("units", "months" if ntime == 12 else "unknown")

        output_coord = ds.createVariable("catchment_id", "i8", ("saved_points",))
        output_coord[:] = mapping.target_ids

        dims = ("time", "saved_points") if has_time else ("saved_points",)
        out_var = ds.createVariable(
            out_name, dtype_nc, dims, **create_options,
        )
        out_var.setncattr("description", description or f"Catchment-aggregated {out_name}")
        out_var.setncattr("units", units)

        if has_time:
            out_var[:, :] = aggregated
        else:
            out_var[:] = aggregated[0]

    return nc_path

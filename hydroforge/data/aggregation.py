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

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

import netCDF4 as nc
import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.contracts.naming import validate_safe_path_component
from hydroforge.data.mapping.build import (
    _build_regular_grid_mapping_trusted,
    build_hires_aggregate_mapping,
)
from hydroforge.data.mapping.cama import (
    read_cama_catchments, read_cama_hires_pixels,
)
from hydroforge.data.mapping.grid import RegularGrid
from hydroforge.data.mapping.table import MappingTable
from hydroforge.data.mapping.target import TargetSupport
from hydroforge.data.numeric import (
    canonical_ids,
    canonical_float64,
    canonical_floating_array,
    positive_finite_float64,
)
from hydroforge.data.distributed import _find_indices_in_trusted
from hydroforge.serialization.netcdf import (
    DEFAULT_NETCDF_OPTIONS,
    _atomic_netcdf_dataset_trusted,
    _create_netcdf_variable_trusted,
    prepare_netcdf_variable_options,
)


class _BuildCamaMappingRequest(HydroForgeModel):
    source_lon: np.ndarray
    source_lat: np.ndarray
    map_dir: str | Path
    hires_tag: str | None = "1min"
    mapinfo_txt: str = "location.txt"
    lowres_idx_precision: str = "<i4"
    hires_idx_precision: str = "<i2"
    map_precision: str = "<f4"
    parameter_nc: str | Path | None = None
    allow_oob_zero: bool = Field(default=False, strict=True)
    producer: str = "build_cama_mapping"

    _source: RegularGrid = PrivateAttr()
    _target_ids: np.ndarray = PrivateAttr()
    _pixel_catchment_id: np.ndarray = PrivateAttr()
    _pixel_area: np.ndarray = PrivateAttr()
    _pixel_lon: np.ndarray = PrivateAttr()
    _pixel_lat: np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def _validate_declaration(self):
        if not self.producer:
            raise ValueError("producer must be non-empty")
        self._source = RegularGrid.from_coordinates(
            self.source_lon, self.source_lat,
        )
        catchment_id, nx, ny, nextxy_data = read_cama_catchments(
            self.map_dir,
            lowres_idx_precision=self.lowres_idx_precision,
        )
        desired_ids = None
        if self.parameter_nc is not None:
            with nc.Dataset(Path(self.parameter_nc), "r") as dataset:
                raw_ids = dataset.variables["catchment_id"][...]
                if np.ma.isMaskedArray(raw_ids) and np.any(
                    np.ma.getmaskarray(raw_ids)
                ):
                    raise ValueError(
                        "parameter catchment_id contains missing IDs"
                    )
                if np.asarray(raw_ids).ndim != 1:
                    raise ValueError(
                        "parameter catchment_id must be one-dimensional"
                    )
                desired_ids = canonical_ids(
                    raw_ids, label="parameter catchment_id",
                )
        (
            pixel_catchment_id,
            pixel_area,
            pixel_lon,
            pixel_lat,
        ) = read_cama_hires_pixels(
            self.map_dir,
            nx,
            ny,
            nextxy_data,
            hires_tag=self.hires_tag,
            mapinfo_txt=self.mapinfo_txt,
            hires_idx_precision=self.hires_idx_precision,
            map_precision=self.map_precision,
        )
        if desired_ids is None:
            target_ids = canonical_ids(
                catchment_id, label="CaMa catchment IDs",
            )
        else:
            present = (
                _find_indices_in_trusted(desired_ids, catchment_id) >= 0
            )
            if not np.all(present):
                missing = desired_ids[~present]
                raise ValueError(
                    f"{missing.size} parameter catchment id(s) are absent "
                    f"from the map; examples={missing[:5].tolist()}"
                )
            target_ids = desired_ids
        self._target_ids = target_ids
        self._pixel_catchment_id = pixel_catchment_id
        self._pixel_area = pixel_area
        self._pixel_lon = pixel_lon
        self._pixel_lat = pixel_lat
        return self

    @property
    def source(self) -> RegularGrid:
        return self._source

    @property
    def mapping_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._target_ids,
            self._pixel_catchment_id,
            self._pixel_area,
            self._pixel_lon,
            self._pixel_lat,
        )


class _BuildPointMappingRequest(HydroForgeModel):
    source_lon: np.ndarray
    source_lat: np.ndarray
    parameter_nc: str | Path
    method: Literal["nearest", "overlap"] = "overlap"
    lon_name: str = "longitude"
    lat_name: str = "latitude"
    id_name: str = "catchment_id"
    gsize: Any = None
    producer: str = "build_point_mapping"

    _source: RegularGrid = PrivateAttr()
    _gsize: float | None = PrivateAttr(default=None)
    _target: TargetSupport = PrivateAttr()

    @model_validator(mode="after")
    def _validate_declaration(self):
        if not self.producer:
            raise ValueError("producer must be non-empty")
        self._gsize = (
            None
            if self.gsize is None
            else positive_finite_float64(self.gsize, label="gsize")
        )
        self._source = RegularGrid.from_coordinates(
            self.source_lon, self.source_lat,
        )
        with nc.Dataset(Path(self.parameter_nc), "r") as dataset:
            lon = dataset.variables[self.lon_name][:]
            lat = dataset.variables[self.lat_name][:]
            raw_ids = dataset.variables[self.id_name][:]
            for label, value in (
                (self.lon_name, lon),
                (self.lat_name, lat),
                (self.id_name, raw_ids),
            ):
                if np.ma.isMaskedArray(value) and np.any(
                    np.ma.getmaskarray(value)
                ):
                    raise ValueError(
                        f"parameter variable {label!r} contains missing "
                        "values"
                    )
            if np.asarray(raw_ids).ndim != 1:
                raise ValueError(
                    f"parameter variable {self.id_name!r} must be 1-D"
                )
            target_ids = canonical_ids(raw_ids, label=self.id_name)
            gsize = self._gsize
            if gsize is None and "gsize" in dataset.ncattrs():
                gsize = positive_finite_float64(
                    dataset.getncattr("gsize"), label="gsize",
                )
        if self.method == "overlap" and gsize is None:
            raise ValueError(
                f"{Path(self.parameter_nc).name} requests overlap mapping "
                "but has no 'gsize' attribute (and none was passed) to "
                "build cell bounds; add 'gsize' or use method='nearest'."
            )
        self._gsize = gsize
        self._target = TargetSupport._from_points(
            lon,
            lat,
            target_ids=target_ids,
            cell_size=gsize if self.method == "overlap" else None,
        )
        return self

    @property
    def source(self) -> RegularGrid:
        return self._source

    @property
    def normalized_gsize(self) -> float | None:
        return self._gsize

    @property
    def target(self) -> TargetSupport:
        return self._target


class _AggregateFieldRequest(HydroForgeModel):
    field_nc: str | Path
    var_name: str
    mapping_npz: str | Path
    out_dir: str | Path
    out_name: str | None = None
    dtype: Literal["float32", "float64"] = "float32"
    netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS
    units: str = "mm"
    description: str | None = None
    normalized: bool = Field(default=False, strict=True)

    _plan: _AggregateFieldPlan = PrivateAttr()

    @model_validator(mode="after")
    def _validate_declaration(self):
        if not self.units:
            raise ValueError("units must be non-empty")
        if self.out_name is None:
            object.__setattr__(self, "out_name", self.var_name)
        object.__setattr__(
            self,
            "out_name",
            validate_safe_path_component(self.out_name, label="out_name"),
        )
        self._plan = _compile_aggregate_field_plan(self)
        return self

    @property
    def plan(self) -> _AggregateFieldPlan:
        return self._plan


@dataclass(frozen=True, slots=True)
class _AggregateFieldPlan:
    mapping: MappingTable
    field: np.ndarray
    has_time: bool
    time_values: np.ndarray | None
    time_attributes: Mapping[str, Any]


def _compile_aggregate_field_plan(
    request: _AggregateFieldRequest,
) -> _AggregateFieldPlan:
    """Validate and own all external arrays before aggregation begins."""

    mapping = MappingTable._load(request.mapping_npz)
    if request.normalized:
        mapping = mapping._row_normalized()

    time_values = None
    time_attributes: dict[str, Any] = {}
    with nc.Dataset(Path(request.field_nc), "r") as dataset:
        variable = dataset.variables[request.var_name]
        if variable.ndim not in {2, 3}:
            raise ValueError(
                f"field variable {request.var_name!r} must be 2-D or 3-D; "
                f"got shape {variable.shape}"
            )
        if tuple(variable.shape[-2:]) != mapping._source_shape:
            raise ValueError(
                f"field variable {request.var_name!r} has spatial shape "
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
                    f"spatial coordinate {dimension!r} must be "
                    "one-dimensional"
                )
            raw_coordinate = coordinate[:]
            if np.ma.isMaskedArray(raw_coordinate) and np.any(
                np.ma.getmaskarray(raw_coordinate)
            ):
                raise ValueError(
                    f"spatial coordinate {dimension!r} contains missing "
                    "values"
                )
            observed = canonical_float64(
                raw_coordinate,
                label=f"spatial coordinate {dimension!r}",
            )
            if (
                observed.shape != expected.shape
                or not np.isfinite(observed).all()
                or not np.array_equal(observed, expected)
            ):
                raise ValueError(
                    f"spatial coordinate {dimension!r} does not match the "
                    f"mapping source {label}-axis"
                )
        if np.dtype(variable.dtype).kind not in {"f", "i", "u"}:
            raise ValueError(
                f"field variable {request.var_name!r} must contain real "
                "numeric values"
            )
        has_time = variable.ndim == 3
        ntime = int(variable.shape[0]) if has_time else 1
        field = variable[:]
        if has_time:
            time_dimension = variable.dimensions[0]
            time_variable = dataset.variables[time_dimension]
            if time_variable.dimensions != (time_dimension,):
                raise ValueError(
                    f"time coordinate {time_dimension!r} must be "
                    "one-dimensional"
                )
            raw_time = time_variable[:]
            if np.ma.isMaskedArray(raw_time) and np.any(
                np.ma.getmaskarray(raw_time)
            ):
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
            time_attributes = {
                name: time_variable.getncattr(name)
                for name in time_variable.ncattrs()
                if name != "_FillValue"
            }

    if np.ma.isMaskedArray(field):
        mask = np.ma.getmaskarray(field)
        raw_field = np.asarray(field.data)
        valid_values = canonical_floating_array(
            raw_field[~mask],
            dtype="float64",
            label=f"field variable {request.var_name!r}",
        )
        canonical_field = np.empty(raw_field.shape, dtype=np.float64)
        canonical_field[~mask] = valid_values
        canonical_field[mask] = np.nan
    else:
        canonical_field = canonical_floating_array(
            field,
            dtype="float64",
            label=f"field variable {request.var_name!r}",
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
    request = _BuildCamaMappingRequest(
        source_lon=source_lon,
        source_lat=source_lat,
        map_dir=map_dir,
        hires_tag=hires_tag,
        mapinfo_txt=mapinfo_txt,
        lowres_idx_precision=lowres_idx_precision,
        hires_idx_precision=hires_idx_precision,
        map_precision=map_precision,
        parameter_nc=parameter_nc,
        allow_oob_zero=allow_oob_zero,
        producer=producer,
    )
    allow_oob_zero = request.allow_oob_zero
    producer = request.producer
    source = request.source
    (
        target_ids,
        catchment_id_hires,
        valid_areas,
        valid_lon,
        valid_lat,
    ) = request.mapping_inputs

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
    method: Literal["nearest", "overlap"] = "overlap",
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
    request = _BuildPointMappingRequest(
        source_lon=source_lon,
        source_lat=source_lat,
        parameter_nc=parameter_nc,
        method=method,
        lon_name=lon_name,
        lat_name=lat_name,
        id_name=id_name,
        gsize=gsize,
        producer=producer,
    )
    method = request.method
    producer = request.producer
    source = request.source
    return _build_regular_grid_mapping_trusted(
        source=source,
        target=request.target,
        method=method,
        normalization="mean",
        metadata={"producer": producer},
    )


def aggregate_field_to_nc(
    field_nc: Union[str, Path],
    var_name: str,
    mapping_npz: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    out_name: Optional[str] = None,
    dtype: Literal["float32", "float64"] = "float32",
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
    request = _AggregateFieldRequest(
        field_nc=field_nc,
        var_name=var_name,
        mapping_npz=mapping_npz,
        out_dir=out_dir,
        out_name=out_name,
        dtype=dtype,
        netcdf_options=netcdf_options,
        units=units,
        description=description,
        normalized=normalized,
    )
    out_dir = request.out_dir
    out_name = request.out_name
    dtype = request.dtype
    netcdf_options = request.netcdf_options
    units = request.units
    description = request.description
    plan = request.plan
    mapping = plan.mapping
    field = plan.field
    has_time = plan.has_time
    time_values = plan.time_values
    time_attributes = plan.time_attributes

    aggregated_float64 = canonical_floating_array(
        mapping._apply_trusted(field, layout="grid"),
        dtype="float64",
        label="aggregated values",
        allow_nan=True,
    )
    aggregated = canonical_floating_array(
        aggregated_float64,
        dtype=dtype,
        label="aggregated values",
        allow_nan=True,
    )
    n_catch = mapping.matrix.shape[0]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / f"{out_name}_rank0.nc"
    dtype_nc = "f4" if dtype == "float32" else "f8"

    with _atomic_netcdf_dataset_trusted(nc_path, format="NETCDF4") as ds:
        ds.setncattr("title", f"Aggregated catchment parameter ({out_name})")
        if has_time:
            ds.createDimension("time", None)
        ds.createDimension("saved_points", n_catch)

        if has_time:
            time_var = ds.createVariable("time", time_values.dtype, ("time",))
            time_var.setncatts(time_attributes)
            time_var[:] = time_values

        output_coord = ds.createVariable("catchment_id", "i8", ("saved_points",))
        output_coord[:] = mapping.target_ids

        dims = ("time", "saved_points") if has_time else ("saved_points",)
        create_options = prepare_netcdf_variable_options(
            netcdf_options, dtype=dtype_nc, dimensions=dims, name=out_name,
        )
        out_var = _create_netcdf_variable_trusted(
            ds,
            out_name,
            dtype_nc,
            dims,
            options=create_options,
        )
        resolved_description = (
            f"Catchment-aggregated {out_name}"
            if description is None else description
        )
        out_var.setncattr("description", resolved_description)
        out_var.setncattr("units", units)

        if has_time:
            out_var[:, :] = aggregated
        else:
            out_var[:] = aggregated[0]

    return nc_path

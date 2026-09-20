"""Overlap engines that turn source/target geometry into mapping weights.

Two engines share a CSR output via :mod:`hydroforge.data.mapping.build`:

* :func:`regular_overlap_rows` -- analytic separable overlap between a source
  regular grid and axis-aligned rectangular target cells.  On geographic grids
  the per-cell weight is the true spherical overlap area
  (``R^2 * dlon_rad * (sin(lat_hi) - sin(lat_lo))``), so the area weighting is
  latitude-correct without any external dependency.
* :func:`aggregate_hires_coo` -- vectorized area-weighted aggregation of
  high-resolution pixels (e.g. MERIT ``catmxy``) onto source grid cells, for
  catchments that are unions of many hires pixels.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ValidationInfo, field_validator, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.mapping.grid import RegularGrid
from hydroforge.data.mapping.target import TargetSupport
from hydroforge.data.numeric import (
    canonical_float64,
    canonical_floating_array,
    canonical_ids,
)

_EARTH_RADIUS_M = 6371007.2


def normalise_row(values: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError("mapping row weights contain missing values")
    row = np.asarray(values)
    if row.ndim != 1:
        raise ValueError("mapping row weights must be one-dimensional")
    if row.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("mapping row weights must contain real numbers")
    row = canonical_floating_array(
        row,
        dtype="float64",
        label="mapping row weights",
    )
    if np.any(row < 0.0):
        raise ValueError("mapping row weights must be finite and nonnegative")
    scale = float(row.max(initial=0.0))
    if scale <= 0.0:
        raise ValueError("mapping row weights must have a positive sum")
    scaled = row / scale
    total = float(scaled.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise OverflowError("mapping row weights cannot be normalized in float64")
    return scaled / total


def regular_overlap_rows(
    source: RegularGrid,
    target: TargetSupport,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Analytic separable overlap between ``source`` cells and target rectangles.

    For each target cell the overlap with the source grid is separable into a
    1-D longitude interval overlap and a 1-D latitude interval overlap.  On a
    geographic grid the weight is the spherical overlap area
    ``R^2 * dlon_rad * (sin(phi_hi) - sin(phi_lo))`` (latitude-correct); on a
    projected grid it is the planar overlap area.

    Returns one ``(source_cols, weights, coverage)`` tuple per target, where
    ``source_cols`` index the C-order ``(y, x)`` flattened source grid and
    ``coverage`` is the covered-area fraction in the same geometry used by
    the returned weights.
    """
    if target.bounds is None:
        raise ValueError("overlap requires target cell bounds")

    x_lo = source.x_bounds[:, 0]
    x_hi = source.x_bounds[:, 1]
    y_lo = source.y_bounds[:, 0]
    y_hi = source.y_bounds[:, 1]
    nx = source.x.size
    geographic = bool(source.is_geographic)
    shifted_longitude_convention = geographic and bool(
        np.min(source.x) < -180.0 or np.max(source.x) > 180.0
    )
    periodic_x = geographic and source._periodic_x
    align_longitude = periodic_x or shifted_longitude_convention
    source_center = 0.5 * (float(np.min(x_lo)) + float(np.max(x_hi)))

    rows: list[tuple[np.ndarray, np.ndarray, float]] = []
    for xmin, xmax, ymin, ymax in target.bounds:
        target_width = float(xmax - xmin)
        if geographic and (ymin < -90.0 or ymax > 90.0):
            raise ValueError(
                "geographic target latitude bounds must lie within [-90, 90]"
            )
        if periodic_x:
            if target_width > 360.0 + 1e-9:
                raise ValueError(
                    "geographic target longitude width cannot exceed 360 degrees"
                )
        base_shift = 0.0
        if align_longitude:
            target_center = 0.5 * (float(xmin) + float(xmax))
            base_shift = 360.0 * round((source_center - target_center) / 360.0)
        if periodic_x:
            # Align one target copy with the source convention, then include
            # its neighbours so seam-crossing cells are split across both
            # ends of a periodic grid.  Summing per source cell avoids
            # duplicate column indices in the resulting sparse row.
            shifts = (base_shift - 360.0, base_shift, base_shift + 360.0)
        else:
            shifts = (base_shift,)
        for index, shift in enumerate(shifts):
            overlap = np.clip(
                np.minimum(xmax + shift, x_hi) - np.maximum(xmin + shift, x_lo),
                0.0,
                None,
            )
            if index == 0:
                lon_overlap = overlap
            else:
                lon_overlap += overlap
        lat_lo = np.maximum(ymin, y_lo)
        lat_hi = np.minimum(ymax, y_hi)
        lat_overlap = np.clip(lat_hi - lat_lo, 0.0, None)

        col_idx = np.nonzero(lon_overlap > 0.0)[0]
        row_idx = np.nonzero(lat_overlap > 0.0)[0]
        if col_idx.size == 0 or row_idx.size == 0:
            rows.append(
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0.0)
            )
            continue

        if geographic:
            lon_weight = np.radians(lon_overlap[col_idx]) * _EARTH_RADIUS_M
            lat_weight = (
                np.sin(np.radians(lat_hi[row_idx]))
                - np.sin(np.radians(lat_lo[row_idx]))
            ) * _EARTH_RADIUS_M
        else:
            lon_weight = lon_overlap[col_idx]
            lat_weight = lat_overlap[row_idx]

        area = lat_weight[:, None] * lon_weight[None, :]
        cols = (row_idx[:, None] * nx + col_idx[None, :]).ravel().astype(np.int64)
        values = area.ravel().astype(np.float64)

        if geographic:
            target_area = (
                np.radians(target_width)
                * _EARTH_RADIUS_M
                * _EARTH_RADIUS_M
                * (np.sin(np.radians(ymax)) - np.sin(np.radians(ymin)))
            )
        else:
            target_area = float(target_width * (ymax - ymin))
        coverage = (
            float(values.sum(dtype=np.float64)) / target_area
            if target_area > 0.0
            else 0.0
        )
        rows.append((cols, values, float(coverage)))
    return rows


class _HiresPixelDeclaration(HydroForgeModel):
    """Canonical high-resolution pixel inputs shared by both mapping entries."""

    source: RegularGrid
    target_ids: np.ndarray
    pixel_catchment_id: np.ndarray
    pixel_area: np.ndarray
    pixel_lon: np.ndarray
    pixel_lat: np.ndarray
    allow_oob_zero: bool = False

    @field_validator("target_ids", "pixel_catchment_id")
    @classmethod
    def _validate_ids(cls, value: np.ndarray, info: ValidationInfo):
        return canonical_ids(value, label=info.field_name)

    @field_validator("pixel_lon", "pixel_lat")
    @classmethod
    def _validate_coordinate(cls, value: np.ndarray, info: ValidationInfo):
        if value.ndim != 1:
            raise ValueError(f"{info.field_name} must be one-dimensional")
        return canonical_float64(value, label=info.field_name)

    @field_validator("pixel_area")
    @classmethod
    def _validate_area(cls, value: np.ndarray):
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError("pixel_area contains missing values")
        raw = np.asarray(value)
        if raw.ndim != 1:
            raise ValueError("pixel_area must be one-dimensional")
        areas = canonical_floating_array(raw, dtype="float64", label="pixel_area")
        if np.any(areas < 0.0):
            raise ValueError("pixel_area must be nonnegative")
        return areas

    @model_validator(mode="after")
    def _validate_pixels(self) -> Self:
        if np.unique(self.target_ids).size != self.target_ids.size:
            raise ValueError("target_ids must be unique")
        sizes = {
            name: getattr(self, name).size
            for name in ("pixel_catchment_id", "pixel_area", "pixel_lon", "pixel_lat")
        }
        if len(set(sizes.values())) != 1:
            raise ValueError(f"hires pixel arrays must have equal sizes: {sizes}")
        return self


def aggregate_hires_coo(
    source: RegularGrid,
    target_ids: np.ndarray,
    pixel_catchment_id: np.ndarray,
    pixel_area: np.ndarray,
    pixel_lon: np.ndarray,
    pixel_lat: np.ndarray,
    *,
    allow_oob_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted aggregation of hires pixels onto source grid cells.

    Returns ``(rows, cols, data)`` COO triplets where ``rows`` index into
    ``target_ids`` (the catchment that each pixel drains to) and ``cols`` index
    flattened source grid cells.  Pixels whose catchment is absent from
    ``target_ids`` are dropped.  Coordinates outside the source grid raise by
    default; when ``allow_oob_zero`` is true, those pixels are dropped so their
    contribution is zero.
    """
    declaration = _HiresPixelDeclaration(
        source=source,
        target_ids=target_ids,
        pixel_catchment_id=pixel_catchment_id,
        pixel_area=pixel_area,
        pixel_lon=pixel_lon,
        pixel_lat=pixel_lat,
        allow_oob_zero=allow_oob_zero,
    )
    return _aggregate_hires_coo_trusted(declaration)


def _aggregate_hires_coo_trusted(
    declaration: _HiresPixelDeclaration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from hydroforge.data.distributed import _find_indices_in_trusted

    catchment_idx = _find_indices_in_trusted(
        declaration.pixel_catchment_id, declaration.target_ids
    )
    allow_oob_zero = declaration.allow_oob_zero
    try:
        source_idx = declaration.source._index_of_points_trusted(
            declaration.pixel_lon,
            declaration.pixel_lat,
            allow_oob=allow_oob_zero,
        )
    except ValueError as exc:
        if not allow_oob_zero and "points fall outside the source grid" in str(exc):
            raise ValueError(
                f"{exc}; set allow_oob_zero=True to ignore out-of-bounds "
                "hires pixels as zero contribution"
            ) from exc
        raise
    valid = (catchment_idx != -1) & (source_idx != -1)
    rows = catchment_idx[valid].astype(np.int64, copy=False)
    cols = source_idx[valid]
    # Keep source areas in float64 until duplicate COO entries have been
    # coalesced by scipy.  Casting each pixel before that reduction loses
    # measurable area when hundreds of hires pixels map to one source cell.
    data = declaration.pixel_area[valid]
    return rows, cols, data

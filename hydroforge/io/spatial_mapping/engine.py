"""Overlap engines that turn source/target geometry into mapping weights.

Two engines share a CSR output via :mod:`hydroforge.io.spatial_mapping.build`:

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

import numpy as np

from hydroforge.io.spatial_mapping.grid import RegularGrid
from hydroforge.io.spatial_mapping.target import TargetSupport


_EARTH_RADIUS_M = 6371007.2


def normalise_row(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        return values.astype(np.float32, copy=False)
    return (values / total).astype(np.float32, copy=False)


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
    ``coverage`` is the planar covered-area fraction used for validation.
    """
    if target.bounds is None:
        raise ValueError("overlap requires target cell bounds")

    x_lo = source.x_bounds[:, 0]
    x_hi = source.x_bounds[:, 1]
    y_lo = source.y_bounds[:, 0]
    y_hi = source.y_bounds[:, 1]
    nx = source.x.size
    geographic = bool(source.is_geographic)

    rows: list[tuple[np.ndarray, np.ndarray, float]] = []
    for xmin, xmax, ymin, ymax in target.bounds:
        lon_overlap = np.clip(np.minimum(xmax, x_hi) - np.maximum(xmin, x_lo), 0.0, None)
        lat_lo = np.maximum(ymin, y_lo)
        lat_hi = np.minimum(ymax, y_hi)
        lat_overlap = np.clip(lat_hi - lat_lo, 0.0, None)

        col_idx = np.nonzero(lon_overlap > 0.0)[0]
        row_idx = np.nonzero(lat_overlap > 0.0)[0]
        if col_idx.size == 0 or row_idx.size == 0:
            rows.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0.0))
            continue

        if geographic:
            lon_weight = np.radians(lon_overlap[col_idx]) * _EARTH_RADIUS_M
            lat_weight = (
                np.sin(np.radians(lat_hi[row_idx])) - np.sin(np.radians(lat_lo[row_idx]))
            ) * _EARTH_RADIUS_M
        else:
            lon_weight = lon_overlap[col_idx]
            lat_weight = lat_overlap[row_idx]

        area = lat_weight[:, None] * lon_weight[None, :]
        cols = (row_idx[:, None] * nx + col_idx[None, :]).ravel().astype(np.int64)
        values = area.ravel().astype(np.float64)

        covered_planar = float(
            np.sum(lat_overlap[row_idx][:, None] * lon_overlap[col_idx][None, :])
        )
        target_planar = float((xmax - xmin) * (ymax - ymin))
        coverage = covered_planar / target_planar if target_planar > 0.0 else 0.0
        rows.append((cols, values, float(coverage)))
    return rows


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
    ``target_ids`` or whose coordinates fall outside the source grid are dropped.
    """
    from hydroforge.modeling.distributed import find_indices_in

    target_ids = np.asarray(target_ids, dtype=np.int64)
    catchment_idx = find_indices_in(np.asarray(pixel_catchment_id, dtype=np.int64), target_ids)
    source_idx = source.index_of_points(pixel_lon, pixel_lat, allow_oob=not allow_oob_zero)
    source_idx = np.asarray(source_idx, dtype=np.int64).ravel()

    valid = (catchment_idx != -1) & (source_idx != -1)
    rows = catchment_idx[valid].astype(np.int64)
    cols = source_idx[valid].astype(np.int64)
    data = np.asarray(pixel_area, dtype=np.float32)[valid]
    return rows, cols, data

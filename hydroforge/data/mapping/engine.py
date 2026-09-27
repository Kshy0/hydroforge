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

from typing import NamedTuple, Self

import numpy as np
from pydantic import ValidationInfo, field_validator, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.mapping.grid import RegularGrid, _has_duplicates
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


_OVERLAP_CHUNK_ENTRIES = 1 << 22


def _segment_arange(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(start, start + length)`` over all segments."""

    offsets = np.cumsum(lengths) - lengths
    return np.repeat(starts - offsets, lengths) + np.arange(
        int(lengths.sum()), dtype=np.int64
    )


def _segment_sums(
    values: np.ndarray, offsets: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Per-segment sums in the order of a 1-D ``ndarray.sum`` on each segment.

    Equal-length segments are reduced together along a contiguous last axis,
    which NumPy sums with the same pairwise algorithm as a 1-D reduction.
    """

    sums = np.zeros(lengths.size, dtype=np.float64)
    if lengths.size == 0:
        return sums
    order = np.argsort(lengths, kind="stable")
    sorted_lengths = lengths[order]
    starts = np.flatnonzero(np.r_[True, sorted_lengths[1:] != sorted_lengths[:-1]])
    for begin, end in zip(starts, np.r_[starts[1:], order.size]):
        length = int(sorted_lengths[begin])
        if length == 0:
            continue
        members = order[begin:end]
        block = values[offsets[members, None] + np.arange(length, dtype=np.int64)]
        sums[members] = block.sum(axis=1)
    return sums


def _axis_ranges(
    lower: np.ndarray,
    upper: np.ndarray,
    ascending: bool,
    start: np.ndarray,
    stop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Original-index ``[first, last)`` cells with positive overlap.

    A cell overlaps ``(start, stop)`` with positive length exactly when
    ``upper > start``, ``lower < stop`` and ``stop > start``; ordered,
    non-overlapping cell bounds make that set contiguous.
    """

    size = lower.size
    if not ascending:
        lower = lower[::-1]
        upper = upper[::-1]
    first = np.searchsorted(upper, start, side="right")
    last = np.searchsorted(lower, stop, side="left")
    last = np.where((stop > start) & (last > first), last, first)
    if ascending:
        return first.astype(np.int64), last.astype(np.int64)
    return (size - last).astype(np.int64), (size - first).astype(np.int64)


class _OverlapRows(NamedTuple):
    indptr: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    coverage: np.ndarray


def _regular_overlap_row_trusted(
    source: RegularGrid,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    shifts: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Scan every source cell for one target (periodic seam fallback)."""

    x_lo = source.x_bounds[:, 0]
    x_hi = source.x_bounds[:, 1]
    y_lo = source.y_bounds[:, 0]
    y_hi = source.y_bounds[:, 1]
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
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if source.is_geographic:
        lon_weight = np.radians(lon_overlap[col_idx]) * _EARTH_RADIUS_M
        lat_weight = (
            np.sin(np.radians(lat_hi[row_idx])) - np.sin(np.radians(lat_lo[row_idx]))
        ) * _EARTH_RADIUS_M
    else:
        lon_weight = lon_overlap[col_idx]
        lat_weight = lat_overlap[row_idx]
    area = lat_weight[:, None] * lon_weight[None, :]
    cols = (row_idx[:, None] * source.x.size + col_idx[None, :]).ravel()
    return cols.astype(np.int64), area.ravel().astype(np.float64)


def _regular_overlap_csr_trusted(
    source: RegularGrid,
    target: TargetSupport,
) -> _OverlapRows:
    """Vectorised separable overlap of every target, as CSR rows.

    Rows are bit-identical to scanning all source cells per target: the same
    elementwise float64 expressions are evaluated for exactly the cells with
    positive overlap, located by binary search on the ordered cell bounds.
    """

    bounds = target.bounds
    n_target = bounds.shape[0]
    xmin, xmax, ymin, ymax = (np.ascontiguousarray(bounds[:, k]) for k in range(4))
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

    target_width = xmax - xmin
    invalid_latitude = (
        (ymin < -90.0) | (ymax > 90.0) if geographic else np.zeros(n_target, bool)
    )
    invalid_width = (
        target_width > 360.0 + 1e-9 if periodic_x else np.zeros(n_target, bool)
    )
    failed = np.flatnonzero(invalid_latitude | invalid_width)
    if failed.size:
        if invalid_latitude[failed[0]]:
            raise ValueError(
                "geographic target latitude bounds must lie within [-90, 90]"
            )
        raise ValueError("geographic target longitude width cannot exceed 360 degrees")

    base_shift = np.zeros(n_target, dtype=np.float64)
    if align_longitude:
        target_center = 0.5 * (xmin + xmax)
        base_shift = 360.0 * np.round((source_center - target_center) / 360.0)
    shifts = (
        np.stack((base_shift - 360.0, base_shift, base_shift + 360.0), axis=1)
        if periodic_x
        else base_shift[:, None]
    )
    x_start = xmin[:, None] + shifts
    x_stop = xmax[:, None] + shifts
    x_ascending = nx == 1 or bool(source.x[1] > source.x[0])
    col_first, col_last = _axis_ranges(
        x_lo, x_hi, x_ascending, x_start.ravel(), x_stop.ravel()
    )
    col_first = col_first.reshape(shifts.shape)
    col_last = col_last.reshape(shifts.shape)
    # Order each target's (up to three) column ranges by start so the flat
    # column list is ascending like np.nonzero over all source columns.
    range_order = np.argsort(col_first, axis=1, kind="stable")
    col_first = np.take_along_axis(col_first, range_order, axis=1)
    col_last = np.take_along_axis(col_last, range_order, axis=1)
    x_start = np.take_along_axis(x_start, range_order, axis=1)
    x_stop = np.take_along_axis(x_stop, range_order, axis=1)
    col_lengths = col_last - col_first
    nonempty = col_lengths > 0
    # A seam cell reached through two shifts sums both overlaps; leave those
    # rare near-360-degree targets to the exhaustive per-target scan.
    shared_columns = np.zeros(n_target, dtype=bool)
    for k in range(1, shifts.shape[1]):
        previous_last = np.max(
            np.where(nonempty[:, :k], col_last[:, :k], np.iinfo(np.int64).min), axis=1
        )
        shared_columns |= nonempty[:, k] & (col_first[:, k] < previous_last)

    y_ascending = source.y.size == 1 or bool(source.y[1] > source.y[0])
    row_first, row_last = _axis_ranges(y_lo, y_hi, y_ascending, ymin, ymax)
    n_rows = row_last - row_first
    n_cols = col_lengths.sum(axis=1)
    n_rows = np.where(n_cols > 0, n_rows, 0)
    n_cols = np.where(n_rows > 0, n_cols, 0)
    entries = n_rows * n_cols
    entries[shared_columns] = 0

    if geographic:
        target_area = (
            np.radians(target_width)
            * _EARTH_RADIUS_M
            * _EARTH_RADIUS_M
            * (np.sin(np.radians(ymax)) - np.sin(np.radians(ymin)))
        )
    else:
        target_area = target_width * (ymax - ymin)

    fallback = np.flatnonzero(shared_columns)
    fallback_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in fallback:
        fallback_rows[int(index)] = _regular_overlap_row_trusted(
            source,
            xmin[index],
            xmax[index],
            ymin[index],
            ymax[index],
            tuple(float(shift) for shift in shifts[index]),
        )
        entries[index] = fallback_rows[int(index)][0].size

    indptr = np.zeros(n_target + 1, dtype=np.int64)
    np.cumsum(entries, out=indptr[1:])
    cols = np.empty(int(indptr[-1]), dtype=np.int64)
    values = np.empty(int(indptr[-1]), dtype=np.float64)

    chunk_start = 0
    while chunk_start < n_target:
        chunk_stop = int(
            np.searchsorted(
                indptr, indptr[chunk_start] + _OVERLAP_CHUNK_ENTRIES, side="right"
            )
        )
        chunk_stop = min(max(chunk_stop - 1, chunk_start + 1), n_target)
        chunk = np.arange(chunk_start, chunk_stop, dtype=np.int64)
        chunk = chunk[(entries[chunk] > 0) & ~shared_columns[chunk]]
        chunk_start = chunk_stop
        if chunk.size == 0:
            continue

        # Column entries: every (target, shift) range, in ascending order.
        seg_lengths = col_lengths[chunk].ravel()
        seg_columns = _segment_arange(col_first[chunk].ravel(), seg_lengths)
        seg_start = np.repeat(x_start[chunk].ravel(), seg_lengths)
        seg_stop = np.repeat(x_stop[chunk].ravel(), seg_lengths)
        lon_overlap = np.minimum(seg_stop, x_hi[seg_columns]) - np.maximum(
            seg_start, x_lo[seg_columns]
        )
        chunk_cols = n_cols[chunk]
        col_offsets = np.cumsum(chunk_cols) - chunk_cols

        chunk_rows = n_rows[chunk]
        row_offsets = np.cumsum(chunk_rows) - chunk_rows
        row_index = _segment_arange(row_first[chunk], chunk_rows)
        row_ymin = np.repeat(ymin[chunk], chunk_rows)
        row_ymax = np.repeat(ymax[chunk], chunk_rows)
        lat_lo = np.maximum(row_ymin, y_lo[row_index])
        lat_hi = np.minimum(row_ymax, y_hi[row_index])
        if geographic:
            lon_weight = np.radians(lon_overlap) * _EARTH_RADIUS_M
            lat_weight = (
                np.sin(np.radians(lat_hi)) - np.sin(np.radians(lat_lo))
            ) * _EARTH_RADIUS_M
        else:
            lon_weight = lon_overlap
            lat_weight = lat_hi - lat_lo

        chunk_entries = entries[chunk]
        owner = np.repeat(np.arange(chunk.size, dtype=np.int64), chunk_entries)
        local = np.arange(owner.size, dtype=np.int64) - np.repeat(
            np.cumsum(chunk_entries) - chunk_entries, chunk_entries
        )
        local_row = local // chunk_cols[owner]
        local_col = local - local_row * chunk_cols[owner]
        row_at = row_offsets[owner] + local_row
        col_at = col_offsets[owner] + local_col
        destination = _segment_arange(indptr[chunk], chunk_entries)
        values[destination] = lat_weight[row_at] * lon_weight[col_at]
        cols[destination] = row_index[row_at] * nx + seg_columns[col_at]

    for index, (row_cols, row_values) in fallback_rows.items():
        cols[indptr[index] : indptr[index + 1]] = row_cols
        values[indptr[index] : indptr[index + 1]] = row_values

    sums = _segment_sums(values, indptr[:-1], entries)
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(target_area > 0.0, sums / target_area, 0.0)
    coverage[entries == 0] = 0.0
    return _OverlapRows(indptr=indptr, cols=cols, values=values, coverage=coverage)


def _normalise_rows_trusted(rows: _OverlapRows) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`normalise_row` to every non-empty CSR row at once.

    Returns the normalised values and a mask of rows that the scalar helper
    would reject; the caller raises for them in row order.
    """

    lengths = np.diff(rows.indptr)
    starts = rows.indptr[:-1]
    values = rows.values
    nonempty = lengths > 0
    scale = np.zeros(lengths.size, dtype=np.float64)
    if values.size:
        scale[nonempty] = np.maximum.reduceat(values, starts[nonempty])
    anomalous = nonempty & ~(np.isfinite(scale) & (scale > 0.0))
    if not np.isfinite(values).all() or np.any(values < 0.0):
        owner = np.repeat(np.arange(lengths.size), lengths)
        bad = ~np.isfinite(values) | (values < 0.0)
        anomalous[owner[bad]] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = values / np.repeat(scale, lengths)
    totals = _segment_sums(scaled, starts, lengths)
    anomalous |= nonempty & ~(np.isfinite(totals) & (totals > 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return scaled / np.repeat(totals, lengths), anomalous


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
    rows = _regular_overlap_csr_trusted(source, target)
    return [
        (
            rows.cols[start:stop].copy(),
            rows.values[start:stop].copy(),
            float(coverage),
        )
        for start, stop, coverage in zip(
            rows.indptr[:-1].tolist(), rows.indptr[1:].tolist(), rows.coverage
        )
    ]


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
        if _has_duplicates(self.target_ids):
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


def _raise_hires_oob_hint(exc: ValueError, *, allow_oob_zero: bool) -> None:
    if not allow_oob_zero and "points fall outside the source grid" in str(exc):
        raise ValueError(
            f"{exc}; set allow_oob_zero=True to ignore out-of-bounds "
            "hires pixels as zero contribution"
        ) from exc


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
        _raise_hires_oob_hint(exc, allow_oob_zero=allow_oob_zero)
        raise
    return _hires_coo_trusted(catchment_idx, source_idx, declaration.pixel_area)


def _hires_coo_trusted(
    catchment_idx: np.ndarray,
    source_idx: np.ndarray,
    pixel_area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (catchment_idx != -1) & (source_idx != -1)
    rows = catchment_idx[valid].astype(np.int64, copy=False)
    cols = source_idx[valid]
    # Keep source areas in float64 until duplicate COO entries have been
    # coalesced by scipy.  Casting each pixel before that reduction loses
    # measurable area when hundreds of hires pixels map to one source cell.
    data = pixel_area[valid]
    return rows, cols, data

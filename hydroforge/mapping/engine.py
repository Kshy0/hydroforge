# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Overlap engine that turns source/target geometry into mapping weights.

:func:`_regular_overlap_csr` computes the analytic separable overlap between a
source regular grid and axis-aligned rectangular target cells as CSR rows for
:mod:`hydroforge.mapping.build`.  On geographic grids the per-cell weight is
the true spherical overlap area
(``R^2 * dlon_rad * (sin(lat_hi) - sin(lat_lo))``), so the area weighting is
latitude-correct without any external dependency.  High-resolution pixel
aggregation (e.g. MERIT ``catmxy``) is assembled from COO triplets by
:func:`_hires_coo`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from hydroforge.core.arrays import canonical_floating_array
from hydroforge.mapping.grid import RegularGrid
from hydroforge.mapping.target import TargetSupport

_EARTH_RADIUS_M = 6371007.2


def _normalise_row(values: np.ndarray) -> np.ndarray:
    """Scale one row's weights to sum one, rejecting unusable rows."""

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


def _area_factors(
    width: np.ndarray, ymin: np.ndarray, ymax: np.ndarray, *, geographic: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Separable ``(lon, lat)`` factors of the cell area ``lat * lon``.

    Geographic factors are ``R * dlon_rad`` and ``R * (sin(lat_hi) -
    sin(lat_lo))``, so the area is spherical (m^2); overlap rows evaluate
    each factor once per source column or row.
    """

    if geographic:
        return (
            np.radians(width) * _EARTH_RADIUS_M,
            (np.sin(np.radians(ymax)) - np.sin(np.radians(ymin))) * _EARTH_RADIUS_M,
        )
    return width, ymax - ymin


def _cell_area(
    width: np.ndarray, ymin: np.ndarray, ymax: np.ndarray, *, geographic: bool
) -> np.ndarray:
    """Area of axis-aligned cells: spherical (m^2) on geographic grids."""

    lon_weight, lat_weight = _area_factors(width, ymin, ymax, geographic=geographic)
    return lat_weight * lon_weight


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


def _regular_overlap_row(
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
    lon_weight, lat_weight = _area_factors(
        lon_overlap[col_idx],
        lat_lo[row_idx],
        lat_hi[row_idx],
        geographic=source.is_geographic,
    )
    area = lat_weight[:, None] * lon_weight[None, :]
    cols = (row_idx[:, None] * source.x.size + col_idx[None, :]).ravel()
    return cols.astype(np.int64), area.ravel().astype(np.float64)


def _regular_overlap_csr(
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
    periodic_x = geographic and source._periodic_x
    align_longitude = geographic
    source_center = 0.5 * float(np.min(x_lo)) + 0.5 * float(np.max(x_hi))

    target_width = xmax - xmin
    target_height = ymax - ymin
    if not np.isfinite(target_width).all() or not np.isfinite(target_height).all():
        raise ValueError("target widths and heights must be finite")
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
        target_center = 0.5 * xmin + 0.5 * xmax
        base_shift = 360.0 * np.round((source_center - target_center) / 360.0)
    shifts = (
        np.stack((base_shift - 360.0, base_shift, base_shift + 360.0), axis=1)
        if periodic_x
        else base_shift[:, None]
    )
    x_start = xmin[:, None] + shifts
    x_stop = xmax[:, None] + shifts
    if not np.isfinite(x_start).all() or not np.isfinite(x_stop).all():
        raise ValueError("shifted longitude bounds must be finite")
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

    target_area = _cell_area(target_width, ymin, ymax, geographic=geographic)
    if not np.all(np.isfinite(target_area) & (target_area > 0)):
        raise ValueError("target areas must be finite and positive")

    fallback = np.flatnonzero(shared_columns)
    fallback_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in fallback:
        fallback_rows[int(index)] = _regular_overlap_row(
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
        lon_weight, lat_weight = _area_factors(
            lon_overlap, lat_lo, lat_hi, geographic=geographic
        )

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


def _normalise_rows(rows: _OverlapRows) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`_normalise_row` to every non-empty CSR row at once.

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


def _raise_hires_oob_hint(exc: ValueError, *, allow_oob_zero: bool) -> None:
    if not allow_oob_zero and "points fall outside the source grid" in str(exc):
        raise ValueError(
            f"{exc}; set allow_oob_zero=True to ignore out-of-bounds "
            "hires pixels as zero contribution"
        ) from exc


def _hires_coo(
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

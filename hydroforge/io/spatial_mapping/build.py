"""Orchestrators that assemble :class:`MappingTable` objects from engines."""
from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
from scipy.sparse import csr_matrix

from hydroforge.io.spatial_mapping.engine import (
    aggregate_hires_coo,
    normalise_row,
    regular_overlap_rows,
)
from hydroforge.io.spatial_mapping.grid import RegularGrid
from hydroforge.io.spatial_mapping.table import MappingTable
from hydroforge.io.spatial_mapping.target import TargetSupport


MappingMethod = Literal["nearest", "overlap"]
Normalization = Literal["mean", "sum"]
_MIN_FULL_COVERAGE = 1.0 - 1e-6


def build_regular_grid_mapping(
    source: RegularGrid,
    target: TargetSupport,
    *,
    method: MappingMethod = "overlap",
    normalization: Normalization = "mean",
    metadata: Mapping[str, Any] | None = None,
) -> MappingTable:
    """Build a sparse ``target x source`` mapping table."""
    if normalization not in ("mean", "sum"):
        raise ValueError("normalization must be 'mean' or 'sum'")
    method_name = str(method)
    if method_name not in ("nearest", "overlap"):
        raise ValueError(f"Unsupported mapping method: {method}")
    if method_name == "overlap" and target.bounds is None:
        raise ValueError("overlap requires target bounds")
    if method_name == "nearest" and (target.x is None or target.y is None):
        raise ValueError("nearest requires target center coordinates")

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    coverage = np.zeros(target.target_ids.size, dtype=np.float32)

    if method_name == "nearest":
        source_idx = source.index_of_points(target.x, target.y, allow_oob=False)
        target_rows = np.arange(source_idx.size, dtype=np.int64)
        rows.extend(target_rows.tolist())
        cols.extend(source_idx.astype(np.int64).tolist())
        values.extend(np.ones(target_rows.size, dtype=np.float32).tolist())
        coverage[:] = 1.0
    else:
        for row, (row_cols, row_values, row_coverage) in enumerate(regular_overlap_rows(source, target)):
            coverage[row] = row_coverage
            if row_cols.size == 0:
                raise ValueError(f"target {int(target.target_ids[row])} has no source-grid overlap")
            if row_coverage < _MIN_FULL_COVERAGE:
                raise ValueError(
                    f"target {int(target.target_ids[row])} coverage {row_coverage:.4f} < {_MIN_FULL_COVERAGE:.4f}"
                )
            if normalization == "mean":
                row_values = normalise_row(row_values)
            rows.extend([row] * row_cols.size)
            cols.extend(row_cols.tolist())
            values.extend(row_values.tolist())

    matrix = csr_matrix(
        (np.asarray(values, dtype=np.float32), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(target.target_ids.size, source.size),
        dtype=np.float32,
    )
    matrix.eliminate_zeros()
    out_metadata = {
        "method": method_name,
        "normalization": normalization,
        "source_shape": list(source.shape),
        "source_order": source.order,
        "source_x_name": source.x_name,
        "source_y_name": source.y_name,
        "target_kind": target.metadata.get("kind", "unknown"),
        "overlap_engine": "separable" if method_name == "overlap" else None,
    }
    if metadata:
        out_metadata.update(dict(metadata))
    return MappingTable(target.target_ids, matrix, source.x, source.y, coverage=coverage, metadata=out_metadata)


def build_hires_aggregate_mapping(
    source: RegularGrid,
    target_ids: np.ndarray,
    pixel_catchment_id: np.ndarray,
    pixel_area: np.ndarray,
    pixel_lon: np.ndarray,
    pixel_lat: np.ndarray,
    *,
    allow_oob_zero: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> MappingTable:
    """Build an area-weighted catchment x source mapping from hires pixels.

    Weights are raw pixel areas (no per-row normalization), matching the legacy
    CaMa ``generate_mapping_table`` semantics.  ``target_ids`` defines both the
    row order and the catchment subset to keep.
    """
    target_ids = np.asarray(target_ids, dtype=np.int64)
    rows, cols, data = aggregate_hires_coo(
        source,
        target_ids,
        pixel_catchment_id,
        pixel_area,
        pixel_lon,
        pixel_lat,
        allow_oob_zero=allow_oob_zero,
    )
    matrix = csr_matrix(
        (data.astype(np.float32), (rows, cols)),
        shape=(target_ids.size, source.size),
        dtype=np.float32,
    )
    matrix.eliminate_zeros()
    out_metadata = {
        "method": "hires_aggregate",
        "normalization": "sum",
        "source_shape": list(source.shape),
        "source_order": source.order,
        "source_x_name": source.x_name,
        "source_y_name": source.y_name,
        "target_kind": "catchment",
        "overlap_engine": "hires_aggregate",
    }
    if metadata:
        out_metadata.update(dict(metadata))
    return MappingTable(target_ids, matrix, source.x, source.y, metadata=out_metadata)

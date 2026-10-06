# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Orchestrators that assemble :class:`MappingTable` objects from engines."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import AfterValidator, InstanceOf, validate_call
from scipy.sparse import csr_matrix

from hydroforge.core.validation import HydroForgeModel
from hydroforge.mapping.cama import _cama_cell_targets, _cama_hires_source_cells
from hydroforge.mapping.engine import (
    _EARTH_RADIUS_M,
    _hires_coo,
    _normalise_row,
    _normalise_rows,
    _raise_hires_oob_hint,
    _regular_overlap_csr,
)
from hydroforge.mapping.grid import RegularGrid
from hydroforge.mapping.table import MappingTable, _canonical_metadata_value
from hydroforge.mapping.target import TargetSupport

MappingMethod = Literal["nearest", "overlap"]
Normalization = Literal["mean", "sum"]
_MIN_FULL_COVERAGE = 1.0 - 1e-6
_MAPPING_METADATA_KEYS = frozenset(
    {
        "method",
        "normalization",
        "source_shape",
        "source_order",
        "source_is_geographic",
        "source_periodic_x",
        "source_x_name",
        "source_y_name",
        "target_kind",
        "overlap_engine",
    }
)


def _mapping_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if any(type(name) is not str or not name for name in value):
        raise ValueError("mapping metadata keys must be non-empty exact strings")
    reserved = sorted(set(value).intersection({*_MAPPING_METADATA_KEYS, "schema"}))
    if reserved:
        raise ValueError(f"mapping metadata cannot override derived keys: {reserved}")
    return _canonical_metadata_value(dict(value), path="mapping metadata")


_MappingMetadata = Annotated[
    Mapping[str, Any] | None, AfterValidator(_mapping_metadata)
]


def _source_metadata(source: RegularGrid) -> dict[str, Any]:
    return {
        "source_shape": list(source._shape),
        "source_order": source.order,
        "source_is_geographic": source.is_geographic,
        "source_periodic_x": source._periodic_x,
        "source_x_name": source.x_name,
        "source_y_name": source.y_name,
    }


@validate_call(config=HydroForgeModel.model_config)
def build_regular_grid_mapping(
    source: InstanceOf[RegularGrid],
    target: InstanceOf[TargetSupport],
    *,
    method: MappingMethod = "overlap",
    normalization: Normalization = "mean",
    metadata: _MappingMetadata = None,
) -> MappingTable:
    """Build a sparse ``target x source`` mapping table.

    ``overlap`` uses the spherical (geographic) or planar overlap area of each
    target's cell bounds and requires full coverage; ``nearest`` assigns each
    target center to the source cell containing it.
    """

    if method == "overlap" and target.bounds is None:
        raise ValueError("overlap requires target bounds")
    if method == "nearest" and (target.x is None or target.y is None):
        raise ValueError("nearest requires target center coordinates")
    n_target = target.target_ids.size
    coverage = np.zeros(n_target, dtype=np.float32)

    if method == "nearest":
        source_idx = source._index_of_points(target.x, target.y)
        indptr = np.arange(source_idx.size + 1, dtype=np.int64)
        cols = source_idx.astype(np.int64, copy=False)
        values = np.ones(source_idx.size, dtype=np.float64)
        coverage[:] = 1.0
    else:
        overlap = _regular_overlap_csr(source, target)
        indptr = overlap.indptr
        cols = overlap.cols
        values = overlap.values
        lengths = np.diff(indptr)
        rejected = np.zeros(n_target, dtype=bool)
        if normalization == "mean":
            values, rejected = _normalise_rows(overlap)
        failed = np.flatnonzero(
            (lengths == 0) | (overlap.coverage < _MIN_FULL_COVERAGE) | rejected
        )
        if failed.size:
            row = int(failed[0])
            row_coverage = float(overlap.coverage[row])
            if lengths[row] == 0:
                raise ValueError(
                    f"target {int(target.target_ids[row])} has no source-grid overlap"
                )
            if row_coverage < _MIN_FULL_COVERAGE:
                raise ValueError(
                    f"target {int(target.target_ids[row])} coverage "
                    f"{row_coverage:.4f} < {_MIN_FULL_COVERAGE:.4f}"
                )
            _normalise_row(overlap.values[indptr[row] : indptr[row + 1]])
        coverage[:] = overlap.coverage

    out_metadata = dict(metadata)
    out_metadata.update(
        {
            "method": method,
            "normalization": normalization,
            **_source_metadata(source),
            "target_kind": target.metadata.get("kind", "unknown"),
            "overlap_engine": "separable" if method == "overlap" else None,
        }
    )
    return MappingTable._assemble(
        target_ids=target.target_ids,
        matrix=csr_matrix(
            (values, cols, indptr),
            shape=(n_target, source._size),
            dtype=np.float64,
        ),
        source_x=source.x,
        source_y=source.y,
        coverage=coverage,
        metadata=out_metadata,
    )


def _hires_mapping(
    source: RegularGrid,
    target_ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    coverage: np.ndarray,
    metadata: Mapping[str, Any],
) -> MappingTable:
    """Assemble summed hires pixel areas as a ``catchment x source`` table.

    ``coverage`` is each catchment's area fraction that lies on the source
    grid, like the covered-area fraction of overlap mappings.
    """

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(target_ids.size, source._size),
        dtype=np.float64,
    )
    return MappingTable._assemble(
        target_ids=target_ids,
        matrix=matrix,
        source_x=source.x,
        source_y=source.y,
        coverage=coverage,
        metadata={
            **metadata,
            "method": "hires_aggregate",
            "normalization": "sum",
            **_source_metadata(source),
            "target_kind": "catchment",
            "overlap_engine": "hires_aggregate",
        },
    )


def _covered_fraction(
    rows: np.ndarray, areas: np.ndarray, covered: np.ndarray, count: int
) -> np.ndarray:
    """Per-row fraction of ``areas`` flagged ``covered``; zero for empty rows."""

    total = np.bincount(rows, weights=areas, minlength=count)
    inside = np.bincount(rows[covered], weights=areas[covered], minlength=count)
    return np.divide(inside, total, out=np.zeros(count), where=total > 0.0)


def _cama_cell_mapping(
    source: RegularGrid,
    target_ids: np.ndarray,
    map_dir: Path,
    nx: int,
    ny: int,
    *,
    map_precision: str,
    allow_oob_zero: bool,
    metadata: Mapping[str, Any],
) -> MappingTable:
    """Spread each CaMa catchment area over the source cells its cell overlaps.

    Without a high-resolution map a catchment is known only through its
    low-resolution cell. Its area (``ctmare.bin`` or the spherical cell
    area) is distributed over the separable spherical overlap of that cell
    with the source grid, so a finer source grid is area-averaged instead
    of sampled at the cell center.
    """

    bounds, areas = _cama_cell_targets(
        map_dir, nx, ny, target_ids, map_precision=map_precision
    )
    overlap = _regular_overlap_csr(
        source, TargetSupport(target_ids=target_ids, bounds=bounds)
    )
    coverage = overlap.coverage
    uncovered = int(np.count_nonzero(coverage < _MIN_FULL_COVERAGE))
    if uncovered and not allow_oob_zero:
        raise ValueError(
            f"{uncovered}/{target_ids.size} points fall outside the source grid "
            "(CaMa cells not fully covered by the source grid)"
        )
    if source.is_geographic:
        cell_area = (
            np.radians(bounds[:, 1] - bounds[:, 0])
            * _EARTH_RADIUS_M
            * _EARTH_RADIUS_M
            * (np.sin(np.radians(bounds[:, 3])) - np.sin(np.radians(bounds[:, 2])))
        )
    else:
        cell_area = (bounds[:, 1] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 2])
    lengths = np.diff(overlap.indptr)
    # Pixels outside the source grid are dropped, as on the hires path.
    values = overlap.values * np.repeat(areas / cell_area, lengths)
    return MappingTable._assemble(
        target_ids=target_ids,
        matrix=csr_matrix(
            (values, overlap.cols, overlap.indptr),
            shape=(target_ids.size, source._size),
            dtype=np.float64,
        ),
        source_x=source.x,
        source_y=source.y,
        coverage=np.minimum(coverage, 1.0),
        metadata={
            **metadata,
            "method": "cama_cell_overlap",
            "normalization": "sum",
            **_source_metadata(source),
            "target_kind": "catchment",
            "overlap_engine": "separable",
        },
    )


def _cama_hires_mapping(
    source: RegularGrid,
    target_ids: np.ndarray,
    map_dir: Path,
    nx: int,
    ny: int,
    nextxy_data: np.ndarray,
    *,
    hires_tag: str | None,
    mapinfo_txt: str,
    hires_idx_precision: str,
    map_precision: str,
    allow_oob_zero: bool,
    metadata: Mapping[str, Any],
) -> MappingTable:
    """Aggregate the hires pixels of a CaMa map directory onto ``source``.

    Each hires tile's longitude and latitude axes are located on ``source``
    once instead of once per pixel.  With ``hires_tag=None`` the
    low-resolution cells are overlapped instead (:func:`_cama_cell_mapping`).
    Inputs are validated by
    :func:`~hydroforge.mapping.aggregation.build_cama_mapping`.
    """

    try:
        if hires_tag is None:
            return _cama_cell_mapping(
                source,
                target_ids,
                map_dir,
                nx,
                ny,
                map_precision=map_precision,
                allow_oob_zero=allow_oob_zero,
                metadata=metadata,
            )
        catchment_idx, pixel_area, source_idx = _cama_hires_source_cells(
            map_dir,
            nx,
            ny,
            nextxy_data,
            source,
            allow_oob=allow_oob_zero,
            target_ids=target_ids,
            hires_tag=hires_tag,
            mapinfo_txt=mapinfo_txt,
            hires_idx_precision=hires_idx_precision,
            map_precision=map_precision,
        )
    except ValueError as exc:
        _raise_hires_oob_hint(exc, allow_oob_zero=allow_oob_zero)
        raise
    rows, cols, data = _hires_coo(catchment_idx, source_idx, pixel_area)
    coverage = _covered_fraction(
        catchment_idx, pixel_area, source_idx >= 0, target_ids.size
    )
    return _hires_mapping(source, target_ids, rows, cols, data, coverage, metadata)

"""Orchestrators that assemble :class:`MappingTable` objects from engines."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import AfterValidator, InstanceOf, validate_call
from scipy.sparse import csr_matrix

from hydroforge.core.arrays import find_indices_in
from hydroforge.core.validation import HydroForgeModel
from hydroforge.mapping.cama import _cama_hires_source_cells
from hydroforge.mapping.engine import (
    _aggregate_hires_coo,
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
    if method == "overlap" and source.is_geographic:
        bounds = target.bounds
        if np.any((bounds[:, 2] < -90.0) | (bounds[:, 3] > 90.0)):
            raise ValueError(
                "geographic target latitude bounds must lie within [-90, 90]"
            )
        if source._periodic_x and np.any(bounds[:, 1] - bounds[:, 0] > 360.0 + 1e-9):
            raise ValueError(
                "geographic target longitude width cannot exceed 360 degrees"
            )
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
    metadata: Mapping[str, Any],
) -> MappingTable:
    """Assemble summed hires pixel areas as a ``catchment x source`` table."""

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
        coverage=np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel(),
        metadata={
            **metadata,
            "method": "hires_aggregate",
            "normalization": "sum",
            **_source_metadata(source),
            "target_kind": "catchment",
            "overlap_engine": "hires_aggregate",
        },
    )


def _build_hires_aggregate_mapping(
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
    """Area-weighted ``catchment x source`` mapping from explicit hires pixels.

    The per-pixel reference for :func:`_cama_hires_mapping`; inputs are the
    canonical arrays of :func:`~hydroforge.mapping.engine._aggregate_hires_coo`.
    Weights are raw pixel areas without per-row normalization.
    """

    rows, cols, data = _aggregate_hires_coo(
        source,
        target_ids,
        pixel_catchment_id,
        pixel_area,
        pixel_lon,
        pixel_lat,
        allow_oob_zero=allow_oob_zero,
    )
    return _hires_mapping(source, target_ids, rows, cols, data, metadata or {})


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

    Equal to :func:`_build_hires_aggregate_mapping` of the same pixels, but
    each hires tile's longitude and latitude axes are located on ``source``
    once instead of once per pixel.  Inputs are validated by
    :func:`~hydroforge.mapping.aggregation.build_cama_mapping`.
    """

    try:
        pixel_catchment_id, pixel_area, source_idx = _cama_hires_source_cells(
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
    catchment_idx = find_indices_in(pixel_catchment_id, target_ids)
    rows, cols, data = _hires_coo(catchment_idx, source_idx, pixel_area)
    return _hires_mapping(source, target_ids, rows, cols, data, metadata)

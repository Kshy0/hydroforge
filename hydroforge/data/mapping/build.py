"""Orchestrators that assemble :class:`MappingTable` objects from engines."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import AfterValidator, Field, PrivateAttr, model_validator
from scipy.sparse import csr_matrix

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.distributed import _find_indices_in_trusted
from hydroforge.data.mapping.cama import _read_cama_hires_source_cells
from hydroforge.data.mapping.engine import (
    _aggregate_hires_coo_trusted,
    _hires_coo_trusted,
    _HiresPixelDeclaration,
    _normalise_rows_trusted,
    _raise_hires_oob_hint,
    _regular_overlap_csr_trusted,
    normalise_row,
)
from hydroforge.data.mapping.grid import RegularGrid, _has_duplicates
from hydroforge.data.mapping.table import MappingTable
from hydroforge.data.mapping.target import TargetSupport
from hydroforge.data.numeric import canonical_floating_array, canonical_ids

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


def _float32_mapping_matrix(matrix: csr_matrix, *, label: str) -> csr_matrix:
    data = canonical_floating_array(
        matrix.data,
        dtype="float32",
        label=label,
    )
    converted = csr_matrix(
        (data, matrix.indices.copy(), matrix.indptr.copy()),
        shape=matrix.shape,
        dtype=np.float32,
    )
    converted.eliminate_zeros()
    converted.sort_indices()
    return converted


def _mapping_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if any(type(name) is not str or not name for name in value):
        raise ValueError("mapping metadata keys must be non-empty exact strings")
    reserved = sorted(set(value).intersection(_MAPPING_METADATA_KEYS))
    if reserved:
        raise ValueError(f"mapping metadata cannot override derived keys: {reserved}")
    return deepcopy(dict(value))


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


class _RegularGridMappingDeclaration(HydroForgeModel):
    """Validated public declaration for one regular-grid mapping build."""

    source: RegularGrid
    target: TargetSupport
    method: MappingMethod = "overlap"
    normalization: Normalization = "mean"
    metadata: _MappingMetadata = None

    _mapping: MappingTable = PrivateAttr()

    @model_validator(mode="after")
    def _validate_mapping(self) -> Self:
        if self.method == "overlap" and self.target.bounds is None:
            raise ValueError("overlap requires target bounds")
        if self.method == "nearest" and (
            self.target.x is None or self.target.y is None
        ):
            raise ValueError("nearest requires target center coordinates")
        self._mapping = _build_regular_grid_mapping_trusted(
            source=self.source,
            target=self.target,
            method=self.method,
            normalization=self.normalization,
            metadata=self.metadata,
        )
        return self

    @property
    def mapping(self) -> MappingTable:
        return self._mapping


def _build_regular_grid_mapping_trusted(
    *,
    source: RegularGrid,
    target: TargetSupport,
    method: MappingMethod,
    normalization: Normalization,
    metadata: Mapping[str, Any] | None,
) -> MappingTable:
    """Materialize a mapping from already validated immutable inputs."""

    n_target = target.target_ids.size
    coverage = np.zeros(n_target, dtype=np.float32)

    if method == "nearest":
        source_idx = source._index_of_points_trusted(
            target.x,
            target.y,
            allow_oob=False,
        )
        indptr = np.arange(source_idx.size + 1, dtype=np.int64)
        cols = source_idx.astype(np.int64, copy=False)
        values = np.ones(source_idx.size, dtype=np.float64)
        coverage[:] = 1.0
    else:
        overlap = _regular_overlap_csr_trusted(source, target)
        indptr = overlap.indptr
        cols = overlap.cols
        values = overlap.values
        lengths = np.diff(indptr)
        rejected = np.zeros(n_target, dtype=bool)
        if normalization == "mean":
            values, rejected = _normalise_rows_trusted(overlap)
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
            normalise_row(overlap.values[indptr[row] : indptr[row + 1]])
        coverage[:] = overlap.coverage

    matrix = csr_matrix(
        (values, cols, indptr),
        shape=(n_target, source._size),
        dtype=np.float64,
    )
    matrix = _float32_mapping_matrix(
        matrix,
        label="regular-grid mapping weights",
    )
    out_metadata = dict(metadata or {})
    out_metadata.update(
        {
            "method": method,
            "normalization": normalization,
            **_source_metadata(source),
            "target_kind": target.metadata.get("kind", "unknown"),
            "overlap_engine": "separable" if method == "overlap" else None,
        }
    )
    return MappingTable(
        target_ids=target.target_ids,
        matrix=matrix,
        source_x=source.x,
        source_y=source.y,
        coverage=coverage,
        metadata=out_metadata,
    )


def build_regular_grid_mapping(
    source: RegularGrid,
    target: TargetSupport,
    *,
    method: MappingMethod = "overlap",
    normalization: Normalization = "mean",
    metadata: Mapping[str, Any] | None = None,
) -> MappingTable:
    """Build a sparse ``target x source`` mapping table."""
    declaration = _RegularGridMappingDeclaration(
        source=source,
        target=target,
        method=method,
        normalization=normalization,
        metadata=metadata,
    )
    return declaration.mapping


class _HiresAggregateMappingDeclaration(_HiresPixelDeclaration):
    """Validated public declaration for a high-resolution aggregate build."""

    metadata: _MappingMetadata = None


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

    Weights are raw pixel areas without per-row normalization. ``target_ids``
    defines the row order and catchment subset.
    """
    declaration = _HiresAggregateMappingDeclaration(
        source=source,
        target_ids=target_ids,
        pixel_catchment_id=pixel_catchment_id,
        pixel_area=pixel_area,
        pixel_lon=pixel_lon,
        pixel_lat=pixel_lat,
        allow_oob_zero=allow_oob_zero,
        metadata=metadata,
    )
    rows, cols, data = _aggregate_hires_coo_trusted(declaration)
    return _hires_mapping_trusted(
        declaration.source,
        declaration.target_ids,
        rows,
        cols,
        data,
        declaration.metadata,
    )


class _CamaHiresMappingDeclaration(HydroForgeModel):
    """Validated declaration for a CaMa hires aggregate mapping build."""

    source: RegularGrid
    target_ids: np.ndarray
    map_dir: Path = Field(strict=False)
    nx: int = Field(ge=1)
    ny: int = Field(ge=1)
    nextxy_data: np.ndarray
    hires_tag: str | None = "1min"
    mapinfo_txt: str = "location.txt"
    hires_idx_precision: str = "<i2"
    map_precision: str = "<f4"
    allow_oob_zero: bool = False
    metadata: _MappingMetadata = None

    @model_validator(mode="after")
    def _validate_targets(self) -> Self:
        target_ids = canonical_ids(self.target_ids, label="target_ids")
        if _has_duplicates(target_ids):
            raise ValueError("target_ids must be unique")
        object.__setattr__(self, "target_ids", target_ids)
        return self


def build_cama_hires_aggregate_mapping(
    source: RegularGrid,
    target_ids: np.ndarray,
    map_dir: str | Path,
    nx: int,
    ny: int,
    nextxy_data: np.ndarray,
    *,
    hires_tag: str | None = "1min",
    mapinfo_txt: str = "location.txt",
    hires_idx_precision: str = "<i2",
    map_precision: str = "<f4",
    allow_oob_zero: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> MappingTable:
    """Build the :func:`build_hires_aggregate_mapping` table straight from a
    CaMa map directory.

    The result is identical to reading :func:`read_cama_hires_pixels` and
    aggregating its per-pixel coordinates, but each hires tile's longitude and
    latitude axes are located on ``source`` once instead of once per pixel.
    """
    declaration = _CamaHiresMappingDeclaration(
        source=source,
        target_ids=target_ids,
        map_dir=map_dir,
        nx=nx,
        ny=ny,
        nextxy_data=nextxy_data,
        hires_tag=hires_tag,
        mapinfo_txt=mapinfo_txt,
        hires_idx_precision=hires_idx_precision,
        map_precision=map_precision,
        allow_oob_zero=allow_oob_zero,
        metadata=metadata,
    )
    try:
        pixel_catchment_id, pixel_area, source_idx = _read_cama_hires_source_cells(
            declaration.map_dir,
            declaration.nx,
            declaration.ny,
            declaration.nextxy_data,
            declaration.source,
            allow_oob=declaration.allow_oob_zero,
            hires_tag=declaration.hires_tag,
            mapinfo_txt=declaration.mapinfo_txt,
            hires_idx_precision=declaration.hires_idx_precision,
            map_precision=declaration.map_precision,
        )
    except ValueError as exc:
        _raise_hires_oob_hint(exc, allow_oob_zero=declaration.allow_oob_zero)
        raise
    catchment_idx = _find_indices_in_trusted(pixel_catchment_id, declaration.target_ids)
    rows, cols, data = _hires_coo_trusted(catchment_idx, source_idx, pixel_area)
    return _hires_mapping_trusted(
        declaration.source,
        declaration.target_ids,
        rows,
        cols,
        data,
        declaration.metadata,
    )


def _hires_mapping_trusted(
    source: RegularGrid,
    target_ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    metadata: Mapping[str, Any] | None,
) -> MappingTable:
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(target_ids.size, source._size),
        dtype=np.float64,
    )
    coverage = canonical_floating_array(
        np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel(),
        dtype="float32",
        label="hires mapping coverage",
    )
    matrix = _float32_mapping_matrix(
        matrix,
        label="hires mapping weights",
    )
    out_metadata = dict(metadata or {})
    out_metadata.update(
        {
            "method": "hires_aggregate",
            "normalization": "sum",
            **_source_metadata(source),
            "target_kind": "catchment",
            "overlap_engine": "hires_aggregate",
        }
    )
    return MappingTable(
        target_ids=target_ids,
        matrix=matrix,
        source_x=source.x,
        source_y=source.y,
        coverage=coverage,
        metadata=out_metadata,
    )

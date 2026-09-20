"""Orchestrators that assemble :class:`MappingTable` objects from engines."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import AfterValidator, PrivateAttr, model_validator
from scipy.sparse import csr_matrix

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.mapping.engine import (
    _aggregate_hires_coo_trusted,
    _HiresPixelDeclaration,
    normalise_row,
    regular_overlap_rows,
)
from hydroforge.data.mapping.grid import RegularGrid
from hydroforge.data.mapping.table import MappingTable
from hydroforge.data.mapping.target import TargetSupport
from hydroforge.data.numeric import canonical_floating_array

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

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    coverage = np.zeros(target.target_ids.size, dtype=np.float32)

    if method == "nearest":
        source_idx = source._index_of_points_trusted(
            target.x,
            target.y,
            allow_oob=False,
        )
        target_rows = np.arange(source_idx.size, dtype=np.int64)
        rows.extend(target_rows.tolist())
        cols.extend(source_idx.tolist())
        values.extend(np.ones(target_rows.size, dtype=np.float32).tolist())
        coverage[:] = 1.0
    else:
        for row, (
            row_cols,
            row_values,
            row_coverage,
        ) in enumerate(regular_overlap_rows(source, target)):
            coverage[row] = row_coverage
            if row_cols.size == 0:
                raise ValueError(
                    f"target {int(target.target_ids[row])} has no source-grid overlap"
                )
            if row_coverage < _MIN_FULL_COVERAGE:
                raise ValueError(
                    f"target {int(target.target_ids[row])} coverage "
                    f"{row_coverage:.4f} < {_MIN_FULL_COVERAGE:.4f}"
                )
            if normalization == "mean":
                row_values = normalise_row(row_values)
            rows.extend([row] * row_cols.size)
            cols.extend(row_cols.tolist())
            values.extend(row_values.tolist())

    matrix = csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=(target.target_ids.size, source._size),
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
    source = declaration.source
    target_ids = declaration.target_ids
    rows, cols, data = _aggregate_hires_coo_trusted(declaration)
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
    out_metadata = dict(declaration.metadata or {})
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

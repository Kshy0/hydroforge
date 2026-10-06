# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Sparse mapping table from flattened source grid cells to target supports."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import torch
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
    validate_call,
)
from scipy.sparse import csr_matrix

from hydroforge.core.arrays import (
    UniqueIds,
    canonical_floating_array,
    canonical_ids,
    find_indices_in,
    immutable_array,
    immutable_metadata,
)
from hydroforge.core.validation import HydroForgeModel
from hydroforge.io.files import atomic_output_path
from hydroforge.mapping.engine import _segment_sums
from hydroforge.mapping.grid import (
    _axis_bounds,
    _bounds_are_periodic,
    _has_duplicates,
)

_MAPPING_SCHEMA = "hydroforge.spatial_mapping.v2"
_MAPPING_ARCHIVE_KEYS = frozenset(
    {
        "target_ids",
        "sparse_data",
        "sparse_indices",
        "sparse_indptr",
        "matrix_shape",
        "coord_lon",
        "coord_lat",
        "coverage",
        "metadata_json",
    }
)
_FilePath = Annotated[Path, Field(strict=False)]


def normalize_target_weights(
    targets: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> np.ndarray:
    """Scale float64 ``weights`` so the weights of each target sum to one.

    ``targets`` gives each weight's target in ``range(count)``; targets
    without a positive total keep zero weights.  This is the only mapping
    normalization rule, shared by row-normalized tables and exports.
    """

    totals = np.bincount(targets, weights=weights, minlength=count)[targets]
    return np.divide(
        weights,
        totals,
        out=np.zeros_like(weights, dtype=np.float64),
        where=totals > 0.0,
    )


def _apply_input(value: Any) -> np.ndarray:
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError("mapping input contains missing values")
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("mapping input must have at least one dimension")
    return canonical_floating_array(
        array, dtype="float64", label="mapping input", allow_nan=True
    )


def _boolean_mask(value: Any) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError("valid_source_mask must not be a masked array")
    if value.dtype != np.dtype(np.bool_):
        raise ValueError("valid_source_mask must use exact boolean dtype")
    return value


def _validate_csr_components(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
) -> None:
    for name, component in (
        ("sparse_data", data),
        ("sparse_indices", indices),
        ("sparse_indptr", indptr),
    ):
        if not isinstance(component, np.ndarray) or component.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if np.ma.isMaskedArray(component) and np.any(np.ma.getmaskarray(component)):
            raise ValueError(f"{name} contains missing values")
    for name, component in (("sparse_indices", indices), ("sparse_indptr", indptr)):
        if component.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
            raise ValueError(f"{name} must use signed int32 or int64 storage")
    if indices.size != data.size:
        raise ValueError("sparse_indices must match sparse_data size")
    if indptr.shape != (shape[0] + 1,):
        raise ValueError(f"sparse_indptr must have shape ({shape[0] + 1},)")
    if indptr[0] != 0 or indptr[-1] != data.size or np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("sparse_indptr is not a valid CSR row pointer")
    if indices.size and (indices.min() < 0 or indices.max() >= shape[1]):
        raise ValueError("sparse_indices fall outside matrix_shape")


class _ImmutableCSR(csr_matrix):
    """CSR read view whose structural storage cannot be replaced in place."""

    _IMMUTABLE_ATTRIBUTES = frozenset({"data", "indices", "indptr", "_shape"})

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_hydroforge_sealed", False) and (
            name in self._IMMUTABLE_ATTRIBUTES
        ):
            raise TypeError("frozen mapping CSR storage is immutable")
        super().__setattr__(name, value)

    def _reject_mutation(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if getattr(self, "_hydroforge_sealed", False):
            raise TypeError("frozen mapping CSR storage is immutable")

    def __setitem__(self, key: Any, value: Any) -> None:
        self._reject_mutation()
        super().__setitem__(key, value)

    def eliminate_zeros(self) -> None:
        self._reject_mutation()
        super().eliminate_zeros()

    def prune(self) -> None:
        self._reject_mutation()
        super().prune()

    def resize(self, *shape: Any) -> None:
        self._reject_mutation()
        super().resize(*shape)

    def setdiag(self, values: Any, k: int = 0) -> None:
        self._reject_mutation()
        super().setdiag(values, k=k)

    def sort_indices(self) -> None:
        # SciPy reductions call these unconditionally; canonical storage
        # needs no change, so only a real mutation is rejected.
        if self.has_sorted_indices:
            return
        self._reject_mutation()
        super().sort_indices()

    def sum_duplicates(self) -> None:
        if self.has_canonical_format:
            return
        self._reject_mutation()
        super().sum_duplicates()


def _freeze_csr_storage(matrix: csr_matrix) -> csr_matrix:
    """Replace CSR component arrays with immutable-buffer-backed arrays."""

    # The new CSR shell may share inputs until each component is replaced by
    # immutable_array's independently owned byte storage below.
    frozen = _ImmutableCSR(matrix, copy=False)
    frozen.data = immutable_array(frozen.data, order="C")
    frozen.indices = immutable_array(frozen.indices, order="C")
    frozen.indptr = immutable_array(frozen.indptr, order="C")
    frozen._hydroforge_sealed = True
    return frozen


def _canonical_metadata_value(value: Any, *, path: str) -> Any:
    """Canonicalize one strict JSON metadata value without coercion."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if type(value) in {list, tuple}:
        return tuple(
            _canonical_metadata_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if type(value) is dict:
        if any(type(name) is not str or not name for name in value):
            raise ValueError(f"{path} keys must be non-empty exact strings")
        return {
            name: _canonical_metadata_value(
                item,
                path=f"{path}.{name}",
            )
            for name, item in value.items()
        }
    raise ValueError(
        f"{path} must contain only exact JSON scalar, list, or object values"
    )


class LocalMapping(HydroForgeModel):
    """Target-selected mapping with unused source columns removed."""

    target_ids: np.ndarray
    source_indices: np.ndarray
    source_to_target: csr_matrix

    # ``target x active_source`` CSR derived once for :meth:`to_torch`.
    _target_rows: csr_matrix | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_local_mapping(self) -> Self:
        target_ids = canonical_ids(self.target_ids, label="target_ids")
        source_indices = canonical_ids(
            self.source_indices,
            label="source_indices",
        )
        if _has_duplicates(target_ids):
            raise ValueError("target_ids must be unique")
        if _has_duplicates(source_indices):
            raise ValueError("source_indices must be unique")
        if source_indices.size and np.any(source_indices < 0):
            raise ValueError("source_indices must be nonnegative")
        _validate_csr_components(
            self.source_to_target.data,
            self.source_to_target.indices,
            self.source_to_target.indptr,
            self.source_to_target.shape,
        )
        matrix = self.source_to_target.copy()
        if matrix.dtype != np.dtype(np.float32):
            raise ValueError("source_to_target must use float32 storage")
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.shape != (
            source_indices.size,
            target_ids.size,
        ):
            raise ValueError(
                "source_to_target shape must match source_indices and target_ids"
            )
        if not np.isfinite(matrix.data).all():
            raise ValueError("source_to_target weights must be finite")
        if np.any(matrix.data < 0):
            raise ValueError("source_to_target weights must be nonnegative")
        object.__setattr__(
            self,
            "target_ids",
            immutable_array(target_ids, order="C"),
        )
        object.__setattr__(
            self,
            "source_indices",
            immutable_array(source_indices, order="C"),
        )
        object.__setattr__(
            self,
            "source_to_target",
            _freeze_csr_storage(matrix),
        )
        return self

    @classmethod
    def _assemble(
        cls,
        *,
        target_ids: np.ndarray,
        source_indices: np.ndarray,
        source_to_target: csr_matrix,
    ) -> LocalMapping:
        """Own a local projection derived solely from a validated mapping."""

        matrix = source_to_target.astype(np.float32, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return cls.model_construct(
            target_ids=immutable_array(target_ids, dtype=np.int64, order="C"),
            source_indices=immutable_array(
                source_indices,
                dtype=np.int64,
                order="C",
            ),
            source_to_target=_freeze_csr_storage(matrix),
        )

    def to_torch(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        layout: torch.layout,
    ) -> torch.Tensor:
        """Materialize the ``target x active_source`` matrix as a sparse tensor.

        ``layout`` is ``torch.sparse_csr`` or ``torch.sparse_coo``; COO
        tensors are coalesced.  Column indices ascend within each target row.
        """

        if dtype not in {torch.float32, torch.float64}:
            raise TypeError("sparse mapping dtype must be float32 or float64")
        if layout not in {torch.sparse_csr, torch.sparse_coo}:
            raise ValueError(f"unsupported sparse mapping layout {layout}")
        matrix = self._target_rows
        if matrix is None:
            matrix = self.source_to_target.T.tocsr()
            matrix.sort_indices()
            self._target_rows = matrix
        data = canonical_floating_array(
            matrix.data,
            dtype="float32" if dtype == torch.float32 else "float64",
            label="sparse mapping weights",
        )
        values = torch.tensor(data, device=device, dtype=dtype)
        if layout == torch.sparse_csr:
            with warnings.catch_warnings():
                # PyTorch labels every sparse CSR tensor as beta.
                warnings.filterwarnings(
                    "ignore",
                    message=r"Sparse CSR tensor support is in beta state",
                    category=UserWarning,
                )
                return torch.sparse_csr_tensor(
                    torch.tensor(matrix.indptr, device=device, dtype=torch.int64),
                    torch.tensor(matrix.indices, device=device, dtype=torch.int64),
                    values,
                    size=matrix.shape,
                    device=device,
                    dtype=dtype,
                    check_invariants=True,
                )
        matrix = matrix.tocoo()
        indices = torch.tensor(
            np.stack((matrix.row, matrix.col)),
            device=device,
            dtype=torch.int64,
        )
        with torch.sparse.check_sparse_tensor_invariants(enable=True):
            tensor = torch.sparse_coo_tensor(
                indices,
                values,
                size=matrix.shape,
                device=device,
                dtype=dtype,
            )
        return tensor.coalesce()


class MappingTable(HydroForgeModel):
    """CSR mapping from flattened source grid cells to target supports."""

    target_ids: np.ndarray
    matrix: csr_matrix
    source_x: np.ndarray
    source_y: np.ndarray
    coverage: np.ndarray
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if type(value) is not dict:
            raise ValueError("mapping metadata must be an exact dict")
        if "schema" in value:
            raise ValueError("mapping metadata key 'schema' is reserved")
        canonical = _canonical_metadata_value(value, path="mapping metadata")
        return immutable_metadata(canonical, label="mapping metadata")

    @field_serializer("metadata")
    def _serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return deepcopy(dict(value))

    @model_validator(mode="after")
    def _validate_mapping(self) -> Self:
        for name in ("target_ids", "source_x", "source_y", "coverage"):
            value = getattr(self, name)
            if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
                raise ValueError(f"mapping {name} contains missing values")
        if self.target_ids.ndim != 1:
            raise ValueError("mapping target_ids must be one-dimensional")
        if self.target_ids.dtype != np.dtype(np.int64):
            raise ValueError("mapping target_ids must use exact int64 dtype")
        if _has_duplicates(self.target_ids):
            raise ValueError("mapping target_ids must be unique")
        _validate_csr_components(
            self.matrix.data,
            self.matrix.indices,
            self.matrix.indptr,
            self.matrix.shape,
        )
        if self.matrix.dtype != np.dtype(np.float32):
            raise ValueError("mapping matrix must use exact float32 dtype")
        # A fresh shell recomputes the canonical-format flag without copying
        # storage; freezing below takes the only owned copy.
        matrix = csr_matrix(self.matrix, copy=False)
        if not matrix.has_canonical_format:
            raise ValueError("mapping matrix must use canonical CSR storage")
        if not np.isfinite(self.matrix.data).all():
            raise ValueError("mapping matrix values must be finite")
        if np.any(self.matrix.data < 0):
            raise ValueError("mapping matrix values must be nonnegative")
        if self.source_x.ndim != 1 or self.source_y.ndim != 1:
            raise ValueError("mapping source coordinates must be one-dimensional")
        if self.source_x.dtype != np.dtype(
            np.float64
        ) or self.source_y.dtype != np.dtype(np.float64):
            raise ValueError("mapping source coordinates must use exact float64 dtype")
        if self.source_x.size == 0 or self.source_y.size == 0:
            raise ValueError("mapping source coordinates must be non-empty")
        if (
            not np.isfinite(self.source_x).all()
            or not np.isfinite(
                self.source_y,
            ).all()
        ):
            raise ValueError("mapping source coordinates must be finite")
        if _has_duplicates(self.source_x) or _has_duplicates(self.source_y):
            raise ValueError("mapping source coordinates must be unique")
        expected_shape = (
            self.target_ids.size,
            self.source_x.size * self.source_y.size,
        )
        if self.matrix.shape != expected_shape:
            raise ValueError(
                f"matrix shape {self.matrix.shape} is inconsistent with "
                f"{self.target_ids.size} targets and {expected_shape[1]} "
                "source cells"
            )
        if self.coverage.ndim != 1:
            raise ValueError("mapping coverage must be one-dimensional")
        if self.coverage.dtype != np.dtype(np.float32):
            raise ValueError("mapping coverage must use exact float32 dtype")
        if self.coverage.size != self.target_ids.size:
            raise ValueError("mapping coverage size must match target_ids")
        if not np.isfinite(self.coverage).all() or np.any(self.coverage < 0):
            raise ValueError("mapping coverage must be finite and nonnegative")
        for name in ("target_ids", "source_x", "source_y", "coverage"):
            object.__setattr__(
                self, name, immutable_array(getattr(self, name), order="C")
            )
        object.__setattr__(
            self,
            "matrix",
            _freeze_csr_storage(matrix),
        )
        return self

    @classmethod
    def _assemble(
        cls,
        *,
        target_ids: np.ndarray,
        matrix: csr_matrix,
        source_x: np.ndarray,
        source_y: np.ndarray,
        coverage: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> MappingTable:
        """Own a table produced by a transformation of validated storage."""

        owned_matrix = matrix.copy()
        owned_matrix.sum_duplicates()
        owned_matrix.data = canonical_floating_array(
            owned_matrix.data,
            dtype="float32",
            label="transformed mapping weights",
        )
        owned_matrix.eliminate_zeros()
        owned_matrix.sort_indices()
        coverage = canonical_floating_array(
            coverage, dtype="float32", label="transformed mapping coverage"
        )
        if coverage.shape != target_ids.shape or np.any(coverage < 0):
            raise ValueError(
                "transformed mapping coverage must match target IDs and be nonnegative"
            )
        return cls.model_construct(
            target_ids=immutable_array(target_ids, dtype=np.int64, order="C"),
            matrix=_freeze_csr_storage(owned_matrix),
            source_x=immutable_array(source_x, dtype=np.float64, order="C"),
            source_y=immutable_array(source_y, dtype=np.float64, order="C"),
            coverage=immutable_array(coverage, dtype=np.float32, order="C"),
            metadata=immutable_metadata(
                dict(metadata),
                label="mapping metadata",
            ),
        )

    @property
    def _source_shape(self) -> tuple[int, int]:
        return (self.source_y.size, self.source_x.size)

    def require_source_grid(self, longitude: np.ndarray, latitude: np.ndarray) -> None:
        """Reject source coordinates other than the grid this table was built on."""

        if not (
            np.array_equal(longitude, self.source_x)
            and np.array_equal(latitude, self.source_y)
        ):
            raise ValueError(
                "dataset coordinates do not match the mapping source grid; "
                "regenerate the mapping for this dataset"
            )

    def row_normalized(self) -> MappingTable:
        """Return a copy with each row scaled to sum 1 (empty rows stay zero)."""

        matrix = self.matrix.astype(np.float64, copy=True)
        rows = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr)
        )
        matrix.data = normalize_target_weights(rows, matrix.data, matrix.shape[0])
        return MappingTable._assemble(
            target_ids=self.target_ids,
            matrix=matrix,
            source_x=self.source_x,
            source_y=self.source_y,
            coverage=self.coverage,
            metadata={**self.metadata, "normalization": "row_sum"},
        )

    @validate_call(config=HydroForgeModel.model_config)
    def local(self, target_ids: UniqueIds | None = None) -> LocalMapping:
        """Select targets and remove source columns unused by that selection.

        ``None`` keeps every target in mapping order.
        """

        if target_ids is None:
            selected_ids = self.target_ids
            selected = self.matrix
        else:
            selected_ids = target_ids
            rows = find_indices_in(target_ids, self.target_ids)
            missing = rows < 0
            missing_count = np.count_nonzero(missing)
            if missing_count:
                raise ValueError(
                    f"{missing_count} requested target id(s) are absent from "
                    f"the mapping; examples={target_ids[missing][:5].tolist()}"
                )
            selected = self.matrix[rows, :].tocsr()
        selected = selected.copy()
        selected.eliminate_zeros()
        active = np.unique(selected.indices).astype(np.int64)
        compact = csr_matrix(
            (selected.data, np.searchsorted(active, selected.indices), selected.indptr),
            shape=(selected.shape[0], active.size),
        )
        return LocalMapping._assemble(
            target_ids=selected_ids,
            source_indices=active,
            source_to_target=compact.T.tocsr(),
        )

    @staticmethod
    def _nearest_valid_cols(
        valid_grid: np.ndarray,
        start_y: np.ndarray,
        start_x: np.ndarray,
        source_x: np.ndarray,
        source_y: np.ndarray,
        geographic: bool,
    ) -> np.ndarray | None:
        """Nearest valid source cell center of every start cell center.

        Geographic grids compare great-circle distances (chords between unit
        vectors), so longitude convergence and a periodic axis are exact;
        other grids compare planar coordinate distances.
        """
        if not np.any(valid_grid):
            return None
        from scipy.spatial import cKDTree

        nx = valid_grid.shape[1]

        def points(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
            if not geographic:
                return np.column_stack((source_x[cols], source_y[rows]))
            lat = np.radians(source_y[rows])
            lon = np.radians(source_x[cols])
            return np.column_stack(
                (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
            )

        valid_y, valid_x = np.nonzero(valid_grid)
        tree = cKDTree(points(valid_y, valid_x))
        _distance, nearest = tree.query(points(start_y, start_x))
        return valid_y[nearest].astype(np.int64) * nx + valid_x[nearest]

    def _source_periodic_x(self) -> bool:
        """Whether the source longitude axis wraps, from its bound span."""
        recorded = self.metadata.get("source_periodic_x")
        if type(recorded) is bool:
            return recorded
        # Archives written before the flag existed: infer bounds from centers
        # exactly as RegularGrid does for a grid without explicit bounds.
        if self.metadata.get("source_is_geographic") is not True:
            return False
        if self.source_x.size < 2:
            return False
        return _bounds_are_periodic(_axis_bounds(self.source_x))

    @staticmethod
    def _weighted_center_index(
        cols: np.ndarray,
        weights: np.ndarray,
        nx: int,
        periodic_x: bool,
    ) -> tuple[int, int]:
        """Return a weighted source-grid center as integer ``(y, x)`` indices."""
        ys = cols // nx
        xs = cols % nx
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return int(np.round(float(ys.mean()))), int(np.round(float(xs.mean())))

        y0 = int(np.round(float(np.average(ys, weights=weights))))
        if periodic_x:
            angles = 2.0 * np.pi * (xs.astype(np.float64) / float(nx))
            sin_mean = float(np.average(np.sin(angles), weights=weights))
            cos_mean = float(np.average(np.cos(angles), weights=weights))
            x_angle = np.arctan2(sin_mean, cos_mean)
            if x_angle < 0.0:
                x_angle += 2.0 * np.pi
            x0 = int(np.round(x_angle / (2.0 * np.pi) * nx)) % nx
        else:
            x0 = int(np.round(float(np.average(xs, weights=weights))))
        return y0, x0

    @classmethod
    def _weighted_center_indices(
        cls,
        matrix: csr_matrix,
        rows: np.ndarray,
        nx: int,
        periodic_x: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`_weighted_center_index` for non-empty CSR rows.

        Weighted sums use the pairwise order of ``np.average`` on each row.
        """
        starts = matrix.indptr[rows]
        lengths = matrix.indptr[rows + 1] - starts
        entries = np.repeat(
            starts - (np.cumsum(lengths) - lengths), lengths
        ) + np.arange(int(lengths.sum()), dtype=np.int64)
        cols = matrix.indices[entries].astype(np.int64)
        weights = matrix.data[entries].astype(np.float64, copy=False)
        offsets = np.cumsum(lengths) - lengths
        ys = (cols // nx).astype(np.float64)
        weight_sums = _segment_sums(weights, offsets, lengths)
        y0 = np.round(_segment_sums(ys * weights, offsets, lengths) / weight_sums)
        if periodic_x:
            angles = 2.0 * np.pi * ((cols % nx).astype(np.float64) / float(nx))
            sin_mean = _segment_sums(np.sin(angles) * weights, offsets, lengths)
            cos_mean = _segment_sums(np.cos(angles) * weights, offsets, lengths)
            x_angle = np.arctan2(sin_mean / weight_sums, cos_mean / weight_sums)
            x_angle = np.where(x_angle < 0.0, x_angle + 2.0 * np.pi, x_angle)
            x0 = np.round(x_angle / (2.0 * np.pi) * nx).astype(np.int64) % nx
        else:
            x0 = np.round(
                _segment_sums(
                    (cols % nx).astype(np.float64) * weights, offsets, lengths
                )
                / weight_sums
            ).astype(np.int64)
        y0 = y0.astype(np.int64)
        # Rows without positive weight keep the scalar helper's mean fallback.
        for index in np.flatnonzero(~(weight_sums > 0.0)):
            start = starts[index]
            y0[index], x0[index] = cls._weighted_center_index(
                matrix.indices[start : start + lengths[index]].astype(np.int64),
                matrix.data[start : start + lengths[index]],
                nx,
                periodic_x,
            )
        return y0, x0

    @validate_call(config=HydroForgeModel.model_config)
    def with_source_mask(
        self,
        valid_source_mask: Annotated[np.ndarray, AfterValidator(_boolean_mask)],
        *,
        empty_row_policy: Literal["zero", "nearest"] = "zero",
        preserve_row_sum: bool = True,
    ) -> Self:
        """Return a mapping with invalid source cells removed.

        ``valid_source_mask`` has the ``(y, x)`` source grid shape.
        ``empty_row_policy="nearest"`` repairs rows that originally had source
        support but become empty after masking by assigning the original row sum
        to the nearest valid source cell.  Coverage is scaled by the fraction
        of each row's original weight that remains on valid source cells.
        """

        if valid_source_mask.shape != self._source_shape:
            raise ValueError(
                f"valid_source_mask shape {valid_source_mask.shape} does not "
                f"match mapping source shape {self._source_shape}"
            )
        valid = valid_source_mask.reshape(-1)

        original = self.matrix.astype(np.float64, copy=True)
        original_row_sums = np.asarray(
            original.sum(axis=1),
            dtype=np.float64,
        ).ravel()

        coo = original.tocoo()
        keep = valid[coo.col]
        masked = csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=original.shape,
            dtype=np.float64,
        )
        valid_row_sums = np.asarray(
            masked.sum(axis=1),
            dtype=np.float64,
        ).ravel()

        scaled_rows = 0
        if preserve_row_sum:
            scale = np.ones_like(original_row_sums, dtype=np.float64)
            can_scale = (original_row_sums > 0.0) & (valid_row_sums > 0.0)
            changed = can_scale & ~np.isclose(original_row_sums, valid_row_sums)
            scale[can_scale] = original_row_sums[can_scale] / valid_row_sums[can_scale]
            scaled_rows = int(np.sum(changed))
            masked = masked.multiply(scale[:, None]).tocsr()

        empty_rows = np.where((original_row_sums > 0.0) & (valid_row_sums <= 0.0))[0]
        repaired_rows = 0
        if empty_row_policy == "nearest" and empty_rows.size:
            ny, nx = self._source_shape
            valid_grid = valid.reshape(ny, nx)
            periodic_x = self._source_periodic_x()

            repair_row: list[int] = []
            repair_col: list[int] = []
            repair_val: list[float] = []
            lengths = np.diff(original.indptr)[empty_rows]
            candidates = empty_rows[lengths > 0]
            if candidates.size:
                y0, x0 = self._weighted_center_indices(
                    original, candidates, nx, periodic_x
                )
                nearest = self._nearest_valid_cols(
                    valid_grid,
                    y0,
                    x0,
                    self.source_x,
                    self.source_y,
                    self.metadata.get("source_is_geographic") is True,
                )
                if nearest is not None:
                    repair_row = candidates.tolist()
                    repair_col = nearest.tolist()
                    repair_val = original_row_sums[candidates].tolist()

            if repair_row:
                repair = csr_matrix(
                    (
                        np.asarray(repair_val, dtype=np.float64),
                        (
                            np.asarray(repair_row, dtype=np.int64),
                            np.asarray(repair_col, dtype=np.int64),
                        ),
                    ),
                    shape=original.shape,
                    dtype=np.float64,
                )
                masked = (masked + repair).tocsr()
                repaired_rows = len(repair_row)

        metadata = {
            **self.metadata,
            "source_mask_valid_cells": int(valid.sum()),
            "source_mask_invalid_cells": int(valid.size - valid.sum()),
            "source_mask_preserve_row_sum": bool(preserve_row_sum),
            "source_mask_empty_row_policy": empty_row_policy,
            "source_mask_empty_rows": int(empty_rows.size),
            "source_mask_repaired_rows": repaired_rows,
            "source_mask_scaled_rows": scaled_rows,
        }
        coverage = np.asarray(self.coverage, dtype=np.float64) * np.divide(
            valid_row_sums,
            original_row_sums,
            out=np.ones_like(original_row_sums),
            where=original_row_sums > 0.0,
        )
        return MappingTable._assemble(
            target_ids=self.target_ids,
            matrix=masked,
            source_x=self.source_x,
            source_y=self.source_y,
            coverage=coverage,
            metadata=metadata,
        )

    @validate_call(config=HydroForgeModel.model_config)
    def save(self, path: _FilePath) -> Path:
        """Write the mapping as a v2 ``.npz`` archive and return its path."""

        out_path = path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "schema": _MAPPING_SCHEMA}
        with atomic_output_path(out_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    target_ids=self.target_ids,
                    sparse_data=self.matrix.data,
                    sparse_indices=self.matrix.indices.astype(np.int64, copy=False),
                    sparse_indptr=self.matrix.indptr.astype(np.int64, copy=False),
                    matrix_shape=np.asarray(self.matrix.shape, dtype=np.int64),
                    coord_lon=self.source_x,
                    coord_lat=self.source_y,
                    coverage=self.coverage,
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
        return out_path

    @classmethod
    @validate_call(config=HydroForgeModel.model_config)
    def load(cls, path: _FilePath) -> Self:
        """Read and fully validate one v2 mapping archive."""

        with np.load(path, allow_pickle=False) as data:
            keys = frozenset(data.files)
            missing = _MAPPING_ARCHIVE_KEYS - keys
            unexpected = keys - _MAPPING_ARCHIVE_KEYS
            if missing or unexpected:
                raise ValueError(
                    "mapping archive does not use the exact v2 schema; "
                    "regenerate the mapping. "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )

            raw_shape = np.asarray(data["matrix_shape"])
            shape = canonical_ids(raw_shape, label="matrix_shape")
            if shape.shape != (2,) or shape[0] < 0 or shape[1] < 1:
                raise ValueError(
                    "matrix_shape must contain a nonnegative row count and "
                    "a positive column count"
                )
            n_rows, n_cols = map(int, shape)
            sparse_data = np.asarray(data["sparse_data"])
            raw_indices = np.asarray(data["sparse_indices"])
            raw_indptr = np.asarray(data["sparse_indptr"])
            indices = canonical_ids(raw_indices, label="sparse_indices")
            indptr = canonical_ids(raw_indptr, label="sparse_indptr")
            _validate_csr_components(sparse_data, indices, indptr, (n_rows, n_cols))

            target_ids = canonical_ids(
                np.asarray(data["target_ids"]),
                label="target_ids",
            )

            matrix = csr_matrix(
                (sparse_data, indices, indptr),
                shape=(n_rows, n_cols),
            )
            raw_metadata = np.asarray(data["metadata_json"])
            if raw_metadata.shape != () or raw_metadata.dtype.kind not in "US":
                raise ValueError("metadata_json must be a scalar JSON string")
            metadata = json.loads(str(raw_metadata.item()))
            if type(metadata) is not dict:
                raise TypeError("mapping metadata JSON must decode to an object")
            if metadata.pop("schema", None) != _MAPPING_SCHEMA:
                raise ValueError(
                    "mapping archive has an unsupported schema; regenerate "
                    "the mapping with this HydroForge version"
                )
            return cls(
                target_ids=target_ids,
                matrix=matrix,
                source_x=np.asarray(data["coord_lon"]),
                source_y=np.asarray(data["coord_lat"]),
                coverage=np.asarray(data["coverage"]),
                metadata=metadata,
            )

    @validate_call(config=HydroForgeModel.model_config)
    def apply(
        self,
        data: Annotated[np.ndarray, BeforeValidator(_apply_input)],
        *,
        layout: Literal["flat", "grid"],
    ) -> np.ndarray:
        """Apply the mapping over the trailing source axes of ``layout``.

        ``"grid"`` data end with the ``(y, x)`` source shape, ``"flat"`` data
        with the flattened source size; NaN values propagate.
        """

        n_source = self.matrix.shape[1]
        if layout == "grid":
            if data.ndim < 2 or data.shape[-2:] != self._source_shape:
                raise ValueError(
                    f"grid mapping input shape {data.shape} does not end with "
                    f"source shape {self._source_shape}"
                )
            leading = data.shape[:-2]
        else:
            if data.shape[-1] != n_source:
                raise ValueError(
                    f"flat mapping input shape {data.shape} does not end with "
                    f"source size {n_source}"
                )
            leading = data.shape[:-1]
        flat_2d = data.reshape(-1, n_source)
        out = (self.matrix @ flat_2d.T).T
        return np.asarray(out).reshape(*leading, self.matrix.shape[0])

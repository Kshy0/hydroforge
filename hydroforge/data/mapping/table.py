"""Sparse mapping table from flattened source grid cells to target supports."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, Self

import numpy as np
import torch
from pydantic import (
    Field,
    PrivateAttr,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.sparse import csr_matrix

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.numeric import (
    canonical_floating_array,
    canonical_ids,
    immutable_array,
    immutable_metadata,
)
from hydroforge.serialization.files import atomic_output_path


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
_MAPPING_APPLY_CONTEXT = "hydroforge_mapping_apply"


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
        self._reject_mutation()
        super().sort_indices()

    def sum_duplicates(self) -> None:
        self._reject_mutation()
        super().sum_duplicates()


def _freeze_csr_storage(matrix: csr_matrix) -> csr_matrix:
    """Replace CSR component arrays with immutable-buffer-backed arrays."""

    frozen = _ImmutableCSR(matrix, copy=True)
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
            raise ValueError(
                f"{path} keys must be non-empty exact strings"
            )
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


class _MappingSaveRequest(HydroForgeModel):
    path: str | Path


class _MappingApplyRequest(HydroForgeModel):
    data: np.ndarray
    layout: Literal["flat", "grid"]

    @model_validator(mode="after")
    def _validate_input(self, info: ValidationInfo):
        mapping = (
            info.context.get(_MAPPING_APPLY_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if mapping is None:
            raise ValueError("mapping apply requires MappingTable context")
        if np.ma.isMaskedArray(self.data) and np.any(np.ma.getmaskarray(self.data)):
            raise ValueError("mapping input contains missing values")
        arr = np.asarray(self.data)
        if arr.ndim == 0:
            raise ValueError("mapping input must have at least one dimension")
        if arr.dtype.kind not in {"f", "i", "u"}:
            raise ValueError("mapping input must contain real numbers")
        arr = canonical_floating_array(
            arr,
            dtype="float64",
            label="mapping input",
            allow_nan=True,
        )
        if self.layout == "grid":
            if arr.ndim < 2 or arr.shape[-2:] != mapping._source_shape:
                raise ValueError(
                    f"grid mapping input shape {arr.shape} does not end with "
                    f"source shape {mapping._source_shape}"
                )
        elif arr.shape[-1] != mapping.matrix.shape[1]:
            raise ValueError(
                f"flat mapping input shape {arr.shape} does not end with "
                f"source size {mapping.matrix.shape[1]}"
            )
        object.__setattr__(self, "data", arr)
        return self


class LocalMapping(HydroForgeModel):
    """Target-selected mapping with unused source columns removed."""

    target_ids: np.ndarray
    source_indices: np.ndarray
    source_to_target: csr_matrix

    @model_validator(mode="after")
    def _validate_local_mapping(self) -> Self:
        target_ids = canonical_ids(self.target_ids, label="target_ids")
        source_indices = canonical_ids(
            self.source_indices,
            label="source_indices",
        )
        if np.unique(target_ids).size != target_ids.size:
            raise ValueError("target_ids must be unique")
        if np.unique(source_indices).size != source_indices.size:
            raise ValueError("source_indices must be unique")
        if source_indices.size and np.any(source_indices < 0):
            raise ValueError("source_indices must be nonnegative")
        if not isinstance(self.source_to_target, csr_matrix):
            raise ValueError("source_to_target must be a scipy CSR matrix")
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
            self, "target_ids", immutable_array(target_ids, order="C"),
        )
        object.__setattr__(
            self,
            "source_indices",
            immutable_array(source_indices, order="C"),
        )
        object.__setattr__(
            self, "source_to_target", _freeze_csr_storage(matrix),
        )
        return self

    @classmethod
    def _from_trusted(
        cls,
        *,
        target_ids: np.ndarray,
        source_indices: np.ndarray,
        source_to_target: csr_matrix,
    ) -> "LocalMapping":
        """Own a local projection derived solely from a validated mapping."""

        owned_target_ids = np.array(
            target_ids, dtype=np.int64, order="C", copy=True,
        )
        owned_source_indices = np.array(
            source_indices, dtype=np.int64, order="C", copy=True,
        )
        matrix = source_to_target.copy().astype(np.float32)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return cls.model_construct(
            target_ids=immutable_array(owned_target_ids, order="C"),
            source_indices=immutable_array(
                owned_source_indices, order="C",
            ),
            source_to_target=_freeze_csr_storage(matrix),
        )

    def to_torch(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize the canonical ``active_source x target`` sparse tensor."""

        matrix = self.source_to_target.tocoo()
        indices = torch.tensor(
            np.stack((matrix.row, matrix.col)),
            device=device,
            dtype=torch.int64,
        )
        values = torch.tensor(matrix.data, device=device, dtype=dtype)
        with torch.sparse.check_sparse_tensor_invariants(enable=True):
            tensor = torch.sparse_coo_tensor(
                indices,
                values,
                size=matrix.shape,
                device=device,
                dtype=dtype,
            )
        return tensor.coalesce()


class _LocalMappingRequest(HydroForgeModel):
    """Bind a validated target selection to one loaded mapping identity."""

    mapping: Any = Field(exclude=True)
    target_ids: np.ndarray | None = Field(default=None, exclude=True)

    _local_mapping: LocalMapping = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self):
        mapping = self.mapping
        selected_ids = (
            mapping.target_ids.copy() if self.target_ids is None else self.target_ids
        )
        row_by_id = {
            int(target_id): row for row, target_id in enumerate(mapping.target_ids)
        }
        missing = tuple(
            int(target_id)
            for target_id in selected_ids
            if int(target_id) not in row_by_id
        )
        if missing:
            raise ValueError(
                f"{len(missing)} requested target id(s) are absent from the "
                f"mapping; examples={list(missing[:5])}"
            )
        rows = np.asarray(
            [row_by_id[int(target_id)] for target_id in selected_ids],
            dtype=np.int64,
        )
        selected = mapping.matrix[rows, :].tocsr()
        active = np.flatnonzero(
            np.asarray(selected.sum(axis=0)).ravel() != 0,
        ).astype(np.int64)
        self._local_mapping = LocalMapping._from_trusted(
            target_ids=selected_ids,
            source_indices=active,
            source_to_target=(selected[:, active].T.tocsr().astype(np.float32)),
        )
        return self

    @property
    def local_mapping(self) -> LocalMapping:
        return self._local_mapping


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
        if self.target_ids.ndim != 1:
            raise ValueError("mapping target_ids must be one-dimensional")
        if self.target_ids.dtype != np.dtype(np.int64):
            raise ValueError("mapping target_ids must use exact int64 dtype")
        if np.unique(self.target_ids).size != self.target_ids.size:
            raise ValueError("mapping target_ids must be unique")
        if not isinstance(self.matrix, csr_matrix):
            raise ValueError("mapping matrix must be a scipy CSR matrix")
        if self.matrix.dtype != np.dtype(np.float32):
            raise ValueError("mapping matrix must use exact float32 dtype")
        if not self.matrix.has_canonical_format:
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
        if (
            np.unique(self.source_x).size != self.source_x.size
            or np.unique(self.source_y).size != self.source_y.size
        ):
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
        target_ids = np.array(
            self.target_ids, dtype=np.int64, order="C", copy=True,
        )
        matrix = self.matrix.copy()
        source_x = np.array(
            self.source_x, dtype=np.float64, order="C", copy=True,
        )
        source_y = np.array(
            self.source_y, dtype=np.float64, order="C", copy=True,
        )
        coverage = np.array(
            self.coverage, dtype=np.float32, order="C", copy=True,
        )
        object.__setattr__(
            self, "target_ids", immutable_array(target_ids, order="C"),
        )
        object.__setattr__(
            self, "matrix", _freeze_csr_storage(matrix),
        )
        object.__setattr__(
            self, "source_x", immutable_array(source_x, order="C"),
        )
        object.__setattr__(
            self, "source_y", immutable_array(source_y, order="C"),
        )
        object.__setattr__(
            self, "coverage", immutable_array(coverage, order="C"),
        )
        return self

    @classmethod
    def _from_trusted(
        cls,
        *,
        target_ids: np.ndarray,
        matrix: csr_matrix,
        source_x: np.ndarray,
        source_y: np.ndarray,
        coverage: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> "MappingTable":
        """Own a table produced by a transformation of validated storage."""

        owned_target_ids = np.array(
            target_ids, dtype=np.int64, order="C", copy=True,
        )
        owned_matrix = matrix.copy().astype(np.float32)
        owned_matrix.sum_duplicates()
        owned_matrix.eliminate_zeros()
        owned_matrix.sort_indices()
        owned_source_x = np.array(
            source_x, dtype=np.float64, order="C", copy=True,
        )
        owned_source_y = np.array(
            source_y, dtype=np.float64, order="C", copy=True,
        )
        owned_coverage = np.array(
            coverage, dtype=np.float32, order="C", copy=True,
        )
        return cls.model_construct(
            target_ids=immutable_array(owned_target_ids, order="C"),
            matrix=_freeze_csr_storage(owned_matrix),
            source_x=immutable_array(owned_source_x, order="C"),
            source_y=immutable_array(owned_source_y, order="C"),
            coverage=immutable_array(owned_coverage, order="C"),
            metadata=immutable_metadata(
                dict(metadata), label="mapping metadata",
            ),
        )

    @property
    def _source_shape(self) -> tuple[int, int]:
        return (self.source_y.size, self.source_x.size)

    def _row_normalized(self) -> "MappingTable":
        """Return a copy with each row scaled to sum 1 (empty rows stay zero)."""
        matrix = self.matrix.tocsr(copy=True).astype(np.float64)
        row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        scale = np.zeros_like(row_sums)
        nz = row_sums > 0
        scale[nz] = 1.0 / row_sums[nz]
        matrix = matrix.multiply(scale[:, None]).tocsr().astype(np.float32)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return MappingTable._from_trusted(
            target_ids=self.target_ids,
            matrix=matrix,
            source_x=self.source_x,
            source_y=self.source_y,
            coverage=self.coverage,
            metadata={**self.metadata, "normalization": "row_sum"},
        )

    def _local(self, target_ids: np.ndarray | None = None) -> LocalMapping:
        """Select targets and remove source columns unused by that selection."""
        return _LocalMappingRequest(
            mapping=self,
            target_ids=target_ids,
        ).local_mapping

    @staticmethod
    def _nearest_valid_col(
        valid_grid: np.ndarray,
        start_y: int,
        start_x: int,
        periodic_x: bool,
    ) -> int | None:
        """Find the nearest valid source cell in index space."""
        ny, nx = valid_grid.shape
        if periodic_x:
            start_x %= nx
        if valid_grid[start_y, start_x]:
            return start_y * nx + start_x
        if not np.any(valid_grid):
            return None

        max_radius = max(ny, nx)
        for radius in range(1, max_radius + 1):
            cand_y: list[np.ndarray] = []
            cand_x: list[np.ndarray] = []

            x_range = np.arange(start_x - radius, start_x + radius + 1, dtype=np.int64)
            if periodic_x:
                x_idx = np.mod(x_range, nx)
            else:
                x_idx = x_range[(x_range >= 0) & (x_range < nx)]

            for y in (start_y - radius, start_y + radius):
                if 0 <= y < ny and x_idx.size:
                    hit = valid_grid[y, x_idx]
                    if np.any(hit):
                        cand_y.append(np.full(int(hit.sum()), y, dtype=np.int64))
                        cand_x.append(x_idx[hit])

            y_inner = np.arange(
                max(0, start_y - radius + 1),
                min(ny, start_y + radius),
                dtype=np.int64,
            )
            for x in (start_x - radius, start_x + radius):
                if y_inner.size and (periodic_x or 0 <= x < nx):
                    x_mod = int(x % nx) if periodic_x else int(x)
                    hit = valid_grid[y_inner, x_mod]
                    if np.any(hit):
                        cand_y.append(y_inner[hit])
                        cand_x.append(np.full(int(hit.sum()), x_mod, dtype=np.int64))

            if cand_y:
                ys = np.concatenate(cand_y)
                xs = np.concatenate(cand_x)
                dy = ys - start_y
                dx = np.abs(xs - start_x)
                if periodic_x:
                    dx = np.minimum(dx, nx - dx)
                best = int(np.argmin(dy * dy + dx * dx))
                return int(ys[best] * nx + xs[best])
        return None

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

    def _with_source_mask_trusted(
        self,
        valid_source_mask: np.ndarray,
        *,
        empty_row_policy: Literal["zero", "nearest"] = "zero",
        preserve_row_sum: bool = True,
    ) -> "MappingTable":
        """Return a mapping with invalid source cells removed.

        ``empty_row_policy="nearest"`` repairs rows that originally had source
        support but become empty after masking by assigning the original row sum
        to the nearest valid source cell.
        """
        valid = valid_source_mask.reshape(-1)

        original = self.matrix.tocsr(copy=True).astype(np.float64)
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
            dx = (
                abs(float(self.source_x[1] - self.source_x[0]))
                if self.source_x.size > 1
                else 0.0
            )
            longitude_span = (
                abs(float(self.source_x[-1] - self.source_x[0]))
                * self.source_x.size
                / (self.source_x.size - 1)
                if self.source_x.size > 1
                else 0.0
            )
            periodic_x = bool(
                self.metadata.get("source_is_geographic") is True
                and dx > 0.0
                and np.isclose(
                    longitude_span,
                    360.0,
                    rtol=0.0,
                    atol=2.0e-5,
                )
            )

            repair_row: list[int] = []
            repair_col: list[int] = []
            repair_val: list[float] = []
            for row in empty_rows:
                start, end = original.indptr[row], original.indptr[row + 1]
                cols = original.indices[start:end]
                weights = original.data[start:end]
                if cols.size == 0:
                    continue
                y0, x0 = self._weighted_center_index(
                    cols.astype(np.int64),
                    weights,
                    nx,
                    periodic_x,
                )
                nearest_col = self._nearest_valid_col(valid_grid, y0, x0, periodic_x)
                if nearest_col is None:
                    continue
                repair_row.append(int(row))
                repair_col.append(int(nearest_col))
                repair_val.append(
                    float(original_row_sums[row] if preserve_row_sum else 1.0)
                )

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

        masked.eliminate_zeros()
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
        masked = masked.astype(np.float32)
        masked.eliminate_zeros()
        masked.sort_indices()
        return MappingTable._from_trusted(
            target_ids=self.target_ids,
            matrix=masked,
            source_x=self.source_x,
            source_y=self.source_y,
            coverage=self.coverage,
            metadata=metadata,
        )

    def save(self, path: str | Path) -> Path:
        request = _MappingSaveRequest(path=path)
        out_path = Path(request.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "schema": _MAPPING_SCHEMA}
        with atomic_output_path(out_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    target_ids=self.target_ids.astype(np.int64),
                    sparse_data=self.matrix.data.astype(np.float32),
                    sparse_indices=self.matrix.indices.astype(np.int64),
                    sparse_indptr=self.matrix.indptr.astype(np.int64),
                    matrix_shape=np.asarray(self.matrix.shape, dtype=np.int64),
                    coord_lon=self.source_x.astype(np.float64),
                    coord_lat=self.source_y.astype(np.float64),
                    coverage=self.coverage.astype(np.float32),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
        return out_path

    @classmethod
    def _load(
        cls,
        path: str | Path,
    ) -> Self:
        with np.load(Path(path), allow_pickle=False) as data:
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
            if raw_shape.ndim != 1:
                raise ValueError("matrix_shape must be one-dimensional")
            shape = canonical_ids(raw_shape, label="matrix_shape")
            if (
                shape.shape != (2,)
                or shape[0] < 0
                or shape[1] < 1
            ):
                raise ValueError(
                    "matrix_shape must contain a nonnegative row count and "
                    "a positive column count"
                )
            n_rows, n_cols = map(int, shape)
            sparse_data = np.asarray(data["sparse_data"])
            if sparse_data.ndim != 1:
                raise ValueError("sparse_data must be one-dimensional")
            raw_indices = np.asarray(data["sparse_indices"])
            raw_indptr = np.asarray(data["sparse_indptr"])
            if raw_indices.ndim != 1:
                raise ValueError("sparse_indices must be one-dimensional")
            if raw_indptr.ndim != 1:
                raise ValueError("sparse_indptr must be one-dimensional")
            indices = canonical_ids(raw_indices, label="sparse_indices")
            indptr = canonical_ids(raw_indptr, label="sparse_indptr")
            if indices.size != sparse_data.size:
                raise ValueError(
                    "sparse_indices must be one-dimensional and match sparse_data"
                )
            if indptr.shape != (n_rows + 1,):
                raise ValueError(f"sparse_indptr must have shape ({n_rows + 1},)")
            if (
                indptr[0] != 0
                or indptr[-1] != sparse_data.size
                or np.any(np.diff(indptr) < 0)
            ):
                raise ValueError("sparse_indptr is not a valid CSR row pointer")
            if indices.size and (indices.min() < 0 or indices.max() >= n_cols):
                raise ValueError("sparse_indices fall outside matrix_shape")

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

    def apply(
        self,
        data: np.ndarray,
        *,
        layout: Literal["flat", "grid"],
    ) -> np.ndarray:
        """Apply mapping using one explicit source-axis layout."""
        request = _MappingApplyRequest.model_validate(
            {"data": data, "layout": layout},
            context={_MAPPING_APPLY_CONTEXT: self},
        )
        return self._apply_trusted(request.data, layout=request.layout)

    def _apply_trusted(
        self,
        data: np.ndarray,
        *,
        layout: Literal["flat", "grid"],
    ) -> np.ndarray:
        """Apply a compiler-owned canonical array without rebuilding a query."""

        arr = data
        if layout == "grid":
            leading = arr.shape[:-2]
            flat = arr.reshape(*leading, self.matrix.shape[1])
        else:
            leading = arr.shape[:-1]
            flat = arr
        flat_2d = flat.reshape(-1, self.matrix.shape[1])
        out = (self.matrix @ flat_2d.T).T
        return np.asarray(out).reshape(*leading, self.matrix.shape[0])

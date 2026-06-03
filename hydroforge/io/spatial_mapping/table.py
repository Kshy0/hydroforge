"""Sparse mapping table from flattened source grid cells to target supports."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class MappingTable:
    """CSR mapping from flattened source grid cells to target supports."""

    target_ids: np.ndarray
    matrix: csr_matrix
    source_x: np.ndarray
    source_y: np.ndarray
    coverage: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_ids = np.asarray(self.target_ids, dtype=np.int64).ravel()
        self.matrix = self.matrix.tocsr().astype(np.float32)
        self.source_x = np.asarray(self.source_x, dtype=np.float64).ravel()
        self.source_y = np.asarray(self.source_y, dtype=np.float64).ravel()
        if self.coverage is None:
            self.coverage = np.asarray(self.matrix.sum(axis=1)).ravel().astype(np.float32)
        else:
            self.coverage = np.asarray(self.coverage, dtype=np.float32).ravel()
        if self.matrix.shape != (self.target_ids.size, self.source_x.size * self.source_y.size):
            raise ValueError(
                f"matrix shape {self.matrix.shape} is inconsistent with "
                f"{self.target_ids.size} targets and {self.source_x.size * self.source_y.size} source cells"
            )

    @property
    def source_shape(self) -> tuple[int, int]:
        return (self.source_y.size, self.source_x.size)

    def row_normalized(self) -> "MappingTable":
        """Return a copy with each row scaled to sum 1 (empty rows stay zero)."""
        matrix = self.matrix.tocsr(copy=True).astype(np.float32)
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.zeros_like(row_sums)
        nz = row_sums > 0
        scale[nz] = 1.0 / row_sums[nz]
        matrix = matrix.multiply(scale[:, None]).tocsr().astype(np.float32)
        return MappingTable(
            self.target_ids,
            matrix,
            self.source_x,
            self.source_y,
            coverage=self.coverage,
            metadata={**self.metadata, "normalization": "row_sum"},
        )

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema": "hydroforge.spatial_mapping.v1", **self.metadata}
        np.savez_compressed(
            out_path,
            catchment_ids=self.target_ids.astype(np.int64),
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
    def load(cls, path: str | Path) -> "MappingTable":
        data = np.load(Path(path), allow_pickle=False)
        matrix = csr_matrix(
            (data["sparse_data"], data["sparse_indices"], data["sparse_indptr"]),
            shape=tuple(data["matrix_shape"].astype(np.int64)),
        )
        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"]))
        target_key = "target_ids" if "target_ids" in data else "catchment_ids"
        return cls(
            target_ids=data[target_key],
            matrix=matrix,
            source_x=data["coord_lon"],
            source_y=data["coord_lat"],
            coverage=data["coverage"] if "coverage" in data else None,
            metadata=metadata,
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply mapping to ``(..., source)`` or ``(..., y, x)`` arrays."""
        arr = np.asarray(data)
        if arr.shape[-2:] == self.source_shape:
            leading = arr.shape[:-2]
            flat = arr.reshape(*leading, self.matrix.shape[1])
        elif arr.shape[-1] == self.matrix.shape[1]:
            leading = arr.shape[:-1]
            flat = arr
        else:
            raise ValueError(
                f"Input shape {arr.shape} does not end with source shape "
                f"{self.source_shape} or source size {self.matrix.shape[1]}"
            )
        flat_2d = flat.reshape(-1, self.matrix.shape[1])
        out = (self.matrix @ flat_2d.T).T
        return np.asarray(out).reshape(*leading, self.matrix.shape[0])

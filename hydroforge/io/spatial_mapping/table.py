"""Sparse mapping table from flattened source grid cells to target supports."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

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

    @staticmethod
    def _nearest_valid_col(
        valid_grid: np.ndarray,
        start_y: int,
        start_x: int,
        periodic_x: bool,
    ) -> int | None:
        """Find the nearest valid source cell in index space."""
        ny, nx = valid_grid.shape
        start_y = int(np.clip(start_y, 0, ny - 1))
        start_x = int(start_x % nx) if periodic_x else int(np.clip(start_x, 0, nx - 1))
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

    def with_source_mask(
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
        if empty_row_policy not in ("zero", "nearest"):
            raise ValueError("empty_row_policy must be 'zero' or 'nearest'")

        valid = np.asarray(valid_source_mask, dtype=bool).reshape(-1)
        if valid.size != self.matrix.shape[1]:
            raise ValueError(
                f"valid_source_mask size {valid.size} does not match "
                f"mapping source size {self.matrix.shape[1]}"
            )

        original = self.matrix.tocsr(copy=True).astype(np.float32)
        original_row_sums = np.asarray(original.sum(axis=1)).ravel()

        coo = original.tocoo()
        keep = valid[coo.col]
        masked = csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=original.shape,
            dtype=np.float32,
        )
        valid_row_sums = np.asarray(masked.sum(axis=1)).ravel()

        scaled_rows = 0
        if preserve_row_sum:
            scale = np.ones_like(original_row_sums, dtype=np.float32)
            can_scale = (original_row_sums > 0.0) & (valid_row_sums > 0.0)
            changed = can_scale & ~np.isclose(original_row_sums, valid_row_sums)
            scale[can_scale] = original_row_sums[can_scale] / valid_row_sums[can_scale]
            scaled_rows = int(np.sum(changed))
            masked = masked.multiply(scale[:, None]).tocsr().astype(np.float32)

        empty_rows = np.where((original_row_sums > 0.0) & (valid_row_sums <= 0.0))[0]
        repaired_rows = 0
        if empty_row_policy == "nearest" and empty_rows.size:
            ny, nx = self.source_shape
            valid_grid = valid.reshape(ny, nx)
            dx = abs(float(self.source_x[1] - self.source_x[0])) if self.source_x.size > 1 else 0.0
            periodic_x = bool(dx > 0.0 and np.isclose(dx * self.source_x.size, 360.0, atol=1e-6))

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
                repair_val.append(float(original_row_sums[row] if preserve_row_sum else 1.0))

            if repair_row:
                repair = csr_matrix(
                    (
                        np.asarray(repair_val, dtype=np.float32),
                        (np.asarray(repair_row, dtype=np.int64), np.asarray(repair_col, dtype=np.int64)),
                    ),
                    shape=original.shape,
                    dtype=np.float32,
                )
                masked = (masked + repair).tocsr().astype(np.float32)
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
        return MappingTable(
            self.target_ids,
            masked,
            self.source_x,
            self.source_y,
            coverage=self.coverage,
            metadata=metadata,
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

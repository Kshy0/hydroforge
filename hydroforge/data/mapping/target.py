"""Target supports that consume values from a source grid.

A :class:`TargetSupport` is the destination geometry of a mapping: regular-grid
mask cells, per-cell points (e.g. VIC), or CaMa catchments reconstructed from a
``parameters.nc`` ``GridSpec`` annotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hydroforge.data.mapping.grid import RegularGrid


@dataclass
class TargetSupport:
    """Target areas that consume values from a source grid."""

    target_ids: np.ndarray
    bounds: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    flat_indices: np.ndarray | None = None
    target_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_ids = np.asarray(self.target_ids, dtype=np.int64).ravel()
        n_target = self.target_ids.size
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds, dtype=np.float64)
            if self.bounds.shape != (n_target, 4):
                raise ValueError(f"bounds must have shape ({n_target}, 4), got {self.bounds.shape}")
        if self.x is not None:
            self.x = np.asarray(self.x, dtype=np.float64).ravel()
        if self.y is not None:
            self.y = np.asarray(self.y, dtype=np.float64).ravel()
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y target centers must be provided together")
        if self.x is not None and self.x.size != n_target:
            raise ValueError("target center size does not match target_ids")

    @classmethod
    def from_mask(
        cls,
        longitude: np.ndarray,
        latitude: np.ndarray,
        mask: np.ndarray,
        *,
        target_ids: np.ndarray | None = None,
        is_geographic: bool | None = None,
        x_bounds: np.ndarray | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> "TargetSupport":
        grid = RegularGrid.from_coordinates(
            longitude,
            latitude,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != grid.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match grid shape {grid.shape}")
        rows, cols = np.where(mask_arr)
        flat_indices = np.ravel_multi_index((rows, cols), grid.shape).astype(np.int64)
        ids = flat_indices if target_ids is None else np.asarray(target_ids, dtype=np.int64).ravel()
        if ids.size != flat_indices.size:
            raise ValueError("target_ids size does not match the active mask size")
        bounds = np.column_stack((
            grid.x_bounds[cols, 0], grid.x_bounds[cols, 1],
            grid.y_bounds[rows, 0], grid.y_bounds[rows, 1],
        ))
        return cls(
            ids,
            bounds=bounds,
            x=grid.x[cols],
            y=grid.y[rows],
            flat_indices=flat_indices,
            target_shape=grid.shape,
            metadata={"kind": "regular_mask"},
        )

    @classmethod
    def from_points(
        cls,
        longitude: np.ndarray,
        latitude: np.ndarray,
        *,
        target_ids: np.ndarray | None = None,
        cell_size: float | tuple[float, float] | None = None,
    ) -> "TargetSupport":
        """Build point targets from per-cell ``(longitude, latitude)`` centers.

        Each target is a single regular grid cell located at its center.  This
        is the support for models stored as a sparse 1D list of regular cells
        (e.g. VIC), as opposed to the CaMa MERIT sub-pixel scaffold.  Passing
        ``cell_size`` (scalar or ``(dx, dy)`` in the coordinate units) adds cell
        bounds so the ``overlap`` method can be used; without it only
        ``nearest`` is available.
        """
        lon = np.asarray(longitude, dtype=np.float64).ravel()
        lat = np.asarray(latitude, dtype=np.float64).ravel()
        if lon.size != lat.size:
            raise ValueError("longitude and latitude must have the same length")
        ids = (
            np.arange(lon.size, dtype=np.int64)
            if target_ids is None
            else np.asarray(target_ids, dtype=np.int64).ravel()
        )
        bounds = None
        if cell_size is not None:
            if np.isscalar(cell_size):
                dx = dy = float(cell_size)
            else:
                dx, dy = (float(v) for v in cell_size)
            bounds = np.column_stack((
                lon - 0.5 * dx, lon + 0.5 * dx,
                lat - 0.5 * dy, lat + 0.5 * dy,
            ))
        return cls(ids, bounds=bounds, x=lon, y=lat, metadata={"kind": "points"})


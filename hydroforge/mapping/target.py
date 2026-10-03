"""Target supports that consume values from a source grid.

A :class:`TargetSupport` is the destination geometry of a mapping: regular-grid
mask cells, per-cell points (e.g. VIC), or CaMa catchments reconstructed from a
``parameters.nc`` ``GridSpec`` annotation.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

import numpy as np
from pydantic import (
    BeforeValidator,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
    validate_call,
)

from hydroforge.core.arrays import (
    canonical_float64,
    canonical_ids,
    immutable_array,
    immutable_metadata,
    positive_finite_float64,
)
from hydroforge.core.validation import HydroForgeModel
from hydroforge.mapping.grid import RegularGrid, _has_duplicates


def _cell_size(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if np.isscalar(value):
        size = positive_finite_float64(value, label="cell_size")
        return size, size
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(
            "cell_size must be a real scalar or a two-value sequence"
        ) from error
    if len(values) != 2:
        raise ValueError("cell_size must be a scalar or a two-value sequence")
    return (
        positive_finite_float64(values[0], label="cell_size x value"),
        positive_finite_float64(values[1], label="cell_size y value"),
    )


_CellSize = Annotated[tuple[float, float] | None, BeforeValidator(_cell_size)]


class TargetSupport(HydroForgeModel):
    """Target areas that consume values from a source grid."""

    target_ids: np.ndarray
    bounds: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    flat_indices: np.ndarray | None = None
    target_shape: (
        tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]] | None
    ) = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_ids", "flat_indices")
    @classmethod
    def _validate_ids(cls, value: np.ndarray | None, info: ValidationInfo):
        return None if value is None else canonical_ids(value, label=info.field_name)

    @field_validator("x", "y")
    @classmethod
    def _validate_center(cls, value: np.ndarray | None, info: ValidationInfo):
        if value is None:
            return None
        label = f"{info.field_name} target centers"
        if value.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        return canonical_float64(value, label=label)

    @field_validator("metadata")
    @classmethod
    def _freeze_metadata(cls, value: dict[str, Any]):
        return immutable_metadata(value, label="target metadata")

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        n_target = self.target_ids.size
        if _has_duplicates(self.target_ids):
            raise ValueError("target_ids must be unique")
        if self.bounds is not None:
            object.__setattr__(
                self,
                "bounds",
                canonical_float64(self.bounds, label="target bounds"),
            )
            if self.bounds.shape != (n_target, 4):
                raise ValueError(
                    f"bounds must have shape ({n_target}, 4), got {self.bounds.shape}"
                )
            if np.any(self.bounds[:, 0] >= self.bounds[:, 1]) or np.any(
                self.bounds[:, 2] >= self.bounds[:, 3]
            ):
                raise ValueError(
                    "target bounds must satisfy xmin < xmax and ymin < ymax"
                )
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y target centers must be provided together")
        if self.x is not None:
            if self.x.size != n_target or self.y.size != n_target:
                raise ValueError("target center size does not match target_ids")
        if (self.flat_indices is None) != (self.target_shape is None):
            raise ValueError("flat_indices and target_shape must be provided together")
        if self.target_shape is not None:
            if self.flat_indices.size != n_target:
                raise ValueError("flat_indices size does not match target_ids")
            if _has_duplicates(self.flat_indices):
                raise ValueError("flat_indices must be unique")
            extent = self.target_shape[0] * self.target_shape[1]
            if self.flat_indices.size and (
                self.flat_indices.min() < 0 or self.flat_indices.max() >= extent
            ):
                raise ValueError("flat_indices fall outside target_shape")
        for name in ("target_ids", "bounds", "x", "y", "flat_indices"):
            array = getattr(self, name)
            if array is not None:
                object.__setattr__(
                    self,
                    name,
                    immutable_array(array, order="C"),
                )
        return self

    @classmethod
    @validate_call(config=HydroForgeModel.model_config)
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
    ) -> Self:
        """Build one target per active cell of a boolean ``(y, x)`` grid mask."""

        if np.ma.isMaskedArray(mask) and np.any(np.ma.getmaskarray(mask)):
            raise ValueError("mask contains missing values")
        if np.asarray(mask).dtype != np.dtype(np.bool_):
            raise ValueError("mask must contain boolean values")
        grid = RegularGrid.from_coordinates(
            longitude,
            latitude,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        if mask.shape != grid._shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match grid shape {grid._shape}"
            )
        rows, columns = np.where(np.asarray(mask))
        flat_indices = np.ravel_multi_index((rows, columns), grid._shape).astype(
            np.int64
        )
        if target_ids is None:
            target_ids = flat_indices
        elif target_ids.size != flat_indices.size:
            raise ValueError("target_ids size does not match the active mask size")
        return cls(
            target_ids=target_ids,
            bounds=np.column_stack(
                (
                    grid.x_bounds[columns, 0],
                    grid.x_bounds[columns, 1],
                    grid.y_bounds[rows, 0],
                    grid.y_bounds[rows, 1],
                )
            ),
            x=grid.x[columns],
            y=grid.y[rows],
            flat_indices=flat_indices,
            target_shape=grid._shape,
            metadata={"kind": "regular_mask"},
        )

    @classmethod
    @validate_call(config=HydroForgeModel.model_config)
    def from_points(
        cls,
        longitude: np.ndarray,
        latitude: np.ndarray,
        *,
        target_ids: np.ndarray | None = None,
        cell_size: _CellSize = None,
    ) -> Self:
        """Build point targets from per-cell ``(longitude, latitude)`` centers.

        Each target is a single regular grid cell located at its center.  This
        is the support for models stored as a sparse 1D list of regular cells
        (e.g. VIC), as opposed to the CaMa MERIT sub-pixel scaffold.  Passing
        ``cell_size`` (scalar or ``(dx, dy)`` in the coordinate units) adds cell
        bounds so the ``overlap`` method can be used; without it only
        ``nearest`` is available.
        """

        if longitude.ndim != 1 or latitude.ndim != 1:
            raise ValueError("longitude and latitude must be one-dimensional")
        lon = canonical_float64(longitude, label="longitude")
        lat = canonical_float64(latitude, label="latitude")
        if lon.size != lat.size:
            raise ValueError("longitude and latitude must have the same length")
        bounds = None
        if cell_size is not None:
            dx, dy = cell_size
            bounds = np.column_stack(
                (
                    lon - 0.5 * dx,
                    lon + 0.5 * dx,
                    lat - 0.5 * dy,
                    lat + 0.5 * dy,
                )
            )
        return cls(
            target_ids=(
                np.arange(lon.size, dtype=np.int64)
                if target_ids is None
                else target_ids
            ),
            bounds=bounds,
            x=lon,
            y=lat,
            metadata={"kind": "points"},
        )

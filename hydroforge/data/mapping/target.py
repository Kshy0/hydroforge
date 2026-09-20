"""Target supports that consume values from a source grid.

A :class:`TargetSupport` is the destination geometry of a mapping: regular-grid
mask cells, per-cell points (e.g. VIC), or CaMa catchments reconstructed from a
``parameters.nc`` ``GridSpec`` annotation.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

import numpy as np
from pydantic import (
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.mapping.grid import RegularGrid
from hydroforge.data.numeric import (
    canonical_float64,
    canonical_ids,
    immutable_array,
    immutable_metadata,
    positive_finite_float64,
)


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
        if np.unique(self.target_ids).size != n_target:
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
            if np.unique(self.flat_indices).size != n_target:
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
        declaration = _TargetMaskDeclaration(
            longitude=longitude,
            latitude=latitude,
            mask=mask,
            target_ids=target_ids,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        grid = declaration.grid
        rows, cols = declaration.active_rows, declaration.active_columns
        return cls(
            target_ids=declaration.resolved_target_ids,
            bounds=declaration.resolved_bounds,
            x=grid.x[cols],
            y=grid.y[rows],
            flat_indices=declaration.flat_indices,
            target_shape=grid._shape,
            metadata={"kind": "regular_mask"},
        )

    @classmethod
    def _from_points(
        cls,
        longitude: np.ndarray,
        latitude: np.ndarray,
        *,
        target_ids: np.ndarray | None = None,
        cell_size: float | tuple[float, float] | None = None,
    ) -> Self:
        """Build point targets from per-cell ``(longitude, latitude)`` centers.

        Each target is a single regular grid cell located at its center.  This
        is the support for models stored as a sparse 1D list of regular cells
        (e.g. VIC), as opposed to the CaMa MERIT sub-pixel scaffold.  Passing
        ``cell_size`` (scalar or ``(dx, dy)`` in the coordinate units) adds cell
        bounds so the ``overlap`` method can be used; without it only
        ``nearest`` is available.
        """
        declaration = _TargetPointsDeclaration(
            longitude=longitude,
            latitude=latitude,
            target_ids=target_ids,
            cell_size=cell_size,
        )
        lon = declaration.longitude
        lat = declaration.latitude
        ids = (
            np.arange(lon.size, dtype=np.int64)
            if declaration.target_ids is None
            else declaration.target_ids
        )
        bounds = None
        if declaration.cell_size is not None:
            dx, dy = declaration.cell_size
            bounds = np.column_stack(
                (
                    lon - 0.5 * dx,
                    lon + 0.5 * dx,
                    lat - 0.5 * dy,
                    lat + 0.5 * dy,
                )
            )
        return cls(
            target_ids=ids,
            bounds=bounds,
            x=lon,
            y=lat,
            metadata={"kind": "points"},
        )


class _TargetMaskDeclaration(HydroForgeModel):
    """Validated declaration consumed by ``TargetSupport.from_mask``."""

    longitude: np.ndarray
    latitude: np.ndarray
    mask: np.ndarray
    target_ids: np.ndarray | None = None
    is_geographic: bool | None = None
    x_bounds: np.ndarray | None = None
    y_bounds: np.ndarray | None = None

    _grid: RegularGrid = PrivateAttr()
    _active_rows: np.ndarray = PrivateAttr()
    _active_columns: np.ndarray = PrivateAttr()
    _flat_indices: np.ndarray = PrivateAttr()
    _resolved_target_ids: np.ndarray = PrivateAttr()
    _resolved_bounds: np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def _validate_mask(self) -> Self:
        if np.ma.isMaskedArray(self.mask) and np.any(np.ma.getmaskarray(self.mask)):
            raise ValueError("mask contains missing values")
        mask = np.asarray(self.mask)
        if mask.dtype != np.dtype(np.bool_):
            raise ValueError("mask must contain boolean values")
        mask = np.array(mask, dtype=np.bool_, order="C", copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        self._grid = RegularGrid.from_coordinates(
            self.longitude,
            self.latitude,
            is_geographic=self.is_geographic,
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
        )
        if mask.shape != self._grid._shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match grid shape {self._grid._shape}"
            )
        rows, columns = np.where(mask)
        flat_indices = np.ravel_multi_index(
            (rows, columns),
            self._grid._shape,
        ).astype(np.int64)
        resolved_ids = flat_indices if self.target_ids is None else self.target_ids
        if resolved_ids.size != flat_indices.size:
            raise ValueError("target_ids size does not match the active mask size")
        self._active_rows = rows
        self._active_columns = columns
        self._flat_indices = flat_indices
        self._resolved_target_ids = resolved_ids
        self._resolved_bounds = np.column_stack(
            (
                self._grid.x_bounds[columns, 0],
                self._grid.x_bounds[columns, 1],
                self._grid.y_bounds[rows, 0],
                self._grid.y_bounds[rows, 1],
            )
        )
        return self

    @property
    def grid(self) -> RegularGrid:
        return self._grid

    @property
    def active_rows(self) -> np.ndarray:
        return self._active_rows

    @property
    def active_columns(self) -> np.ndarray:
        return self._active_columns

    @property
    def flat_indices(self) -> np.ndarray:
        return self._flat_indices

    @property
    def resolved_target_ids(self) -> np.ndarray:
        return self._resolved_target_ids

    @property
    def resolved_bounds(self) -> np.ndarray:
        return self._resolved_bounds


class _TargetPointsDeclaration(HydroForgeModel):
    """Validated declaration consumed by ``TargetSupport.from_points``."""

    longitude: np.ndarray
    latitude: np.ndarray
    target_ids: np.ndarray | None = None
    cell_size: Any = None

    @model_validator(mode="after")
    def _validate_points(self) -> Self:
        if self.longitude.ndim != 1 or self.latitude.ndim != 1:
            raise ValueError("longitude and latitude must be one-dimensional")
        longitude = canonical_float64(self.longitude, label="longitude")
        latitude = canonical_float64(self.latitude, label="latitude")
        if longitude.size != latitude.size:
            raise ValueError("longitude and latitude must have the same length")
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "latitude", latitude)
        if self.cell_size is None:
            return self
        if np.isscalar(self.cell_size):
            size = positive_finite_float64(
                self.cell_size,
                label="cell_size",
            )
            object.__setattr__(self, "cell_size", (size, size))
            return self
        try:
            values = tuple(self.cell_size)
        except TypeError as error:
            raise ValueError(
                "cell_size must be a real scalar or a two-value sequence"
            ) from error
        if len(values) != 2:
            raise ValueError("cell_size must be a scalar or a two-value sequence")
        object.__setattr__(
            self,
            "cell_size",
            (
                positive_finite_float64(
                    values[0],
                    label="cell_size x value",
                ),
                positive_finite_float64(
                    values[1],
                    label="cell_size y value",
                ),
            ),
        )
        return self

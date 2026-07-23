"""Rectilinear regular-grid geometry for spatial mapping.

The :class:`RegularGrid` describes a source or target rectilinear grid with
C-order ``(y, x)`` flattening, cell bounds, and a single point-to-cell index
routine (:meth:`RegularGrid.index_of_points`) reused across the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from netCDF4 import Dataset as NCDataset


_X_NAMES = ("lon", "longitude", "x")
_Y_NAMES = ("lat", "latitude", "y")


def _as_axis_names(names: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def _find_variable(ds: NCDataset, names: Sequence[str]) -> str:
    wanted = {name.lower() for name in names}
    for name in names:
        if name in ds.variables:
            return name
    for name in ds.variables:
        if name.lower() in wanted:
            return name
    raise ValueError(f"None of {tuple(names)!r} found in {Path(ds.filepath()).name}")


def _regular_axes(x_coord: np.ndarray, y_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_coord, dtype=np.float64)
    y_arr = np.asarray(y_coord, dtype=np.float64)
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return x_arr, y_arr
    if x_arr.ndim == 2 and y_arr.ndim == 2 and x_arr.shape == y_arr.shape:
        x_axis = x_arr[0, :]
        y_axis = y_arr[:, 0]
        if np.allclose(x_arr, x_axis[None, :]) and np.allclose(y_arr, y_axis[:, None]):
            return x_axis, y_axis
    raise ValueError("Coordinates are not a rectilinear regular grid")


def _validate_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).ravel()
    if axis.size == 0:
        raise ValueError(f"{name} axis must not be empty")
    if axis.size == 1:
        return axis
    diffs = np.diff(axis)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError(f"{name} axis must be strictly monotonic")
    step = diffs[0]
    if not np.allclose(diffs, step, rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} axis must be regularly spaced")
    return axis


def _validate_axis_bounds(bounds: np.ndarray, axis_size: int, name: str) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape == (2,) and axis_size == 1:
        arr = arr.reshape(1, 2)
    if arr.shape != (axis_size, 2):
        raise ValueError(f"{name} bounds must have shape ({axis_size}, 2), got {arr.shape}")
    out = np.column_stack((np.minimum(arr[:, 0], arr[:, 1]), np.maximum(arr[:, 0], arr[:, 1])))
    if np.any(out[:, 1] <= out[:, 0]):
        raise ValueError(f"{name} bounds must have positive widths")
    return out


def _axis_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64).ravel()
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate(([first], mids, [last])).astype(np.float64)


def _axis_bounds(values: np.ndarray) -> np.ndarray:
    edges = _axis_edges(values)
    return np.column_stack((np.minimum(edges[:-1], edges[1:]), np.maximum(edges[:-1], edges[1:])))


def _looks_geographic(x: np.ndarray, y: np.ndarray) -> bool:
    return (
        float(np.nanmin(y)) >= -90.0 and float(np.nanmax(y)) <= 90.0
        and float(np.nanmin(x)) >= -360.0 and float(np.nanmax(x)) <= 360.0
    )


def _wrap_longitude_like(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis_min = float(np.nanmin(axis))
    axis_max = float(np.nanmax(axis))
    if axis_min >= 0.0 and axis_max <= 360.0:
        return np.mod(values, 360.0)
    return np.mod(values + 180.0, 360.0) - 180.0


@dataclass
class RegularGrid:
    """A rectilinear regular grid with C-order flattening ``(y, x)``."""

    x: np.ndarray
    y: np.ndarray
    x_name: str = "lon"
    y_name: str = "lat"
    is_geographic: bool | None = None
    order: Literal["C"] = "C"
    x_bounds: np.ndarray | None = None
    y_bounds: np.ndarray | None = None
    x_edges: np.ndarray = field(init=False)
    y_edges: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x = _validate_axis(self.x, self.x_name)
        self.y = _validate_axis(self.y, self.y_name)
        if self.x_bounds is None:
            if self.x.size < 2:
                raise ValueError(f"{self.x_name} axis needs bounds when it has one cell")
            self.x_edges = _axis_edges(self.x)
            self.x_bounds = _axis_bounds(self.x)
        else:
            self.x_bounds = _validate_axis_bounds(self.x_bounds, self.x.size, self.x_name)
            self.x_edges = np.empty(0, dtype=np.float64)
        if self.y_bounds is None:
            if self.y.size < 2:
                raise ValueError(f"{self.y_name} axis needs bounds when it has one cell")
            self.y_edges = _axis_edges(self.y)
            self.y_bounds = _axis_bounds(self.y)
        else:
            self.y_bounds = _validate_axis_bounds(self.y_bounds, self.y.size, self.y_name)
            self.y_edges = np.empty(0, dtype=np.float64)
        if self.is_geographic is None:
            self.is_geographic = _looks_geographic(self.x, self.y)
        if self.order != "C":
            raise ValueError("Only C-order flattening is supported")

    @classmethod
    def from_coordinates(
        cls,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
        *,
        x_name: str = "lon",
        y_name: str = "lat",
        is_geographic: bool | None = None,
        x_bounds: np.ndarray | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> "RegularGrid":
        x_axis, y_axis = _regular_axes(x_coord, y_coord)
        return cls(
            x_axis,
            y_axis,
            x_name=x_name,
            y_name=y_name,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        *,
        x_names: str | Sequence[str] = _X_NAMES,
        y_names: str | Sequence[str] = _Y_NAMES,
        is_geographic: bool | None = None,
        x_bounds: np.ndarray | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> "RegularGrid":
        with NCDataset(str(path), "r") as ds:
            x_name = _find_variable(ds, _as_axis_names(x_names))
            y_name = _find_variable(ds, _as_axis_names(y_names))
            x_coord = np.asarray(ds.variables[x_name][:], dtype=np.float64)
            y_coord = np.asarray(ds.variables[y_name][:], dtype=np.float64)
        return cls.from_coordinates(
            x_coord,
            y_coord,
            x_name=x_name,
            y_name=y_name,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.size, self.x.size)

    @property
    def size(self) -> int:
        return int(self.x.size * self.y.size)

    @property
    def periodic_x(self) -> bool:
        if self.x.size < 2:
            return False
        dx = abs(float(self.x[1] - self.x[0]))
        return bool(self.is_geographic and np.isclose(dx * self.x.size, 360.0, atol=1e-6))

    def index_of_points(self, x_coord: np.ndarray, y_coord: np.ndarray, *, allow_oob: bool = False) -> np.ndarray:
        """Return flattened source indices for point coordinates."""
        x_raw = np.asarray(x_coord, dtype=np.float64)
        y_raw = np.asarray(y_coord, dtype=np.float64)
        if x_raw.shape != y_raw.shape:
            raise ValueError(f"x/y coordinate shape mismatch: {x_raw.shape} != {y_raw.shape}")

        x_val = x_raw.ravel()
        y_val = y_raw.ravel()
        if self.is_geographic:
            x_val = _wrap_longitude_like(x_val, self.x)

        if self.x.size == 1:
            ix = np.where(
                (x_val >= self.x_bounds[0, 0] - 1e-10) & (x_val <= self.x_bounds[0, 1] + 1e-10),
                0,
                -1,
            ).astype(np.int64)
        elif self.x[1] > self.x[0]:
            dx = abs(float(self.x[1] - self.x[0]))
            ix = np.floor((x_val - (self.x[0] - 0.5 * dx)) / dx).astype(np.int64)
        else:
            dx = abs(float(self.x[1] - self.x[0]))
            ix = np.floor(((self.x[0] + 0.5 * dx) - x_val) / dx).astype(np.int64)
        if self.y.size == 1:
            iy = np.where(
                (y_val >= self.y_bounds[0, 0] - 1e-10) & (y_val <= self.y_bounds[0, 1] + 1e-10),
                0,
                -1,
            ).astype(np.int64)
        elif self.y[1] > self.y[0]:
            dy = abs(float(self.y[1] - self.y[0]))
            iy = np.floor((y_val - (self.y[0] - 0.5 * dy)) / dy).astype(np.int64)
        else:
            dy = abs(float(self.y[1] - self.y[0]))
            iy = np.floor(((self.y[0] + 0.5 * dy) - y_val) / dy).astype(np.int64)

        if self.periodic_x:
            ix[ix == self.x.size] = 0
        else:
            ix[(ix == self.x.size) & (x_val <= self.x_bounds[:, 1].max() + 1e-10)] = self.x.size - 1
            ix[(ix == -1) & (x_val >= self.x_bounds[:, 0].min() - 1e-10)] = 0
        iy[(iy == self.y.size) & (y_val <= self.y_bounds[:, 1].max() + 1e-10)] = self.y.size - 1
        iy[(iy == -1) & (y_val >= self.y_bounds[:, 0].min() - 1e-10)] = 0

        valid = (ix >= 0) & (ix < self.x.size) & (iy >= 0) & (iy < self.y.size)
        out = np.full(ix.shape, -1, dtype=np.int64)
        out[valid] = iy[valid] * self.x.size + ix[valid]
        if not allow_oob and np.any(~valid):
            bad = int((~valid).sum())
            raise ValueError(f"{bad}/{out.size} points fall outside the source grid")
        return out.reshape(x_raw.shape)

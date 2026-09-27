"""Rectilinear grid geometry for spatial mapping.

The :class:`RegularGrid` describes a source or target rectilinear grid with
C-order ``(y, x)`` flattening, cell bounds, and a single point-to-cell index
routine (:meth:`RegularGrid.index_of_points`) reused across the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import numpy as np
from netCDF4 import Dataset as NCDataset
from pydantic import Field, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.numeric import canonical_float64, immutable_array

_X_NAMES = ("lon", "longitude", "x")
_Y_NAMES = ("lat", "latitude", "y")
# Longitude bound span tolerance for a periodic (global) axis: two float32
# roundings at 360 degrees.  Always capped below a quarter of the narrowest
# cell so a regional axis missing one cell is never treated as global.
_GLOBAL_LONGITUDE_ATOL = 2.0 * float(np.finfo(np.float32).eps) * 360.0


def _has_duplicates(values: np.ndarray) -> bool:
    """Exact duplicate test; a sort is much cheaper than ``np.unique`` here."""

    ordered = np.sort(values, axis=None)
    return bool(np.any(ordered[1:] == ordered[:-1]))


def _as_axis_names(names: str | Sequence[str]) -> tuple[str, ...]:
    if type(names) is str:
        result = (names,)
    elif type(names) in {tuple, list}:
        result = tuple(names)
    else:
        raise ValueError("coordinate names must be a string, tuple, or list")
    if not result:
        raise ValueError("coordinate names must not be empty")
    if any(type(name) is not str or not name for name in result):
        raise ValueError("coordinate names must contain non-empty exact strings")
    if len(set(result)) != len(result):
        raise ValueError("coordinate names must not contain duplicates")
    return result


def _find_variable(ds: NCDataset, names: Sequence[str]) -> str:
    exact = [name for name in names if name in ds.variables]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"Ambiguous coordinate variables {exact!r} in {Path(ds.filepath()).name}"
        )
    raise ValueError(f"None of {tuple(names)!r} found in {Path(ds.filepath()).name}")


def _regular_axes(
    x_coord: np.ndarray, y_coord: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Preserve the source dtype until validation so coordinate quantisation is
    # distinguished from a genuinely non-monotonic axis.
    for name, value in (("x", x_coord), ("y", y_coord)):
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"{name} coordinates contain missing values")
    x_arr = np.asarray(x_coord)
    y_arr = np.asarray(y_coord)
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return x_arr, y_arr
    if x_arr.ndim == 2 and y_arr.ndim == 2 and x_arr.shape == y_arr.shape:
        if x_arr.size == 0:
            raise ValueError("coordinate grids must not be empty")
        x_axis = x_arr[0, :]
        y_axis = y_arr[:, 0]
        if np.array_equal(
            x_arr,
            np.broadcast_to(x_axis, x_arr.shape),
        ) and np.array_equal(
            y_arr,
            np.broadcast_to(y_axis[:, None], y_arr.shape),
        ):
            return x_axis, y_axis
    raise ValueError("Coordinates are not a rectilinear regular grid")


def _validate_axis(values: np.ndarray, name: str) -> np.ndarray:
    source = np.asanyarray(values)
    if source.ndim != 1:
        raise ValueError(f"{name} axis must be one-dimensional")
    axis = canonical_float64(values, label=f"{name} axis")
    if axis.size == 0:
        raise ValueError(f"{name} axis must not be empty")
    if axis.size == 1:
        return axis
    with np.errstate(over="ignore", invalid="ignore"):
        diffs = np.diff(axis)
    if not np.isfinite(diffs).all():
        raise ValueError(f"{name} axis spacing exceeds float64 range")
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError(f"{name} axis must be strictly monotonic")
    return axis


def _bound_snap_tolerance(dtype: np.dtype, scale: float) -> float:
    """Largest seam mismatch treated as quantisation of shared cell edges."""

    if dtype.kind == "f" and dtype.itemsize <= 4:
        return 2.0 * float(np.finfo(dtype).eps) * scale
    return 64.0 * float(np.finfo(np.float64).eps) * scale


def _wrap_bounds_to_centers(
    arr: np.ndarray, axis: np.ndarray, period: float
) -> np.ndarray:
    return (
        axis[:, None]
        + np.mod(arr - axis[:, None] + 0.5 * period, period)
        - 0.5 * period
    )


def _raw_axis_bounds(
    bounds: np.ndarray, axis_size: int, name: str
) -> tuple[np.ndarray, np.dtype]:
    source_dtype = np.asanyarray(bounds).dtype
    arr = canonical_float64(bounds, label=f"{name} bounds")
    if arr.shape == (2,) and axis_size == 1:
        arr = arr.reshape(1, 2)
    if arr.shape != (axis_size, 2):
        raise ValueError(
            f"{name} bounds must have shape ({axis_size}, 2), got {arr.shape}"
        )
    return arr, source_dtype


def _bounds_are_periodic(bounds: np.ndarray) -> bool:
    """Whether ordered cell bounds span exactly one 360-degree period."""

    lower = np.minimum(bounds[:, 0], bounds[:, 1])
    upper = np.maximum(bounds[:, 0], bounds[:, 1])
    widths = upper - lower
    if widths.size == 0 or float(np.min(widths)) <= 0.0:
        return False
    span = float(np.max(upper) - np.min(lower))
    tolerance = min(_GLOBAL_LONGITUDE_ATOL, 0.25 * float(np.min(widths)))
    return abs(span - 360.0) <= tolerance


def _validate_axis_bounds(
    bounds: np.ndarray,
    axis: np.ndarray,
    name: str,
    *,
    period: float | None = None,
) -> np.ndarray:
    axis_size = axis.size
    arr, source_dtype = _raw_axis_bounds(bounds, axis_size, name)
    if period is not None:
        arr = _wrap_bounds_to_centers(arr, axis, period)
    out = np.column_stack(
        (np.minimum(arr[:, 0], arr[:, 1]), np.maximum(arr[:, 0], arr[:, 1]))
    )
    if np.any(out[:, 1] <= out[:, 0]):
        raise ValueError(f"{name} bounds must have positive widths")
    if axis_size > 1:
        ascending = axis[1] > axis[0]
        scale = max(
            float(np.max(np.abs(axis))),
            float(np.max(np.abs(out))),
            1.0 if period is None else period,
        )
        tolerance = _bound_snap_tolerance(source_dtype, scale)
        # Signed seam mismatch between each cell and its successor along the
        # axis: positive is a gap, negative an overlap.
        lower_cell, upper_cell = (
            (slice(None, -1), slice(1, None))
            if ascending
            else (slice(1, None), slice(None, -1))
        )
        seam = out[upper_cell, 0] - out[lower_cell, 1]
        if np.any(seam < -tolerance):
            raise ValueError(f"{name} bounds must not overlap")
        if np.any(seam > tolerance):
            index = int(np.argmax(seam))
            first, second = (index, index + 1) if ascending else (index + 1, index)
            raise ValueError(
                f"{name} bounds leave a gap of {float(seam[index]):.6g} between "
                f"cells {first} and {second} (tolerance {tolerance:.3g}); "
                "adjacent cell bounds must share their edges"
            )
        # Shared edges stored independently (for example float32 CF bounds)
        # can differ by rounding.  Snap them together so the cells tile the
        # axis without slivers that no point lookup or overlap can reach.
        snap = seam != 0.0
        if np.any(snap):
            edge = 0.5 * (out[upper_cell, 0] + out[lower_cell, 1])
            out[upper_cell, 0] = np.where(snap, edge, out[upper_cell, 0])
            out[lower_cell, 1] = np.where(snap, edge, out[lower_cell, 1])
            if np.any(out[:, 1] <= out[:, 0]):
                raise ValueError(f"{name} bounds must have positive widths")
    if np.any((axis < out[:, 0]) | (axis > out[:, 1])):
        raise ValueError(f"{name} bounds must contain their cell centers")
    if axis_size > 1:
        ordered = (
            np.all(np.diff(out[:, 0]) > 0.0) and np.all(np.diff(out[:, 1]) > 0.0)
            if ascending
            else np.all(np.diff(out[:, 0]) < 0.0) and np.all(np.diff(out[:, 1]) < 0.0)
        )
        if not ordered:
            raise ValueError(f"{name} bounds must follow the axis order")
    return out


def _index_axis_points(
    values: np.ndarray,
    bounds: np.ndarray,
    *,
    ascending: bool,
) -> np.ndarray:
    """Locate values in canonical non-overlapping cell bounds."""

    oriented = values if ascending else -values
    lower = bounds[:, 0] if ascending else -bounds[:, 1]
    upper = bounds[:, 1] if ascending else -bounds[:, 0]
    indices = np.searchsorted(lower, oriented, side="right") - 1
    candidates = np.clip(indices, 0, bounds.shape[0] - 1)
    valid = (
        np.isfinite(oriented)
        & (indices >= 0)
        & (indices < bounds.shape[0])
        & (oriented <= upper[candidates])
    )
    return np.where(valid, indices, -1).astype(np.int64)


def _axis_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64).ravel()
    # Divide before adding so a representable midpoint does not overflow when
    # two large, same-sign centers are added.  Extrapolated outer edges may
    # genuinely lie outside float64 and must be rejected rather than retained
    # as infinite cell bounds.
    with np.errstate(over="ignore", invalid="ignore"):
        mids = 0.5 * centers[:-1] + 0.5 * centers[1:]
        first = centers[0] + (centers[0] - mids[0])
        last = centers[-1] + (centers[-1] - mids[-1])
    edges = np.concatenate(([first], mids, [last])).astype(np.float64)
    if not np.isfinite(edges).all():
        raise ValueError("inferred axis bounds exceed float64 range")
    return edges


def _axis_bounds(values: np.ndarray) -> np.ndarray:
    edges = _axis_edges(values)
    return np.column_stack(
        (np.minimum(edges[:-1], edges[1:]), np.maximum(edges[:-1], edges[1:]))
    )


def _names_are_geographic(x_name: str, y_name: str) -> bool:
    return x_name.lower() in {"lon", "longitude"} and y_name.lower() in {
        "lat",
        "latitude",
    }


def _netcdf_axes_are_geographic(x_var, y_var) -> bool:
    if _names_are_geographic(x_var.name, y_var.name):
        return True
    x_standard = str(getattr(x_var, "standard_name", "")).lower()
    y_standard = str(getattr(y_var, "standard_name", "")).lower()
    if x_standard == "longitude" and y_standard == "latitude":
        return True
    x_units = str(getattr(x_var, "units", "")).lower()
    y_units = str(getattr(y_var, "units", "")).lower()
    return x_units in {
        "degree_east",
        "degrees_east",
        "degree_e",
        "degrees_e",
    } and y_units in {
        "degree_north",
        "degrees_north",
        "degree_n",
        "degrees_n",
    }


def _wrap_longitude_like(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    center = 0.5 * (float(np.min(axis)) + float(np.max(axis)))
    return values + 360.0 * np.floor((center - values) / 360.0 + 0.5)


class _RegularGridCoordinatesDeclaration(HydroForgeModel):
    """Validated declaration consumed by ``RegularGrid.from_coordinates``."""

    x_coord: np.ndarray
    y_coord: np.ndarray
    x_name: str = Field(default="lon", min_length=1)
    y_name: str = Field(default="lat", min_length=1)
    is_geographic: bool | None = None
    x_bounds: np.ndarray | None = None
    y_bounds: np.ndarray | None = None

    @model_validator(mode="after")
    def _validate_coordinates(self) -> Self:
        x_axis, y_axis = _regular_axes(self.x_coord, self.y_coord)
        object.__setattr__(self, "x_coord", x_axis)
        object.__setattr__(self, "y_coord", y_axis)
        return self


class _RegularGridNetCDFDeclaration(HydroForgeModel):
    """Validated declaration consumed by ``RegularGrid.from_netcdf``."""

    path: Path = Field(strict=False)
    x_names: str | tuple[str, ...] | list[str] = _X_NAMES
    y_names: str | tuple[str, ...] | list[str] = _Y_NAMES
    is_geographic: bool | None = None
    x_bounds: np.ndarray | None = None
    y_bounds: np.ndarray | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        object.__setattr__(self, "x_names", _as_axis_names(self.x_names))
        object.__setattr__(self, "y_names", _as_axis_names(self.y_names))
        return self


class _GridPointLookupDeclaration(HydroForgeModel):
    """Validated declaration consumed by ``RegularGrid.index_of_points``."""

    x_coord: np.ndarray
    y_coord: np.ndarray
    allow_oob: bool = False

    @model_validator(mode="after")
    def _validate_points(self) -> Self:
        x_coord = canonical_float64(
            self.x_coord,
            label="x point coordinates",
        )
        y_coord = canonical_float64(
            self.y_coord,
            label="y point coordinates",
        )
        if x_coord.shape != y_coord.shape:
            raise ValueError(
                f"x/y coordinate shape mismatch: {x_coord.shape} != {y_coord.shape}"
            )
        object.__setattr__(self, "x_coord", x_coord)
        object.__setattr__(self, "y_coord", y_coord)
        return self


class RegularGrid(HydroForgeModel):
    """A rectilinear regular grid with C-order flattening ``(y, x)``."""

    x: np.ndarray
    y: np.ndarray
    x_name: str = Field(default="lon", min_length=1)
    y_name: str = Field(default="lat", min_length=1)
    is_geographic: bool | None = None
    order: Literal["C"] = "C"
    x_bounds: np.ndarray | None = None
    y_bounds: np.ndarray | None = None

    @model_validator(mode="after")
    def _validate_grid(self) -> Self:
        object.__setattr__(self, "x", _validate_axis(self.x, self.x_name))
        object.__setattr__(self, "y", _validate_axis(self.y, self.y_name))
        if self.is_geographic is None:
            object.__setattr__(
                self,
                "is_geographic",
                _names_are_geographic(self.x_name, self.y_name),
            )
        if self.is_geographic and (np.any(self.y < -90.0) or np.any(self.y > 90.0)):
            raise ValueError(
                "geographic latitude coordinates must lie within [-90, 90]"
            )
        if self.x_bounds is None:
            if self.x.size < 2:
                raise ValueError(
                    f"{self.x_name} axis needs bounds when it has one cell"
                )
            object.__setattr__(
                self,
                "x_bounds",
                _validate_axis_bounds(_axis_bounds(self.x), self.x, self.x_name),
            )
        else:
            # Bounds of a global longitude axis may use another convention
            # than the centers (e.g. [-180, 180) edges for [0, 360) centers);
            # unwrap them around each center when that spans one period.
            periodic_x = (
                bool(self.is_geographic)
                and self.x.size > 1
                and _bounds_are_periodic(
                    _wrap_bounds_to_centers(
                        _raw_axis_bounds(self.x_bounds, self.x.size, self.x_name)[0],
                        self.x,
                        360.0,
                    )
                )
            )
            object.__setattr__(
                self,
                "x_bounds",
                _validate_axis_bounds(
                    self.x_bounds,
                    self.x,
                    self.x_name,
                    period=360.0 if periodic_x else None,
                ),
            )
        if self.y_bounds is None:
            if self.y.size < 2:
                raise ValueError(
                    f"{self.y_name} axis needs bounds when it has one cell"
                )
            inferred_y_bounds = _axis_bounds(self.y)
            if self.is_geographic:
                # Some operational latitude axes (notably ERA5-Land) include
                # grid points exactly at both poles. Their nearest-neighbour
                # support ends at the pole instead of half a cell beyond it.
                inferred_y_bounds = np.clip(
                    inferred_y_bounds,
                    -90.0,
                    90.0,
                )
            object.__setattr__(
                self,
                "y_bounds",
                _validate_axis_bounds(inferred_y_bounds, self.y, self.y_name),
            )
        else:
            object.__setattr__(
                self,
                "y_bounds",
                _validate_axis_bounds(
                    self.y_bounds,
                    self.y,
                    self.y_name,
                ),
            )
        if self.is_geographic and (
            np.any(self.y_bounds < -90.0) or np.any(self.y_bounds > 90.0)
        ):
            raise ValueError("geographic latitude bounds must lie within [-90, 90]")
        for name in ("x", "y", "x_bounds", "y_bounds"):
            object.__setattr__(
                self,
                name,
                immutable_array(getattr(self, name), order="C"),
            )
        return self

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
    ) -> Self:
        declaration = _RegularGridCoordinatesDeclaration(
            x_coord=x_coord,
            y_coord=y_coord,
            x_name=x_name,
            y_name=y_name,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        return cls(
            x=declaration.x_coord,
            y=declaration.y_coord,
            x_name=declaration.x_name,
            y_name=declaration.y_name,
            is_geographic=declaration.is_geographic,
            x_bounds=declaration.x_bounds,
            y_bounds=declaration.y_bounds,
        )

    @classmethod
    def _from_netcdf(
        cls,
        path: str | Path,
        *,
        x_names: str | Sequence[str] = _X_NAMES,
        y_names: str | Sequence[str] = _Y_NAMES,
        is_geographic: bool | None = None,
        x_bounds: np.ndarray | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> Self:
        declaration = _RegularGridNetCDFDeclaration(
            path=path,
            x_names=x_names,
            y_names=y_names,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        with NCDataset(str(declaration.path), "r") as ds:
            x_name = _find_variable(ds, declaration.x_names)
            y_name = _find_variable(ds, declaration.y_names)
            x_var = ds.variables[x_name]
            y_var = ds.variables[y_name]
            x_coord = x_var[:]
            y_coord = y_var[:]
            x_bounds = (
                declaration.x_bounds
                if declaration.x_bounds is not None
                else cls._netcdf_coordinate_bounds(ds, x_var, x_name)
            )
            y_bounds = (
                declaration.y_bounds
                if declaration.y_bounds is not None
                else cls._netcdf_coordinate_bounds(ds, y_var, y_name)
            )
            is_geographic = declaration.is_geographic
            if is_geographic is None:
                is_geographic = _netcdf_axes_are_geographic(x_var, y_var)
        return cls.from_coordinates(
            x_coord,
            y_coord,
            x_name=x_name,
            y_name=y_name,
            is_geographic=is_geographic,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

    @staticmethod
    def _netcdf_coordinate_bounds(
        dataset: NCDataset,
        coordinate,
        coordinate_name: str,
    ) -> np.ndarray | None:
        """Read CF cell bounds declared by a one-dimensional coordinate."""

        if len(coordinate.dimensions) != 1:
            return None
        declared = getattr(coordinate, "bounds", None)
        bounds_name = (
            declared.strip()
            if isinstance(declared, str) and declared.strip()
            else f"{coordinate_name}_bnds"
        )
        if bounds_name not in dataset.variables:
            return None
        bounds = dataset.variables[bounds_name]
        raw = bounds[:]
        if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
            raise ValueError(
                f"coordinate bounds {bounds_name!r} contains missing values"
            )
        values = np.asarray(raw)
        expected_shape = (coordinate.shape[0], 2)
        if values.shape != expected_shape:
            raise ValueError(
                f"coordinate bounds {bounds_name!r} must have shape "
                f"{expected_shape}, got {values.shape}"
            )
        return values

    @property
    def _shape(self) -> tuple[int, int]:
        return (self.y.size, self.x.size)

    @property
    def _size(self) -> int:
        return int(self.x.size * self.y.size)

    @property
    def _periodic_x(self) -> bool:
        # The contiguous bound span, not the center spacing, decides
        # periodicity: rectilinear axes need not be uniformly spaced.
        return bool(self.is_geographic) and _bounds_are_periodic(self.x_bounds)

    def _index_of_points(
        self, x_coord: np.ndarray, y_coord: np.ndarray, *, allow_oob: bool = False
    ) -> np.ndarray:
        """Return flattened source indices for point coordinates."""
        declaration = _GridPointLookupDeclaration(
            x_coord=x_coord,
            y_coord=y_coord,
            allow_oob=allow_oob,
        )
        return self._index_of_points_trusted(
            declaration.x_coord,
            declaration.y_coord,
            allow_oob=declaration.allow_oob,
        )

    def _x_indices_trusted(self, x_val: np.ndarray) -> np.ndarray:
        """Elementwise x-cell index of canonical float64 values, or -1."""

        periodic_x = self._periodic_x
        normalize_longitude = bool(self.is_geographic) and (
            periodic_x or bool(np.min(self.x) < -180.0 or np.max(self.x) > 180.0)
        )
        if normalize_longitude:
            x_val = _wrap_longitude_like(x_val, self.x)

        x_ascending = self.x.size == 1 or self.x[1] > self.x[0]
        ix = _index_axis_points(
            x_val,
            self.x_bounds,
            ascending=x_ascending,
        )
        if periodic_x:
            for shift in (-360.0, 360.0):
                missing = ix < 0
                if not np.any(missing):
                    break
                ix[missing] = _index_axis_points(
                    x_val[missing] + shift,
                    self.x_bounds,
                    ascending=x_ascending,
                )
        return ix

    def _y_indices_trusted(self, y_val: np.ndarray) -> np.ndarray:
        """Elementwise y-cell index of canonical float64 values, or -1."""

        return _index_axis_points(
            y_val,
            self.y_bounds,
            ascending=self.y.size == 1 or self.y[1] > self.y[0],
        )

    def _index_of_points_trusted(
        self, x_raw: np.ndarray, y_raw: np.ndarray, *, allow_oob: bool
    ) -> np.ndarray:
        """Locate canonical equal-shaped coordinates from a validated request."""

        ix = self._x_indices_trusted(x_raw.ravel())
        iy = self._y_indices_trusted(y_raw.ravel())

        # Axis lookup returns either a valid cell index or -1.
        valid = (ix >= 0) & (iy >= 0)
        out = np.full(ix.shape, -1, dtype=np.int64)
        out[valid] = iy[valid] * self.x.size + ix[valid]
        if not allow_oob and np.any(~valid):
            bad = int((~valid).sum())
            raise ValueError(f"{bad}/{out.size} points fall outside the source grid")
        return out.reshape(x_raw.shape)

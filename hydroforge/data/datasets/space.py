"""Spatial identities of forcing datasets: regular grids and ID points."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hydroforge.core.arrays import (
    exact_numeric_array_equal,
    find_indices_in,
    immutable_array,
)


def _require_same(
    reference: tuple[np.ndarray, ...],
    observed: tuple[np.ndarray, ...],
    *,
    label: str,
) -> None:
    """Accept identical axes; name an order-only difference explicitly."""

    if all(
        exact_numeric_array_equal(left, right)
        for left, right in zip(reference, observed, strict=True)
    ):
        return
    if all(
        left.shape == right.shape
        and exact_numeric_array_equal(np.sort(left), np.sort(right))
        for left, right in zip(reference, observed, strict=True)
    ):
        raise ValueError(f"{label} uses a different spatial coordinate order")
    raise ValueError(f"{label} uses a different spatial domain")


@dataclass(frozen=True, slots=True)
class GridSpace:
    """A regular ``(latitude, longitude)`` grid, optionally reduced to a selection.

    ``selection`` holds flat ``y * nx + x`` source positions in mapping column
    order; values then have one column per selected cell.
    """

    longitude: np.ndarray
    latitude: np.ndarray
    longitude_bounds: np.ndarray | None = None
    latitude_bounds: np.ndarray | None = None
    selection: np.ndarray | None = None
    longitude_units: str | None = None
    latitude_units: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.latitude.size, self.longitude.size)

    @property
    def size(self) -> int:
        if self.selection is not None:
            return int(self.selection.size)
        return self.latitude.size * self.longitude.size

    def select(self, indices: np.ndarray) -> GridSpace:
        """Return this grid reduced to flat source ``indices``."""

        if np.ma.isMaskedArray(indices):
            raise TypeError("grid selection must not be a masked array")
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError("grid selection requires a one-dimensional integer array")
        if (
            indices.size
            and indices.dtype.kind == "u"
            and int(indices.max()) > np.iinfo(np.int64).max
        ):
            raise OverflowError("grid selection exceeds int64 range")
        cells = self.latitude.size * self.longitude.size
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= cells):
            raise ValueError(f"grid selection exceeds the {cells} source cells")
        return replace(
            self, selection=immutable_array(indices, dtype=np.int64, order="C")
        )

    def require_compatible(self, other: GridSpace, *, label: str) -> None:
        """Require the same grid and the same selected cells."""

        for axis in ("longitude_units", "latitude_units"):
            left, right = getattr(self, axis), getattr(other, axis)
            if left != right:
                raise ValueError(f"{label} uses different coordinate units ({axis})")
        _require_same(
            (self.longitude, self.latitude),
            (other.longitude, other.latitude),
            label=label,
        )
        for name in ("longitude_bounds", "latitude_bounds"):
            left, right = getattr(self, name), getattr(other, name)
            if (left is None) != (right is None) or (
                left is not None and not exact_numeric_array_equal(left, right)
            ):
                raise ValueError(f"{label} uses different spatial bounds ({name})")
        if (self.selection is None) != (other.selection is None) or (
            self.selection is not None
            and not np.array_equal(self.selection, other.selection)
        ):
            raise ValueError(f"{label} uses a different spatial selection")


@dataclass(frozen=True, slots=True)
class PointSpace:
    """Unique integer point IDs in storage order, optionally reordered."""

    ids: np.ndarray
    selection: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.ids.size if self.selection is None else self.selection.size)

    @property
    def selected_ids(self) -> np.ndarray:
        return self.ids if self.selection is None else self.ids[self.selection]

    def select(self, target_ids: np.ndarray, *, label: str) -> PointSpace:
        """Return the storage positions of ``target_ids`` in their order."""

        positions = find_indices_in(target_ids, self.ids)
        missing = int(np.count_nonzero(positions == -1))
        if missing:
            raise ValueError(f"{missing} requested IDs were not found in {label}")
        return replace(
            self, selection=immutable_array(positions, dtype=np.int64, order="C")
        )

    def require_compatible(self, other: PointSpace, *, label: str) -> None:
        """Require the same selected IDs in the same order."""

        _require_same((self.selected_ids,), (other.selected_ids,), label=label)

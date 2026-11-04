# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Grid axes and the spatial extent of statistics outputs.

A field's logical axes are named: the leading axis of a field declaring
``dim_coords`` is its coordinate (rows identified by integer IDs), any other
symbolic axis takes the name its model declares in ``grid_axes`` (or the
symbol itself), and a literal axis is ``dim_{position}``. Outputs select by
global identity, never by a rank's local position: coordinate rows through
their ``SelectionField`` (IDs), grid axes through an :class:`OutputView` of
global logical indices (:class:`Range`, :class:`Indices`). A staggered axis
not named by a view follows the axis it staggers: the faces bounding a
selected unit-step range, otherwise the same indices.

Views combine with ``|`` into an :class:`OutputUnion`, the distinct cells
any of them selects: ``OutputView(y=Range(10, 20), x=Range(5, 9)) |
OutputView(points={"y": [3, 40], "x": [7, 2]}) | OutputView(y=Indices([60]))``
is a box, two points and row 60, each cell once.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import Field, model_validator

from hydroforge.core.naming import Identifier
from hydroforge.core.validation import FrozenMapping, HydroForgeModel


class GridAxis(HydroForgeModel):
    """The name of a symbolic axis and the axis it staggers, if any.

    ``staggers`` names the cell axis whose faces this axis holds (one more
    entry than cells), so a view of the cell axis also bounds this one.
    """

    name: Identifier
    staggers: Identifier | None = None


class Range(HydroForgeModel):
    """Global logical indices ``start, start + step, ...`` below ``stop``.

    Spelled like ``range``; ``stop=None`` runs to the end of the axis.
    """

    start: int = Field(default=0, ge=0)
    stop: int | None = Field(default=None, ge=0)
    step: int = Field(default=1, ge=1)

    def __init__(self, *bounds: int | None, **fields: Any) -> None:
        # Spelled like range(): Range(stop), Range(start, stop[, step]).
        if len(bounds) > 3 or (bounds and fields):
            raise TypeError("Range takes (stop), (start, stop[, step]) or keywords")
        if len(bounds) == 1:
            fields["stop"] = bounds[0]
        elif bounds:
            fields.update(zip(("start", "stop", "step"), bounds))
        super().__init__(**fields)

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.stop is not None and self.stop <= self.start:
            raise ValueError(f"empty range [{self.start}, {self.stop})")
        return self

    def indices(self, extent: int) -> range:
        stop = extent if self.stop is None else self.stop
        if stop > extent:
            raise ValueError(f"range stop {stop} exceeds the axis extent {extent}")
        return range(self.start, stop, self.step)


class Indices(HydroForgeModel):
    """Explicit, strictly increasing global logical indices."""

    values: tuple[int, ...] = Field(min_length=1)

    def __init__(self, values: Any = None, /, **fields: Any) -> None:
        if values is not None:
            fields["values"] = tuple(values)
        super().__init__(**fields)

    @model_validator(mode="after")
    def _validate_indices(self) -> Self:
        values = self.values
        if values[0] < 0 or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("indices must be non-negative and strictly increasing")
        return self

    def indices(self, extent: int) -> tuple[int, ...]:
        if self.values[-1] >= extent:
            raise ValueError(
                f"index {self.values[-1]} exceeds the axis extent {extent}"
            )
        return self.values


Selector = Range | Indices


class OutputView(HydroForgeModel):
    """Selected global indices of named grid axes; others are complete.

    ``OutputView(k=Range(0, 10), west_east=Indices([4, 8]))``. Coordinate
    rows are selected by ID through their ``SelectionField``, not here.

    ``OutputView(points={"y": [4, 1], "x": [2, 8]})`` instead samples the
    paired grid points (4, 2), (1, 8), in that order, into ``point_dimension``.
    A point view names every logical grid axis of its output. Repeated points
    remain repeated; ranks save only their owned points, indexed by their
    position in the complete point list.
    """

    axes: FrozenMapping[Identifier, Selector] = Field(default_factory=dict)
    points: FrozenMapping[Identifier, tuple[int, ...]] = Field(default_factory=dict)
    point_dimension: Identifier = "point"

    def __init__(
        self, axes: Mapping[str, Selector] | None = None, *,
        points: Mapping[str, Any] | None = None, point_dimension: str = "point",
        **named: Any,
    ):
        if axes is not None and named:
            raise ValueError("pass axes as a mapping or as keywords, not both")
        super().__init__(
            axes=dict(axes if axes is not None else named),
            points={} if points is None else {
                axis: tuple(indices) for axis, indices in points.items()
            },
            point_dimension=point_dimension,
        )

    @model_validator(mode="after")
    def _validate_points(self) -> Self:
        if self.points:
            if self.axes:
                raise ValueError("a view selects axis ranges or paired points, not both")
            sizes = {len(indices) for indices in self.points.values()}
            if len(sizes) != 1 or 0 in sizes:
                raise ValueError("point axes must have equal nonzero lengths")
            if any(index < 0 for indices in self.points.values() for index in indices):
                raise ValueError("point indices must be non-negative")
        return self

    def __or__(self, other: OutputView | OutputUnion) -> OutputUnion:
        return OutputUnion(parts=(self,)) | other


class OutputUnion(HydroForgeModel):
    """The distinct grid cells any of ``parts`` selects, written along one
    point dimension in the row-major order of the output's axes.

    A part is a box (axis ranges; axes it does not name are complete) or
    paired points (an axis they do not name is complete for each point). As
    a point view, a union names every grid axis of its output. The point
    dimension's coordinate is each cell's flat row-major index on the global
    grid, with its index on every axis beside it; each rank writes the cells
    it owns, so ranks' files join by that coordinate. ``point_dimension`` is
    that of the first point view among the parts, else ``"point"``.
    """

    parts: tuple[OutputView, ...] = Field(min_length=1)
    point_dimension: Identifier = "point"

    @model_validator(mode="before")
    @classmethod
    def _name_dimension(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "point_dimension" not in data:
            named = next(
                (
                    part
                    for part in data.get("parts", ())
                    if isinstance(part, OutputView) and part.points
                ),
                None,
            )
            if named is not None:
                data = {**data, "point_dimension": named.point_dimension}
        return data

    def __or__(self, other: OutputView | OutputUnion) -> OutputUnion:
        added = other.parts if isinstance(other, OutputUnion) else (other,)
        return OutputUnion(parts=(*self.parts, *added))

    @property
    def axes(self) -> tuple[str, ...]:
        """Every axis a part names, in the order parts first name them."""

        return tuple(
            dict.fromkeys(
                axis for part in self.parts for axis in (*part.axes, *part.points)
            )
        )


class Out(HydroForgeModel):
    """One statistics output: a field or alias, its expression and view.

    ``Out("u", view="box")`` or ``Out("t2c", expression="t2 - 273.15",
    view="box")`` inside ``OutputConfig.variables``; ``view`` names an entry
    of ``OutputConfig.views``.
    """

    name: str = Field(min_length=1)
    expression: str | None = Field(default=None, min_length=1)
    view: Identifier | None = None

    def __init__(self, name: Any = None, /, **fields: Any) -> None:
        if name is not None:
            fields["name"] = name
        super().__init__(**fields)


def view_indices(
    axes: tuple[str, ...],
    extents: tuple[int, ...],
    view: OutputView | None,
    grid_axes: Mapping[str, GridAxis],
) -> tuple[tuple[int, ...] | None, ...]:
    """Selected global indices of every axis; ``None`` keeps an axis whole.

    A staggered axis the view does not name follows its cell axis.
    """

    if view is None:
        return (None,) * len(axes)
    staggers = {axis.name: axis.staggers for axis in grid_axes.values()}
    result: list[tuple[int, ...] | None] = []
    for axis, extent in zip(axes, extents, strict=True):
        selector = view.axes.get(axis)
        cell = staggers.get(axis)
        if selector is None and cell is not None and cell in view.axes:
            selector = view.axes[cell]
            if isinstance(selector, Range) and selector.step == 1:
                # The faces bounding the selected cells.
                stop = None if selector.stop is None else selector.stop + 1
                selector = Range(start=selector.start, stop=stop)
        result.append(None if selector is None else tuple(selector.indices(extent)))
    return tuple(result)


__all__ = [
    "GridAxis",
    "Indices",
    "Out",
    "OutputUnion",
    "OutputView",
    "Range",
    "Selector",
]

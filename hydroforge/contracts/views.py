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
    """

    axes: FrozenMapping[Identifier, Selector] = Field(default_factory=dict)

    def __init__(self, axes: Mapping[str, Selector] | None = None, /, **named: Any):
        if axes is not None and named:
            raise ValueError("pass axes as a mapping or as keywords, not both")
        super().__init__(axes=dict(axes if axes is not None else named))


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


__all__ = ["GridAxis", "Indices", "Out", "OutputView", "Range", "Selector"]

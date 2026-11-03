# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Which fields hold rows of a model's slab-decomposed axis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from hydroforge.compiler.partition import bare
from hydroforge.contracts.slab import CELL, FACE, INTERIOR, SlabAxis
from hydroforge.contracts.views import GridAxis

if TYPE_CHECKING:
    from hydroforge.compiler.fields import FieldEntry


@dataclass(frozen=True, slots=True)
class SlabField:
    """Where a field's slab axis sits: ``position`` in its logical shape and
    its ``kind`` (cell rows, staggered faces or interior rows); ``halo``:
    whether halo exchanges carry it (``TensorField(halo=True)``)."""

    position: int
    kind: str
    halo: bool = False

    @property
    def face(self) -> bool:
        return self.kind == FACE


def slab_tokens(slab: SlabAxis, grid_axes: Mapping[str, GridAxis]) -> dict[str, str]:
    """Symbolic dimensions of the slab axis and their kind: the cell rows,
    the axes staggering them (one more entry) and the interior axes."""

    tokens = {slab.dim: CELL}
    cell = grid_axes.get(slab.dim)
    if cell is not None:
        for symbol, axis in grid_axes.items():
            if axis.staggers == cell.name:
                tokens[symbol] = FACE
    tokens.update(dict.fromkeys(slab.interior, INTERIOR))
    return tokens


@dataclass(frozen=True, slots=True)
class LocatedDim:
    """Entities of one dimension, located on a grid by a ``grid_index`` table.

    ``index`` (a qualified field name) holds each entity's flat row-major
    index into the grid of dimensions ``grid``; every field naming ``dim``
    describes the same entities. ``axis`` and ``kind`` place the slab axis
    in ``grid`` (``None`` without a slab axis there): on several ranks each
    keeps the entities in the rows it owns.
    """

    dim: str
    index: str
    grid: tuple[str, ...]
    axis: int | None = None
    kind: str | None = None


def compile_located(
    entries: Iterable[FieldEntry],
    slab: SlabAxis | None,
    grid_axes: Mapping[str, GridAxis],
) -> Mapping[str, LocatedDim]:
    """The located entity dimensions of the active fields, by dimension."""

    tokens = {} if slab is None else slab_tokens(slab, grid_axes)
    found: dict[str, LocatedDim] = {}
    for entry in entries:
        grid = tuple(bare(item) for item in entry.tensor.grid_index)
        if not grid:
            continue
        dim = bare(entry.tensor.shape[0])
        if dim in found:
            raise ValueError(
                f"entities {dim!r} are located by both {found[dim].index!r} and "
                f"{entry.qualified!r}; one grid_index table locates a dimension"
            )
        if dim in tokens or dim in grid:
            raise ValueError(
                f"{entry.qualified!r} locates entities of a grid dimension {dim!r}"
            )
        hits = [position for position, item in enumerate(grid) if item in tokens]
        if slab is not None and len(hits) != 1:
            raise ValueError(
                f"{entry.qualified!r} locates entities on grid {grid}, which must "
                f"name the slab axis {slab.dim!r} (or an axis of it) exactly once"
            )
        axis = hits[0] if hits else None
        found[dim] = LocatedDim(
            dim,
            entry.qualified,
            grid,
            axis,
            None if axis is None else tokens[grid[axis]],
        )
    return MappingProxyType(found)


def compile_slab(
    entries: Iterable[FieldEntry],
    slab: SlabAxis | None,
    grid_axes: Mapping[str, GridAxis],
) -> Mapping[str, SlabField]:
    """Slab axis of every active field that names it, by qualified name."""

    if slab is None:
        return MappingProxyType({})
    tokens = slab_tokens(slab, grid_axes)
    found: dict[str, SlabField] = {}
    for entry in entries:
        shape = [
            bare(item) if isinstance(item, str) else item for item in entry.tensor.shape
        ]
        hits = [
            (position, tokens[item])
            for position, item in enumerate(shape)
            if item in tokens
        ]
        if len(hits) > 1:
            raise ValueError(
                f"field {entry.qualified!r} names the slab axis {slab.dim!r} "
                f"in more than one dimension of {tuple(entry.tensor.shape)!r}"
            )
        if hits:
            found[entry.qualified] = SlabField(*hits[0], entry.tensor.halo)
    if not found:
        raise ValueError(
            f"slab axis {slab.dim!r} is named by no field of the opened modules"
        )
    return MappingProxyType(found)

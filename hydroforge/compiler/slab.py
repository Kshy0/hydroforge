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

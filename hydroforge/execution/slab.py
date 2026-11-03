# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Rank-local rows of a model's slab-decomposed axis, and the entities
located in them."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from hydroforge.compiler.partition import bare
from hydroforge.contracts.slab import CELL, FACE, INTERIOR
from hydroforge.core.events import emit
from hydroforge.parallel.slab import Slab

if TYPE_CHECKING:
    from hydroforge.compiler.plan import ModelPlan
    from hydroforge.compiler.slab import LocatedDim
    from hydroforge.execution.inputs import InputBinding
    from hydroforge.execution.session import ModelRuntime
    from hydroforge.parallel.topology import Topology


def build_slab(
    plan: ModelPlan, source: InputBinding, topology: Topology
) -> Slab | None:
    """This rank's rows, from the row count every input holding the axis
    agrees on (``None`` for a model without a slab axis). Blocks follow the
    ranks' node order, so neighbouring blocks share a node where they can."""

    declared = plan.spec.slab
    if declared is None:
        return None
    if plan.slab_periodic and declared.interior and plan.spatial_world_size > 1:
        # A periodic axis has no perimeter rows: the interior rows a rank
        # reads would not be the rows it stores and saves.
        raise ValueError(
            f"slab axis {declared.dim!r} is periodic here, so it cannot have "
            f"interior axes {declared.interior} on several ranks"
        )
    offset = {CELL: 0, FACE: -1, INTERIOR: 2}
    extents: dict[int, list[str]] = {}
    for name, (axis, kind) in source.slab_axes.items():
        rows = source.get_var_shape(name)[axis] + offset[kind]
        extents.setdefault(rows, []).append(name)
    if not extents:
        raise ValueError(
            f"no input holds rows of the slab axis {declared.dim!r}; the model "
            "needs at least one of its fields as construction input"
        )
    if len(extents) > 1:
        detail = "; ".join(
            f"{rows} rows: {sorted(names)[:3]}" for rows, names in extents.items()
        )
        raise ValueError(
            f"inputs disagree on the rows of slab axis {declared.dim!r} ({detail}); "
            "a staggered axis holds one more entry than the cells"
        )
    (total,) = extents
    depth = declared.halo
    if isinstance(depth, str):
        depth = getattr(plan.options, depth)
        if type(depth) is not int or depth < 1:
            raise ValueError("the slab halo option must resolve to a positive integer")
    return Slab.build(
        plan.spatial_rank,
        plan.spatial_world_size,
        total,
        depth,
        plan.slab_periodic,
        topology.positions()[plan.spatial_rank],
    )


@dataclass(frozen=True, slots=True)
class LocatedEntities:
    """This rank's share of the entities of one located dimension.

    ``positions`` are their places in the complete table (increasing, so
    the rank keeps their order), of ``total`` entities. ``index`` (the
    locating field) holds flat indices of this rank's stored grid,
    ``stored``; the complete table indexes the global grid, ``grid``, whose
    slab ``axis`` row is the stored row plus ``offset``.
    """

    dim: str
    index: str
    positions: np.ndarray
    total: int
    grid: tuple[int, ...]
    stored: tuple[int, ...]
    axis: int
    offset: int

    def global_index(self, local: Any) -> np.ndarray:
        """The global flat indices of stored-grid flat indices ``local``."""

        coords = list(np.unravel_index(np.asarray(local, dtype=np.int64), self.stored))
        coords[self.axis] = coords[self.axis] + self.offset
        return np.ravel_multi_index(coords, self.grid).astype(np.int64)


def _shape_tokens(shape: tuple[Any, ...]) -> list[Any]:
    return [bare(item) if isinstance(item, str) else item for item in shape]


def _grid_extents(source: InputBinding, located: LocatedDim) -> tuple[int, ...]:
    """The global extent of every grid dimension, from an input naming it."""

    extents = []
    for dim in located.grid:
        for name, spec in source.fields.items():
            tensor = spec.field.tensor
            if tensor is None or name not in source:
                continue
            tokens = _shape_tokens(tensor.shape)
            if dim in tokens:
                shape = source.get_var_shape(name)
                extents.append(int(shape[source.axes[name] + tokens.index(dim)]))
                break
        else:
            raise ValueError(
                f"{located.index!r} locates entities on grid {located.grid}, "
                f"but no input names its dimension {dim!r}"
            )
    return tuple(extents)


def cut_located(runtime: ModelRuntime, values: Any) -> Any:
    """Keep this rank's entities of every located dimension in ``values``.

    The complete tables (what ``prepare_model_input`` saw) index the global
    grid. Each entity belongs to the rank owning its row of the slab axis:
    every input naming its dimension keeps the rank's entities, in order,
    and the locating field is renumbered to the rank's stored grid. A rank
    may keep none. Without several ranks only the indices are checked.
    """

    runtime.located = {}
    declared = runtime.plan.fields.located
    if not declared or not isinstance(values, Mapping):
        return values
    values = dict(values)
    source = runtime.input
    slab = runtime.slab
    kept = {}
    for dim, located in declared.items():
        name = bare(located.index)
        index = values.get(name)
        if index is None:
            continue
        grid = _grid_extents(source, located)
        flat = index.detach().reshape(-1).to("cpu", torch.int64).numpy()
        cells = math.prod(grid)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= cells):
            raise ValueError(
                f"{name} holds indices outside its grid {located.grid} of {cells} cells"
            )
        if slab is None or slab.world == 1 or located.axis is None:
            continue
        coords = list(np.unravel_index(flat, grid))
        owned = slab.global_rows(located.kind)
        first = int(owned[0]) if owned.size else 0
        rows = coords[located.axis]
        positions = np.flatnonzero((rows >= first) & (rows < first + owned.size))
        stored = list(grid)
        stored[located.axis] = int(slab.rows_of(located.kind).size)
        offset = first - slab.owned_slice(located.kind).start
        coords = [coord[positions] for coord in coords]
        coords[located.axis] = coords[located.axis] - offset
        selected = torch.as_tensor(positions, device=index.device)
        for field, value in values.items():
            spec = source.fields.get(field)
            if (
                spec is None
                or spec.field.tensor is None
                or not isinstance(value, torch.Tensor)
            ):
                continue
            tokens = _shape_tokens(spec.field.tensor.shape)
            if dim in tokens:
                axis = value.ndim - len(tokens) + tokens.index(dim)
                values[field] = value.index_select(axis, selected)
        values[name] = torch.as_tensor(
            np.ravel_multi_index(coords, stored), dtype=index.dtype
        ).to(index.device)
        runtime.located[dim] = LocatedEntities(
            dim,
            name,
            positions,
            int(flat.size),
            grid,
            tuple(stored),
            located.axis,
            offset,
        )
        kept[dim] = (int(positions.size), int(flat.size))
    if kept:
        emit(
            runtime,
            "info",
            "model.data_located",
            "Kept the entities located in this rank's rows",
            rank=runtime.plan.rank,
            entities=kept,
        )
    return values


__all__ = ["LocatedEntities", "build_slab", "cut_located"]

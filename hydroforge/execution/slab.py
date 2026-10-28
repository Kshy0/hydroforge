# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Rank-local rows of a model's slab-decomposed axis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydroforge.contracts.slab import CELL, FACE, INTERIOR
from hydroforge.parallel.slab import Slab

if TYPE_CHECKING:
    from hydroforge.compiler.plan import ModelPlan
    from hydroforge.execution.inputs import InputBinding
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
    return Slab.build(
        plan.spatial_rank,
        plan.spatial_world_size,
        total,
        declared.halo,
        plan.slab_periodic,
        topology.positions()[plan.spatial_rank],
    )


__all__ = ["build_slab"]

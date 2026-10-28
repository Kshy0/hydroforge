# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Row-slab decomposition of one structured-grid axis."""

from __future__ import annotations

from typing import Any

from pydantic import Field, InstanceOf, field_validator, model_validator

from hydroforge.contracts.conditions import Condition, as_condition
from hydroforge.core.naming import Identifier
from hydroforge.core.validation import HydroForgeModel

CELL, FACE, INTERIOR = "cell", "face", "interior"
"""Kinds of slab axis: the cell rows, the faces staggering them (one more
entry) and the interior rows of a padded axis."""


class SlabAxis(HydroForgeModel):
    """One grid axis split into contiguous row blocks over the spatial ranks.

    ``dim`` is the symbolic dimension of the axis's cell rows (``num_j``);
    every field whose shape names it, or a staggered axis of it declared in
    ``grid_axes`` (``GridAxis(name="south_north_stag", staggers="south_north")``,
    one more entry than the cells), holds this rank's rows of that axis only.
    ``halo`` ghost rows on each side follow from the neighbouring blocks: a
    stencil that reads up to ``halo`` rows away from an owned row finds them
    after :meth:`AbstractModel.halo_exchange_`. A ``periodic`` axis wraps its
    ghost rows around the domain across ranks; one rank keeps neither ghost
    rows nor wrapping, and its kernels close a periodic axis themselves.

    Fields of the axis are ``lower + owned + upper`` rows long: ghost rows
    sit next to the owned rows in global order, so a kernel that clamps its
    stencil to the stored rows sees the domain's own boundary at the first and
    last rank and ghost rows everywhere else. A global notion (a boundary
    zone, a distance to the edge) reads the rank's ``slab`` geometry.
    """

    dim: Identifier
    halo: int = Field(ge=1, strict=True)
    periodic: InstanceOf[Condition] | None = None
    interior: tuple[Identifier, ...] = ()
    """Axes holding only the interior rows: a grid padded by one perimeter row
    on each side of the domain (``dim``) has an interior axis two rows
    shorter, whose rows are the owned ones without the perimeter. With a
    ghost row on a side, the perimeter row of the padded axis is the ghost,
    so every rank holds the interior rows between its first and last stored
    row; this needs ``halo=1``."""

    @model_validator(mode="after")
    def _interior_needs_one_row(self) -> SlabAxis:
        if self.interior and self.halo != 1:
            raise ValueError("a slab with interior axes needs halo=1")
        return self

    @field_validator("periodic", mode="before")
    @classmethod
    def _condition(cls, value: Any) -> Any:
        return None if value is None else as_condition(value)


__all__ = ["CELL", "FACE", "INTERIOR", "SlabAxis"]

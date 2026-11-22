# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Declared exchange points between the modules of one model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Port:
    """One field a module offers to, or receives from, another module.

    In ``exports``, ``Port("heat")`` offers the module's field ``heat``. In
    ``imports``, ``Port("forcing", source="atmosphere.heat")`` receives export
    ``heat`` of module ``atmosphere`` into the module's field ``forcing``.
    The model decides when an exchange runs; ports only name what connects.

    ``dims`` names the field's dimensions a link-table ID indexes
    (:func:`hydroforge.execution.exchange_`), adjacent and in shape order;
    the field's other dimensions are carried along each point. By default a
    slab field's IDs index every dimension (``("num_lat", "num_lon")``:
    cells) and a partition or located field's its key dimension.
    """

    field: str
    source: str | None = None
    dims: tuple[str, ...] | None = None


__all__ = ["Port"]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Row-slab decomposition of a structured-grid axis.

A model declares ``slab = SlabAxis(dim="num_j", halo=3)``. The cell rows of
that axis are split into contiguous blocks, one per spatial rank, and every
field whose shape names the axis (or a staggered axis of it) is read, held
and saved as that rank's rows plus ``halo`` ghost rows on each side that has
a neighbour: ``lower + owned + upper`` rows in global order. Kernels keep
stencils inside the stored rows by clamping at the stored edge, which is the
domain's own edge at the first and last block and a ghost row elsewhere.

Blocks follow the ranks' node order (:meth:`Topology.positions
<hydroforge.parallel.topology.Topology.positions>`), so neighbouring blocks
share a node wherever they can. :class:`Slab` is one rank's geometry;
:func:`slab_layouts` gives every rank's stored rows to the halo
(:mod:`hydroforge.parallel.halo`), which exchanges ghost rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hydroforge.contracts.slab import CELL, FACE, INTERIOR


def blocks(total: int, world: int) -> list[tuple[int, int]]:
    """Contiguous row blocks of ``total`` rows over ``world`` ranks.

    Sizes differ by at most one row, the first ranks taking the remainder.
    """

    base, extra = divmod(total, world)
    start = 0
    result = []
    for rank in range(world):
        stop = start + base + (rank < extra)
        result.append((start, stop))
        start = stop
    return result


@dataclass(frozen=True, slots=True)
class Slab:
    """One rank's rows of a slab-decomposed axis.

    The rank owns cell rows ``[start, stop)`` of ``total``, block
    ``position`` of ``world``. Its arrays hold
    ``lower`` ghost rows, the owned rows and ``upper`` ghost rows; a staggered
    axis holds one more face than its cells. ``periodic`` ghost rows wrap
    around the domain. One rank has no ghost rows even on a periodic axis;
    its kernels close the axis themselves.
    """

    rank: int
    position: int
    world: int
    total: int
    start: int
    stop: int
    lower: int
    upper: int
    periodic: bool
    halo: int

    @classmethod
    def build(
        cls,
        rank: int,
        world: int,
        total: int,
        halo: int,
        periodic: bool,
        position: int | None = None,
    ) -> Slab:
        # A periodic block wraps its closing face onto the next block's
        # first, which must not be its own.
        least = halo + int(periodic)
        if world > 1 and total < world * least:
            raise ValueError(
                f"slab axis has {total} rows, too few for {world} ranks with "
                f"{halo} ghost rows each (needs at least {world * least})"
            )
        position = rank if position is None else position
        start, stop = blocks(total, world)[position]
        if world == 1:
            lower = upper = 0
        elif periodic:
            lower = upper = halo
        else:
            lower = min(halo, start)
            upper = min(halo, total - stop)
        return cls(
            rank, position, world, total, start, stop, lower, upper, periodic, halo
        )

    @property
    def owned(self) -> int:
        return self.stop - self.start

    @property
    def rows(self) -> int:
        """Cell rows stored: ghost rows below, owned and ghost rows above."""

        return self.lower + self.owned + self.upper

    @property
    def first(self) -> int:
        """Global row of stored row 0, before periodic wrapping."""

        return self.start - self.lower

    @property
    def owns_end_face(self) -> bool:
        """Whether this rank owns the closing face of a staggered axis."""

        return self.stop == self.total and not self.periodic

    @property
    def owned_faces(self) -> int:
        """Faces this rank is the authority of (what the exchange mirrors)."""

        return self.owned + int(self.owns_end_face)

    @property
    def saved_faces(self) -> int:
        """Faces this rank saves and writes: its owned ones, and the last rank's
        closing face, which on a periodic axis copies the first face."""

        return self.owned + int(self.stop == self.total)

    def _wrap(self, rows: np.ndarray) -> np.ndarray:
        return rows % self.total if self.periodic else rows

    def cell_rows(self) -> np.ndarray:
        """The global row each stored cell row holds."""

        return self._wrap(self.first + np.arange(self.rows, dtype=np.int64))

    def face_rows(self) -> np.ndarray:
        """The global face each stored face holds (cell rows plus one)."""

        return self._wrap(self.first + np.arange(self.rows + 1, dtype=np.int64))

    def owned_slice(self, kind: str = CELL) -> slice:
        """Stored rows this rank owns (and saves, and writes to output).

        An interior axis (see :class:`~hydroforge.contracts.slab.SlabAxis`)
        stores exactly its owned rows.
        """

        if kind == INTERIOR:
            return slice(0, self.rows - 2)
        count = self.saved_faces if kind == FACE else self.owned
        return slice(self.lower, self.lower + count)

    def global_rows(self, kind: str = CELL) -> np.ndarray:
        """Global indices of the owned rows, in stored order (an interior
        axis counts from the first interior row of the domain)."""

        if kind == INTERIOR:
            first = max(self.start, 1)
            last = min(self.stop, self.total - 1)
            return np.arange(first - 1, last - 1, dtype=np.int64)
        count = self.saved_faces if kind == FACE else self.owned
        return np.arange(self.start, self.start + count, dtype=np.int64)

    def rows_of(self, kind: str) -> np.ndarray:
        """Global rows to read from an input for each stored row of ``kind``."""

        if kind == FACE:
            return self.face_rows()
        if kind == INTERIOR:
            return self.global_rows(INTERIOR)
        return self.cell_rows()

    def total_of(self, kind: str) -> int:
        """Rows of ``kind`` in the whole domain."""

        return {CELL: self.total, FACE: self.total + 1, INTERIOR: self.total - 2}[kind]


def slab_layouts(slabs: Sequence[Slab], kind: str):
    """Every rank's stored rows of ``kind`` (cells or faces) for the halo."""

    from hydroforge.parallel.halo import RowLayout

    layouts = []
    for slab in slabs:
        stored = slab.face_rows() if kind == FACE else slab.cell_rows()
        count = slab.owned_faces if kind == FACE else slab.owned
        local = np.arange(stored.size)
        layouts.append(
            RowLayout(stored, (local >= slab.lower) & (local < slab.lower + count))
        )
    return layouts


__all__ = ["Slab", "blocks", "slab_layouts"]

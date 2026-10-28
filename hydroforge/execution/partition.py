# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Rank-local partition service over the bound global inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numba import njit
from pydantic import model_validator

from hydroforge.compiler.partition import bare, coordinate_is_partitioned
from hydroforge.core.arrays import immutable_array
from hydroforge.core.validation import HydroForgeModel

if TYPE_CHECKING:
    from hydroforge.compiler.plan import ModelPlan
    from hydroforge.execution.balance import Pieces
    from hydroforge.execution.inputs import InputBinding
    from hydroforge.parallel.topology import Topology


class GroupRankLookup(HydroForgeModel):
    """Sparse group-ID to rank mapping with NumPy-style lookup."""

    group_ids: np.ndarray
    ranks: np.ndarray

    @model_validator(mode="after")
    def _canonicalize(self):
        if (
            self.group_ids.ndim != 1
            or self.ranks.ndim != 1
            or self.group_ids.dtype != np.dtype(np.int64)
            or self.ranks.dtype != np.dtype(np.int64)
        ):
            raise ValueError("group IDs and ranks must be one-dimensional int64 arrays")
        if self.group_ids.shape != self.ranks.shape:
            raise ValueError("group IDs and ranks must have identical shape")
        if self.group_ids.size > 1 and np.any(
            self.group_ids[1:] <= self.group_ids[:-1]
        ):
            raise ValueError("group IDs must be strictly increasing")
        if np.any(self.ranks < 0):
            raise ValueError("partition ranks must be non-negative")
        object.__setattr__(
            self,
            "group_ids",
            immutable_array(self.group_ids, dtype=np.int64, order="C"),
        )
        object.__setattr__(
            self,
            "ranks",
            immutable_array(self.ranks, dtype=np.int64, order="C"),
        )
        return self

    def __getitem__(
        self,
        values: int | np.integer | np.ndarray,
    ) -> int | np.ndarray:
        if type(values) is int or isinstance(values, np.integer):
            integer = int(values)
            if not -(1 << 63) <= integer < (1 << 63):
                raise ValueError("group ID is outside the int64 range")
            array = np.asarray(integer, dtype=np.int64)
        elif isinstance(values, np.ndarray):
            if values.dtype != np.dtype(np.int64):
                raise ValueError("group ID arrays must use exact int64 dtype")
            array = values
        else:
            raise ValueError(
                "group IDs must be an exact int, NumPy integer, or int64 array"
            )
        flat = array.reshape(-1)
        positions = np.searchsorted(self.group_ids, flat)
        matched = positions < self.group_ids.size
        if np.any(matched):
            matched[matched] &= self.group_ids[positions[matched]] == flat[matched]
        if not np.all(matched):
            missing = flat[~matched][:5].tolist()
            raise ValueError(
                f"group IDs are absent from the model partition: {missing}"
            )
        result = self.ranks[positions].reshape(array.shape)
        return result.item() if array.ndim == 0 else result

    def _lookup_trusted(self, values: np.ndarray) -> np.ndarray:
        """Resolve compiled group IDs known to belong to this lookup."""

        positions = searchsorted_batch(self.group_ids, values)
        return self.ranks[positions]

    def __len__(self) -> int:
        return len(self.group_ids)


def searchsorted_batch(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """``np.searchsorted`` that sorts large query batches for memory locality."""

    queries = np.asarray(queries)
    if queries.size < 65536:
        return np.searchsorted(sorted_values, queries)
    flat = queries.reshape(-1)
    order = np.argsort(flat, kind="stable")
    positions = np.empty(flat.shape, dtype=np.intp)
    positions[order] = np.searchsorted(sorted_values, flat[order])
    return positions.reshape(queries.shape)


@njit(cache=True)
def _compute_group_to_rank(
    world_size: int,
    group_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily balance original group IDs over ranks."""
    if world_size <= 0 or group_assignments.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    ordered = np.sort(group_assignments)
    starts = np.empty(ordered.size, np.bool_)
    starts[0] = True
    starts[1:] = ordered[1:] != ordered[:-1]
    first = np.nonzero(starts)[0]
    unique_ids = ordered[first]
    sizes = np.empty(first.size, np.int64)
    sizes[:-1] = first[1:] - first[:-1]
    sizes[-1] = ordered.size - first[-1]
    order = np.argsort(sizes)
    loads = np.zeros(world_size, np.int64)
    ranks = np.empty(unique_ids.size, np.int64)
    for position in range(order.size - 1, -1, -1):
        group = order[position]
        rank = int(np.argmin(loads))
        ranks[group] = rank
        loads[rank] += sizes[group]
    return unique_ids, ranks


def compute_group_to_rank(
    world_size: int,
    group_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily balance already validated group IDs over validated ranks."""

    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive exact int")
    if np.ma.isMaskedArray(group_assignments):
        raise ValueError("group assignments must not be masked")
    values = np.asarray(group_assignments)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("group assignments must be a one-dimensional integer array")
    if (
        values.size
        and values.dtype.kind == "u"
        and int(values.max()) > np.iinfo(np.int64).max
    ):
        raise ValueError("group assignments exceed int64 range")
    canonical = values.astype(np.int64, copy=False)
    return _compute_group_to_rank(world_size, canonical)


def _host_ids(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    # Mixed signed/unsigned searches can otherwise promote IDs to float64.
    if array.dtype.kind == "u":
        if array.size and int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError("partition ID is outside the int64 range")
        array = array.astype(np.int64)
    return array


class PartitionRuntime:
    """Global reference integrity and rank ownership of the bound inputs.

    Construction reads the global reference and coordinate values and rejects
    references absent from their target coordinate, as well as inverse
    references that are not one-to-one.
    """

    def __init__(
        self, plan: ModelPlan, source: InputBinding, topology: Topology | None = None
    ) -> None:
        self.schema = plan.fields.partition
        self.topology = topology
        self.variable_groups = plan.fields.variable_groups
        self.partition_key = plan.spec.partition_key
        self.partition_group = plan.spec.partition_group
        self.spatial_rank = plan.spatial_rank
        self.spatial_world_size = plan.spatial_world_size
        self.source = source
        self._coordinate_groups: dict[str, np.ndarray] = {}
        self._reference_indices: dict[str, np.ndarray] = {}
        self._rank_indices: dict[str, np.ndarray] = {}
        self._sorted_keys: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
        self._group_ranks: GroupRankLookup | None = None
        self._pieces: Pieces | None = None
        self.plan = plan
        self._halo: tuple[list[np.ndarray], list[np.ndarray]] | None | bool = False
        self._validate_global_reference_integrity()
        self._validate_inverse_reference_integrity(plan.fields.inverse_sources)

    def sorted_global_key(
        self,
        name: str,
        load: Callable[[], Any],
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return one shared ``(order, sorted, unique)`` view of a global key."""

        cached = self._sorted_keys.get(name)
        if cached is None:
            values = _host_ids(load()).reshape(-1)
            order = np.argsort(values, kind="stable")
            ordered = values[order]
            unique = not (
                ordered.size > 1 and bool(np.any(ordered[1:] == ordered[:-1]))
            )
            cached = (order, ordered, unique)
            self._sorted_keys[name] = cached
        return cached

    def _validate_global_reference_integrity(self) -> None:
        """Validate external reference values before runtime slicing."""

        source = self.source
        for name, metadata in self.schema.fields.items():
            target = bare(metadata.references)
            if not target or name not in source or target not in source:
                continue
            values = _host_ids(source.value(name))
            shape = values.shape
            values = values.reshape(-1)
            order, sorted_target, unique = self.sorted_global_key(
                target,
                lambda target=target: source.value(target),
            )
            if not unique:
                raise ValueError(
                    f"Reference target coordinate '{target}' must contain "
                    "unique values."
                )
            position = searchsorted_batch(sorted_target, values)
            found = position < sorted_target.size
            found[found] = sorted_target[position[found]] == values[found]
            missing = ~found
            if np.any(missing):
                raise ValueError(
                    f"Reference field '{name}' has {int(missing.sum())} "
                    f"value(s) absent from global coordinate '{target}'; "
                    f"examples: {values[missing][:5].tolist()}."
                )
            coordinate = self.variable_groups.get(name)
            if (
                coordinate is not None
                and coordinate != self.partition_key
                and name
                == (bare(self.schema.fields[coordinate].partition_by) or coordinate)
            ):
                # Ownership slicing needs these exact global positions next.
                # Other references remain lazy to avoid retaining unused indices.
                self._reference_indices[name] = order[position].reshape(shape)

    def _validate_inverse_reference_integrity(
        self,
        inverse_sources: frozenset[str],
    ) -> None:
        """Prove every inverse reference is a one-to-one global relation."""

        source = self.source
        for name in inverse_sources:
            if name not in source:
                continue
            values = _host_ids(source.value(name)).reshape(-1)
            if np.unique(values).size != values.size:
                raise ValueError(
                    f"Inverse reference field {name!r} must contain unique "
                    "target references"
                )

    def _reference_index(self, name: str) -> np.ndarray:
        cached = self._reference_indices.get(name)
        if cached is not None:
            return cached
        target = bare(self.schema.fields[name].references)
        values = _host_ids(self.source.value(name))
        order, sorted_target, _unique = self.sorted_global_key(
            target, lambda: self.source.value(target)
        )
        index = order[searchsorted_batch(sorted_target, values)]
        self._reference_indices[name] = index
        return index

    def _coordinate_group_values(self, coordinate: str) -> np.ndarray:
        cached = self._coordinate_groups.get(coordinate)
        if cached is not None:
            return cached
        metadata = self.schema.fields[coordinate]
        if coordinate == self.partition_key:
            groups = (
                _host_ids(self.source.value(self.partition_group))
                if self.spatial_world_size == 1
                else self.pieces.ids
            )
        else:
            via = bare(metadata.partition_by)
            if via:
                target = bare(self.schema.fields[via].references)
                index = self._reference_index(via)
            else:
                target = bare(metadata.references)
                index = self._reference_index(coordinate)
            groups = self._coordinate_group_values(target)[index]
        self._coordinate_groups[coordinate] = groups
        return groups

    @property
    def group_ranks(self) -> GroupRankLookup:
        """Rank of every partition id (:attr:`pieces`)."""

        cached = self._group_ranks
        if cached is None:
            if self.spatial_world_size == 1:
                ids, ranks = compute_group_to_rank(
                    1, _host_ids(self.source.value(self.partition_group))
                )
            else:
                ids, ranks = self.pieces.piece_ids, self.pieces.ranks
            cached = GroupRankLookup(group_ids=ids, ranks=ranks)
            self._group_ranks = cached
        return cached

    @property
    def pieces(self) -> Pieces:
        """The partition ids of the key rows, derived before any slicing.

        The partition group when the groups balance over the ranks (or the
        halo is not exchanged, or ``partition_imbalance`` is ``None``), else
        pieces of groups cut along the crossing references
        (:mod:`hydroforge.execution.balance`). Every rank derives them alike
        from the global partition group and reference fields; they are not a
        model field.
        """

        if self._pieces is None:
            from hydroforge.execution.balance import balance

            groups = _host_ids(self.source.value(self.partition_group)).reshape(-1)
            tolerance = self.plan.partition_imbalance
            tree = edges = (np.zeros(0, np.int64), np.zeros(0, np.int64))
            if tolerance is not None and self.halo_problem() is None:
                tree, edges = self._key_edges()
            else:
                tolerance = None
            nodes = None if self.topology is None else self.topology.nodes
            self._pieces = balance(
                self.spatial_world_size, groups, tree, edges, tolerance, nodes
            )
        return self._pieces

    def _key_rows(self, coordinate: str) -> np.ndarray:
        """The partition-key row that owns each row of ``coordinate``."""

        if coordinate == self.partition_key:
            return np.arange(_host_ids(self.source.value(coordinate)).size)
        metadata = self.schema.fields[coordinate]
        via = bare(metadata.partition_by)
        name = via or coordinate
        target = bare(self.schema.fields[name].references)
        return self._key_rows(target)[self._reference_index(name)].reshape(-1)

    def _key_edges(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Key-row pairs of the crossing references: ``(tree, edges)``.

        ``tree`` holds the references within the key coordinate with one
        target per row (``downstream``), along which groups are cut;
        ``edges`` every crossing reference.
        """

        tree: list[tuple[np.ndarray, np.ndarray]] = []
        edges: list[tuple[np.ndarray, np.ndarray]] = []
        for name in self.crossing_references():
            coordinate = self.variable_groups[name]
            targets = self._reference_index(name)
            sources = self._key_rows(coordinate)
            width = targets.size // max(sources.size, 1)
            pair = (np.repeat(sources, width), targets.reshape(-1).astype(np.int64))
            edges.append(pair)
            if coordinate == self.partition_key and width == 1:
                tree.append(pair)

        def joined(pairs):
            if not pairs:
                return np.zeros(0, np.int64), np.zeros(0, np.int64)
            return tuple(np.concatenate(side) for side in zip(*pairs))

        return joined(tree), joined(edges)

    def crossing_references(self) -> tuple[str, ...]:
        """Reference fields to the partition key that may cross ranks.

        Every reference to a partition-key row except the ones that define a
        coordinate's ownership (``partition_by`` or a referencing coordinate),
        which keep their rows on the target's rank by construction.
        """

        key = self.partition_key
        fields = self.schema.fields
        owning = {
            bare(fields[coordinate].partition_by) or coordinate
            for coordinate in self.schema.coordinates
            if coordinate != key
        }
        return tuple(
            name
            for name, metadata in fields.items()
            if bare(metadata.references) == key
            and name not in owning
            and name in self.variable_groups
            and name in self.source
            and coordinate_is_partitioned(self.schema, key, self.variable_groups[name])
        )

    def halo_rows(self) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        """Every rank's owned and ghost global rows of the partition key.

        ``None`` when every partition group is closed under the crossing
        references (always on one rank): no rank then holds ghost rows and
        nothing is exchanged. A group that is not closed needs the model to
        exchange the halo of every open module (``ModuleRequirement(halo=True)``).
        """

        if self._halo is not False:
            return self._halo
        self._halo = None
        if self.spatial_world_size == 1 or self.partition_key is None:
            return None
        references = self.crossing_references()
        if not references:
            return None
        from hydroforge.parallel.halo import ghost_rows

        world = self.spatial_world_size
        lookup = self.group_ranks._lookup_trusted

        def rows(coordinate: str) -> list[np.ndarray]:
            ranks = lookup(self._coordinate_group_values(coordinate))
            return [np.nonzero(ranks == r)[0] for r in range(world)]

        owned = rows(self.partition_key)
        crossing = {
            name: (
                self._reference_index(name).reshape(-1),
                rows(self.variable_groups[name]),
            )
            for name in references
        }
        ghosts = ghost_rows(owned, tuple(crossing.values()))
        if not any(rows.size for rows in ghosts):
            return None
        self._require_halo(crossing, owned)
        self._halo = (owned, ghosts)
        return self._halo

    def halo_problem(self) -> str | None:
        """Why this model cannot exchange a halo, or ``None``."""

        from hydroforge.contracts.runtime import DEFAULT_MODULE_REQUIREMENT

        plan = self.plan
        requirements = plan.spec.module_requirements
        unsupported = [
            name
            for name in plan.modules
            if not requirements.get(name, DEFAULT_MODULE_REQUIREMENT).halo
        ]
        if unsupported:
            return f"the model does not exchange the halo of modules {unsupported}"
        if plan.ensemble_size is not None:
            return "halo exchange supports one ensemble member"
        return None

    def _require_halo(self, crossing, owned) -> None:
        """Reject a split partition the model does not exchange."""

        problem = self.halo_problem()
        if problem is None:
            return
        owner = np.empty(sum(rows.size for rows in owned), np.int64)
        for rank, rows in enumerate(owned):
            owner[rows] = rank
        split = {
            name: int(
                sum(
                    np.count_nonzero(owner[targets[rows]] != rank)
                    for rank, rows in enumerate(sources)
                )
            )
            for name, (targets, sources) in crossing.items()
        }
        raise ValueError(
            f"partition groups {self.partition_group!r} split references across "
            f"ranks ({', '.join(f'{n}: {c} rows' for n, c in split.items() if c)}), "
            f"but {problem}; keep connected structure in one group"
        )

    def ghost_indices(self, coordinate: str) -> np.ndarray:
        """Global rows this rank holds as ghosts of ``coordinate``."""

        halo = self.halo_rows() if coordinate == self.partition_key else None
        if halo is None:
            return np.zeros(0, dtype=np.int64)
        return halo[1][self.spatial_rank]

    def rank_indices(self, coordinate: str) -> np.ndarray:
        cached = self._rank_indices.get(coordinate)
        if cached is not None:
            return cached
        groups = self._coordinate_group_values(coordinate)
        if self.spatial_world_size == 1:
            indices = np.arange(groups.size, dtype=np.int64)
        else:
            ranks = self.group_ranks._lookup_trusted(groups)
            indices = np.nonzero(ranks == self.spatial_rank)[0]
        self._rank_indices[coordinate] = indices
        return indices

"""Rank-local partition service over the bound global inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numba import njit
from pydantic import model_validator

from hydroforge.compiler.partition import bare
from hydroforge.core.arrays import immutable_array
from hydroforge.core.validation import HydroForgeModel

if TYPE_CHECKING:
    from hydroforge.compiler.plan import ModelPlan
    from hydroforge.execution.inputs import InputBinding


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

        positions = _searchsorted_batch(self.group_ids, values)
        return self.ranks[positions]

    def __len__(self) -> int:
        return len(self.group_ids)


def _searchsorted_batch(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
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

    def __init__(self, plan: ModelPlan, source: InputBinding) -> None:
        self.schema = plan.fields.partition
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
            values = _host_ids(source.value(name)).reshape(-1)
            _order, sorted_target, unique = self.sorted_global_key(
                target,
                lambda target=target: source.value(target),
            )
            if not unique:
                raise ValueError(
                    f"Reference target coordinate '{target}' must contain "
                    "unique values."
                )
            position = _searchsorted_batch(sorted_target, values)
            found = position < sorted_target.size
            found[found] = sorted_target[position[found]] == values[found]
            missing = ~found
            if np.any(missing):
                raise ValueError(
                    f"Reference field '{name}' has {int(missing.sum())} "
                    f"value(s) absent from global coordinate '{target}'; "
                    f"examples: {values[missing][:5].tolist()}."
                )

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
        index = order[_searchsorted_batch(sorted_target, values)]
        self._reference_indices[name] = index
        return index

    def _coordinate_group_values(self, coordinate: str) -> np.ndarray:
        cached = self._coordinate_groups.get(coordinate)
        if cached is not None:
            return cached
        metadata = self.schema.fields[coordinate]
        if coordinate == self.partition_key:
            groups = _host_ids(self.source.value(self.partition_group))
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
        cached = self._group_ranks
        if cached is None:
            ids, ranks = compute_group_to_rank(
                self.spatial_world_size,
                _host_ids(self.source.value(self.partition_group)),
            )
            cached = GroupRankLookup(group_ids=ids, ranks=ranks)
            self._group_ranks = cached
        return cached

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

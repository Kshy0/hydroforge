"""Compiled coordinate ownership and rank-local partition service."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from numba import njit

from hydroforge.data.distributed import find_indices_in, find_indices_in_torch
from hydroforge.contracts.fields import (
    PartitionSchema, RuntimeTensorMetadata, tensor_is_active,
)


@dataclass(frozen=True)
class GroupRankLookup:
    """Sparse group-ID to rank mapping with NumPy-style lookup."""

    group_ids: np.ndarray
    ranks: np.ndarray

    def __getitem__(self, values):
        array = np.asarray(values)
        flat = array.reshape(-1)
        positions = np.searchsorted(self.group_ids, flat)
        valid = positions < len(self.group_ids)
        matched = np.zeros(flat.shape, dtype=np.bool_)
        matched[valid] = self.group_ids[positions[valid]] == flat[valid]
        if not np.all(matched):
            raise KeyError(f"Unknown group ID(s): {flat[~matched][:5].tolist()}")
        result = self.ranks[positions].reshape(array.shape)
        return result.item() if array.ndim == 0 else result

    def __len__(self) -> int:
        return len(self.group_ids)


@njit
def compute_group_to_rank(
    world_size: int, group_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily balance original group IDs over ranks."""
    if world_size <= 0 or group_assignments.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    unique_ids = np.unique(group_assignments).astype(np.int64)
    inverse = np.searchsorted(unique_ids, group_assignments)
    sizes = np.bincount(inverse, minlength=unique_ids.size).astype(np.int64)
    order = np.argsort(sizes)
    loads = np.zeros(world_size, np.int64)
    ranks = np.empty(unique_ids.size, np.int64)
    for position in range(order.size - 1, -1, -1):
        group = order[position]
        rank = int(np.argmin(loads))
        ranks[group] = rank
        loads[rank] += sizes[group]
    return unique_ids, ranks


class PartitionCompiler:
    """Own the immutable partition graph and every derived rank-local index."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._schema: PartitionSchema | None = None
        self._variable_groups: MappingProxyType | None = None
        self._coordinate_groups: dict[str, np.ndarray] = {}
        self._reference_indices: dict[str, np.ndarray] = {}
        self._group_ranks: GroupRankLookup | None = None

    @property
    def schema(self) -> PartitionSchema:
        cached = self._schema
        if cached is not None:
            return cached
        model = self.model
        fields = {
            field.name: field.tensor
            for module_name in model.opened_modules
            for field in model.compiled_schema().fields(module_name)
            if (
                not field.computed
                and field.tensor is not None
                and tensor_is_active(field.tensor, model.opened_modules)
            )
        }
        coordinates = {
            name for name, metadata in fields.items() if metadata.is_coordinate
        }
        selections: dict[str, str] = {}

        # Structured-grid models without logical CoordinateField axes are
        # unpartitioned by construction; they need no model-side override.
        if not coordinates:
            cached = PartitionSchema(
                fields=MappingProxyType(fields),
                coordinates=frozenset(),
                selections=MappingProxyType({}),
            )
            self._schema = cached
            return cached

        if model.partition_key is None:
            raise ValueError("Model partition_key must be configured.")
        if model.partition_key not in coordinates:
            raise ValueError(
                f"partition_key '{model.partition_key}' must be a CoordinateField."
            )
        if model.partition_group not in fields:
            raise ValueError(
                f"partition_group '{model.partition_group}' is not declared."
            )

        for name, metadata in fields.items():
            coordinate = self._bare(metadata.dim_coords)
            if coordinate and coordinate not in coordinates:
                raise ValueError(
                    f"Field '{name}' uses dim_coords='{coordinate}', but it is "
                    "not a CoordinateField."
                )
            references = self._bare(metadata.references)
            if references and references not in coordinates:
                raise ValueError(
                    f"Field '{name}' references unknown coordinate '{references}'."
                )
            selects = self._bare(metadata.selects)
            if selects:
                if name not in coordinates:
                    raise ValueError(f"Selection '{name}' must be a CoordinateField.")
                if references != selects:
                    raise ValueError(
                        f"Selection '{name}' must reference the coordinate it "
                        f"selects ('{selects}')."
                    )
                previous = selections.get(selects)
                if previous is not None:
                    raise ValueError(
                        f"Coordinate '{selects}' has multiple default selections: "
                        f"'{previous}' and '{name}'."
                    )
                selections[selects] = name

            partition_by = self._bare(metadata.partition_by)
            if metadata.replicated and name not in coordinates:
                raise ValueError(
                    f"replicated=True is only valid on CoordinateField, got '{name}'."
                )
            if metadata.replicated and (
                name == model.partition_key or partition_by or references
            ):
                raise ValueError(
                    f"Replicated coordinate '{name}' cannot define partition lineage."
                )
            if (
                name in coordinates
                and name != model.partition_key
                and not partition_by
                and not references
                and not metadata.replicated
            ):
                raise ValueError(
                    f"Coordinate '{name}' has no ownership lineage. Declare "
                    "partition_by/references or set replicated=True."
                )
            if partition_by:
                if name not in coordinates:
                    raise ValueError(
                        f"partition_by is only valid on CoordinateField, got '{name}'."
                    )
                if partition_by not in fields:
                    raise ValueError(
                        f"Coordinate '{name}' partitions by undeclared field "
                        f"'{partition_by}'."
                    )
                via = fields[partition_by]
                if self._bare(via.dim_coords) != name:
                    raise ValueError(
                        f"Partition field '{partition_by}' must be aligned to "
                        f"coordinate '{name}', got dim_coords={via.dim_coords!r}."
                    )
                if not via.references:
                    raise ValueError(
                        f"Partition field '{partition_by}' must declare references."
                    )

        cached = PartitionSchema(
            fields=MappingProxyType(fields),
            coordinates=frozenset(coordinates),
            selections=MappingProxyType(selections),
        )
        self._schema = cached
        return cached

    @staticmethod
    def _bare(name: str | None) -> str | None:
        return name.rsplit(".", 1)[-1] if name else None

    def coordinate_is_partitioned(self, coordinate: str) -> bool:
        metadata = self.schema.fields[coordinate]
        return bool(
            not metadata.replicated
            and (
                coordinate == self.model.partition_key
                or metadata.partition_by
                or metadata.references
            )
        )

    @property
    def variable_groups(self) -> MappingProxyType:
        cached = self._variable_groups
        if cached is not None:
            return cached
        mapping: dict[str, str] = {}
        for name, metadata in self.schema.fields.items():
            if name in self.schema.coordinates:
                if self.coordinate_is_partitioned(name):
                    mapping[name] = name
                continue
            coordinate = self._bare(metadata.dim_coords)
            if coordinate and self.coordinate_is_partitioned(coordinate):
                mapping[name] = coordinate
        cached = MappingProxyType(mapping)
        self._variable_groups = cached
        return cached

    def field_coordinate(self, field: Any) -> str | None:
        if field.tensor is None:
            return None
        coordinate = self._bare(field.tensor.dim_coords)
        return (
            coordinate
            if coordinate and self.coordinate_is_partitioned(coordinate)
            else None
        )

    def logical_axis(
        self, field_name: str, field: Any, shape: tuple[int, ...],
    ) -> int:
        if field.tensor is None:
            raise ValueError(f"Field {field_name!r} is not a tensor field")
        logical_ndim = len(field.tensor.shape)
        if len(shape) == logical_ndim:
            return 0
        trials = self.model.num_trials
        if trials is not None and len(shape) == logical_ndim + 1:
            if shape[0] != trials:
                raise ValueError(
                    f"Batched field '{field_name}' has leading size {shape[0]}, "
                    f"expected num_trials={trials}."
                )
            return 1
        raise ValueError(
            f"Field '{field_name}' has rank {len(shape)}, but tensor_shape declares "
            f"{logical_ndim} logical dimension(s)."
        )

    def validate_input_axes(self, fields: dict[str, Any]) -> None:
        proxy = self.model.input_proxy
        for name, field in fields.items():
            if name not in proxy or field.tensor is None:
                continue
            coordinate = self._bare(field.tensor.dim_coords)
            if not coordinate:
                continue
            if coordinate not in proxy:
                raise ValueError(
                    f"Field '{name}' requires missing dim_coords '{coordinate}'."
                )
            shape = proxy.get_var_shape(name)
            coordinate_shape = proxy.get_var_shape(coordinate)
            if len(coordinate_shape) != 1:
                raise ValueError(
                    f"Coordinate '{coordinate}' must be 1-D, got {coordinate_shape}."
                )
            axis = self.logical_axis(name, field, shape)
            if shape[axis] != coordinate_shape[0]:
                raise ValueError(
                    f"Field '{name}' logical axis length {shape[axis]} does not match "
                    f"dim_coords '{coordinate}' length {coordinate_shape[0]}."
                )

    def reference_index(self, name: str) -> np.ndarray:
        cached = self._reference_indices.get(name)
        if cached is not None:
            return cached
        target = self._bare(self.schema.fields[name].references)
        if not target:
            raise ValueError(f"Field '{name}' is not a reference field.")
        proxy = self.model.input_proxy
        if name not in proxy or target not in proxy:
            raise ValueError(
                f"Reference field '{name}' requires loaded coordinate '{target}'."
            )
        values = np.asarray(proxy[name])
        target_values = np.asarray(proxy[target])
        if values.ndim != 1 or target_values.ndim != 1:
            raise ValueError(
                f"Reference field '{name}' and coordinate '{target}' must be 1-D."
            )
        index = find_indices_in(values, target_values)
        missing = index < 0
        if np.any(missing):
            raise ValueError(
                f"Reference field '{name}' has {int(missing.sum())} value(s) "
                f"absent from coordinate '{target}'; examples: "
                f"{values[missing][:5].tolist()}."
            )
        self._reference_indices[name] = index
        return index

    def coordinate_group_values(
        self, coordinate: str, resolving: set[str] | None = None,
    ) -> np.ndarray:
        cached = self._coordinate_groups.get(coordinate)
        if cached is not None:
            return cached
        resolving = set() if resolving is None else resolving
        if coordinate in resolving:
            raise ValueError(f"Partition coordinate cycle detected at '{coordinate}'.")
        resolving.add(coordinate)
        model = self.model
        metadata = self.schema.fields[coordinate]
        keys = np.asarray(model.input_proxy[coordinate])
        if keys.ndim != 1 or len(np.unique(keys)) != len(keys):
            raise ValueError(f"Coordinate '{coordinate}' must contain unique 1-D values.")
        if coordinate == model.partition_key:
            groups = np.asarray(model.input_proxy[model.partition_group])
            if groups.ndim != 1 or len(groups) != len(keys):
                raise ValueError(
                    f"partition_group '{model.partition_group}' must align with "
                    f"partition_key '{coordinate}'."
                )
        else:
            via = self._bare(metadata.partition_by)
            references = self._bare(metadata.references)
            if via:
                target = self._bare(self.schema.fields[via].references)
                index = self.reference_index(via)
            elif references:
                target = references
                index = self.reference_index(coordinate)
            else:
                raise ValueError(
                    f"Coordinate '{coordinate}' has no partition lineage."
                )
            groups = self.coordinate_group_values(target, resolving)[index]
        resolving.remove(coordinate)
        self._coordinate_groups[coordinate] = groups
        return groups

    @property
    def group_ranks(self) -> GroupRankLookup:
        cached = self._group_ranks
        if cached is not None:
            return cached
        model = self.model
        if model.partition_group not in model.input_proxy:
            raise ValueError(
                f"Missing partition_group '{model.partition_group}' in InputProxy."
            )
        ids, ranks = compute_group_to_rank(
            model.world_size, np.asarray(model.input_proxy[model.partition_group]),
        )
        cached = GroupRankLookup(group_ids=ids, ranks=ranks)
        self._group_ranks = cached
        return cached

    def rank_indices(self, coordinate: str) -> np.ndarray:
        groups = self.coordinate_group_values(coordinate)
        if not np.issubdtype(groups.dtype, np.integer) or np.any(groups < 0):
            raise ValueError(
                f"Resolved groups for coordinate '{coordinate}' must be nonnegative integers."
            )
        try:
            ranks = self.group_ranks[groups]
        except KeyError as exc:
            raise ValueError(
                f"Resolved groups for coordinate '{coordinate}' are unknown."
            ) from exc
        return np.nonzero(ranks == self.model.rank)[0]

    def validate_reference_integrity(self, module_data: dict[str, Any]) -> None:
        proxy = self.model.input_proxy
        for name, metadata in self.schema.fields.items():
            target = self._bare(metadata.references)
            if not target or name not in module_data or target not in proxy:
                continue
            values = self._numpy(module_data[name])
            target_values = np.asarray(proxy[target])
            index = find_indices_in(values.reshape(-1), target_values.reshape(-1))
            missing = index < 0
            if np.any(missing):
                raise ValueError(
                    f"Reference field '{name}' has {int(missing.sum())} value(s) "
                    f"absent from global coordinate '{target}'; examples: "
                    f"{values.reshape(-1)[missing][:5].tolist()}."
                )

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def bind_output(
        self, field: Any,
    ) -> tuple[RuntimeTensorMetadata, dict[str, torch.Tensor]]:
        if field.tensor is None:
            raise TypeError(f"{field.module_name}.{field.name} is not a tensor")
        policy = field.tensor.output
        coordinate = (
            None if policy == "disabled" else self._bare(field.tensor.dim_coords)
        )
        index_name = None
        indices = None
        coordinate_tensor = None
        variable_map = self.model._namespace.build()
        if policy != "disabled" and coordinate:
            if coordinate not in variable_map:
                raise ValueError(
                    f"Output coordinate '{coordinate}' is not available in opened modules."
                )
            module, attribute, _ = variable_map[coordinate]
            coordinate_tensor = getattr(module, attribute)
            selection = (
                self.schema.selections.get(coordinate) if policy == "auto" else None
            )
            if selection:
                module, attribute, _ = variable_map[selection]
                selected = getattr(module, attribute)
                if selected is not None:
                    indices = (
                        torch.empty(0, dtype=torch.int32, device=self.model.device)
                        if selected.numel() == 0
                        else find_indices_in_torch(selected, coordinate_tensor)
                    )
                    if torch.any(indices < 0):
                        missing = selected[indices < 0][:5].detach().cpu().tolist()
                        raise ValueError(
                            f"Selection '{selection}' contains values absent from "
                            f"coordinate '{coordinate}'; examples: {missing}."
                        )
                    indices = indices.to(self.model.device)
                    index_name = f"__selection_idx__{selection}"
                    coordinate = selection
                    coordinate_tensor = selected
        bound = RuntimeTensorMetadata(
            tensor=field.tensor,
            description=field.description or f"Variable {field.name}",
            output_index=index_name,
            output_coord=coordinate,
        )
        tensors: dict[str, torch.Tensor] = {}
        if index_name is not None and indices is not None:
            tensors[index_name] = indices
        if coordinate and coordinate_tensor is not None:
            tensors[coordinate] = coordinate_tensor
        return bound, tensors

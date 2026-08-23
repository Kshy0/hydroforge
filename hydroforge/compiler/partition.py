"""Compiled coordinate ownership and rank-local partition service."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numba import njit
from pydantic import Field, PrivateAttr, model_validator

from hydroforge.data.distributed import (
    _find_indices_in_trusted,
    _find_indices_in_torch_trusted,
)
from hydroforge.compiler.model import _ReferenceTargetPlan
from hydroforge.contracts.fields import (
    ModuleFieldSchema, PartitionSchema, RuntimeTensorMetadata,
    tensor_is_active,
)
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.data.numeric import immutable_array

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class _GroupRankQuery(HydroForgeModel):
    """One strict public lookup against a compiled group/rank identity."""

    values: Any
    group_ids: np.ndarray = Field(exclude=True)
    ranks: np.ndarray = Field(exclude=True)

    _result: int | np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self):
        values = self.values
        if type(values) is int:
            if not -(1 << 63) <= values < (1 << 63):
                raise ValueError("group ID is outside the int64 range")
            array = np.asarray(values, dtype=np.int64)
        elif isinstance(values, np.integer) and not isinstance(values, np.bool_):
            integer = int(values)
            if not -(1 << 63) <= integer < (1 << 63):
                raise ValueError("group ID is outside the int64 range")
            array = np.asarray(integer, dtype=np.int64)
        elif isinstance(values, np.ndarray):
            if values.dtype != np.dtype(np.int64):
                raise ValueError(
                    "group ID arrays must use exact int64 dtype"
                )
            array = values
        else:
            raise ValueError(
                "group IDs must be an exact int, NumPy integer, or int64 array"
            )

        flat = array.reshape(-1)
        positions = np.searchsorted(self.group_ids, flat)
        matched = positions < self.group_ids.size
        if np.any(matched):
            matched[matched] &= (
                self.group_ids[positions[matched]] == flat[matched]
            )
        if not np.all(matched):
            missing = flat[~matched][:5].tolist()
            raise ValueError(
                f"group IDs are absent from the model partition: {missing}"
            )
        result = self.ranks[positions].reshape(array.shape)
        self._result = result.item() if array.ndim == 0 else result
        return self

    @property
    def result(self) -> int | np.ndarray:
        return self._result


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
            raise ValueError(
                "group IDs and ranks must be one-dimensional int64 arrays"
            )
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
        self, values: int | np.integer | np.ndarray,
    ) -> int | np.ndarray:
        return _GroupRankQuery(
            values=values,
            group_ids=self.group_ids,
            ranks=self.ranks,
        ).result

    def _lookup_trusted(self, values: np.ndarray) -> np.ndarray:
        """Resolve compiled group IDs known to belong to this lookup."""

        positions = np.searchsorted(self.group_ids, values)
        return self.ranks[positions]

    def __len__(self) -> int:
        return len(self.group_ids)


@njit
def _compute_group_to_rank(
    world_size: int, group_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily balance original group IDs over ranks."""
    if world_size <= 0 or group_assignments.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    unique_ids = np.unique(group_assignments)
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


def compute_group_to_rank(
    world_size: int, group_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily balance already validated group IDs over validated ranks."""

    canonical = np.asarray(group_assignments, dtype=np.int64)
    return _compute_group_to_rank(world_size, canonical)


class _PartitionSemanticCompiler:
    """Validate and compile the immutable model partition declaration."""

    def __init__(
        self,
        model: AbstractModel,
        *,
        schema: PartitionSchema | None = None,
        variable_groups: MappingProxyType | None = None,
    ) -> None:
        self.model = model
        self._schema = schema
        self._variable_groups = variable_groups
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
            for field in model._compiled_schema().fields(module_name)
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

        lineage: dict[str, str] = {}
        for coordinate in coordinates:
            if coordinate == model.partition_key:
                continue
            metadata = fields[coordinate]
            via = self._bare(metadata.partition_by)
            target = (
                self._bare(fields[via].references)
                if via else self._bare(metadata.references)
            )
            if target is not None:
                lineage[coordinate] = target
        for origin in lineage:
            seen: set[str] = set()
            coordinate = origin
            while coordinate in lineage:
                if coordinate in seen:
                    raise ValueError(
                        "partition coordinate lineage must be acyclic; "
                        f"cycle includes {coordinate!r}"
                    )
                seen.add(coordinate)
                coordinate = lineage[coordinate]

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

    def field_coordinate(self, field: ModuleFieldSchema) -> str | None:
        if field.tensor is None:
            return None
        coordinate = self._bare(field.tensor.dim_coords)
        return (
            coordinate
            if coordinate and self.coordinate_is_partitioned(coordinate)
            else None
        )

    def logical_axis(
        self,
        field_name: str,
        field: ModuleFieldSchema,
        shape: tuple[int, ...],
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

    def compile_input_axes(
        self, fields: dict[str, Any],
    ) -> MappingProxyType:
        proxy = self.model._input
        axes: dict[str, int] = {}
        for name, field in fields.items():
            if name not in proxy or field.tensor is None:
                continue
            coordinate = self._bare(field.tensor.dim_coords)
            if (
                coordinate is None
                and name in self.schema.coordinates
                and self.coordinate_is_partitioned(name)
            ):
                coordinate = name
            if not coordinate:
                continue
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
            axes[name] = axis
        return MappingProxyType(axes)

    def validate_global_reference_integrity(self) -> None:
        """Validate external reference values before runtime slicing."""

        proxy = self.model._input.proxy
        for name, metadata in self.schema.fields.items():
            target = self._bare(metadata.references)
            if not target or name not in proxy or target not in proxy:
                continue
            values = self._numpy(
                proxy._get_value_trusted(name),
                label=f"reference field {name!r}",
            ).reshape(-1)
            target_values = self._numpy(
                proxy._get_value_trusted(target),
                label=f"reference coordinate {target!r}",
            ).reshape(-1)
            if np.unique(target_values).size != target_values.size:
                raise ValueError(
                    f"Reference target coordinate '{target}' must contain "
                    "unique values."
                )
            index = _find_indices_in_trusted(values, target_values)
            missing = index < 0
            if np.any(missing):
                raise ValueError(
                    f"Reference field '{name}' has {int(missing.sum())} "
                    f"value(s) absent from global coordinate '{target}'; "
                    f"examples: {values[missing][:5].tolist()}."
                )

    def compile_reference_targets(
        self,
    ) -> tuple[
        MappingProxyType,
        frozenset[str],
    ]:
        """Resolve every derived reference-index target before runtime."""

        model = self.model
        module_types = model._module_types()
        opened = frozenset(model.opened_modules)
        compiled: dict[str, MappingProxyType] = {}
        inverse_sources: set[str] = set()

        for module_name in model.opened_modules:
            module_type = module_types[module_name]
            module_references = module_type._module_reference_fields()
            plans: dict[str, _ReferenceTargetPlan] = {}
            for descriptor in module_type._reference_index_fields().values():
                source = module_type._tensor_schema_map().get(
                    descriptor.reference,
                )
                if source is None or source.tensor is None:
                    raise ValueError(
                        f"ReferenceIndexField {descriptor.reference!r} in "
                        f"module {module_name!r} does not name a tensor field"
                    )
                if not tensor_is_active(source.tensor, opened):
                    continue
                target_name = source.tensor.references
                if not target_name:
                    raise ValueError(
                        f"ReferenceIndexField {descriptor.reference!r} in "
                        f"module {module_name!r} refers to a field without "
                        "reference metadata"
                    )

                parts = target_name.split(".")
                target_field = parts[-1]
                candidates: list[tuple[str, str]] = []
                if len(parts) > 1:
                    owner_name = parts[-2]
                    if owner_name == module_name:
                        owner_type = module_type
                    else:
                        reference = module_references.get(owner_name)
                        owner_type = (
                            None
                            if reference is None
                            or reference.module_name not in opened
                            else reference.module_type
                        )
                    if owner_type is not None:
                        target = owner_type._tensor_schema_map().get(
                            target_field,
                        )
                        if (
                            target is not None
                            and target.tensor is not None
                            and tensor_is_active(target.tensor, opened)
                        ):
                            candidates.append((
                                owner_type.module_name,
                                target_field,
                            ))
                else:
                    local = module_type._tensor_schema_map().get(target_field)
                    if (
                        local is not None
                        and local.tensor is not None
                        and tensor_is_active(local.tensor, opened)
                    ):
                        candidates.append((module_name, target_field))
                    for reference in module_references.values():
                        if reference.module_name not in opened:
                            continue
                        target = reference.module_type._tensor_schema_map().get(
                            target_field,
                        )
                        if (
                            target is not None
                            and target.tensor is not None
                            and tensor_is_active(target.tensor, opened)
                        ):
                            candidates.append((
                                reference.module_name,
                                target_field,
                            ))

                if len(candidates) != 1:
                    raise ValueError(
                        f"Reference target {target_name!r} for "
                        f"{module_name}.{descriptor.reference} resolves to "
                        f"{len(candidates)} opened tensor fields; qualify the "
                        "target with its module name"
                    )
                target_module, resolved_field = candidates[0]
                plans[descriptor.reference] = _ReferenceTargetPlan(
                    target_module=target_module,
                    target_field=resolved_field,
                    qualified_name=f"{target_module}.{resolved_field}",
                )
                if descriptor.inverse:
                    inverse_sources.add(descriptor.reference)
            compiled[module_name] = MappingProxyType(plans)

        return MappingProxyType(compiled), frozenset(inverse_sources)

    def validate_inverse_reference_integrity(
        self, inverse_sources: frozenset[str],
    ) -> None:
        """Prove every inverse reference is a one-to-one global relation."""

        proxy = self.model._input.proxy
        for name in inverse_sources:
            if name not in proxy:
                continue
            values = self._numpy(
                proxy._get_value_trusted(name),
                label=f"inverse reference field {name!r}",
            ).reshape(-1)
            if np.unique(values).size != values.size:
                raise ValueError(
                    f"Inverse reference field {name!r} must contain unique "
                    "target references"
                )

    def reference_index(self, name: str) -> np.ndarray:
        cached = self._reference_indices.get(name)
        if cached is not None:
            return cached
        target = self._bare(self.schema.fields[name].references)
        proxy = self.model._input
        values = self._numpy(proxy[name])
        target_values = self._numpy(proxy[target])
        index = _find_indices_in_trusted(values, target_values)
        self._reference_indices[name] = index
        return index

    def coordinate_group_values(self, coordinate: str) -> np.ndarray:
        cached = self._coordinate_groups.get(coordinate)
        if cached is not None:
            return cached
        model = self.model
        metadata = self.schema.fields[coordinate]
        if coordinate == model.partition_key:
            groups = self._numpy(model._input[model.partition_group])
        else:
            via = self._bare(metadata.partition_by)
            references = self._bare(metadata.references)
            if via:
                target = self._bare(self.schema.fields[via].references)
                index = self.reference_index(via)
            else:
                target = references
                index = self.reference_index(coordinate)
            groups = self.coordinate_group_values(target)[index]
        self._coordinate_groups[coordinate] = groups
        return groups

    @property
    def group_ranks(self) -> GroupRankLookup:
        cached = self._group_ranks
        if cached is not None:
            return cached
        model = self.model
        ids, ranks = compute_group_to_rank(
            model.world_size,
            self._numpy(model._input[model.partition_group]),
        )
        cached = GroupRankLookup(group_ids=ids, ranks=ranks)
        self._group_ranks = cached
        return cached

    def rank_indices(self, coordinate: str) -> np.ndarray:
        groups = self.coordinate_group_values(coordinate)
        ranks = self.group_ranks._lookup_trusted(groups)
        return np.nonzero(ranks == self.model.rank)[0]

    @staticmethod
    def _numpy(value: Any, *, label: str | None = None) -> np.ndarray:
        del label
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def bind_output(
        self, field: ModuleFieldSchema,
    ) -> tuple[RuntimeTensorMetadata, dict[str, torch.Tensor]]:
        policy = field.tensor.output
        coordinate = (
            None if policy == "disabled" else self._bare(field.tensor.dim_coords)
        )
        index_name = None
        indices = None
        coordinate_tensor = None
        variable_map = self.model._namespace.build()
        if policy != "disabled" and coordinate:
            coordinate_entry = variable_map[coordinate]
            coordinate_tensor = getattr(
                coordinate_entry.module, coordinate_entry.field_name,
            )
            selection = (
                self.schema.selections.get(coordinate) if policy == "auto" else None
            )
            if selection:
                selection_entry = variable_map[selection]
                selected = getattr(
                    selection_entry.module, selection_entry.field_name,
                )
                if selected is not None:
                    indices = (
                        torch.empty(0, dtype=torch.int32, device=self.model.device)
                        if selected.numel() == 0
                        else _find_indices_in_torch_trusted(
                            selected, coordinate_tensor,
                        )
                    )
                    indices = indices.to(self.model.device)
                    index_name = f"__selection_idx__{selection}"
                    coordinate = selection
                    coordinate_tensor = selected
        bound = RuntimeTensorMetadata(
            tensor=field.tensor,
            description=field.description,
            output_index=index_name,
            output_coord=coordinate,
        )
        tensors: dict[str, torch.Tensor] = {}
        if index_name is not None and indices is not None:
            tensors[index_name] = indices
        if coordinate and coordinate_tensor is not None:
            tensors[coordinate] = coordinate_tensor
        return bound, tensors


class PartitionCompiler(_PartitionSemanticCompiler):
    """Trusted runtime partition service built from validated semantics."""

    def __init__(
        self,
        model: AbstractModel,
        *,
        schema: PartitionSchema,
        variable_groups: MappingProxyType,
    ) -> None:
        super().__init__(
            model,
            schema=schema,
            variable_groups=variable_groups,
        )

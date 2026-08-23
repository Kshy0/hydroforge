"""Internal parameter loading and rank-local slicing service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydroforge.contracts.events import emit

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class ModelDataCompiler:
    def __init__(self, model: AbstractModel) -> None:
        self.model = model

    def shard(self) -> dict[str, Any]:
        model = self.model
        partition = model._partition
        source = model._input
        fields = source.fields

        group_names = {
            partition.variable_groups[name]
            for name in fields
            if name in source
            and name in partition.variable_groups
        }
        group_indices = {
            group: partition.rank_indices(group)
            for group in group_names
        }
        emit(
            model, "info", "model.data_loading", "Loading module data",
            rank=model.rank, modules=tuple(model.opened_modules),
        )

        result: dict[str, Any] = {}
        missing: list[str] = []
        empty: dict[str, list[str]] = {}
        distributed: dict[tuple[tuple[int, ...], str], list[str]] = {}
        full: list[str] = []
        def field_order(item: tuple[str, Any]) -> tuple[int, str, str]:
            name = item[0]
            group = partition.variable_groups.get(name)
            return (0, "", name) if group is None else (1, group, name)

        ordered = sorted(fields.items(), key=field_order)
        for name, info in ordered:
            if name not in source:
                missing.append(name)
                continue
            group = partition.variable_groups.get(name)
            if group is None:
                result[name] = source[name]
                full.append(name)
                continue
            indices = group_indices[group]
            shape = source.get_var_shape(name)
            axis = model._semantic_plan.input_axes[name]
            selector = (slice(None), indices) if axis == 1 else indices
            local = source.get_subset(name, selector)
            result[name] = local
            if indices.size == 0:
                empty.setdefault(group, []).append(name)
            else:
                distributed.setdefault((local.shape, group), []).append(name)

        for group, names in empty.items():
            emit(
                model, "info", "model.data_empty_partition",
                "No local data for distributed fields", rank=model.rank,
                fields=tuple(names), coordinate=group,
            )
        for (shape, group), names in distributed.items():
            emit(
                model, "info", "model.data_distributed",
                "Loaded distributed fields", rank=model.rank,
                fields=tuple(names), shape=shape, coordinate=group,
            )
        if full:
            emit(
                model, "info", "model.data_full", "Loaded full-domain fields",
                rank=model.rank, fields=tuple(full),
            )
        if missing:
            emit(
                model, "info", "model.data_defaults",
                "Optional fields are absent; using defaults", rank=model.rank,
                fields=tuple(missing),
            )
        return result

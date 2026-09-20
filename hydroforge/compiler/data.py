"""Internal parameter loading and rank-local slicing service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydroforge.contracts.events import emit
from hydroforge.model.tensors import _ModulePayload

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class ModelDataCompiler:
    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self._prepared_modules: dict[str, _ModulePayload] | None = None
        self.consumed = False

    def prepare_modules(self) -> dict[str, _ModulePayload]:
        """Prepare rank-local module inputs once, without allocating defaults."""
        if self._prepared_modules is not None:
            return self._prepared_modules
        if self.consumed:
            raise RuntimeError("Consumed module inputs require a new data compiler")
        model = self.model
        module_data = model.shard_param()
        module_types = model._module_types()
        prepared: dict[str, _ModulePayload] = {}
        for name in model._module_order:
            module_class = module_types[name]
            payload = {
                field_name: value
                for field_name, value in module_data.items()
                if field_name in module_class.model_fields
            }
            payload.update(
                {
                    "opened_modules": model.opened_modules,
                    "rank": model.spatial_rank,
                    "device": model.device,
                    "precision": model.dtype,
                    "mixed_precision": model.mixed_precision,
                    "ensemble_size": model.local_ensemble_size,
                }
            )
            try:
                payload = module_class.prepare_module_input(payload)
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"module {name!r} prepare_module_input must return a dict"
                    )
                prepared[name] = _ModulePayload(
                    module_class,
                    payload,
                    prepared,
                    model._field_demand.required_for(name),
                    defer_defaults=True,
                )
            except (KeyError, TypeError, OverflowError) as error:
                raise ValueError(str(error)) from error
        self._prepared_modules = prepared
        return prepared

    def release(self) -> None:
        """Transfer input ownership to modules, or discard a failed attempt."""
        self._prepared_modules = None
        self.model._input.clear_cache()
        self.consumed = True

    def shard(self) -> dict[str, Any]:
        model = self.model
        partition = model._partition
        source = model._input
        fields = source.fields

        group_names = {
            partition.variable_groups[name]
            for name in fields
            if name in source and name in partition.variable_groups
        }
        group_indices = {group: partition.rank_indices(group) for group in group_names}
        emit(
            model,
            "info",
            "model.data_loading",
            "Loading module data",
            rank=model.rank,
            modules=tuple(model.opened_modules),
        )

        result: dict[str, Any] = {}
        missing: list[str] = []
        empty: dict[str, list[str]] = {}
        distributed: dict[tuple[tuple[int, ...], str], list[str]] = {}
        full: list[str] = []

        def field_order(name: str) -> tuple[int, str, str]:
            group = partition.variable_groups.get(name)
            return (0, "", name) if group is None else (1, group, name)

        for name in sorted(fields, key=field_order):
            if name not in source:
                missing.append(name)
                continue
            group = partition.variable_groups.get(name)
            if group is None:
                result[name] = source.read_local(name)
                full.append(name)
                continue
            indices = group_indices[group]
            local = source.read_local(name, indices)
            result[name] = local
            if indices.size == 0:
                empty.setdefault(group, []).append(name)
            else:
                distributed.setdefault((local.shape, group), []).append(name)

        for group, names in empty.items():
            emit(
                model,
                "info",
                "model.data_empty_partition",
                "No local data for distributed fields",
                rank=model.rank,
                fields=tuple(names),
                coordinate=group,
            )
        for (shape, group), names in distributed.items():
            emit(
                model,
                "info",
                "model.data_distributed",
                "Loaded distributed fields",
                rank=model.rank,
                fields=tuple(names),
                shape=shape,
                coordinate=group,
            )
        if full:
            emit(
                model,
                "info",
                "model.data_full",
                "Loaded full-domain fields",
                rank=model.rank,
                fields=tuple(full),
            )
        if missing:
            emit(
                model,
                "info",
                "model.data_defaults",
                "Optional fields are absent; using defaults",
                rank=model.rank,
                fields=tuple(missing),
            )
        return result

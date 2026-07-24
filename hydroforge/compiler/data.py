"""Internal parameter loading and rank-local slicing service."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hydroforge.contracts.events import emit
from hydroforge.contracts.fields import (
    cast_declared_tensor, concrete_tensor_dtype, tensor_is_active,
)


class ModelDataCompiler:
    def __init__(self, model: Any) -> None:
        self.model = model

    def shard(self) -> dict[str, Any]:
        model = self.model
        partition = model._partition
        fields: dict[str, Any] = {}
        schema = model.compiled_schema()
        for module_name in model.opened_modules:
            for field in schema.fields(module_name):
                if (
                    not field.computed
                    and not field.excluded
                    and tensor_is_active(field.tensor, model.opened_modules)
                ):
                    fields.setdefault(field.name, field)

        injected = getattr(model.input_proxy, "injected_vars", set())
        unknown = sorted(set(injected).difference(fields))
        if unknown:
            raise KeyError(
                f"Injected InputProxy variables are not opened-module fields: "
                f"{unknown}; available={sorted(fields)}"
            )
        missing_required = [
            name for name, info in fields.items()
            if info.required and name not in model.input_proxy
        ]
        if missing_required:
            raise KeyError(
                f"Required fields are missing from InputProxy: {missing_required}; "
                f"available={list(model.input_proxy.data)}"
            )
        partition.validate_input_axes(fields)

        group_names = {
            partition.variable_groups[name]
            for name in fields
            if name in model.input_proxy
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
        ordered = sorted(
            fields.items(),
            key=lambda item: (
                str(partition.variable_groups.get(item[0]) or ""), item[0],
            ),
        )
        for name, info in ordered:
            if name not in model.input_proxy:
                missing.append(name)
                continue
            group = partition.variable_groups.get(name)
            if group is None:
                result[name] = self._prepare(model.input_proxy[name], info)
                full.append(name)
                continue
            indices = group_indices[group]
            shape = model.input_proxy.get_var_shape(name)
            axis = partition.logical_axis(name, info, shape)
            selector = (slice(None), indices) if axis == 1 else indices
            local = model.input_proxy.get_subset(name, selector)
            result[name] = self._prepare(local, info)
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
        partition.validate_reference_integrity(result)
        return result

    def _prepare(self, value: Any, info: Any) -> Any:
        if info.tensor is not None:
            tensor = torch.as_tensor(value)
            dtype = concrete_tensor_dtype(
                info.tensor.dtype, self.model.dtype,
                self.model.mixed_precision,
            )
            if tensor.dtype != dtype:
                tensor = cast_declared_tensor(
                    tensor, dtype,
                    name=f"{info.module_name}.{info.name}",
                )
            return tensor if tensor.is_contiguous() else tensor.contiguous()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and (value.ndim == 0 or value.size == 1):
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

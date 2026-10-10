# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Typed boundary between external input storage and model tensors.

``InputProxy`` deliberately remains a storage abstraction: it can expose
NetCDF variables lazily and can also hold in-memory NumPy or Torch values.  A
model, however, has a much narrower contract.  ``InputBinding`` binds the
proxy to the compiled input fields when the runtime materializes, and is the
only place where external arrays become internal model tensors.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from hydroforge.compiler.partition import bare, coordinate_is_partitioned
from hydroforge.contracts.fields import cast_declared_tensor
from hydroforge.core.errors import error_message
from hydroforge.core.events import EventSink, ModelEvent, emit
from hydroforge.data.input import select_values
from hydroforge.declare.spec import FieldSpec, ModuleBinding
from hydroforge.declare.tensors import ModulePayload
from hydroforge.io.netcdf.read import normalize_selection

if TYPE_CHECKING:
    from hydroforge.compiler.plan import ModelPlan
    from hydroforge.data.input import InputProxy
    from hydroforge.execution.session import ModelRuntime


_TORCH_DTYPE_KINDS: Mapping[torch.dtype, str] = MappingProxyType(
    {
        torch.bool: "bool",
        torch.uint8: "integer",
        torch.uint16: "integer",
        torch.uint32: "integer",
        torch.int8: "integer",
        torch.int16: "integer",
        torch.int32: "integer",
        torch.int64: "integer",
        torch.float16: "floating",
        torch.bfloat16: "floating",
        torch.float32: "floating",
        torch.float64: "floating",
    }
)


def _dtype_kind(dtype: Any) -> str | None:
    if isinstance(dtype, torch.dtype):
        return _TORCH_DTYPE_KINDS.get(dtype)
    try:
        numpy_dtype = np.dtype(dtype)
    except TypeError:
        return None
    if numpy_dtype.kind == "b":
        return "bool"
    if numpy_dtype.kind in {"i", "u"}:
        return "integer"
    if numpy_dtype.kind == "f":
        return "floating"
    return None


def normalize_input_tensor(
    value: Any,
    field: FieldSpec,
    dtype: torch.dtype,
    *,
    device: torch.device,
    members: int | None,
    event_sink: EventSink,
) -> torch.Tensor:
    """Return one external value as independent, contiguous model storage.

    ``members`` is the local ensemble size; shared state then gains a member
    axis with independent storage per member.
    """

    name = field.name
    external_contiguous: bool | None = None
    owns_storage = False
    if isinstance(value, torch.Tensor):
        source = value.detach()
    elif isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        external_contiguous = bool(array.flags.c_contiguous)
        if not array.dtype.isnative or any(stride < 0 for stride in array.strides):
            array = np.array(
                array,
                dtype=array.dtype.newbyteorder("="),
                order="C",
                copy=True,
            )
            owns_storage = True
        # A read-only array is shared only until the copy below, which
        # every path without owned storage makes; it is never written.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The given NumPy array is not writable")
            source = torch.as_tensor(array)
    else:
        source = torch.as_tensor(np.asarray(value))

    original_dtype = source.dtype
    original_device = source.device
    original_contiguous = (
        source.is_contiguous() if external_contiguous is None else external_contiguous
    )
    if source.dtype != dtype:
        previous = source
        source = cast_declared_tensor(source, dtype, name=f"input.{name}")
        owns_storage = owns_storage or source is not previous
    tensor = source.to(device=device)
    owns_storage = owns_storage or tensor is not source
    shared_ensemble_state = (
        members is not None
        and field.tensor.category in {"state", "init_state"}
        and tensor.ndim == len(field.tensor.shape)
    )
    if shared_ensemble_state:
        tensor = tensor.unsqueeze(0).expand(members, *tensor.shape)
    if shared_ensemble_state or not owns_storage:
        tensor = tensor.detach().clone(memory_format=torch.contiguous_format)
    else:
        tensor = tensor.detach().contiguous()

    if (
        original_dtype != tensor.dtype
        and original_dtype.is_floating_point
        and tensor.dtype.is_floating_point
    ):
        event_sink.emit(
            ModelEvent(
                level="info",
                name="model.input_normalized",
                message="Normalized external input at the model boundary",
                fields={
                    "field": name,
                    "source_dtype": str(original_dtype),
                    "target_dtype": str(tensor.dtype),
                    "source_device": str(original_device),
                    "target_device": str(tensor.device),
                    "source_contiguous": original_contiguous,
                },
            )
        )
    return tensor


def _copy_scalar_or_object(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return np.array(value, order="C", copy=True)
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


class InputBinding:
    """Read view of one ``InputProxy`` bound to the compiled input fields.

    Construction validates the inventory, dtype families, ranks and logical
    axes against the proxy metadata. Every dtype/layout/device conversion
    happens here: consumers receive independent, contiguous tensors with the
    exact declared dtype on the model device.  Complete reads of stored values
    are cached here only, until ``clear_cache()`` after module construction.
    """

    def __init__(
        self, plan: ModelPlan, proxy: InputProxy, event_sink: EventSink
    ) -> None:
        self.plan = plan
        self.proxy = proxy
        self.event_sink = event_sink
        self.fields = plan.fields.inputs
        self._values: dict[str, Any] = {}
        self._validate_inventory()
        self._validate_declared_sources()
        self.axes = self._compile_axes()

    def _validate_inventory(self) -> None:
        injected = self.proxy.injected_vars
        unknown = sorted(set(injected).difference(self.fields))
        if unknown:
            raise KeyError(
                "Injected InputProxy variables are not opened-module fields: "
                f"{unknown}; available={sorted(self.fields)}"
            )
        missing = sorted(
            name
            for name, spec in self.fields.items()
            if spec.field.required and name not in self.proxy
        )
        spec = self.plan.spec
        # Every rank's ownership derives from the group of the partition key.
        if spec.partition_key is not None and spec.partition_group not in self.proxy:
            missing = sorted({*missing, spec.partition_group})
        if missing:
            raise KeyError(
                f"Required fields are missing from InputProxy: {missing}; "
                f"available={sorted(self.proxy.keys())}"
            )

    def _validate_declared_sources(self) -> None:
        members = self.plan.ensemble_size
        for name, spec in self.fields.items():
            if spec.dtype is None or name not in self.proxy:
                continue
            source_dtype = self.proxy._dtype(name)
            source_kind = _dtype_kind(source_dtype)
            target_kind = _dtype_kind(spec.dtype)
            if source_kind != target_kind:
                raise TypeError(
                    f"Input field {name!r} declares {target_kind} data "
                    f"({spec.dtype}) but source storage uses {source_dtype}"
                )
            shape = self.proxy._shape(name)
            tensor = spec.field.tensor
            logical_rank = len(tensor.shape)
            if members is None:
                allowed_ranks = (logical_rank,)
            elif tensor.category in {"state", "init_state", "param", "forcing"}:
                allowed_ranks = (logical_rank, logical_rank + 1)
            else:
                allowed_ranks = (logical_rank,)
            if len(shape) not in allowed_ranks:
                raise ValueError(
                    f"Input field {name!r} has rank {len(shape)}, but "
                    f"category {tensor.category!r} permits rank(s) "
                    f"{allowed_ranks} for ensemble_size={members}"
                )
            if len(shape) == logical_rank + 1 and (
                members is None or shape[0] != members
            ):
                raise ValueError(
                    f"Input field {name!r} has leading member size "
                    f"{shape[0]}, expected ensemble_size={members}"
                )

    def _compile_axes(self) -> Mapping[str, int]:
        """Compile global logical axes before any rank-local slicing."""

        plan = self.plan
        schema = plan.fields.partition
        partition_key = plan.spec.partition_key
        members = plan.ensemble_size
        axes: dict[str, int] = {}
        for name, spec in self.fields.items():
            tensor = spec.field.tensor
            if name not in self.proxy or tensor is None:
                continue
            shape = self.get_var_shape(name)
            logical_ndim = len(tensor.shape)
            if len(shape) == logical_ndim:
                axis = 0
            elif members is not None and len(shape) == logical_ndim + 1:
                if shape[0] != members:
                    raise ValueError(
                        f"Batched field '{name}' has leading size {shape[0]}, "
                        f"expected ensemble_size={members}."
                    )
                axis = 1
            else:
                raise ValueError(
                    f"Field '{name}' has rank {len(shape)}, but tensor_shape "
                    f"declares {logical_ndim} logical dimension(s)."
                )
            axes[name] = axis
            coordinate = bare(tensor.dim_coords)
            if (
                coordinate is None
                and name in schema.coordinates
                and coordinate_is_partitioned(schema, partition_key, name)
            ):
                coordinate = name
            if not coordinate:
                continue
            coordinate_shape = self.get_var_shape(coordinate)
            if len(coordinate_shape) != 1:
                raise ValueError(
                    f"Coordinate '{coordinate}' must be 1-D, got {coordinate_shape}."
                )
            if shape[axis] != coordinate_shape[0]:
                raise ValueError(
                    f"Field '{name}' logical axis length {shape[axis]} does not match "
                    f"dim_coords '{coordinate}' length {coordinate_shape[0]}."
                )
        return MappingProxyType(axes)

    def __contains__(self, name: str) -> bool:
        return name in self.proxy

    def get_var_shape(self, name: str) -> tuple[int, ...]:
        return self.proxy._shape(name)

    def value(self, name: str) -> Any:
        """Return the complete stored value; the caller must not modify it."""

        value = self._values.get(name)
        if value is None:
            value = self._values[name] = self.proxy._value(name)
        return value

    def __getitem__(self, name: str) -> Any:
        """Return the complete value as independent model storage."""

        return self._prepare(name, self.value(name))

    def _local_selector(
        self, name: str, spatial_indices=None
    ) -> tuple[Any, ...] | None:
        """The normalized rank-local selector of ``name``; ``None`` reads all."""

        axis = self.axes.get(name, 0)
        mesh = self.plan.parallel
        members = slice(None) if mesh is None else mesh.member_slice
        if spatial_indices is not None:
            selector = (members, spatial_indices) if axis == 1 else spatial_indices
        elif axis == 1:
            selector = members
        else:
            return None
        return normalize_selection(selector, self.proxy._shape(name))

    def local_metadata(
        self, name: str, spatial_indices=None
    ) -> tuple[tuple[int, ...], torch.dtype] | None:
        """Shape and dtype :meth:`read_local` returns, without reading values.

        ``None`` for fields without a declared tensor dtype, whose prepared
        value is a host scalar or object.
        """

        spec = self.fields.get(name)
        if spec is None or spec.dtype is None:
            return None
        stored = self.proxy._shape(name)
        selector = self._local_selector(name, spatial_indices)
        if selector is None:
            shape = stored
        else:
            selected: list[int] = []
            for axis, item in enumerate(selector):
                if isinstance(item, slice):
                    selected.append(len(range(*item.indices(stored[axis]))))
                elif isinstance(item, np.ndarray):
                    selected.append(int(item.size))
            shape = (*selected, *stored[len(selector) :])
        members = self.plan.local_ensemble_size
        tensor = spec.field.tensor
        if (
            members is not None
            and tensor.category in {"state", "init_state"}
            and len(shape) == len(tensor.shape)
        ):
            shape = (members, *shape)
        return tuple(int(extent) for extent in shape), spec.dtype

    def read_local(self, name: str, spatial_indices=None) -> Any:
        selector = self._local_selector(name, spatial_indices)
        if selector is None:
            return self[name]
        value = self._values.get(name)
        return self._prepare(
            name,
            self.proxy._subset(name, selector)
            if value is None
            else select_values(value, selector),
        )

    def clear_cache(self) -> None:
        """Release complete reads after transferring module input ownership."""
        self._values.clear()

    def _prepare(self, name: str, value: Any) -> Any:
        spec = self.fields.get(name)
        if spec is None or spec.dtype is None:
            return _copy_scalar_or_object(value)
        return normalize_input_tensor(
            value,
            spec.field,
            spec.dtype,
            device=self.plan.device,
            members=self.plan.local_ensemble_size,
            event_sink=self.event_sink,
        )


def shard_inputs(runtime: ModelRuntime) -> dict[str, Any]:
    """Read every present input field rank-locally, grouped by coordinate."""

    plan = runtime.plan
    source = runtime.input
    partition = runtime.partition
    groups = plan.fields.variable_groups
    group_indices = {
        group: partition.rank_indices(group)
        for group in {
            groups[name] for name in source.fields if name in source and name in groups
        }
    }
    emit(
        runtime,
        "info",
        "model.data_loading",
        "Loading module data",
        rank=plan.rank,
        modules=plan.modules,
    )
    result: dict[str, Any] = {}
    missing: list[str] = []
    empty: dict[str, list[str]] = {}
    distributed: dict[tuple[tuple[int, ...], str], list[str]] = {}
    full: list[str] = []

    def field_order(name: str) -> tuple[int, str, str]:
        group = groups.get(name)
        return (0, "", name) if group is None else (1, group, name)

    for name in sorted(source.fields, key=field_order):
        if name not in source:
            missing.append(name)
            continue
        group = groups.get(name)
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
            runtime,
            "info",
            "model.data_empty_partition",
            "No local data for distributed fields",
            rank=plan.rank,
            fields=tuple(names),
            coordinate=group,
        )
    for (shape, group), names in distributed.items():
        emit(
            runtime,
            "info",
            "model.data_distributed",
            "Loaded distributed fields",
            rank=plan.rank,
            fields=tuple(names),
            shape=shape,
            coordinate=group,
        )
    if full:
        emit(
            runtime,
            "info",
            "model.data_full",
            "Loaded full-domain fields",
            rank=plan.rank,
            fields=tuple(full),
        )
    if missing:
        emit(
            runtime,
            "info",
            "model.data_defaults",
            "Optional fields are absent; using defaults",
            rank=plan.rank,
            fields=tuple(missing),
        )
    return result


def prepare_payloads(
    runtime: ModelRuntime, values: Mapping[str, Any]
) -> dict[str, ModulePayload]:
    """Split rank-local values into prepared module payload views.

    Defaults stay unevaluated; sibling views let declared symbolic dimensions
    resolve before any module is constructed.
    """

    if not isinstance(values, Mapping):
        raise TypeError("prepare_model_input must return a Mapping")
    plan = runtime.plan
    prepared: dict[str, ModulePayload] = {}
    for name in plan.module_order:
        spec = plan.spec.modules[name]
        module_type = spec.module_type
        binding = plan.fields.modules[name]
        # Field names are shared across modules; an inactive declaration
        # must not receive another module's active value.
        payload = {
            field_name: value
            for field_name, value in values.items()
            if field_name in module_type.model_fields
            and (spec.fields[field_name].tensor is None or field_name in binding.active)
        }
        payload.update(
            {
                "opened_modules": plan.modules,
                "rank": plan.spatial_rank,
                "device": plan.device,
                "precision": plan.dtype,
                "mixed_precision": plan.mixed_precision,
                "metal_emulation": plan.metal_emulation,
                "ensemble_size": plan.local_ensemble_size,
                "init_mode": plan.init_mode,
                "options": plan.options,
            }
        )
        try:
            payload = module_type.prepare_module_input(payload)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(error_message(error)) from error
        if not isinstance(payload, dict):
            raise TypeError(f"module {name!r} prepare_module_input must return a dict")
        try:
            prepared[name] = ModulePayload(
                module_type,
                payload,
                ModuleBinding(
                    plan=binding,
                    references={
                        reference: prepared.get(reference)
                        for reference in spec.references
                    },
                    event_sink=runtime.event_sink,
                    prepared=True,
                ),
                defer_defaults=True,
            )
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(error_message(error)) from error
    return prepared

"""Typed boundary between external input storage and model tensors.

``InputProxy`` deliberately remains a storage abstraction: it can expose
NetCDF variables lazily and can also hold in-memory NumPy or Torch values.  A
model, however, has a much narrower contract.  This module binds the proxy to
the opened modules' Pydantic/TensorField schema exactly once and is the only
place where external arrays become internal model tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pydantic import Field, PrivateAttr, model_validator

from hydroforge.contracts.events import emit
from hydroforge.contracts.fields import (
    ModuleFieldSchema,
    cast_declared_tensor,
    concrete_tensor_dtype,
)
from hydroforge.contracts.runtime import MODEL_OWNED_MODULE_FIELDS
from hydroforge.contracts.validation import HydroForgeModel

if TYPE_CHECKING:
    from hydroforge.compiler.partition import PartitionCompiler
    from hydroforge.model.model import AbstractModel


_TORCH_DTYPE_KINDS: Mapping[torch.dtype, str] = MappingProxyType({
    torch.bool: "bool",
    torch.uint8: "integer",
    torch.int8: "integer",
    torch.int16: "integer",
    torch.int32: "integer",
    torch.int64: "integer",
    torch.float16: "floating",
    torch.bfloat16: "floating",
    torch.float32: "floating",
    torch.float64: "floating",
})


@dataclass(frozen=True, slots=True)
class TensorInputSpec:
    """One immutable external-to-internal tensor declaration."""

    name: str
    owners: tuple[str, ...]
    dtype: torch.dtype
    logical_rank: int
    category: str
    required: bool


class _ModelTensorPayload(HydroForgeModel):
    """Normalize one external model tensor into its canonical runtime value."""

    model: Any = Field(exclude=True)
    contract: Any = Field(exclude=True)
    value: Any = Field(exclude=True)

    _tensor: torch.Tensor = PrivateAttr()

    @model_validator(mode="after")
    def _normalize(self):
        contract = self.contract
        value = self.value
        name = contract.name
        external_contiguous: bool | None = None
        if isinstance(value, torch.Tensor):
            source = value.detach()
        elif isinstance(value, (np.ndarray, np.generic)):
            array = np.asarray(value)
            external_contiguous = bool(array.flags.c_contiguous)
            if (
                not array.dtype.isnative
                or not array.flags.writeable
                or any(stride < 0 for stride in array.strides)
            ):
                array = np.array(
                    array,
                    dtype=array.dtype.newbyteorder("="),
                    order="C",
                    copy=True,
                )
            source = torch.as_tensor(array)
        else:
            array = np.asarray(value)
            source = torch.as_tensor(array)

        original_dtype = source.dtype
        original_device = source.device
        original_contiguous = (
            source.is_contiguous()
            if external_contiguous is None else external_contiguous
        )
        if source.dtype != contract.dtype:
            source = cast_declared_tensor(
                source, contract.dtype, name=f"input.{name}",
            )
        tensor = source.to(device=self.model.device)
        trials = getattr(self.model, "num_trials", None)
        shared_trial_state = (
            trials is not None
            and contract.category in {"state", "init_state"}
            and tensor.ndim == contract.logical_rank
        )
        if shared_trial_state:
            tensor = tensor.unsqueeze(0).expand(trials, *tensor.shape)
        tensor = tensor.detach().clone(memory_format=torch.contiguous_format)

        if (
            original_dtype != tensor.dtype
            and original_dtype.is_floating_point
            and tensor.dtype.is_floating_point
        ):
            emit(
                self.model,
                "info",
                "model.input_normalized",
                "Normalized external input at the model boundary",
                field=name,
                source_dtype=str(original_dtype),
                target_dtype=str(tensor.dtype),
                source_device=str(original_device),
                target_device=str(tensor.device),
                source_contiguous=original_contiguous,
            )
        self._tensor = tensor
        return self

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor


class ModelInput:
    """Schema-bound read view used by initialization and partitioning.

    Every dtype/layout/device transformation happens here.  Consumers receive
    independent, contiguous tensors with the exact declared dtype and model
    device; downstream code therefore validates invariants but never repairs
    input types.
    """

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self.proxy = model.input_proxy
        fields = self._active_fields(model)
        self.fields: Mapping[str, ModuleFieldSchema] = MappingProxyType(fields)
        self.tensor_contracts = self._tensor_contracts(model, fields)
        self._full_cache: dict[str, Any] = {}
        self._validate_inventory()
        self._validate_declared_sources()

    @staticmethod
    def _active_fields(model: AbstractModel) -> dict[str, ModuleFieldSchema]:
        fields: dict[str, ModuleFieldSchema] = {}
        schema = model._compiled_schema()
        for module_name in model.opened_modules:
            for field in schema.fields(module_name):
                if (
                    field.computed
                    or field.excluded
                    or field.name in MODEL_OWNED_MODULE_FIELDS
                    or (
                        field.tensor is not None
                        and field.tensor.category == "forcing"
                    )
                    or not model._is_tensor_field_active(module_name, field)
                ):
                    continue
                fields.setdefault(field.name, field)
        return fields

    @staticmethod
    def _tensor_contracts(
        model: AbstractModel,
        fields: Mapping[str, ModuleFieldSchema],
    ) -> Mapping[str, TensorInputSpec]:
        owners: dict[str, list[str]] = {name: [] for name in fields}
        schema = model._compiled_schema()
        for module_name in model.opened_modules:
            for field in schema.fields(module_name):
                if field.name in fields and field.tensor is not None:
                    owners[field.name].append(module_name)
        contracts: dict[str, TensorInputSpec] = {}
        for name, field in fields.items():
            if field.tensor is None:
                continue
            contracts[name] = TensorInputSpec(
                name=name,
                owners=tuple(owners[name]),
                dtype=concrete_tensor_dtype(
                    field.tensor.dtype, model.dtype, model.mixed_precision,
                ),
                logical_rank=len(field.tensor.shape),
                category=field.tensor.category,
                required=field.required,
            )
        return MappingProxyType(contracts)

    def _validate_inventory(self) -> None:
        injected = self.proxy.injected_vars
        unknown = sorted(set(injected).difference(self.fields))
        if unknown:
            raise KeyError(
                "Injected InputProxy variables are not opened-module fields: "
                f"{unknown}; available={sorted(self.fields)}"
            )
        missing = sorted(
            name for name, field in self.fields.items()
            if field.required and name not in self.proxy
        )
        if missing:
            raise KeyError(
                f"Required fields are missing from InputProxy: {missing}; "
                f"available={sorted(self.proxy.keys())}"
            )

    def _validate_declared_sources(self) -> None:
        for name, contract in self.tensor_contracts.items():
            if name not in self.proxy:
                continue
            source_dtype = self.proxy._get_var_dtype(name)
            self._validate_dtype_family(
                name, source_dtype, contract.dtype,
            )
            self._validate_source_shape(
                contract, self.proxy._shape_for_trusted(name),
            )

    def _validate_source_shape(
        self, contract: TensorInputSpec, shape: tuple[int, ...],
    ) -> None:
        logical_rank = contract.logical_rank
        trials = self.model.num_trials
        if trials is None:
            allowed_ranks = (logical_rank,)
        elif contract.category in {"state", "init_state", "param", "forcing"}:
            allowed_ranks = (logical_rank, logical_rank + 1)
        else:
            allowed_ranks = (logical_rank,)
        if len(shape) not in allowed_ranks:
            raise ValueError(
                f"Input field {contract.name!r} has rank {len(shape)}, but "
                f"category {contract.category!r} permits rank(s) "
                f"{allowed_ranks} for num_trials={trials}"
            )
        if len(shape) == logical_rank + 1 and (
            trials is None or shape[0] != trials
        ):
            raise ValueError(
                f"Input field {contract.name!r} has leading trial size "
                f"{shape[0]}, expected num_trials={trials}"
            )

    @staticmethod
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

    @classmethod
    def _validate_dtype_family(
        cls, name: str, source_dtype: Any, target_dtype: torch.dtype,
    ) -> None:
        source_kind = cls._dtype_kind(source_dtype)
        target_kind = cls._dtype_kind(target_dtype)
        if source_kind != target_kind:
            raise TypeError(
                f"Input field {name!r} declares {target_kind} data "
                f"({target_dtype}) but source storage uses {source_dtype}"
            )

    def compile_partition_axes(
        self, partition: PartitionCompiler,
    ) -> Mapping[str, int]:
        """Compile global logical axes before any rank-local slicing."""

        return partition.compile_input_axes(dict(self.fields))

    @property
    def injected_vars(self) -> set[str]:
        return set(self.proxy.injected_vars)

    def keys(self) -> set[str]:
        return set(self.proxy.keys())

    def __contains__(self, name: str) -> bool:
        return name in self.proxy

    def get_var_shape(self, name: str) -> tuple[int, ...]:
        return self.proxy._shape_for_trusted(name)

    def __getitem__(self, name: str) -> Any:
        try:
            return self._full_cache[name]
        except KeyError:
            pass
        value = self._prepare(name, self.proxy._get_value_trusted(name))
        self._full_cache[name] = value
        return value

    def get_subset(self, name: str, selector: Any) -> Any:
        return self._prepare(
            name,
            self.proxy._get_subset_trusted(name, selector),
        )

    def _prepare(self, name: str, value: Any) -> Any:
        contract = self.tensor_contracts.get(name)
        if contract is None:
            return self._copy_scalar_or_object(name, value)
        return self._prepare_tensor(contract, value)

    @staticmethod
    def _copy_scalar_or_object(name: str, value: Any) -> Any:
        del name
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                return np.array(
                    value.detach().cpu().numpy(), order="C", copy=True,
                )
            return value.detach().cpu().item()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return np.array(value, order="C", copy=True)
        if isinstance(value, np.generic):
            return value.item()
        return deepcopy(value)

    def _prepare_tensor(
        self, contract: TensorInputSpec, value: Any,
    ) -> torch.Tensor:
        return _ModelTensorPayload(
            model=self.model,
            contract=contract,
            value=value,
        ).tensor


__all__: list[str] = []

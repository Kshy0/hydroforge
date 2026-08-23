"""Initialization-only canonical kernel argument resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

import torch
from pydantic import Field, PrivateAttr, model_validator

from hydroforge.contracts.fields import concrete_tensor_dtype
from hydroforge.contracts.kernels import ModuleEnabled, ModuleFlag
from hydroforge.contracts.runtime import (
    DEFAULT_BLOCK_SIZE,
)
from hydroforge.contracts.validation import HydroForgeModel

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class UnboundKernelArgument(KeyError):
    """A canonical ABI parameter has no owner in the model namespace."""


@dataclass(frozen=True, slots=True)
class BindingResolution:
    """One initialization-time canonical parameter resolution."""

    value: Any
    source: Literal["field", "feature", "optional", "model_config", "batched"]
    owner: str | None = None


class _KernelBindingRequest(HydroForgeModel):
    """One complete model-bound kernel call validated during recording."""

    binder: Any = Field(exclude=True)
    kernel: Any = Field(exclude=True)
    supplied: Mapping[str, Any]

    _arguments: Mapping[str, Any] = PrivateAttr()
    _buffer_dtypes: Mapping[str, torch.dtype] = PrivateAttr()

    @model_validator(mode="after")
    def _bind(self):
        try:
            arguments = self.binder._complete_trusted(
                self.kernel, dict(self.supplied),
            )
            buffer_dtypes = self.binder._buffer_dtypes_trusted(
                self.kernel, arguments,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        self._arguments = MappingProxyType(arguments)
        self._buffer_dtypes = buffer_dtypes
        return self

    @property
    def arguments(self) -> Mapping[str, Any]:
        return self._arguments

    @property
    def buffer_dtypes(self) -> Mapping[str, torch.dtype]:
        return self._buffer_dtypes


class KernelBinder:
    """Resolve exact KernelSpec names against one immutable model namespace."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        # Kernel entries are process-lifetime nominal operator objects.  Keep
        # the object itself as the key: an integer ``id`` can be reused after
        # collection and could otherwise return another kernel's ABI binding.
        self._complete_cache: dict[Any, Mapping[str, Any]] = {}

    def invalidate(self) -> None:
        """Drop bindings whose scalar specializations may have changed."""

        self._complete_cache.clear()

    @property
    def _field_index(self):
        """Use the immutable namespace compiled during model initialization."""
        return self.model._field_namespace

    def bind(
        self, kernel: Any, supplied: dict[str, Any],
    ) -> _KernelBindingRequest:
        return _KernelBindingRequest(
            binder=self, kernel=kernel, supplied=supplied,
        )

    def _complete_trusted(
        self, kernel: Any, supplied: dict[str, Any],
    ) -> dict[str, Any]:
        spec = kernel._spec
        if not supplied:
            cached = self._complete_cache.get(kernel)
            if cached is None:
                metadata = kernel.metadata
                values = {
                    parameter: self.resolve(
                        parameter,
                        metadata.optional_buffers,
                        metadata.optional_values,
                        spec.feature_sources,
                    ).value
                    for parameter in metadata.parameters
                    if parameter != "BLOCK_SIZE"
                }
                values["BLOCK_SIZE"] = self._block_size(kernel)
                cached = MappingProxyType(values)
                self._complete_cache[kernel] = cached
            return dict(cached)
        metadata = kernel.metadata
        for parameter in supplied:
            self.validate_dynamic(parameter, spec)
        arguments = dict(supplied)
        for parameter in metadata.parameters:
            if parameter == "BLOCK_SIZE":
                continue
            if parameter not in arguments:
                arguments[parameter] = self.resolve(
                    parameter,
                    metadata.optional_buffers,
                    metadata.optional_values,
                    spec.feature_sources,
                ).value
        arguments["BLOCK_SIZE"] = self._block_size(kernel)
        return arguments

    def validate_dynamic(self, parameter: str, spec: Any) -> None:
        """Require a call-site argument to have no model binding."""

        try:
            resolution = self.resolve(
                parameter,
                spec.optional_buffers,
                spec.optional_values,
                spec.feature_sources,
            )
        except UnboundKernelArgument:
            if parameter.startswith(("HAS_", "batched_")):
                raise
            return
        raise TypeError(
            f"{spec.name}.{parameter} is already resolved from "
            f"{resolution.source} {resolution.owner!r}; omit the redundant "
            "call-site value"
        )

    def _buffer_dtypes_trusted(
        self, kernel: Any, arguments: dict[str, Any],
    ) -> Mapping[str, torch.dtype]:
        """Compile buffer dtypes from declared model fields."""

        metadata = kernel.metadata
        feature_sources = kernel._spec.feature_sources
        result: dict[str, torch.dtype] = {}
        for parameter in metadata.buffers:
            value = arguments[parameter]
            declared = self._declared_buffer_dtype(
                parameter,
                metadata.optional_buffers.get(parameter),
                feature_sources,
                optional=parameter in metadata.optional_buffers,
            )
            if isinstance(value, torch.Tensor):
                if declared is not None and value.dtype != declared:
                    raise TypeError(
                        f"{metadata.name}.{parameter} has dtype {value.dtype}, "
                        f"but its model field declares {declared}"
                    )
                result[parameter] = value.dtype if declared is None else declared
                continue
            if value is not None or parameter not in metadata.optional_buffers:
                raise TypeError(
                    f"{metadata.name}.{parameter} has no concrete tensor dtype"
                )
            if declared is None:
                raise TypeError(
                    f"{metadata.name}.{parameter} is disabled and its dtype "
                    "cannot be resolved from a declared model/module field"
                )
            result[parameter] = declared
        return MappingProxyType(result)

    def _declared_buffer_dtype(
        self, parameter: str, feature: str | None,
        feature_sources: Mapping[str, Any], *, optional: bool,
    ) -> torch.dtype | None:
        field = parameter[:-4] if parameter.endswith("_ptr") else parameter
        matches = self._field_index.get(field, ())
        typed = []
        for match in matches:
            schema_getter = getattr(match.owner, "_get_tensor_schema", None)
            if schema_getter is not None and schema_getter(field) is None:
                continue
            getter = getattr(match.owner, "_get_expected_dtype", None)
            if getter is not None:
                typed.append((match.module_name, getter(field)))
        if len(typed) == 1:
            return typed[0][1]
        if len(typed) > 1:
            raise ValueError(
                f"buffer {parameter!r} has ambiguous dtype declarations in "
                f"{[name for name, _dtype in typed]}"
            )

        if feature is None and optional:
            declared = []
            for module_name, module_type in self.model._module_types().items():
                schema = module_type._get_tensor_schema(field)
                if (
                    schema is not None
                    and schema.tensor is not None
                    and not schema.tensor.expression
                ):
                    declared.append((module_name, schema.tensor.dtype))
            if len(declared) == 1:
                return self._concrete_dtype(declared[0][1])
            if len(declared) > 1:
                raise ValueError(
                    f"optional buffer {parameter!r} has ambiguous declarations "
                    f"in {[name for name, _kind in declared]}"
                )
            return None
        if feature is None:
            return None
        source = feature_sources.get(feature)
        if not isinstance(source, (ModuleEnabled, ModuleFlag)):
            return None
        module_name = source.module
        module_type = self.model._module_types().get(module_name)
        if module_type is None:
            return None
        schema = module_type._get_tensor_schema(field)
        if schema is None or schema.tensor is None:
            return None
        return self._concrete_dtype(schema.tensor.dtype)

    def _concrete_dtype(self, kind: str) -> torch.dtype:
        return concrete_tensor_dtype(
            kind, self.model.dtype, self.model.mixed_precision,
        )

    def _block_size(self, kernel: Any) -> int:
        """Resolve the configured launch width."""

        model = self.model
        backend = model._execution.backend
        value = model.BLOCK_SIZE
        if value is None:
            value = kernel.metadata.block_sizes.get(
                backend, DEFAULT_BLOCK_SIZE,
            )

        return value

    def resolve(
        self,
        parameter: str,
        optional_buffers: Any,
        optional_values: Any,
        feature_sources: Mapping[str, Any],
    ) -> BindingResolution:
        model = self.model
        if parameter in optional_values:
            flag, disabled = optional_values[parameter]
            if not self._feature(flag, feature_sources):
                return BindingResolution(disabled, "optional", flag)
        if parameter in optional_buffers:
            feature = optional_buffers[parameter]
            if feature is None:
                field = parameter.removesuffix("_ptr")
                matches = self._field_index.get(field, ())
                if not matches:
                    return BindingResolution(None, "optional", None)
                if len(matches) != 1:
                    self._raise_resolution(
                        parameter, [match.module_name for match in matches],
                    )
                match = matches[0]
                schema_getter = getattr(match.owner, "get_tensor_schema", None)
                schema = None if schema_getter is None else schema_getter(field)
                if (
                    schema is not None
                    and schema.tensor is not None
                    and schema.tensor.category == "virtual"
                    and field not in match.owner.__dict__
                ):
                    return BindingResolution(None, "optional", match.module_name)
                return BindingResolution(
                    getattr(match.owner, field),
                    "optional",
                    f"{match.module_name}.{field}",
                )
            if not self._feature(feature, feature_sources):
                return BindingResolution(
                    None, "optional", feature,
                )
        if parameter == "num_trials":
            return BindingResolution(
                1 if model.num_trials is None else model.num_trials,
                "model_config", "model",
            )
        if parameter in feature_sources:
            return BindingResolution(
                self._feature(parameter, feature_sources), "feature", parameter,
            )

        field = parameter[:-4] if parameter.endswith("_ptr") else parameter
        if field.startswith("batched_"):
            source = field.removeprefix("batched_")
            matches = self._field_index.get(source, ())
            if not matches:
                declared = [
                    module_name
                    for module_name in self.model._module_types()
                    if any(
                        item.name == source
                        for item in self.model._compiled_schema().fields(module_name)
                    )
                ]
                if (
                    len(declared) == 1
                    and declared[0] not in self.model.opened_modules
                ):
                    return BindingResolution(
                        False, "batched", declared[0],
                    )
            if len(matches) != 1:
                self._raise_resolution(
                    parameter, [match.module_name for match in matches],
                )
            return BindingResolution(
                matches[0].owner._is_batched_trusted(source),
                "batched",
                f"{matches[0].module_name}.{source}",
            )

        matches = self._field_index.get(field, ())
        if len(matches) != 1:
            self._raise_resolution(
                parameter, [match.module_name for match in matches],
            )
        match = matches[0]
        value = getattr(match.owner, field)
        if isinstance(value, Enum):
            value = value.value
        return BindingResolution(
            value,
            "field",
            f"{match.module_name}.{field}",
        )

    def _feature(
        self, parameter: str, feature_sources: Mapping[str, Any],
    ) -> bool:
        model = self.model
        source = feature_sources.get(parameter)
        if source is None:
            raise KeyError(
                f"kernel feature {parameter!r} has no explicit feature_source"
            )
        if isinstance(source, ModuleEnabled):
            if source.module not in model._module_types():
                raise KeyError(
                    f"kernel feature {parameter!r} references unknown model "
                    f"module {source.module!r}"
                )
            return source.module in model.opened_modules
        if isinstance(source, ModuleFlag):
            module_type = model._module_types().get(source.module)
            if module_type is None:
                raise KeyError(
                    f"kernel feature {parameter!r} references unknown "
                    f"model module {source.module!r}"
                )
            module = model._modules.get(source.module)
            if module is None:
                raise KeyError(
                    f"kernel feature {parameter!r} requires closed module "
                    f"{source.module!r}"
                )
            if source.field not in module_type.model_fields and not hasattr(
                module_type, source.field,
            ):
                raise KeyError(
                    f"kernel feature {parameter!r} references unknown field "
                    f"{source.module}.{source.field}"
                )
            value = getattr(module, source.field)
            if type(value) is not bool:
                raise TypeError(
                    f"kernel feature {parameter!r} source "
                    f"{source.module}.{source.field} must be an exact bool, "
                    f"got {type(value).__name__}"
                )
            return value
        raise TypeError(
            f"kernel feature {parameter!r} has invalid source "
            f"{type(source).__name__}"
        )

    @staticmethod
    def _raise_resolution(parameter: str, matches: Any) -> None:
        if matches:
            raise ValueError(
                f"kernel argument {parameter!r} is ambiguous across "
                f"{list(matches)}; kernel ABI names must match unique fields"
            )
        raise UnboundKernelArgument(
            f"kernel argument {parameter!r} has no model/module field; "
            "rename the ABI/field to match or supply it explicitly at the "
            "recording call site"
        )

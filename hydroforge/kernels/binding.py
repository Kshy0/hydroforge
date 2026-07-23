"""Initialization-only canonical kernel argument resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, Mapping

import torch

from hydroforge.contracts.fields import concrete_tensor_dtype


class UnboundKernelArgument(KeyError):
    """A canonical ABI parameter has no owner in the model namespace."""


@dataclass(frozen=True, slots=True)
class BindingResolution:
    """One initialization-time canonical parameter resolution."""

    value: Any
    source: Literal["field", "feature", "optional", "model_config", "batched"]
    owner: str | None = None


class KernelBinder:
    """Resolve exact KernelSpec names against one immutable model namespace."""

    def __init__(self, model: Any) -> None:
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
        return self.model._plan.kernels.fields.owners

    def bind(self, kernel: Any, *, dynamic: tuple[str, ...] = ()) -> Any:
        metadata = kernel.metadata
        if len(dynamic) != len(set(dynamic)):
            raise ValueError("dynamic kernel argument names must be unique")
        dynamic_names = frozenset(dynamic)
        unknown = dynamic_names.difference(metadata.parameters)
        if unknown:
            raise ValueError(
                f"dynamic kernel arguments are outside {metadata.name}: "
                f"{sorted(unknown)}"
            )
        for parameter in dynamic_names:
            self.validate_dynamic(parameter, metadata)
        bindings = {
            parameter: self.resolve(
                parameter, metadata.optional_buffers, metadata.optional_values,
            ).value
            for parameter in metadata.parameters
            if parameter not in dynamic_names
        }
        bindings["BLOCK_SIZE"] = self._block_size(kernel)
        return partial(kernel, **bindings)

    def complete(self, kernel: Any, supplied: dict[str, Any]) -> dict[str, Any]:
        if not supplied:
            cached = self._complete_cache.get(kernel)
            if cached is None:
                metadata = kernel.metadata
                values = {
                    parameter: self.resolve(
                        parameter,
                        metadata.optional_buffers,
                        metadata.optional_values,
                    ).value
                    for parameter in metadata.parameters
                }
                values["BLOCK_SIZE"] = self._block_size(kernel)
                cached = MappingProxyType(values)
                self._complete_cache[kernel] = cached
            return dict(cached)
        metadata = kernel.metadata
        extra = set(supplied).difference(metadata.parameters, {"BLOCK_SIZE"})
        if extra:
            raise TypeError(
                f"{metadata.name} received arguments outside its KernelSpec: "
                f"{sorted(extra)}"
            )
        if "BLOCK_SIZE" in supplied:
            raise TypeError(
                f"{metadata.name}.BLOCK_SIZE is compiler-owned; configure "
                "model.BLOCK_SIZE once instead of overriding a kernel launch"
            )
        # A call-site value is dynamic only when the canonical model namespace
        # has no matching field.  Never let an explicit value override a
        # unique field, a capability decision, or a disabled optional value;
        # those would create two competing sources for one ABI parameter.
        for parameter in supplied:
            self.validate_dynamic(parameter, metadata)
        arguments = dict(supplied)
        for parameter in metadata.parameters:
            if parameter not in arguments:
                arguments[parameter] = self.resolve(
                    parameter,
                    metadata.optional_buffers,
                    metadata.optional_values,
                ).value
        arguments["BLOCK_SIZE"] = self._block_size(kernel)
        return arguments

    def validate_dynamic(self, parameter: str, metadata: Any) -> None:
        """Prove that one call-site argument has no canonical model source.

        Runtime binding and the explicit ``check_step`` audit share this exact
        decision; neither is allowed to reinterpret a resolution failure as a
        dynamic value through a broader exception policy.
        """

        try:
            resolution = self.resolve(
                parameter,
                metadata.optional_buffers,
                metadata.optional_values,
            )
        except UnboundKernelArgument:
            # Capability and batching parameters are compiler-owned by
            # definition.  A missing declaration is an invalid model contract,
            # never permission to turn them into caller-provided values.
            if parameter.startswith(("HAS_", "batched_")):
                raise
            return
        raise TypeError(
            f"{metadata.name}.{parameter} is already resolved from "
            f"{resolution.source} {resolution.owner!r}; omit the redundant "
            "call-site value"
        )

    def buffer_dtypes(
        self, kernel: Any, arguments: dict[str, Any],
    ) -> Mapping[str, torch.dtype]:
        """Compile the concrete buffer ABI from declared model fields once.

        Present tensors are checked against their owning field declaration.
        Disabled optional buffers have no tensor value, so their declared
        field dtype is resolved without constructing the disabled module.
        Backends therefore never infer a pointee type from a null value.
        """

        metadata = kernel.metadata
        result: dict[str, torch.dtype] = {}
        for parameter in metadata.buffers:
            value = arguments[parameter]
            declared = self._declared_buffer_dtype(
                parameter,
                metadata.optional_buffers.get(parameter),
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
        self, parameter: str, feature: str | None, *, optional: bool,
    ) -> torch.dtype | None:
        field = parameter[:-4] if parameter.endswith("_ptr") else parameter
        matches = self._field_index.get(field, ())
        typed = []
        for match in matches:
            getter = getattr(match.owner, "get_expected_dtype", None)
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
            for module_name, module_type in self.model.module_list.items():
                schema = module_type.get_tensor_schema(field)
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
        if not feature.startswith("HAS_"):
            return None
        module_name = feature.removeprefix("HAS_").lower()
        module_type = self.model.module_list.get(module_name)
        if module_type is None:
            return None
        schema = module_type.get_tensor_schema(field)
        if schema is None or schema.tensor is None:
            return None
        return self._concrete_dtype(schema.tensor.dtype)

    def _concrete_dtype(self, kind: str) -> torch.dtype:
        return concrete_tensor_dtype(
            kind, self.model.dtype, self.model.mixed_precision,
        )

    def _block_size(self, kernel: Any) -> int:
        """Resolve one launch width from the canonical initialization policy.

        An explicit model value is the single user override. Otherwise the
        logical KernelSpec may choose a backend-specific performance default;
        kernels without one inherit the model's declared default. Backend
        adapters never own or infer launch widths.
        """

        model = self.model
        value = model.BLOCK_SIZE
        if "BLOCK_SIZE" not in model.model_fields_set:
            backend = model._execution.backend
            value = kernel.metadata.block_sizes.get(backend, value)
        else:
            backend = model._execution.backend
        from hydroforge.contracts.runtime import DEFAULT_BACKEND_REQUIREMENT

        rule = model.backend_requirements.get(
            backend, DEFAULT_BACKEND_REQUIREMENT,
        )
        rule.validate_block_size(value, backend=backend)
        return value

    def resolve(
        self,
        parameter: str,
        optional_buffers: Any,
        optional_values: Any,
    ) -> BindingResolution:
        model = self.model
        if parameter in optional_values:
            flag, disabled = optional_values[parameter]
            if not self._feature(flag):
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
            if not self._feature(feature):
                return BindingResolution(
                    None, "optional", feature,
                )
        if parameter == "BLOCK_SIZE":
            return BindingResolution(model.BLOCK_SIZE, "model_config", "model")
        if parameter == "num_trials":
            return BindingResolution(
                1 if model.num_trials is None else model.num_trials,
                "model_config", "model",
            )
        if parameter.startswith("HAS_"):
            return BindingResolution(
                self._feature(parameter), "feature", parameter,
            )

        field = parameter[:-4] if parameter.endswith("_ptr") else parameter
        if parameter.isupper() and field not in self._field_index:
            field = field.lower()
        if field.startswith("batched_"):
            source = field.removeprefix("batched_")
            matches = self._field_index.get(source, ())
            if len(matches) != 1:
                self._raise_resolution(
                    parameter, [match.module_name for match in matches],
                )
            return BindingResolution(
                matches[0].owner.is_batched(source),
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

    def _feature(self, parameter: str) -> bool:
        model = self.model
        if not parameter.startswith("HAS_"):
            raise ValueError(
                f"kernel feature {parameter!r} must use the canonical HAS_* name"
            )
        feature = parameter.removeprefix("HAS_").lower()
        if feature in model.module_list:
            return model.has_module(feature)
        if feature in model.feature_rules:
            return model.has_feature(feature)
        raise KeyError(
            f"kernel feature {parameter!r} does not name a module or feature rule"
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
            "rename the ABI/field to match or mark the argument dynamic"
        )

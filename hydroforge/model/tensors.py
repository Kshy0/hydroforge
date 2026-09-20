"""Initialization-only tensor compilation for one physics module."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from numbers import Integral
from typing import Any, get_args

import torch
from pydantic_core import PydanticUndefined

from hydroforge.contracts.fields import (
    concrete_tensor_dtype,
)


def copy_tensor_inputs(
    inputs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Stage inputs in mapping order using native ``Tensor.copy_`` semantics.

    Broadcasting, dtype conversion and device transfers are owned by Torch.
    Model setters validate their physical and optional-module semantics before
    calling this helper. Copies are not rolled back on failure: setters must
    use ``@between_steps`` to prevent subsequent execution of partial state.
    """

    for value, target in inputs.values():
        target.copy_(value)


class _ModulePayload:
    """Attribute view for declared shapes and Pydantic input completion.

    This is deliberately not a partially constructed ``AbstractModule``.  It
    exposes the prepared field mapping, sibling declaration/module views and class
    descriptors needed to evaluate declared symbolic dimensions.  The
    completed mapping is then passed to Pydantic for the one real module
    construction.
    """

    def __init__(
        self,
        module_type: type,
        payload: dict[str, Any],
        module_references: dict[str, Any],
        output_required_fields: frozenset[str],
        *,
        default_values: dict[str, Any] | None = None,
        defer_defaults: bool = False,
    ) -> None:
        object.__setattr__(self, "_module_type", module_type)
        object.__setattr__(self, "_input_values", dict(payload))
        object.__setattr__(self, "_model_fields_set", frozenset(payload))
        object.__setattr__(
            self,
            "_default_values",
            {} if default_values is None else default_values,
        )
        object.__setattr__(
            self,
            "_output_required_fields",
            output_required_fields,
        )
        object.__setattr__(
            self,
            "_reference_values",
            {
                name: module_references.get(descriptor.module_name)
                for name, descriptor in module_type._module_reference_fields().items()
            },
        )
        for name in module_type.model_fields:
            if name in payload:
                value = payload[name]
            elif defer_defaults:
                continue
            else:
                value = self._default_value(name)
                if value is PydanticUndefined:
                    continue
            object.__setattr__(self, name, value)

    def _default_value(self, name: str) -> Any:
        defaults = self._default_values
        if name not in defaults:
            defaults[name] = self._module_type.model_fields[name].get_default(
                call_default_factory=True,
            )
        return defaults[name]

    @property
    def model_fields_set(self) -> frozenset[str]:
        return self._model_fields_set

    def __getattr__(self, name: str) -> Any:
        references = self._reference_values
        if name in references:
            return references[name]
        module_type = self._module_type
        if name in module_type.model_fields:
            value = self._default_value(name)
            if value is PydanticUndefined:
                raise AttributeError(name)
            object.__setattr__(self, name, value)
            return value
        try:
            descriptor = inspect.getattr_static(module_type, name)
        except AttributeError as error:
            raise AttributeError(name) from error
        if hasattr(descriptor, "__get__"):
            return descriptor.__get__(self, module_type)
        return descriptor

    def _completed(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self._module_type.model_fields
            if hasattr(self, name)
        }


class ModuleTensors:
    """Materialize and validate a module's declared tensor schema once."""

    def __init__(
        self,
        module: Any,
        *,
        batched_fields: tuple[str, ...] = (),
    ) -> None:
        self.module = module
        self.batched_fields: set[str] = set(batched_fields)

    @classmethod
    def _prepare_payload(
        cls,
        module_type: type,
        payload: dict[str, Any],
        *,
        module_references: dict[str, Any],
        batched_fields: tuple[str, ...] = (),
        output_required_fields: frozenset[str] = frozenset(),
        default_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Complete scalar tensor defaults inside Pydantic validation."""

        view = _ModulePayload(
            module_type,
            payload,
            module_references,
            output_required_fields,
            default_values=None if default_values is None else dict(default_values),
        )
        tensors = cls(view, batched_fields=batched_fields)
        tensors._deactivate_declared()
        tensors._initialize_optional()
        return view._completed()

    def _finalize_computed(self) -> None:
        """Resolve computed tensor residency and validate active values."""

        module = self.module
        computed_fields = tuple(
            field
            for field in module.tensor_schema()
            if field.computed and field.tensor.category != "virtual"
        )
        for field in computed_fields:
            if not module._is_tensor_field_active(field):
                object.__setattr__(module, field.name, None)
        for field in computed_fields:
            if module._is_tensor_field_active(field):
                self._validate_computed_field(
                    field.name,
                    getattr(module, field.name),
                )
        # Derived reference indices are descriptors rather than Pydantic
        # computed fields, but belong to the same stable cold-start phase.
        for name in module._reference_index_fields(
            opened_modules=module.opened_modules,
            field_demand=module._field_demand,
        ):
            getattr(module, name)

    def _deactivate_declared(self) -> None:
        module = self.module
        for field in module.tensor_schema():
            if field.computed or module._is_tensor_field_active(field):
                continue
            if field.name in module.model_fields_set:
                required = ", ".join(field.tensor.depends_on)
                consumers = ", ".join(field.tensor.required_by)
                dependencies = required
                if consumers:
                    dependencies = (
                        f"{dependencies}; required by any of: {consumers}"
                        if dependencies
                        else f"required by any of: {consumers}"
                    )
                raise ValueError(
                    f"Inactive field {module.module_name}.{field.name} was "
                    f"supplied explicitly; open its dependencies: {dependencies}"
                )
            object.__setattr__(module, field.name, None)

    def _expected_shape(self, field_name: str) -> tuple[int, ...] | None:
        module = self.module
        schema = module._tensor_schema_map().get(field_name)
        if schema is None:
            raise ValueError(f"Field {field_name} is not a tensor field")
        if not module._is_tensor_field_active(schema):
            return None
        values: dict[Any, Any] = {}
        for dimension in schema.tensor.shape:
            if dimension in values:
                continue
            if isinstance(dimension, int):
                values[dimension] = dimension
                continue
            if "." in dimension:
                owner_name, attribute = dimension.split(".", 1)
                owner = getattr(module, owner_name, None)
            else:
                owner, attribute = module, dimension
            try:
                values[dimension] = getattr(owner, attribute)
            except AttributeError as error:
                raise ValueError(
                    f"Dimension {dimension!r} is not available to "
                    f"module {module.module_name!r}"
                ) from error
        shape = tuple(values[dimension] for dimension in schema.tensor.shape)
        for dimension, size in zip(schema.tensor.shape, shape, strict=True):
            if isinstance(size, bool) or not isinstance(size, Integral):
                raise ValueError(
                    f"Dimension '{dimension}' used by field '{field_name}' must "
                    f"be an integer, got {type(size).__name__}"
                )
            if size < 0:
                raise ValueError(
                    f"Dimension '{dimension}' used by field '{field_name}' must "
                    f"be non-negative, got {size}"
                )
        if module.ensemble_size is not None:
            category = schema.tensor.category
            batched = category in {"state", "init_state"} or (
                category in {"param", "derived_param", "forcing"}
                and field_name in self.batched_fields
            )
            if batched:
                return (module.ensemble_size, *shape)
        return shape

    def _expected_dtype(self, field_name: str) -> torch.dtype:
        module = self.module
        schema = module._tensor_schema_map().get(field_name)
        if schema is None:
            schema = module._get_tensor_schema(
                field_name,
                opened_modules=module.opened_modules,
                field_demand=module._field_demand,
            )
        if schema is None:
            raise ValueError(f"Field {field_name} is not a tensor field")
        return concrete_tensor_dtype(
            schema.tensor.dtype,
            module.precision,
            module.mixed_precision,
        )

    def _initialize_optional(self) -> None:
        module = self.module
        for schema in module.tensor_schema():
            if (
                schema.computed
                or schema.name in module.model_fields_set
                or not module._is_tensor_field_active(schema)
            ):
                continue
            shape = self._expected_shape(schema.name)
            if shape is None:
                continue
            value = getattr(module, schema.name, None)
            if value is None:
                tensor = None
            elif isinstance(value, (int, float, bool)):
                tensor = torch.full(
                    shape,
                    value,
                    dtype=self._expected_dtype(schema.name),
                    device=module.device,
                )
            else:
                raise ValueError(
                    f"Unsupported default type for {schema.name}: {type(value)}"
                )
            object.__setattr__(module, schema.name, tensor)

    def _validate_declared(self) -> None:
        """Assert the input-boundary contract without repairing tensors."""

        module = self.module
        fields = tuple(
            field
            for field in module.tensor_schema()
            if not field.computed and module._is_tensor_field_active(field)
        )
        for field in fields:
            name = field.name
            tensor = getattr(module, name, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            expected = self._expected_shape(name)
            if expected is not None and tuple(tensor.shape) != expected:
                tensor = self._resolve_batch_shape(field, tensor, expected)
            if not tensor.is_contiguous():
                raise ValueError(
                    f"Input field {module.module_name}.{name} must be "
                    "contiguous before module construction"
                )
            if not self._on_device(tensor):
                raise ValueError(
                    f"Input field {module.module_name}.{name} must already be "
                    f"on device {module.device}, got {tensor.device}"
                )
            dtype = self._expected_dtype(name)
            if tensor.dtype != dtype:
                raise ValueError(
                    f"Input field {module.module_name}.{name} must already use "
                    f"dtype {dtype}, got {tensor.dtype}"
                )
            self._validate_key(field, tensor)

    def _resolve_batch_shape(
        self,
        field: Any,
        tensor: torch.Tensor,
        expected: tuple[int, ...],
    ) -> torch.Tensor:
        module = self.module
        name = field.name
        category = field.tensor.category
        if (
            category in {"param", "derived_param"}
            and module.ensemble_size is not None
            and tensor.ndim > 0
            and tensor.shape[0] == module.ensemble_size
            and tuple(tensor.shape[1:]) == expected
        ):
            self.batched_fields.add(name)
            if tuple(tensor.shape) == self._expected_shape(name):
                return tensor
        raise ValueError(
            f"Shape mismatch for {name}: expected {expected}, got {tuple(tensor.shape)}"
        )

    @staticmethod
    def _validate_key(field: Any, tensor: torch.Tensor) -> None:
        if not field.tensor.is_key:
            return
        if tensor.dtype not in {torch.int32, torch.int64} or tensor.ndim != 1:
            raise ValueError(
                f"Key field '{field.name}' must be a one-dimensional integer tensor"
            )
        if not tensor.numel():
            return
        values, counts = torch.unique(tensor, return_counts=True)
        duplicate = counts > 1
        if bool(duplicate.any()):
            raise ValueError(
                f"Key field '{field.name}' has "
                f"{int(duplicate.sum().item())} duplicate value(s); first few: "
                f"{values[duplicate][:5].tolist()}"
            )

    def _validate_computed_field(
        self,
        field_name: str,
        value: Any,
    ) -> None:
        module = self.module
        field = module._tensor_schema_map()[field_name]
        return_type = type(module).model_computed_fields[field_name].return_type
        if value is None and type(None) in get_args(return_type):
            return
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Computed field {field.name} must be a torch.Tensor, got "
                f"{type(value).__name__}"
            )
        tensor = value
        if not self._on_device(tensor):
            raise ValueError(
                f"Computed field {field.name} must be on device "
                f"{module.device}, but is on {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"Computed field {field.name} must be contiguous; computed "
                "fields are never repaired implicitly"
            )
        expected = self._expected_shape(field.name)
        if expected is not None and tuple(tensor.shape) != expected:
            if (
                field.tensor.category == "derived_param"
                and module.ensemble_size is not None
                and tuple(tensor.shape) == (module.ensemble_size, *expected)
            ):
                self.batched_fields.add(field.name)
            else:
                raise ValueError(
                    f"Computed field {field.name} has shape "
                    f"{tuple(tensor.shape)}, expected {expected}"
                )
        dtype = self._expected_dtype(field.name)
        if tensor.dtype != dtype:
            raise ValueError(
                f"Computed field {module.module_name}.{field.name} must use "
                f"dtype {dtype}, got {tensor.dtype}"
            )

    def _on_device(self, tensor: torch.Tensor) -> bool:
        expected = self.module.device
        return bool(
            tensor.device.type == expected.type
            and (
                tensor.device.index is None
                or expected.index is None
                or tensor.device.index == expected.index
            )
        )

    def _apply_modes(self) -> None:
        module = self.module
        for field in module.tensor_schema():
            if (
                field.computed
                or not module._is_tensor_field_active(field)
                or field.tensor.mode == "device"
            ):
                continue
            value = getattr(module, field.name)
            if not isinstance(value, torch.Tensor):
                continue
            if field.tensor.mode == "cpu":
                object.__setattr__(module, field.name, value.cpu())
            elif field.tensor.mode == "discard":
                object.__setattr__(module, field.name, None)

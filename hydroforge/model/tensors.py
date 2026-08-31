"""Initialization-only tensor compilation for one physics module."""

from __future__ import annotations

import inspect
from numbers import Integral
from typing import Any, get_args

import torch
from pydantic_core import PydanticUndefined

from hydroforge.contracts.fields import (
    concrete_tensor_dtype,
)


class _ModulePayload:
    """Attribute view used only while a Pydantic before-validator completes input.

    This is deliberately not a partially constructed ``AbstractModule``.  It
    exposes the raw field mapping, already validated sibling modules and class
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
    ) -> None:
        object.__setattr__(self, "_module_type", module_type)
        object.__setattr__(self, "_model_fields_set", frozenset(payload))
        object.__setattr__(
            self, "_output_required_fields", output_required_fields,
        )
        object.__setattr__(self, "_reference_values", {
            name: module_references.get(descriptor.module_name)
            for name, descriptor in
            module_type._module_reference_fields().items()
        })
        for name, field in module_type.model_fields.items():
            if name in payload:
                value = payload[name]
            else:
                value = field.get_default(call_default_factory=True)
                if value is PydanticUndefined:
                    continue
            object.__setattr__(self, name, value)

    @property
    def model_fields_set(self) -> frozenset[str]:
        return self._model_fields_set

    def __getattr__(self, name: str) -> Any:
        references = self._reference_values
        if name in references:
            return references[name]
        module_type = self._module_type
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
    ) -> dict[str, Any]:
        """Complete scalar tensor defaults inside Pydantic validation."""

        view = _ModulePayload(
            module_type, payload, module_references, output_required_fields,
        )
        tensors = cls(view, batched_fields=batched_fields)
        tensors._deactivate_declared()
        tensors._initialize_optional()
        return view._completed()

    def _initialize_declared(self) -> None:
        """Validate the complete tensor payload supplied to Pydantic."""

        self._validate_declared()

    def _finalize_computed(self) -> None:
        """Resolve computed tensor residency and validate active values."""

        module = self.module
        computed_fields = tuple(
            field for field in module.tensor_schema()
            if field.computed and field.tensor.category != "virtual"
        )
        for field in computed_fields:
            if not module._is_tensor_field_active(field):
                object.__setattr__(module, field.name, None)
        for field in computed_fields:
            if module._is_tensor_field_active(field):
                self._validate_computed_field(
                    field.name, getattr(module, field.name),
                )
        # Derived reference indices are descriptors rather than Pydantic
        # computed fields, but belong to the same stable cold-start phase.
        for name in module._reference_index_fields():
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
                        if dependencies else f"required by any of: {consumers}"
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
            if isinstance(dimension, int):
                values[dimension] = dimension
                continue
            if "." in dimension:
                owner_name, attribute = dimension.split(".", 1)
                owner = getattr(module, owner_name, None)
                if owner is None or not hasattr(owner, attribute):
                    raise ValueError(
                        f"Dimension {dimension!r} is not available to "
                        f"module {module.module_name!r}"
                    )
                values[dimension] = getattr(owner, attribute)
            elif hasattr(module, dimension):
                values[dimension] = getattr(module, dimension)
            else:
                raise ValueError(
                    f"Dimension {dimension!r} is not available to "
                    f"module {module.module_name!r}"
                )
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
        if module.num_trials is not None:
            category = schema.tensor.category
            batched = category in {"state", "init_state"} or (
                category in {"param", "derived_param", "forcing"}
                and field_name in self.batched_fields
            )
            if batched:
                return (module.num_trials, *shape)
        return shape

    def _expected_dtype(self, field_name: str) -> torch.dtype:
        module = self.module
        schema = module._get_tensor_schema(field_name)
        if schema is None:
            raise ValueError(f"Field {field_name} is not a tensor field")
        return concrete_tensor_dtype(
            schema.tensor.dtype, module.precision, module.mixed_precision,
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
                    shape, value, dtype=self._expected_dtype(schema.name),
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
            field for field in module.tensor_schema()
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
        self, field: Any, tensor: torch.Tensor, expected: tuple[int, ...],
    ) -> torch.Tensor:
        module = self.module
        name = field.name
        category = field.tensor.category
        if (
            category in {"param", "derived_param"}
            and module.num_trials is not None
            and tensor.ndim > 0
            and tensor.shape[0] == module.num_trials
            and tuple(tensor.shape[1:]) == expected
        ):
            self.batched_fields.add(name)
            if tuple(tensor.shape) == self._expected_shape(name):
                return tensor
        raise ValueError(
            f"Shape mismatch for {name}: expected {expected}, "
            f"got {tuple(tensor.shape)}"
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
        self, field_name: str, value: Any,
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
                and module.num_trials is not None
                and tuple(tensor.shape) == (module.num_trials, *expected)
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

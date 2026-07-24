"""Initialization-only tensor compilation for one physics module."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch

from hydroforge.contracts.fields import (
    cast_declared_tensor, concrete_tensor_dtype, tensor_is_active,
)


class ModuleTensors:
    """Materialize and validate a module's declared tensor schema once."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.expanded_parameters: set[str] = set()

    def initialize(self) -> None:
        self._deactivate_declared()
        self._normalize_declared()
        self._initialize_optional()
        self._validate_computed()

    def _deactivate_declared(self) -> None:
        module = self.module
        for field in module.tensor_schema():
            if field.computed or module.is_tensor_field_active(field):
                continue
            if field.name in module.model_fields_set:
                dependencies = ", ".join(field.tensor.depends_on)
                raise ValueError(
                    f"Inactive field {module.module_name}.{field.name} was "
                    f"supplied explicitly; open its dependencies: {dependencies}"
                )
            setattr(module, field.name, None)

    def expected_shape(self, field_name: str) -> tuple[int, ...] | None:
        module = self.module
        schema = module.tensor_schema_map().get(field_name)
        if schema is None:
            raise ValueError(f"Field {field_name} is not a tensor field")
        if not tensor_is_active(schema.tensor, module.opened_modules):
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
                raise TypeError(
                    f"Dimension '{dimension}' used by field '{field_name}' must "
                    f"be an integer, got {type(size).__name__}"
                )
            if size < 0 or (size == 0 and not schema.tensor.allow_empty):
                requirement = (
                    "non-negative" if schema.tensor.allow_empty else "positive"
                )
                raise ValueError(
                    f"Dimension '{dimension}' used by field '{field_name}' must "
                    f"be {requirement}, got {size}"
                )
        if module.num_trials is not None:
            category = schema.tensor.category
            batched = category in {"state", "init_state"} or (
                category in {"param", "derived_param"}
                and field_name in self.expanded_parameters
            )
            if batched:
                return (module.num_trials, *shape)
        return shape

    def expected_dtype(self, field_name: str) -> torch.dtype:
        module = self.module
        schema = module.get_tensor_schema(field_name)
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
                or not module.is_tensor_field_active(schema)
            ):
                continue
            shape = self.expected_shape(schema.name)
            if shape is None:
                continue
            value = getattr(module, schema.name, None)
            if value is None:
                tensor = None
            elif isinstance(value, (int, float, bool)):
                tensor = torch.full(
                    shape, value, dtype=self.expected_dtype(schema.name),
                    device=module.device,
                )
            else:
                raise TypeError(
                    f"Unsupported default type for {schema.name}: {type(value)}"
                )
            setattr(module, schema.name, tensor)

    def _normalize_declared(self) -> None:
        module = self.module
        conversions: dict[str, list[str]] = {}
        fields = tuple(
            field for field in module.tensor_schema()
            if not field.computed and module.is_tensor_field_active(field)
        )
        # Shape expressions can depend on topology tensors, so place every
        # supplied tensor before evaluating any expected shape.
        for field in fields:
            tensor = getattr(module, field.name, None)
            if isinstance(tensor, torch.Tensor) and not self._on_device(tensor):
                setattr(module, field.name, tensor.to(module.device))

        for field in fields:
            name = field.name
            tensor = getattr(module, name, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            expected = self.expected_shape(name)
            if expected is not None and tuple(tensor.shape) != expected:
                tensor = self._resolve_batch_shape(field, tensor, expected)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if not self._on_device(tensor):
                tensor = tensor.to(module.device)
            dtype = self.expected_dtype(name)
            if tensor.dtype != dtype:
                conversion = f"{tensor.dtype} -> {dtype}"
                conversions.setdefault(conversion, []).append(name)
                tensor = cast_declared_tensor(
                    tensor, dtype, name=f"{module.module_name}.{name}",
                )
            self._validate_key(field, tensor)
            setattr(module, name, tensor)
        self._emit_conversions("module.dtype_fixed", conversions)

    def _resolve_batch_shape(
        self, field: Any, tensor: torch.Tensor, expected: tuple[int, ...],
    ) -> torch.Tensor:
        module = self.module
        name = field.name
        category = field.tensor.category
        if (
            category in {"state", "init_state"}
            and module.num_trials is not None
            and tuple(tensor.shape) == expected[1:]
        ):
            return tensor.unsqueeze(0).expand(expected).clone()
        if (
            category in {"param", "derived_param"}
            and module.num_trials is not None
            and tensor.ndim > 0
            and tensor.shape[0] == module.num_trials
            and tuple(tensor.shape[1:]) == expected
        ):
            self.expanded_parameters.add(name)
            if tuple(tensor.shape) == self.expected_shape(name):
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

    def _validate_computed(self) -> None:
        module = self.module
        conversions: dict[str, list[str]] = {}
        for field in module.tensor_schema():
            if not field.computed or field.tensor.category == "virtual":
                continue
            if not module.is_tensor_field_active(field):
                continue
            tensor = getattr(module, field.name)
            if not isinstance(tensor, torch.Tensor):
                continue
            if not self._on_device(tensor):
                raise ValueError(
                    f"Computed field {field.name} must be on device "
                    f"{module.device}, but is on {tensor.device}"
                )
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            expected = self.expected_shape(field.name)
            if expected is not None and tuple(tensor.shape) != expected:
                if (
                    module.num_trials is not None
                    and tuple(tensor.shape) == (module.num_trials, *expected)
                ):
                    self.expanded_parameters.add(field.name)
                else:
                    raise ValueError(
                        f"Computed field {field.name} has shape "
                        f"{tuple(tensor.shape)}, expected {expected}"
                    )
            dtype = self.expected_dtype(field.name)
            if tensor.dtype != dtype:
                conversion = f"{tensor.dtype} -> {dtype}"
                conversions.setdefault(conversion, []).append(field.name)
                tensor = cast_declared_tensor(
                    tensor, dtype,
                    name=f"{module.module_name}.{field.name}",
                )
            setattr(module, field.name, tensor)
        self._emit_conversions("module.computed_dtype_fixed", conversions)

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

    def _emit_conversions(
        self, event: str, conversions: dict[str, list[str]],
    ) -> None:
        for conversion, fields in conversions.items():
            self.module._emit(
                "info", event, "Normalized module field dtypes",
                module=self.module.module_name, fields=fields,
                conversion=conversion,
            )

    def apply_modes(self) -> None:
        module = self.module
        for field in module.tensor_schema():
            if (
                field.computed
                or not module.is_tensor_field_active(field)
                or field.tensor.mode == "device"
            ):
                continue
            value = getattr(module, field.name)
            if not isinstance(value, torch.Tensor):
                continue
            if field.tensor.mode == "cpu":
                setattr(module, field.name, value.cpu())
            elif field.tensor.mode == "discard":
                setattr(module, field.name, None)

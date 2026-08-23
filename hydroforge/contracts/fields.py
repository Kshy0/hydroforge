"""Generic field contracts extracted from module declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import cache
from typing import Any, Self, TypeAlias

import torch
from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.validation import (
    HydroForgeModel,
    _immutable_dict,
)


ModuleType: TypeAlias = type[Any]
DimensionToken: TypeAlias = str | int


def tensor_is_active(
    metadata: TensorMetadata | RuntimeTensorMetadata | None,
    opened_modules: Iterable[str],
) -> bool:
    """Return whether field dependencies and consumer requirements hold."""
    opened = set(opened_modules)
    required = getattr(metadata, "depends_on", ())
    consumers = getattr(metadata, "required_by", ())
    return all(
        dependency in opened
        for dependency in required
    ) and (not consumers or any(item in opened for item in consumers))


def concrete_tensor_dtype(
    kind: str, base_dtype: torch.dtype, mixed_precision: bool,
) -> torch.dtype:
    """Resolve one semantic TensorField dtype without intermediate casting."""

    if base_dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            f"base tensor precision must be float32 or float64, got {base_dtype}"
        )
    if type(mixed_precision) is not bool:
        raise TypeError("mixed_precision must be an exact bool")
    try:
        return {
            "float": base_dtype,
            "hpfloat": torch.float64 if mixed_precision else base_dtype,
            "int": torch.int64,
            "idx": torch.int32,
            "bool": torch.bool,
        }[kind]
    except KeyError as error:
        raise TypeError(f"unsupported tensor dtype declaration {kind!r}") from error


def cast_declared_tensor(
    tensor: torch.Tensor, target: torch.dtype, *, name: str,
) -> torch.Tensor:
    """Convert one external tensor at the model-input boundary.

    Internal compilers and modules must never call this helper: once input is
    bound, declared tensors already have their exact runtime dtype.
    """

    if tensor.dtype == target:
        return tensor
    integer_types = {
        torch.int8, torch.uint8, torch.int16, torch.uint16,
        torch.int32, torch.uint32, torch.int64,
    }
    if target in {torch.float32, torch.float64}:
        if not tensor.is_floating_point():
            raise TypeError(
                f"{name} declares {target} but received non-floating "
                f"dtype {tensor.dtype}"
            )
        if target == torch.float32 and tensor.numel():
            finite = torch.isfinite(tensor)
            outside = finite & (
                torch.abs(tensor) > torch.finfo(torch.float32).max
            )
            if bool(outside.any().item()):
                raise OverflowError(
                    f"{name} cannot convert {tensor.dtype} to {target}: "
                    "finite values exceed the float32 range"
                )
    elif target in {torch.int32, torch.int64}:
        if tensor.dtype not in integer_types:
            raise TypeError(
                f"{name} declares {target} but received non-integer "
                f"dtype {tensor.dtype}"
            )
        if tensor.numel():
            range_tensor = (
                tensor.to(torch.int64)
                if tensor.dtype in {torch.uint16, torch.uint32} else tensor
            )
            lower = int(range_tensor.min().item())
            upper = int(range_tensor.max().item())
            limits = torch.iinfo(target)
            if lower < limits.min or upper > limits.max:
                raise OverflowError(
                    f"{name} cannot convert {tensor.dtype} to {target}: "
                    f"observed range [{lower}, {upper}]"
                )
    elif target == torch.bool:
        raise TypeError(
            f"{name} declares bool but received dtype {tensor.dtype}"
        )
    else:
        raise TypeError(f"{name} has unsupported declared dtype {target}")
    converted = tensor.to(target)
    if (
        target == torch.float32
        and tensor.numel()
        and bool((torch.isfinite(tensor) & (tensor != 0) & (converted == 0)).any().item())
    ):
        raise OverflowError(
            f"{name} cannot convert {tensor.dtype} to {target}: "
            "nonzero values underflow to zero"
        )
    return converted


def _resolve_dimension(
    dimensions: Mapping[DimensionToken, Any],
    dimension: DimensionToken,
) -> Any:
    """Resolve a logical dimension, including ``module.attribute`` tokens."""
    try:
        return dimensions[dimension]
    except KeyError:
        if isinstance(dimension, str) and "." in dimension:
            try:
                return dimensions[dimension.rsplit(".", 1)[1]]
            except KeyError:
                raise KeyError(dimension) from None
        raise


class TensorMetadata(HydroForgeModel):
    """Typed TensorField metadata compiled from Pydantic exactly once."""

    shape: tuple[DimensionToken, ...]
    dtype: str
    category: str
    mode: str
    dim_coords: str | None
    is_key: bool
    is_coordinate: bool
    partition_by: str | None
    references: str | None
    selects: str | None
    replicated: bool
    allow_empty: bool
    output: str
    depends_on: tuple[str, ...]
    required_by: tuple[str, ...]
    expression: str

    @classmethod
    def compile(cls, raw: Mapping[str, Any]) -> Self:
        def enum_value(
            key: str, default: str, allowed: frozenset[str],
        ) -> str:
            value = raw.get(key, default)
            if type(value) is not str:
                raise ValueError(f"{key} must be a string")
            if value not in allowed:
                choices = ", ".join(sorted(allowed))
                raise ValueError(f"{key} must be one of: {choices}")
            return value

        def exact_bool(key: str) -> bool:
            value = raw.get(key, False)
            if type(value) is not bool:
                raise ValueError(f"{key} must be an exact bool")
            return value

        def optional_name(key: str) -> str | None:
            value = raw.get(key)
            if value is None:
                return None
            if type(value) is not str or not value:
                raise ValueError(f"{key} must be a non-empty string or None")
            return value

        def dependencies(key: str) -> tuple[str, ...]:
            values = raw.get(key)
            if values is None:
                return ()
            if isinstance(values, str):
                values = (values,)
            elif type(values) is not tuple:
                raise ValueError(
                    f"{key} must be a module name, a tuple of module "
                    "names, or None"
                )
            if any(
                type(dependency) is not str or not dependency
                for dependency in values
            ):
                raise ValueError(f"{key} must contain non-empty module names")
            if len(values) != len(set(values)):
                raise ValueError(f"{key} contains duplicate module names")
            return tuple(values)

        raw_shape = raw["tensor_shape"]
        if type(raw_shape) is not tuple:
            raise ValueError("tensor_shape must be an exact tuple")
        shape = raw_shape
        for dimension in shape:
            if type(dimension) is int:
                if dimension < 0:
                    raise ValueError(
                        "integer tensor_shape dimensions must be non-negative"
                    )
            elif type(dimension) is not str or not dimension:
                raise ValueError(
                    "tensor_shape dimensions must be exact non-negative ints "
                    "or non-empty strings"
                )

        depends_on = dependencies("depends_on")
        required_by = dependencies("required_by")
        expression_value = raw.get("expr")
        if expression_value is None:
            expression = ""
        elif type(expression_value) is not str:
            raise ValueError("expr must be a string or None")
        else:
            expression = expression_value
        return cls(
            shape=shape,
            dtype=enum_value(
                "tensor_dtype", "float",
                frozenset({"float", "hpfloat", "int", "idx", "bool"}),
            ),
            category=enum_value(
                "category", "param",
                frozenset({
                    "topology", "param", "forcing", "init_state", "state",
                    "derived_param", "shared_state", "virtual",
                }),
            ),
            mode=enum_value(
                "mode", "device", frozenset({"device", "cpu", "discard"}),
            ),
            dim_coords=optional_name("dim_coords"),
            is_key=exact_bool("is_key"),
            is_coordinate=exact_bool("is_coordinate"),
            partition_by=optional_name("partition_by"),
            references=optional_name("references"),
            selects=optional_name("selects"),
            replicated=exact_bool("replicated"),
            allow_empty=exact_bool("allow_empty"),
            output=enum_value(
                "output", "auto", frozenset({"auto", "full", "disabled"}),
            ),
            depends_on=depends_on,
            required_by=required_by,
            expression=expression,
        )


class ModuleFieldSchema(HydroForgeModel):
    """Framework-neutral description of one declared tensor field."""

    module_name: str
    name: str
    shape: tuple[DimensionToken, ...]
    dtype: str
    required: bool
    computed: bool
    tensor: TensorMetadata | None
    excluded: bool
    annotation: Any = None
    description: str = ""

    @property
    def category(self) -> str | None:
        return None if self.tensor is None else self.tensor.category

    @property
    def output(self) -> str | None:
        return None if self.tensor is None else self.tensor.output

    @property
    def selects(self) -> str | None:
        return None if self.tensor is None else self.tensor.selects

class ModuleSchema(HydroForgeModel):
    """Tensor fields grouped by their owning module."""

    modules: Mapping[str, tuple[ModuleFieldSchema, ...]]

    @model_validator(mode="after")
    def _freeze_modules(self) -> Self:
        object.__setattr__(self, "modules", _immutable_dict(self.modules))
        return self

    def resolve_dimensions(
        self,
        dimensions: Mapping[DimensionToken, str],
        *,
        include: Callable[[ModuleFieldSchema], bool] | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        """Translate logical tensor shapes into consumer-specific dimensions."""
        resolved: dict[str, dict[str, tuple[str, ...]]] = {}
        for module_name, fields in self.modules.items():
            module_fields: dict[str, tuple[str, ...]] = {}
            for field in fields:
                if field.tensor is None:
                    continue
                if include is not None and not include(field):
                    continue
                try:
                    module_fields[field.name] = tuple(
                        str(dimension) if isinstance(dimension, int)
                        else _resolve_dimension(dimensions, dimension)
                        for dimension in field.shape
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"{module_name}.{field.name} uses unresolved dimension "
                        f"{exc.args[0]!r}"
                    ) from exc
            resolved[module_name] = module_fields
        return resolved

    def fields(self, module_name: str) -> tuple[ModuleFieldSchema, ...]:
        """Return fields owned by ``module_name``."""
        try:
            return self.modules[module_name]
        except KeyError as exc:
            raise KeyError(f"Module {module_name!r} is absent from schema") from exc

def _field_schema(
    module_name: str,
    name: str,
    field: Any,
    *,
    computed: bool,
) -> ModuleFieldSchema:
    raw_metadata = getattr(field, "json_schema_extra", None)
    if raw_metadata is None:
        metadata: Mapping[str, Any] = {}
    elif not isinstance(raw_metadata, Mapping):
        raise ValueError(
            f"{module_name}.{name} json_schema_extra must be a mapping or None"
        )
    else:
        metadata = raw_metadata
    tensor = (
        TensorMetadata.compile(metadata)
        if "tensor_shape" in metadata
        else None
    )
    excluded = getattr(field, "exclude", None)
    if excluded is None:
        excluded = False
    elif type(excluded) is not bool:
        raise ValueError(
            f"{module_name}.{name} exclude must be an exact bool or None"
        )
    description = getattr(field, "description", None)
    if description is None:
        description = ""
    elif type(description) is not str:
        raise ValueError(
            f"{module_name}.{name} description must be an exact string or None"
        )
    return ModuleFieldSchema(
        module_name=module_name,
        name=name,
        shape=() if tensor is None else tensor.shape,
        dtype="" if tensor is None else tensor.dtype,
        required=not computed and field.is_required(),
        computed=computed,
        tensor=tensor,
        excluded=excluded,
        annotation=getattr(field, "annotation", getattr(field, "return_type", None)),
        description=description,
    )


@cache
def _parse_module_schema_cached(
    modules: tuple[ModuleType, ...],
    *,
    include_computed: bool = False,
) -> ModuleSchema:
    """Parse tensor declarations without instantiating any module.

    The parser preserves logical dimension names and module metadata. File
    formats or applications can subsequently map those dimensions and apply
    their own required/optional policy with :meth:`ModuleSchema.resolve_dimensions`.
    """
    parsed: dict[str, tuple[ModuleFieldSchema, ...]] = {}
    for module in modules:
        module_name = module.module_name
        if module_name in parsed:
            raise ValueError(f"Duplicate module name {module_name!r}")

        fields: list[ModuleFieldSchema] = []
        for name, field in module.model_fields.items():
            schema = _field_schema(
                module_name,
                name,
                field,
                computed=False,
            )
            fields.append(schema)
        if include_computed:
            for name, field in module.model_computed_fields.items():
                schema = _field_schema(
                    module_name,
                    name,
                    field,
                    computed=True,
                )
                fields.append(schema)
        parsed[module_name] = tuple(fields)

    return ModuleSchema(modules=parsed)


class _ModuleSchemaDeclaration(HydroForgeModel):
    modules: tuple[ModuleType, ...]
    include_computed: bool = False

    _schema: ModuleSchema = PrivateAttr()

    @model_validator(mode="after")
    def _compile_schema(self) -> Self:
        from hydroforge.model.module import AbstractModule

        if not self.modules:
            raise ValueError("module schema requires at least one module type")
        invalid = [
            getattr(module, "__name__", type(module).__name__)
            for module in self.modules
            if not isinstance(module, type)
            or not issubclass(module, AbstractModule)
        ]
        if invalid:
            raise ValueError(
                "module schema entries must be AbstractModule classes: "
                f"{invalid}"
            )
        self._schema = _parse_module_schema_cached(
            self.modules,
            include_computed=self.include_computed,
        )
        return self

    @property
    def schema(self) -> ModuleSchema:
        return self._schema


def parse_module_schema(
    modules: tuple[ModuleType, ...],
    *,
    include_computed: bool = False,
) -> ModuleSchema:
    """Return one immutable schema shared by all instances of these modules."""
    declaration = _ModuleSchemaDeclaration(
        modules=modules, include_computed=include_computed,
    )
    return declaration.schema


class PartitionSchema(HydroForgeModel):
    """Validated coordinate/reference graph used by data partitioning."""

    fields: Mapping[str, TensorMetadata]
    coordinates: frozenset[str]
    selections: Mapping[str, str]

    @model_validator(mode="after")
    def _freeze_mappings(self) -> Self:
        object.__setattr__(self, "fields", _immutable_dict(self.fields))
        object.__setattr__(
            self, "selections", _immutable_dict(self.selections),
        )
        return self


class RuntimeTensorMetadata(HydroForgeModel):
    """Typed tensor metadata with per-model output bindings attached."""

    tensor: TensorMetadata
    description: str
    output_index: str | None = None
    output_coord: str | None = None

"""Generic field contracts extracted from module declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any, TypeAlias

import torch


ModuleType: TypeAlias = type[Any]
DimensionToken: TypeAlias = str | int


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
    """Apply one explicit schema conversion without numeric reinterpretation."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
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
    return tensor.to(target)


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


@dataclass(frozen=True, slots=True)
class TensorMetadata:
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
    expression: str

    @classmethod
    def compile(cls, raw: Mapping[str, Any]) -> TensorMetadata:
        depends_on = raw.get("depends_on") or ()
        if isinstance(depends_on, str):
            depends_on = (depends_on,)
        return cls(
            shape=tuple(raw["tensor_shape"]),
            dtype=str(raw.get("tensor_dtype", "float")),
            category=str(raw.get("category", "param")),
            mode=str(raw.get("mode", "device")),
            dim_coords=raw.get("dim_coords"),
            is_key=bool(raw.get("is_key", False)),
            is_coordinate=bool(raw.get("is_coordinate", False)),
            partition_by=raw.get("partition_by"),
            references=raw.get("references"),
            selects=raw.get("selects"),
            replicated=bool(raw.get("replicated", False)),
            allow_empty=bool(raw.get("allow_empty", False)),
            output=str(raw.get("output", "auto")),
            depends_on=tuple(depends_on),
            expression=str(raw.get("expr") or ""),
        )


@dataclass(frozen=True, slots=True)
class ModuleFieldSchema:
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

@dataclass(frozen=True, slots=True)
class ModuleSchema:
    """Tensor fields grouped by their owning module."""

    modules: Mapping[str, tuple[ModuleFieldSchema, ...]]

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
                        _resolve_dimension(dimensions, dimension)
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

    def resolve_field_dimensions(
        self,
        dimensions: Mapping[DimensionToken, str],
        *,
        include: Callable[[ModuleFieldSchema], bool] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve and flatten fields, rejecting ambiguous duplicate names."""
        flattened: dict[str, tuple[str, ...]] = {}
        for module_name, fields in self.resolve_dimensions(
            dimensions, include=include,
        ).items():
            for name, dims in fields.items():
                previous = flattened.setdefault(name, dims)
                if previous != dims:
                    raise ValueError(
                        f"Field {name!r} has conflicting dimensions "
                        f"{previous} and {dims} in module {module_name!r}"
                    )
        return flattened

    def validate_shapes(
        self,
        values: Mapping[str, Any],
        dimensions: Mapping[DimensionToken, int],
        *,
        include: Callable[[ModuleFieldSchema], bool] | None = None,
    ) -> None:
        """Validate present values without requiring every schema field."""
        for fields in self.modules.values():
            for field in fields:
                if field.tensor is None:
                    continue
                if field.name not in values:
                    continue
                if include is not None and not include(field):
                    continue
                try:
                    expected = tuple(
                        dimension if isinstance(dimension, int)
                        else _resolve_dimension(dimensions, dimension)
                        for dimension in field.shape
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"{field.module_name}.{field.name} uses unresolved size "
                        f"{exc.args[0]!r}"
                    ) from exc
                actual = tuple(getattr(values[field.name], "shape", ()))
                if actual != expected:
                    raise ValueError(
                        f"{field.module_name}.{field.name} has shape {actual}, "
                        f"expected {expected}"
                    )


def _field_schema(
    module_name: str,
    name: str,
    field: Any,
    *,
    computed: bool,
) -> ModuleFieldSchema:
    metadata = field.json_schema_extra or {}
    tensor = (
        TensorMetadata.compile(metadata)
        if isinstance(metadata, Mapping) and "tensor_shape" in metadata
        else None
    )
    return ModuleFieldSchema(
        module_name=module_name,
        name=name,
        shape=() if tensor is None else tensor.shape,
        dtype="" if tensor is None else tensor.dtype,
        required=not computed and field.is_required(),
        computed=computed,
        tensor=tensor,
        excluded=bool(getattr(field, "exclude", False)),
        annotation=getattr(field, "annotation", getattr(field, "return_type", None)),
        description=str(getattr(field, "description", "") or ""),
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

    return ModuleSchema(MappingProxyType(parsed))


def parse_module_schema(
    modules: Iterable[ModuleType],
    *,
    include_computed: bool = False,
) -> ModuleSchema:
    """Return one immutable schema shared by all instances of these modules."""
    return _parse_module_schema_cached(
        tuple(modules), include_computed=include_computed,
    )


@dataclass(frozen=True, slots=True)
class PartitionSchema:
    """Validated coordinate/reference graph used by data partitioning."""

    fields: Mapping[str, TensorMetadata]
    coordinates: frozenset[str]
    selections: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeTensorMetadata:
    """Typed tensor metadata with per-model output bindings attached."""

    tensor: TensorMetadata
    description: str
    output_index: str | None = None
    output_coord: str | None = None

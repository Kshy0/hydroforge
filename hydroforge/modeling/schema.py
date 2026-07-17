"""Generic schema extraction from HydroForge module declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

from hydroforge.modeling.module import AbstractModule


ModuleType: TypeAlias = type[AbstractModule]
DimensionToken: TypeAlias = str | int


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
class ModuleFieldSchema:
    """Framework-neutral description of one declared tensor field."""

    module_name: str
    name: str
    shape: tuple[DimensionToken, ...]
    dtype: str
    required: bool
    computed: bool
    metadata: Mapping[str, Any]

    @property
    def category(self) -> str | None:
        return self.metadata.get("category")

    @property
    def output(self) -> str | None:
        return self.metadata.get("output")

    @property
    def selects(self) -> str | None:
        return self.metadata.get("selects")


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
) -> ModuleFieldSchema | None:
    metadata = field.json_schema_extra or {}
    if not isinstance(metadata, Mapping) or "tensor_shape" not in metadata:
        return None
    copied_metadata = MappingProxyType(dict(metadata))
    return ModuleFieldSchema(
        module_name=module_name,
        name=name,
        shape=tuple(metadata["tensor_shape"]),
        dtype=str(metadata.get("tensor_dtype", "float")),
        required=not computed and field.is_required(),
        computed=computed,
        metadata=copied_metadata,
    )


def parse_module_schema(
    modules: Iterable[ModuleType],
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
            if schema is not None:
                fields.append(schema)
        if include_computed:
            for name, field in module.model_computed_fields.items():
                schema = _field_schema(
                    module_name,
                    name,
                    field,
                    computed=True,
                )
                if schema is not None:
                    fields.append(schema)
        parsed[module_name] = tuple(fields)

    return ModuleSchema(MappingProxyType(parsed))

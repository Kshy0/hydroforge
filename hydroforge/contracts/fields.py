"""Generic field contracts extracted from module declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeAlias, get_args

import torch
from pydantic import (
    AfterValidator,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.contracts.validation import (
    FrozenMapping,
    HydroForgeModel,
)

ModuleType: TypeAlias = type[Any]
DimensionToken: TypeAlias = str | int

TensorName: TypeAlias = Annotated[str, Field(min_length=1)]
TensorShape: TypeAlias = tuple[TensorName | Annotated[int, Field(ge=0)], ...]
TensorDType = Literal["float", "hpfloat", "int", "idx", "bool"]
TensorOutput = Literal["auto", "full", "disabled"]


def _unique_module_names(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError("contains duplicate module names")
    return values


_ModuleNames: TypeAlias = Annotated[
    tuple[TensorName, ...], AfterValidator(_unique_module_names)
]
TensorDependencies: TypeAlias = TensorName | _ModuleNames | None


@dataclass(frozen=True, slots=True)
class FieldDemandPlan:
    """Immutable output demand for one model specialization.

    ``required_fields`` controls storage; ``observed_fields`` also tracks
    alias and virtual-expression dependencies.
    """

    required_fields: Mapping[str, frozenset[str]]
    observed_fields: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        def freeze(
            values: Mapping[str, Iterable[str]],
        ) -> Mapping[str, frozenset[str]]:
            if not isinstance(values, Mapping):
                raise TypeError("field demand maps must be mappings")
            normalized: dict[str, frozenset[str]] = {}
            for module, fields in values.items():
                if type(module) is not str or not module:
                    raise TypeError(
                        "field demand module names must be non-empty strings"
                    )
                if isinstance(fields, str):
                    raise TypeError(
                        f"field demand for module {module!r} must be an "
                        "iterable of field names, not a string"
                    )
                try:
                    normalized[module] = frozenset(fields)
                except TypeError as error:
                    raise TypeError(
                        f"field demand for module {module!r} must be iterable"
                    ) from error
                if any(
                    type(field) is not str or not field for field in normalized[module]
                ):
                    raise TypeError(
                        f"field demand for module {module!r} must contain "
                        "non-empty strings"
                    )
            return MappingProxyType(normalized)

        object.__setattr__(self, "required_fields", freeze(self.required_fields))
        object.__setattr__(self, "observed_fields", freeze(self.observed_fields))

    @classmethod
    def empty(cls) -> Self:
        """Return a plan with no output-driven field demand."""

        return cls({}, {})

    @classmethod
    def from_sets(
        cls,
        required_fields: Mapping[str, Iterable[str]],
        observed_fields: Mapping[str, Iterable[str]],
    ) -> Self:
        """Build a canonical plan from compiler-owned mutable sets."""

        return cls(required_fields, observed_fields)

    def required_for(self, module_name: str) -> frozenset[str]:
        """Return direct output requests belonging to one module."""

        return self.required_fields.get(module_name, frozenset())

    def observed_for(self, module_name: str) -> frozenset[str]:
        """Return all output observations belonging to one module."""

        return self.observed_fields.get(module_name, frozenset())

    @property
    def specialization_key(
        self,
    ) -> tuple[
        tuple[tuple[str, tuple[str, ...]], ...],
        tuple[tuple[str, tuple[str, ...]], ...],
    ]:
        """Return an order-independent cache key."""

        def canonical(
            values: Mapping[str, frozenset[str]],
        ) -> tuple[tuple[str, tuple[str, ...]], ...]:
            return tuple(
                (module, tuple(sorted(fields)))
                for module, fields in sorted(values.items())
            )

        return canonical(self.required_fields), canonical(self.observed_fields)

    def __hash__(self) -> int:
        """Make the immutable plan safe to use as a specialization key."""

        return hash(self.specialization_key)

    def is_required(self, module_name: str, field_name: str) -> bool:
        """Return whether one field is directly requested as output."""

        return field_name in self.required_for(module_name)

    def is_observed(self, module_name: str, field_name: str) -> bool:
        """Return whether one field participates in output observation."""

        return field_name in self.observed_for(module_name)


def tensor_is_active(
    metadata: TensorMetadata | RuntimeTensorMetadata | None,
    opened_modules: Iterable[str],
    *,
    output_required: bool = False,
) -> bool:
    """Evaluate conditional tensor residency for one specialization.

    Output demand activates ``output_only`` and ``required_by`` storage but
    never bypasses ``depends_on``.
    """
    if isinstance(metadata, RuntimeTensorMetadata):
        metadata = metadata.tensor
    opened = set(opened_modules)
    required = getattr(metadata, "depends_on", ())
    consumers = getattr(metadata, "required_by", ())
    output_only = getattr(metadata, "output_only", False)
    return (
        all(dependency in opened for dependency in required)
        and (not output_only or output_required)
        and (
            not consumers
            or output_required
            or any(item in opened for item in consumers)
        )
    )


def concrete_tensor_dtype(
    kind: str,
    base_dtype: torch.dtype,
    mixed_precision: bool,
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
    tensor: torch.Tensor,
    target: torch.dtype,
    *,
    name: str,
) -> torch.Tensor:
    """Convert one external tensor at the model-input boundary.

    Internal compilers and modules must never call this helper: once input is
    bound, declared tensors already have their exact runtime dtype.
    """

    if tensor.dtype == target:
        return tensor
    integer_types = {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.uint16,
        torch.int32,
        torch.uint32,
        torch.int64,
    }
    if target in {torch.float32, torch.float64}:
        if not tensor.is_floating_point():
            raise TypeError(
                f"{name} declares {target} but received non-floating "
                f"dtype {tensor.dtype}"
            )
        if target == torch.float32 and tensor.numel():
            finite = torch.isfinite(tensor)
            outside = finite & (torch.abs(tensor) > torch.finfo(torch.float32).max)
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
                if tensor.dtype in {torch.uint16, torch.uint32}
                else tensor
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
        raise TypeError(f"{name} declares bool but received dtype {tensor.dtype}")
    else:
        raise TypeError(f"{name} has unsupported declared dtype {target}")
    converted = tensor.to(target)
    if (
        target == torch.float32
        and tensor.numel()
        and bool(
            (torch.isfinite(tensor) & (tensor != 0) & (converted == 0)).any().item()
        )
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
    """Canonical field metadata validated once when class metadata is read."""

    shape: TensorShape
    dtype: TensorDType = "float"
    category: Literal[
        "topology",
        "param",
        "forcing",
        "init_state",
        "state",
        "derived_param",
        "shared_state",
        "virtual",
    ] = "param"
    mode: Literal["device", "cpu", "discard"] = "device"
    dim_coords: TensorName | None = None
    is_key: bool = False
    is_coordinate: bool = False
    partition_by: TensorName | None = None
    references: TensorName | None = None
    selects: TensorName | None = None
    replicated: bool = False
    output: TensorOutput = "auto"
    depends_on: _ModuleNames = ()
    required_by: _ModuleNames = ()
    expression: str = ""
    output_only: bool = False

    @field_validator("shape", mode="before")
    @classmethod
    def _canonical_shape(cls, value: Any, info: ValidationInfo):
        # Raw FieldInfo metadata must already contain canonical Python types.
        # A before validator receives JSON arrays as Python lists. Restore the
        # tuple representation before strict Python validation resumes.
        if info.mode == "json" and type(value) is list:
            return tuple(value)
        if info.mode == "python" and (
            type(value) is not tuple
            or any(type(dimension) not in (str, int) for dimension in value)
        ):
            raise ValueError("tensor_shape must be an exact tuple of strings or ints")
        return value

    @field_validator("depends_on", "required_by", mode="before")
    @classmethod
    def _canonical_dependencies(cls, value: Any, info: ValidationInfo):
        if value is None:
            return ()
        if type(value) is str:
            return (value,)
        if info.mode == "json" and type(value) is list:
            return tuple(value)
        if info.mode == "python" and (
            type(value) is not tuple or any(type(name) is not str for name in value)
        ):
            raise ValueError("dependencies must be a module name, exact tuple, or None")
        return value

    @field_validator(
        "dtype",
        "category",
        "mode",
        "dim_coords",
        "partition_by",
        "references",
        "selects",
        "output",
        "expression",
        mode="before",
    )
    @classmethod
    def _canonical_string(cls, value: Any, info: ValidationInfo):
        if value is None and info.field_name == "expression":
            return ""
        if value is not None and type(value) is not str:
            raise ValueError("metadata strings must use exact str values")
        return value

    @classmethod
    def compile(cls, raw: Mapping[str, Any]) -> Self:
        # Class FieldInfo remains externally mutable, so validate its current
        # metadata here. Unrelated JSON schema annotations are intentionally ignored.
        aliases = {
            "shape": "tensor_shape",
            "dtype": "tensor_dtype",
            "expression": "expr",
        }
        values = {
            name: raw[source]
            for name in cls.model_fields
            if (source := aliases.get(name, name)) in raw
        }
        return cls(**values)


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

    modules: FrozenMapping[str, tuple[ModuleFieldSchema, ...]]

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
                        str(dimension)
                        if isinstance(dimension, int)
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
    tensor = TensorMetadata.compile(metadata) if "tensor_shape" in metadata else None
    annotation = getattr(
        field,
        "annotation",
        getattr(field, "return_type", None),
    )
    if tensor is not None and tensor.category != "virtual":
        may_be_inactive = bool(
            tensor.depends_on
            or tensor.required_by
            or tensor.output_only
            or tensor.mode == "discard"
        )
        if (
            may_be_inactive
            and annotation is not Any
            and annotation is not None
            and annotation is not type(None)
            and type(None) not in get_args(annotation)
        ):
            raise ValueError(
                f"{module_name}.{name} may be None because of its tensor "
                "lifecycle metadata; annotate it as torch.Tensor | None"
            )
    excluded = getattr(field, "exclude", None)
    if excluded is None:
        excluded = False
    elif type(excluded) is not bool:
        raise ValueError(f"{module_name}.{name} exclude must be an exact bool or None")
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
        annotation=annotation,
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
            if not isinstance(module, type) or not issubclass(module, AbstractModule)
        ]
        if invalid:
            raise ValueError(
                f"module schema entries must be AbstractModule classes: {invalid}"
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
        modules=modules,
        include_computed=include_computed,
    )
    return declaration.schema


class PartitionSchema(HydroForgeModel):
    """Validated coordinate/reference graph used by data partitioning."""

    fields: FrozenMapping[str, TensorMetadata]
    coordinates: frozenset[str]
    selections: FrozenMapping[str, str]


class RuntimeTensorMetadata(HydroForgeModel):
    """Typed tensor metadata with per-model output bindings attached."""

    tensor: TensorMetadata
    description: str
    output_index: str | None = None
    output_coord: str | None = None

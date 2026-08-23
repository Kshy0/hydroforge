# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Abstract base class for hydroforge physics modules using Pydantic v2.
This is the highest level abstraction that all modules inherit from.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Literal,
    Optional,
    Self,
    Tuple,
    TypeVar,
    overload,
)

import torch
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)

from hydroforge.data.distributed import _find_indices_in_torch_trusted
from hydroforge.contracts.events import EventSink, ModelEvent, NullEventSink
from hydroforge.contracts.fields import tensor_is_active
from hydroforge.contracts.kernel_field import _KernelField
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.contracts.runtime import MODEL_OWNED_MODULE_FIELDS
from hydroforge.model.tensors import ModuleTensors


_NO_FIELD_DEFAULT = object()
_MODULE_INITIALIZATION_CONTEXT = "hydroforge_model_initialization"
_MODULE_REFERENCES_CONTEXT = "hydroforge_module_references"
_MODULE_EVENT_SINK_CONTEXT = "hydroforge_module_event_sink"
_MODULE_REFERENCE_TARGETS_CONTEXT = "hydroforge_module_reference_targets"


class _ModuleTensorQuery(HydroForgeModel):
    """One validated public lookup into a module's compiled tensor schema."""

    module_type: Any = Field(exclude=True)
    field_name: str

    _schema: Any = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self) -> Self:
        if not self.field_name:
            raise ValueError("tensor field name must be non-empty")
        schema = self.module_type._get_tensor_schema(self.field_name)
        if schema is None:
            raise ValueError(
                f"unknown tensor field {self.module_type.module_name}.{self.field_name}"
            )
        self._schema = schema
        return self

    @property
    def schema(self) -> Any:
        return self._schema


class _ModuleBatchQuery(HydroForgeModel):
    """One validated public batched-field classification request."""

    module: Any = Field(exclude=True)
    field: str | torch.Tensor

    _tensor: torch.Tensor = PrivateAttr()
    _schema: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve(self) -> Self:
        if isinstance(self.field, str):
            query = _ModuleTensorQuery(
                module_type=type(self.module),
                field_name=self.field,
            )
            self._schema = query.schema
            self._tensor = getattr(self.module, query.field_name)
        else:
            self._tensor = self.field
        return self

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def schema(self) -> Any:
        return self._schema


class _ModuleGatherRequest(HydroForgeModel):
    """Canonical tensors accepted by the public module gather helper."""

    tensor: torch.Tensor
    indices: torch.Tensor
    batched: bool
    num_trials: int | None = Field(exclude=True)

    @model_validator(mode="after")
    def _validate_gather(self) -> Self:
        if self.batched and (
            self.num_trials is None or self.tensor.shape[0] != self.num_trials
        ):
            raise ValueError("batched gather requires the declared leading trial axis")
        if self.indices.numel() and int(self.indices.min().item()) < 0:
            raise ValueError("gather indices must be non-negative")
        return self


class _ModuleClassDeclaration(HydroForgeModel):
    """Validated subclass-authoring declaration for ``AbstractModule``."""

    module_name: str
    description: str
    conflicts: tuple[str, ...]
    nc_excluded_fields: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_declaration(self) -> Self:
        if not self.module_name.isidentifier():
            raise ValueError("module_name must be a Python identifier")
        if not self.description:
            raise ValueError("description must be a non-empty string")
        for label in ("conflicts", "nc_excluded_fields"):
            values = getattr(self, label)
            if any(not value or not value.isidentifier() for value in values):
                raise ValueError(f"{label} must contain Python identifiers")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must not contain duplicates")
        return self


class _TensorFieldDeclaration(HydroForgeModel):
    """Validated public declaration consumed by the TensorField adapter."""

    description: str
    shape: tuple[str | int, ...]
    dtype: Literal["float", "int", "idx", "bool", "hpfloat"] = "float"
    dim_coords: str | None = None
    category: Literal["topology", "param", "init_state", "state"] = "param"
    mode: Literal["device", "cpu", "discard"] = "device"
    is_key: bool = False
    is_coordinate: bool = False
    partition_by: str | None = None
    references: str | None = None
    selects: str | None = None
    replicated: bool = False
    allow_empty: bool = False
    output: Literal["auto", "full", "disabled"] = "auto"
    depends_on: str | tuple[str, ...] | None = None
    required_by: str | tuple[str, ...] | None = None


class _TensorFieldDefault(HydroForgeModel):
    """The finite scalar-default grammar accepted by ``TensorField``."""

    value: bool | int | float | None


def TensorField(
    description: str,
    shape: Tuple[str | int, ...],
    dtype: Literal["float", "int", "idx", "bool", "hpfloat"] = "float",
    dim_coords: Optional[str] = None,
    category: Literal["topology", "param", "init_state", "state"] = "param",
    mode: Literal["device", "cpu", "discard"] = "device",
    is_key: bool = False,
    is_coordinate: bool = False,
    partition_by: Optional[str] = None,
    references: Optional[str] = None,
    selects: Optional[str] = None,
    replicated: bool = False,
    allow_empty: bool = False,
    output: Literal["auto", "full", "disabled"] = "auto",
    depends_on: str | Tuple[str, ...] | None = None,
    required_by: str | Tuple[str, ...] | None = None,
    default: Any = _NO_FIELD_DEFAULT,
):
    """
    Create a tensor field with shape information directly in AbstractModule.

    ``is_key=True`` marks the field as a unique 1D integer key. Such
    fields are validated at startup (1D, int dtype, all values unique)
    and are the only fields that ``PlanItem`` may use for ``target_ids``
    lookup (either via ``dim_coords`` or ``target_id_field``).

    Args:
        description: Human-readable description of the variable
        shape: Tuple of dimension names (scalar variable names)
        dtype: Data type ('float', 'int', 'idx', 'bool', 'hpfloat')
        dim_coords: Variable name that provides coordinates (IDs) for the 0th dimension.
                    Useful for selecting elements by ID (e.g. for parameter changes).
        replicated: Coordinate ownership exception.  ``True`` means every rank
                    receives the complete coordinate and its aligned fields.
                    Valid only for CoordinateField declarations.
        allow_empty: Whether a tensor may contain a zero-length declared axis.
                     Ordinary model dimensions are non-empty by default.
        output: Output policy. ``auto`` inherits the default SelectionField for
                ``dim_coords``; ``full`` writes the full local axis; ``disabled``
                rejects explicit output requests.
        depends_on: Module name, or names, that must all be open for this field
                    to be loaded, allocated, and exposed to runtime compilers.
        required_by: Consumer module names. The field is active when at least
                     one listed consumer module is open.
        category: Category of the variable:
                  - 'topology': Static structure (NEVER batched)
                  - 'param': Input parameter (can be batched)
                  - 'init_state': Initializable restart state (persisted in model
                    checkpoints; ALWAYS batched if num_trials > 1)
        mode: Handling of variables after initialization:
                  - 'device': Keep on current device (default)
                  - 'cpu': Move to CPU memory to save GPU memory
                  - 'discard': Set to None after initialization to maximize memory saving
        default: Exact default declaration validated later against the model field.
    """
    declaration = _TensorFieldDeclaration(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        mode=mode,
        is_key=is_key,
        is_coordinate=is_coordinate,
        partition_by=partition_by,
        references=references,
        selects=selects,
        replicated=replicated,
        allow_empty=allow_empty,
        output=output,
        depends_on=depends_on,
        required_by=required_by,
    )
    default_argument = {}
    if default is not _NO_FIELD_DEFAULT:
        default_argument["default"] = _TensorFieldDefault(
            value=default,
        ).value
    return Field(
        **default_argument,
        description=declaration.description,
        json_schema_extra={
            "tensor_shape": declaration.shape,
            "tensor_dtype": declaration.dtype,
            "dim_coords": declaration.dim_coords,
            "category": declaration.category,
            "mode": declaration.mode,
            "is_key": declaration.is_key,
            "is_coordinate": declaration.is_coordinate,
            "partition_by": declaration.partition_by,
            "references": declaration.references,
            "selects": declaration.selects,
            "replicated": declaration.replicated,
            "allow_empty": declaration.allow_empty,
            "output": declaration.output,
            "depends_on": declaration.depends_on,
            "required_by": declaration.required_by,
        },
    )


def CoordinateField(
    description: str,
    shape: Tuple[str | int, ...],
    dtype: Literal["int", "idx"] = "int",
    partition_by: Optional[str] = None,
    references: Optional[str] = None,
    replicated: bool = False,
    allow_empty: bool = False,
    default: Any = _NO_FIELD_DEFAULT,
):
    """Declare an axis coordinate; ownership is inferred from its relations."""
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=None,
        category="topology",
        mode="cpu",
        is_key=True,
        is_coordinate=True,
        partition_by=partition_by,
        references=references,
        replicated=replicated,
        allow_empty=allow_empty,
        default=default,
    )


def SelectionField(
    description: str,
    shape: Tuple[str | int, ...],
    selects: str,
    dtype: Literal["int", "idx"] = "int",
    allow_empty: bool = True,
    default: Any = _NO_FIELD_DEFAULT,
):
    """Declare a unique coordinate subset used as the default output view."""
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=None,
        category="topology",
        mode="cpu",
        is_key=True,
        is_coordinate=True,
        references=selects,
        selects=selects,
        allow_empty=allow_empty,
        output="disabled",
        default=default,
    )


def ReferenceField(
    description: str,
    shape: Tuple[str | int, ...],
    references: str,
    dim_coords: str,
    dtype: Literal["int", "idx"] = "int",
    is_key: bool = False,
    allow_empty: bool = False,
    default: Any = _NO_FIELD_DEFAULT,
):
    """Declare a globally valid foreign key to another coordinate."""
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category="topology",
        mode="cpu",
        is_key=is_key,
        references=references,
        allow_empty=allow_empty,
        default=default,
    )


class _ReferenceIndexDescriptor:
    """Address-stable, lazily derived local index for a reference field."""

    def __init__(self, reference: str, *, inverse: bool, device: bool) -> None:
        self.reference = reference
        self.inverse = inverse
        self.device = device
        self.name = ""

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        cache_name = f"__derived_reference_index_{self.name}"
        cached = instance.__dict__.get(cache_name)
        if cached is None:
            if self.inverse:
                cached = instance._inverse_reference_index(self.reference)
            else:
                cached = instance._reference_index(self.reference)
            if self.device:
                cached = cached.to(instance.device)
            instance.__dict__[cache_name] = cached
        return cached


class _ReferenceIndexDeclaration(HydroForgeModel):
    reference: str
    inverse: bool = False
    device: bool = True


def ReferenceIndexField(
    reference: str,
    *,
    inverse: bool = False,
    device: bool = True,
):
    """Declare an automatically derived local index for a reference field.

    ``inverse=False`` maps every relation row to its referenced local row;
    ``inverse=True`` maps every target row back to its unique relation row,
    using ``-1`` when it is not referenced.
    """
    declaration = _ReferenceIndexDeclaration(
        reference=reference,
        inverse=inverse,
        device=device,
    )
    return _ReferenceIndexDescriptor(
        declaration.reference,
        inverse=declaration.inverse,
        device=declaration.device,
    )


_TModule = TypeVar("_TModule", bound="AbstractModule")
_TReference = TypeVar("_TReference", covariant=True)


class ModuleReference(HydroForgeModel, Generic[_TReference]):
    """Typed module declaration shared by models and sibling modules."""

    module_type: type[AbstractModule]
    optional: bool

    @model_validator(mode="after")
    def _validate_reference(self) -> Self:
        if not isinstance(self.module_type, type) or not issubclass(
            self.module_type, AbstractModule
        ):
            raise ValueError("module_ref requires an AbstractModule class")
        return self

    @property
    def module_name(self) -> str:
        return self.module_type.module_name

    def __set_name__(self, owner: type, name: str) -> None:
        if name != self.module_name:
            raise ValueError(
                f"module reference attribute {name!r} must match "
                f"{self.module_type.__name__}.module_name "
                f"{self.module_name!r}"
            )

    @classmethod
    def collect(cls, owner: type) -> Mapping[str, ModuleReference]:
        """Collect active declarations using normal Python MRO lookup."""

        return cls._collect(owner)

    @classmethod
    @cache
    def _collect(cls, owner: type) -> Mapping[str, ModuleReference]:

        fields: Dict[str, ModuleReference] = {}
        seen: set[str] = set()
        for base in owner.mro():
            for name, value in vars(base).items():
                if name in seen:
                    continue
                seen.add(name)
                if isinstance(value, cls):
                    fields[name] = value
        return MappingProxyType(fields)

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(
        self,
        instance: object,
        owner: type | None = None,
    ) -> _TReference: ...

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        if isinstance(instance, AbstractModule):
            links = instance._module_references
        else:
            links = instance._module_links
            if links is None:
                instance._ensure_runtime_materialized()
                links = instance._module_links
        return links.get(self.module_name)

    def __set__(self, instance: Any, value: Any) -> None:
        del value
        raise AttributeError(
            f"Module reference {type(instance).__name__}."
            f"{self.module_name} is read-only"
        )


@overload
def module_ref(
    module_type: type[_TModule],
) -> ModuleReference[_TModule]: ...


def module_ref(
    module_type: type[AbstractModule],
) -> ModuleReference[AbstractModule]:
    """Declare a required typed module on a model or sibling module."""

    return ModuleReference(module_type=module_type, optional=False)


@overload
def optional_module_ref(
    module_type: type[_TModule],
) -> ModuleReference[_TModule | None]: ...


def optional_module_ref(
    module_type: type[AbstractModule],
) -> ModuleReference[AbstractModule | None]:
    """Declare an optional typed module that is ``None`` when closed."""

    return ModuleReference(module_type=module_type, optional=True)


class _ComputedTensorFieldDeclaration(HydroForgeModel):
    description: str
    shape: tuple[str | int, ...]
    dtype: Literal["float", "int", "idx", "bool", "hpfloat"] = "float"
    dim_coords: str | None = None
    category: Literal[
        "topology", "derived_param", "state", "shared_state", "virtual"
    ] = "derived_param"
    expr: str | None = None
    depends_on: str | tuple[str, ...] | None = None
    required_by: str | tuple[str, ...] | None = None
    output: Literal["auto", "full", "disabled"] = "auto"
    allow_empty: bool = False

    @model_validator(mode="after")
    def _validate_expression(self) -> Self:
        if self.expr is not None and self.category != "virtual":
            raise ValueError("expr can only be provided when category is 'virtual'")
        return self


def computed_tensor_field(
    description: str,
    shape: Tuple[str | int, ...],
    dtype: Literal["float", "int", "idx", "bool", "hpfloat"] = "float",
    dim_coords: Optional[str] = None,
    category: Literal[
        "topology", "derived_param", "state", "shared_state", "virtual"
    ] = "derived_param",
    expr: Optional[str] = None,
    depends_on: str | Tuple[str, ...] | None = None,
    required_by: str | Tuple[str, ...] | None = None,
    output: Literal["auto", "full", "disabled"] = "auto",
    allow_empty: bool = False,
):
    """
    Create a computed tensor field with shape information for AbstractModule.

    Args:
        description: Human-readable description of the variable
        shape: Tuple of dimension names (scalar variable names)
        dtype: Data type ('float', 'int', 'idx', 'bool', 'hpfloat')
        dim_coords: Variable name that provides coordinates (IDs) for the 0th dimension.
        output: Output policy (``auto``, ``full``, or ``disabled``).
        category: Category of the variable:
                  - 'topology': Static structure (NEVER batched)
                  - 'derived_param': Computed parameter (can be batched)
                  - 'state': Reconstructed runtime state (ALWAYS batched if
                    num_trials > 1; never checkpointed)
                  - 'shared_state': Reconstructed runtime state (NEVER batched
                    or checkpointed)
                  - 'virtual': Computed on-demand during analysis/output (not stored in memory)
        expr: Expression string for virtual variables
        depends_on: Module name, or names, that must all be active before this
            computed tensor is evaluated or validated.
        required_by: Consumer module names. At least one must be active before
            this computed tensor is evaluated or validated.
        allow_empty: Whether a symbolic tensor dimension may resolve to zero.
    """
    declaration = _ComputedTensorFieldDeclaration(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        expr=expr,
        depends_on=depends_on,
        required_by=required_by,
        output=output,
        allow_empty=allow_empty,
    )

    return computed_field(
        description=declaration.description,
        json_schema_extra={
            "tensor_shape": declaration.shape,
            "tensor_dtype": declaration.dtype,
            "dim_coords": declaration.dim_coords,
            "category": declaration.category,
            "expr": declaration.expr,
            "depends_on": declaration.depends_on,
            "required_by": declaration.required_by,
            "allow_empty": declaration.allow_empty,
            "output": declaration.output,
        },
    )


class AbstractModule(HydroForgeModel, ABC):
    """
    Abstract base class for all hydroforge physics modules.

    This class provides the fundamental framework that all modules must follow:
    - Field discovery and validation using Pydantic v2
    - Shape information for tensor fields
    - Type safety for variables
    - Distinction between input variables and computed fields
    - Integration with PyTorch tensors
    - Device and precision management
    - Support for distributed data splitting

    All specific modules (base, bifurcation, reservoir, etc.) inherit from this class.
    """

    # Pydantic configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow torch.Tensor types
        frozen=True,
        # The model compiler materializes every TensorField declaration before
        # Pydantic construction; ordinary scalar defaults are validated here.
        validate_default=True,
        extra="forbid",
        strict=True,
        ignored_types=(
            _ReferenceIndexDescriptor,
            ModuleReference,
            _KernelField,
        ),
    )

    # Module metadata - must be overridden in subclasses
    module_name: ClassVar[str] = "abstract"
    description: ClassVar[str] = "Abstract base module"
    conflicts: ClassVar[Tuple[str, ...]] = ()
    nc_excluded_fields: ClassVar[Tuple[str, ...]] = MODEL_OWNED_MODULE_FIELDS
    """Fields owned by the model runtime rather than module input data."""

    opened_modules: tuple[str, ...] = Field(
        default_factory=tuple,
    )
    rank: int = Field(
        default=0,
        ge=0,
        strict=True,
        description="Current process rank in distributed setup",
    )
    device: torch.device = Field(
        default=torch.device("cpu"),
        description="Device for tensors (e.g., 'cuda:0', 'cpu')",
    )
    precision: torch.dtype = Field(
        default=torch.float32,
        description="Data type for tensors",
    )
    mixed_precision: bool = Field(
        default=True,
        strict=True,
        description=(
            "Enable mixed precision for hpfloat tensors (storage variables).\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  float32 → float64, float64 → float64 (no promotion)."
        ),
    )
    num_trials: Optional[int] = Field(
        default=None,
        strict=True,
        description="Number of parallel simulations (ensemble members)",
    )

    _event_sink: EventSink = PrivateAttr(default_factory=NullEventSink)
    _tensors: ModuleTensors = PrivateAttr()
    _module_references: Dict[str, Optional["AbstractModule"]] = PrivateAttr(
        default_factory=dict,
    )
    _reference_targets: Mapping[str, Any] = PrivateAttr(default_factory=dict)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declaration = _ModuleClassDeclaration(
            module_name=cls.module_name,
            description=cls.description,
            conflicts=cls.conflicts,
            nc_excluded_fields=cls.nc_excluded_fields,
        )
        cls.module_name = declaration.module_name
        cls.description = declaration.description
        cls.conflicts = declaration.conflicts
        cls.nc_excluded_fields = declaration.nc_excluded_fields

    @classmethod
    @cache
    def _field_schema(cls):
        """Return every declared field from the immutable compiled schema."""
        from hydroforge.contracts.fields import parse_module_schema

        return parse_module_schema(
            (cls,),
            include_computed=True,
        ).fields(cls.module_name)

    @classmethod
    @cache
    def _field_schema_map(cls):
        return MappingProxyType(
            {field.name: field for field in cls._field_schema()}
        )

    @classmethod
    @cache
    def tensor_schema(cls):
        """Return the tensor subset of the cached field schema."""
        return tuple(field for field in cls._field_schema() if field.tensor is not None)

    @classmethod
    @cache
    def _tensor_schema_map(cls):
        """Index the compiled schema without reparsing ``json_schema_extra``."""
        return MappingProxyType(
            {field.name: field for field in cls.tensor_schema()}
        )

    def _is_tensor_field_active(self, field: str | Any) -> bool:
        """Return whether a tensor field belongs to this module specialization."""
        schema = (
            self._tensor_schema_map().get(field) if isinstance(field, str) else field
        )
        if schema is None or schema.tensor is None:
            raise KeyError(f"Unknown tensor field: {field}")
        return tensor_is_active(schema.tensor, self.opened_modules)

    def _emit(self, level: str, name: str, message: str, **fields: Any) -> None:
        self._event_sink.emit(
            ModelEvent(
                level=level,
                name=name,
                message=message,
                fields=fields,
            )
        )

    @field_validator("num_trials")
    @classmethod
    def _validate_num_trials(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 1:
            raise ValueError(
                "num_trials must be greater than 1 if specified. For single trial, use None."
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _complete_module_input(
        cls,
        values: Any,
        info: ValidationInfo,
    ) -> Any:
        """Complete the tensor payload as the first module validation step."""

        context = info.context
        if (
            not isinstance(context, Mapping)
            or context.get(_MODULE_INITIALIZATION_CONTEXT) is not True
        ):
            raise ValueError(
                f"module {cls.module_name!r} must be initialized through "
                "an AbstractModel"
            )
        references = context.get(_MODULE_REFERENCES_CONTEXT)
        if not isinstance(references, dict):
            raise ValueError(
                "model initialization must provide the module reference graph"
            )
        sink = context.get(_MODULE_EVENT_SINK_CONTEXT)
        if not isinstance(sink, EventSink):
            raise ValueError("model initialization must provide an EventSink")
        if not isinstance(values, Mapping):
            return values
        try:
            return ModuleTensors.prepare_payload(
                cls,
                dict(values),
                module_references=references,
            )
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def _canonicalize_module_payload(self, info: ValidationInfo) -> Self:
        """Complete and validate tensor fields inside Pydantic validation.

        Base-class after validators execute before subclass after validators,
        preserving the model-author guarantee that downstream semantic
        validators observe canonical tensors rather than scalar declarations.
        """

        context = info.context
        references = context[_MODULE_REFERENCES_CONTEXT]
        try:
            self._module_references = {
                name: references.get(descriptor.module_name)
                for name, descriptor in self._module_reference_fields().items()
            }
            self._reference_targets = context[_MODULE_REFERENCE_TARGETS_CONTEXT]
            self._event_sink = context[_MODULE_EVENT_SINK_CONTEXT]
            self._tensors = ModuleTensors(self)
            if self.module_name not in self.opened_modules:
                raise ValueError(
                    f"`{self.module_name}` is not listed in `opened_modules`. "
                    "All active modules must include themselves in that list."
                )
            self._tensors.initialize_declared()
            self._tensors.materialize_computed()
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self

    @classmethod
    def _module_reference_fields(
        cls,
    ) -> Mapping[str, ModuleReference]:
        return ModuleReference.collect(cls)

    @classmethod
    def _required_modules(cls) -> Tuple[str, ...]:
        """Return sibling modules that must be open with this module."""

        return tuple(
            descriptor.module_name
            for descriptor in cls._module_reference_fields().values()
            if not descriptor.optional
        )

    @classmethod
    def _reference_index_fields(cls) -> Dict[str, _ReferenceIndexDescriptor]:
        fields: Dict[str, _ReferenceIndexDescriptor] = {}
        for owner in reversed(cls.mro()):
            for name, value in vars(owner).items():
                if isinstance(value, _ReferenceIndexDescriptor):
                    fields[name] = value
        return fields

    @classmethod
    def _reference_index_metadata(cls, name: str):
        """Compile derived-index tensor metadata once per module class."""

        return cls._get_reference_index_metadata(name)

    @classmethod
    @cache
    def _get_reference_index_metadata(cls, name: str):
        from hydroforge.contracts.fields import TensorMetadata

        descriptor = cls._reference_index_fields().get(name)
        if descriptor is None:
            return None
        source = cls._tensor_schema_map().get(descriptor.reference)
        if source is None:
            raise ValueError(
                f"ReferenceIndexField {name!r} refers to non-tensor field "
                f"{descriptor.reference!r}"
            )
        return TensorMetadata.compile(
            {
                "tensor_shape": source.tensor.shape,
                "tensor_dtype": "idx",
                "dim_coords": source.tensor.dim_coords,
                "category": "topology",
                "mode": "device" if descriptor.device else "cpu",
                "output": "disabled",
                "allow_empty": source.tensor.allow_empty,
            }
        )

    @classmethod
    def get_tensor_schema(cls, name: str):
        """Resolve a regular, computed, or derived-index typed schema."""

        return _ModuleTensorQuery(
            module_type=cls,
            field_name=name,
        ).schema

    @classmethod
    @cache
    def _get_tensor_schema(cls, name: str):
        schema = cls._tensor_schema_map().get(name)
        if schema is not None:
            return schema
        metadata = cls._reference_index_metadata(name)
        if metadata is None:
            return None
        from hydroforge.contracts.fields import ModuleFieldSchema

        return ModuleFieldSchema(
            module_name=cls.module_name,
            name=name,
            shape=metadata.shape,
            dtype=metadata.dtype,
            required=False,
            computed=True,
            tensor=metadata,
            excluded=False,
            annotation=torch.Tensor,
            description=f"Derived local index {name}",
        )

    def _reference_target(self, field_name: str) -> Tuple[str, torch.Tensor]:
        """Return the construction-time-resolved local target tensor."""

        plan = self._reference_targets[field_name]
        owner = (
            self
            if plan.target_module == self.module_name
            else self._module_references[plan.target_module]
        )
        return plan.qualified_name, getattr(owner, plan.target_field)

    def _reference_index(
        self,
        field_name: str,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve a validated ReferenceField to rank-local indices."""
        values = getattr(self, field_name)

        if target is None:
            _, target = self._reference_target(field_name)
        return _find_indices_in_torch_trusted(values, target)

    def _inverse_reference_index(
        self,
        field_name: str,
        target: Optional[torch.Tensor] = None,
        *,
        fill_value: int = -1,
    ) -> torch.Tensor:
        """Return referencing-row indices aligned to the target coordinate."""
        if target is None:
            _, target = self._reference_target(field_name)
        indices = self._reference_index(field_name, target)
        inverse = torch.full(
            (target.shape[0],),
            fill_value,
            dtype=torch.int32,
            device=indices.device,
        )
        inverse[indices.to(torch.int64)] = torch.arange(
            indices.numel(),
            dtype=torch.int32,
            device=indices.device,
        )
        return inverse

    def get_expected_dtype(self, field_name: str) -> torch.dtype:
        query = _ModuleTensorQuery(
            module_type=type(self),
            field_name=field_name,
        )
        return self._get_expected_dtype(query.field_name)

    def _get_expected_dtype(self, field_name: str) -> torch.dtype:
        return self._tensors.expected_dtype(field_name)

    @model_validator(mode="after")
    def _validate_opened_modules(self) -> Self:
        v = self.opened_modules
        present_conflicts = [
            c for c in self.conflicts if c in v and c != self.module_name
        ]
        if present_conflicts:
            raise ValueError(
                f"Module '{self.module_name}' conflicts with modules present in opened_modules: {present_conflicts}. "
                f"These modules cannot be enabled together."
            )

        return self

    def gather_tensor(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        *,
        batched: bool,
    ) -> torch.Tensor:
        """
        Gather values along the declared coordinate axis.

        If tensor is (N, ...), returns (L, ...) where L = len(indices).
        If tensor is (T, N, ...), returns (T, L, ...).

        ``batched`` must come from field metadata (for example
        ``module.is_batched("field")``). Shape-only inference is ambiguous
        whenever a shared tensor's leading dimension equals ``num_trials``.
        """
        request = _ModuleGatherRequest(
            tensor=tensor,
            indices=indices,
            batched=batched,
            num_trials=self.num_trials,
        )
        if request.batched:
            return request.tensor[:, request.indices]
        return request.tensor[request.indices]

    def is_batched(self, field: str | torch.Tensor) -> bool:
        """Return whether a tensor has HydroForge's leading trial axis.

        Declared fields are decided from their schema rank, so a shared tensor
        whose first dimension happens to equal ``num_trials`` is never
        misclassified. Passing a raw tensor retains the shape-only behavior for
        callers that do not have field metadata.
        """
        query = _ModuleBatchQuery(module=self, field=field)
        if query.schema is not None:
            return self._is_batched_trusted(query.schema.name)
        if self.num_trials is None:
            return False
        tensor = query.tensor
        return tensor.ndim > 0 and tensor.shape[0] == self.num_trials

    def _is_batched_trusted(self, field_name: str) -> bool:
        if self.num_trials is None:
            return False
        schema = type(self)._get_tensor_schema(field_name)
        if schema.tensor.category == "topology":
            return False
        tensor = getattr(self, field_name)
        return tensor.ndim == len(schema.tensor.shape) + 1

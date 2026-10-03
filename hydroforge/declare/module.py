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

import warnings
from abc import ABC
from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypeVar,
    get_args,
    get_origin,
    overload,
)

import torch
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationInfo,
    computed_field,
    model_validator,
    validate_call,
)

from hydroforge.contracts.fields import TensorMetadata
from hydroforge.contracts.runtime import MODEL_OWNED_MODULE_FIELDS
from hydroforge.core.arrays import find_indices_in_torch
from hydroforge.core.events import ModelEvent
from hydroforge.core.naming import Identifier
from hydroforge.core.validation import HydroForgeModel
from hydroforge.declare.kernel_field import _KernelField
from hydroforge.declare.spec import (
    FieldSpec,
    ModuleBinding,
    ModuleRefSpec,
    ModuleSchema,
    ModuleSpec,
    ReferenceIndexSpec,
)
from hydroforge.declare.tensors import ModuleTensors

_NO_FIELD_DEFAULT = object()
_MODULE_BINDING = "hydroforge_module_binding"
_STORED_CATEGORIES = frozenset({"topology", "param", "forcing", "init_state", "state"})
_COMPUTED_CATEGORIES = frozenset(
    {"topology", "derived_param", "state", "shared_state", "virtual"}
)
_TENSOR_METADATA = "__hydroforge_tensor__"
_SPEC = "__hydroforge_spec__"


def _require_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError("must not contain duplicates")
    return values


_STRICT = ConfigDict(strict=True)
_IDENTIFIER = TypeAdapter(Identifier, config=_STRICT)
_DESCRIPTION = TypeAdapter(Annotated[str, Field(min_length=1)], config=_STRICT)
_UNIQUE_IDENTIFIERS = TypeAdapter(
    Annotated[tuple[Identifier, ...], AfterValidator(_require_unique)],
    config=_STRICT,
)


def TensorField(
    description: str,
    shape: tuple[str | int, ...],
    dtype: Literal["float", "int", "idx", "bool", "hpfloat"] = "float",
    dim_coords: str | None = None,
    category: Literal["topology", "param", "forcing", "init_state", "state"] = "param",
    mode: Literal["device", "cpu", "discard"] = "device",
    is_key: bool = False,
    is_coordinate: bool = False,
    partition_by: str | None = None,
    references: str | None = None,
    selects: str | None = None,
    replicated: bool = False,
    output: Literal["auto", "full", "disabled"] = "auto",
    depends_on: str | tuple[str, ...] | None = None,
    required_by: str | tuple[str, ...] | None = None,
    default: Any = _NO_FIELD_DEFAULT,
) -> Any:
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
                  - 'forcing': Transient per-step input; shared unless listed
                    in the model's construction-time ensemble_forcing_fields
                  - 'init_state': Initializable restart state (persisted in model
                    checkpoints, which currently require a non-ensemble model;
                    ALWAYS batched if ensemble_size > 1)
        mode: Handling of variables after initialization:
                  - 'device': Keep on current device (default)
                  - 'cpu': Move to CPU memory to save GPU memory
                  - 'discard': Set to None after initialization to maximize memory saving
        default: Scalar default (bool, int, float or None) expanded to the
                 declared shape when the module is constructed.
    """
    if category not in _STORED_CATEGORIES:
        raise ValueError(
            f"TensorField category must be one of {sorted(_STORED_CATEGORIES)}, "
            f"got {category!r}"
        )
    if default is not _NO_FIELD_DEFAULT and type(default) not in (
        bool,
        int,
        float,
        type(None),
    ):
        raise ValueError("TensorField default must be a bool, int, float or None")
    metadata = TensorMetadata(
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
        output=output,
        depends_on=depends_on,
        required_by=required_by,
    )
    info = (
        Field(description=description)
        if default is _NO_FIELD_DEFAULT
        else Field(default, description=description)
    )
    info.metadata.append(metadata)
    return info


def CoordinateField(
    description: str,
    shape: tuple[str | int, ...],
    dtype: Literal["int", "idx"] = "int",
    partition_by: str | None = None,
    references: str | None = None,
    replicated: bool = False,
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
        default=default,
    )


def SelectionField(
    description: str,
    shape: tuple[str | int, ...],
    selects: str,
    dtype: Literal["int", "idx"] = "int",
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
        output="disabled",
        default=default,
    )


def ReferenceField(
    description: str,
    shape: tuple[str | int, ...],
    references: str,
    dim_coords: str,
    dtype: Literal["int", "idx"] = "int",
    is_key: bool = False,
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

    @property
    def cache_name(self) -> str:
        return f"__derived_reference_index_{self.name}"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        cached = instance.__dict__.get(self.cache_name)
        if cached is None:
            if self.name not in instance._binding.plan.active:
                return None
            cached = self.derive(instance)
            instance.__dict__[self.cache_name] = cached
        return cached

    def derive(
        self,
        instance,
        values: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the index from explicit or live source and target tensors."""
        if self.inverse:
            index = instance._inverse_reference_index(
                self.reference, target, values=values
            )
        else:
            index = instance._reference_index(self.reference, target, values=values)
        return index.to(instance.device if self.device else torch.device("cpu"))


@validate_call(config=_STRICT)
def ReferenceIndexField(
    reference: str,
    *,
    inverse: bool = False,
    device: bool = True,
) -> Any:
    """Declare an automatically derived local index for a reference field.

    ``inverse=False`` maps every relation row to its referenced local row;
    ``inverse=True`` maps every target row back to its unique relation row,
    using ``-1`` when it is not referenced.
    """
    return _ReferenceIndexDescriptor(reference, inverse=inverse, device=device)


_TModule = TypeVar("_TModule", bound="AbstractModule")
_TReference = TypeVar("_TReference", covariant=True)


class ModuleReference(HydroForgeModel, Generic[_TReference]):
    """Typed module declaration shared by models and sibling modules."""

    module_type: type[AbstractModule]
    optional: bool

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
            return instance.__pydantic_private__["_binding"].references.get(
                self.module_name
            )
        runtime = instance.__pydantic_private__["_runtime"]
        links = runtime.module_links
        if links is None:
            raise runtime.unavailable(f"{type(instance).__name__}.{self.module_name}")
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
    """Declare a required module of this class or a subclass, on a model or sibling."""

    return ModuleReference(module_type=module_type, optional=False)


@overload
def optional_module_ref(
    module_type: type[_TModule],
) -> ModuleReference[_TModule | None]: ...


def optional_module_ref(
    module_type: type[AbstractModule],
) -> ModuleReference[AbstractModule | None]:
    """Declare this class or a subclass, or ``None`` when the module is closed."""

    return ModuleReference(module_type=module_type, optional=True)


def _property_function(prop: Any) -> Any:
    """Return the function that carries a computed field's tensor metadata."""
    if isinstance(prop, cached_property):
        return prop.func
    if isinstance(prop, property):
        return prop.fget
    return prop


def computed_tensor_field(
    description: str,
    shape: tuple[str | int, ...],
    dtype: Literal["float", "int", "idx", "bool", "hpfloat"] = "float",
    dim_coords: str | None = None,
    category: Literal[
        "topology", "derived_param", "state", "shared_state", "virtual"
    ] = "derived_param",
    expr: str | None = None,
    depends_on: str | tuple[str, ...] | None = None,
    required_by: str | tuple[str, ...] | None = None,
    output: Literal["auto", "full", "disabled"] = "auto",
    output_only: bool = False,
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
                    ensemble_size > 1; never checkpointed)
                  - 'shared_state': Reconstructed runtime state (NEVER batched
                    or checkpointed)
                  - 'virtual': Computed on-demand during analysis/output (not stored in memory)
        expr: Expression string for virtual variables
        depends_on: Module name, or names, that must all be active before this
            computed tensor is evaluated or validated.
        required_by: Consumer module names. At least one must be active before
            this computed tensor is evaluated or validated.
        output_only: Keep this computed tensor unmaterialized unless it is
            directly requested by statistics. This is an output-storage
            policy; it does not introduce a new checkpoint/state lifecycle
            category. HydroForge exposes an inactive computed tensor as
            ``None`` after specialization, so field implementations do not
            need an activation guard.
    """
    if category not in _COMPUTED_CATEGORIES:
        raise ValueError(
            "computed_tensor_field category must be one of "
            f"{sorted(_COMPUTED_CATEGORIES)}, got {category!r}"
        )
    metadata = TensorMetadata(
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        expression="" if expr is None else expr,
        depends_on=depends_on,
        required_by=required_by,
        output=output,
        output_only=output_only,
    )
    declare = computed_field(description=description)

    def decorate(prop: Any) -> Any:
        setattr(_property_function(prop), _TENSOR_METADATA, metadata)
        return declare(prop)

    return decorate


def _declared_attributes(owner: type) -> dict[str, Any]:
    """Return class attributes by name, resolved in normal MRO order."""

    attributes: dict[str, Any] = {}
    for base in owner.__mro__:
        for name, value in vars(base).items():
            attributes.setdefault(name, value)
    return attributes


def _field_spec(
    module_name: str,
    name: str,
    info: Any,
    *,
    computed: bool,
    excluded_names: tuple[str, ...],
) -> FieldSpec:
    if computed:
        tensor = getattr(
            _property_function(info.wrapped_property), _TENSOR_METADATA, None
        )
        annotation = info.return_type
    else:
        tensor = next(
            (item for item in info.metadata if isinstance(item, TensorMetadata)),
            None,
        )
        annotation = info.annotation
    if tensor is not None and tensor.category != "virtual":
        declared = annotation
        while get_origin(declared) is Annotated:
            declared = get_args(declared)[0]
        types = get_args(declared) or (declared,)
        if declared not in {Any, None} and any(
            item is not type(None)
            and (not isinstance(item, type) or not issubclass(item, torch.Tensor))
            for item in types
        ):
            raise ValueError(
                f"{module_name}.{name}: physical tensor fields must be annotated as torch.Tensor or torch.Tensor | None"
            )
        if computed and not isinstance(info.wrapped_property, cached_property):
            raise ValueError(
                f"{module_name}.{name} is a stored computed tensor and must wrap "
                "functools.cached_property; a plain property would reallocate "
                "storage on every access and cannot be deactivated"
            )
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
    return FieldSpec(
        module_name=module_name,
        name=name,
        tensor=tensor,
        computed=computed,
        required=not computed and info.is_required(),
        excluded=getattr(info, "exclude", None) is True or name in excluded_names,
        annotation=annotation,
        description=info.description or "",
    )


def _module_spec(cls: type[AbstractModule]) -> ModuleSpec:
    """Validate and freeze one module class declaration."""

    module_name = _IDENTIFIER.validate_python(cls.module_name)
    description = _DESCRIPTION.validate_python(cls.description)
    conflicts = _UNIQUE_IDENTIFIERS.validate_python(cls.conflicts)
    excluded_names = _UNIQUE_IDENTIFIERS.validate_python(cls.nc_excluded_fields)
    fields = {
        name: _field_spec(
            module_name, name, info, computed=False, excluded_names=excluded_names
        )
        for name, info in cls.model_fields.items()
    }
    fields.update(
        {
            name: _field_spec(
                module_name, name, info, computed=True, excluded_names=excluded_names
            )
            for name, info in cls.model_computed_fields.items()
        }
    )
    tensor_fields = {
        name: field for name, field in fields.items() if field.tensor is not None
    }
    attributes = _declared_attributes(cls)
    reference_indices: dict[str, ReferenceIndexSpec] = {}
    references: dict[str, ModuleRefSpec] = {}
    kernel_fields: list[str] = []
    for name, value in attributes.items():
        if isinstance(value, _ReferenceIndexDescriptor):
            source = tensor_fields.get(value.reference)
            if source is None:
                raise ValueError(
                    f"ReferenceIndexField {value.reference!r} in module "
                    f"{module_name!r} does not name a tensor field"
                )
            if not source.tensor.references:
                raise ValueError(
                    f"ReferenceIndexField {value.reference!r} in module "
                    f"{module_name!r} refers to a field without reference metadata"
                )
            if len(source.tensor.shape) != 1 or source.tensor.dtype not in {
                "int",
                "idx",
            }:
                raise ValueError(
                    "ReferenceIndexField requires a one-dimensional integer reference source"
                )
            reference_indices[name] = ReferenceIndexSpec(
                name=name,
                reference=value.reference,
                inverse=value.inverse,
                device=value.device,
            )
        elif isinstance(value, ModuleReference):
            references[name] = ModuleRefSpec(
                name=name, module_type=value.module_type, optional=value.optional
            )
        elif isinstance(value, _KernelField):
            kernel_fields.append(name)
    return ModuleSpec(
        module_type=cls,
        name=module_name,
        description=description,
        conflicts=conflicts,
        fields=MappingProxyType(fields),
        tensor_fields=MappingProxyType(tensor_fields),
        reference_indices=MappingProxyType(reference_indices),
        references=MappingProxyType(references),
        required_modules=tuple(
            name for name, reference in references.items() if not reference.optional
        ),
        kernel_fields=tuple(kernel_fields),
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
        ignored_types=(
            _ReferenceIndexDescriptor,
            ModuleReference,
            _KernelField,
        ),
    )

    # Module metadata - must be overridden in subclasses
    module_name: ClassVar[str] = "abstract"
    description: ClassVar[str] = "Abstract base module"
    conflicts: ClassVar[tuple[str, ...]] = ()
    nc_excluded_fields: ClassVar[tuple[str, ...]] = MODEL_OWNED_MODULE_FIELDS
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
    metal_emulation: Literal["native", "float32x2"] = "native"
    mixed_precision: bool = Field(
        default=True,
        strict=True,
        description=(
            "Enable mixed precision for hpfloat tensors (storage variables).\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  float32 → float64, float64 → float64 (no promotion)."
        ),
    )
    ensemble_size: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Number of parallel simulations (ensemble members)",
    )

    _binding: ModuleBinding = PrivateAttr()
    _tensors: ModuleTensors = PrivateAttr()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # An incomplete class (unresolved forward references) builds its spec
        # on the first ``spec()`` call instead.
        if cls.__pydantic_complete__:
            setattr(cls, _SPEC, _module_spec(cls))

    @classmethod
    def spec(cls) -> ModuleSpec:
        """Return the frozen declaration of this module class."""

        spec = cls.__dict__.get(_SPEC)
        if spec is None:
            cls.model_rebuild()
            spec = _module_spec(cls)
            setattr(cls, _SPEC, spec)
        return spec

    def validate_parameters(self) -> Self:
        """Check live physical parameters after a scheduled refresh.

        Override with a read-only ``@model_validator(mode="after")`` to reuse
        the same checks at construction. Keep initial-state validation separate.
        """
        return self

    def _tensor_metadata(self, name: str) -> TensorMetadata | None:
        """Declared metadata, or resolved metadata of an active reference index."""

        field = self.spec().tensor_fields.get(name)
        if field is not None:
            return field.tensor
        return self._binding.plan.reference_indices.get(name)

    def _is_tensor_field_active(self, name: str) -> bool:
        """Return whether a tensor field belongs to this module specialization."""

        return name in self._binding.plan.active

    def _is_field_requested_for_output(self, field: str) -> bool:
        """Return whether statistics observes this declared field directly."""

        return field in self._binding.plan.observed

    def _emit(self, level: str, name: str, message: str, **fields: Any) -> None:
        self._binding.event_sink.emit(
            ModelEvent(
                level=level,
                name=name,
                message=message,
                fields=fields,
            )
        )

    def update_structure(self, context: Any) -> None:
        """Stage rare between-step storage changes in module order.

        Implementations call ``context.stage`` when their declared tensor
        dimensions must change. The model commits every staged module
        atomically after the ordered pass completes.
        """

        del context

    @classmethod
    def prepare_module_input(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Normalize one module payload before declared tensors materialize."""

        return values

    @model_validator(mode="before")
    @classmethod
    def _complete_module_input(
        cls,
        values: Any,
        info: ValidationInfo,
    ) -> Any:
        """Complete the tensor payload as the first module validation step."""

        context = info.context
        binding = context.get(_MODULE_BINDING) if isinstance(context, Mapping) else None
        if not isinstance(binding, ModuleBinding):
            raise ValueError(
                f"module {cls.module_name!r} must be constructed by an "
                "AbstractModel or hydroforge.testing.build_module"
            )
        if not isinstance(values, Mapping):
            return values
        try:
            payload = (
                dict(values)
                if binding.prepared
                else cls.prepare_module_input(dict(values))
            )
            if not isinstance(payload, dict):
                raise TypeError(
                    f"module {cls.module_name!r} prepare_module_input must "
                    "return a dict"
                )
            unknown = sorted(set(payload).difference(cls.model_fields))
            if unknown:
                warnings.warn(
                    f"module {cls.module_name!r} ignores unknown fields: {unknown}",
                    UserWarning,
                    stacklevel=2,
                )
            return ModuleTensors._prepare_payload(cls, payload, binding)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def _canonicalize_module_payload(self, info: ValidationInfo) -> Self:
        """Complete and validate tensor fields inside Pydantic validation.

        Base-class after validators execute before subclass after validators,
        preserving the model-author guarantee that downstream semantic
        validators observe canonical tensors rather than scalar declarations.
        """

        binding = info.context[_MODULE_BINDING]
        try:
            self._binding = binding
            self._tensors = ModuleTensors(
                self,
                batched_fields=binding.plan.batched_forcing,
            )
            if self.module_name not in self.opened_modules:
                raise ValueError(
                    f"`{self.module_name}` is not listed in `opened_modules`. "
                    "All active modules must include themselves in that list."
                )
            self._tensors._validate_declared()
            self._tensors._finalize_computed()
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self

    def _reference_target(self, field_name: str) -> torch.Tensor:
        """Return the construction-time-resolved local target tensor."""

        target = self._binding.plan.reference_targets[field_name]
        owner = (
            self
            if target.module == self.module_name
            else self._binding.references[target.module]
        )
        return getattr(owner, target.field)

    def _reference_index(
        self,
        field_name: str,
        target: torch.Tensor | None = None,
        *,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resolve a validated ReferenceField to rank-local indices."""
        if values is None:
            values = getattr(self, field_name)
        if target is None:
            target = self._reference_target(field_name)
        # Source and target residencies are declared independently.
        return find_indices_in_torch(values, target.to(values.device))

    def _inverse_reference_index(
        self,
        field_name: str,
        target: torch.Tensor | None = None,
        *,
        values: torch.Tensor | None = None,
        fill_value: int = -1,
    ) -> torch.Tensor:
        """Return referencing-row indices aligned to the target coordinate.

        Relation rows whose target is not local (for example replicated
        relations over a partitioned target) leave no inverse entry.
        """
        if target is None:
            target = self._reference_target(field_name)
        indices = self._reference_index(field_name, target, values=values).reshape(-1)
        rows = torch.arange(indices.numel(), dtype=torch.int32, device=indices.device)
        hits = indices >= 0
        local = indices[hits].to(torch.int64)
        if local.numel() and torch.unique(local).numel() != local.numel():
            raise ValueError(
                f"Inverse reference field {self.module_name}.{field_name} must "
                "contain unique target references"
            )
        inverse = torch.full(
            (target.shape[0],),
            fill_value,
            dtype=torch.int32,
            device=indices.device,
        )
        inverse[local] = rows[hits]
        return inverse

    def _stale_reference_indices(
        self,
        replacements: Mapping[int, torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return materialized derived indices invalidated by staged storage.

        A derived index staged explicitly must equal its derived value; one
        whose source or target is replaced is returned with fresh contents.
        """
        stale: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name in self.spec().reference_indices:
            descriptor = getattr(type(self), name)
            cached = self.__dict__.get(descriptor.cache_name)
            if cached is None:
                continue
            values = getattr(self, descriptor.reference)
            target = self._reference_target(descriptor.reference)
            staged = replacements.get(id(cached))
            if (
                staged is None
                and id(values) not in replacements
                and id(target) not in replacements
            ):
                continue
            if not isinstance(values, torch.Tensor) or not isinstance(
                target, torch.Tensor
            ):
                raise ValueError(
                    f"derived reference index {self.module_name}.{name} cannot "
                    "be refreshed because its source or target was discarded"
                )
            fresh = descriptor.derive(
                self,
                replacements.get(id(values), values),
                replacements.get(id(target), target),
            )
            if staged is None:
                stale.append((cached, fresh))
            elif not torch.equal(staged, fresh.to(staged.device)):
                raise ValueError(
                    f"staged derived reference index {self.module_name}.{name} "
                    f"does not match {descriptor.reference!r} resolved against "
                    "its target coordinate"
                )
        return stale

    def get_expected_dtype(self, field_name: str) -> torch.dtype:
        """Return the concrete dtype of one declared tensor field."""

        return self._tensors._expected_dtype(field_name)

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
        whenever a shared tensor's leading dimension equals ``ensemble_size``.
        """
        if type(batched) is not bool:
            raise TypeError("batched must be an exact bool")
        if not isinstance(tensor, torch.Tensor) or not isinstance(
            indices, torch.Tensor
        ):
            raise TypeError("gather requires tensors")
        if indices.ndim != 1 or indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("gather indices must be a one-dimensional integer tensor")
        axis = int(batched)
        if tensor.ndim <= axis:
            raise ValueError("gather tensor has no coordinate axis")
        if batched and (
            self.ensemble_size is None or tensor.shape[0] != self.ensemble_size
        ):
            raise ValueError("batched gather requires the declared leading member axis")
        if indices.numel() and int(indices.min().item()) < 0:
            raise ValueError("gather indices must be non-negative")
        if indices.numel() and int(indices.max().item()) >= tensor.shape[axis]:
            raise ValueError("gather indices exceed the coordinate axis")
        if batched:
            return tensor[:, indices]
        return tensor[indices]

    def is_batched(self, field: str | torch.Tensor) -> bool:
        """Return whether a tensor has HydroForge's leading member axis.

        Declared fields are decided from their schema rank, so a shared tensor
        whose first dimension happens to equal ``ensemble_size`` is never
        misclassified. Passing a raw tensor retains the shape-only behavior for
        callers that do not have field metadata.
        """
        if isinstance(field, str):
            metadata = self._tensor_metadata(field)
            if metadata is None:
                raise ValueError(f"unknown tensor field {self.module_name}.{field}")
            if self.ensemble_size is None or metadata.category == "topology":
                return False
            if metadata.category == "forcing":
                return field in self._tensors.batched_fields
            tensor = getattr(self, field)
            # An inactive field has no storage and therefore no member axis.
            return (
                isinstance(tensor, torch.Tensor)
                and tensor.ndim == len(metadata.shape) + 1
            )
        if self.ensemble_size is None:
            return False
        return field.ndim > 0 and field.shape[0] == self.ensemble_size

    def forcing_layout(
        self,
        field_name: str,
    ) -> Literal["shared", "batched"]:
        """Return the construction-time layout of one forcing field."""

        metadata = self._tensor_metadata(field_name)
        if metadata is None or metadata.category != "forcing":
            raise ValueError(f"{self.module_name}.{field_name} is not a forcing field")
        if not self._is_tensor_field_active(field_name):
            raise ValueError(
                f"forcing field {self.module_name}.{field_name} is inactive"
            )
        return "batched" if field_name in self._tensors.batched_fields else "shared"

    def materialize_fields(self, *field_names: str) -> None:
        """Materialize declared lazy fields needed by a module execution path.

        Computed virtual fields intentionally remain lazy so output selection
        does not allocate every diagnostic.  A model's
        ``initialize_model_state`` method can request a module's internal
        workspace through this small, schema-aware API instead of reaching
        through the controller with private ``_ = module.field`` accesses.
        Modules do not receive an implicit initialization callback.
        """

        for field_name in field_names:
            if self._tensor_metadata(field_name) is None:
                raise KeyError(f"Unknown tensor field {self.module_name}.{field_name}")
            if not self._is_tensor_field_active(field_name):
                raise ValueError(
                    f"Cannot materialize inactive field {self.module_name}.{field_name}"
                )
            value = getattr(self, field_name)
            if value is None:
                raise ValueError(
                    f"Active field {self.module_name}.{field_name} resolved to None"
                )


def construct_module(
    module_type: type[AbstractModule],
    values: Mapping[str, Any],
    binding: ModuleBinding,
) -> AbstractModule:
    """Validate one module from its field values and model binding."""

    return module_type.model_validate(values, context={_MODULE_BINDING: binding})


@validate_call(config=ConfigDict(strict=True))
def module_schema(
    modules: Annotated[tuple[type[AbstractModule], ...], Field(min_length=1)],
    *,
    include_computed: bool = False,
) -> ModuleSchema:
    """Return the declared fields of these modules, grouped by module name."""

    schema: dict[str, tuple[FieldSpec, ...]] = {}
    for module in modules:
        spec = module.spec()
        if spec.name in schema:
            raise ValueError(f"Duplicate module name {spec.name!r}")
        schema[spec.name] = tuple(
            field
            for field in spec.fields.values()
            if include_computed or not field.computed
        )
    return ModuleSchema(MappingProxyType(schema))

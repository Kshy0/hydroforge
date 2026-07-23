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
from functools import cache
from typing import Any, ClassVar, Dict, List, Literal, Optional, Self, Tuple

import torch
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      computed_field, field_validator, model_validator)

from hydroforge.data.distributed import find_indices_in_torch
from hydroforge.contracts.events import EventSink, ModelEvent, NullEventSink
from hydroforge.model.tensors import ModuleTensors


def TensorField(
    description: str,
    shape: Tuple[str, ...],
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
    **kwargs
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
        category: Category of the variable:
                  - 'topology': Static structure (NEVER batched)
                  - 'param': Input parameter (can be batched)
                  - 'init_state': Initializable state variable (ALWAYS batched if num_trials > 1)
        mode: Handling of variables after initialization:
                  - 'device': Keep on current device (default)
                  - 'cpu': Move to CPU memory to save GPU memory
                  - 'discard': Set to None after initialization to maximize memory saving
        **kwargs: Additional Field parameters
    """
    legacy = {"group_by", "save_idx", "save_coord", "partition", "locality"}
    unsupported = sorted(legacy.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise TypeError(
            f"TensorField no longer accepts {names}; declare coordinate relations "
            "with dim_coords/CoordinateField/ReferenceField/SelectionField instead"
        )
    return Field(
        description=description,
        **kwargs,
        json_schema_extra={
            "tensor_shape": list(shape),
            "tensor_dtype": dtype,
            "dim_coords": dim_coords,
            "category": category,
            "mode": mode,
            "is_key": is_key,
            "is_coordinate": is_coordinate,
            "partition_by": partition_by,
            "references": references,
            "selects": selects,
            "replicated": replicated,
            "allow_empty": allow_empty,
            "output": output,
        }
    )


def CoordinateField(
    description: str,
    shape: Tuple[str, ...],
    dtype: Literal["int", "idx"] = "int",
    partition_by: Optional[str] = None,
    references: Optional[str] = None,
    replicated: bool = False,
    **kwargs,
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
        **kwargs,
    )


def SelectionField(
    description: str,
    shape: Tuple[str, ...],
    selects: str,
    dtype: Literal["int", "idx"] = "int",
    allow_empty: bool = True,
    **kwargs,
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
        **kwargs,
    )


def ReferenceField(
    description: str,
    shape: Tuple[str, ...],
    references: str,
    dim_coords: str,
    dtype: Literal["int", "idx"] = "int",
    **kwargs,
):
    """Declare a globally valid foreign key to another coordinate."""
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category="topology",
        mode="cpu",
        references=references,
        **kwargs,
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
                cached = instance.inverse_reference_index(self.reference)
            else:
                cached = instance.reference_index(self.reference)
            if self.device:
                cached = cached.to(instance.device)
            instance.__dict__[cache_name] = cached
        return cached

def ReferenceIndexField(
    reference: str, *, inverse: bool = False, device: bool = True,
):
    """Declare an automatically derived local index for a reference field.

    ``inverse=False`` maps every relation row to its referenced local row;
    ``inverse=True`` maps every target row back to its unique relation row,
    using ``-1`` when it is not referenced.
    """
    return _ReferenceIndexDescriptor(reference, inverse=inverse, device=device)

def computed_tensor_field(
    description: str,
    shape: Tuple[str, ...],
    dtype: Literal["float", "int", "idx", "bool", "hpfloat"] = "float",
    dim_coords: Optional[str] = None,
    category: Literal["topology", "derived_param", "state", "shared_state", "virtual"] = "derived_param",
    expr: Optional[str] = None,
    depends_on: Optional[str] = None,
    output: Literal["auto", "full", "disabled"] = "auto",
    allow_empty: bool = False,
    **kwargs
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
                  - 'state': Computed state variable (ALWAYS batched if num_trials > 1)
                  - 'shared_state': Computed state variable (NEVER batched)
                  - 'virtual': Computed on-demand during analysis/output (not stored in memory)
        expr: Expression string for virtual variables
        depends_on: Optional module that must be active before this computed
            tensor is evaluated or validated.
        allow_empty: Whether a symbolic tensor dimension may resolve to zero.
        **kwargs: Additional computed_field parameters
    """
    legacy = {
        "group_by", "save_idx", "save_coord", "partition", "locality",
        "static_output",
    }
    unsupported = sorted(legacy.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise TypeError(
            f"computed_tensor_field no longer accepts {names}; output selection "
            "is inferred from dim_coords and SelectionField"
        )
    if expr is not None and category != "virtual":
        raise ValueError("expr can only be provided when category is 'virtual'")

    return computed_field(
        description=description,
        json_schema_extra={
            "tensor_shape": list(shape),
            "tensor_dtype": dtype,
            "dim_coords": dim_coords,
            "category": category,
            "expr": expr,
            "depends_on": depends_on,
            "allow_empty": allow_empty,
            "output": output,
        },
        **kwargs
    )


class AbstractModule(BaseModel, ABC):
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
        validate_assignment=False,      # Validate on assignment
        extra='ignore',
        ignored_types=(_ReferenceIndexDescriptor,),
    )

    # Module metadata - must be overridden in subclasses
    module_name: ClassVar[str] = "abstract"
    description: ClassVar[str] = "Abstract base module"
    dependencies: ClassVar[List[str]] = []  # List of modules this module depends on
    conflicts: ClassVar[List[str]] = []  # List of modules that cannot co-exist with this module
    nc_excluded_fields: ClassVar[List[str]] = [
        "opened_modules", "device", "precision", "mixed_precision", "rank",
        "num_trials",
    ]  # Fields to exclude from HDF5

    opened_modules: List[str] = Field(
        default_factory=list,
    )
    rank: int = Field(
        default=0,
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
        description=(
            "Enable mixed precision for hpfloat tensors (storage variables).\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  float32 → float64, float64 → float64 (no promotion)."
        ),
    )
    num_trials: Optional[int] = Field(
        default=None,
        description="Number of parallel simulations (ensemble members)",
    )

    _event_sink: EventSink = PrivateAttr(default_factory=NullEventSink)
    _tensors: ModuleTensors = PrivateAttr()
    _sealed_tensor_bindings: Optional[Dict[str, object]] = PrivateAttr(
        default=None,
    )
    _sealed_declared_fields: Optional[Dict[str, object]] = PrivateAttr(
        default=None,
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("dependencies", "conflicts", "nc_excluded_fields"):
            values = tuple(getattr(cls, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise TypeError(
                    f"{cls.__name__}.{name} must contain non-empty strings"
                )
            if len(values) != len(set(values)):
                raise ValueError(
                    f"{cls.__name__}.{name} contains duplicate declarations"
                )
            setattr(cls, name, values)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject tensor-object replacement after model initialization.

        Native kernels, CUDA graphs, Metal ICBs and recorded ATen nodes retain
        the tensor objects observed while the sub-step program is compiled.
        Replacing a declared field would therefore make the public module and
        the cached program refer to different storage.  Values remain mutable
        through ordinary in-place tensor operations; only object identity is
        sealed.
        """

        private = getattr(self, "__pydantic_private__", None)
        sealed = (
            None if private is None
            else private.get("_sealed_tensor_bindings")
        )
        if sealed is not None and name in sealed and value is not sealed[name]:
            raise RuntimeError(
                f"declared tensor binding {self.module_name}.{name} is sealed; "
                "update its value in place instead of replacing the tensor "
                "object"
            )
        declared = (
            None if private is None
            else private.get("_sealed_declared_fields")
        )
        if (
            declared is not None
            and name in declared
            and value is not declared[name]
        ):
            raise RuntimeError(
                f"declared module field {self.module_name}.{name} is sealed "
                "after initialization"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        private = getattr(self, "__pydantic_private__", None)
        sealed = (
            None if private is None
            else private.get("_sealed_tensor_bindings")
        )
        if sealed is not None and name in sealed:
            raise RuntimeError(
                f"declared tensor binding {self.module_name}.{name} is sealed "
                "and cannot be deleted"
            )
        declared = (
            None if private is None
            else private.get("_sealed_declared_fields")
        )
        if declared is not None and name in declared:
            raise RuntimeError(
                f"declared module field {self.module_name}.{name} is sealed "
                "and cannot be deleted"
            )
        super().__delattr__(name)

    def _seal_declared_tensor_bindings(self) -> None:
        """Seal every declared tensor slot once initialization is complete."""

        if self._sealed_tensor_bindings is not None:
            raise RuntimeError(
                f"module {self.module_name!r} tensor bindings are already sealed"
            )
        # This configuration participates in cached module specialization but
        # was historically a mutable Pydantic list.  Normalize it before the
        # identity seal so append/clear cannot diverge the module's public
        # configuration from the compiled model capability plan.
        self.opened_modules = tuple(self.opened_modules)
        self._sealed_tensor_bindings = {
            field.name: getattr(self, field.name, None)
            for field in self.tensor_schema()
        }
        self._sealed_declared_fields = {
            name: getattr(self, name)
            for name in type(self).model_fields
        }

    @classmethod
    @cache
    def field_schema(cls):
        """Return every declared field from the immutable compiled schema."""
        from hydroforge.contracts.fields import parse_module_schema

        return parse_module_schema(
            (cls,), include_computed=True,
        ).fields(cls.module_name)

    @classmethod
    @cache
    def field_schema_map(cls):
        return {field.name: field for field in cls.field_schema()}

    @classmethod
    @cache
    def tensor_schema(cls):
        """Return the tensor subset of the cached field schema."""
        return tuple(
            field for field in cls.field_schema() if field.tensor is not None
        )

    @classmethod
    @cache
    def tensor_schema_map(cls):
        """Index the compiled schema without reparsing ``json_schema_extra``."""
        return {field.name: field for field in cls.tensor_schema()}

    def _emit(self, level: str, name: str, message: str, **fields: Any) -> None:
        self._event_sink.emit(ModelEvent(level, name, message, fields))

    @property
    def high_precision(self) -> torch.dtype:
        """Return the dtype for hpfloat tensors.

        When ``mixed_precision`` is False (default), hpfloat uses the same
        dtype as ``precision`` — all tensors share one precision level.

        When ``mixed_precision`` is True, hpfloat is promoted one level:
          float32 → float64, float64 → float64.
        High-precision storage follows the pattern of Fortran-based solvers
        (e.g. double-precision for storage variables) (P2RIVSTO, P2FLDSTO, etc.).
        """
        if not self.mixed_precision:
            return self.precision
        _hp_map = {
            torch.float32: torch.float64,
            torch.float64: torch.float64,
        }
        return _hp_map.get(self.precision, self.precision)

    @field_validator('num_trials')
    @classmethod
    def validate_num_trials(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 1:
            raise ValueError("num_trials must be greater than 1 if specified. For single trial, use None.")
        return v

    def model_post_init(self, __context: Any):
        if self.module_name not in self.opened_modules:
            raise ValueError(
                f"`{self.module_name}` is not listed in `opened_modules`. "
                f"All active modules must include themselves in that list."
            )
        self._tensors = ModuleTensors(self)
        self._tensors.initialize()
        # Derived reference indices are declared as descriptors rather than
        # Pydantic computed fields, but retain eager topology validation and
        # device placement during module initialization.
        for name in self.get_reference_index_fields():
            getattr(self, name)

    @classmethod
    def get_reference_index_fields(cls) -> Dict[str, _ReferenceIndexDescriptor]:
        fields: Dict[str, _ReferenceIndexDescriptor] = {}
        for owner in reversed(cls.mro()):
            for name, value in vars(owner).items():
                if isinstance(value, _ReferenceIndexDescriptor):
                    fields[name] = value
        return fields

    @classmethod
    @cache
    def get_reference_index_metadata(cls, name: str):
        """Compile derived-index tensor metadata once per module class."""
        from hydroforge.contracts.fields import TensorMetadata

        descriptor = cls.get_reference_index_fields().get(name)
        if descriptor is None:
            return None
        source = cls.tensor_schema_map().get(descriptor.reference)
        if source is None:
            raise TypeError(
                f"ReferenceIndexField {name!r} refers to non-tensor field "
                f"{descriptor.reference!r}"
            )
        return TensorMetadata.compile({
            "tensor_shape": source.tensor.shape,
            "tensor_dtype": "idx",
            "dim_coords": source.tensor.dim_coords,
            "category": "topology",
            "mode": "device" if descriptor.device else "cpu",
            "output": "disabled",
        })

    @classmethod
    @cache
    def get_tensor_schema(cls, name: str):
        """Resolve a regular, computed, or derived-index typed schema."""
        schema = cls.tensor_schema_map().get(name)
        if schema is not None:
            return schema
        metadata = cls.get_reference_index_metadata(name)
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
        """Resolve the uniquely visible tensor referenced by ``field_name``."""
        schema = self.tensor_schema_map().get(field_name)
        target_name = None if schema is None else schema.tensor.references
        if not target_name:
            raise ValueError(f"Field '{field_name}' does not declare references.")

        parts = target_name.split(".")
        attr_name = parts[-1]
        candidates: List[torch.Tensor] = []
        if len(parts) > 1:
            owner_name = parts[-2]
            owner = self if owner_name == self.module_name else getattr(
                self, owner_name, None
            )
            value = getattr(owner, attr_name, None) if owner is not None else None
            if isinstance(value, torch.Tensor):
                candidates.append(value)
        else:
            value = getattr(self, attr_name, None)
            if isinstance(value, torch.Tensor):
                candidates.append(value)
            for dependency in self.dependencies:
                owner = getattr(self, dependency, None)
                value = getattr(owner, attr_name, None) if owner is not None else None
                if isinstance(value, torch.Tensor):
                    candidates.append(value)
        if len(candidates) != 1:
            raise ValueError(
                f"Reference target '{target_name}' for '{field_name}' resolved "
                f"to {len(candidates)} local tensors; use a qualified reference."
            )
        return target_name, candidates[0]

    def reference_index(
        self,
        field_name: str,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve a ReferenceField to rank-local indices.

        ``ReferenceField`` itself guarantees only foreign-key semantics.  This
        method is the explicit request for a local array representation and
        therefore fails if any referenced ID is not colocated on this rank.
        """
        values = getattr(self, field_name)
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"Reference field '{field_name}' is not a tensor.")

        if target is None:
            target_name, target = self._reference_target(field_name)
        else:
            schema = self.tensor_schema_map().get(field_name)
            target_name = None if schema is None else schema.tensor.references

        indices = find_indices_in_torch(values, target)
        missing = indices < 0
        if torch.any(missing):
            examples = values[missing][:5].detach().cpu().tolist()
            raise ValueError(
                f"Reference field '{field_name}' is not local on rank {self.rank}; "
                f"IDs absent from '{target_name}' include {examples}."
            )
        return indices

    def inverse_reference_index(
        self,
        field_name: str,
        target: Optional[torch.Tensor] = None,
        *,
        fill_value: int = -1,
    ) -> torch.Tensor:
        """Return referencing-row indices aligned to the target coordinate."""
        if target is None:
            _, target = self._reference_target(field_name)
        indices = self.reference_index(field_name, target)
        if indices.numel() and torch.unique(indices).numel() != indices.numel():
            raise ValueError(
                f"Reference field '{field_name}' contains duplicate target "
                "references and therefore has no unique inverse."
            )
        inverse = torch.full(
            (target.shape[0],), fill_value,
            dtype=torch.int32, device=indices.device,
        )
        inverse[indices.to(torch.int64)] = torch.arange(
            indices.numel(), dtype=torch.int32, device=indices.device,
        )
        return inverse

    def get_expected_shape(self, field_name: str) -> Optional[Tuple[int, ...]]:
        return self._tensors.expected_shape(field_name)

    def get_expected_dtype(self, field_name: str) -> torch.dtype:
        return self._tensors.expected_dtype(field_name)

    @model_validator(mode="after")
    def validate_opened_modules(self) -> Self:
        v = self.opened_modules
        if self.module_name not in v:
            raise ValueError(
                f"Current module '{self.module_name}' must be included in opened_modules. "
                f"Available modules: {v}"
            )

        missing_deps = [dep for dep in self.dependencies if dep not in v]
        if missing_deps:
            raise ValueError(
                f"Module '{self.module_name}' has missing dependencies in opened_modules: {missing_deps}. "
                f"Required dependencies: {self.dependencies}. "
                f"Available modules: {v}"
            )

        present_conflicts = [c for c in self.conflicts if c in v and c != self.module_name]
        if present_conflicts:
            raise ValueError(
                f"Module '{self.module_name}' conflicts with modules present in opened_modules: {present_conflicts}. "
                f"These modules cannot be enabled together."
            )

        return self


    def gather_tensor(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Gather values from a tensor using indices, handling potential batch dimensions.

        If tensor is (N, ...), returns (L, ...) where L = len(indices).
        If tensor is (T, N, ...), returns (T, L, ...).
        """
        if self.num_trials is not None and tensor.shape[0] == self.num_trials:
            # Batched tensor (T, N, ...)
            # We want to gather along the second dimension (N)
            # indices is (L,)
            # Result should be (T, L, ...)
            # We can use tensor[:, indices]
            return tensor[:, indices]
        else:
            # Shared tensor (N, ...)
            # Result should be (L, ...)
            return tensor[indices]


    def is_batched(self, field: str | torch.Tensor) -> bool:
        """Return whether a tensor has HydroForge's leading trial axis.

        Declared fields are decided from their schema rank, so a shared tensor
        whose first dimension happens to equal ``num_trials`` is never
        misclassified. Passing a raw tensor retains the shape-only behavior for
        callers that do not have field metadata.
        """
        if self.num_trials is None:
            return False
        if isinstance(field, str):
            tensor = getattr(self, field)
            schema = self.tensor_schema_map().get(field)
            if schema is None:
                raise KeyError(f"Unknown tensor field: {field}")
            if schema.tensor.category == "topology":
                return False
            declared_rank = len(schema.tensor.shape)
            return tensor.ndim == declared_rank + 1
        tensor = field
        return tensor.ndim > 0 and tensor.shape[0] == self.num_trials

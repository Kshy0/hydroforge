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
from numbers import Integral
from typing import Any, ClassVar, Dict, List, Literal, Optional, Self, Tuple

import torch
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      computed_field, field_validator, model_validator)
from pydantic.fields import FieldInfo

from hydroforge.modeling.distributed import find_indices_in_torch


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

    def field_info(self, owner: type["AbstractModule"]) -> FieldInfo:
        """Build tensor metadata from the referenced relation field."""
        reference_info = owner.get_model_fields().get(self.reference)
        if reference_info is None:
            raise TypeError(
                f"ReferenceIndexField '{self.name}' refers to unknown field "
                f"'{self.reference}'"
            )
        reference_extra = reference_info.json_schema_extra or {}
        return Field(
            description=f"Local index derived from {self.reference}",
            json_schema_extra={
                "tensor_shape": reference_extra.get("tensor_shape", []),
                "tensor_dtype": "idx",
                "dim_coords": reference_extra.get("dim_coords"),
                "category": "topology",
                "output": "disabled",
            },
        )


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

    _expanded_params: set = PrivateAttr(default_factory=set)

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
        self.validate_tensors()
        self.init_optional_tensors()
        self.validate_computed_tensors()
        # Derived reference indices are declared as descriptors rather than
        # Pydantic computed fields, but retain eager topology validation and
        # device placement during module initialization.
        for name in self.get_reference_index_fields():
            getattr(self, name)

    @classmethod
    def get_model_fields(cls) -> Dict[str, FieldInfo]:
        return cls.model_fields

    @classmethod
    def get_model_computed_fields(cls) -> Dict[str, Any]:
        return cls.model_computed_fields

    @classmethod
    def get_reference_index_fields(cls) -> Dict[str, _ReferenceIndexDescriptor]:
        fields: Dict[str, _ReferenceIndexDescriptor] = {}
        for owner in reversed(cls.mro()):
            for name, value in vars(owner).items():
                if isinstance(value, _ReferenceIndexDescriptor):
                    fields[name] = value
        return fields

    @classmethod
    def get_reference_index_field_info(cls, name: str) -> Optional[FieldInfo]:
        descriptor = cls.get_reference_index_fields().get(name)
        return descriptor.field_info(cls) if descriptor is not None else None

    @classmethod
    def get_tensor_field_info(cls, name: str) -> Optional[FieldInfo]:
        """Resolve regular, computed, and derived-index tensor metadata."""
        return (
            cls.get_model_fields().get(name)
            or cls.get_model_computed_fields().get(name)
            or cls.get_reference_index_field_info(name)
        )

    def _reference_target(self, field_name: str) -> Tuple[str, torch.Tensor]:
        """Resolve the uniquely visible tensor referenced by ``field_name``."""
        field_info = self.get_model_fields().get(field_name)
        extra = getattr(field_info, "json_schema_extra", {}) or {}
        target_name = extra.get("references")
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
            field_info = self.get_model_fields().get(field_name)
            extra = getattr(field_info, "json_schema_extra", {}) or {}
            target_name = extra.get("references")

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

    def init_optional_tensors(self) -> None:
        """
        Initialize optional tensor fields:
        - If None -> zeros with expected shape
        - If scalar default -> full with that value and expected shape
        - If already a tensor -> skip
        """
        for name, field_info in self.get_model_fields().items():
            # Check if it is a TensorField by looking for tensor_shape in json_schema_extra
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if not isinstance(json_schema_extra, dict) or 'tensor_shape' not in json_schema_extra:
                continue

            if name in self.model_fields_set:
                continue
            value = getattr(self, name, None)
            # shape
            expected_shape = self.get_expected_shape(name)
            if expected_shape is None:
                continue
            # dtype
            tensor_dtype = str(json_schema_extra.get('tensor_dtype', 'float'))
            dtype_map = {
                'float': self.precision,
                'hpfloat': self.high_precision,
                'int': torch.int64,
                'idx': torch.int32,
                'bool': torch.bool
            }
            target_dtype = dtype_map.get(tensor_dtype)

            if value is None:
                tensor_value = None
            elif isinstance(value, (int, float, bool)):
                tensor_value = torch.full(expected_shape, fill_value=value, dtype=target_dtype, device=self.device)
            else:
                raise TypeError(f"Unsupported default type for {name}: {type(value)}")

            setattr(self, name, tensor_value)

    def get_expected_shape(self, field_name: str) -> Optional[Tuple[int, ...]]:
        """
        Get the expected shape for a tensor field based on current scalar values.

        Args:
            field_name: Name of the tensor field

        Returns:
            Tuple of integer dimensions, or None if no shape is defined
        """
        model_fields = self.get_model_fields() | self.get_model_computed_fields()
        if field_name not in model_fields:
            raise ValueError(f"Field {field_name} is not a tensor field")
        json_schema_extra = getattr(model_fields[field_name], 'json_schema_extra', None)
        if not isinstance(json_schema_extra, dict):
            json_schema_extra = {}
        shape_spec = json_schema_extra.get('tensor_shape', None)
        if shape_spec is None:
            return None

        # Get current scalar values from instance
        scalar_values = {}
        for dim_name in shape_spec:
            # Literal integer → use directly
            if isinstance(dim_name, int):
                scalar_values[dim_name] = dim_name
                continue

            # Handle dotted notation (e.g., "base.num_flood_levels")
            if "." in dim_name:
                parts = dim_name.split(".")
                if len(parts) != 2:
                    raise ValueError(f"Invalid dimension format: {dim_name}. Expected 'module.attribute'")
                module_name, attr_name = parts
                if not hasattr(self, module_name):
                    raise ValueError(f"Module {module_name} not found in {self.module_name} for dimension {dim_name}")
                module_obj = getattr(self, module_name)
                if not hasattr(module_obj, attr_name):
                    raise ValueError(f"Attribute {attr_name} not found in module {module_name} for dimension {dim_name}")
                scalar_values[dim_name] = getattr(module_obj, attr_name)
                continue

            if hasattr(self, dim_name):
                scalar_values[dim_name] = getattr(self, dim_name)
            else:
                raise ValueError(f"Dimension {dim_name} not found in module")

        shape = tuple(scalar_values[dim] for dim in shape_spec)
        allow_empty = bool(json_schema_extra.get("allow_empty", False))
        for dim_name, size in zip(shape_spec, shape, strict=True):
            if isinstance(size, bool) or not isinstance(size, Integral):
                raise TypeError(
                    f"Dimension '{dim_name}' used by field '{field_name}' must "
                    f"be an integer, got {type(size).__name__}"
                )
            if size < 0 or (size == 0 and not allow_empty):
                requirement = "non-negative" if allow_empty else "positive"
                raise ValueError(
                    f"Dimension '{dim_name}' used by field '{field_name}' must "
                    f"be {requirement}, got {size}"
                )

        category = json_schema_extra.get('category', 'param')
        if self.num_trials is not None:
            is_batched = category in ('state', 'init_state') or (
                category in ('param', 'derived_param')
                and field_name in self._expanded_params
            )
            if is_batched:
                return (self.num_trials,) + shape

        return shape


    def get_expected_dtype(self, field_name: str) -> torch.dtype:
        """
        Get the expected data type for a tensor field based on its definition.

        Args:
            field_name: Name of the tensor field

        Returns:
            Expected torch.dtype for the tensor
        """
        model_fields = self.get_model_fields() | self.get_model_computed_fields()
        if field_name not in model_fields:
            raise ValueError(f"Field {field_name} is not a tensor field")
        json_schema_extra = getattr(model_fields[field_name], 'json_schema_extra', None)
        if not isinstance(json_schema_extra, dict):
            json_schema_extra = {}
        dtype_str = str(json_schema_extra.get('tensor_dtype', 'float'))

        dtype_map = {
            'float': self.precision,
            'hpfloat': self.high_precision,
            'int': torch.int64,
            'idx': torch.int32,
            'bool': torch.bool
        }

        return dtype_map.get(dtype_str, torch.float32)

    def validate_tensors(self) -> bool:
        """
        Validate and auto-fix tensor consistency issues.
        - Ensures contiguity
        - Validates shapes (fails on mismatch)
        - Ensures device consistency (moves to self.device if needed)
        - Ensures precision consistency for floating-point tensors
        - Ensures int tensors are int64, idx tensors are int32
        """
        auto_fix_log = {}

        # Pass 0: Move ALL tensors to the correct device first.
        for field_name, field_info in self.get_model_fields().items():
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if not isinstance(json_schema_extra, dict) or 'tensor_shape' not in json_schema_extra:
                continue
            tensor = getattr(self, field_name, None)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != self.device.type or (
                tensor.device.index is not None
                and self.device.index is not None
                and tensor.device.index != self.device.index
            ):
                setattr(self, field_name, tensor.to(self.device))

        for field_name, field_info in self.get_model_fields().items():
            # Check if it is a TensorField by looking for tensor_shape in json_schema_extra
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if not isinstance(json_schema_extra, dict) or 'tensor_shape' not in json_schema_extra:
                continue

            tensor = getattr(self, field_name, None)

            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue

            # 1. Shape validation (fail fast)
            expected_shape = self.get_expected_shape(field_name)
            if expected_shape is None:
                continue
            if tensor.shape != expected_shape:
                # Try to expand if it's a state/init_state variable and shape matches without batch dim
                category = json_schema_extra.get('category', 'param')
                if category in ('state', 'init_state') and self.num_trials is not None:
                    # Check if it matches expected shape excluding the first dim (num_trials)
                    if tensor.shape == expected_shape[1:]:
                        tensor = tensor.unsqueeze(0).expand(expected_shape).clone()
                        setattr(self, field_name, tensor)
                    else:
                        raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")
                elif category in ('param', 'derived_param') and self.num_trials is not None:
                    # Check if it is already batched but not marked as expanded
                    if tensor.shape[0] == self.num_trials and tensor.shape[1:] == expected_shape:
                        # It is batched, so we should mark it as expanded
                        self._expanded_params.add(field_name)
                        # Re-check expected shape now that it is marked expanded
                        expected_shape = self.get_expected_shape(field_name)
                        if tensor.shape != expected_shape:
                             raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")
                    else:
                        raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")
                else:
                    raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")

            # 2. Auto-fix contiguity
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            # 3. Auto-fix device mismatch
            if tensor.device.type != self.device.type or (
                tensor.device.index is not None
                and self.device.index is not None
                and tensor.device.index != self.device.index
            ):
                tensor = tensor.to(self.device)

            # 4. Auto-fix precision for floating-point tensors
            expected_dtype = self.get_expected_dtype(field_name)
            tensor_dtype = tensor.dtype
            if tensor_dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
                key = f"{tensor_dtype} -> {expected_dtype}"
                if key not in auto_fix_log:
                    auto_fix_log[key] = []
                auto_fix_log[key].append(field_name)

            # 5. Key field validation: 1D integer + unique values.
            if json_schema_extra.get("is_key", False):
                if tensor.dtype not in (torch.int32, torch.int64):
                    raise ValueError(
                        f"Key field '{field_name}' must be int32/int64, got {tensor.dtype}"
                    )
                if tensor.ndim != 1:
                    raise ValueError(
                        f"Key field '{field_name}' must be 1D, got shape {tuple(tensor.shape)}"
                    )
                if tensor.numel() > 0:
                    vals, counts = torch.unique(tensor, return_counts=True)
                    dup_mask = counts > 1
                    if bool(dup_mask.any()):
                        n_dup = int(dup_mask.sum().item())
                        sample = vals[dup_mask][:5].tolist()
                        raise ValueError(
                            f"Key field '{field_name}' has {n_dup} duplicate value(s); "
                            f"first few: {sample}"
                        )

            # Update tensor if it was modified
            setattr(self, field_name, tensor)

        if auto_fix_log:
            for key, fields in auto_fix_log.items():
                print(f"Auto-fixed dtype for fields: {', '.join(fields)} ({key})")
        return True

    def validate_computed_tensors(self) -> bool:
        """
        Validate computed tensors to ensure they are correctly defined.
        """
        auto_fix_log = {}
        for field_name, field_info in self.get_model_computed_fields().items():
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if not isinstance(json_schema_extra, dict):
                continue

            # Need to skip virtual fields
            category = json_schema_extra.get('category', 'derived_param')
            if category == 'virtual':
                continue

            # Skip fields whose dependency module is not opened
            depends_on = json_schema_extra.get('depends_on')
            if depends_on is not None and depends_on not in self.opened_modules:
                continue

            tensor = getattr(self, field_name)
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != self.device.type or (
                tensor.device.index is not None
                and self.device.index is not None
                and tensor.device.index != self.device.index
            ):
                raise ValueError(
                    f"Computed field {field_name} must be on device {self.device}, "
                    f"but is on {tensor.device}"
                )

            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
                setattr(self, field_name, tensor)

            expected_shape = self.get_expected_shape(field_name)
            if expected_shape is None:
                continue
            if tensor.shape != expected_shape:
                # Check if it is batched (num_trials > 1)
                if self.num_trials is not None and tensor.shape == (self.num_trials,) + expected_shape:
                    # Mark as expanded so get_expected_shape returns the correct batched shape
                    self._expanded_params.add(field_name)
                else:
                    raise ValueError(
                        f"Computed field {field_name} has shape {tensor.shape}, "
                        f"but expected shape is {expected_shape}"
                    )

            expected_dtype = self.get_expected_dtype(field_name)
            if tensor.dtype != expected_dtype:
                key = f"{tensor.dtype} -> {expected_dtype}"
                if key not in auto_fix_log:
                    auto_fix_log[key] = []
                auto_fix_log[key].append(field_name)
                tensor = tensor.to(expected_dtype)
                setattr(self, field_name, tensor)

        if auto_fix_log:
            for key, fields in auto_fix_log.items():
                print(f"Auto-fixed dtype for computed fields: {', '.join(fields)} ({key})")
        return True

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


    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of the module in bytes.
        Excludes intermediate tensors.
        Only counts tensors currently on the active computing device.
        Uses data_ptr deduplication to avoid double-counting shared tensors.
        """
        total_bytes = 0
        seen_ptrs: set = set()

        # Combine fields and computed fields
        all_fields = self.get_model_fields().copy()
        all_fields.update(self.get_model_computed_fields())

        for name in all_fields:

            # Get the value
            if not hasattr(self, name):
                continue
            value = getattr(self, name)

            # Check if it's a tensor
            if isinstance(value, torch.Tensor):
                # Only count if on the main computing device
                if value.device.type == self.device.type:
                    ptr = value.data_ptr()
                    if ptr not in seen_ptrs:
                        seen_ptrs.add(ptr)
                        total_bytes += value.element_size() * value.nelement()

        return total_bytes

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
            info = (
                self.get_model_fields().get(field)
                or self.get_model_computed_fields().get(field)
            )
            if info is None:
                raise KeyError(f"Unknown tensor field: {field}")
            extra = getattr(info, "json_schema_extra", {}) or {}
            if extra.get("category") == "topology":
                return False
            declared_rank = len(extra.get("tensor_shape", ()))
            return tensor.ndim == declared_rank + 1
        tensor = field
        return tensor.ndim > 0 and tensor.shape[0] == self.num_trials

    _is_batched = is_batched

    @property
    def batched_params(self) -> Dict[str, bool]:
        """
        Returns a dictionary mapping parameter names to whether they are batched.
        Checks both fields and computed fields.
        """
        res = {}
        # Helper to check a field
        def check_field(name, field_info):
            # Skip state variables and topology
            category = field_info.json_schema_extra.get("category", "param") if field_info.json_schema_extra else "param"
            if category == "state" or category == "topology":
                return

            if not hasattr(self, name):
                return

            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                res[name] = self.is_batched(name)

        for name, field in self.get_model_fields().items():
            check_field(name, field)

        for name, field in self.get_model_computed_fields().items():
            check_field(name, field)

        return res

    def handle_tensor_mode(self) -> None:
        """
        Process variables based on their mode setting:
        - 'device': Keep on current device (default)
        - 'cpu': Move to CPU memory
        - 'discard': Remove from memory (set to None) to save space
        """
        for name, field_info in self.get_model_fields().items():
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            if json_schema_extra is None:
                continue

            mode = json_schema_extra.get('mode', 'device')

            if mode == 'device':
                continue

            if not hasattr(self, name):
                continue

            val = getattr(self, name)
            if not isinstance(val, torch.Tensor):
                # Could be None already
                continue

            if mode == 'cpu':
                if val.device.type != 'cpu':
                    setattr(self, name, val.cpu())
            elif mode == 'discard':
                setattr(self, name, None)

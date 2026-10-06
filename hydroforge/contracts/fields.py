# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Generic field contracts extracted from module declarations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeAlias

import torch
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from hydroforge.contracts.conditions import (
    Condition,
    any_condition_satisfied,
    conditions_satisfied,
)
from hydroforge.core.arrays import TORCH_INTEGER_DTYPES
from hydroforge.core.expr import parse_value_source
from hydroforge.core.naming import DottedPath
from hydroforge.core.validation import (
    FrozenMapping,
    HydroForgeModel,
    require_unique,
)


def _tensor_name(value: str) -> str:
    if len(value.split(".")) > 2:
        raise ValueError("tensor references must be identifier or module.identifier")
    return value


TensorName: TypeAlias = Annotated[DottedPath, AfterValidator(_tensor_name)]
TensorShape: TypeAlias = tuple[TensorName | Annotated[int, Field(ge=0)], ...]
TensorDType = Literal["float", "hpfloat", "int", "idx", "bool"]
TensorOutput = Literal["auto", "full", "disabled"]


_Conditions: TypeAlias = Annotated[
    tuple[Condition, ...],
    AfterValidator(require_unique),
]
TensorDependencies: TypeAlias = Any
"""A condition, a module name, an ``options.<path>`` string, or a tuple."""


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
    conditions: Mapping[str, bool] | None = None,
) -> bool:
    """Evaluate conditional tensor residency for one specialization.

    Output demand activates ``output_only`` and ``required_by`` storage but
    never bypasses ``depends_on``.
    """
    if isinstance(metadata, RuntimeTensorMetadata):
        metadata = metadata.tensor
    opened = frozenset(opened_modules)
    required = getattr(metadata, "depends_on", ())
    consumers = getattr(metadata, "required_by", ())
    output_only = getattr(metadata, "output_only", False)
    return (
        conditions_satisfied(required, opened, conditions)
        and (not output_only or output_required)
        and (
            not consumers
            or output_required
            or any_condition_satisfied(consumers, opened, conditions)
        )
    )


def offending_indices(mask: torch.Tensor, limit: int = 5) -> list[Any]:
    """Return the first ``limit`` indices where ``mask`` is true.

    One-dimensional masks yield ints; higher ranks yield index tuples.
    """

    return [
        tuple(index) if len(index) != 1 else index[0]
        for index in torch.nonzero(mask)[:limit].tolist()
    ]


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


def precision_dtype(name: str, precision: torch.dtype) -> torch.dtype:
    """Resolve a ``"precision"`` placeholder or an exact torch dtype name."""

    return precision if name == "precision" else getattr(torch, name)


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
        if tensor.dtype not in TORCH_INTEGER_DTYPES:
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


@pydantic_dataclass(frozen=True, slots=True, config=ConfigDict(strict=True))
class TensorMetadata:
    """Canonical tensor-field metadata, validated once by its field factory.

    A validated dataclass rather than a model: field factories append it to
    ``FieldInfo.metadata``, where Pydantic ignores it for core schemas.
    """

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
    depends_on: _Conditions = ()
    required_by: _Conditions = ()
    expression: str = ""
    output_only: bool = False
    units: str | None = None
    finite: bool = False
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None

    @field_validator("depends_on", "required_by", mode="before")
    @classmethod
    def _canonical_dependencies(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return ()
        if type(value) is not tuple and type(value) is not list:
            return (value,)
        # A before validator hands JSON arrays on as Python lists.
        if info.mode == "json" and type(value) is list:
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        if self.expression:
            parse_value_source(self.expression)
        if self.is_key and (len(self.shape) != 1 or self.dtype not in {"int", "idx"}):
            raise ValueError("key fields require one-dimensional integer metadata")
        if self.is_coordinate and not self.is_key:
            raise ValueError("coordinate fields require key semantics")
        if (
            self.selects or self.replicated or self.partition_by
        ) and not self.is_coordinate:
            raise ValueError(
                "selects, replicated and partition_by require a coordinate field"
            )
        if self.selects and self.references != self.selects:
            raise ValueError("selection references must name its selects target")
        if self.replicated and (self.partition_by or self.references):
            raise ValueError("replicated coordinates cannot declare partition lineage")
        if self.dim_coords is not None and not self.shape:
            raise ValueError(
                "dim_coords names the coordinate of dimension 0 and requires "
                "a non-scalar shape"
            )
        if self.mode == "discard":
            if self.is_coordinate:
                raise ValueError(
                    "coordinate fields cannot use mode='discard'; outputs and "
                    "checkpoints read their runtime values"
                )
            if self.category not in {"topology", "param"}:
                raise ValueError(
                    "mode='discard' is only valid for construction-time "
                    "topology or parameter fields"
                )
            if self.output != "disabled":
                raise ValueError("mode='discard' fields must use output='disabled'")
        if self.category == "forcing":
            if self.mode != "device":
                raise ValueError("forcing fields must use mode='device'")
            if self.output != "disabled":
                raise ValueError("forcing fields must use output='disabled'")
            if self.is_key or self.is_coordinate or self.references or self.selects:
                raise ValueError(
                    "forcing fields cannot define topology/key relationships"
                )
        if self.expression and self.category != "virtual":
            raise ValueError("expr can only be provided when category is 'virtual'")
        if self.output_only:
            if self.category == "virtual":
                raise ValueError("output_only is invalid for virtual fields")
            if self.output == "disabled":
                raise ValueError(
                    "output_only fields must permit explicit statistics output"
                )
        self._validate_value_contract()
        return self

    def _validate_value_contract(self) -> None:
        """Validate units and value constraints."""

        if self.units is not None and (
            not self.units or self.units != self.units.strip()
        ):
            raise ValueError("units must be a non-empty string without padding")
        bounds = {
            name: getattr(self, name)
            for name in ("ge", "gt", "le", "lt")
            if getattr(self, name) is not None
        }
        if (self.finite or bounds) and (
            self.dtype == "bool" or self.category == "virtual"
        ):
            raise ValueError(
                "finite and value bounds require a stored numeric tensor field"
            )
        if self.finite and self.dtype not in {"float", "hpfloat"}:
            raise ValueError("finite applies only to floating dtypes")
        if any(not math.isfinite(value) for value in bounds.values()):
            raise ValueError("value bounds must be finite numbers")
        if "ge" in bounds and "gt" in bounds:
            raise ValueError("declare at most one of ge and gt")
        if "le" in bounds and "lt" in bounds:
            raise ValueError("declare at most one of le and lt")
        lower = bounds.get("ge", bounds.get("gt"))
        upper = bounds.get("le", bounds.get("lt"))
        if (
            lower is not None
            and upper is not None
            and (
                lower > upper or (lower == upper and ("gt" in bounds or "lt" in bounds))
            )
        ):
            raise ValueError(f"value bounds {bounds} admit no value")

    @property
    def has_value_constraints(self) -> bool:
        """Whether materialization checks finiteness or bounds of this field."""

        return self.finite or any(
            getattr(self, name) is not None for name in ("ge", "gt", "le", "lt")
        )

    def value_violations(self, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Return ``(constraint, mask)`` for every violated value constraint.

        With ``finite`` NaN violates the bounds too; a non-finite field may
        hold NaN, and its bounds apply to the other values.
        """

        return value_violations(
            tensor, finite=self.finite, ge=self.ge, gt=self.gt, le=self.le, lt=self.lt
        )


def value_violations(
    tensor: torch.Tensor,
    *,
    finite: bool = False,
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Return ``(constraint, mask)`` for every violated value constraint.

    With ``finite`` NaN and infinities are rejected and NaN also violates the
    bounds; without it NaN is allowed and the bounds apply to the other
    values. Reads one flag array back from the tensor's device.
    """

    return violated_masks(
        constraint_masks(tensor, finite=finite, ge=ge, gt=gt, le=le, lt=lt)
    )


def constraint_masks(
    tensor: torch.Tensor,
    *,
    finite: bool = False,
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Return ``(constraint, mask)`` of every declared constraint, unchecked.

    A mask is true where the value violates its constraint. Building the masks
    never synchronizes the device; :func:`violated_masks` reads them back.
    """

    checks: list[tuple[str, torch.Tensor]] = []
    allow_nan = not finite and tensor.is_floating_point()
    if finite and tensor.is_floating_point():
        checks.append(("finite", ~torch.isfinite(tensor)))
    for name, bound, holds in (
        ("ge", ge, torch.greater_equal),
        ("gt", gt, torch.greater),
        ("le", le, torch.less_equal),
        ("lt", lt, torch.less),
    ):
        if bound is not None:
            violated = ~holds(tensor, bound)
            if allow_nan:
                violated &= ~torch.isnan(tensor)
            checks.append((f"{name}={bound!r}", violated))
    return checks


def violated_masks(
    checks: list[tuple[Any, torch.Tensor]],
) -> list[tuple[Any, torch.Tensor]]:
    """Keep the ``(key, mask)`` entries whose mask has a true value.

    Flags are reduced per device and read back with one synchronization per
    device, however many masks are checked.
    """

    by_device: dict[torch.device, list[int]] = {}
    for index, (_key, mask) in enumerate(checks):
        by_device.setdefault(mask.device, []).append(index)
    violated: set[int] = set()
    for indices in by_device.values():
        flags = torch.stack([checks[index][1].any() for index in indices]).tolist()
        violated.update(index for index, flag in zip(indices, flags) if flag)
    return [check for index, check in enumerate(checks) if index in violated]


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
    resolved_shape: tuple[int, ...] | None = None

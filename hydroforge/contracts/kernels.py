"""Canonical kernel contracts shared by every HydroForge backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias

if TYPE_CHECKING:
    import torch


AccessMode = Literal[
    "read", "write", "read_write",
    "atomic_write", "atomic_add", "atomic_min", "atomic_max",
]
ScalarKind = Literal["bool", "int32", "float32"]
RuntimeScalarKind = Literal["bool", "int32", "index", "float32"]
LoweringMode = Literal["canonical", "plan", "declared"]
ParameterOrder = Literal["canonical", "native"]
BufferAccessLowering = Literal["exact", "conservative"]
BufferElementLowering = Literal["tensor", "specialized"]
BufferDTypeABI: TypeAlias = Mapping[str, "torch.dtype"]
_LAUNCH_BACKENDS = frozenset({"cuda", "triton", "metal"})


@dataclass(frozen=True, slots=True)
class BufferAccessSemantics:
    """Canonical dependency and native-storage meaning of one access mode."""

    reads: bool
    writes: bool
    atomic: bool
    dependency: Literal["read", "write", "read_write"]


_BUFFER_ACCESS_SEMANTICS = MappingProxyType({
    "read": BufferAccessSemantics(True, False, False, "read"),
    "write": BufferAccessSemantics(False, True, False, "write"),
    "read_write": BufferAccessSemantics(True, True, False, "read_write"),
    "atomic_write": BufferAccessSemantics(False, True, True, "write"),
    "atomic_add": BufferAccessSemantics(True, True, True, "read_write"),
    "atomic_min": BufferAccessSemantics(True, True, True, "read_write"),
    "atomic_max": BufferAccessSemantics(True, True, True, "read_write"),
})
BUFFER_ACCESS_MODES = tuple(_BUFFER_ACCESS_SEMANTICS)


def buffer_access_semantics(access: str) -> BufferAccessSemantics:
    """Return the one defined meaning of a KernelSpec buffer access."""

    try:
        return _BUFFER_ACCESS_SEMANTICS[access]
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid buffer access mode {access!r}") from error


def _validate_buffer_accesses(name: str, buffers: Mapping[str, Any]) -> None:
    invalid = []
    for access in buffers.values():
        try:
            buffer_access_semantics(access)
        except ValueError:
            invalid.append(access)
    invalid.sort(key=repr)
    if invalid:
        raise ValueError(f"{name}: invalid buffer access: {invalid}")


def _frozen_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


def _host_scalar_is_valid(value: Any, kind: RuntimeScalarKind) -> bool:
    """Define canonical host scalar semantics once, without coercion."""

    if kind == "bool":
        return type(value) is bool
    if kind == "int32":
        return type(value) is int and -(2 ** 31) <= value < 2 ** 31
    if kind == "index":
        return type(value) is int and -(2 ** 63) <= value < 2 ** 63
    if kind == "float32":
        return bool(
            type(value) is float
            and math.isfinite(value)
            and abs(value) <= 3.4028234663852886e38
        )
    raise RuntimeError(f"unknown canonical host scalar kind {kind!r}")


def _optional_value_declaration(
    kernel: str, argument: str, declaration: Any,
) -> tuple[str, Any]:
    """Validate one optional-value declaration before interpreting it."""

    if type(declaration) is not tuple or len(declaration) != 2:
        raise TypeError(
            f"{kernel}: optional value {argument!r} must be declared as the "
            "exact tuple (feature, disabled_sentinel)"
        )
    feature, disabled = declaration
    if not isinstance(feature, str) or not feature.isidentifier():
        raise ValueError(
            f"{kernel}: optional value {argument!r} feature must be a valid "
            f"Python identifier, got {feature!r}"
        )
    return feature, disabled


def validate_launch_extent(
    name: str,
    size_key: str | tuple[str, ...],
    arguments: Mapping[str, Any],
) -> int:
    """Validate and flatten one logical launch geometry without coercion."""

    keys = (size_key,) if isinstance(size_key, str) else size_key
    extent = 1
    for key in keys:
        value = arguments[key]
        if type(value) is not int:
            raise TypeError(
                f"{name}.{key} launch extent must be an exact host int, got "
                f"{type(value).__name__}"
            )
        if value < 0:
            raise ValueError(
                f"{name}.{key} launch extent must be non-negative"
            )
        extent *= value
        if extent >= 2 ** 63:
            raise OverflowError(
                f"{name} launch extent exceeds signed int64 range"
            )
    return extent


@dataclass(frozen=True)
class KernelMetadata:
    """Concrete metadata exposed by one specialized backend implementation."""

    name: str
    parameters: tuple[str, ...]
    size_key: str | tuple[str, ...]
    buffers: Mapping[str, AccessMode]
    optional_buffers: Mapping[str, str | None]
    compile_time: Mapping[str, ScalarKind]
    runtime_scalars: Mapping[str, RuntimeScalarKind] = field(default_factory=dict)
    optional_values: Mapping[str, tuple[str, Any]] = field(default_factory=dict)
    block_sizes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        for name in (
            "buffers", "optional_buffers", "compile_time",
            "runtime_scalars", "optional_values", "block_sizes",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"KernelMetadata.{name} must be a mapping")
            object.__setattr__(self, name, _frozen_mapping(value))
        _validate_buffer_accesses(self.name, self.buffers)


@dataclass(frozen=True, slots=True)
class BackendLoweringSpec:
    """Explicit representation of canonical constants in a native adapter."""

    mode: LoweringMode
    native_constants: Mapping[str, ScalarKind] = field(default_factory=dict)
    parameter_order: ParameterOrder = "native"
    buffer_access: BufferAccessLowering = "exact"
    buffer_elements: BufferElementLowering = field(kw_only=True)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mode, str)
            or self.mode not in {"canonical", "plan", "declared"}
        ):
            raise ValueError(f"invalid backend lowering mode {self.mode!r}")
        if (
            not isinstance(self.parameter_order, str)
            or self.parameter_order not in {"canonical", "native"}
        ):
            raise ValueError(
                f"invalid backend parameter order {self.parameter_order!r}"
            )
        if (
            not isinstance(self.buffer_access, str)
            or self.buffer_access not in {"exact", "conservative"}
        ):
            raise ValueError(
                f"invalid backend buffer access {self.buffer_access!r}"
            )
        if (
            not isinstance(self.buffer_elements, str)
            or self.buffer_elements not in {"tensor", "specialized"}
        ):
            raise ValueError(
                f"invalid backend buffer elements {self.buffer_elements!r}"
            )
        if not isinstance(self.native_constants, Mapping):
            raise TypeError("backend native_constants must be a mapping")
        invalid_names = [
            name for name in self.native_constants
            if not isinstance(name, str) or not name.isidentifier()
        ]
        if invalid_names:
            raise ValueError(
                "backend native constant names must be identifiers: "
                f"{sorted(invalid_names, key=repr)}"
            )
        invalid_kinds = [
            kind for kind in self.native_constants.values()
            if kind not in ("bool", "int32", "float32")
        ]
        if invalid_kinds:
            raise ValueError(
                "invalid backend native constant kind(s): "
                f"{sorted(invalid_kinds, key=repr)}"
            )
        if self.mode != "declared" and self.native_constants:
            raise ValueError(
                f"{self.mode} backend lowering may not declare native_constants"
            )
        object.__setattr__(
            self, "native_constants", _frozen_mapping(self.native_constants),
        )

    def compile_time_for(self, spec: KernelSpec) -> Mapping[str, ScalarKind]:
        """Derive the native constexpr ABI from one canonical KernelSpec."""

        values = {
            "canonical": spec.compile_time,
            "plan": {},
            "declared": self.native_constants,
        }[self.mode]
        unknown = set(values).difference(spec.compile_time)
        mistyped = {
            name: kind for name, kind in values.items()
            if spec.compile_time.get(name) != kind
        }
        if unknown or mistyped:
            raise TypeError(
                f"{spec.name}: BackendLoweringSpec contains undeclared or "
                f"mistyped constants={sorted(unknown)!r}/{mistyped!r}"
            )
        return values

    @classmethod
    def canonical(
        cls, *, buffer_elements: BufferElementLowering,
    ) -> "BackendLoweringSpec":
        return cls(
            "canonical", parameter_order="canonical",
            buffer_elements=buffer_elements,
        )

    @classmethod
    def plan_specialized(
        cls, *, buffer_elements: BufferElementLowering,
    ) -> "BackendLoweringSpec":
        return cls(
            "plan", parameter_order="canonical", buffer_access="exact",
            buffer_elements=buffer_elements,
        )

    @classmethod
    def declared(
        cls, constants: Mapping[str, ScalarKind], *,
        buffer_elements: BufferElementLowering,
    ) -> "BackendLoweringSpec":
        return cls(
            "declared", constants, buffer_elements=buffer_elements,
        )


@dataclass(frozen=True)
class KernelSpec:
    """The single backend-neutral ABI for one logical kernel.

    A spec is declared beside the logical :class:`BackendRegistry`; backend
    factories only provide implementations.  No backend is allowed to infer,
    add, remove, reorder, or rename public arguments.
    """

    name: str
    parameters: tuple[str, ...]
    size_key: str | tuple[str, ...]
    buffers: Mapping[str, AccessMode]
    optional_buffers: Mapping[str, str | None] = field(default_factory=dict)
    compile_time: Mapping[str, ScalarKind] = field(default_factory=dict)
    runtime_scalars: Mapping[str, RuntimeScalarKind] = field(default_factory=dict)
    optional_values: Mapping[str, tuple[str, Any]] = field(default_factory=dict)
    block_sizes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("KernelSpec.name must be a valid Python identifier")
        for field_name in (
            "buffers", "optional_buffers", "compile_time",
            "runtime_scalars", "optional_values", "block_sizes",
        ):
            if not isinstance(getattr(self, field_name), Mapping):
                raise TypeError(
                    f"{self.name}: {field_name} must be a mapping"
                )
        invalid_parameters = [
            value for value in parameters
            if not isinstance(value, str) or not value.isidentifier()
        ]
        if invalid_parameters:
            raise ValueError(
                f"{self.name}: canonical parameters must be valid Python "
                f"identifiers: {invalid_parameters!r}"
            )
        if len(parameters) != len(set(parameters)):
            raise ValueError(f"{self.name}: duplicate canonical parameters")
        parameter_set = set(parameters)
        if isinstance(self.size_key, str):
            size_keys = (self.size_key,)
        elif isinstance(self.size_key, tuple):
            size_keys = self.size_key
        else:
            raise TypeError(
                f"{self.name}: size_key must be a string or tuple of strings"
            )
        if (
            not size_keys
            or any(
                not isinstance(value, str) or not value.isidentifier()
                for value in size_keys
            )
            or len(size_keys) != len(set(size_keys))
        ):
            raise ValueError(
                f"{self.name}: size_key must contain one or more unique "
                "Python identifiers"
            )
        missing_size = set(size_keys).difference(parameter_set)
        if missing_size:
            raise ValueError(
                f"{self.name}: size key(s) outside canonical ABI: {sorted(missing_size)}"
            )
        unknown_buffers = set(self.buffers).difference(parameter_set)
        if unknown_buffers:
            raise ValueError(
                f"{self.name}: buffer(s) outside canonical ABI: {sorted(unknown_buffers)}"
            )
        buffer_extents = set(size_keys).intersection(self.buffers)
        if buffer_extents:
            raise ValueError(
                f"{self.name}: size key(s) must be host scalars, not buffers: "
                f"{sorted(buffer_extents)}"
            )
        _validate_buffer_accesses(self.name, self.buffers)
        unknown_optional = set(self.optional_buffers).difference(self.buffers)
        if unknown_optional:
            raise ValueError(
                f"{self.name}: optional non-buffer argument(s): {sorted(unknown_optional)}"
            )
        unknown_values = set(self.optional_values).difference(parameter_set)
        if unknown_values:
            raise ValueError(
                f"{self.name}: optional value(s) outside ABI: {sorted(unknown_values)}"
            )
        unknown_constants = set(self.compile_time).difference(parameter_set)
        if unknown_constants:
            raise ValueError(
                f"{self.name}: compile-time value(s) outside ABI: "
                f"{sorted(unknown_constants)}"
            )
        invalid_constant_kinds = set(self.compile_time.values()).difference(
            {"bool", "int32", "float32"},
        )
        if invalid_constant_kinds:
            raise ValueError(
                f"{self.name}: invalid compile-time scalar kind(s): "
                f"{sorted(invalid_constant_kinds)}"
            )
        noncanonical_features = sorted(
            name for name, kind in self.compile_time.items()
            if kind == "bool" and name.isupper() and not name.startswith("HAS_")
        )
        if noncanonical_features:
            raise ValueError(
                f"{self.name}: uppercase capability flags must use the "
                f"canonical HAS_* spelling: {noncanonical_features}"
            )
        unknown_runtime = set(self.runtime_scalars).difference(parameter_set)
        if unknown_runtime:
            raise ValueError(
                f"{self.name}: runtime scalar(s) outside canonical ABI: "
                f"{sorted(unknown_runtime)}"
            )
        invalid_runtime_kinds = set(self.runtime_scalars.values()).difference(
            {"bool", "int32", "index", "float32"},
        )
        if invalid_runtime_kinds:
            raise ValueError(
                f"{self.name}: invalid runtime scalar kind(s): "
                f"{sorted(invalid_runtime_kinds)}"
            )
        contradictory = set(self.buffers).intersection(self.compile_time)
        if contradictory:
            raise ValueError(
                f"{self.name}: parameter(s) cannot be both buffers and "
                f"compile-time scalars: {sorted(contradictory)}"
            )
        classified = (
            set(self.buffers) | set(self.compile_time)
            | set(self.runtime_scalars)
        )
        overlaps = (
            set(self.buffers).intersection(self.runtime_scalars)
            | set(self.compile_time).intersection(self.runtime_scalars)
        )
        if overlaps:
            raise ValueError(
                f"{self.name}: parameter(s) have multiple ABI classes: "
                f"{sorted(overlaps)}"
            )
        unclassified = parameter_set.difference(classified)
        if unclassified:
            raise ValueError(
                f"{self.name}: every parameter must be declared exactly once "
                "as a buffer, compile-time scalar, or runtime scalar; "
                f"unclassified={sorted(unclassified)}"
            )
        invalid_extents = {
            name for name in size_keys
            if self.runtime_scalars.get(name) != "index"
        }
        if invalid_extents:
            raise ValueError(
                f"{self.name}: launch extent scalar(s) must have runtime kind "
                f"'index': {sorted(invalid_extents)}"
            )
        duplicate_optional = set(self.optional_buffers).intersection(
            self.optional_values,
        )
        if duplicate_optional:
            raise ValueError(
                f"{self.name}: optional parameters must be declared as either "
                "buffers or values, not both: "
                f"{sorted(duplicate_optional)}"
            )
        feature_names = set(self.compile_time)
        for argument, feature in self.optional_buffers.items():
            if feature is None:
                continue
            if not isinstance(feature, str) or not feature.isidentifier():
                raise ValueError(
                    f"{self.name}: optional buffer {argument!r} feature must "
                    f"be a valid Python identifier, got {feature!r}"
                )
            if feature not in feature_names:
                raise ValueError(
                    f"{self.name}: optional buffer {argument!r} references "
                    f"undeclared compile-time feature {feature!r}"
                )
            if self.compile_time[feature] != "bool":
                raise ValueError(
                    f"{self.name}: optional buffer {argument!r} requires bool "
                    f"feature {feature!r}, got {self.compile_time[feature]!r}"
                )
        for argument, declaration in self.optional_values.items():
            feature, disabled = _optional_value_declaration(
                self.name, argument, declaration,
            )
            if feature not in feature_names:
                raise ValueError(
                    f"{self.name}: optional value {argument!r} references "
                    f"undeclared compile-time feature {feature!r}"
                )
            if self.compile_time[feature] != "bool":
                raise ValueError(
                    f"{self.name}: optional value {argument!r} requires bool "
                    f"feature {feature!r}, got {self.compile_time[feature]!r}"
                )
            if argument not in self.runtime_scalars:
                raise ValueError(
                    f"{self.name}: optional value {argument!r} must be a "
                    "runtime scalar"
                )
            kind = self.runtime_scalars[argument]
            if not _host_scalar_is_valid(disabled, kind):
                raise TypeError(
                    f"{self.name}: optional value {argument!r} disabled "
                    f"sentinel must be an exact finite {kind} host scalar, "
                    f"got {disabled!r} ({type(disabled).__name__})"
                )
        if not isinstance(self.block_sizes, Mapping):
            raise TypeError(f"{self.name}: block_sizes must be a mapping")
        unknown_launch_backends = set(self.block_sizes).difference(
            _LAUNCH_BACKENDS,
        )
        if unknown_launch_backends:
            raise ValueError(
                f"{self.name}: block_sizes contains unsupported backends: "
                f"{sorted(unknown_launch_backends)}"
            )
        invalid_block_sizes = {
            backend: value
            for backend, value in self.block_sizes.items()
            if type(value) is not int or not 1 <= value <= 1024
        }
        if invalid_block_sizes:
            raise ValueError(
                f"{self.name}: backend block sizes must be exact ints in "
                f"[1, 1024]: {invalid_block_sizes}"
            )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "buffers", _frozen_mapping(self.buffers))
        object.__setattr__(
            self, "optional_buffers", _frozen_mapping(self.optional_buffers),
        )
        object.__setattr__(self, "compile_time", _frozen_mapping(self.compile_time))
        object.__setattr__(
            self, "runtime_scalars", _frozen_mapping(self.runtime_scalars),
        )
        object.__setattr__(
            self, "optional_values", _frozen_mapping(self.optional_values),
        )
        object.__setattr__(
            self, "block_sizes", _frozen_mapping(self.block_sizes),
        )

    def _metadata(
        self, compile_time: Mapping[str, ScalarKind],
    ) -> KernelMetadata:
        return KernelMetadata(
            name=self.name,
            parameters=self.parameters,
            size_key=self.size_key,
            buffers=self.buffers,
            optional_buffers=self.optional_buffers,
            compile_time=compile_time,
            runtime_scalars=self.runtime_scalars,
            optional_values=self.optional_values,
            block_sizes=self.block_sizes,
        )

    def metadata_for_lowering(
        self, lowering: BackendLoweringSpec,
    ) -> KernelMetadata:
        """Project native metadata entirely from this Spec and its lowering."""

        if not isinstance(lowering, BackendLoweringSpec):
            raise TypeError("metadata lowering must be BackendLoweringSpec")
        return self._metadata(lowering.compile_time_for(self))

    @cached_property
    def metadata(self) -> KernelMetadata:
        return self._metadata(self.compile_time)

    def launch_extent(self, arguments: Mapping[str, Any]) -> int:
        """Return the exact flattened launch extent for this ABI.

        Launch geometry is part of the logical kernel contract, not a backend
        adapter convenience.  In particular, accepting ``bool`` or truncating
        a float with ``int()`` would let different backends launch different
        numbers of threads for the same public call.
        """

        return validate_launch_extent(self.name, self.size_key, arguments)

    def execution_size_key(
        self, additional_axes: tuple[str, ...] = (),
    ) -> str | tuple[str, ...]:
        """Return a validated backend execution layout over canonical axes."""

        if type(additional_axes) is not tuple or any(
            type(axis) is not str or not axis.isidentifier()
            for axis in additional_axes
        ):
            raise TypeError("additional execution axes must be a tuple of identifiers")
        if len(additional_axes) != len(set(additional_axes)):
            raise ValueError("additional execution axes must be unique")
        base = (
            (self.size_key,) if isinstance(self.size_key, str)
            else tuple(self.size_key)
        )
        overlap = set(base).intersection(additional_axes)
        if overlap:
            raise ValueError(
                f"{self.name}: additional execution axes repeat logical size "
                f"keys {sorted(overlap)}"
            )
        invalid = {
            axis: self.runtime_scalars.get(axis)
            for axis in additional_axes
            if self.runtime_scalars.get(axis) != "index"
        }
        if invalid:
            raise TypeError(
                f"{self.name}: additional execution axes must be canonical "
                f"runtime index scalars: {invalid}"
            )
        return (*base, *additional_axes) if additional_axes else self.size_key

    def validate_runtime_scalars(self, arguments: Mapping[str, Any]) -> None:
        """Validate semantic host values before any backend representation."""

        for name, kind in self.runtime_scalars.items():
            value = arguments[name]
            if not _host_scalar_is_valid(value, kind):
                raise TypeError(
                    f"{self.name}.{name} must be an exact finite {kind} host "
                    f"scalar, got {value!r} ({type(value).__name__})"
                )

    def validate_compile_time(self, arguments: Mapping[str, Any]) -> None:
        """Validate canonical specialization values without backend coercion."""

        for name, kind in self.compile_time.items():
            value = arguments[name]
            if not _host_scalar_is_valid(value, kind):
                raise TypeError(
                    f"{self.name}.{name} must be an exact finite {kind} "
                    f"host scalar, got {value!r} ({type(value).__name__})"
                )

    def validate_host_arguments(self, arguments: Mapping[str, Any]) -> None:
        """Apply every backend-independent host-side ABI invariant."""

        self.validate_compile_time(arguments)
        self.launch_extent(arguments)
        self.validate_runtime_scalars(arguments)
        self.validate_optional(arguments)

    def validate_optional(self, arguments: Mapping[str, Any]) -> None:
        """Require one exact representation for every disabled feature."""

        for buffer, feature in self.optional_buffers.items():
            if feature is None:
                continue
            enabled = arguments[feature]
            present = arguments[buffer] is not None
            if enabled != present:
                expected = "a tensor" if enabled else "None"
                raise ValueError(f"{self.name}.{buffer} must be {expected}")
        for argument, (feature, disabled) in self.optional_values.items():
            if not arguments[feature] and arguments[argument] != disabled:
                raise ValueError(
                    f"{self.name}.{argument} must equal disabled sentinel "
                    f"{disabled!r} when {feature}=False"
                )

    def project(
        self, *, omit: tuple[str, ...] = (), name: str | None = None,
        size_key: str | tuple[str, ...] | None = None,
    ) -> "KernelSpec":
        """Declare the exact private ABI consumed by one native variant."""
        omitted = frozenset(omit)
        unknown = omitted.difference(self.parameters)
        if unknown:
            raise ValueError(
                f"{self.name}: projection omits unknown parameters "
                f"{sorted(unknown)}"
            )
        projected_size = self.size_key if size_key is None else size_key
        size_names = (
            (projected_size,) if isinstance(projected_size, str)
            else projected_size
        )
        if omitted.intersection(size_names):
            raise ValueError(
                f"{self.name}: projection cannot omit size key(s) "
                f"{sorted(omitted.intersection(size_names))}"
            )
        orphaned = {
            parameter for parameter, feature in (
                *self.optional_buffers.items(), *self.optional_values.items(),
            )
            if feature is not None and (
                (feature[0] if isinstance(feature, tuple) else feature)
                in omitted and parameter not in omitted
            )
        }
        if orphaned:
            raise ValueError(
                f"{self.name}: projection omits feature while retaining "
                f"optional argument(s) {sorted(orphaned)}"
            )
        return KernelSpec(
            name=self.name if name is None else name,
            parameters=tuple(
                parameter for parameter in self.parameters
                if parameter not in omitted
            ),
            size_key=projected_size,
            buffers={
                parameter: access for parameter, access in self.buffers.items()
                if parameter not in omitted
            },
            optional_buffers={
                parameter: feature
                for parameter, feature in self.optional_buffers.items()
                if parameter not in omitted and feature not in omitted
            },
            compile_time={
                parameter: kind
                for parameter, kind in self.compile_time.items()
                if parameter not in omitted
            },
            runtime_scalars={
                parameter: kind
                for parameter, kind in self.runtime_scalars.items()
                if parameter not in omitted
            },
            optional_values={
                parameter: value
                for parameter, value in self.optional_values.items()
                if parameter not in omitted and value[0] not in omitted
            },
            block_sizes=self.block_sizes,
        )

    def validate(self, backend: str, actual: KernelMetadata) -> None:
        """Fail if a public kernel ABI differs from this exact specification."""

        differences: list[str] = []
        if actual.name != self.name:
            differences.append(f"name={actual.name!r}, expected {self.name!r}")
        if tuple(actual.parameters) != self.parameters:
            differences.append(
                f"parameters={tuple(actual.parameters)!r}, expected {self.parameters!r}"
            )
        if actual.size_key != self.size_key:
            differences.append(
                f"size_key={actual.size_key!r}, expected {self.size_key!r}"
            )
        for label, expected, observed in (
            ("buffers", self.buffers, actual.buffers),
            ("optional_buffers", self.optional_buffers, actual.optional_buffers),
            ("compile_time", self.compile_time, actual.compile_time),
            ("runtime_scalars", self.runtime_scalars, actual.runtime_scalars),
            ("optional_values", self.optional_values, actual.optional_values),
            ("block_sizes", self.block_sizes, actual.block_sizes),
        ):
            if dict(observed) != dict(expected):
                differences.append(
                    f"{label}={dict(observed)!r}, expected {dict(expected)!r}"
                )
        if differences:
            detail = "; ".join(differences)
            raise TypeError(
                f"{self.name}: {backend} implementation violates KernelSpec: {detail}"
            )

    def validate_native(
        self, backend: str, actual: KernelMetadata,
        lowering: BackendLoweringSpec,
    ) -> None:
        """Validate the private native launch surface behind the public ABI.

        Native launchers may use a different positional order and conservative
        access modes.  Those details never escape the specialized launch.  They
        must nevertheless consume exactly the canonical names, use the same
        launch extent, and preserve optional-buffer/value semantics.
        """

        differences: list[str] = []
        if actual.name != self.name:
            differences.append(f"name={actual.name!r}, expected={self.name!r}")
        parameters_match = (
            tuple(actual.parameters) == self.parameters
            if lowering.parameter_order == "canonical"
            else set(actual.parameters) == set(self.parameters)
        )
        if not parameters_match:
            differences.append(
                f"parameters={tuple(actual.parameters)!r}, "
                f"lowering={lowering.parameter_order}, expected={self.parameters!r}"
            )
        if actual.size_key != self.size_key:
            differences.append(
                f"size_key={actual.size_key!r}, expected={self.size_key!r}"
            )
        if lowering.buffer_access == "exact":
            buffers_match = dict(actual.buffers) == dict(self.buffers)
        else:
            buffers_match = (
                set(actual.buffers) == set(self.buffers)
                and all(
                    actual.buffers[name] in {access, "read_write"}
                    for name, access in self.buffers.items()
                )
            )
        if not buffers_match:
            differences.append(
                f"buffers={dict(actual.buffers)!r}, lowering={lowering.buffer_access}, "
                f"expected={dict(self.buffers)!r}"
            )
        for label, expected, observed in (
            ("optional_buffers", self.optional_buffers, actual.optional_buffers),
            ("optional_values", self.optional_values, actual.optional_values),
            ("runtime_scalars", self.runtime_scalars, actual.runtime_scalars),
            ("block_sizes", self.block_sizes, actual.block_sizes),
        ):
            if dict(observed) != dict(expected):
                differences.append(
                    f"{label}={dict(observed)!r}, expected={dict(expected)!r}"
                )
        expected_constants = lowering.compile_time_for(self)
        if dict(actual.compile_time) != expected_constants:
            differences.append(
                f"compile_time={dict(actual.compile_time)!r}, "
                f"lowering requires={expected_constants!r}"
            )
        if differences:
            raise TypeError(
                f"{self.name}: {backend} native launch violates KernelSpec: "
                f"{'; '.join(differences)}"
            )

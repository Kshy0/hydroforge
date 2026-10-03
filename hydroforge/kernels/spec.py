"""Backend-neutral kernel declarations.

A :class:`KernelSpec` declares every parameter of one logical kernel exactly
once, in its canonical order: a buffer access mode, a runtime scalar kind, a
compile-time value (with its type) or a step field.  Backend implementations
only state how their native code consumes this ABI.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import Field, PrivateAttr, model_validator

from hydroforge.contracts.step_fields import StepField
from hydroforge.core.naming import DottedPath, Identifier
from hydroforge.core.validation import FrozenMapping, HydroForgeModel, frozen_dict
from hydroforge.platform.backend import backend_named

AccessMode = Literal[
    "read",
    "write",
    "read_write",
    "atomic_write",
    "atomic_add",
    "atomic_min",
    "atomic_max",
]
RuntimeKind = Literal[
    "bool", "int32", "uint32", "index", "float32", "float64", "precision"
]
ConstantKind = Literal["bool", "int32", "float32", "float64", "precision"]
Precision = Literal["float32", "float64"]

_ACCESS_MODES = frozenset(AccessMode.__args__)
_EXTENT_KINDS = frozenset(("int32", "uint32", "index"))


class Constant(HydroForgeModel):
    """A compile-time scalar bound by name from the model namespace."""

    kind: Literal["constant"] = "constant"
    dtype: ConstantKind


class ModuleEnabled(HydroForgeModel):
    """Whether one optional model module is open."""

    kind: Literal["module_enabled"] = "module_enabled"
    module: Identifier
    dtype: Literal["bool"] = "bool"


class ModuleFlag(HydroForgeModel):
    """One boolean field of a constructed module; false when it is closed."""

    kind: Literal["module_flag"] = "module_flag"
    module: Identifier
    field: Identifier
    dtype: Literal["bool"] = "bool"


class OutputRequested(HydroForgeModel):
    """Whether statistics output observes one declared tensor field."""

    kind: Literal["output_requested"] = "output_requested"
    module: Identifier
    field: Identifier
    dtype: Literal["bool"] = "bool"


class ConfigValue(HydroForgeModel):
    """A compile-time scalar read from the model's frozen options."""

    kind: Literal["config_value"] = "config_value"
    path: DottedPath
    dtype: ConstantKind


class OptionCode(HydroForgeModel):
    """The stable integer code of one declared model option."""

    kind: Literal["option_code"] = "option_code"
    path: DottedPath
    dtype: Literal["int32"] = "int32"


class LiteralValue(HydroForgeModel):
    """A fixed compile-time scalar owned by one kernel variant."""

    kind: Literal["literal_value"] = "literal_value"
    value: bool | int | float
    dtype: ConstantKind

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if not host_scalar_is_valid(self.value, self.dtype):
            raise ValueError(
                f"literal value must be an {expected_host_scalar(self.dtype)}, "
                f"got {self.value!r}"
            )
        return self


CompileTimeSource: TypeAlias = (
    ModuleEnabled
    | ModuleFlag
    | OutputRequested
    | ConfigValue
    | OptionCode
    | LiteralValue
)
Parameter: TypeAlias = (
    AccessMode | RuntimeKind | Constant | CompileTimeSource | StepField
)


def constant(dtype: ConstantKind) -> Constant:
    """Declare a compile-time scalar the binder resolves by parameter name."""

    return Constant(dtype=dtype)


def module_enabled(module: str) -> ModuleEnabled:
    return ModuleEnabled(module=module)


def module_flag(module: str, field: str) -> ModuleFlag:
    return ModuleFlag(module=module, field=field)


def output_requested(module: str, field: str) -> OutputRequested:
    """Bind a kernel feature to the model's frozen statistics output plan."""

    return OutputRequested(module=module, field=field)


def config_value(path: str, dtype: ConstantKind) -> ConfigValue:
    return ConfigValue(path=path, dtype=dtype)


def option_code(path: str) -> OptionCode:
    return OptionCode(path=path)


def literal_value(
    value: bool | int | float, dtype: ConstantKind | None = None
) -> LiteralValue:
    """Bind a compile-time parameter to one variant-local scalar.

    ``bool`` and ``int`` values default to ``bool`` and ``int32``; a ``float``
    value names its kind explicitly.
    """

    if dtype is None:
        if type(value) is bool:
            dtype = "bool"
        elif type(value) is int:
            dtype = "int32"
        else:
            raise TypeError("literal_value() of a float requires its dtype")
    return LiteralValue(value=value, dtype=dtype)


class BufferAccessSemantics(HydroForgeModel):
    """Dependency and native-storage meaning of one access mode."""

    reads: bool
    writes: bool
    atomic: bool
    dependency: Literal["read", "write", "read_write"]


_BUFFER_ACCESS_SEMANTICS = MappingProxyType(
    {
        access: BufferAccessSemantics(
            reads=reads, writes=writes, atomic=atomic, dependency=dependency
        )
        for access, reads, writes, atomic, dependency in (
            ("read", True, False, False, "read"),
            ("write", False, True, False, "write"),
            ("read_write", True, True, False, "read_write"),
            ("atomic_write", False, True, True, "write"),
            ("atomic_add", True, True, True, "read_write"),
            ("atomic_min", True, True, True, "read_write"),
            ("atomic_max", True, True, True, "read_write"),
        )
    }
)


def buffer_access_semantics(access: str) -> BufferAccessSemantics:
    """Return the one defined meaning of a buffer access mode."""

    try:
        return _BUFFER_ACCESS_SEMANTICS[access]
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid buffer access mode {access!r}") from error


def host_scalar_is_valid(value: Any, kind: str) -> bool:
    """Define host scalar semantics once, without coercion.

    An unresolved ``precision`` value must hold under either resolution, so it
    has to be representable in float32.
    """

    if kind == "bool":
        return type(value) is bool
    if kind == "int32":
        return type(value) is int and -(2**31) <= value < 2**31
    if kind == "uint32":
        return type(value) is int and 0 <= value < 2**32
    if kind == "index":
        return type(value) is int and -(2**63) <= value < 2**63
    if kind in {"float32", "precision"}:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or abs(value) > 3.4028234663852886e38
        ):
            return False
        encoded = struct.unpack("=f", struct.pack("=f", value))[0]
        return value == 0.0 or encoded != 0.0
    if kind == "float64":
        return type(value) is float and math.isfinite(value)
    raise RuntimeError(f"unknown host scalar kind {kind!r}")


def expected_host_scalar(kind: str) -> str:
    """Describe the host scalar :func:`host_scalar_is_valid` accepts."""

    if kind == "precision":
        return "exact finite float32-representable precision host scalar"
    return f"exact finite {kind} host scalar"


def launch_extent(name: str, keys: tuple[str, ...], values: Mapping[str, Any]) -> int:
    """Validate and flatten one launch geometry without coercion."""

    extent = 1
    for key in keys:
        value = values[key]
        if type(value) is not int:
            raise TypeError(
                f"{name}.{key} launch extent must be an exact host int, got "
                f"{type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"{name}.{key} launch extent must be non-negative")
        extent *= value
        if extent >= 2**63:
            raise OverflowError(f"{name} launch extent exceeds signed int64 range")
    return extent


def _size_keys(size: str | tuple[str, ...]) -> tuple[str, ...]:
    return (size,) if isinstance(size, str) else tuple(size)


def _resolved(declaration: Any, precision: str) -> Any:
    """``declaration`` with its ``precision`` kind replaced by ``precision``."""

    if declaration == "precision":
        return precision
    if isinstance(declaration, (Constant, ConfigValue, LiteralValue)) and (
        declaration.dtype == "precision"
    ):
        return type(declaration).model_construct(
            **{**dict(declaration), "dtype": precision}
        )
    return declaration


@dataclass(frozen=True, slots=True)
class _Views:
    """The per-kind views of one spec's ordered parameter declarations."""

    parameter_names: tuple[str, ...]
    size_keys: tuple[str, ...]
    buffers: Mapping[str, str]
    step_fields: Mapping[str, StepField]
    runtime_scalars: Mapping[str, str]
    compile_time: Mapping[str, str]
    compile_time_sources: Mapping[str, CompileTimeSource]
    uses_precision: bool

    @classmethod
    def of(cls, spec: KernelSpec) -> _Views:
        buffers, step_fields, runtime, compile_time, sources = {}, {}, {}, {}, {}
        for name, declaration in spec.parameters.items():
            if isinstance(declaration, StepField):
                buffers[name] = "read"
                step_fields[name] = declaration
            elif isinstance(declaration, str):
                (buffers if declaration in _ACCESS_MODES else runtime)[name] = (
                    declaration
                )
            else:
                compile_time[name] = declaration.dtype
                if not isinstance(declaration, Constant):
                    sources[name] = declaration
        return cls(
            tuple(spec.parameters),
            _size_keys(spec.size),
            MappingProxyType(buffers),
            MappingProxyType(step_fields),
            MappingProxyType(runtime),
            MappingProxyType(compile_time),
            MappingProxyType(sources),
            "precision" in runtime.values() or "precision" in compile_time.values(),
        )


class KernelWorkspace(HydroForgeModel):
    """Model-owned scratch shared by kernel specs with the same ``key``.

    Dimensions and topology sources bind to unique model/module fields;
    ``ensemble_size`` is the local member count (one without an ensemble).
    CSR offsets omit the terminal entry; CSR order is a stable source ordering.
    Workspaces are built once per execution plan and rebuilt on invalidation.
    They are neither physical state nor checkpoint/output fields.
    """

    key: DottedPath
    dtype: Literal["precision", "float32", "float64", "int32", "int64", "bool"]
    shape: tuple[Identifier | Annotated[int, Field(ge=0)], ...] = Field(min_length=1)
    initialize: Literal["zeros", "csr_offsets", "csr_order"] = "zeros"
    source: Identifier | None = None
    target_count: Identifier | None = None
    module: Identifier | None = None
    emulated_only: bool = False

    @model_validator(mode="after")
    def _validate_initialization(self) -> Self:
        if self.initialize == "zeros":
            if self.source is not None or self.target_count is not None:
                raise ValueError("zero workspace must not declare a topology source")
        elif (
            self.source is None
            or self.target_count is None
            or self.dtype != "int32"
            or len(self.shape) != 1
        ):
            raise ValueError(
                "CSR workspace requires source, target_count and a one-dimensional int32 shape"
            )
        return self


class KernelSpec(HydroForgeModel):
    """The explicit ABI of one logical kernel.

    ``parameters`` maps every parameter, in canonical order, to its
    declaration: an access mode for a buffer, a runtime scalar kind, a
    compile-time value (:func:`constant` or a value source) or a
    :func:`~hydroforge.contracts.step_field` bound buffer.  ``size`` names the
    integer scalars whose product is the launch extent.  ``optional`` maps a
    buffer to the bool compile-time feature that enables it (``None``: present
    when its field exists); ``optional_values`` maps a scalar to its feature
    and the value it takes while the feature is off.  ``block_sizes`` fixes a
    per-backend launch width. Implementations must match the selected spec;
    a registry may explicitly declare backend-specific workspace ABIs.
    ``workspace`` binds ordinary buffer parameters to model-owned scratch,
    shared by declaration key across kernels. Its dtype is explicit even when
    a conditional workspace is disabled; access modes still govern dependencies.
    """

    name: Identifier
    size: Identifier | tuple[Identifier, ...]
    parameters: FrozenMapping[Identifier, Parameter]
    optional: FrozenMapping[Identifier, Identifier | None] = Field(default_factory=dict)
    optional_values: FrozenMapping[Identifier, tuple[Identifier, Any]] = Field(
        default_factory=dict
    )
    block_sizes: FrozenMapping[
        Literal["cuda", "triton", "metal"], Annotated[int, Field(ge=1, le=1024)]
    ] = Field(default_factory=dict)
    workspace: FrozenMapping[Identifier, KernelWorkspace] = Field(default_factory=dict)
    _precision_parameters: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _derived: _Views | None = PrivateAttr(default=None)

    @property
    def _views(self) -> _Views:
        views = self._derived
        if views is None:
            views = self._derived = _Views.of(self)
        return views

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._views.parameter_names

    @property
    def size_keys(self) -> tuple[str, ...]:
        return self._views.size_keys

    @property
    def buffers(self) -> Mapping[str, str]:
        """Access mode of every buffer; step-field buffers are read-only."""

        return self._views.buffers

    @property
    def step_fields(self) -> Mapping[str, StepField]:
        return self._views.step_fields

    @property
    def runtime_scalars(self) -> Mapping[str, str]:
        return self._views.runtime_scalars

    @property
    def compile_time(self) -> Mapping[str, str]:
        """Scalar kind of every compile-time parameter."""

        return self._views.compile_time

    @property
    def compile_time_sources(self) -> Mapping[str, CompileTimeSource]:
        """Compile-time parameters bound to an explicit source."""

        return self._views.compile_time_sources

    @property
    def uses_precision(self) -> bool:
        return self._views.uses_precision

    @property
    def precision_parameters(self) -> frozenset[str]:
        """Scalars a :meth:`resolved` spec declared with the model precision."""

        return self._precision_parameters

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        parameters = self.parameters
        for name, declaration in self.workspace.items():
            if name not in self.buffers or name in self.step_fields:
                raise ValueError(
                    f"{self.name}: workspace {name!r} must name an ordinary buffer"
                )
            if (
                declaration.emulated_only or declaration.module is not None
            ) and name not in self.optional:
                raise ValueError(
                    f"{self.name}: conditional workspace {name!r} must be optional"
                )
        if "BLOCK_SIZE" in parameters:
            raise ValueError(
                f"{self.name}: BLOCK_SIZE is compiler-owned and cannot be a "
                "parameter; declare per-backend widths in block_sizes"
            )
        for backend, block_size in self.block_sizes.items():
            try:
                backend_named(backend).block.validate(block_size, backend=backend)
            except ValueError as error:
                raise ValueError(f"{self.name}: block_sizes: {error}") from error
        self._validate_size(self.size_keys)
        compile_time = self.compile_time
        for name, kind in compile_time.items():
            if kind != "bool":
                continue
            if name.startswith("HAS_"):
                if name not in self.compile_time_sources:
                    raise ValueError(
                        f"{self.name}: capability flag {name!r} requires an "
                        "explicit compile-time source"
                    )
            elif name.isupper():
                raise ValueError(
                    f"{self.name}: uppercase capability flag {name!r} must use "
                    "the HAS_* spelling"
                )
        for buffer, feature in self.optional.items():
            if buffer not in self.buffers:
                raise ValueError(
                    f"{self.name}: optional {buffer!r} is not a buffer parameter"
                )
            if buffer in self.step_fields:
                raise ValueError(
                    f"{self.name}: step field {buffer!r} cannot be optional"
                )
            if feature is not None and compile_time.get(feature) != "bool":
                raise ValueError(
                    f"{self.name}: optional buffer {buffer!r} requires a bool "
                    f"compile-time feature, got {feature!r}"
                )
        for argument, (feature, disabled) in self.optional_values.items():
            kind = self.runtime_scalars.get(argument, compile_time.get(argument))
            if kind is None:
                raise ValueError(
                    f"{self.name}: optional value {argument!r} must be a scalar "
                    "parameter"
                )
            if compile_time.get(feature) != "bool":
                raise ValueError(
                    f"{self.name}: optional value {argument!r} requires a bool "
                    f"compile-time feature, got {feature!r}"
                )
            if not host_scalar_is_valid(disabled, kind):
                raise ValueError(
                    f"{self.name}: optional value {argument!r} disabled "
                    f"sentinel must be an {expected_host_scalar(kind)}, "
                    f"got {disabled!r} ({type(disabled).__name__})"
                )
        return self

    def _validate_size(self, keys: tuple[str, ...]) -> None:
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f"{self.name}: size must name one or more unique scalars")
        invalid = [
            key for key in keys if self.runtime_scalars.get(key) not in _EXTENT_KINDS
        ]
        if invalid:
            raise ValueError(
                f"{self.name}: launch extent parameter(s) {invalid} must be "
                "'int32', 'uint32' or 'index' runtime scalars"
            )

    def resolved(self, precision: Precision | None) -> KernelSpec:
        """This spec with its ``precision`` kinds replaced by ``precision``."""

        if not self.uses_precision:
            return self
        if precision not in {"float32", "float64"}:
            raise ValueError(
                f"{self.name}: precision-dependent KernelSpec requires "
                "precision='float32' or 'float64'"
            )
        resolved = self._derive(
            parameters={
                name: _resolved(declaration, precision)
                for name, declaration in self.parameters.items()
            }
        )
        resolved._precision_parameters = frozenset(
            name
            for name, kind in (
                *self.runtime_scalars.items(),
                *self.compile_time.items(),
            )
            if kind == "precision"
        )
        return resolved

    def project(
        self,
        *,
        omit: tuple[str, ...] = (),
        name: str | None = None,
        size: str | tuple[str, ...] | None = None,
    ) -> KernelSpec:
        """The sub-ABI without ``omit``, optionally renamed or resized."""

        omitted = frozenset(omit)
        if type(omit) is not tuple or len(omit) != len(omitted):
            raise ValueError(f"{self.name}: projection omits a tuple of unique names")
        unknown = omitted.difference(self.parameters)
        if unknown:
            raise ValueError(
                f"{self.name}: projection omits unknown parameters {sorted(unknown)}"
            )
        size = self.size if size is None else size
        if omitted.intersection(_size_keys(size)):
            raise ValueError(
                f"{self.name}: projection cannot omit size key(s) "
                f"{sorted(omitted.intersection(_size_keys(size)))}"
            )
        orphaned = sorted(
            argument
            for argument, feature in (
                *self.optional.items(),
                *(
                    (argument, value[0])
                    for argument, value in self.optional_values.items()
                ),
            )
            if feature in omitted and argument not in omitted
        )
        if orphaned:
            raise ValueError(
                f"{self.name}: projection omits a feature while retaining "
                f"optional argument(s) {orphaned}"
            )
        projected = KernelSpec(
            name=self.name if name is None else name,
            size=size,
            parameters={
                parameter: declaration
                for parameter, declaration in self.parameters.items()
                if parameter not in omitted
            },
            optional={
                parameter: feature
                for parameter, feature in self.optional.items()
                if parameter not in omitted
            },
            optional_values={
                parameter: value
                for parameter, value in self.optional_values.items()
                if parameter not in omitted
            },
            workspace={
                key: value
                for key, value in self.workspace.items()
                if key not in omitted
            },
            block_sizes=self.block_sizes,
        )
        projected._precision_parameters = self._precision_parameters - omitted
        return projected

    def _derive(self, **changes: Any) -> KernelSpec:
        """Build a spec from these validated fields and checked changes."""

        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(
            {
                name: frozen_dict(value) if isinstance(value, Mapping) else value
                for name, value in changes.items()
            }
        )
        return KernelSpec.model_construct(**values)

    def launch_extent(self, values: Mapping[str, Any]) -> int:
        """The exact flattened launch extent of one call."""

        return launch_extent(self.name, self.size_keys, values)

    def validate_scalars(
        self, kinds: Mapping[str, str], values: Mapping[str, Any]
    ) -> None:
        for name, kind in kinds.items():
            value = values[name]
            if not host_scalar_is_valid(value, kind):
                raise TypeError(
                    f"{self.name}.{name} must be an {expected_host_scalar(kind)}, "
                    f"got {value!r} ({type(value).__name__})"
                )

    def validate_host_values(self, values: Mapping[str, Any]) -> None:
        """Apply every backend-independent host-side ABI invariant."""

        self.validate_scalars(self.compile_time, values)
        self.launch_extent(values)
        self.validate_scalars(self.runtime_scalars, values)
        for buffer, feature in self.optional.items():
            if feature is None:
                continue
            enabled = values[feature]
            if enabled != (values[buffer] is not None):
                expected = "a tensor" if enabled else "None"
                raise ValueError(f"{self.name}.{buffer} must be {expected}")
        for argument, (feature, disabled) in self.optional_values.items():
            if not values[feature] and values[argument] != disabled:
                raise ValueError(
                    f"{self.name}.{argument} must equal disabled sentinel "
                    f"{disabled!r} when {feature}=False"
                )


__all__ = [
    "KernelSpec",
    "KernelWorkspace",
    "buffer_access_semantics",
    "config_value",
    "constant",
    "literal_value",
    "module_enabled",
    "module_flag",
    "option_code",
    "output_requested",
]

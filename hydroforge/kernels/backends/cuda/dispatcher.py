"""Declarative CUDA extension namespace and canonical launch adapter."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import math
from typing import Any, Callable, Dict, Mapping, Self

from pydantic import Field, PrivateAttr, ValidationInfo, model_validator

from hydroforge.kernels.backends.cuda.spec import (
    CudaExtensionSpec, _CompiledCudaExtension, cuda_declarations,
    cuda_function_signature, cuda_narrowed_index_parameters,
)
from hydroforge.contracts.kernels import (
    BackendLoweringSpec, BufferDTypeABI, KernelSpec, _host_scalar_is_valid,
)
from hydroforge.contracts.validation import HydroForgeModel, _immutable_dict
from hydroforge.kernels.context import (
    active_kernel_spec, registry_factory,
)


@dataclass(frozen=True, slots=True)
class _CudaTensorVector:
    """The sole inferred private CUDA argument form."""

    target: str
    sources: tuple[str, ...]

    def resolve(self, values: Mapping[str, Any]) -> list[Any]:
        return [values[name] for name in self.sources]


CudaProjectionValue = bool | int | float | None


class CudaNativeProjection(HydroForgeModel):
    """Semantic preconditions for canonical values absent from a launcher."""

    fixed: Mapping[str, CudaProjectionValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_projection(self) -> Self:
        if not isinstance(self.fixed, Mapping):
            raise ValueError("CUDA native projection fixed values must be a mapping")
        fixed = dict(self.fixed)
        invalid = [name for name in fixed if not name.isidentifier()]
        if invalid:
            raise ValueError(
                "CUDA native projection fixed names must be Python "
                f"identifiers: {invalid}"
            )
        invalid_values = {
            name: type(value).__name__
            for name, value in fixed.items()
            if type(value) not in {bool, int, float, type(None)}
        }
        if invalid_values:
            raise ValueError(
                "CUDA native projection fixed values must be exact immutable "
                f"host scalars or None: {invalid_values}"
            )
        nonfinite = [
            name for name, value in fixed.items()
            if type(value) is float and not math.isfinite(value)
        ]
        if nonfinite:
            raise ValueError(
                "CUDA native projection fixed floats must be finite: "
                f"{nonfinite}"
            )
        object.__setattr__(self, "fixed", _immutable_dict(fixed))
        return self

    def _validate(self, values: Mapping[str, Any], *, kernel: str) -> None:
        mismatched = {
            name: (values[name], expected)
            for name, expected in self.fixed.items()
            if type(values[name]) is not type(expected) or values[name] != expected
        }
        if mismatched:
            detail = ", ".join(
                f"{name}={observed!r}, required={expected!r}"
                for name, (observed, expected) in sorted(mismatched.items())
            )
            raise ValueError(
                f"{kernel}: CUDA native projection precondition failed: {detail}"
            )


class CudaRoute(HydroForgeModel):
    """One declarative extension launcher owned by a CUDA extension group."""

    extension: str
    launch: str
    spec: KernelSpec
    projection: CudaNativeProjection | None = None

    @model_validator(mode="after")
    def _validate_route(self) -> Self:
        for field in ("extension", "launch"):
            value = getattr(self, field)
            if type(value) is not str or not value.isidentifier():
                raise ValueError(
                    f"CUDA route {field} must be a Python/C++ identifier"
                )
        return self

    @property
    def _key(self) -> tuple[str, str]:
        return self.extension, self.launch


@dataclass(frozen=True, slots=True)
class _CompiledCudaRoute:
    """Complete construction-time CUDA ABI consumed by trusted dispatch."""

    extension: str
    launch: str
    spec: KernelSpec
    native_signature: tuple[tuple[str, str], ...]
    launch_args: tuple[str, ...]
    tensor_vector: _CudaTensorVector | None
    projection: CudaNativeProjection
    omitted: frozenset[str]
    resolved_specs: tuple[KernelSpec, ...]


def _normalized_native_type(native_type: str) -> str:
    return " ".join(
        token for token in native_type.replace("&", "").split()
        if token != "const"
    )


def _native_kind(native_type: str) -> str:
    normalized = _normalized_native_type(native_type)
    if normalized in {
        "at::Tensor", "std::optional<at::Tensor>",
        "c10::optional<at::Tensor>",
    }:
        return "buffer"
    if normalized == "bool":
        return "bool"
    if normalized in {"int", "int32_t"}:
        return "int32"
    if normalized in {"uint32_t", "std::uint32_t", "unsigned int"}:
        return "uint32"
    if normalized in {"long", "int64_t"}:
        return "index"
    if normalized == "float":
        return "float32"
    if normalized == "double":
        return "float64"
    raise ValueError(
        f"unsupported CUDA launcher parameter type {native_type!r}"
    )


def _native_buffer_optional(native_type: str) -> bool:
    normalized = _normalized_native_type(native_type)
    if normalized == "at::Tensor":
        return False
    if normalized in {
        "std::optional<at::Tensor>", "c10::optional<at::Tensor>",
    }:
        return True
    raise ValueError(f"unsupported CUDA tensor launcher type {native_type!r}")


def _validate_projection_values(
    spec: KernelSpec,
    projection: CudaNativeProjection,
    omitted: set[str],
) -> None:
    """Validate every omitted canonical value against its declared ABI kind."""

    for name in sorted(omitted):
        value = projection.fixed[name]
        if name in spec.buffers:
            if name not in spec.optional_buffers:
                raise ValueError(
                    f"{spec.name}: CUDA launcher omits required canonical "
                    f"buffer {name!r}"
                )
            if value is not None:
                raise ValueError(
                    f"{spec.name}: omitted optional CUDA buffer {name!r} "
                    "must be fixed to None"
                )
            continue

        kind = spec.runtime_scalars.get(name, spec.compile_time.get(name))
        validation_kind = "float32" if kind == "precision" else kind
        if validation_kind is None or not _host_scalar_is_valid(
            value, validation_kind,
        ):
            raise ValueError(
                f"{spec.name}: CUDA native projection value {name!r} must "
                f"be an exact finite {kind} host scalar, got {value!r} "
                f"({type(value).__name__})"
            )


def _compile_cuda_route(
    route: CudaRoute, source: str,
) -> _CompiledCudaRoute:
    """Parse and validate one route exactly once during group construction."""

    spec = route.spec
    native_signature = cuda_function_signature(source, route.launch)
    narrowed = cuda_narrowed_index_parameters(
        source,
        route.launch,
        tuple(
            name for name, kind in spec.runtime_scalars.items()
            if kind == "index"
        ),
    )
    if narrowed:
        raise ValueError(
            f"{spec.name}: CUDA launcher narrows canonical int64 index "
            f"parameter(s) to int32: {list(narrowed)}; declare an int32 "
            "runtime scalar when the device algorithm is truly 32-bit"
        )

    parameters = spec.parameters
    launch_args = tuple(name for name, _kind in native_signature)
    native_names = set(launch_args)
    canonical_names = set(parameters)
    vector_arguments = tuple(
        name for name, native_type in native_signature
        if _normalized_native_type(native_type) == "std::vector<at::Tensor>"
    )
    if len(vector_arguments) > 1:
        raise ValueError(
            f"{spec.name}: CUDA launcher has multiple private tensor vectors; "
            "their canonical partition cannot be inferred"
        )
    tensor_vector = None
    if vector_arguments:
        target = vector_arguments[0]
        if target in canonical_names:
            raise ValueError(
                f"{spec.name}: CUDA tensor vector {target!r} must be a private "
                "physical projection, not a canonical parameter"
            )
        sources = tuple(
            name for name in parameters
            if name in spec.buffers and name not in native_names
        )
        if not sources:
            raise ValueError(
                f"{spec.name}: CUDA tensor vector {target!r} has no canonical "
                "buffer sources to pack"
            )
        optional_sources = set(sources).intersection(spec.optional_buffers)
        if optional_sources:
            raise ValueError(
                f"{spec.name}: CUDA tensor vectors cannot pack optional "
                f"buffers: {sorted(optional_sources)}"
            )
        tensor_vector = _CudaTensorVector(target, sources)

    projection = route.projection or CudaNativeProjection()
    consumed_canonical = native_names.intersection(canonical_names) | {
        source_name for source_name in (
            () if tensor_vector is None else tensor_vector.sources
        )
    }
    omitted_canonical = canonical_names.difference(consumed_canonical)
    unknown_fixed = set(projection.fixed).difference(omitted_canonical)
    if unknown_fixed:
        raise ValueError(
            f"{spec.name}: CUDA native projection fixes values that are still "
            "consumed by the launcher/derived ABI or absent from KernelSpec: "
            f"{sorted(unknown_fixed)}"
        )
    # Mask members are supplied by source specialization, not launcher inputs.
    grouped_features = (
        set().union(*spec.compile_time_masks.values())
        if spec.compile_time_masks
        else set()
    )
    missing_fixed = omitted_canonical.difference(
        projection.fixed,
        grouped_features,
    )
    if missing_fixed:
        raise ValueError(
            f"{spec.name}: CUDA launcher omits canonical inputs "
            f"{sorted(missing_fixed)}; define every omitted value in "
            "CudaNativeProjection.fixed instead of inferring semantics from "
            "an absent native parameter"
        )
    _validate_projection_values(
        spec,
        projection,
        omitted_canonical.difference(grouped_features),
    )
    if "BLOCK_SIZE" not in native_names:
        raise ValueError(
            f"{spec.name}: CUDA launcher must expose compiler-owned "
            "BLOCK_SIZE explicitly"
        )
    unknown = set(launch_args).difference(
        parameters,
        (() if tensor_vector is None else {tensor_vector.target}),
        {"BLOCK_SIZE"},
    )
    if unknown:
        raise ValueError(
            "CUDA launch arguments are outside canonical ABI: "
            f"{sorted(unknown)}"
        )

    canonical_native_kinds = {
        **{name: "buffer" for name in spec.buffers},
        **{
            name: kind for name, kind in spec.runtime_scalars.items()
        },
        **{
            name: kind for name, kind in spec.compile_time.items()
        },
    }
    for name, native_type in native_signature:
        if tensor_vector is not None and name == tensor_vector.target:
            # The private vector target was already identified by its exact
            # native type and its canonical buffer sources were validated
            # above. It has no scalar/buffer kind in the canonical ABI.
            continue
        observed = _native_kind(native_type)
        if name == "BLOCK_SIZE":
            if observed != "index":
                raise ValueError(
                    f"{spec.name}: CUDA launcher BLOCK_SIZE uses "
                    f"{native_type!r} ({observed}), requires int64 index"
                )
            continue
        expected = canonical_native_kinds.get(name)
        if expected is None:
            continue
        compatible = expected == observed or (
            expected == "precision" and observed in {"float32", "float64"}
        )
        if not compatible:
            raise ValueError(
                f"{spec.name}: CUDA launcher parameter {name!r} uses "
                f"{native_type!r} ({observed}), KernelSpec requires "
                f"{expected}"
            )
        if expected == "buffer":
            expected_optional = name in spec.optional_buffers
            observed_optional = _native_buffer_optional(native_type)
            if observed_optional != expected_optional:
                required = "optional" if expected_optional else "required"
                native = "optional" if observed_optional else "required"
                raise ValueError(
                    f"{spec.name}: CUDA launcher buffer {name!r} is {native}, "
                    f"KernelSpec declares it {required}"
                )
    return _CompiledCudaRoute(
        extension=route.extension,
        launch=route.launch,
        spec=spec,
        native_signature=native_signature,
        launch_args=launch_args,
        tensor_vector=tensor_vector,
        projection=projection,
        omitted=frozenset(omitted_canonical),
        resolved_specs=(
            (
                spec._resolve_precision("float32"),
                spec._resolve_precision("float64"),
            )
            if spec._uses_precision else (spec,)
        ),
    )


_CUDA_FACTORY_CONTEXT = "hydroforge_cuda_extension_group"


class _CudaFactoryRequest(HydroForgeModel):
    extension: str
    launch: str

    _route: _CompiledCudaRoute = PrivateAttr()

    @model_validator(mode="after")
    def _resolve_route(self, info: ValidationInfo):
        group = (
            info.context.get(_CUDA_FACTORY_CONTEXT)
            if isinstance(info.context, Mapping) else None
        )
        if group is None:
            raise ValueError("CUDA factory request requires group context")
        try:
            self._route = group._route_index[(self.extension, self.launch)]
        except KeyError as error:
            raise ValueError(
                f"unknown CUDA route {self.extension!r}/{self.launch!r}"
            ) from error
        return self

    @property
    def route(self) -> _CompiledCudaRoute:
        return self._route


class _CudaDispatcherDeclaration(HydroForgeModel):
    """Bind a validated registry precision to one compiled route."""

    route: _CompiledCudaRoute
    spec: KernelSpec

    @model_validator(mode="after")
    def _validate_spec(self):
        if self.spec not in self.route.resolved_specs:
            raise ValueError(
                f"CUDA route {self.route.extension!r}/{self.route.launch!r} "
                f"declares KernelSpec {self.route.spec.name!r}, not "
                f"{self.spec.name!r}"
            )
        return self


class CudaExtensionGroup(HydroForgeModel):
    """Lazily build a named namespace of declarative CUDA extensions."""

    owner_module: str
    specs: Mapping[str, CudaExtensionSpec]
    routes: tuple[CudaRoute, ...]
    binary_prefix: str | None = None
    env_prefix: str = "HYDROFORGE"
    module_extensions: Mapping[
        str, set[str] | frozenset[str]
    ] = Field(default_factory=dict)

    _route_index: Mapping[tuple[str, str], _CompiledCudaRoute] = PrivateAttr(
        default_factory=dict,
    )
    _exports: Mapping[str, tuple[str, ...]] = PrivateAttr(default_factory=dict)
    _compiled_specs: Mapping[str, _CompiledCudaExtension] = PrivateAttr(
        default_factory=dict,
    )
    _loaded: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _variant_loaded: Dict[tuple[str, tuple[tuple[str, int], ...]], Any] = (
        PrivateAttr(default_factory=dict)
    )
    _precompiled: set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="after")
    def _validate_group(self) -> Self:
        if (
            type(self.owner_module) is not str or not self.owner_module
            or any(
                not part.isidentifier()
                for part in self.owner_module.split(".")
            )
        ):
            raise ValueError(
                "CUDA extension owner_module must be a dotted Python name"
            )
        if not isinstance(self.specs, Mapping) or not self.specs:
            raise ValueError("CUDA extension specs must be a non-empty mapping")
        invalid_names = [
            name for name in self.specs
            if type(name) is not str or not name.isidentifier()
        ]
        if invalid_names:
            raise ValueError(
                f"CUDA extension names must be identifiers: {invalid_names}"
            )
        invalid_specs = {
            name: type(spec).__name__
            for name, spec in self.specs.items()
            if not isinstance(spec, CudaExtensionSpec)
        }
        if invalid_specs:
            raise ValueError(
                f"CUDA extension catalog values must be CudaExtensionSpec: "
                f"{invalid_specs}"
            )
        resolved_prefix = (
            self.binary_prefix or self.owner_module.replace(".", "_")
        )
        if type(resolved_prefix) is not str or not resolved_prefix.isidentifier():
            raise ValueError("CUDA binary_prefix must be a Python identifier")
        if type(self.env_prefix) is not str or not self.env_prefix.isidentifier():
            raise ValueError("CUDA env_prefix must be a Python identifier")
        object.__setattr__(self, "binary_prefix", resolved_prefix)
        object.__setattr__(self, "specs", _immutable_dict(self.specs))
        demands = {
            module: frozenset(extensions)
            for module, extensions in self.module_extensions.items()
        }
        invalid_modules = [
            module for module in demands
            if type(module) is not str or not module.isidentifier()
        ]
        if invalid_modules:
            raise ValueError(
                "CUDA module demand names must be identifiers: "
                f"{invalid_modules}"
            )
        unknown_demands = {
            module: sorted(extensions.difference(self.specs))
            for module, extensions in demands.items()
            if extensions.difference(self.specs)
        }
        if unknown_demands:
            raise ValueError(
                "CUDA module demands reference unknown extensions: "
                f"{unknown_demands}"
            )
        object.__setattr__(
            self, "module_extensions", _immutable_dict(demands),
        )
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError("CUDA extension routes must be a non-empty tuple")
        route_index: dict[tuple[str, str], CudaRoute] = {}
        exports: dict[str, list[str]] = {name: [] for name in self.specs}
        for route in self.routes:
            if route.extension not in self.specs:
                raise ValueError(
                    f"CUDA route {route.extension!r}/{route.launch!r} "
                    "references an unknown extension"
                )
            if route._key in route_index:
                raise ValueError(
                    f"CUDA route {route.extension!r}/{route.launch!r} "
                    "is declared more than once"
                )
            route_index[route._key] = route
            exports[route.extension].append(route.launch)
        missing_routes = sorted(
            name for name, launches in exports.items() if not launches
        )
        if missing_routes:
            raise ValueError(
                "CUDA extension specs must each declare at least one route: "
                f"{missing_routes}"
            )
        immutable_exports = {
            name: tuple(launches) for name, launches in exports.items()
        }
        compiled_specs: dict[str, _CompiledCudaExtension] = {}
        compiled_routes: dict[tuple[str, str], _CompiledCudaRoute] = {}
        for name, spec in self.specs.items():
            source = spec._materialize_source()
            functions = immutable_exports[name]
            declarations = cuda_declarations(source, functions)
            compiled_specs[name] = _CompiledCudaExtension(
                source=source,
                functions=functions,
                declarations=declarations,
                cflags=spec.cflags,
                cpp_headers=spec.cpp_headers,
                include_paths=spec.include_paths,
                ldflags=spec.ldflags,
            )
        for key, route in route_index.items():
            try:
                compiled_routes[key] = _compile_cuda_route(
                    route, compiled_specs[route.extension].source,
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(str(error)) from error
        self._route_index = _immutable_dict(compiled_routes)
        self._exports = _immutable_dict(immutable_exports)
        self._compiled_specs = _immutable_dict(compiled_specs)
        return self

    def factory(
        self, extension: str, launch: str,
    ) -> Callable[[], "CudaDispatcher"]:
        """Return the registry factory for one already-declared route."""

        request = _CudaFactoryRequest.model_validate(
            {"extension": extension, "launch": launch},
            context={_CUDA_FACTORY_CONTEXT: self},
        )
        route = request.route

        @registry_factory
        def factory() -> CudaDispatcher:
            return self._dispatcher(route)

        return factory

    def _load(self, name: str) -> Any:
        if name in self._loaded:
            return self._loaded[name]
        spec = self._compiled_specs[name]
        from hydroforge.kernels.backends.cuda.build import load_inline_cu_module

        module = load_inline_cu_module(
            f"{self.binary_prefix}_{name}",
            cpp_sources="\n".join((*spec.cpp_headers, *spec.declarations)),
            cuda_sources=spec.source,
            functions=spec.functions, extra_cuda_cflags=spec.cflags,
            extra_include_paths=tuple(map(str, spec.include_paths)),
            extra_ldflags=spec.ldflags,
            env_prefix=self.env_prefix,
        )
        self._loaded[name] = module
        return module

    def _load_variant(
        self, name: str, masks: tuple[tuple[str, int], ...],
    ) -> Any:
        """Compile a CUDA source variant for one grouped-mask tuple."""

        key = (name, masks)
        cached = self._variant_loaded.get(key)
        if cached is not None:
            return cached
        if not masks:
            return self._load(name)
        spec = self._compiled_specs[name]
        prefix = "".join(
            f"#define HYDROFORGE_{mask} {value}u\n"
            for mask, value in masks
        )
        source = prefix + spec.source
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        from hydroforge.kernels.backends.cuda.build import load_inline_cu_module

        module = load_inline_cu_module(
            f"{self.binary_prefix}_{name}_mask_{digest}",
            cpp_sources="\n".join((*spec.cpp_headers, *cuda_declarations(
                source, spec.functions,
            ))),
            cuda_sources=source,
            functions=spec.functions,
            extra_cuda_cflags=spec.cflags,
            extra_include_paths=tuple(map(str, spec.include_paths)),
            extra_ldflags=spec.ldflags,
            env_prefix=self.env_prefix,
        )
        self._variant_loaded[key] = module
        return module

    def _ensure_precompiled(
        self, extensions: Any = None,
    ) -> Dict[str, Any]:
        """Build and load the requested subset of this extension catalog.

        Repeated calls are cumulative.  Omitting ``extensions`` preserves the
        public whole-catalog precompile behavior used by the CLI.
        """
        requested = (
            set(self.specs) if extensions is None else set(extensions)
        )
        pending = requested.difference(self._precompiled)
        if not pending:
            return {name: self._loaded[name] for name in requested}
        from hydroforge.kernels.backends.cuda.precompile import precompile_extension_specs

        effective = {
            name: spec for name, spec in self._compiled_specs.items()
            if name in pending
        }
        precompile_extension_specs(
            self.binary_prefix, effective, env_prefix=self.env_prefix,
        )
        for name in pending:
            self._load(name)
        self._precompiled.update(pending)
        return {name: self._loaded[name] for name in requested}

    def _ensure_precompiled_for_modules(
        self, opened_modules: Any,
    ) -> Dict[str, Any]:
        """Precompile the exact catalog subset required by model modules."""
        if not self.module_extensions:
            return self._ensure_precompiled()
        opened = set(opened_modules)
        required = set().union(*(
            self.module_extensions[module] for module in opened
        ))
        return self._ensure_precompiled(required)

    def _dispatcher(self, route: _CompiledCudaRoute) -> "CudaDispatcher":
        declaration = _CudaDispatcherDeclaration(
            route=route, spec=active_kernel_spec(),
        )
        return CudaDispatcher(self, route, spec=declaration.spec)

class CudaDispatcher:
    """Trusted adapter over one construction-time compiled CUDA route."""

    def __init__(
        self, group: CudaExtensionGroup, route: _CompiledCudaRoute, *,
        spec: KernelSpec,
    ) -> None:
        self.group = group
        self.route = route
        self.extension = route.extension
        self.launch = route.launch
        self.parameters = spec.parameters
        self.launch_args = route.launch_args
        self.native_signature = route.native_signature
        self.tensor_vector = route.tensor_vector
        self.projection = route.projection
        self.omitted = route.omitted
        self.spec = spec
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    def _validate_specialization_input(
        self, values: Mapping[str, Any], *, buffer_dtypes: BufferDTypeABI,
    ) -> None:
        """Backend-specific validation invoked only by the Pydantic request."""

        del buffer_dtypes
        block_size = values["BLOCK_SIZE"]
        if type(block_size) is not int or not 1 <= block_size <= 1024:
            raise ValueError(
                f"{self.spec.name}: CUDA BLOCK_SIZE must be an exact int in "
                f"[1, 1024], got {block_size!r}"
            )
        self.projection._validate(values, kernel=self.spec.name)

    @cached_property
    def _launcher(self):
        return getattr(self.group._load(self.extension), self.launch)

    def specialize(
        self, arguments: Dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Any:
        del buffer_dtypes
        values = dict(arguments)
        if self.tensor_vector is not None:
            values[self.tensor_vector.target] = self.tensor_vector.resolve(values)
        size_keys = (
            (self.spec.size_key,)
            if isinstance(self.spec.size_key, str)
            else self.spec.size_key
        )
        static_extent = 1
        for name in size_keys:
            static_extent *= values[name]
        if static_extent == 0:
            def no_op() -> None:
                return None

            return no_op
        if self.spec.compile_time_masks:
            masks = tuple(
                (
                    name,
                    self.spec.compile_time_mask(name, values),
                )
                for name in self.spec.compile_time_masks
            )
            launcher = getattr(
                self.group._load_variant(self.extension, masks), self.launch,
            )
        else:
            launcher = self._launcher
        static_launch = tuple(values[name] for name in self.launch_args)

        def launch():
            return launcher(*static_launch)

        return launch

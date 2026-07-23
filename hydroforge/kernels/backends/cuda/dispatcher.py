"""Declarative CUDA extension namespace and canonical launch adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
from typing import Any, Dict, Mapping

from hydroforge.kernels.backends.cuda.spec import (
    CudaExtensionSpec, cuda_declarations, cuda_function_signature,
    cuda_narrowed_index_parameters,
)
from hydroforge.kernels.mutation import record_kernel_writes
from hydroforge.contracts import BackendLoweringSpec, BufferDTypeABI, KernelSpec
from hydroforge.kernels.context import (
    active_kernel_spec, registry_factory, reject_direct_kernel_launch,
)


@dataclass(frozen=True, slots=True)
class _CudaTensorVector:
    """The sole inferred private CUDA argument form."""

    target: str
    sources: tuple[str, ...]

    def resolve(self, values: Mapping[str, Any]) -> list[Any]:
        return [values[name] for name in self.sources]


@dataclass(frozen=True, slots=True, kw_only=True)
class CudaNativeProjection:
    """Semantic preconditions for canonical values absent from a launcher."""

    fixed: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.fixed, Mapping):
            raise TypeError("CUDA native projection fixed values must be a mapping")
        fixed = dict(self.fixed)
        invalid = [name for name in fixed if not name.isidentifier()]
        if invalid:
            raise ValueError(
                "CUDA native projection fixed names must be Python "
                f"identifiers: {invalid}"
            )
        object.__setattr__(self, "fixed", MappingProxyType(fixed))

    def validate(self, values: Mapping[str, Any], *, kernel: str) -> None:
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


class CudaExtensionGroup:
    """Lazily build a named namespace of declarative CUDA extensions."""

    def __init__(
        self, owner_module: str, specs: Mapping[str, CudaExtensionSpec],
        *, binary_prefix: str | None = None, env_prefix: str = "HYDROFORGE",
        module_extensions: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            type(owner_module) is not str or not owner_module
            or any(not part.isidentifier() for part in owner_module.split("."))
        ):
            raise ValueError(
                "CUDA extension owner_module must be a dotted Python name"
            )
        if not isinstance(specs, Mapping) or not specs:
            raise TypeError("CUDA extension specs must be a non-empty mapping")
        invalid_names = [
            name for name in specs
            if type(name) is not str or not name.isidentifier()
        ]
        if invalid_names:
            raise ValueError(
                f"CUDA extension names must be identifiers: {invalid_names}"
            )
        invalid_specs = {
            name: type(spec).__name__
            for name, spec in specs.items()
            if not isinstance(spec, CudaExtensionSpec)
        }
        if invalid_specs:
            raise TypeError(
                f"CUDA extension catalog values must be CudaExtensionSpec: "
                f"{invalid_specs}"
            )
        resolved_prefix = binary_prefix or owner_module.replace(".", "_")
        if type(resolved_prefix) is not str or not resolved_prefix.isidentifier():
            raise ValueError("CUDA binary_prefix must be a Python identifier")
        if type(env_prefix) is not str or not env_prefix.isidentifier():
            raise ValueError("CUDA env_prefix must be a Python identifier")
        self.owner_module = owner_module
        self.binary_prefix = resolved_prefix
        self.specs = MappingProxyType(dict(specs))
        demands = {} if module_extensions is None else {
            module: frozenset(extensions)
            for module, extensions in module_extensions.items()
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
        self.module_extensions = MappingProxyType(demands)
        self._exports = {
            name: list(spec.functions) for name, spec in self.specs.items()
        }
        self._routes: set[tuple[str, str]] = set()
        self._routes_sealed = False
        self.env_prefix = env_prefix
        self._loaded: Dict[str, Any] = {}
        self._materialized_sources: Dict[str, str] = {}
        self._precompiled: set[str] = set()

    def route(
        self,
        extension: str,
        launch: str,
        *,
        projection: CudaNativeProjection | None = None,
    ):
        """Register one native export and return its zero-argument factory.

        The route is the sole Python declaration of the native launcher.  The
        extension export list used by pybind and precompilation is derived from
        these registrations, so downstream packages never repeat it in
        ``CudaExtensionSpec.functions``.
        """
        if self._routes_sealed:
            raise RuntimeError(
                "CUDA routes are sealed after extension materialization starts"
            )
        if extension not in self.specs:
            raise KeyError(f"unknown CUDA extension {extension!r}")
        extension_spec = self.specs[extension]
        if extension_spec.functions or extension_spec.declarations:
            raise TypeError(
                "route-managed CUDA extensions may not repeat functions or "
                "declarations in CudaExtensionSpec"
            )
        key = (extension, launch)
        if key in self._routes:
            raise ValueError(
                f"CUDA route {extension!r}/{launch!r} is already registered"
            )
        self._routes.add(key)
        exports = self._exports[extension]
        if launch not in exports:
            exports.append(launch)
        @registry_factory
        def factory():
            return self._dispatcher(
                extension,
                launch,
                projection=projection,
            )

        return factory

    def _effective_specs(self) -> Dict[str, CudaExtensionSpec]:
        result = {}
        for name, spec in self.specs.items():
            functions = tuple(self._exports[name])
            if not functions:
                raise RuntimeError(
                    f"CUDA extension {name!r} has no registered routes or "
                    "explicit exports"
                )
            result[name] = replace(spec, functions=functions)
        return result

    def _materialized_source(self, name: str) -> str:
        """Return the immutable source snapshot used by every CUDA phase."""

        try:
            spec = self.specs[name]
        except KeyError as error:
            raise KeyError(f"unknown CUDA extension {name!r}") from error
        current = spec.materialize_source()
        source = self._materialized_sources.setdefault(name, current)
        if current != source:
            raise RuntimeError(
                f"CUDA extension {name!r} source changed after its ABI was "
                "materialized; rebuild the model so parsing and compilation "
                "use one immutable source snapshot"
            )
        return source

    def load(self, name: str) -> Any:
        if name in self._loaded:
            return self._loaded[name]
        self._routes_sealed = True
        try:
            spec = self._effective_specs()[name]
        except KeyError as error:
            raise KeyError(f"unknown CUDA extension {name!r}") from error
        from hydroforge.kernels.backends.cuda.build import load_inline_cu_module

        source = self._materialized_source(name)
        declarations = spec.declarations or cuda_declarations(source, spec.functions)
        module = load_inline_cu_module(
            f"{self.binary_prefix}_{name}",
            cpp_sources="\n".join((*spec.cpp_headers, *declarations)),
            cuda_sources=source,
            functions=spec.functions, extra_cuda_cflags=spec.cflags,
            extra_include_paths=tuple(map(str, spec.include_paths)),
            extra_ldflags=spec.ldflags,
            env_prefix=self.env_prefix,
        )
        self._loaded[name] = module
        return module

    def ensure_precompiled(
        self, extensions: Any = None,
    ) -> Dict[str, Any]:
        """Build and load the requested subset of this extension catalog.

        Repeated calls are cumulative.  Omitting ``extensions`` preserves the
        public whole-catalog precompile behavior used by the CLI.
        """
        requested = (
            set(self.specs) if extensions is None else set(extensions)
        )
        unknown = requested.difference(self.specs)
        if unknown:
            raise KeyError(f"unknown CUDA extensions: {sorted(unknown)}")
        pending = requested.difference(self._precompiled)
        if not pending:
            return {name: self._loaded[name] for name in requested}
        self._routes_sealed = True
        from hydroforge.kernels.backends.cuda.precompile import precompile_extension_specs

        effective = {
            name: spec for name, spec in self._effective_specs().items()
            if name in pending
        }
        precompile_extension_specs(
            self.binary_prefix, effective, env_prefix=self.env_prefix,
            materialized_sources={
                name: self._materialized_source(name) for name in effective
            },
        )
        for name in pending:
            self.load(name)
        self._precompiled.update(pending)
        return {name: self._loaded[name] for name in requested}

    def ensure_precompiled_for_modules(
        self, opened_modules: Any,
    ) -> Dict[str, Any]:
        """Precompile the exact catalog subset required by model modules."""
        if not self.module_extensions:
            raise RuntimeError(
                f"{self.owner_module} does not declare module_extensions"
            )
        opened = set(opened_modules)
        unknown = opened.difference(self.module_extensions)
        if unknown:
            raise KeyError(
                "opened modules have no CUDA extension demand declaration: "
                f"{sorted(unknown)}"
            )
        required = set().union(*(
            self.module_extensions[module] for module in opened
        ))
        return self.ensure_precompiled(required)

    def _dispatcher(
        self, extension: str, launch: str, *,
        projection: CudaNativeProjection | None = None,
    ) -> "CudaDispatcher":
        active = active_kernel_spec()
        if active is None:
            raise TypeError(
                "CUDA routes may only materialize inside their "
                "BackendRegistry KernelSpec factory"
            )
        spec = active
        if extension not in self.specs:
            raise KeyError(f"unknown CUDA extension {extension!r}")
        if launch not in self._exports[extension]:
            raise ValueError(
                f"{spec.name}: CUDA launcher {launch!r} is not exported by "
                f"extension {extension!r}; declared exports are "
                f"{self._exports[extension]}"
            )
        source = self._materialized_source(extension)
        native_signature = cuda_function_signature(source, launch)
        narrowed = cuda_narrowed_index_parameters(
            source, launch,
            tuple(
                name for name, kind in spec.runtime_scalars.items()
                if kind == "index"
            ),
        )
        if narrowed:
            raise TypeError(
                f"{spec.name}: CUDA launcher narrows canonical int64 index "
                f"parameter(s) to int32: {list(narrowed)}; declare an int32 "
                "runtime scalar when the device algorithm is truly 32-bit"
            )
        return CudaDispatcher(
            self, extension, launch, spec=spec,
            native_signature=native_signature,
            projection=projection,
        )

class CudaDispatcher:
    """Cached keyword ABI to positional pybind launcher."""

    def __init__(
        self, group: CudaExtensionGroup, extension: str, launch: str, *,
        spec: KernelSpec, native_signature: tuple[tuple[str, str], ...],
        projection: CudaNativeProjection | None = None,
    ) -> None:
        parameters = spec.parameters
        launch_args = tuple(name for name, _kind in native_signature)
        native_names = set(launch_args)
        canonical_names = set(parameters)
        vector_arguments = tuple(
            name for name, native_type in native_signature
            if self._normalized_native_type(native_type)
            == "std::vector<at::Tensor>"
        )
        if len(vector_arguments) > 1:
            raise TypeError(
                f"{spec.name}: CUDA launcher has multiple private tensor "
                "vectors; their canonical partition cannot be inferred"
            )
        tensor_vector = None
        if vector_arguments:
            target = vector_arguments[0]
            if target in canonical_names:
                raise TypeError(
                    f"{spec.name}: CUDA tensor vector {target!r} must be a "
                    "private physical projection, not a canonical parameter"
                )
            sources = tuple(
                name for name in parameters
                if name in spec.buffers
                and name not in native_names
            )
            if not sources:
                raise TypeError(
                    f"{spec.name}: CUDA tensor vector {target!r} has no "
                    "canonical buffer sources to pack"
                )
            optional_sources = set(sources).intersection(spec.optional_buffers)
            if optional_sources:
                raise TypeError(
                    f"{spec.name}: CUDA tensor vectors cannot pack optional "
                    f"buffers: {sorted(optional_sources)}"
                )
            tensor_vector = _CudaTensorVector(target, sources)
        if projection is not None and not isinstance(
            projection, CudaNativeProjection,
        ):
            raise TypeError(
                f"{spec.name}: CUDA native projection must be a "
                "CudaNativeProjection contract"
            )
        projection = projection or CudaNativeProjection()
        consumed_canonical = native_names.intersection(canonical_names) | {
            source for source in (
                () if tensor_vector is None else tensor_vector.sources
            )
        }
        omitted_canonical = canonical_names.difference(consumed_canonical)
        unknown_fixed = set(projection.fixed).difference(omitted_canonical)
        if unknown_fixed:
            raise TypeError(
                f"{spec.name}: CUDA native projection fixes values that are "
                f"still consumed by the launcher/derived ABI or absent from "
                f"KernelSpec: {sorted(unknown_fixed)}"
            )
        missing_fixed = omitted_canonical.difference(projection.fixed)
        if missing_fixed:
            raise TypeError(
                f"{spec.name}: CUDA launcher omits canonical inputs "
                f"{sorted(missing_fixed)}; define every omitted value in "
                "CudaNativeProjection.fixed instead of inferring semantics "
                "from an absent native parameter"
            )
        if "BLOCK_SIZE" not in native_names:
            raise TypeError(
                f"{spec.name}: CUDA launcher must expose compiler-owned "
                "BLOCK_SIZE explicitly"
            )
        unknown = set(launch_args).difference(
            parameters,
            (() if tensor_vector is None else {tensor_vector.target}),
            {"BLOCK_SIZE"},
        )
        if unknown:
            raise ValueError(f"CUDA launch arguments are outside canonical ABI: {sorted(unknown)}")
        canonical_native_kinds = {
            **{name: "buffer" for name in spec.buffers},
            **{
                name: {
                    "bool": "bool", "int32": "int32",
                    "index": "index", "float32": "float32",
                }[kind]
                for name, kind in spec.runtime_scalars.items()
            },
            **{
                name: {
                    "bool": "bool", "int32": "int32", "float32": "float32",
                }[kind]
                for name, kind in spec.compile_time.items()
            },
        }
        for name, native_type in native_signature:
            if name == "BLOCK_SIZE":
                observed = self._native_kind(native_type)
                if observed != "index":
                    raise TypeError(
                        f"{spec.name}: CUDA launcher BLOCK_SIZE uses "
                        f"{native_type!r} ({observed}), requires int64 index"
                    )
                continue
            expected = canonical_native_kinds.get(name)
            if expected is None:
                continue
            observed = self._native_kind(native_type)
            if expected != observed:
                raise TypeError(
                    f"{spec.name}: CUDA launcher parameter {name!r} uses "
                    f"{native_type!r} ({observed}), KernelSpec requires "
                    f"{expected}"
                )
            if expected == "buffer":
                expected_optional = name in spec.optional_buffers
                observed_optional = self._native_buffer_optional(native_type)
                if observed_optional != expected_optional:
                    required = "optional" if expected_optional else "required"
                    native = "optional" if observed_optional else "required"
                    raise TypeError(
                        f"{spec.name}: CUDA launcher buffer {name!r} is "
                        f"{native}, KernelSpec declares it {required}"
                    )
        self.group = group
        self.extension = extension
        self.launch = launch
        self.parameters = parameters
        self.launch_args = launch_args
        self.native_signature = native_signature
        self.tensor_vector = tensor_vector
        self.projection = projection
        self.omitted = frozenset(omitted_canonical)
        self.spec = spec
        self.__hydroforge_kernel__ = spec.metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    @staticmethod
    def _normalized_native_type(native_type: str) -> str:
        return " ".join(
            token for token in native_type.replace("&", "").split()
            if token != "const"
        )

    @classmethod
    def _native_kind(cls, native_type: str) -> str:
        normalized = cls._normalized_native_type(native_type)
        if normalized in {
            "at::Tensor", "std::optional<at::Tensor>",
            "c10::optional<at::Tensor>",
        }:
            return "buffer"
        if normalized == "bool":
            return "bool"
        if normalized in {"int", "int32_t"}:
            return "int32"
        if normalized in {"long", "int64_t"}:
            return "index"
        if normalized == "float":
            return "float32"
        if normalized == "double":
            return "float64"
        raise TypeError(f"unsupported CUDA launcher parameter type {native_type!r}")

    @classmethod
    def _native_buffer_optional(cls, native_type: str) -> bool:
        normalized = cls._normalized_native_type(native_type)
        if normalized == "at::Tensor":
            return False
        if normalized in {
            "std::optional<at::Tensor>", "c10::optional<at::Tensor>",
        }:
            return True
        raise TypeError(
            f"unsupported CUDA tensor launcher type {native_type!r}"
        )

    def _validate_block_size(self, values: Mapping[str, Any]) -> None:
        block_size = values["BLOCK_SIZE"]
        if type(block_size) is not int or not 1 <= block_size <= 1024:
            raise ValueError(
                f"{self.spec.name}: CUDA BLOCK_SIZE must be an exact int in "
                f"[1, 1024], got {block_size!r}"
            )

    @cached_property
    def _launcher(self):
        return getattr(self.group.load(self.extension), self.launch)

    def __call__(self, **kwargs: Any):
        reject_direct_kernel_launch(self.__hydroforge_kernel__.name)
        values = dict(kwargs)
        accepted = set(self.parameters) | {"BLOCK_SIZE"}
        extra = set(values).difference(accepted)
        if extra:
            raise TypeError(f"unexpected CUDA kernel arguments: {sorted(extra)}")
        missing = accepted.difference(values)
        if missing:
            raise TypeError(f"missing CUDA kernel arguments: {sorted(missing)}")
        self._validate_block_size(values)
        self.spec.validate_host_arguments(values)
        self.projection.validate(values, kernel=self.spec.name)
        if self.tensor_vector is not None:
            values[self.tensor_vector.target] = self.tensor_vector.resolve(values)
        record_kernel_writes(self.__hydroforge_kernel__, values)
        return self._launcher(*(values[name] for name in self.launch_args))

    def specialize(
        self, arguments: Dict[str, Any], dynamic: frozenset[str], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Any:
        del buffer_dtypes
        dynamic_vector_sources = (
            () if self.tensor_vector is None else tuple(
                source for source in self.tensor_vector.sources
                if source in dynamic
            )
        )
        if dynamic_vector_sources:
            raise TypeError(
                f"{self.spec.name}: CUDA tensor vector cannot contain dynamic "
                f"canonical buffers {list(dynamic_vector_sources)} because "
                "specialization would freeze their first addresses"
            )
        dynamic_omitted = self.omitted.intersection(dynamic)
        if dynamic_omitted:
            raise TypeError(
                f"{self.spec.name}: CUDA native projection omits dynamic "
                f"canonical inputs {sorted(dynamic_omitted)}; projected "
                "arguments must be specialization-stable"
            )
        values = dict(arguments)
        self._validate_block_size(values)
        self.spec.validate_host_arguments(values)
        self.projection.validate(values, kernel=self.spec.name)
        missing = (set(self.parameters) | {"BLOCK_SIZE"}).difference(values)
        if missing:
            raise TypeError(f"missing CUDA kernel arguments: {sorted(missing)}")
        if self.tensor_vector is not None:
            values[self.tensor_vector.target] = self.tensor_vector.resolve(values)
        launcher = self._launcher
        dynamic_launch = tuple(name for name in self.launch_args if name in dynamic)
        static_launch = {name: values[name] for name in self.launch_args if name not in dynamic_launch}
        args = ", ".join(
            f"_values[{name!r}]" if name in dynamic_launch else f"_static[{name!r}]"
            for name in self.launch_args
        )
        namespace = {"_launcher": launcher, "_static": static_launch}
        exec(f"def launch(**_values): return _launcher({args})", namespace)  # noqa: S102
        return namespace["launch"]

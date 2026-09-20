"""Canonical dispatch adapters for Torch, Triton, and Metal kernels."""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from typing import Any, Literal

from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.kernels import (
    BackendLoweringSpec,
    BufferDTypeABI,
    KernelMetadata,
    KernelSpec,
    validate_launch_extent,
)
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.backends.metal.dispatcher import (
    make_metal_dispatcher as make_metal_dispatcher,
)
from hydroforge.kernels.context import (
    active_triton_precision,
    native_component_factory,
    resolve_factory_spec,
    triton_precision_context,
)


def _reject_unproven_uint32_runtime_scalars(
    spec: KernelSpec,
    backend: str,
) -> None:
    """Reject a backend that cannot prove fixed-width unsigned semantics."""

    names = sorted(
        name for name, kind in spec.runtime_scalars.items() if kind == "uint32"
    )
    if names:
        raise TypeError(
            f"{spec.name}: {backend} backend does not support canonical "
            f"uint32 runtime scalar(s) {names}; it cannot prove a fixed-width "
            "unsigned native representation"
        )


_TRITON_FLOAT_ANNOTATIONS = {
    "float32": "fp32",
    "float64": "fp64",
}


def _precision_scalar_kinds(spec: KernelSpec) -> dict[str, str]:
    return {
        name: kind
        for name, kind in (*spec.compile_time.items(), *spec.runtime_scalars.items())
        if name in spec._precision_parameter_names and kind in _TRITON_FLOAT_ANNOTATIONS
    }


def _precision_scalar_names(spec: KernelSpec) -> frozenset[str]:
    return frozenset(_precision_scalar_kinds(spec))


def _precision_from_spec(spec: KernelSpec) -> str | None:
    precisions = set(_precision_scalar_kinds(spec).values())
    return next(iter(precisions)) if len(precisions) == 1 else None


def _require_resolved_triton_precision(spec: KernelSpec) -> str | None:
    """Require a concrete precision before building a Triton dispatcher."""

    precision = _precision_from_spec(spec)
    if spec._uses_precision and precision is None:
        raise TypeError(
            f"{spec.name}: Triton precision-dependent KernelSpec must be "
            "resolved to float32 or float64 by the model/registry before "
            "constructing a dispatcher"
        )
    return precision


_TRITON_PRECISION_VARIANTS: dict[tuple[int, str, tuple[str, ...]], tuple[Any, Any]] = {}


def _precisionize_triton_kernel(
    kernel: Any,
    precision: str,
    scalar_names: frozenset[str],
) -> Any:
    """Return a Triton JIT variant with the resolved scalar annotations.

    Triton treats an unannotated Python ``float`` as fp32 even when the
    surrounding model is fp64.  Rebuilding the small Python function with
    precision-specific annotations lets Triton generate an ABI-correct
    specialization without changing every downstream launch helper.  Mock
    kernels used by contract tests intentionally do not support this and are
    validated strictly instead.
    """

    if not scalar_names or not hasattr(kernel, "params") or not hasattr(kernel, "fn"):
        return kernel
    expected = _TRITON_FLOAT_ANNOTATIONS.get(precision)
    if expected is None:
        return kernel
    parameters = {parameter.name: parameter for parameter in kernel.params}
    names = tuple(sorted(name for name in scalar_names if name in parameters))
    if not names:
        return kernel
    # A precision-dependent parameter is deliberately lowered to a typed
    # runtime scalar.  Matching the dtype alone is not sufficient: a kernel
    # may already spell the parameter as ``tl.float64`` while retaining the
    # ``tl.constexpr`` qualifier, which would still make Triton specialize the
    # value as a compile-time constant and bypass the model precision contract.
    if all(
        parameters[name].annotation_type == expected
        and not getattr(parameters[name], "is_constexpr", False)
        for name in names
    ):
        return kernel

    # Keep the source object alongside the integer key.  This avoids invoking
    # Triton's relatively expensive ``JITFunction.__hash__`` while also
    # preventing an id-reuse collision after a lazily-created implementation is
    # collected.
    key = (id(kernel), precision, names)
    cached = _TRITON_PRECISION_VARIANTS.get(key)
    if cached is not None:
        source_kernel, variant = cached
        if source_kernel is kernel:
            return variant

    import triton
    import triton.language as tl

    source = kernel.fn
    annotations = dict(getattr(source, "__annotations__", {}))
    dtype = tl.float32 if precision == "float32" else tl.float64
    for name in names:
        annotations[name] = dtype
    variant_name = f"{source.__name__}__hydroforge_{precision}"
    variant = types.FunctionType(
        source.__code__,
        source.__globals__,
        variant_name,
        source.__defaults__,
        source.__closure__,
    )
    variant.__kwdefaults__ = source.__kwdefaults__
    variant.__annotations__ = annotations
    variant.__dict__.update(getattr(source, "__dict__", {}))
    variant.__doc__ = source.__doc__
    variant.__module__ = source.__module__
    variant.__qualname__ = f"{source.__qualname__}__hydroforge_{precision}"
    jit_kwargs = {
        name: getattr(kernel, name)
        for name in (
            "version",
            "do_not_specialize",
            "do_not_specialize_on_alignment",
            "debug",
            "noinline",
            "launch_metadata",
        )
        if hasattr(kernel, name) and getattr(kernel, name) not in (None, [], {})
    }
    compiled = triton.jit(variant, **jit_kwargs)
    _TRITON_PRECISION_VARIANTS[key] = (kernel, compiled)
    return compiled


def _validate_triton_precision_scalars(
    kernel: Any,
    spec: KernelSpec,
    parameters: set[str],
    label: str,
) -> None:
    # Precision-dependent compile-time values are intentionally lowered to
    # typed runtime scalars by ``_precisionize_triton_kernel``.  Validate both
    # canonical ABI classes here: checking runtime_scalars alone would allow a
    # stale ``tl.constexpr`` annotation on a config constant to slip through.
    precision_names = _precision_scalar_names(spec)
    names = precision_names.intersection(parameters)
    if not names:
        return
    parameters_by_name = {
        parameter.name: parameter for parameter in getattr(kernel, "params", ())
    }
    expected = {
        name: _TRITON_FLOAT_ANNOTATIONS[
            spec.runtime_scalars.get(name, spec.compile_time.get(name))
        ]
        for name in names
    }
    invalid = sorted(
        name
        for name in names
        if (
            parameters_by_name.get(name) is None
            or getattr(parameters_by_name[name], "is_constexpr", False)
            or parameters_by_name[name].annotation_type != expected[name]
        )
    )
    if invalid:
        if any(expected[name] == "fp64" for name in invalid):
            detail = "explicit tl.float64 annotations"
        else:
            detail = "explicit tl.float32 annotations"
        raise TypeError(
            f"{spec.name}: {label} Triton precision runtime scalar ABI "
            f"requires {detail}; expected={expected}, invalid={invalid}"
        )


def launch_triton_kernel(kernel: Any, grid: Any) -> Callable:
    """Return a precision-aware launch proxy for an inner Triton kernel.

    This is intended for compound-program helpers.  The returned callable
    binds the actual launch arguments first, identifies declared floating
    scalar parameters, and dispatches a cached fp32/fp64 JIT variant.  Outside
    a :func:`triton_precision_context` it is exactly equivalent to
    ``kernel[grid]``.
    """

    def launch(*args: Any, **kwargs: Any):
        active = active_triton_precision()
        selected = kernel
        if active is not None:
            precision, declared = active
            names: set[str] = set()
            signature = getattr(kernel, "fn", kernel)
            try:
                kernel_signature = inspect.signature(signature)
                kernel_kwargs = {
                    name: value
                    for name, value in kwargs.items()
                    if name in kernel_signature.parameters
                }
                bound = kernel_signature.bind_partial(*args, **kernel_kwargs)
                bound.apply_defaults()
            except (TypeError, ValueError):
                bound = None
            if bound is not None:
                names = {
                    name
                    for name, value in bound.arguments.items()
                    if name in declared and isinstance(value, float)
                }
            selected = _precisionize_triton_kernel(
                kernel,
                precision,
                frozenset(names),
            )
        return selected[grid](*args, **kwargs)

    return launch


def _empty_launch() -> None:
    return None


class _SpecializedDispatcher:
    """Non-callable backend declaration with one trusted specializer."""

    def __init__(
        self,
        metadata: KernelMetadata,
        lowering: BackendLoweringSpec,
        specializer: Callable,
    ) -> None:
        self.__hydroforge_kernel__ = metadata
        self.__hydroforge_lowering__ = lowering
        self._specializer = specializer

    def specialize(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        return self._specializer(arguments, buffer_dtypes=buffer_dtypes)


def _torch_compile(fn: Callable) -> Callable:
    """Apply torch.compile with inference-optimized settings.

    All physics kernels mutate inputs via ``.copy_()`` / indexed assignment,
    so ``reduce-overhead`` mode (which relies on internal CUDA graphs) can
    never actually use its main optimisation and only produces warnings.
    We use ``fullgraph=True`` so that compilation errors surface at the
    first call rather than lazily on a rare code-path hours later.
    """
    import torch

    return torch.compile(fn, fullgraph=True)


class TorchDispatcher:
    """Strict canonical-ABI dispatcher for a native PyTorch implementation."""

    def __init__(
        self,
        kernel: Callable,
        spec: KernelSpec,
        *,
        compile: bool = True,
    ) -> None:
        _reject_unproven_uint32_runtime_scalars(spec, "Torch")
        signature = inspect.signature(kernel)
        parameters = tuple(signature.parameters)
        if parameters != spec.parameters:
            raise TypeError(
                f"{spec.name}: torch signature {parameters!r} must exactly match "
                f"KernelSpec {spec.parameters!r}"
            )
        if any(
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            for parameter in signature.parameters.values()
        ):
            raise TypeError(
                f"{spec.name}: torch kernels must accept canonical arguments "
                "by keyword and may not use positional-only, *args, or **kwargs"
            )
        self._kernel = _torch_compile(kernel) if compile else kernel
        self.spec = spec
        self._parameters = frozenset(spec.parameters)
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    def specialize(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        """Return a zero-argument launch for an already validated call."""
        del buffer_dtypes
        extent = validate_launch_extent(
            self.spec.name,
            self.spec.size_key,
            arguments,
        )
        if extent == 0:
            return _empty_launch
        static = {
            name: value for name, value in arguments.items() if name in self._parameters
        }

        def launch():
            return self._kernel(**static)

        return launch


class _TorchDispatcherDeclaration(HydroForgeModel):
    kernel: Callable
    spec: KernelSpec | None = None
    compile: bool = True

    _dispatcher: TorchDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            spec = resolve_factory_spec(self.spec, factory="make_torch_dispatcher")
            self._dispatcher = TorchDispatcher(
                self.kernel,
                spec,
                compile=self.compile,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self


def make_torch_dispatcher(
    kernel: Callable,
    spec: KernelSpec | None = None,
    *,
    compile: bool = True,
) -> TorchDispatcher:
    """Build a formal Torch backend from the active canonical Spec."""

    return _TorchDispatcherDeclaration(
        kernel=kernel,
        spec=spec,
        compile=compile,
    )._dispatcher


# ── Triton dispatcher factory ─────────────────────────────────────────────


def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _make_triton_dispatcher_trusted(
    kernel: Any,
    *,
    spec: KernelSpec | None = None,
    batched_kernel: Any = None,
    batched_grid: str = "parallel",
) -> _SpecializedDispatcher:
    """Create a unified dispatch function for a Triton kernel pair.

    Shared/batched selection, accepted arguments and launch geometry are fixed
    once by ``specialize`` while an execution plan is initialized.  The launch
    closure performs no variant selection or canonical-ABI argument dropping.

    Args:
        kernel: Non-batched Triton JIT kernel.
        batched_kernel: Batched variant (or ``None``).
        batched_grid: ``"parallel"`` → ``cdiv(n*nt, BS)``; ``"loop"`` → ``cdiv(n, BS)``.
    """
    canonical = resolve_factory_spec(spec, factory="make_triton_dispatcher")
    _reject_unproven_uint32_runtime_scalars(canonical, "Triton")
    precision = _require_resolved_triton_precision(canonical)
    precision_names = _precision_scalar_names(canonical)
    if precision is not None:
        kernel = _precisionize_triton_kernel(
            kernel,
            precision,
            precision_names,
        )
        if batched_kernel is not None:
            batched_kernel = _precisionize_triton_kernel(
                batched_kernel,
                precision,
                precision_names,
            )
    size_key = canonical.size_key

    def specialize(
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        del buffer_dtypes
        bs = arguments["BLOCK_SIZE"]
        members = arguments.get("ensemble_size")
        use_batched = members is not None and members > 1 and batched_kernel is not None
        selected = batched_kernel if use_batched else kernel
        accepted = frozenset(
            name for name in getattr(selected, "arg_names", ()) if name != "BLOCK_SIZE"
        )
        static = {name: value for name, value in arguments.items() if name in accepted}
        size_keys = (size_key,) if isinstance(size_key, str) else size_key
        static_n = 1
        for key in size_keys:
            static_n *= arguments[key]
        if use_batched and batched_grid == "parallel":
            static_n *= members

        if static_n == 0:
            return _empty_launch
        grid = (_cdiv(static_n, bs),)

        def launch():
            selected[grid](BLOCK_SIZE=bs, **static)

        return launch

    canonical_parameters = set(canonical.parameters)

    def validate_variant(candidate, label: str, *, complete: bool) -> None:
        parameters = tuple(
            name for name in getattr(candidate, "arg_names", ()) if name != "BLOCK_SIZE"
        )
        if len(parameters) != len(set(parameters)):
            raise TypeError(
                f"{canonical.name}: {label} Triton variant has duplicate "
                "native parameters"
            )
        observed = set(parameters)
        _validate_triton_precision_scalars(
            candidate,
            canonical,
            observed,
            label,
        )
        extra = observed.difference(canonical_parameters)
        missing = canonical_parameters.difference(observed)
        if extra:
            raise TypeError(
                f"{canonical.name}: {label} Triton variant has parameters "
                f"outside KernelSpec: {sorted(extra)}"
            )
        missing_buffers = missing.intersection(canonical.buffers)
        if missing_buffers:
            raise TypeError(
                f"{canonical.name}: {label} Triton variant omits canonical "
                f"buffers: {sorted(missing_buffers)}"
            )
        if complete and missing:
            raise TypeError(
                f"{canonical.name}: {label} Triton variant must consume the "
                f"complete canonical ABI: missing={sorted(missing)}"
            )
        if not complete:
            # Validate the selected shared surface as a real KernelSpec
            # projection.  This rejects omitted launch extents and orphaned
            # optional arguments instead of treating any scalar subset as an
            # implementation detail.
            canonical.project(
                omit=tuple(name for name in canonical.parameters if name in missing)
            )

    if batched_kernel is None:
        validate_variant(kernel, "single", complete=True)
    else:
        # A shared implementation may project out scalar values used only to
        # select or index the batched layout.  Buffers are never grid-only:
        # every selectable kernel must consume the complete state ABI.
        validate_variant(kernel, "shared", complete=False)
        validate_variant(batched_kernel, "batched", complete=True)
    lowering = BackendLoweringSpec.plan_specialized(
        buffer_elements="tensor",
    )
    return _SpecializedDispatcher(
        canonical._metadata_for_lowering(lowering),
        lowering,
        specialize,
    )


class _TritonDispatcherDeclaration(HydroForgeModel):
    kernel: Any
    spec: KernelSpec | None = None
    batched_kernel: Any = None
    batched_grid: Literal["parallel", "loop"] = "parallel"

    _dispatcher: _SpecializedDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            self._dispatcher = _make_triton_dispatcher_trusted(
                self.kernel,
                spec=self.spec,
                batched_kernel=self.batched_kernel,
                batched_grid=self.batched_grid,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self


def make_triton_dispatcher(
    kernel: Any,
    *,
    spec: KernelSpec | None = None,
    batched_kernel: Any = None,
    batched_grid: Literal["parallel", "loop"] = "parallel",
) -> _SpecializedDispatcher:
    """Validate and build a Triton dispatcher declaration."""

    return _TritonDispatcherDeclaration(
        kernel=kernel,
        spec=spec,
        batched_kernel=batched_kernel,
        batched_grid=batched_grid,
    )._dispatcher


def _make_triton_sequence_dispatcher_trusted(
    *,
    kernels: tuple[tuple[Any, str | tuple[str, ...]], ...],
    spec: KernelSpec | None = None,
) -> _SpecializedDispatcher:
    """Compose ordered native launches under one canonical logical ABI.

    Components consume exact-name subsets of the public ABI. Their complete
    arguments and launch geometry are specialized once, so the hot path is
    only the prebuilt sequence of native launches.
    """
    spec = resolve_factory_spec(spec, factory="make_triton_sequence_dispatcher")
    _reject_unproven_uint32_runtime_scalars(spec, "Triton")
    _require_resolved_triton_precision(spec)
    # Component extents are backend implementation strategy, not alternative
    # public Specs.  Build them in an explicitly isolated native context.
    component_specs = []
    for kernel, component_size in kernels:
        native_parameters = tuple(
            name for name in getattr(kernel, "arg_names", ()) if name != "BLOCK_SIZE"
        )
        component_specs.append(
            spec.project(
                omit=tuple(
                    name for name in spec.parameters if name not in native_parameters
                ),
                size_key=component_size,
            )
        )
    with native_component_factory():
        components = tuple(
            _make_triton_dispatcher_trusted(kernel, spec=component_spec)
            for (kernel, _component_size), component_spec in zip(
                kernels, component_specs, strict=True
            )
        )
    expected = frozenset(spec.parameters)
    component_parameters = tuple(
        frozenset(component.__hydroforge_kernel__.parameters)
        for component in components
    )
    consumed = frozenset().union(*component_parameters)
    if consumed != expected:
        raise ValueError(
            f"{spec.name}: Triton sequence ABI mismatch: "
            f"missing={sorted(expected - consumed)}, "
            f"extra={sorted(consumed - expected)}"
        )

    def specialize(
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        launches = []
        for component, accepted in zip(
            components,
            component_parameters,
            strict=True,
        ):
            selected = {
                key: value
                for key, value in arguments.items()
                if key in accepted or key == "BLOCK_SIZE"
            }
            launch = component.specialize(
                selected,
                buffer_dtypes={
                    name: dtype
                    for name, dtype in buffer_dtypes.items()
                    if name in accepted
                },
            )
            launches.append(launch)

        def run() -> None:
            for launch in launches:
                launch()

        return run

    lowering = BackendLoweringSpec.plan_specialized(
        buffer_elements="tensor",
    )
    return _SpecializedDispatcher(
        spec._metadata_for_lowering(lowering),
        lowering,
        specialize,
    )


class _TritonSequenceDeclaration(HydroForgeModel):
    kernels: tuple[tuple[Any, str | tuple[str, ...]], ...]
    spec: KernelSpec | None = None

    _dispatcher: _SpecializedDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            self._dispatcher = _make_triton_sequence_dispatcher_trusted(
                kernels=self.kernels,
                spec=self.spec,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self


def make_triton_sequence_dispatcher(
    *,
    kernels: tuple[tuple[Any, str | tuple[str, ...]], ...],
    spec: KernelSpec | None = None,
) -> _SpecializedDispatcher:
    """Validate and build an ordered Triton sequence."""

    return _TritonSequenceDeclaration(
        kernels=kernels,
        spec=spec,
    )._dispatcher


def _make_triton_program_dispatcher_trusted(
    prepare: Callable[..., Callable[..., None]],
    spec: KernelSpec | None = None,
) -> _SpecializedDispatcher:
    """Build one cached, ordered Triton program behind a canonical ABI.

    ``prepare`` is initialization/specialization work: it receives the stable
    argument mapping and concrete buffer dtype ABI, then returns the hot-path
    launch callable. This adapter is for physical
    operators made of several dependent native launches and device tensor
    expressions which cannot be represented as an independent-kernel sequence.
    It does not permit a Python launch fallback: preparation happens once per
    specialization and the returned program is captured by the normal compiled
    operator runtime.
    """
    spec = resolve_factory_spec(spec, factory="make_triton_program_dispatcher")
    _reject_unproven_uint32_runtime_scalars(spec, "Triton")
    precision = _require_resolved_triton_precision(spec)
    precision_names = _precision_scalar_names(spec)
    signature = inspect.signature(prepare)
    if tuple(signature.parameters) != ("arguments", "buffer_dtypes"):
        raise TypeError(
            f"{spec.name}: Triton program prepare signature must be exactly "
            "(arguments, buffer_dtypes)"
        )

    def specialize(
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        if (
            validate_launch_extent(
                spec.name,
                spec.size_key,
                arguments,
            )
            == 0
        ):
            return _empty_launch
        if precision is None:
            return prepare(arguments, buffer_dtypes)
        with triton_precision_context(precision, precision_names):
            prepared = prepare(arguments, buffer_dtypes)

        def launch() -> None:
            with triton_precision_context(precision, precision_names):
                prepared()

        return launch

    lowering = BackendLoweringSpec.plan_specialized(buffer_elements="tensor")
    return _SpecializedDispatcher(
        spec._metadata_for_lowering(lowering),
        lowering,
        specialize,
    )


class _TritonProgramDeclaration(HydroForgeModel):
    prepare: Callable[..., Callable[..., None]]
    spec: KernelSpec | None = None

    _dispatcher: _SpecializedDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            self._dispatcher = _make_triton_program_dispatcher_trusted(
                self.prepare,
                self.spec,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self


def make_triton_program_dispatcher(
    prepare: Callable[..., Callable[..., None]],
    spec: KernelSpec | None = None,
) -> _SpecializedDispatcher:
    """Validate and build a specialized Triton program."""

    return _TritonProgramDeclaration(
        prepare=prepare,
        spec=spec,
    )._dispatcher

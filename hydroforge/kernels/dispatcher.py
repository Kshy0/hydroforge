"""Canonical dispatch adapters for Torch, Triton, and Metal kernels."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal

from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.kernels import (
    BackendLoweringSpec, BufferDTypeABI, KernelMetadata, KernelSpec,
)
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.context import (
    active_kernel_spec, native_component_factory,
)


def _reject_unproven_uint32_runtime_scalars(
    spec: KernelSpec, backend: str,
) -> None:
    """Reject a backend that cannot prove fixed-width unsigned semantics."""

    names = sorted(
        name for name, kind in spec.runtime_scalars.items()
        if kind == "uint32"
    )
    if names:
        raise TypeError(
            f"{spec.name}: {backend} backend does not support canonical "
            f"uint32 runtime scalar(s) {names}; it cannot prove a fixed-width "
            "unsigned native representation"
        )


def _validate_triton_float64_scalars(
    kernel: Any,
    spec: KernelSpec,
    parameters: set[str],
    label: str,
) -> None:
    names = {
        name for name in parameters
        if spec.runtime_scalars.get(name) == "float64"
    }
    if not names:
        return
    parameters_by_name = {
        parameter.name: parameter
        for parameter in getattr(kernel, "params", ())
    }
    invalid = sorted(
        name for name in names
        if (
            parameters_by_name.get(name) is None
            or (
                parameters_by_name[name].annotation_type != "fp64"
                and not getattr(parameters_by_name[name], "is_constexpr", False)
            )
        )
    )
    if invalid:
        raise TypeError(
            f"{spec.name}: {label} Triton float64 runtime scalar(s) require "
            f"explicit tl.float64 annotations: {invalid}"
        )


class _SpecializedDispatcher:
    """Non-callable backend declaration with one trusted specializer."""

    def __init__(
        self, metadata: KernelMetadata, lowering: BackendLoweringSpec,
        specializer: Callable,
    ) -> None:
        self.__hydroforge_kernel__ = metadata
        self.__hydroforge_lowering__ = lowering
        self._specializer = specializer

    def specialize(
        self, arguments: dict[str, Any], *,
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
        import inspect

        _reject_unproven_uint32_runtime_scalars(spec, "Torch")
        signature = inspect.signature(kernel)
        parameters = tuple(signature.parameters)
        if parameters != spec.parameters:
            raise TypeError(
                f"{spec.name}: torch signature {parameters!r} must exactly match "
                f"KernelSpec {spec.parameters!r}"
            )
        if any(
            parameter.kind in {
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
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        """Return a zero-argument launch for an already validated call."""
        del buffer_dtypes
        static = {
            name: value for name, value in arguments.items()
            if name in self._parameters
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
        active = active_kernel_spec()
        if active is not None:
            if self.spec is not None:
                raise ValueError(
                    "Torch factory may not repeat active KernelSpec metadata"
                )
            spec = active
        elif self.spec is None:
            raise ValueError(
                "make_torch_dispatcher requires a KernelSpec outside a "
                "BackendRegistry factory"
            )
        else:
            spec = self.spec
        try:
            self._dispatcher = TorchDispatcher(
                self.kernel, spec, compile=self.compile,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self

    @property
    def dispatcher(self) -> TorchDispatcher:
        return self._dispatcher


def make_torch_dispatcher(
    kernel: Callable,
    spec: KernelSpec | None = None,
    *,
    compile: bool = True,
) -> TorchDispatcher:
    """Build a formal Torch backend from the active canonical Spec."""

    return _TorchDispatcherDeclaration(
        kernel=kernel, spec=spec, compile=compile,
    ).dispatcher


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
    from hydroforge.kernels.context import active_kernel_spec

    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(
                "Triton factory may not repeat active KernelSpec metadata "
                "through spec"
            )
        canonical = active
    elif spec is None:
        raise TypeError(
            "make_triton_dispatcher requires a BackendRegistry KernelSpec "
            "context or an explicit spec"
        )
    else:
        canonical = spec
    _reject_unproven_uint32_runtime_scalars(canonical, "Triton")
    size_key = canonical.size_key
    if batched_grid not in {"parallel", "loop"}:
        raise ValueError(
            "batched_grid must be exactly 'parallel' or 'loop', got "
            f"{batched_grid!r}"
        )

    def specialize(
        arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        del buffer_dtypes
        bs = arguments["BLOCK_SIZE"]
        trials = arguments.get("num_trials")
        use_batched = (
            trials is not None and trials > 1 and batched_kernel is not None
        )
        selected = batched_kernel if use_batched else kernel
        accepted = frozenset(
            name for name in getattr(selected, "arg_names", ())
            if name != "BLOCK_SIZE"
        )
        static = {
            name: value for name, value in arguments.items()
            if name in accepted
        }
        size_keys = (size_key,) if isinstance(size_key, str) else size_key
        static_n = 1
        for key in size_keys:
            static_n *= arguments[key]
        if use_batched and batched_grid == "parallel":
            static_n *= trials

        def launch():
            if static_n == 0:
                return None
            grid = (_cdiv(static_n, bs),)
            selected[grid](BLOCK_SIZE=bs, **static)

        return launch

    canonical_parameters = set(canonical.parameters)

    def validate_variant(candidate, label: str, *, complete: bool) -> None:
        parameters = tuple(
            name for name in getattr(candidate, "arg_names", ())
            if name != "BLOCK_SIZE"
        )
        if len(parameters) != len(set(parameters)):
            raise TypeError(
                f"{canonical.name}: {label} Triton variant has duplicate "
                "native parameters"
            )
        observed = set(parameters)
        _validate_triton_float64_scalars(
            candidate, canonical, observed, label,
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
            canonical.project(omit=tuple(
                name for name in canonical.parameters if name in missing
            ))

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
        canonical._metadata_for_lowering(lowering), lowering, specialize,
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

    @property
    def dispatcher(self) -> _SpecializedDispatcher:
        return self._dispatcher


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
    ).dispatcher


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
    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(
                "Triton sequence may not repeat active KernelSpec metadata"
            )
        spec = active
    elif spec is None:
        raise TypeError(
            "make_triton_sequence_dispatcher requires a KernelSpec outside "
            "a BackendRegistry factory"
        )
    _reject_unproven_uint32_runtime_scalars(spec, "Triton")
    # Component extents are backend implementation strategy, not alternative
    # public Specs.  Build them in an explicitly isolated native context.
    component_specs = []
    for kernel, component_size in kernels:
        native_parameters = tuple(
            name for name in getattr(kernel, "arg_names", ())
            if name != "BLOCK_SIZE"
        )
        component_specs.append(spec.project(
            omit=tuple(
                name for name in spec.parameters
                if name not in native_parameters
            ),
            size_key=component_size,
        ))
    with native_component_factory():
        components = tuple(
            _make_triton_dispatcher_trusted(kernel, spec=component_spec)
            for (kernel, _component_size), component_spec
            in zip(kernels, component_specs, strict=True)
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
        arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        launches = []
        for component, accepted in zip(
            components, component_parameters, strict=True,
        ):
            selected = {
                key: value for key, value in arguments.items()
                if key in accepted or key == "BLOCK_SIZE"
            }
            launch = component.specialize(
                selected,
                buffer_dtypes={
                    name: dtype for name, dtype in buffer_dtypes.items()
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
        spec._metadata_for_lowering(lowering), lowering, specialize,
    )


class _TritonSequenceDeclaration(HydroForgeModel):
    kernels: tuple[tuple[Any, str | tuple[str, ...]], ...]
    spec: KernelSpec | None = None

    _dispatcher: _SpecializedDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            self._dispatcher = _make_triton_sequence_dispatcher_trusted(
                kernels=self.kernels, spec=self.spec,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self

    @property
    def dispatcher(self) -> _SpecializedDispatcher:
        return self._dispatcher


def make_triton_sequence_dispatcher(
    *,
    kernels: tuple[tuple[Any, str | tuple[str, ...]], ...],
    spec: KernelSpec | None = None,
) -> _SpecializedDispatcher:
    """Validate and build an ordered Triton sequence."""

    return _TritonSequenceDeclaration(
        kernels=kernels, spec=spec,
    ).dispatcher


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
    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(
                "Triton program may not repeat active KernelSpec metadata"
            )
        spec = active
    elif spec is None:
        raise TypeError(
            "make_triton_program_dispatcher requires a KernelSpec outside "
            "a BackendRegistry factory"
        )
    _reject_unproven_uint32_runtime_scalars(spec, "Triton")
    signature = inspect.signature(prepare)
    if tuple(signature.parameters) != ("arguments", "buffer_dtypes"):
        raise TypeError(
            f"{spec.name}: Triton program prepare signature must be exactly "
            "(arguments, buffer_dtypes)"
        )

    def specialize(
        arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        return prepare(arguments, buffer_dtypes)

    lowering = BackendLoweringSpec.plan_specialized(buffer_elements="tensor")
    return _SpecializedDispatcher(
        spec._metadata_for_lowering(lowering), lowering, specialize,
    )


class _TritonProgramDeclaration(HydroForgeModel):
    prepare: Callable[..., Callable[..., None]]
    spec: KernelSpec | None = None

    _dispatcher: _SpecializedDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            self._dispatcher = _make_triton_program_dispatcher_trusted(
                self.prepare, self.spec,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self

    @property
    def dispatcher(self) -> _SpecializedDispatcher:
        return self._dispatcher


def make_triton_program_dispatcher(
    prepare: Callable[..., Callable[..., None]],
    spec: KernelSpec | None = None,
) -> _SpecializedDispatcher:
    """Validate and build a specialized Triton program."""

    return _TritonProgramDeclaration(
        prepare=prepare, spec=spec,
    ).dispatcher


# Metal is a separate adapter; this import preserves the public factory.
from hydroforge.kernels.backends.metal.dispatcher import make_metal_dispatcher  # noqa: E402, F401

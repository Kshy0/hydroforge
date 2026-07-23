"""Canonical dispatch adapters for Torch, Triton, and Metal kernels."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from hydroforge.contracts import (
    BackendLoweringSpec, BufferDTypeABI, KernelMetadata, KernelSpec,
)
from hydroforge.kernels.context import (
    active_kernel_spec, native_component_factory, reject_direct_kernel_launch,
)


def _metadata(callable_: Callable) -> KernelMetadata | None:
    return getattr(callable_, "__hydroforge_kernel__", None)


def require_specializer(implementation: Any, *, label: str) -> Callable:
    """Return one strict backend specializer or reject Python launch fallback."""

    specializer = getattr(implementation, "specialize", None)
    if not callable(specializer):
        raise TypeError(
            f"{label} must implement specialize(); Python launch fallback "
            "is forbidden"
        )
    parameter = inspect.signature(specializer).parameters.get("buffer_dtypes")
    if (
        parameter is None
        or parameter.kind is not inspect.Parameter.KEYWORD_ONLY
    ):
        raise TypeError(
            f"{label} specialize() must accept the keyword-only canonical "
            "buffer_dtypes ABI"
        )
    return specializer


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

        signature = inspect.signature(kernel)
        parameters = tuple(signature.parameters)
        if parameters != spec.parameters:
            raise TypeError(
                f"{spec.name}: torch signature {parameters!r} must exactly match "
                f"KernelSpec {spec.parameters!r}"
            )
        if any(
            parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            for parameter in signature.parameters.values()
        ):
            raise TypeError(f"{spec.name}: torch kernels may not use *args/**kwargs")
        self._kernel = _torch_compile(kernel) if compile else kernel
        self.spec = spec
        self._parameters = frozenset(spec.parameters)
        self.__hydroforge_kernel__ = spec.metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    def __call__(self, **kwargs: Any):
        reject_direct_kernel_launch(self.__hydroforge_kernel__.name)
        supplied = set(kwargs)
        if supplied != self._parameters:
            raise TypeError(
                "torch kernel ABI mismatch: "
                f"missing={sorted(self._parameters - supplied)}, "
                f"extra={sorted(supplied - self._parameters)}"
            )
        self.spec.validate_host_arguments(kwargs)
        return self._kernel(**kwargs)

    def specialize(
        self, arguments: dict[str, Any], dynamic: frozenset[str], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        """Validate once and return a canonical launch without hot-path ABI work."""
        del buffer_dtypes
        supplied = set(arguments).difference({"BLOCK_SIZE"})
        if supplied != self._parameters:
            raise TypeError(
                "torch kernel specialization ABI mismatch: "
                f"missing={sorted(self._parameters - supplied)}, "
                f"extra={sorted(supplied - self._parameters)}"
            )
        for buffer, feature in self.spec.optional_buffers.items():
            if buffer in dynamic or (feature is not None and feature in dynamic):
                raise TypeError(
                    f"optional Torch ABI ({buffer}, {feature}) must be static"
                )
        self.spec.validate_host_arguments(arguments)
        static = {
            name: value for name, value in arguments.items()
            if name in self._parameters and name not in dynamic
        }

        def launch(**values: Any):
            return self._kernel(**static, **values)

        return launch


def make_torch_dispatcher(
    kernel: Callable,
    spec: KernelSpec | None = None,
    *,
    compile: bool = True,
) -> TorchDispatcher:
    """Build a formal Torch backend from the active canonical Spec."""

    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(
                "Torch factory may not repeat active KernelSpec metadata"
            )
        spec = active
    elif spec is None:
        raise TypeError(
            "make_torch_dispatcher requires a KernelSpec outside a "
            "BackendRegistry factory"
        )
    return TorchDispatcher(kernel, spec, compile=compile)


class VariantDispatcher:
    """Initialization-specialized shared/batched implementation pair."""

    def __init__(
        self, shared: Callable, batched: Callable, *, batch_key: str,
        spec: Any,
    ) -> None:
        self.shared = shared
        self.batched = batched
        self.batch_key = batch_key
        shared_metadata = _metadata(shared)
        batched_metadata = _metadata(batched)
        if shared_metadata is None or batched_metadata is None:
            raise TypeError("variant implementations require KernelMetadata")
        if batch_key not in spec.parameters:
            raise TypeError(
                f"variant batch key {batch_key!r} is absent from KernelSpec"
            )
        canonical_parameters = set(spec.parameters)
        batched_parameters = set(batched_metadata.parameters)
        if batched_parameters != canonical_parameters:
            raise TypeError(
                f"{spec.name}: batched variant must consume the complete "
                "canonical ABI: "
                f"missing={sorted(canonical_parameters - batched_parameters)}, "
                f"extra={sorted(batched_parameters - canonical_parameters)}"
            )
        if batch_key in shared_metadata.parameters:
            raise TypeError(
                f"{spec.name}: shared variant may not consume selection key "
                f"{batch_key!r}; that key belongs to the batched projection"
            )
        expected_shared = canonical_parameters.difference({batch_key})
        shared_parameters = set(shared_metadata.parameters)
        if shared_parameters != expected_shared:
            raise TypeError(
                f"{spec.name}: shared variant must consume exactly the "
                "canonical ABI except for its selection key: "
                f"missing={sorted(expected_shared - shared_parameters)}, "
                f"extra={sorted(shared_parameters - expected_shared)}"
            )
        for label, metadata in (("shared", shared_metadata), ("batched", batched_metadata)):
            unknown = set(metadata.parameters).difference(spec.parameters)
            if unknown:
                raise TypeError(
                    f"{spec.name}: {label} variant has parameters outside "
                    f"KernelSpec: {sorted(unknown)}"
                )
            for name, access in metadata.buffers.items():
                expected_access = spec.buffers.get(name)
                if expected_access != access:
                    raise TypeError(
                        f"{spec.name}: {label} variant buffer {name!r} has "
                        f"access={access}, KernelSpec requires={expected_access}"
                    )
            for name, kind in metadata.compile_time.items():
                if spec.compile_time.get(name) != kind:
                    raise TypeError(
                        f"{spec.name}: {label} variant constant {name!r}={kind} "
                        "differs from KernelSpec"
                    )
            for name, feature in metadata.optional_buffers.items():
                if spec.optional_buffers.get(name) != feature:
                    raise TypeError(
                        f"{spec.name}: {label} variant optional buffer "
                        f"{name!r} differs from KernelSpec"
                    )
            for name, value in metadata.optional_values.items():
                if spec.optional_values.get(name) != value:
                    raise TypeError(
                        f"{spec.name}: {label} variant optional value "
                        f"{name!r} differs from KernelSpec"
                    )
        self.__hydroforge_kernel__ = spec.metadata
        shared_lowering = getattr(shared, "__hydroforge_lowering__", None)
        batched_lowering = getattr(batched, "__hydroforge_lowering__", None)
        if shared_lowering is None or batched_lowering is None:
            raise TypeError("shared/batched variants require lowering strategies")
        if shared_lowering.buffer_elements != batched_lowering.buffer_elements:
            raise TypeError(
                "shared/batched variants require identical buffer-element lowering"
            )
        specializers = []
        for label, implementation in (
            ("shared", shared), ("batched", batched),
        ):
            specializers.append(require_specializer(
                implementation, label=f"{spec.name}: {label} variant",
            ))
        self._specializers = tuple(specializers)
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements=shared_lowering.buffer_elements,
        )
        for implementation, is_batched in ((shared, False), (batched, True)):
            bind_role = getattr(implementation, "bind_variant_role", None)
            if bind_role is not None:
                bind_role(batch_key, batched=is_batched)

    def specialize(
        self, arguments: dict[str, Any], dynamic: frozenset[str], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        trials = arguments[self.batch_key]
        selected = self.batched if trials is not None and trials > 1 else self.shared
        metadata = _metadata(selected)
        accepted = frozenset(metadata.parameters)
        selected_arguments = {
            name: value for name, value in arguments.items()
            if name in accepted or name == "BLOCK_SIZE"
        }
        dynamic_accepted = accepted & dynamic
        specializer = self._specializers[1 if selected is self.batched else 0]
        return specializer(
            selected_arguments, dynamic_accepted,
            buffer_dtypes={
                name: dtype for name, dtype in buffer_dtypes.items()
                if name in accepted
            },
        )

    def __call__(self, **kwargs: Any):
        del kwargs
        raise RuntimeError(
            "shared/batched variants may run only inside an explicit compiled "
            "substep; eager calls would reselect and filter the ABI every launch"
        )


def make_variant_dispatcher(
    shared: Callable,
    batched: Callable,
    *,
    batch_key: str = "num_trials",
    spec: KernelSpec | None = None,
) -> VariantDispatcher:
    """Create a variant pair from the enclosing registry's canonical Spec.

    Backend factories run under :func:`kernel_factory_contract`, so repeating
    ``spec=`` there creates a second source of ABI truth.  An explicit Spec is
    accepted only for standalone construction in tests or low-level tooling.
    """
    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(
                "make_variant_dispatcher may not repeat active KernelSpec metadata"
            )
        spec = active
    elif spec is None:
        raise TypeError(
            "make_variant_dispatcher requires a KernelSpec outside a "
            "BackendRegistry factory"
        )
    return VariantDispatcher(shared, batched, batch_key=batch_key, spec=spec)


# ── Triton dispatcher factory ─────────────────────────────────────────────

def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def make_triton_dispatcher(
    kernel,
    *,
    spec: KernelSpec | None = None,
    batched_kernel=None,
    batched_grid: str = "parallel",
) -> Callable:
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
    size_key = canonical.size_key
    if batched_grid not in {"parallel", "loop"}:
        raise ValueError(
            "batched_grid must be exactly 'parallel' or 'loop', got "
            f"{batched_grid!r}"
        )

    def specialize(
        arguments: dict[str, Any], dynamic: frozenset[str], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        canonical.validate_host_arguments(arguments)
        del buffer_dtypes
        try:
            bs = arguments["BLOCK_SIZE"]
        except KeyError as error:
            raise TypeError(
                f"{canonical.name}: compiler-owned BLOCK_SIZE was not bound"
            ) from error
        if type(bs) is not int or not 1 <= bs <= 1024:
            raise ValueError(
                "Triton BLOCK_SIZE must be an exact int in [1, 1024]"
            )
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
            if name in accepted and name not in dynamic
        }
        size_keys = (size_key,) if isinstance(size_key, str) else size_key
        static_n = None
        if not set(size_keys) & dynamic:
            static_n = 1
            for key in size_keys:
                static_n *= arguments[key]

        def launch(**values: Any):
            n = static_n
            if n is None:
                n = 1
                merged = static | values
                for key in size_keys:
                    n *= merged[key]
            if use_batched and batched_grid == "parallel":
                n *= trials
            grid = (_cdiv(n, bs),)
            selected[grid](BLOCK_SIZE=bs, **static, **values)

        return launch

    def dispatch(**kw):
        """One-shot eager call through the same initialization specializer."""
        reject_direct_kernel_launch(
            getattr(kernel, "__name__", "triton_kernel"),
        )
        return specialize(
            kw, frozenset(),
            buffer_dtypes={
                name: value.dtype for name, value in kw.items()
                if name in canonical.buffers and hasattr(value, "dtype")
            },
        )()
    kernel_parameters = tuple(dict.fromkeys(
        name for candidate in (kernel, batched_kernel)
        if candidate is not None
        for name in getattr(candidate, "arg_names", ())
        if name != "BLOCK_SIZE"
    ))
    if batched_kernel is not None and "num_trials" not in kernel_parameters:
        kernel_parameters = (*kernel_parameters, "num_trials")
    if set(kernel_parameters) != set(canonical.parameters):
        raise TypeError(
            f"{canonical.name}: Triton native parameters differ from "
            "KernelSpec: "
            f"missing={sorted(set(canonical.parameters) - set(kernel_parameters))}, "
            f"extra={sorted(set(kernel_parameters) - set(canonical.parameters))}"
        )
    lowering = BackendLoweringSpec.plan_specialized(
        buffer_elements="tensor",
    )
    dispatch.__hydroforge_kernel__ = canonical.metadata_for_lowering(lowering)
    dispatch.__hydroforge_lowering__ = lowering
    dispatch.specialize = specialize
    return dispatch


def make_triton_sequence_dispatcher(
    *,
    kernels: tuple[tuple[Any, str | tuple[str, ...]], ...],
    spec: KernelSpec | None = None,
) -> Callable:
    """Compose ordered native launches under one canonical logical ABI.

    Components consume exact-name subsets of the public ABI.  Their launch
    geometry and accepted dynamic values are specialized once, so the hot path
    is only the prebuilt sequence of native launches.
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
            make_triton_dispatcher(kernel, spec=component_spec)
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
        arguments: dict[str, Any], dynamic: frozenset[str], *,
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
            dynamic_selected = dynamic & accepted
            launch = component.specialize(
                selected, dynamic_selected,
                buffer_dtypes={
                    name: dtype for name, dtype in buffer_dtypes.items()
                    if name in accepted
                },
            )
            launches.append((launch, dynamic_selected))

        def run(**values: Any) -> None:
            for launch, accepted_dynamic in launches:
                launch(**{
                    key: value for key, value in values.items()
                    if key in accepted_dynamic
                })

        return run

    def dispatch(**values: Any) -> None:
        reject_direct_kernel_launch(spec.name)
        return specialize(
            values, frozenset(),
            buffer_dtypes={
                name: value.dtype for name, value in values.items()
                if name in spec.buffers and hasattr(value, "dtype")
            },
        )()

    lowering = BackendLoweringSpec.plan_specialized(
        buffer_elements="tensor",
    )
    dispatch.__hydroforge_kernel__ = spec.metadata_for_lowering(lowering)
    dispatch.__hydroforge_lowering__ = lowering
    dispatch.specialize = specialize
    return dispatch


def make_triton_program_dispatcher(
    prepare: Callable[..., Callable[..., None]],
    spec: KernelSpec | None = None,
) -> Callable:
    """Build one cached, ordered Triton program behind a canonical ABI.

    ``prepare`` is initialization/specialization work: it receives the stable
    argument mapping, the dynamic-name set, and the concrete buffer dtype ABI,
    and returns the hot-path launch callable.  This adapter is for physical
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
    signature = inspect.signature(prepare)
    if tuple(signature.parameters) != (
        "arguments", "dynamic", "buffer_dtypes",
    ):
        raise TypeError(
            f"{spec.name}: Triton program prepare signature must be exactly "
            "(arguments, dynamic, buffer_dtypes)"
        )

    def specialize(
        arguments: dict[str, Any], dynamic: frozenset[str], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        launch = prepare(arguments, dynamic, buffer_dtypes)
        if not callable(launch):
            raise TypeError(
                f"{spec.name}: Triton program prepare must return a callable"
            )
        return launch

    def dispatch(**values: Any) -> None:
        reject_direct_kernel_launch(spec.name)
        return specialize(
            values, frozenset(),
            buffer_dtypes={
                name: value.dtype for name, value in values.items()
                if name in spec.buffers and hasattr(value, "dtype")
            },
        )()

    lowering = BackendLoweringSpec.plan_specialized(buffer_elements="tensor")
    dispatch.__hydroforge_kernel__ = spec.metadata_for_lowering(lowering)
    dispatch.__hydroforge_lowering__ = lowering
    dispatch.specialize = specialize
    return dispatch


# Metal is a separate adapter; this import preserves the public factory.
from hydroforge.kernels.backends.metal.dispatcher import make_metal_dispatcher  # noqa: E402, F401

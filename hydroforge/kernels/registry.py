# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Kernel backend selection and registry for hydroforge.

Set ``HYDROFORGE_BACKEND`` to choose the backend explicitly::

    export HYDROFORGE_BACKEND=metal    # Metal shaders (Apple Silicon)
    export HYDROFORGE_BACKEND=triton   # Triton JIT kernels (NVIDIA/AMD)
    export HYDROFORGE_BACKEND=cuda     # Compiled CUDA extensions (NVIDIA, or
                                       # AMD/ROCm via PyTorch's automatic hipify)
    export HYDROFORGE_BACKEND=torch    # Formal pure-PyTorch backend

When unset, auto-detection picks the best available backend:
``triton`` → ``metal`` → ``torch``.  Triton covers both NVIDIA and AMD GPUs;
the ``cuda`` backend doubles as the AMD compiled path when explicitly selected.

Torch is an optional but formal backend: projects that register it must expose
the same exact :class:`KernelSpec` ABI as native backends.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable

from hydroforge.contracts import BufferDTypeABI, KernelMetadata, KernelSpec
from hydroforge.kernels.context import (
    active_operator_recorder, kernel_factory_contract, registry_factory,
    require_active_kernel_spec,
)
from hydroforge.kernels.devices import devices_match
from hydroforge.kernels.mutation import record_kernel_writes
from hydroforge.kernels.dispatcher import (
    TorchDispatcher, VariantDispatcher, make_metal_dispatcher,
    make_torch_dispatcher, make_triton_dispatcher,
    make_triton_program_dispatcher, make_triton_sequence_dispatcher,
    make_variant_dispatcher,
    require_specializer,
)
from hydroforge.kernels.backends.metal.template import make_spec_metal_dispatcher
from hydroforge.kernels.backends.cuda.template import make_spec_cuda_dispatcher

__all__ = [
    "BackendRegistry", "KERNEL_BACKEND", "KernelEntry", "StrictImplementation",
    "TorchDispatcher", "VariantDispatcher",
    "devices_match", "make_metal_dispatcher", "make_torch_dispatcher",
    "registry_factory", "require_active_kernel_spec",
    "make_spec_metal_dispatcher",
    "make_spec_cuda_dispatcher",
    "make_triton_dispatcher", "make_triton_program_dispatcher",
    "make_triton_sequence_dispatcher",
    "make_variant_dispatcher", "resolve_model_backend",
]


def _metadata(callable_: Callable) -> KernelMetadata | None:
    return getattr(callable_, "__hydroforge_kernel__", None)


_ACTIVE_AUTO_BINDER: ContextVar[Any | None] = ContextVar(
    "hydroforge_automatic_kernel_binder", default=None,
)


@contextmanager
def automatic_kernel_binding(binder: Any):
    """Complete omitted kernel arguments inside a compiled orchestration body."""

    token = _ACTIVE_AUTO_BINDER.set(binder)
    try:
        yield
    finally:
        _ACTIVE_AUTO_BINDER.reset(token)


def _resolve_backend() -> str:
    """Resolve kernel backend from HYDROFORGE_BACKEND environment variable.

    When the variable is unset, auto-detect in priority order:
    ``triton`` → ``metal`` → ``torch``.  Triton handles both NVIDIA and AMD
    GPUs, so AMD/ROCm users get Triton by default; the compiled ``cuda``
    backend (which PyTorch hipifies under ROCm) must be requested explicitly.
    """
    env = os.environ.get("HYDROFORGE_BACKEND", "").strip().lower()
    if env:
        supported = {"torch", "triton", "cuda", "metal"}
        if env not in supported:
            raise ValueError(
                "HYDROFORGE_BACKEND must be one of "
                f"{sorted(supported)}, got {env!r}"
            )
        return env
    import torch

    if torch.cuda.is_available():
        try:
            import triton  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "CUDA is available but Triton is not installed; set "
                "HYDROFORGE_BACKEND=cuda for compiled extensions or "
                "HYDROFORGE_BACKEND=torch for the formal Torch backend. "
                "HydroForge will not silently downgrade an accelerator model"
            ) from exc
        return "triton"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "metal"
    return "torch"


def resolve_model_backend(device: Any) -> str:
    """Resolve one model's backend from its declared device.

    An explicit ``HYDROFORGE_BACKEND`` remains authoritative.  In automatic
    mode the model device, rather than accelerator visibility elsewhere in the
    process, selects the backend.  This permits CPU and accelerator models to
    coexist without silently assigning a native GPU backend to CPU state.
    """

    env = os.environ.get("HYDROFORGE_BACKEND", "").strip().lower()
    if env:
        return _resolve_backend()
    import torch

    device_type = torch.device(device).type
    if device_type == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "a CUDA model requires Triton in automatic mode; set "
                "HYDROFORGE_BACKEND=cuda for compiled extensions or "
                "HYDROFORGE_BACKEND=torch for the formal Torch backend"
            ) from exc
        return "triton"
    if device_type == "mps":
        return "metal"
    return "torch"


KERNEL_BACKEND: str = _resolve_backend()


@dataclass(frozen=True)
class BackendRegistry:
    """Explicit lazy implementations of one logical kernel by backend."""

    implementations: dict[str, Callable[[], Callable]]
    name: str = "kernel"
    spec: KernelSpec | None = None

    def __post_init__(self) -> None:
        if self.spec is None:
            raise TypeError(f"{self.name}: BackendRegistry requires KernelSpec")
        if self.spec.name != self.name:
            raise ValueError(
                f"registry name {self.name!r} differs from KernelSpec "
                f"name {self.spec.name!r}"
            )

    @cached_property
    def selected(self) -> Callable:
        return KernelEntry(self)

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(self.implementations)

    def resolve(self, backend: str | None = None) -> Callable:
        """Build the implementation for ``backend`` or the active backend."""
        backend = KERNEL_BACKEND if backend is None else backend
        try:
            factory = self.implementations[backend]
        except KeyError as exc:
            raise ValueError(
                f"Backend {backend!r} is not registered for {self.name}; "
                f"available={self.available}"
            ) from exc
        with kernel_factory_contract(self.spec):
            implementation = factory()
        return StrictImplementation(implementation, self.spec, backend)

    def __call__(self, **kwargs):
        return self.selected(**kwargs)

    def __getitem__(self, grid):
        return self.selected[grid]


class KernelEntry:
    """A lazy registered operator recorded by an active compiled substep."""

    __slots__ = ("registry", "_implementations")

    def __init__(self, registry: BackendRegistry):
        self.registry = registry
        self._implementations: dict[str, Callable] = {}

    def _active_backend(self) -> str:
        recorder = active_operator_recorder()
        if recorder is not None:
            return recorder.execution.backend
        binder = _ACTIVE_AUTO_BINDER.get()
        if binder is not None:
            return binder.model._execution.backend
        return KERNEL_BACKEND

    def implementation(self, backend: str | None = None) -> Callable:
        """Return one backend implementation, constructed and checked once."""
        backend = self._active_backend() if backend is None else backend
        implementation = self._implementations.get(backend)
        if implementation is None:
            implementation = self.registry.resolve(backend)
            self._implementations[backend] = implementation
        return implementation

    @property
    def raw(self) -> Callable:
        return self.implementation()

    @property
    def metadata(self) -> KernelMetadata:
        # KernelSpec is the canonical public ABI. Merely inspecting or binding
        # an entry must not construct whichever backend happens to be active.
        return self.registry.spec.metadata

    def metadata_by_backend(self) -> dict[str, KernelMetadata]:
        result = {}
        for backend in self.registry.available:
            callable_ = self.registry.resolve(backend)
            metadata = _metadata(callable_)
            if metadata is None:
                raise TypeError(
                    f"{self.registry.name}: {backend} implementation has no "
                    "KernelMetadata"
                )
            result[backend] = metadata
        return result

    def __call__(self, **kwargs):
        recorder = active_operator_recorder()
        if recorder is not None:
            return recorder.record(self, kwargs)
        binder = _ACTIVE_AUTO_BINDER.get()
        if binder is not None:
            kwargs = binder.complete(self, kwargs)
            implementation = self.implementation()
            launch = implementation.specialize(
                kwargs,
                frozenset(),
                buffer_dtypes=binder.buffer_dtypes(self, kwargs),
            )
            return launch()
        return self.raw(**kwargs)

    def __getitem__(self, grid):
        return self.raw[grid]


class StrictImplementation:
    """Expose one backend implementation through the exact canonical ABI."""

    def __init__(self, implementation: Callable, spec: KernelSpec, backend: str):
        self.implementation = implementation
        self.spec = spec
        self.backend = backend
        metadata = _metadata(implementation)
        if metadata is None:
            raise TypeError(
                f"{spec.name}: {backend} factory returned an unregistered "
                "callable; build implementations with a HydroForge backend "
                "dispatcher so KernelMetadata and lowering semantics are explicit"
            )
        lowering = getattr(implementation, "__hydroforge_lowering__", None)
        if lowering is None:
            raise TypeError(
                f"{spec.name}: {backend} adapter has KernelMetadata but no "
                "BackendLoweringSpec"
            )
        spec.validate_native(backend, metadata, lowering)
        self.__hydroforge_kernel__ = spec.metadata
        self._specializer = require_specializer(
            implementation, label=f"{spec.name}: {backend} adapter",
        )

    def _validate_buffers(self, arguments: dict[str, Any]) -> None:
        import torch

        tensors: list[tuple[str, torch.Tensor]] = []
        for name in self.spec.buffers:
            value = arguments[name]
            if value is None and name in self.spec.optional_buffers:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{self.spec.name}.{name} must be a tensor, got "
                    f"{type(value).__name__}"
                )
            if value.layout is not torch.strided or not value.is_contiguous():
                raise ValueError(
                    f"{self.spec.name}.{name} must be a contiguous strided tensor"
                )
            tensors.append((name, value))
        required_device = {
            "cuda": "cuda", "triton": "cuda", "metal": "mps",
        }.get(self.backend)
        if required_device is not None:
            wrong = [
                f"{name}={tensor.device}"
                for name, tensor in tensors if tensor.device.type != required_device
            ]
            if wrong:
                raise ValueError(
                    f"{self.spec.name}: {self.backend} buffers must be on "
                    f"{required_device}; got {', '.join(wrong)}"
                )
        if tensors:
            reference = tensors[0][1].device
            mismatched = [
                f"{name}={tensor.device}" for name, tensor in tensors[1:]
                if not devices_match(tensor.device, reference)
            ]
            if mismatched:
                raise ValueError(
                    f"{self.spec.name}: buffers must share one device; "
                    f"expected {reference}, got {', '.join(mismatched)}"
                )

    def __call__(self, **kwargs: Any):
        supplied = set(kwargs).difference({"BLOCK_SIZE"})
        expected = set(self.spec.parameters)
        if supplied != expected:
            raise TypeError(
                f"{self.spec.name} ABI mismatch: "
                f"missing={sorted(expected - supplied)}, "
                f"extra={sorted(supplied - expected)}"
            )
        self.spec.validate_host_arguments(kwargs)
        self._validate_buffers(kwargs)
        record_kernel_writes(self.spec.metadata, kwargs)
        return self.implementation(**kwargs)

    def specialize(
        self, arguments: dict[str, Any], dynamic: frozenset[str], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        supplied = set(arguments).difference({"BLOCK_SIZE"})
        expected = set(self.spec.parameters)
        if supplied != expected:
            raise TypeError(
                f"{self.spec.name} specialization ABI mismatch: "
                f"missing={sorted(expected - supplied)}, "
                f"extra={sorted(supplied - expected)}"
            )
        dynamic_constants = dynamic.intersection(self.spec.compile_time)
        if dynamic_constants:
            raise TypeError(
                f"{self.spec.name}: compile-time parameters may not be dynamic: "
                f"{sorted(dynamic_constants)}"
            )
        dynamic_scalars = dynamic.intersection(self.spec.runtime_scalars)
        if dynamic_scalars:
            raise TypeError(
                f"{self.spec.name}: runtime host scalars may not be dynamic; "
                "copy changing inputs into stable tensors or include host "
                f"values in substep specialization: {sorted(dynamic_scalars)}"
            )
        self.spec.validate_host_arguments(arguments)
        self._validate_buffers(arguments)
        expected_buffers = set(self.spec.buffers)
        if set(buffer_dtypes) != expected_buffers:
            raise TypeError(
                f"{self.spec.name}: specialization buffer ABI mismatch: "
                f"missing={sorted(expected_buffers - set(buffer_dtypes))}, "
                f"extra={sorted(set(buffer_dtypes) - expected_buffers)}"
            )
        import torch
        for name, dtype in buffer_dtypes.items():
            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    f"{self.spec.name}.{name} buffer dtype must be torch.dtype"
                )
            value = arguments[name]
            if isinstance(value, torch.Tensor) and value.dtype != dtype:
                raise TypeError(
                    f"{self.spec.name}.{name} specialization declares {dtype}, "
                    f"but the tensor has dtype {value.dtype}"
                )
        launch = self._specializer(
            arguments, dynamic, buffer_dtypes=buffer_dtypes,
        )
        if not callable(launch):
            raise TypeError(
                f"{self.spec.name}: {self.backend} specialize() must return "
                f"a callable launch, got {type(launch).__name__}; Python "
                "launch fallback is forbidden"
            )
        return launch

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Kernel backend selection and registry for hydroforge.

Set ``HYDROFORGE_BACKEND`` to choose the backend explicitly::

    export HYDROFORGE_BACKEND=metal    # Metal shaders (Apple Silicon)
    export HYDROFORGE_BACKEND=triton   # Triton JIT kernels (NVIDIA/AMD/Intel)
    export HYDROFORGE_BACKEND=cuda     # Compiled CUDA extensions (NVIDIA, or
                                       # AMD/ROCm via PyTorch's automatic hipify)
    export HYDROFORGE_BACKEND=torch    # Formal pure-PyTorch backend

When unset, auto-detection picks the best available backend:
``triton`` → ``metal`` → ``torch``.  Triton covers NVIDIA and AMD GPUs
through its in-tree backends and Intel XPU when an Intel Triton backend is
installed.  The ``cuda`` backend doubles as the AMD compiled path when
explicitly selected.

Torch is an optional but formal backend: projects that register it must expose
the same exact :class:`KernelSpec` ABI as native backends.
"""

import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from typing import Any, Self

import torch
from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.kernels import (
    BackendLoweringSpec, BufferDTypeABI, KernelMetadata, KernelSpec,
)
from hydroforge.contracts.validation import HydroForgeModel, _immutable_dict
from hydroforge.kernels.context import (
    active_operator_recorder, kernel_factory_contract, registry_factory,
)
from hydroforge.kernels.devices import devices_match
from hydroforge.kernels.dispatcher import (
    make_metal_dispatcher, make_torch_dispatcher, make_triton_dispatcher,
    make_triton_program_dispatcher, make_triton_sequence_dispatcher,
)
from hydroforge.kernels.backends.metal.template import make_spec_metal_dispatcher
from hydroforge.kernels.backends.cuda.template import make_spec_cuda_dispatcher

__all__ = [
    "BackendRegistry",
    "devices_match", "make_metal_dispatcher", "make_torch_dispatcher",
    "registry_factory",
    "make_spec_metal_dispatcher",
    "make_spec_cuda_dispatcher",
    "make_triton_dispatcher", "make_triton_program_dispatcher",
    "make_triton_sequence_dispatcher", "resolve_model_backend",
]


_ACTIVE_AUTO_BINDER: ContextVar[Any | None] = ContextVar(
    "hydroforge_automatic_kernel_binder", default=None,
)

_BACKEND_DEVICE_TYPES: Mapping[str, tuple[str, ...]] = {
    "cuda": ("cuda",),
    "triton": ("cuda", "xpu"),
    "metal": ("mps",),
}

_TRITON_DEVICE_REGISTRATION_HINTS: Mapping[str, tuple[str, ...]] = {
    # ROCm intentionally uses PyTorch's ``cuda`` device spelling.
    "cuda": ("nvidia", "amd"),
    # Intel's Triton fork registers the backend as ``intel``. Accept ``xpu``
    # as well so an out-of-tree plugin can use the PyTorch device spelling.
    "xpu": ("intel", "xpu"),
}

_TRITON_DEVICE_TARGETS: Mapping[str, tuple[str, ...]] = {
    "cuda": ("cuda", "hip"),
    "xpu": ("xpu", "intel"),
}


def _backend_device_types(backend: str) -> tuple[str, ...] | None:
    """Return the physical device types accepted by one native backend."""

    return _BACKEND_DEVICE_TYPES.get(backend)


def _installed_triton_backends() -> frozenset[str]:
    """Discover in-tree and entry-point Triton backends without activating one."""

    try:
        from triton.backends import backends
    except ImportError:
        return frozenset()
    return frozenset(str(name).strip().lower() for name in backends)


class _TritonDriverSelectionError(RuntimeError):
    """Triton exposed more than one driver matching the requested device."""


def _inspect_triton_driver(active: Any) -> tuple[str, torch.device]:
    """Return one compiler-proven target and Torch device for ``active``."""

    from triton.compiler.compiler import make_backend

    target = active.get_current_target()
    make_backend(target)
    target_backend = str(getattr(target, "backend", "")).strip().lower()
    active_device = torch.device(active.get_active_torch_device())
    return target_backend, active_device


def _matching_triton_drivers(
    device: torch.device,
) -> tuple[tuple[str, Any, str, torch.device], ...]:
    """Construct every active Triton driver that matches ``device`` exactly."""

    from triton.backends import backends

    expected_targets = _TRITON_DEVICE_TARGETS[device.type]
    candidates: list[tuple[str, Any, str, torch.device]] = []
    seen_driver_types: set[type[Any]] = set()
    for registration, backend in backends.items():
        driver_type = getattr(backend, "driver", None)
        if driver_type is None or driver_type in seen_driver_types:
            continue
        seen_driver_types.add(driver_type)
        try:
            if not driver_type.is_active():
                continue
            active = driver_type()
            target_backend, active_device = _inspect_triton_driver(active)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            continue
        if (
            target_backend in expected_targets
            and devices_match(active_device, device)
        ):
            candidates.append((
                str(registration), active, target_backend, active_device,
            ))
    return tuple(candidates)


def _active_triton_runtime(
    device: torch.device,
) -> tuple[str, torch.device]:
    """Select and return the Triton runtime matching one known model device.

    Triton's default ``DriverConfig.active`` requires exactly one globally
    active driver. A process may legitimately expose both CUDA and Intel XPU,
    however, so use the model device to select the sole matching driver through
    ``DriverConfig.set_active``. An already explicit matching selection remains
    authoritative and avoids enumerating other drivers.
    """

    from triton.runtime import driver

    original_error: BaseException | None = None
    current: tuple[str, torch.device] | None = None
    try:
        current = _inspect_triton_driver(driver.active)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as error:
        original_error = error
    else:
        expected_targets = _TRITON_DEVICE_TARGETS[device.type]
        if (
            current[0] in expected_targets
            and devices_match(current[1], device)
        ):
            return current

    candidates = _matching_triton_drivers(device)
    if len(candidates) == 1:
        _registration, active, target_backend, active_device = candidates[0]
        driver.set_active(active)
        return target_backend, active_device
    if len(candidates) > 1:
        descriptions = [
            f"{registration}:{target_backend}@{active_device}"
            for registration, _active, target_backend, active_device
            in candidates
        ]
        reason = (
            "none"
            if original_error is None
            else f"{type(original_error).__name__}: {original_error}"
        )
        raise _TritonDriverSelectionError(
            f"Triton exposes multiple active drivers matching model device "
            f"{str(device)!r}: {descriptions!r}; original driver selection "
            f"reason={reason}. Hide non-target accelerators or explicitly call "
            "triton.runtime.driver.set_active(...) with the intended driver "
            "before constructing the model"
        ) from original_error
    if original_error is not None:
        raise original_error
    assert current is not None
    return current


def _require_triton_device_backend(device: torch.device) -> None:
    """Prove Triton's registered and active runtime match ``device``."""

    device_type = device.type
    if device_type in {"xla", "lazy"}:
        raise RuntimeError(
            "TPU/XLA tensors do not implement the device-pointer ABI used by "
            "HydroForge Triton kernels; use a dedicated torch_xla/PJRT "
            "adapter instead"
        )
    registration_hints = _TRITON_DEVICE_REGISTRATION_HINTS.get(device_type)
    if registration_hints is None:
        raise ValueError(
            "HydroForge Triton kernels require a CUDA/ROCm or XPU model "
            f"device, got {device_type!r}"
        )
    try:
        target_backend, active_device = _active_triton_runtime(device)
    except _TritonDriverSelectionError as error:
        raise RuntimeError(str(error)) from error
    except (
        AttributeError,
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        installed = _installed_triton_backends()
        discovered = sorted(installed) or ["none"]
        reason = f"{type(error).__name__}: {error}"
        matching_registration = bool(installed.intersection(
            registration_hints,
        ))
        if matching_registration:
            advice = (
                "hide non-target accelerators or explicitly select the matching "
                "Triton driver with triton.runtime.driver.set_active(...); "
                "otherwise select a non-Triton backend explicitly"
            )
        elif device_type == "xpu":
            advice = (
                "install an Intel XPU build/plugin for Triton, or explicitly "
                "set HYDROFORGE_BACKEND=torch"
            )
        else:
            advice = (
                "install Triton with an NVIDIA or AMD backend, or explicitly "
                "set HYDROFORGE_BACKEND=cuda or HYDROFORGE_BACKEND=torch"
            )
        raise RuntimeError(
            f"Triton has no usable active driver/compiler target for "
            f"{device_type!r}; registered backends={discovered!r}; common "
            f"registrations are {list(registration_hints)!r}; original "
            f"reason={reason}; {advice}"
        ) from error

    expected_targets = _TRITON_DEVICE_TARGETS[device_type]
    if target_backend not in expected_targets:
        raise RuntimeError(
            f"active Triton target {target_backend!r} does not match model "
            f"device {str(device)!r}; expected one of {list(expected_targets)!r}"
        )
    if not devices_match(active_device, device):
        raise RuntimeError(
            f"active Triton torch device {str(active_device)!r} does not match "
            f"model device {str(device)!r}; select the process-local device "
            "before constructing the model"
        )


@contextmanager
def automatic_kernel_binding(binder: Any):
    """Complete omitted kernel arguments inside a compiled orchestration body."""

    token = _ACTIVE_AUTO_BINDER.set(binder)
    try:
        yield
    finally:
        _ACTIVE_AUTO_BINDER.reset(token)


def _configured_backend() -> str | None:
    """Return and validate the explicitly configured model backend."""

    env = os.environ.get("HYDROFORGE_BACKEND", "").strip().lower()
    supported = {"torch", "triton", "cuda", "metal"}
    if env and env not in supported:
        raise ValueError(
            "HYDROFORGE_BACKEND must be one of "
            f"{sorted(supported)}, got {env!r}"
        )
    return env or None


class _ModelBackendRequest(HydroForgeModel):
    device: Any

    _backend: str = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self):
        import torch

        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device")
        self._backend = _resolve_model_backend_trusted(self.device)
        return self

    @property
    def backend(self) -> str:
        return self._backend


def resolve_model_backend(device: Any) -> str:
    """Resolve one model's backend from its declared device.

    An explicit ``HYDROFORGE_BACKEND`` remains authoritative.  In automatic
    mode the model device, rather than accelerator visibility elsewhere in the
    process, selects the backend.  This permits CPU and accelerator models to
    coexist without silently assigning a native GPU backend to CPU state.
    """

    return _ModelBackendRequest(device=device).backend


def _resolve_model_backend_trusted(device: Any) -> str:
    """Resolve a backend from an already validated torch device."""

    device_type = device.type
    if device_type in {"xla", "lazy"}:
        _require_triton_device_backend(device)
    configured = _configured_backend()
    if configured is not None:
        if configured == "triton":
            _require_triton_device_backend(device)
        required_devices = _backend_device_types(configured)
        if (
            required_devices is not None
            and device_type not in required_devices
        ):
            required_label = " or ".join(
                repr(item) for item in required_devices
            )
            raise ValueError(
                f"HydroForge backend {configured!r} requires a "
                f"{required_label} model device, got {str(device)!r}"
            )
        return configured
    if device_type in {"cuda", "xpu"}:
        _require_triton_device_backend(device)
        return "triton"
    if device_type == "mps":
        return "metal"
    return "torch"


class _KernelInvocationRequest(HydroForgeModel):
    """Validate the caller-supplied portion of one canonical kernel ABI."""

    spec: KernelSpec
    arguments: Mapping[str, Any]

    @model_validator(mode="after")
    def _validate_arguments(self) -> Self:
        if not isinstance(self.arguments, Mapping):
            raise ValueError("kernel arguments must be a mapping")
        supplied = dict(self.arguments)
        if any(type(name) is not str or not name for name in supplied):
            raise ValueError("kernel argument names must be non-empty strings")
        unknown = set(supplied).difference(self.spec.parameters)
        if unknown:
            raise ValueError(
                f"{self.spec.name} received arguments outside its KernelSpec: "
                f"{sorted(unknown)}"
            )
        if "BLOCK_SIZE" in supplied:
            raise ValueError(
                f"{self.spec.name}.BLOCK_SIZE is compiler-owned; configure "
                "model.BLOCK_SIZE instead"
            )
        if (
            active_operator_recorder() is None
            and _ACTIVE_AUTO_BINDER.get() is None
        ):
            raise ValueError(
                f"{self.spec.name} may be called only while HydroForge records "
                "or executes a validated model step"
            )
        object.__setattr__(self, "arguments", _immutable_dict(supplied))
        return self


class BackendRegistry(HydroForgeModel):
    """Explicit lazy implementations of one logical kernel by backend."""

    implementations: Mapping[str, Callable[[], Any]]
    name: str = "kernel"
    spec: KernelSpec

    @model_validator(mode="after")
    def _validate_registry(self) -> Self:
        if self.spec.name != self.name:
            raise ValueError(
                f"registry name {self.name!r} differs from KernelSpec "
                f"name {self.spec.name!r}"
            )
        object.__setattr__(
            self,
            "implementations",
            _immutable_dict(self.implementations),
        )
        return self

    @cached_property
    def selected(self) -> Callable:
        return KernelEntry(self)

    @property
    def _available(self) -> tuple[str, ...]:
        return tuple(self.implementations)

    def resolve(self, backend: str, *, precision: str | None = None) -> Any:
        """Build the implementation for one explicit model backend."""
        request = _BackendResolutionRequest(
            registry=self,
            backend=backend,
            precision=precision,
        )
        return request.materialize()

    def __call__(self, **kwargs: Any):
        request = _KernelInvocationRequest(
            spec=self.spec,
            arguments=kwargs,
        )
        return self.selected._invoke_trusted(dict(request.arguments))

class KernelEntry(HydroForgeModel):
    """A lazy registered operator recorded by an active compiled substep."""

    _registry: BackendRegistry = PrivateAttr()
    _implementations: dict[tuple[str, str | None], Any] = PrivateAttr(
        default_factory=dict,
    )

    def __init__(self, registry: BackendRegistry):
        super().__init__()
        self._registry = registry

    def _implementation(
        self, backend: str, *, precision: str | None = None,
    ) -> Any:
        """Return one backend implementation, constructed and checked once."""
        precision_key = (
            precision if self._registry.spec._uses_precision else None
        )
        key = (backend, precision_key)
        implementation = self._implementations.get(key)
        if implementation is None:
            implementation = self._registry.resolve(
                backend, precision=precision_key,
            )
            self._implementations[key] = implementation
        return implementation

    @property
    def _spec(self) -> KernelSpec:
        return self._registry.spec

    @property
    def metadata(self) -> KernelMetadata:
        # KernelSpec is the canonical public ABI. Merely inspecting or binding
        # an entry must not construct whichever backend happens to be active.
        return self._registry.spec._canonical_metadata

    def __call__(self, **kwargs: Any):
        request = _KernelInvocationRequest(
            spec=self._registry.spec,
            arguments=kwargs,
        )
        return self._invoke_trusted(dict(request.arguments))

    def _invoke_trusted(self, kwargs: dict[str, Any]):
        recorder = active_operator_recorder()
        if recorder is not None:
            return recorder.record(self, kwargs)
        binder = _ACTIVE_AUTO_BINDER.get()
        binding = binder.bind(self, kwargs)
        kwargs = dict(binding.arguments)
        implementation = self._implementation(
            binder.model._execution.backend,
            precision=getattr(binder.model, "precision", None),
        )
        launch = implementation.specialize(
            kwargs,
            buffer_dtypes=binding.buffer_dtypes,
        )
        return launch()

class StrictImplementation(HydroForgeModel):
    """Trusted backend implementation built by a validated adapter factory."""

    spec: KernelSpec
    backend: str

    _implementation: Any = PrivateAttr()
    _specializer: Callable = PrivateAttr()

    @classmethod
    def _from_validated(
        cls, request: "_ResolvedImplementationRequest",
    ) -> "StrictImplementation":
        result = cls(spec=request.spec, backend=request.backend)
        result._implementation = request.implementation
        result._specializer = request.specializer
        result.__hydroforge_kernel__ = request.spec._canonical_metadata
        return result

    def __call__(self, **arguments: Any):
        """Validate, compile and execute one explicit backend invocation."""

        buffer_dtypes = {
            name: getattr(arguments.get(name), "dtype", None)
            for name in self.spec.buffers
        }
        return self.specialize(
            arguments,
            buffer_dtypes=buffer_dtypes,
        )()

    def _compile_trusted(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        return self._specializer(
            arguments, buffer_dtypes=buffer_dtypes,
        )

    def specialize(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        """Validate one public kernel call and return its compiled launch."""

        request = _KernelSpecializationRequest(
            implementation=self,
            arguments=arguments,
            buffer_dtypes=buffer_dtypes,
        )
        return request.materialize()


class _ResolvedImplementationRequest(HydroForgeModel):
    """Validate one lazy backend factory result before it becomes trusted."""

    implementation: Any
    spec: KernelSpec
    backend: str

    _specializer: Callable = PrivateAttr()

    @model_validator(mode="after")
    def _validate_implementation(self) -> Self:
        metadata = getattr(
            self.implementation, "__hydroforge_kernel__", None,
        )
        lowering = getattr(
            self.implementation, "__hydroforge_lowering__", None,
        )
        specializer = getattr(self.implementation, "specialize", None)
        if not isinstance(metadata, KernelMetadata):
            raise ValueError(
                f"{self.spec.name}: {self.backend} factory must return a "
                "validated HydroForge dispatcher with KernelMetadata"
            )
        if not isinstance(lowering, BackendLoweringSpec):
            raise ValueError(
                f"{self.spec.name}: {self.backend} factory must return a "
                "dispatcher with BackendLoweringSpec"
            )
        if not callable(specializer):
            raise ValueError(
                f"{self.spec.name}: {self.backend} dispatcher must define "
                "a callable specialize()"
            )
        try:
            self.spec._validate_native(self.backend, metadata, lowering)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        self._specializer = specializer
        return self

    @property
    def specializer(self) -> Callable:
        return self._specializer

    def materialize(self) -> StrictImplementation:
        return StrictImplementation._from_validated(self)


class _BackendResolutionRequest(HydroForgeModel):
    """One validated public backend-selection request."""

    registry: BackendRegistry
    backend: str
    precision: str | None = None

    _spec: KernelSpec = PrivateAttr()

    @model_validator(mode="after")
    def _validate_resolution(self):
        if self.backend not in self.registry.implementations:
            raise ValueError(
                f"Backend {self.backend!r} is not registered for "
                f"{self.registry.name}; available={self.registry._available}"
            )
        self._spec = self.registry.spec._resolve_precision(self.precision)
        return self

    def materialize(self) -> StrictImplementation:
        """Build one implementation after semantic request validation."""

        factory = self.registry.implementations[self.backend]
        with kernel_factory_contract(self._spec):
            implementation = factory()
        return _ResolvedImplementationRequest(
            implementation=implementation,
            spec=self._spec,
            backend=self.backend,
        ).materialize()


class _KernelSpecializationRequest(HydroForgeModel):
    """One complete canonical kernel call validated before compilation."""

    implementation: StrictImplementation
    arguments: Mapping[str, Any]
    buffer_dtypes: Mapping[str, Any]

    @model_validator(mode="after")
    def _validate_specialization_request(self):
        implementation = self.implementation
        spec = implementation.spec
        arguments = dict(self.arguments)
        buffer_dtypes = self.buffer_dtypes
        try:
            supplied = set(arguments).difference({"BLOCK_SIZE"})
            expected = set(spec.parameters)
            if supplied != expected:
                raise ValueError(
                    f"{spec.name} specialization ABI mismatch: "
                    f"missing={sorted(expected - supplied)}, "
                    f"extra={sorted(supplied - expected)}"
                )
            spec._validate_host_arguments(arguments)

            tensors: list[tuple[str, torch.Tensor]] = []
            for name in spec.buffers:
                value = arguments[name]
                if value is None and name in spec.optional_buffers:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"{spec.name}.{name} must be a tensor, got "
                        f"{type(value).__name__}"
                    )
                if (
                    value.layout is not torch.strided
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"{spec.name}.{name} must be a contiguous strided tensor"
                    )
                tensors.append((name, value))

            required_devices = _backend_device_types(implementation.backend)
            if required_devices is not None:
                wrong = [
                    f"{name}={tensor.device}"
                    for name, tensor in tensors
                    if tensor.device.type not in required_devices
                ]
                if wrong:
                    required_label = (
                        required_devices[0]
                        if len(required_devices) == 1
                        else "one of " + "/".join(required_devices)
                    )
                    raise ValueError(
                        f"{spec.name}: {implementation.backend} buffers must "
                        f"be on {required_label}; got {', '.join(wrong)}"
                    )
            if tensors:
                reference = tensors[0][1].device
                mismatched = [
                    f"{name}={tensor.device}"
                    for name, tensor in tensors[1:]
                    if not devices_match(tensor.device, reference)
                ]
                if mismatched:
                    raise ValueError(
                        f"{spec.name}: buffers must share one device; expected "
                        f"{reference}, got {', '.join(mismatched)}"
                    )
                if implementation.backend == "triton":
                    _require_triton_device_backend(reference)

            expected_buffers = set(spec.buffers)
            if set(buffer_dtypes) != expected_buffers:
                raise ValueError(
                    f"{spec.name}: specialization buffer ABI mismatch: "
                    f"missing={sorted(expected_buffers - set(buffer_dtypes))}, "
                    f"extra={sorted(set(buffer_dtypes) - expected_buffers)}"
                )
            for name, dtype in buffer_dtypes.items():
                value = arguments[name]
                if dtype is None:
                    if value is None and name in spec.optional_buffers:
                        continue
                    raise ValueError(
                        f"{spec.name}.{name} buffer dtype must be torch.dtype"
                    )
                if not isinstance(dtype, torch.dtype):
                    raise ValueError(
                        f"{spec.name}.{name} buffer dtype must be torch.dtype"
                    )
                if isinstance(value, torch.Tensor) and value.dtype != dtype:
                    raise ValueError(
                        f"{spec.name}.{name} specialization declares {dtype}, "
                        f"but the tensor has dtype {value.dtype}"
                    )
            backend_validator = getattr(
                implementation._implementation,
                "_validate_specialization_input",
                None,
            )
            if backend_validator is not None:
                backend_validator(
                    arguments, buffer_dtypes=buffer_dtypes,
                )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        object.__setattr__(
            self,
            "arguments",
            _immutable_dict(self.arguments),
        )
        object.__setattr__(
            self,
            "buffer_dtypes",
            _immutable_dict(self.buffer_dtypes),
        )
        return self

    def materialize(self) -> Callable:
        """Compile a zero-argument launch after semantic validation."""

        launch = self.implementation._compile_trusted(
            dict(self.arguments),
            buffer_dtypes=self.buffer_dtypes,
        )
        return launch

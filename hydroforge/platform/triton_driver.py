# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Triton driver discovery and validation for a selected model device."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import Any

import torch

from hydroforge.core.devices import devices_match

_DRIVER_LOCK = RLock()


_TRITON_DEVICE_REGISTRATION_HINTS: Mapping[str, tuple[str, ...]] = {
    "cpu": ("cpu",),
    # ROCm intentionally uses PyTorch's ``cuda`` device spelling.
    "cuda": ("nvidia", "amd"),
    # Intel's Triton fork registers the backend as ``intel``. Accept ``xpu``
    # as well so an out-of-tree plugin can use the PyTorch device spelling.
    "xpu": ("intel", "xpu"),
}

_TRITON_DEVICE_TARGETS: Mapping[str, tuple[str, ...]] = {
    "cpu": ("cpu",),
    "cuda": ("cuda", "hip"),
    "xpu": ("xpu", "intel"),
}


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
        if target_backend in expected_targets and devices_match(active_device, device):
            candidates.append(
                (
                    str(registration),
                    active,
                    target_backend,
                    active_device,
                )
            )
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
        if current[0] in expected_targets and devices_match(current[1], device):
            return current

    candidates = _matching_triton_drivers(device)
    if len(candidates) == 1:
        _registration, active, target_backend, active_device = candidates[0]
        driver.set_active(active)
        return target_backend, active_device
    if len(candidates) > 1:
        descriptions = [
            f"{registration}:{target_backend}@{active_device}"
            for registration, _active, target_backend, active_device in candidates
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


def _require_triton_device(device: torch.device) -> Any:
    """Prove Triton's registered and active runtime match ``device``.

    Returns the proven active driver.
    """

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
            "HydroForge Triton kernels require a CPU, CUDA/ROCm or XPU model "
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
        matching_registration = bool(
            installed.intersection(
                registration_hints,
            )
        )
        if matching_registration:
            advice = (
                "hide non-target accelerators or explicitly select the matching "
                "Triton driver with triton.runtime.driver.set_active(...); "
                "otherwise select a non-Triton backend explicitly"
            )
        elif device_type == "cpu":
            advice = (
                "install an official Triton CPU build, or explicitly "
                "set HYDROFORGE_BACKEND=torch"
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
    from triton.runtime import driver

    return driver.active


def require_triton_device(device: torch.device) -> Any:
    """Prove a matching driver while serializing HydroForge driver selection."""
    with _DRIVER_LOCK:
        return _require_triton_device(device)


class ProvenTritonDevice:
    """A device proof used during binding and compilation.

    CPU storage has no selectable index. Driver selection is process-global,
    so HydroForge serializes selection and compilation. Bound launches do not
    enter this context or recheck the driver; their caller owns the active
    driver/device for execution. Rebind after changing that execution context.
    """

    __slots__ = ("device", "_config", "_driver", "_current", "_select")

    def __init__(self, device: torch.device) -> None:
        from triton.runtime import driver

        self.device = torch.device("cpu") if device.type == "cpu" else device
        self._config = driver
        if device.type == "cpu":
            self._current = lambda: None
            self._select = lambda _index: nullcontext()
        else:
            runtime = getattr(torch, device.type)
            self._current = runtime.current_device
            self._select = runtime.device
        with _DRIVER_LOCK, self._select(self.device.index):
            self._driver = _require_triton_device(self.device)

    @contextmanager
    def active(self):
        """Keep device selection, launch binding and compilation on one target."""
        with _DRIVER_LOCK:
            if (
                self._config.active is self._driver
                and self._current() == self.device.index
            ):
                yield
                return
            with self._select(self.device.index):
                if self._config.active is not self._driver:
                    self._driver = _require_triton_device(self.device)
                yield


_PROVEN_TRITON_DEVICES: dict[tuple[str, int | None], ProvenTritonDevice] = {}


def proven_triton_device(device: torch.device) -> ProvenTritonDevice:
    """Return a cached proof, canonicalizing the singleton CPU device."""
    key = (device.type, None if device.type == "cpu" else device.index)
    with _DRIVER_LOCK:
        proven = _PROVEN_TRITON_DEVICES.get(key)
        if proven is None:
            proven = _PROVEN_TRITON_DEVICES[key] = ProvenTritonDevice(device)
        return proven


def triton_call_device(
    values: Iterable[Any], *, device: torch.device | None = None
) -> torch.device:
    """Resolve a native call's tensor device before exposing any pointers.

    A bound device also supports calls without buffers. Standalone bufferless
    calls retain Triton's explicitly selected current device as their default.
    """
    pending = list(values)
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, (tuple, list, Mapping)):
            if id(value) not in seen:
                seen.add(id(value))
                pending.extend(value.values() if isinstance(value, Mapping) else value)
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if device is None:
            device = value.device
        elif not devices_match(value.device, device):
            raise ValueError(
                f"Triton buffers must share the bound device {device}; "
                f"got {value.device}"
            )
    if device is None:
        from triton.runtime import driver

        device = torch.device(driver.active.get_active_torch_device())
    return device

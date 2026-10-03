"""The four kernel backends and the selection of one for a model device.

A :class:`Backend` states everything HydroForge relies on about a backend:
the torch devices it runs on, the toolchain that compiles its kernels, the
dialect of framework-generated code, its capture executor, launch widths and
extents, the scalar kinds and precisions it represents, and the math policy.
Code that differs by backend reads these facts instead of comparing names.

``HYDROFORGE_BACKEND`` selects a backend explicitly::

    export HYDROFORGE_BACKEND=metal    # Metal shaders (Apple Silicon)
    export HYDROFORGE_BACKEND=triton   # Triton JIT kernels (NVIDIA/AMD/Intel)
    export HYDROFORGE_BACKEND=cuda     # NVRTC / HIPRTC device kernels
    export HYDROFORGE_BACKEND=torch    # Formal pure-PyTorch backend

When unset, the model device selects Triton for CUDA/ROCm and XPU, Metal for
MPS, and Torch otherwise. Triton selection requires a usable matching driver
and compiler; another backend must be selected explicitly if it is
unavailable. The ``cuda`` backend also supports AMD/ROCm.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Literal

import torch
from pydantic import validate_call

from hydroforge.core.validation import HydroForgeModel
from hydroforge.platform import env
from hydroforge.platform.devices import float64_supported
from hydroforge.platform.triton_driver import require_triton_device

BackendName = Literal["torch", "triton", "cuda", "metal"]
Dialect = Literal["torch", "triton", "cuda", "msl"]
Toolchain = Literal["python", "triton", "rtc", "metal"]
Capture = Literal["cuda_graph", "metal_icb"]
Precision = Literal["float32", "float64"]
ScalarKind = Literal["bool", "int32", "uint32", "index", "float32", "float64"]

_ALL_SCALARS: frozenset[ScalarKind] = frozenset(
    ("bool", "int32", "uint32", "index", "float32", "float64")
)


@dataclass(frozen=True, slots=True)
class BlockWidth:
    """Launch-width rule of one backend runtime."""

    default: int = 256
    fixed: int | None = None
    power_of_two: bool = False
    limit: int = 1024

    def validate(self, value: int, *, backend: str) -> int:
        if type(value) is not int or not 1 <= value <= self.limit:
            raise ValueError(
                f"backend {backend!r} BLOCK_SIZE must be an exact int in "
                f"[1, {self.limit}], got {value!r}"
            )
        if self.power_of_two and value & (value - 1):
            raise ValueError(
                f"backend {backend!r} BLOCK_SIZE must be a power of two, got {value}"
            )
        if self.fixed is not None and value != self.fixed:
            raise ValueError(
                f"backend {backend!r} launches a fixed BLOCK_SIZE={self.fixed}, "
                f"got {value}"
            )
        return value

    def resolve(
        self, configured: int | None, *, kernel: int | None = None, backend: str
    ) -> int:
        """Choose fixed, then model-configured, then kernel, then default."""

        if self.fixed is not None:
            value = self.fixed
        elif configured is not None:
            value = configured
        elif kernel is not None:
            value = kernel
        else:
            value = self.default
        return self.validate(value, backend=backend)


@dataclass(frozen=True, slots=True)
class MathPolicy:
    """Process-wide numerical policy of physics kernels."""

    physics_fast_math: bool


@cache
def _math_policy() -> MathPolicy:
    return MathPolicy(physics_fast_math=env.flag(env.FAST_MATH, default=False))


@dataclass(frozen=True, slots=True)
class Backend:
    """Immutable facts of one kernel backend.

    ``devices`` is ``None`` for a backend that runs on any torch device.
    ``max_extent`` bounds a flat launch extent including ensemble members
    and the block remainder.  ``mixed_precision_default`` lists the device
    types whose models store ``hpfloat`` state in FP64 unless configured.
    """

    name: BackendName
    devices: frozenset[str] | None
    toolchain: Toolchain
    dialect: Dialect
    capture: Capture | None
    block: BlockWidth
    max_extent: int | None
    scalars: frozenset[ScalarKind]
    precisions: frozenset[Precision]
    mixed_precision: bool
    mixed_precision_default: frozenset[str]

    @property
    def math(self) -> MathPolicy:
        """Read once per process from ``HYDROFORGE_FAST_MATH``."""

        return _math_policy()

    def accepts(self, device: torch.device) -> bool:
        return self.devices is None or device.type in self.devices

    def require_device(self, device: torch.device) -> None:
        if not self.accepts(device):
            label = " or ".join(repr(item) for item in sorted(self.devices))
            raise ValueError(
                f"HydroForge backend {self.name!r} requires a {label} model "
                f"device, got {str(device)!r}"
            )

    def default_mixed_precision(self, device: torch.device) -> bool:
        """Whether a model on ``device`` defaults to mixed precision."""

        if device.type not in self.mixed_precision_default:
            return False
        return device.type != "xpu" or float64_supported(device)

    def validate_precision(self, precision: Precision, mixed: bool) -> None:
        if precision not in self.precisions:
            raise ValueError(
                f"backend {self.name!r} requires precision in "
                f"{sorted(self.precisions)}, got {precision!r}"
            )
        if mixed and not self.mixed_precision:
            raise ValueError(f"backend {self.name!r} does not support mixed precision")

    def validate_scalars(self, kernel: str, kinds: Mapping[str, str]) -> None:
        """Reject scalar kinds this backend cannot represent exactly."""

        unsupported = {
            name: kind for name, kind in kinds.items() if kind not in self.scalars
        }
        if unsupported:
            raise TypeError(
                f"{kernel}: backend {self.name!r} has no exact representation "
                f"of scalar kinds {unsupported}; it supports "
                f"{sorted(self.scalars)}"
            )

    def validate_extent(self, name: str, extent: int, padding: int) -> None:
        """Reject a flat launch extent whose offsets overflow the kernel index.

        ``padding`` is how far the launch's last block may reach past
        ``extent``; the extent includes ensemble members.
        """

        if self.max_extent is not None and extent > self.max_extent - padding:
            raise OverflowError(
                f"{name}: {self.name} launch extent {extent} (including batched "
                f"members) exceeds the kernel offset range for BLOCK_SIZE="
                f"{padding}; the extent must be <= {self.max_extent - padding}; "
                "split the domain or ensemble across launches"
            )


TORCH = Backend(
    name="torch",
    devices=None,
    toolchain="python",
    dialect="torch",
    capture=None,
    block=BlockWidth(),
    max_extent=None,
    scalars=_ALL_SCALARS - {"uint32"},
    precisions=frozenset(("float32", "float64")),
    mixed_precision=True,
    mixed_precision_default=frozenset(),
)
TRITON = Backend(
    name="triton",
    devices=frozenset(("cuda", "xpu")),
    toolchain="triton",
    dialect="triton",
    capture="cuda_graph",
    # ``tl.arange`` widths are powers of two; ``pid * BLOCK + arange`` is int32.
    block=BlockWidth(power_of_two=True),
    max_extent=2**31 - 1,
    scalars=_ALL_SCALARS - {"uint32"},
    precisions=frozenset(("float32", "float64")),
    mixed_precision=True,
    mixed_precision_default=frozenset(("cuda", "xpu")),
)
CUDA = Backend(
    name="cuda",
    devices=frozenset(("cuda",)),
    toolchain="rtc",
    dialect="cuda",
    capture="cuda_graph",
    block=BlockWidth(),
    # Grid dimensions are checked per launch by the runtime-compiled launcher.
    max_extent=None,
    scalars=_ALL_SCALARS,
    precisions=frozenset(("float32", "float64")),
    mixed_precision=True,
    mixed_precision_default=frozenset(("cuda",)),
)
METAL = Backend(
    name="metal",
    devices=frozenset(("mps",)),
    toolchain="metal",
    dialect="msl",
    capture="metal_icb",
    block=BlockWidth(fixed=256),
    # MSL grid coordinates are 32-bit unsigned.
    max_extent=2**32 - 1,
    scalars=_ALL_SCALARS - {"float64"},
    precisions=frozenset(("float32",)),
    mixed_precision=False,
    mixed_precision_default=frozenset(),
)

BACKENDS: Mapping[str, Backend] = MappingProxyType(
    {backend.name: backend for backend in (TORCH, TRITON, CUDA, METAL)}
)


def backend_named(name: str) -> Backend:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown HydroForge backend {name!r}; expected one of {sorted(BACKENDS)}"
        ) from None


@validate_call(config=HydroForgeModel.model_config)
def resolve_backend(device: torch.device) -> Backend:
    """Resolve one model's backend from its declared device.

    An explicit ``HYDROFORGE_BACKEND`` remains authoritative.  In automatic
    mode the model device, rather than accelerator visibility elsewhere in the
    process, selects the backend, so CPU and accelerator models coexist
    without assigning a native GPU backend to CPU state.  Selecting Triton
    proves that its driver matches the device.
    """

    if device.type in {"xla", "lazy"}:
        require_triton_device(device)
    configured = env.choice(env.BACKEND, frozenset(BACKENDS))
    if configured is not None:
        backend = BACKENDS[configured]
        if backend is TRITON:
            require_triton_device(device)
        backend.require_device(device)
        return backend
    if device.type in {"cuda", "xpu"}:
        require_triton_device(device)
        return TRITON
    if device.type == "mps":
        return METAL
    return TORCH

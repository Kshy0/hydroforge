"""Kernel contracts, binding, and backend dispatch."""

from hydroforge.kernels.registry import (
    BackendRegistry,
    KernelEntry,
    make_metal_dispatcher,
    make_torch_dispatcher,
    make_triton_dispatcher,
    make_variant_dispatcher,
)

__all__ = [
    "BackendRegistry",
    "KernelEntry",
    "make_metal_dispatcher",
    "make_torch_dispatcher",
    "make_triton_dispatcher",
    "make_variant_dispatcher",
]

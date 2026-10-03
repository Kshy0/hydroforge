"""Capabilities of torch devices that decide backend defaults."""

from __future__ import annotations

import torch


def float64_supported(device: torch.device) -> bool:
    """Return whether an XPU device computes in FP64, or fail if unknown."""

    runtime = getattr(torch, "xpu", None)
    properties_getter = getattr(runtime, "get_device_properties", None)
    if properties_getter is None:
        raise RuntimeError(
            "this PyTorch XPU runtime cannot report FP64 capability; HydroForge "
            "cannot safely select float64 Triton storage"
        )
    try:
        properties = properties_getter(device)
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"cannot query FP64 capability for XPU device {str(device)!r}"
        ) from error
    supported = getattr(properties, "has_fp64", None)
    if type(supported) is not bool:
        raise RuntimeError(
            f"XPU device {str(device)!r} did not expose an exact has_fp64 "
            "capability; HydroForge cannot safely select float64 Triton storage"
        )
    return supported

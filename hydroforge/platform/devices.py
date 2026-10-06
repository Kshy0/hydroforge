# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Capabilities of torch devices that decide backend defaults."""

from __future__ import annotations

import torch


def float64_supported(device: torch.device) -> bool:
    """Return whether ``device`` computes in FP64, or fail if unknown.

    CPU and CUDA/ROCm devices always do and MPS never does; an XPU device
    reports its own capability.
    """

    if device.type in {"cpu", "cuda"}:
        return True
    if device.type == "mps":
        return False
    if device.type != "xpu":
        raise ValueError(f"unknown FP64 capability of device {str(device)!r}")
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

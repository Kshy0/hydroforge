"""Explicit math options for HydroForge-owned Triton launches."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from hydroforge.kernels.math_mode import fast_math


@cache
def _toolkit_libdevice() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME, include_paths

    if CUDA_HOME is None:
        return None
    for root in include_paths("cuda"):
        candidate = Path(root).parent / "nvvm" / "libdevice" / "libdevice.10.bc"
        if candidate.is_file():
            return str(candidate)
    return None


def kernel_options(kernel: Any, *, physics: bool) -> dict[str, Any]:
    """Resolve defaults for a JIT launch; explicit caller options take priority.

    Already lowered launch adapters have no JIT parameters or compiler options.
    Framework statistics always preserve subnormals, regardless of physics mode.
    """
    if not hasattr(kernel, "params"):
        return {}
    from triton import knobs
    from triton.runtime import driver

    if driver.active.get_current_target().backend != "cuda":
        return {}
    options: dict[str, Any] = {"enable_reflect_ftz": physics and fast_math()}
    if not knobs.nvidia.libdevice_path:
        libdevice = _toolkit_libdevice()
        if libdevice is not None:
            options["extern_libs"] = (("libdevice", libdevice),)
    return options

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Read-only discovery of system and NVIDIA wheel CUDA toolkit files."""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from pathlib import Path


@cache
def include_roots() -> tuple[Path, ...]:
    """Find CUDA headers without changing the environment or PyTorch globals.

    An explicit CUDA_HOME wins over automatic discovery. PyTorch's system
    toolkit remains the default; NVIDIA wheels supply a toolkit when no system
    installation exists. Both NVRTC and Triton use this same discovery.
    """
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, include_paths

    declared = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if declared:
        return (Path(declared) / "include",)
    if CUDA_HOME is not None:
        return tuple(dict.fromkeys(Path(path) for path in include_paths("cuda")))
    if torch.version.cuda is None:
        return ()
    major = int(torch.version.cuda.split(".")[0])
    packages = (
        (f"nvidia.cu{major}",)
        if major >= 13
        else ("nvidia.cuda_runtime", "nvidia.cuda_cccl")
    )
    roots = []
    for package in packages:
        try:
            spec = importlib.util.find_spec(package)
        except (ModuleNotFoundError, ValueError):
            continue
        if spec is not None:
            for location in spec.submodule_search_locations or ():
                candidate = Path(location) / "include"
                if candidate.is_dir():
                    roots.append(candidate)
    return tuple(dict.fromkeys(roots))


@cache
def libdevice_path() -> str | None:
    """The libdevice accompanying the selected toolkit, if it is installed."""
    for root in include_roots():
        candidate = root.parent / "nvvm/libdevice/libdevice.10.bc"
        if candidate.is_file():
            return str(candidate)
    return None

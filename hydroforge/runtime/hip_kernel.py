# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""HIP kernel loader for CUDA-like PyTorch extensions.

PyTorch's extension builder hipifies ``.cu`` sources automatically when running
under a ROCm-enabled PyTorch build.  This module adds the cmfgpu-facing guard
rails around that path: clear environment errors on CUDA-only PyTorch, HIP-safe
fast-math flags, and a tiny source shim for PyTorch stream headers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch

from hydroforge.runtime.cuda_kernel import load_inline_cu_module


def _as_list(src: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(src, str):
        return [src]
    return list(src)


def _hipify_cuda_like_source(src: str) -> str:
    """Apply the narrow source edits PyTorch hipify does not cover reliably."""
    replacements = (
        ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
        ("#include <c10/cuda/CUDAStream.h>", "#include <c10/hip/HIPStream.h>"),
        ("cudaStream_t", "hipStream_t"),
        ("c10::cuda::getCurrentCUDAStream()", "c10::hip::getCurrentHIPStream()"),
    )
    for old, new in replacements:
        src = src.replace(old, new)
    return src


def _normalise_hip_flags(flags: Sequence[str]) -> list[str]:
    out: list[str] = []
    for flag in flags:
        if flag == "--use_fast_math":
            out.append("-ffast-math")
        else:
            out.append(flag)
    return out


def load_inline_hip_module(
    name: str,
    *,
    cpp_sources: Union[str, Sequence[str]],
    cuda_sources: Union[str, Sequence[str]],
    functions: Sequence[str],
    extra_cuda_cflags: Sequence[str] = ("-O3", "-ffast-math"),
    verbose: bool = False,
    build_directory: Optional[Union[str, Path]] = None,
    env_prefix: str = "HYDROFORGE",
    force_rebuild: Optional[bool] = None,
    system_compiler: Optional[str] = None,
) -> Any:
    """Compile a CUDA-like inline extension with HIP/ROCm.

    The public kwargs intentionally mirror
    :func:`hydroforge.runtime.cuda_kernel.load_inline_cu_module` so downstream
    CUDA backends can switch loader functions without changing dispatch code.
    """
    if torch.version.hip is None:
        raise RuntimeError(
            "HYDROFORGE_BACKEND=hip requires a ROCm-enabled PyTorch build "
            f"(torch.version.hip is None for torch {torch.__version__})."
        )

    hip_sources = [_hipify_cuda_like_source(src) for src in _as_list(cuda_sources)]
    return load_inline_cu_module(
        name,
        cpp_sources=cpp_sources,
        cuda_sources=hip_sources,
        functions=functions,
        extra_cuda_cflags=_normalise_hip_flags(extra_cuda_cflags),
        verbose=verbose,
        build_directory=build_directory,
        env_prefix=env_prefix,
        force_rebuild=force_rebuild,
        system_compiler=system_compiler,
        toolchain="hip",
    )

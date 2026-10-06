# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Stable physical-kernel authoring API."""

from hydroforge.kernels.cuda import (
    CudaCall,
    CudaFill,
    CudaKernel,
    CudaSource,
    CudaWorkspace,
)
from hydroforge.kernels.metal import MetalKernel
from hydroforge.kernels.registry import BackendRegistry
from hydroforge.kernels.spec import (
    KernelSpec,
    KernelWorkspace,
    constant,
    literal_value,
    module_enabled,
    module_flag,
    output_requested,
)
from hydroforge.kernels.torch import TorchKernel
from hydroforge.kernels.triton import (
    TritonKernel,
    TritonProgram,
    TritonSequence,
    launch_triton_kernel,
)

__all__ = [
    "BackendRegistry",
    "CudaCall",
    "CudaFill",
    "CudaKernel",
    "CudaSource",
    "CudaWorkspace",
    "KernelSpec",
    "KernelWorkspace",
    "MetalKernel",
    "TorchKernel",
    "TritonKernel",
    "TritonProgram",
    "TritonSequence",
    "constant",
    "launch_triton_kernel",
    "literal_value",
    "module_enabled",
    "module_flag",
    "output_requested",
]

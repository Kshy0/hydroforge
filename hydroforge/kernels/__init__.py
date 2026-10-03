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
    config_value,
    constant,
    literal_value,
    module_enabled,
    module_flag,
    option_code,
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
    "config_value",
    "constant",
    "launch_triton_kernel",
    "literal_value",
    "module_enabled",
    "module_flag",
    "option_code",
    "output_requested",
]

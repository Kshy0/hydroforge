"""Declarative CUDA/HIP extension backend."""

from hydroforge.kernels.backends.cuda.dispatcher import (
    CudaExtensionGroup, CudaNativeProjection,
)
from hydroforge.kernels.backends.cuda.spec import CudaExtensionSpec
from hydroforge.kernels.backends.cuda.template import (
    SpecCudaTemplateDispatcher, make_spec_cuda_dispatcher,
)

__all__ = [
    "CudaExtensionGroup", "CudaExtensionSpec",
    "CudaNativeProjection",
    "SpecCudaTemplateDispatcher", "make_spec_cuda_dispatcher",
]

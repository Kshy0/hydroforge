"""Declarative CUDA/HIP extension backend."""

from hydroforge.kernels.backends.cuda.dispatcher import (
    CudaExtensionGroup, CudaNativeProjection, CudaRoute,
)
from hydroforge.kernels.backends.cuda.spec import CudaExtensionSpec

__all__ = [
    "CudaExtensionGroup", "CudaExtensionSpec", "CudaRoute",
    "CudaNativeProjection",
]

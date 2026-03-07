"""
hydroforge: Generic framework for GPU-accelerated hydrological modelling.

Subpackages
-----------
modeling    Lowest-level abstractions: AbstractModule, AbstractModel, distributed utils.
io          AbstractDataset + concrete dataset implementations, multi-rank I/O.
aggregator  Streaming statistics aggregation with NetCDF / in-memory output.
runtime     GPU kernel backend selection and adapters (Triton / CUDA / Metal / PyTorch).
"""

from hydroforge.modeling.distributed import (find_indices_in,
                                             find_indices_in_torch,
                                             get_global_rank, get_local_rank,
                                             get_world_size, is_rank_zero,
                                             setup_distributed,
                                             torch_to_numpy_dtype)
from hydroforge.modeling.input_proxy import InputProxy
from hydroforge.modeling.model import AbstractModel
from hydroforge.modeling.module import (AbstractModule, TensorField,
                                        computed_tensor_field)
from hydroforge.runtime.cuda_graph import CUDAGraphMixin

__all__ = [
    "AbstractModel",
    "AbstractModule",
    "CUDAGraphMixin",
    "InputProxy",
    "TensorField",
    "computed_tensor_field",
    "find_indices_in",
    "find_indices_in_torch",
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "is_rank_zero",
    "setup_distributed",
    "torch_to_numpy_dtype",
]

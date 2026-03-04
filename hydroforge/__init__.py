"""
hydroforge: Generic framework for GPU-accelerated hydrological modelling.

Subpackages
-----------
core        Lowest-level abstractions: AbstractModule, AbstractModel, distributed utils.
datasets    AbstractDataset + concrete dataset implementations.
io          Input/output helpers: InputProxy.
aggregator  Streaming statistics aggregation with NetCDF / in-memory output.
compute     GPU kernel backend selection and adapters (Triton / PyTorch).
"""

from hydroforge.core.distributed import (find_indices_in,
                                         find_indices_in_torch,
                                         get_global_rank, get_local_rank,
                                         get_world_size, is_rank_zero,
                                         setup_distributed,
                                         torch_to_numpy_dtype)
from hydroforge.core.model import AbstractModel
from hydroforge.core.module import (AbstractModule, TensorField,
                                    computed_tensor_field)
from hydroforge.core.input_proxy import InputProxy

__all__ = [
    "AbstractModel",
    "AbstractModule",
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

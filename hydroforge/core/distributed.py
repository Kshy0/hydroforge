"""
hydroforge.core.distributed
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Distributed-training helpers (rank / world-size queries, process-group
setup) plus low-level numeric utilities shared across modules.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import distributed as dist

# ---------------------------------------------------------------------------
# Rank / world-size helpers
# ---------------------------------------------------------------------------

def get_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def is_rank_zero() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def setup_distributed():
    """Initialise the NCCL process group and return (local_rank, rank, world_size)."""
    torch.multiprocessing.set_start_method("spawn", force=True)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = get_local_rank()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    return local_rank, rank, world_size


# ---------------------------------------------------------------------------
# Binary / map file I/O (used by dataset implementations)
# ---------------------------------------------------------------------------

def binread(filename, shape, dtype_str):
    """Read a Fortran-ordered binary file and reshape to *shape*."""
    count = 1
    for s in shape:
        count *= s
    arr = np.fromfile(filename, dtype=dtype_str, count=count)
    return arr.reshape(shape, order='F')


def read_map(filename, map_shape, precision):
    """Read a spatial map binary file (Fortran-ordered)."""
    if len(map_shape) == 2:
        nx, ny = map_shape
        data = binread(filename, (nx, ny), dtype_str=precision)
    elif len(map_shape) == 3:
        nx, ny, nlfp = map_shape
        data = binread(filename, (nx, ny, nlfp), dtype_str=precision)
    else:
        raise ValueError("Unsupported map_shape dimension.")
    return data


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------

def find_indices_in(a, b):
    """Return indices in *b* for each element of *a* (NumPy version)."""
    order = np.argsort(b)
    sorted_b = b[order]
    pos_in_sorted = np.searchsorted(sorted_b, a)
    valid_mask = pos_in_sorted < len(sorted_b)
    hit_mask = np.zeros_like(a, dtype=bool)
    hit_mask[valid_mask] = (sorted_b[pos_in_sorted[valid_mask]] == a[valid_mask])
    index = np.full_like(pos_in_sorted, -1, dtype=int)
    index[hit_mask] = order[pos_in_sorted[hit_mask]]
    return index


def find_indices_in_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return indices in *b* for each element of *a* (Torch version)."""
    sorted_b, order = torch.sort(b)
    pos = torch.bucketize(a, sorted_b, right=False)
    valid_mask = pos < len(sorted_b)
    hit_mask = torch.zeros_like(a, dtype=torch.bool)
    hit_mask[valid_mask] = (sorted_b[pos[valid_mask]] == a[valid_mask])
    index = torch.full_like(pos, -1, dtype=torch.int32)
    index[hit_mask] = order[pos[hit_mask]].to(torch.int32)
    return index


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def torch_to_numpy_dtype(torch_dtype: torch.dtype) -> type:
    dtype_mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
    }
    if torch_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return dtype_mapping[torch_dtype]

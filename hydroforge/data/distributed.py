# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
hydroforge.data.distributed

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

LOCAL_PROCESS_RANK_ENV = (
    "LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK",
)


def get_local_process_rank() -> int:
    """Resolve one strict local rank directly from launcher environment."""

    observed: dict[str, int] = {}
    for name in LOCAL_PROCESS_RANK_ENV:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(
                f"{name} must be a non-negative integer, got {raw!r}"
            ) from error
        if value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer, got {raw!r}"
            )
        observed[name] = value
    ranks = set(observed.values())
    if len(ranks) > 1:
        raise ValueError(f"conflicting local-rank environment: {observed}")
    return next(iter(ranks), 0)

def get_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return get_local_process_rank()
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
    raw_world_size = os.environ.get("WORLD_SIZE")
    try:
        ws_env = 1 if raw_world_size is None else int(raw_world_size)
    except ValueError as error:
        raise ValueError(
            f"WORLD_SIZE must be a positive integer, got {raw_world_size!r}"
        ) from error
    if ws_env < 1:
        raise ValueError(
            f"WORLD_SIZE must be a positive integer, got {raw_world_size!r}"
        )
    local_rank = get_local_process_rank()
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if raw_world_size is not None and ws_env != world_size:
            raise ValueError(
                f"WORLD_SIZE={ws_env} disagrees with the initialized process "
                f"group world_size={world_size}"
            )
        return local_rank, rank, world_size
    if ws_env > 1:
        if not dist.is_available():
            raise RuntimeError(
                "WORLD_SIZE > 1 requires torch.distributed support"
            )
        if torch.cuda.is_available():
            n_dev = torch.cuda.device_count()
            if local_rank >= n_dev:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} is outside the {n_dev} CUDA "
                    f"device(s) visible on this node (CUDA_VISIBLE_DEVICES="
                    f"{os.environ.get('CUDA_VISIBLE_DEVICES')}). "
                    "The launcher must assign one valid local device index "
                    "per process. WORLD_SIZE may legitimately exceed this "
                    "node-local device count in a multi-node job."
                )
            torch.cuda.set_device(local_rank)
            device_id = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(
                backend="nccl", init_method="env://",
                device_id=device_id,
            )
        else:
            dist.init_process_group(backend="gloo", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        if local_rank != 0:
            raise ValueError(
                f"local rank is {local_rank}, but WORLD_SIZE={ws_env}; "
                "launcher topology is incomplete"
            )
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
    if b.numel() == 0:
        return torch.full_like(a, -1, dtype=torch.int32)
    if len(b) > torch.iinfo(torch.int32).max:
        raise OverflowError(
            f"b has {len(b)} elements, exceeding int32 range. "
            "find_indices_in_torch returns int32 indices."
        )
    sorted_b, order = torch.sort(b)
    pos = torch.bucketize(a, sorted_b, right=False)
    # bucketize on MPS/some GPU backends may return len(sorted_b) for values
    # that equal the last element; clamp to keep indexing safe — the equality
    # check below still rejects true misses.
    pos = pos.clamp(max=len(sorted_b) - 1)
    hit_mask = sorted_b[pos] == a
    index = torch.full_like(a, -1, dtype=torch.int32)
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

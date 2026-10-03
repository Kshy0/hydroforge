# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Launcher environment: process ranks and ``env://`` rendezvous variables."""

from __future__ import annotations

import os
from collections.abc import Mapping


def communication_backend(value: object) -> str:
    """Decode the supported PyTorch communication enum/string spellings."""
    name = str(value).lower().removeprefix("backend.")
    if name not in {"gloo", "nccl", "xccl"}:
        raise ValueError(f"unsupported communication backend {value!r}")
    return name


LOCAL_PROCESS_RANK_ENV = (
    "SLURM_LOCALID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_LOCAL_RANK",
)

LOCAL_PROCESS_COUNT_ENV = (
    "SLURM_NTASKS_PER_NODE",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MPI_LOCALNRANKS",
    "MV2_COMM_WORLD_LOCAL_SIZE",
)


def require_launcher_environment(environment: Mapping[str, str]) -> None:
    """Native env:// must not consume a different launcher observation."""
    names = (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
        *LOCAL_PROCESS_RANK_ENV,
        *LOCAL_PROCESS_COUNT_ENV,
    )
    if any(os.environ.get(name) != environment.get(name) for name in names):
        raise RuntimeError("launcher environment changed during distributed setup")


def _environment_index(
    name: str, *, minimum: int = 0, environment: Mapping[str, str] | None = None
) -> int | None:
    raw = (os.environ if environment is None else environment).get(name)
    if raw is None:
        return None
    kind = "non-negative" if minimum == 0 else "positive"
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a {kind} integer, got {raw!r}") from error
    if value < minimum:
        raise ValueError(f"{name} must be a {kind} integer, got {raw!r}")
    return value


def get_local_process_rank(environment: Mapping[str, str] | None = None) -> int:
    """Resolve one strict local rank directly from launcher environment.

    ``LOCAL_RANK`` is written by the innermost launcher (``torchrun``) and is
    authoritative: scheduler variables inherited from an outer ``srun`` or
    ``mpirun`` describe the launcher task, not this worker. Without
    ``LOCAL_RANK`` the scheduler variables must agree with each other.
    """

    local_rank = _environment_index("LOCAL_RANK", environment=environment)
    if local_rank is not None:
        return local_rank
    observed: dict[str, int] = {}
    for name in LOCAL_PROCESS_RANK_ENV:
        value = _environment_index(name, environment=environment)
        if value is not None:
            observed[name] = value
    ranks = set(observed.values())
    if len(ranks) > 1:
        raise ValueError(f"conflicting local-rank environment: {observed}")
    return next(iter(ranks), 0)


def _local_process_count(environment: Mapping[str, str] | None = None) -> int | None:
    """Return the launcher's process count on this node, when declared."""

    if (os.environ if environment is None else environment).get(
        "LOCAL_RANK"
    ) is not None:
        return _environment_index(
            "LOCAL_WORLD_SIZE", minimum=1, environment=environment
        )
    observed = {}
    for name in LOCAL_PROCESS_COUNT_ENV:
        value = _environment_index(name, minimum=1, environment=environment)
        if value is not None:
            observed[name] = value
    if len(set(observed.values())) > 1:
        raise ValueError(f"conflicting local-count environment: {observed}")
    return next(iter(observed.values()), None)


def _world_size_environment(
    environment: Mapping[str, str] | None = None,
) -> tuple[str | None, int]:
    """Return the validated launcher world size and its original spelling."""

    raw_world_size = (os.environ if environment is None else environment).get(
        "WORLD_SIZE"
    )
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
    return raw_world_size, ws_env


def _rank_environment(
    world_size: int,
    *,
    required: bool,
    environment: Mapping[str, str] | None = None,
) -> tuple[str | None, int | None]:
    """Return a validated launcher rank, optionally requiring its presence."""

    raw_rank = (os.environ if environment is None else environment).get("RANK")
    if raw_rank is None and not required:
        return None, None
    try:
        rank = int(raw_rank) if raw_rank is not None else -1
    except ValueError as error:
        raise ValueError(
            f"RANK must be an integer in [0, WORLD_SIZE), got {raw_rank!r}"
        ) from error
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"RANK must be in [0, WORLD_SIZE), got {raw_rank!r} for "
            f"WORLD_SIZE={world_size}"
        )
    return raw_rank, rank


def _rendezvous_environment(
    world_size: int, environment: Mapping[str, str] | None = None
) -> int:
    """Validate the complete ``env://`` rendezvous before creating a group."""

    _, rank = _rank_environment(world_size, required=True, environment=environment)
    master_addr = (os.environ if environment is None else environment).get(
        "MASTER_ADDR"
    )
    if master_addr is None or not master_addr.strip():
        raise ValueError("MASTER_ADDR must be a non-empty string for env:// rendezvous")
    raw_port = (os.environ if environment is None else environment).get("MASTER_PORT")
    try:
        port = int(raw_port) if raw_port is not None else -1
    except ValueError as error:
        raise ValueError(
            f"MASTER_PORT must be an integer in [1, 65535], got {raw_port!r}"
        ) from error
    if not 1 <= port <= 65535:
        raise ValueError(
            f"MASTER_PORT must be an integer in [1, 65535], got {raw_port!r}"
        )
    assert rank is not None
    return rank

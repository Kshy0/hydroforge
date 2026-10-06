# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Launcher environment: process ranks and ``env://`` rendezvous variables."""

from __future__ import annotations

import os
from collections.abc import Mapping

_SUPPORTED_COMMUNICATION_BACKENDS = frozenset({"gloo", "nccl", "xccl"})
# PyTorch's default collective backend of each device type; an ``undefined``
# default group creates these lazily for the devices it meets.
_DEFAULT_DEVICE_BACKENDS = {"cpu": "gloo", "cuda": "nccl", "xpu": "xccl"}
_ACCELERATOR_PREFERENCE = ("cuda", "xpu")


def _available_accelerators() -> tuple[str, ...]:
    import torch

    return tuple(
        kind
        for kind in _ACCELERATOR_PREFERENCE
        if getattr(torch, kind, None) is not None
        and getattr(torch, kind).is_available()
    )


def _device_backend_entries(text: str, value: object) -> dict[str, str]:
    entries: dict[str, str] = {}
    for item in text.split(","):
        kind, separator, name = item.partition(":")
        kind, name = kind.strip(), name.strip()
        if not separator or not kind or not name or kind in entries:
            raise ValueError(f"malformed communication backend {value!r}")
        entries[kind] = name
    return entries


def _select_device_backend(
    entries: Mapping[str, str], device_type: str | None, value: object
) -> str:
    if device_type is not None:
        kind = str(device_type).lower()
        if kind not in entries and kind == "mps":
            # MPS collectives are staged through CPU tensors.
            kind = "cpu"
        if kind not in entries:
            raise ValueError(
                f"communication backend {value!r} has no entry for device type "
                f"{device_type!r}"
            )
        return entries[kind]
    accelerators = [kind for kind in entries if kind != "cpu"]
    if len(accelerators) == 1:
        return entries[accelerators[0]]
    for kind in _ACCELERATOR_PREFERENCE:
        if kind in entries:
            return entries[kind]
    if accelerators:
        raise ValueError(
            f"communication backend {value!r} names several accelerator "
            "backends; pass the device type"
        )
    return entries["cpu"]


def communication_backend(value: object, device_type: str | None = None) -> str:
    """Decode the supported PyTorch communication backend spellings.

    ``value`` is a backend name or enum (``"nccl"``, ``Backend.NCCL``), a
    per-device list (``"cpu:gloo,cuda:nccl"``), or ``"undefined"`` (a default
    group created without a backend, whose per-device backends PyTorch
    creates lazily with the defaults ``cpu:gloo,cuda:nccl,xpu:xccl``).
    For a list, ``device_type`` selects its entry (MPS falls back to the CPU
    entry). Without ``device_type`` the sole accelerator entry is chosen,
    otherwise CUDA, then XPU, then the CPU entry; ``"undefined"`` then only
    considers the accelerators available in this runtime. A plain name is
    returned unchanged whatever ``device_type`` is.
    """
    text = str(value).strip().lower().removeprefix("backend.")
    if text == "undefined":
        kinds = (
            tuple(_DEFAULT_DEVICE_BACKENDS)
            if device_type is not None
            else ("cpu", *_available_accelerators())
        )
        name = _select_device_backend(
            {kind: _DEFAULT_DEVICE_BACKENDS[kind] for kind in kinds},
            device_type,
            value,
        )
    elif ":" in text:
        name = _select_device_backend(
            _device_backend_entries(text, value), device_type, value
        )
    else:
        name = text
    if name not in _SUPPORTED_COMMUNICATION_BACKENDS:
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

# Global task counts of launchers that do not export ``WORLD_SIZE``.
SCHEDULER_TASK_COUNT_ENV = (
    "SLURM_STEP_NUM_TASKS",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
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
        *SCHEDULER_TASK_COUNT_ENV,
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
    if raw_world_size is None:
        # A scheduler that started several tasks without the env:// variables
        # would otherwise run every task as an independent single process.
        for name in SCHEDULER_TASK_COUNT_ENV:
            tasks = _environment_index(name, minimum=1, environment=environment)
            if tasks is not None and tasks > 1:
                raise ValueError(
                    f"{name}={tasks} reports a multi-process launch, but "
                    "WORLD_SIZE is not set; export WORLD_SIZE, RANK, "
                    "MASTER_ADDR and MASTER_PORT for every task or launch "
                    "with torchrun"
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

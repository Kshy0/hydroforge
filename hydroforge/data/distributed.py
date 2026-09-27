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
import socket
from math import prod
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, Self

import numpy as np
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator
from torch import distributed as dist

from hydroforge.contracts.errors import cleanup_on_exit
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.devices import devices_match

# ---------------------------------------------------------------------------
# Rank / world-size helpers
# ---------------------------------------------------------------------------

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


_COMMUNICATION_BACKENDS = {"cpu": "gloo", "cuda": "nccl", "xpu": "xccl"}


class _ProcessTopology(HydroForgeModel):
    """Immutable process-group identity captured for one model instance."""

    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_topology(self) -> Self:
        if self.rank >= self.world_size:
            raise ValueError("process rank must be smaller than world_size")
        return self


class ProcessTopology(_ProcessTopology):
    """Capture the rank identity of the initialized default group."""

    @classmethod
    def capture(cls) -> ProcessTopology:
        """Capture the initialized default group, or the local 0/1 topology."""

        if dist.is_available() and dist.is_initialized():
            return cls(rank=dist.get_rank(), world_size=dist.get_world_size())
        return cls(rank=0, world_size=1)


class DistributedContext(_ProcessTopology):
    """Resolved process topology, communication backend, and model device."""

    local_rank: int = Field(ge=0)
    device: torch.device
    backend: Literal["gloo", "nccl", "xccl"] | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> Self:
        if self.local_rank >= self.world_size:
            raise ValueError("local_rank must be smaller than world_size")
        if self.device.type not in {"cpu", "cuda", "xpu", "mps"}:
            raise ValueError(
                "distributed context device must be CPU, CUDA, XPU, or MPS"
            )
        if self.device.type == "cpu" and self.device.index is not None:
            raise ValueError("CPU distributed devices must not have an index")
        if self.device.type in {"cuda", "xpu"}:
            if self.device.index is None:
                raise ValueError(
                    "accelerator distributed devices must have a concrete index"
                )
            # Per-process device binding (one visible device per task) maps
            # several local ranks onto index 0, so the index may be smaller.
            if self.world_size > 1 and self.device.index > self.local_rank:
                raise ValueError(
                    "multi-process accelerator device index must not exceed "
                    "local_rank"
                )
        if self.device.type == "mps":
            if self.world_size != 1 or self.local_rank != 0:
                raise ValueError("MPS distributed context must be single-process")
            if self.device.index not in {None, 0}:
                raise ValueError("MPS exposes only device index 0")
            if self.backend is not None:
                raise ValueError("MPS does not have a torch.distributed backend")
        if self.world_size > 1 and self.backend is None:
            raise ValueError(
                "multi-process distributed context requires a communication backend"
            )
        required_backend = _COMMUNICATION_BACKENDS.get(self.device.type)
        if self.backend is not None and self.backend != required_backend:
            raise ValueError(
                f"communication backend {self.backend!r} is incompatible with "
                f"device {str(self.device)!r}"
            )
        return self

    def __iter__(self) -> NoReturn:
        """Reject tuple-style compatibility; callers must use named fields."""

        raise TypeError(
            "DistributedContext is not iterable; use .local_rank, .rank, "
            ".world_size, .device, and .backend"
        )


class _DistributedSetupRequest(HydroForgeModel):
    """Validated public request consumed by distributed initialization."""

    allowed_devices: tuple[torch.device, ...]
    required_kernel_backend: Literal["torch", "triton", "cuda", "metal"] | None = None

    @field_validator("allowed_devices", mode="before")
    @classmethod
    def _validate_allowed_devices(
        cls,
        value: Any,
    ) -> tuple[torch.device, ...]:
        if type(value) is not tuple:
            raise ValueError("allowed_devices must be an exact tuple")
        if not value:
            raise ValueError("allowed_devices must not be empty")
        devices: list[torch.device] = []
        for index, candidate in enumerate(value):
            if type(candidate) is str and candidate.lower().partition(":")[0] == "tpu":
                raise ValueError(
                    "TPU/XLA requires a torch_xla PJRT adapter and cannot be "
                    "initialized by HydroForge setup_distributed"
                )
            if isinstance(candidate, torch.device):
                device = candidate
            elif type(candidate) is str:
                try:
                    device = torch.device(candidate)
                except (TypeError, RuntimeError) as error:
                    raise ValueError(
                        f"allowed_devices[{index}] is not a valid torch device: "
                        f"{candidate!r}"
                    ) from error
            else:
                raise ValueError(
                    "allowed_devices entries must be exact strings or "
                    "torch.device objects"
                )
            if device.type in {"xla", "lazy"}:
                raise ValueError(
                    "TPU/XLA requires a torch_xla PJRT adapter and cannot be "
                    "initialized by HydroForge setup_distributed"
                )
            if device.type not in {"cpu", "cuda", "xpu", "mps"}:
                raise ValueError(
                    f"allowed device {str(device)!r} is unsupported; expected "
                    "CPU, CUDA, XPU, or MPS"
                )
            if device.type == "cpu" and device.index is not None:
                raise ValueError("CPU device candidates must not have an index")
            if device.type == "mps" and device.index not in {None, 0}:
                raise ValueError("MPS exposes only device index 0")
            devices.append(device)
        if len(set(devices)) != len(devices):
            raise ValueError("allowed_devices must not contain duplicates")
        return tuple(devices)

    @model_validator(mode="after")
    def _validate_kernel_backend_candidates(self) -> Self:
        compatible_types = {
            "triton": {"cuda", "xpu"},
            "cuda": {"cuda"},
            "metal": {"mps"},
        }.get(self.required_kernel_backend)
        if compatible_types is not None and not any(
            device.type in compatible_types for device in self.allowed_devices
        ):
            raise ValueError(
                f"required kernel backend {self.required_kernel_backend!r} "
                f"has no compatible candidate in allowed_devices"
            )
        return self


def _environment_index(name: str, *, minimum: int = 0) -> int | None:
    raw = os.environ.get(name)
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


def get_local_process_rank() -> int:
    """Resolve one strict local rank directly from launcher environment.

    ``LOCAL_RANK`` is written by the innermost launcher (``torchrun``) and is
    authoritative: scheduler variables inherited from an outer ``srun`` or
    ``mpirun`` describe the launcher task, not this worker. Without
    ``LOCAL_RANK`` the scheduler variables must agree with each other.
    """

    local_rank = _environment_index("LOCAL_RANK")
    if local_rank is not None:
        return local_rank
    observed: dict[str, int] = {}
    for name in LOCAL_PROCESS_RANK_ENV:
        value = _environment_index(name)
        if value is not None:
            observed[name] = value
    ranks = set(observed.values())
    if len(ranks) > 1:
        raise ValueError(f"conflicting local-rank environment: {observed}")
    return next(iter(ranks), 0)


def _local_process_count() -> int | None:
    """Return the launcher's process count on this node, when declared."""

    if os.environ.get("LOCAL_RANK") is not None:
        return _environment_index("LOCAL_WORLD_SIZE", minimum=1)
    for name in LOCAL_PROCESS_COUNT_ENV:
        value = _environment_index(name, minimum=1)
        if value is not None:
            return value
    return None


def is_rank_zero() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _world_size_environment() -> tuple[str | None, int]:
    """Return the validated launcher world size and its original spelling."""

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
    return raw_world_size, ws_env


def _rank_environment(
    world_size: int,
    *,
    required: bool,
) -> tuple[str | None, int | None]:
    """Return a validated launcher rank, optionally requiring its presence."""

    raw_rank = os.environ.get("RANK")
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


def _backend_name(value: Any) -> str:
    """Normalize PyTorch Backend enum/string representations."""

    normalized = str(value).lower()
    for name in ("nccl", "xccl", "gloo"):
        if name in normalized:
            return name
    return normalized


def _local_device_index(
    local_rank: int,
    device_count: int,
    *,
    device_type: str,
    visibility: str | None,
) -> int:
    """Bind one local rank to a visible accelerator index.

    Node-wide visibility binds ``LOCAL_RANK`` directly. A single visible
    device is per-process binding (``--gpus-per-task=1`` or a per-rank
    visibility mask). Fewer visible devices than local ranks are shared
    round-robin only when the declared node-local process count is an exact
    multiple of the visible device count; anything else is rejected as a
    likely launcher misconfiguration.
    """

    if local_rank < device_count:
        return local_rank
    if device_count == 1:
        return 0
    local_count = _local_process_count()
    if (
        device_count > 1
        and local_count is not None
        and local_rank < local_count
        and local_count % device_count == 0
    ):
        return local_rank % device_count
    raise RuntimeError(
        f"LOCAL_RANK/device index {local_rank} is outside the {device_count} "
        f"{device_type.upper()} device(s) visible on this node "
        f"(visibility mask={visibility!r}), and the node-local process count "
        f"{local_count!r} is not a multiple of the visible device count. The "
        "launcher must assign one valid local device index per process or "
        "expose one device per process. WORLD_SIZE may legitimately exceed "
        "this node-local device count in a multi-node job."
    )


def _accelerator_candidate(
    candidate: torch.device,
    *,
    local_rank: int,
    world_size: int,
) -> torch.device:
    """Validate and activate one CUDA/ROCm or Intel XPU candidate."""

    device_type = candidate.type
    runtime = getattr(torch, device_type, None)
    if runtime is None or not runtime.is_available():
        raise RuntimeError(f"{device_type!r} is not available in this PyTorch runtime")
    device_count = runtime.device_count()
    if type(device_count) is not int or device_count < 0:
        raise RuntimeError(
            f"{device_type.upper()} runtime returned invalid device_count "
            f"{device_count!r}"
        )
    visibility = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if device_type == "cuda"
        else os.environ.get("ZE_AFFINITY_MASK")
    )
    if world_size > 1:
        bound = _local_device_index(
            local_rank, device_count, device_type=device_type, visibility=visibility
        )
        index = bound if candidate.index is None else candidate.index
        if index != bound:
            raise RuntimeError(
                f"device index {index} disagrees with the device index {bound} "
                f"bound to LOCAL_RANK={local_rank}"
            )
    else:
        index = local_rank if candidate.index is None else candidate.index
    if index >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK/device index {index} is outside the {device_count} "
            f"{device_type.upper()} device(s) visible on this node "
            f"(visibility mask={visibility!r}). The launcher must assign one "
            "valid local device index per process. WORLD_SIZE may legitimately "
            "exceed this node-local device count in a multi-node job."
        )
    runtime.set_device(index)
    return torch.device(device_type, index)


def _communication_backend(device: torch.device) -> Literal["gloo", "nccl", "xccl"]:
    """Return the only supported collective backend for one device type."""

    return _COMMUNICATION_BACKENDS[device.type]


def _require_communication_backend(
    device: torch.device,
) -> Literal["gloo", "nccl", "xccl"]:
    """Preflight the process-group backend required by *device*."""

    backend = _communication_backend(device)
    available = getattr(dist, f"is_{backend}_available", None)
    if available is None or not available():
        raise RuntimeError(
            f"{device.type!r} distributed execution requires the PyTorch "
            f"{backend.upper()} communication backend"
        )
    return backend


def _require_kernel_backend(
    device: torch.device,
    required: Literal["torch", "triton", "cuda", "metal"] | None,
) -> str:
    """Resolve the HydroForge kernel backend before collective initialization."""

    # Import lazily: the CUDA precompile module imports rank helpers from this
    # module, so importing the kernel registry while this module is loading
    # would create a cycle.
    from hydroforge.kernels.registry import resolve_model_backend

    resolved = resolve_model_backend(device)
    if required is not None and resolved != required:
        raise RuntimeError(
            f"device {str(device)!r} resolved HydroForge kernel backend "
            f"{resolved!r}, but {required!r} is required"
        )
    return resolved


def _candidate_device(
    candidate: torch.device,
    *,
    local_rank: int,
    world_size: int,
    initialized_backend: Literal["gloo", "nccl", "xccl"] | None,
    required_kernel_backend: Literal["torch", "triton", "cuda", "metal"] | None,
) -> torch.device:
    """Validate, activate, and kernel-preflight one ordered candidate."""

    required_device_type = next(
        (
            kind
            for kind, backend in _COMMUNICATION_BACKENDS.items()
            if backend == initialized_backend
        ),
        None,
    )
    if required_device_type is not None and candidate.type != required_device_type:
        raise RuntimeError(
            f"initialized {initialized_backend.upper()} communication backend "
            f"requires a {required_device_type!r} device"
        )

    if candidate.type == "cpu":
        device = torch.device("cpu")
    elif candidate.type in {"cuda", "xpu"}:
        device = _accelerator_candidate(
            candidate,
            local_rank=local_rank,
            world_size=world_size,
        )
    else:
        if world_size > 1:
            raise RuntimeError(
                "MPS is available only for single-process HydroForge execution"
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in this PyTorch runtime")
        device = torch.device("mps")

    if initialized_backend is None and world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        _require_communication_backend(device)

    _require_kernel_backend(device, required_kernel_backend)
    return device


def _select_distributed_device(
    request: _DistributedSetupRequest,
    *,
    local_rank: int,
    world_size: int,
    initialized_backend: Literal["gloo", "nccl", "xccl"] | None,
) -> torch.device:
    """Select the first fully usable device in the caller's declared order."""

    failures: list[str] = []
    for candidate in request.allowed_devices:
        try:
            return _candidate_device(
                candidate,
                local_rank=local_rank,
                world_size=world_size,
                initialized_backend=initialized_backend,
                required_kernel_backend=request.required_kernel_backend,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            failures.append(f"{str(candidate)!r}: {error}")
    raise RuntimeError(
        "none of the allowed distributed devices passed preflight: "
        + "; ".join(failures)
    )


_DEVICE_CONSENSUS_PREFIX = "hydroforge/device_consensus"


def _device_identity(device: torch.device) -> str:
    """Physical identity of one bound accelerator, comparable across ranks."""

    runtime = getattr(torch, device.type)
    try:
        uuid = getattr(runtime.get_device_properties(device.index), "uuid", None)
    except Exception:  # noqa: BLE001
        # Best effort only: a rank that raised here would skip the consensus
        # and leave its peers waiting.
        uuid = None
    if uuid is not None:
        return f"{socket.gethostname()}/{device.type}/{uuid}"
    # Without a UUID, a per-process visibility mask still tells per-task
    # binding (different masks) apart from sharing (same mask and index).
    mask = os.environ.get(
        "CUDA_VISIBLE_DEVICES" if device.type == "cuda" else "ZE_AFFINITY_MASK"
    )
    return f"{socket.gethostname()}/{device.type}/mask={mask}/{device.index}"


def _device_consensus(
    device_types: tuple[str, ...],
    local_failures: dict[str, str],
    identities: dict[str, str],
    *,
    rank: int,
    world_size: int,
) -> tuple[str | None, dict[str, tuple[int, str]], tuple[int, str] | None, Any]:
    """Agree on the first device type that passed preflight on every rank.

    Uses the ``env://`` rendezvous store that the default group will reuse.
    Every rank publishes its result, including ranks with no usable candidate,
    so an unusable topology fails on all ranks instead of hanging in
    ``init_process_group``. Accelerator ranks also publish the physical
    identity of their bound device; the third result reports how many ranks
    share one device of the agreed type (NCCL/XCCL need one device per rank).
    The returned store must stay referenced until the default group exists:
    rank 0 hosts the shared TCP store server.
    """

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    store, store_rank, store_world_size = next(
        dist.rendezvous("env://", timeout=dist.default_pg_timeout)
    )
    if store_rank != rank or store_world_size != world_size:
        raise RuntimeError(
            f"env:// rendezvous returned rank={store_rank}, "
            f"world_size={store_world_size}; expected rank={rank}, "
            f"world_size={world_size}"
        )
    # Repeated setup calls within one launcher attempt share the store.
    call = store.add(f"{_DEVICE_CONSENSUS_PREFIX}/calls/{rank}", 1)
    scoped = dist.PrefixStore(f"{_DEVICE_CONSENSUS_PREFIX}/{call}", store)
    for device_type in device_types:
        failure = local_failures.get(device_type)
        scoped.add(f"available/{device_type}", 0 if failure is not None else 1)
        if failure is not None:
            scoped.compare_set(
                f"failure/{device_type}", "", f"rank {rank}: {failure}"
            )
    for device_type, identity in identities.items():
        scoped.add(f"identity/{device_type}/{identity}", 1)
    if scoped.add("arrived", 1) == world_size:
        scoped.set("complete", "1")
    scoped.wait(["complete"])
    report: dict[str, tuple[int, str]] = {}
    agreed: str | None = None
    for device_type in device_types:
        missing = world_size - scoped.add(f"available/{device_type}", 0)
        if missing == 0:
            agreed = agreed or device_type
            continue
        first = scoped.get(f"failure/{device_type}").decode(errors="replace")
        report[device_type] = (missing, first)
    shared: tuple[int, str] | None = None
    if agreed in identities:
        identity = identities[agreed]
        users = scoped.add(f"identity/{agreed}/{identity}", 0)
        if users > 1:
            scoped.add("shared", 1)
            scoped.compare_set(
                "shared_example", "", f"{identity} is bound by {users} ranks"
            )
        if scoped.add("checked", 1) == world_size:
            scoped.set("checked_complete", "1")
        scoped.wait(["checked_complete"])
        sharing_ranks = scoped.add("shared", 0)
        if sharing_ranks:
            shared = (
                sharing_ranks,
                scoped.get("shared_example").decode(errors="replace"),
            )
    # Rank 0 hosts the server: it must outlive every rank's reads, including
    # when all ranks are about to raise.
    if scoped.add("departed", 1) == world_size:
        scoped.set("released", "1")
    if rank == 0:
        scoped.wait(["released"])
    return agreed, report, shared, store


def _select_agreed_device(
    request: _DistributedSetupRequest,
    *,
    local_rank: int,
    rank: int,
    world_size: int,
) -> tuple[torch.device, Any]:
    """Select the first device type that is usable on every process."""

    device_types = tuple(dict.fromkeys(device.type for device in request.allowed_devices))
    selected: dict[str, torch.device] = {}
    failures: dict[str, list[str]] = {}
    for candidate in request.allowed_devices:
        if candidate.type in selected:
            continue
        try:
            selected[candidate.type] = _candidate_device(
                candidate,
                local_rank=local_rank,
                world_size=world_size,
                initialized_backend=None,
                required_kernel_backend=request.required_kernel_backend,
            )
        except Exception as error:  # noqa: BLE001
            # Any local failure (e.g. a deferred CUDA init error) is published
            # to the peers instead of leaving them waiting in the consensus.
            failures.setdefault(candidate.type, []).append(
                f"{str(candidate)!r}: {error}"
            )
    local_failures = {
        device_type: "; ".join(failures[device_type])
        for device_type in device_types
        if device_type not in selected
    }
    identities = {
        device_type: _device_identity(device)
        for device_type, device in selected.items()
        if device_type in {"cuda", "xpu"}
    }
    agreed, report, shared, store = _device_consensus(
        device_types,
        local_failures,
        identities,
        rank=rank,
        world_size=world_size,
    )
    if shared is not None:
        sharing_ranks, example = shared
        raise RuntimeError(
            f"{sharing_ranks} of {world_size} ranks share a physical "
            f"{agreed.upper()} device ({example}); "
            f"{_COMMUNICATION_BACKENDS[agreed].upper()} requires one device per "
            "rank. Bind one device per process (e.g. --gpus-per-task=1 or a "
            "per-rank visibility mask) or start fewer processes per node"
        )
    if agreed is not None:
        return selected[agreed], store
    if not selected:
        raise RuntimeError(
            "none of the allowed distributed devices passed preflight: "
            + "; ".join(item for items in failures.values() for item in items)
        )
    raise RuntimeError(
        f"no allowed distributed device type passed preflight on all "
        f"{world_size} ranks; all ranks must select the same communication "
        "backend: "
        + "; ".join(
            f"{device_type!r} failed on {missing} rank(s), first {first}"
            for device_type, (missing, first) in report.items()
        )
    )


def _rendezvous_environment(world_size: int) -> int:
    """Validate the complete ``env://`` rendezvous before creating a group."""

    _, rank = _rank_environment(world_size, required=True)
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr is None or not master_addr.strip():
        raise ValueError("MASTER_ADDR must be a non-empty string for env:// rendezvous")
    raw_port = os.environ.get("MASTER_PORT")
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


def _setup_distributed_trusted(
    request: _DistributedSetupRequest,
) -> DistributedContext:
    """Initialize one already validated distributed request."""

    raw_world_size, ws_env = _world_size_environment()
    local_rank = get_local_process_rank()
    initialized = dist.is_available() and dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        ProcessTopology(rank=rank, world_size=world_size)
        if raw_world_size is not None and ws_env != world_size:
            raise ValueError(
                f"WORLD_SIZE={ws_env} disagrees with the initialized process "
                f"group world_size={world_size}"
            )
        raw_rank, rank_env = _rank_environment(world_size, required=False)
        if raw_rank is not None and rank_env != rank:
            raise ValueError(
                f"RANK={rank_env} disagrees with the initialized process "
                f"group rank={rank}"
            )
        observed_backend = _backend_name(dist.get_backend())
        if observed_backend not in {"gloo", "nccl", "xccl"}:
            raise RuntimeError(
                f"initialized communication backend {observed_backend!r} is "
                "unsupported; HydroForge supports Gloo, NCCL/RCCL, and XCCL"
            )
        backend: Literal["gloo", "nccl", "xccl"] | None = observed_backend
    else:
        if ws_env == 1 and local_rank != 0:
            raise ValueError(
                f"local rank is {local_rank}, but WORLD_SIZE={ws_env}; "
                "launcher topology is incomplete"
            )
        raw_rank, rank_env = _rank_environment(ws_env, required=False)
        if raw_rank is not None and rank_env != 0 and ws_env == 1:
            raise ValueError(
                f"RANK={rank_env}, but WORLD_SIZE=1; launcher topology is incomplete"
            )
        rank = 0
        world_size = ws_env
        backend = None

    if local_rank >= world_size:
        raise ValueError(
            f"LOCAL_RANK={local_rank} must be smaller than world_size={world_size}"
        )
    expected_rank: int | None = None
    if not initialized and world_size > 1:
        # Validate the env:// rendezvous before activating any process-local
        # accelerator. Invalid launcher state must be side-effect free.
        expected_rank = _rendezvous_environment(world_size)
    consensus_store: Any = None
    candidate_types = {device.type for device in request.allowed_devices}
    if expected_rank is not None and (
        len(candidate_types) > 1 or candidate_types & {"cuda", "xpu"}
    ):
        # Independent per-rank fallback could pick different communication
        # backends, and ranks sharing one accelerator break NCCL/XCCL; both
        # would otherwise surface only as a hang or failure at init time.
        device, consensus_store = _select_agreed_device(
            request,
            local_rank=local_rank,
            rank=expected_rank,
            world_size=world_size,
        )
    else:
        device = _select_distributed_device(
            request,
            local_rank=local_rank,
            world_size=world_size,
            initialized_backend=backend,
        )
    if not initialized and world_size > 1:
        assert expected_rank is not None
        backend = _require_communication_backend(device)
        # Construct the complete result before the external process-group side
        # effect. This proves that all locally predictable contract failures
        # have already been raised.
        context = DistributedContext(
            local_rank=local_rank,
            rank=expected_rank,
            world_size=world_size,
            device=device,
            backend=backend,
        )
        arguments: dict[str, Any] = {
            "backend": backend,
            "init_method": "env://",
        }
        if device.type in {"cuda", "xpu"}:
            arguments["device_id"] = device
        try:
            dist.init_process_group(**arguments)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            observed_backend = _backend_name(dist.get_backend())
            if rank != expected_rank:
                raise RuntimeError(
                    f"initialized rank={rank} disagrees with preflight RANK={expected_rank}"
                )
            if world_size != ws_env:
                raise RuntimeError(
                    f"initialized world_size={world_size} disagrees with "
                    f"preflight WORLD_SIZE={ws_env}"
                )
            if observed_backend != backend:
                raise RuntimeError(
                    f"initialized communication backend {observed_backend!r} "
                    f"disagrees with preflight backend {backend!r}"
                )
            ProcessTopology(rank=rank, world_size=world_size)
        except BaseException:

            def release_owned_group() -> None:
                if dist.is_initialized():
                    dist.destroy_process_group()

            with cleanup_on_exit("new process group", (release_owned_group,)):
                raise
        del consensus_store
        return context

    return DistributedContext(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def setup_distributed(
    *,
    allowed_devices: tuple[str | torch.device, ...],
    required_kernel_backend: Literal["torch", "triton", "cuda", "metal"] | None = None,
) -> DistributedContext:
    """Select a device, preflight kernels, and initialize distributed execution.

    ``allowed_devices`` is an explicit, ordered policy: the first candidate
    that is available, communication-compatible, and kernel-compatible is
    selected. When a new multi-process group is created from several device
    types or from accelerator candidates, the ranks first agree through the
    ``env://`` store on the first device type that passed preflight on every
    rank, so all ranks use one communication backend, and reject ranks that
    share one physical accelerator. CUDA/ROCm and XPU candidates are bound to
    ``LOCAL_RANK`` (index 0 when each process sees exactly one device) unless
    an explicit matching index is supplied. Multi-process CPU, CUDA/ROCm, and
    XPU execution uses Gloo, NCCL/RCCL, and XCCL respectively; MPS is accepted
    only for a single process. TPU/XLA requires a separate PJRT adapter.

    ``required_kernel_backend`` optionally requires HydroForge's resolved
    compute backend (for example ``"triton"``) before any process group is
    initialized. :attr:`DistributedContext.backend` is deliberately separate:
    it names the communication backend, or is ``None`` when no group exists.
    """

    request = _DistributedSetupRequest(
        allowed_devices=allowed_devices,
        required_kernel_backend=required_kernel_backend,
    )
    return _setup_distributed_trusted(request)


# ---------------------------------------------------------------------------
# Binary / map file I/O (used by dataset implementations)
# ---------------------------------------------------------------------------


class _BinaryReadRequest(HydroForgeModel):
    filename: Path = Field(strict=False)
    shape: tuple[Annotated[int, Field(ge=1)], ...] = Field(min_length=1)
    dtype: Any

    _normalized_dtype: np.dtype = PrivateAttr()
    _file_identity: tuple[int, int, int, int] = PrivateAttr()

    @model_validator(mode="after")
    def _validate_binary_identity(self):
        dtype = np.dtype(self.dtype)
        if dtype.hasobject or dtype.subdtype is not None or dtype.fields is not None:
            raise ValueError("binary dtype must be a plain fixed-width scalar dtype")
        self._normalized_dtype = dtype
        path = Path(self.filename).absolute()
        expected_size = prod(self.shape) * dtype.itemsize
        status = path.stat()
        if status.st_size > expected_size:
            raise ValueError(
                f"binary file {path} has trailing data: {status.st_size} "
                f"bytes for an expected {expected_size}-byte shape "
                f"{self.shape} and dtype {dtype.str!r}"
            )
        object.__setattr__(self, "filename", path)
        self._file_identity = (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
        )
        return self

    @property
    def normalized_dtype(self) -> np.dtype:
        return self._normalized_dtype

    def verify_file_identity(self) -> None:
        path = Path(self.filename)
        try:
            status = path.stat()
        except OSError as error:
            raise RuntimeError(
                f"binary file {path} changed after validation"
            ) from error
        observed = (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
        )
        if observed != self._file_identity:
            raise RuntimeError(f"binary file {path} changed after validation")


def _binread_trusted(request: _BinaryReadRequest) -> np.ndarray:
    request.verify_file_identity()
    try:
        array = np.fromfile(
            request.filename,
            dtype=request.normalized_dtype,
            count=prod(request.shape),
        )
    finally:
        request.verify_file_identity()
    return array.reshape(request.shape, order="F")


def binread(
    filename: str | Path,
    shape: tuple[int, ...],
    dtype_str: str | np.dtype,
) -> np.ndarray:
    """Read a Fortran-ordered binary file and reshape to *shape*."""
    request = _BinaryReadRequest(
        filename=filename,
        shape=shape,
        dtype=dtype_str,
    )
    return _binread_trusted(request)


class _MapReadRequest(_BinaryReadRequest):
    shape: tuple[Annotated[int, Field(ge=1)], ...] = Field(min_length=2, max_length=3)


def read_map(
    filename: str | Path,
    map_shape: tuple[int, ...],
    precision: str | np.dtype,
) -> np.ndarray:
    """Read a spatial map binary file (Fortran-ordered)."""
    request = _MapReadRequest(
        filename=filename,
        shape=map_shape,
        dtype=precision,
    )
    return _binread_trusted(request)


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------


class _NumpyIndexLookup(HydroForgeModel):
    """Validated declaration consumed by :func:`find_indices_in`."""

    query: np.ndarray
    target: np.ndarray

    @model_validator(mode="after")
    def _validate_lookup(self) -> Self:
        if np.ma.isMaskedArray(self.query) or np.ma.isMaskedArray(self.target):
            raise ValueError("index lookup arrays must not be masked arrays")
        if self.query.ndim != 1 or self.target.ndim != 1:
            raise ValueError("index lookup arrays must be one-dimensional")
        if self.query.dtype.kind not in {"i", "u"} or self.target.dtype.kind not in {
            "i",
            "u",
        }:
            raise ValueError("index lookup arrays must contain integers")
        if self.query.dtype != self.target.dtype:
            raise ValueError("index lookup arrays must have identical dtypes")
        if np.unique(self.target).size != self.target.size:
            raise ValueError("index lookup target must contain unique values")
        return self


def find_indices_in(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return indices in *b* for each element of *a* (NumPy version)."""
    declaration = _NumpyIndexLookup(query=a, target=b)
    return _find_indices_in_trusted(declaration.query, declaration.target)


def _find_indices_in_trusted(
    query: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Resolve an already validated one-dimensional integer lookup."""

    order = np.argsort(target)
    sorted_b = target[order]
    pos_in_sorted = np.searchsorted(sorted_b, query)
    valid_mask = pos_in_sorted < len(sorted_b)
    hit_mask = np.zeros_like(query, dtype=bool)
    hit_mask[valid_mask] = sorted_b[pos_in_sorted[valid_mask]] == query[valid_mask]
    index = np.full_like(pos_in_sorted, -1, dtype=int)
    index[hit_mask] = order[pos_in_sorted[hit_mask]]
    return index


class _TorchIndexLookup(HydroForgeModel):
    """Validated declaration consumed by :func:`find_indices_in_torch`."""

    query: torch.Tensor
    target: torch.Tensor

    @model_validator(mode="after")
    def _validate_lookup(self) -> Self:
        if self.query.ndim != 1 or self.target.ndim != 1:
            raise ValueError("torch index lookup tensors must be one-dimensional")
        integer_dtypes = {
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.uint16,
            torch.int32,
            torch.uint32,
            torch.int64,
        }
        if (
            self.query.dtype not in integer_dtypes
            or self.target.dtype not in integer_dtypes
        ):
            raise ValueError("torch index lookup tensors must contain integers")
        if self.query.dtype != self.target.dtype:
            raise ValueError("torch index lookup tensors must have identical dtypes")
        if not devices_match(self.query.device, self.target.device):
            raise ValueError("torch index lookup tensors must share one device")
        unique_target = (
            self.target.to(torch.int64)
            if self.target.dtype in {torch.uint16, torch.uint32}
            else self.target
        )
        if torch.unique(unique_target).numel() != self.target.numel():
            raise ValueError("torch index lookup target must contain unique values")
        if self.target.numel() > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"b has {self.target.numel()} elements, exceeding int32 range. "
                "find_indices_in_torch returns int32 indices."
            )
        return self


def find_indices_in_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return indices in *b* for each element of *a* (Torch version)."""
    declaration = _TorchIndexLookup(query=a, target=b)
    return _find_indices_in_torch_trusted(
        declaration.query,
        declaration.target,
    )


def _find_indices_in_torch_trusted(
    query: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Resolve an already validated same-device integer tensor lookup."""

    if target.numel() == 0:
        return torch.full_like(query, -1, dtype=torch.int32)
    if target.dtype in {torch.uint16, torch.uint32}:
        lookup_query = query.to(torch.int64)
        lookup_target = target.to(torch.int64)
    else:
        lookup_query = query
        lookup_target = target
    sorted_b, order = torch.sort(lookup_target)
    pos = torch.bucketize(lookup_query, sorted_b, right=False)
    # bucketize on MPS/some GPU backends may return len(sorted_b) for values
    # that equal the last element; clamp to keep indexing safe — the equality
    # check below still rejects true misses.
    pos = pos.clamp(max=len(sorted_b) - 1)
    hit_mask = sorted_b[pos] == lookup_query
    index = torch.full_like(query, -1, dtype=torch.int32)
    index[hit_mask] = order[pos[hit_mask]].to(torch.int32)
    return index


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


_TORCH_NUMPY_DTYPES = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}


class _TorchDTypeLookup(HydroForgeModel):
    """Validated declaration consumed by :func:`torch_to_numpy_dtype`."""

    torch_dtype: torch.dtype

    @model_validator(mode="after")
    def _validate_dtype(self) -> Self:
        if self.torch_dtype not in _TORCH_NUMPY_DTYPES:
            raise ValueError(f"Unsupported torch dtype: {self.torch_dtype}")
        return self


def torch_to_numpy_dtype(torch_dtype: torch.dtype) -> type:
    torch_dtype = _TorchDTypeLookup(torch_dtype=torch_dtype).torch_dtype
    return _TORCH_NUMPY_DTYPES[torch_dtype]

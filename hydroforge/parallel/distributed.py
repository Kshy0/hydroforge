# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
hydroforge.parallel.distributed

Process topology and one strict distributed setup call: ordered device
selection, kernel-backend preflight, cross-rank device consensus, and
process-group initialization.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from typing import Any, Literal, NoReturn, Self

import torch
from pydantic import Field, field_validator, model_validator
from torch import distributed as dist

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.core.validation import HydroForgeModel
from hydroforge.parallel.launch import (
    _local_process_count,
    _rank_environment,
    _rendezvous_environment,
    _world_size_environment,
    get_local_process_rank,
    require_launcher_environment,
)
from hydroforge.parallel.launch import (
    communication_backend as _backend_name,
)
from hydroforge.platform.backend import backend_named, resolve_backend

_COMMUNICATION_BACKENDS = {"cpu": "gloo", "mps": "gloo", "cuda": "nccl", "xpu": "xccl"}


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
                    "multi-process accelerator device index must not exceed local_rank"
                )
        if self.device.type == "mps":
            if self.device.index not in {None, 0}:
                raise ValueError("MPS exposes only device index 0")
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
        if self.required_kernel_backend is None:
            return self
        backend = backend_named(self.required_kernel_backend)
        if backend.devices is not None and not any(
            backend.accepts(device) for device in self.allowed_devices
        ):
            raise ValueError(
                f"required kernel backend {self.required_kernel_backend!r} "
                f"has no compatible candidate in allowed_devices"
            )
        return self


def is_rank_zero() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _local_device_index(
    local_rank: int,
    device_count: int,
    *,
    device_type: str,
    visibility: str | None,
    environment: Mapping[str, str] | None = None,
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
    local_count = _local_process_count(environment)
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
    environment: Mapping[str, str] | None = None,
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
        (os.environ if environment is None else environment).get("CUDA_VISIBLE_DEVICES")
        if device_type == "cuda"
        else (os.environ if environment is None else environment).get(
            "ZE_AFFINITY_MASK"
        )
    )
    if world_size > 1:
        bound = _local_device_index(
            local_rank,
            device_count,
            device_type=device_type,
            visibility=visibility,
            environment=environment,
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

    resolved = resolve_backend(device).name
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
    environment: Mapping[str, str] | None = None,
) -> torch.device:
    """Validate, activate, and kernel-preflight one ordered candidate."""

    compatible_types = tuple(
        kind
        for kind, backend in _COMMUNICATION_BACKENDS.items()
        if backend == initialized_backend
    )
    if compatible_types and candidate.type not in compatible_types:
        raise RuntimeError(
            f"initialized {initialized_backend.upper()} communication backend "
            f"requires a device from {compatible_types!r}"
        )

    if candidate.type == "cpu":
        device = torch.device("cpu")
    elif candidate.type in {"cuda", "xpu"}:
        device = _accelerator_candidate(
            candidate,
            local_rank=local_rank,
            world_size=world_size,
            environment=environment,
        )
    else:
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
    environment: Mapping[str, str] | None = None,
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
                environment=environment,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            failures.append(f"{str(candidate)!r}: {error}")
    raise RuntimeError(
        "none of the allowed distributed devices passed preflight: "
        + "; ".join(failures)
    )


_DEVICE_CONSENSUS_PREFIX = "hydroforge/device_consensus"


def _device_identity(
    device: torch.device, environment: Mapping[str, str] | None = None
) -> str:
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
    mask = (os.environ if environment is None else environment).get(
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
    environment: Mapping[str, str] | None = None,
) -> tuple[str | None, dict[str, tuple[int, str]], tuple[int, str] | None, Any]:
    """Agree on the first device type that passed preflight on every rank.

    Uses the ``env://`` rendezvous store that the default group will reuse.
    Every rank publishes its ordered device types and its result, including
    ranks with no usable candidate, so differing declarations or an unusable
    topology fail on all ranks instead of hanging in the store or in
    ``init_process_group``. Accelerator ranks also publish the physical
    identity of their bound device; the third result reports how many ranks
    share one device of the agreed type (NCCL/XCCL need one device per rank).
    The returned store must stay referenced until the default group exists:
    rank 0 hosts the shared TCP store server.
    """

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if environment is not None:
        require_launcher_environment(environment)
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
    declared = ",".join(device_types)
    scoped.add(f"declared/{declared}", 1)
    scoped.compare_set("declared_example", "", f"rank {rank}: ({declared})")
    for device_type in device_types:
        failure = local_failures.get(device_type)
        scoped.add(f"available/{device_type}", 0 if failure is not None else 1)
        if failure is not None:
            scoped.compare_set(f"failure/{device_type}", "", f"rank {rank}: {failure}")
    for device_type, identity in identities.items():
        scoped.add(f"identity/{device_type}/{identity}", 1)
    if scoped.add("arrived", 1) == world_size:
        scoped.set("complete", "1")
    scoped.wait(["complete"])
    # Each rank sees a full count for its own declaration exactly when all
    # ranks declared the same order, so every rank takes the same decision.
    same = scoped.add(f"declared/{declared}", 0)
    mismatch: str | None = None
    if same != world_size:
        example = scoped.get("declared_example").decode(errors="replace")
        mismatch = (
            f"rank {rank} declared ({declared}) with {same} of {world_size} "
            f"ranks, first declaration {example}"
        )
    candidates = device_types if mismatch is None else ()
    report: dict[str, tuple[int, str]] = {}
    agreed: str | None = None
    for device_type in candidates:
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
    if mismatch is not None:
        raise RuntimeError(
            "ranks declared different allowed device type orders; all ranks "
            f"must pass the same allowed_devices policy: {mismatch}"
        )
    return agreed, report, shared, store


def _select_agreed_device(
    request: _DistributedSetupRequest,
    *,
    local_rank: int,
    rank: int,
    world_size: int,
    environment: Mapping[str, str] | None = None,
) -> tuple[torch.device, Any]:
    """Select the first device type that is usable on every process."""

    device_types = tuple(
        dict.fromkeys(device.type for device in request.allowed_devices)
    )
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
                environment=environment,
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
        device_type: _device_identity(device, environment)
        for device_type, device in selected.items()
        if device_type in {"cuda", "xpu"}
    }
    agreed, report, shared, store = _device_consensus(
        device_types,
        local_failures,
        identities,
        rank=rank,
        world_size=world_size,
        environment=environment,
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


def _setup_distributed_trusted(
    request: _DistributedSetupRequest,
) -> DistributedContext:
    """Initialize one already validated distributed request."""

    environment = dict(os.environ)
    raw_world_size, ws_env = _world_size_environment(environment)
    local_rank = get_local_process_rank(environment)
    local_count = _local_process_count(environment)
    if local_count is not None and local_rank >= local_count:
        raise ValueError(
            f"local rank {local_rank} must be smaller than local process count {local_count}"
        )
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
        raw_rank, rank_env = _rank_environment(
            world_size, required=False, environment=environment
        )
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
        raw_rank, rank_env = _rank_environment(
            ws_env, required=False, environment=environment
        )
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
        expected_rank = _rendezvous_environment(world_size, environment)
    consensus_store: Any = None
    if expected_rank is not None:
        # Independent per-rank fallback could pick different communication
        # backends, and ranks sharing one accelerator break NCCL/XCCL; both
        # would otherwise surface only as a hang or failure at init time.
        # Every rank joins, whatever it declared, so that a rank-local
        # policy difference cannot leave its peers waiting in the store.
        device, consensus_store = _select_agreed_device(
            request,
            local_rank=local_rank,
            rank=expected_rank,
            world_size=world_size,
            environment=environment,
        )
    else:
        device = _select_distributed_device(
            request,
            local_rank=local_rank,
            world_size=world_size,
            initialized_backend=backend,
            environment=environment,
        )
    require_launcher_environment(environment)
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
    selected. When a new multi-process group is created, the ranks first
    agree through the ``env://`` store on the first device type that passed
    preflight on every rank, so all ranks use one communication backend, and
    reject ranks that share one physical CUDA/XPU accelerator. Every rank
    must declare the same device type order; a difference fails on every rank. CUDA/ROCm
    and XPU candidates are bound to ``LOCAL_RANK`` (index 0 when each process
    sees exactly one device) unless an explicit matching index is supplied.
    Multi-process CPU, CUDA/ROCm, and XPU execution uses Gloo, NCCL/RCCL, and
    XCCL respectively; MPS uses Gloo with CPU staging for collectives,
    while physics remains on the shared MPS device. TPU/XLA
    requires a separate PJRT adapter.

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

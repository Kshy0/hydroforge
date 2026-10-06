# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Explicit distributed operators for compiled substeps.

Every collective goes through one batched path: a batch costs a single
managed-step handshake and, when the communication backend supports it, one
coalescing group rather than one of each per tensor. ``reduce_`` and
``all_reduce_`` are its one-tensor spellings.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from pydantic import Field, PrivateAttr, model_validator

from hydroforge.core.identity import digest63
from hydroforge.core.validation import HydroForgeModel
from hydroforge.execution.channel import (
    ENSEMBLE_COLLECTIVE_FLAG,
    require_process_group,
)
from hydroforge.execution.context import ACTIVE_STEP
from hydroforge.kernels.calls import compiled_operator_entry, recording_sink
from hydroforge.parallel.launch import communication_backend
from hydroforge.parallel.mesh import ParallelAxis

Reduction = Literal["min", "max", "sum"]

_DTYPE_CODES = {
    dtype: index
    for index, dtype in enumerate(
        (
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ),
        start=1,
    )
}
# MPS physics communicates through Gloo using explicit CPU staging.
_ABI_DEVICE_CODES = {"cpu": 1, "cuda": 2, "xpu": 3, "mps": 4}
_REDUCTIONS = {
    "min": (0, dist.ReduceOp.MIN),
    "max": (1, dist.ReduceOp.MAX),
    "sum": (2, dist.ReduceOp.SUM),
}


class _CollectiveRequest(HydroForgeModel):
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor]
    operation: Literal["all_reduce", "reduce"]
    reduction: Reduction
    destination: int | None = Field(default=None, ge=0)
    scope: ParallelAxis = "spatial"

    _abis: tuple[tuple[int, int, int], ...] = PrivateAttr()

    @model_validator(mode="after")
    def _validate_collective(self):
        batch = tuple(self.tensors)
        if self.operation == "all_reduce" and self.destination is not None:
            raise ValueError("all_reduce does not accept a destination")
        if self.operation == "reduce" and self.destination is None:
            raise ValueError("reduce requires a destination")
        devices = {tensor.device for tensor in batch}
        if len(devices) > 1:
            raise ValueError(
                f"{self.operation} tensors must share one device, got "
                f"{sorted(map(str, devices))}"
            )
        self._abis = tuple(
            _tensor_abi(tensor, operation=self.operation) for tensor in batch
        )
        object.__setattr__(self, "tensors", batch)
        return self

    @property
    def abis(self) -> tuple[tuple[int, int, int], ...]:
        return self._abis


def _tensor_abi(
    tensor: torch.Tensor,
    *,
    operation: str,
) -> tuple[int, int, int]:
    """Validate and encode the process-group-independent tensor ABI."""

    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"{operation} tensor must be contiguous and strided")
    if tensor.numel() < 1:
        raise ValueError(f"{operation} tensor must be non-empty")
    try:
        dtype_code = _DTYPE_CODES[tensor.dtype]
    except KeyError as error:
        raise ValueError(
            f"{operation} does not support tensor dtype {tensor.dtype}"
        ) from error
    try:
        device_code = _ABI_DEVICE_CODES[tensor.device.type]
    except KeyError as error:
        raise ValueError(
            f"{operation} does not support device {tensor.device.type!r}"
        ) from error
    return dtype_code, tensor.numel(), device_code


def _batch_signature(
    abis: Sequence[tuple[int, int, int]],
    reduction: Reduction,
    destination: int | None,
) -> tuple[int, int, int]:
    """Fold a whole batch into the three-int managed-step signature.

    Slots 0 and 2 carry the batch length and total element count; slot 1 hashes
    each tensor's ABI in order, so any cross-rank difference is still rejected.
    """

    digest = digest63((_REDUCTIONS[reduction][0], destination, tuple(abis)))
    return len(abis), digest, sum(abi[1] for abi in abis)


def _validate_collective_environment(
    device: torch.device | None,
    *,
    operation: str,
    destination: int | None = None,
    group=None,
    group_size: int | None = None,
) -> str | None:
    """Validate batch-invariant process-group state; return its backend."""

    if group_size != 1:
        require_process_group(operation)
    group_kwargs = {} if group is None else {"group": group}
    if destination is not None:
        size = dist.get_world_size(**group_kwargs) if group_size is None else group_size
        if not 0 <= destination < size:
            raise ValueError(f"{operation} destination is outside the process group")
    if device is None or group_size == 1:
        return None
    backend = communication_backend(
        dist.get_backend(**group_kwargs), device_type=device.type
    )
    required_device = {"nccl": "cuda", "xccl": "xpu"}
    for backend_name, device_type in required_device.items():
        if backend_name == backend and device.type != device_type:
            raise ValueError(
                f"{operation} with {backend_name.upper()} requires a "
                f"{device_type.upper()} tensor"
            )
    required_backend = {"cuda": "nccl", "xpu": "xccl", "mps": "gloo"}.get(
        device.type,
    )
    if required_backend is not None and required_backend != backend:
        raise ValueError(
            f"{operation} of a {device.type.upper()} tensor requires "
            f"the {required_backend.upper()} process-group backend, got "
            f"{backend}"
        )
    return backend


def _event_kind(
    operation: str,
    reduction: Reduction,
    destination: int | None = None,
) -> int:
    reduction_code = _REDUCTIONS[reduction][0]
    if operation == "all_reduce":
        return 10 + reduction_code
    return 100 + destination * 3 + reduction_code


def _coalescing_group(device: torch.device, backend: str | None, group=None):
    """Group the batch into one NCCL submission when the backend allows it."""

    manager = getattr(dist, "_coalescing_manager", None)
    if manager is None or device.type != "cuda" or backend != "nccl":
        return nullcontext()
    group_kwargs = {} if group is None else {"group": group}
    return manager(device=device, async_ops=False, **group_kwargs)


@dataclass(frozen=True, slots=True)
class _BatchEnvironment:
    """Process-group facts of one recorded batch, resolved once."""

    group: Any
    group_size: int | None
    backend: str | None
    global_destination: int | None
    receives: bool


# (mesh id, scope, operation, destination, device) -> (mesh, world, environment).
# The mesh and default group are compared by identity, so a rebuilt mesh or a
# re-initialized process group resolves afresh.
_ENVIRONMENTS: dict[tuple[Any, ...], tuple[Any, Any, _BatchEnvironment]] = {}


def _batch_environment(
    mesh: Any,
    *,
    scope: ParallelAxis,
    operation: str,
    destination: int | None,
    device: torch.device | None,
) -> _BatchEnvironment:
    world = dist.group.WORLD if dist.is_available() and dist.is_initialized() else None
    key = (id(mesh), scope, operation, destination, device)
    cached = _ENVIRONMENTS.get(key)
    if cached is not None and cached[0] is mesh and cached[1] is world:
        return cached[2]
    if mesh is None and scope == "ensemble":
        raise ValueError("ensemble collectives require an EnsembleParallel mesh")
    group = None if mesh is None else mesh.group(scope)
    group_size = (
        None
        if mesh is None
        else mesh.spatial_partitions
        if scope == "spatial"
        else mesh.ensemble_partitions
    )
    backend = _validate_collective_environment(
        device,
        operation=operation,
        destination=destination,
        group=group,
        group_size=group_size,
    )
    global_destination = destination
    if destination is not None and mesh is not None:
        ranks = mesh.rank_groups(scope)[
            mesh.ensemble_rank if scope == "spatial" else mesh.spatial_rank
        ]
        global_destination = ranks[destination]
    receives = destination is None or (
        group_size != 1 and dist.get_rank() == global_destination
    )
    environment = _BatchEnvironment(
        group, group_size, backend, global_destination, receives
    )
    # Keep only the current mesh and process group: a stale entry would
    # otherwise retain a released mesh and its groups.
    for stale in [
        other
        for other, (owner, group_world, _environment) in _ENVIRONMENTS.items()
        if owner is not mesh or group_world is not world
    ]:
        del _ENVIRONMENTS[stale]
    _ENVIRONMENTS[key] = (mesh, world, environment)
    return environment


def _run_validated_batch(
    tensors: tuple[torch.Tensor, ...],
    abis: tuple[tuple[int, int, int], ...],
    *,
    operation: str,
    reduction: Reduction,
    destination: int | None,
    scope: ParallelAxis = "spatial",
) -> None:
    """Synchronize once and launch one already validated batch."""

    step = ACTIVE_STEP.get()
    if step is None:
        raise RuntimeError(
            "HydroForge collectives may be called only inside a managed step "
            "or an operator recorder"
        )
    _code, op = _REDUCTIONS[reduction]
    environment = _batch_environment(
        step.mesh,
        scope=scope,
        operation=operation,
        destination=destination,
        device=tensors[0].device if tensors else None,
    )
    # The handshake runs even for an empty batch: a rank that contributes no
    # tensors must still be seen to disagree with one that does.
    step.channel.event(
        _event_kind(operation, reduction, destination)
        | (ENSEMBLE_COLLECTIVE_FLAG if scope == "ensemble" else 0),
        _batch_signature(abis, reduction, destination),
    )
    if not tensors or environment.group_size == 1:
        return
    group = environment.group
    group_kwargs = {} if group is None else {"group": group}
    with _coalescing_group(tensors[0].device, environment.backend, **group_kwargs):
        for tensor in tensors:
            # cpu() decodes emulated storage; never reduce its integer carrier.
            staging = tensor.cpu() if tensor.device.type == "mps" else tensor
            if destination is None:
                dist.all_reduce(staging, op=op, **group_kwargs)
            else:
                dist.reduce(
                    staging,
                    dst=environment.global_destination,
                    op=op,
                    **group_kwargs,
                )
            if staging is not tensor and environment.receives:
                tensor.copy_(staging)


def _submit_collective(request: _CollectiveRequest) -> None:
    recorder = recording_sink()
    if recorder is not None:
        recorder.record_collective_batch(
            request.tensors,
            request.abis,
            request.reduction,
            operation=request.operation,
            destination=request.destination,
            scope=request.scope,
        )
        return
    _run_validated_batch(
        request.tensors,
        request.abis,
        operation=request.operation,
        reduction=request.reduction,
        destination=request.destination,
        scope=request.scope,
    )


def launch_recorded_collective_batch(
    tensors: tuple[torch.Tensor, ...],
    abis: tuple[tuple[int, int, int], ...],
    *,
    operation: str,
    reduction: Reduction,
    destination: int | None,
    scope: ParallelAxis = "spatial",
) -> None:
    """Replay one compiled batch without repeating its tensor ABI checks."""

    _run_validated_batch(
        tensors,
        abis,
        operation=operation,
        reduction=reduction,
        destination=destination,
        scope=scope,
    )


@compiled_operator_entry
def all_reduce_(
    tensor: torch.Tensor, *, reduction: Reduction, scope: ParallelAxis = "spatial"
) -> None:
    """Apply an in-place distributed reduction as an explicit IR operator.

    Unlike calling ``torch.distributed`` directly inside a lexical substep,
    this operation is recorded once and replayed on every physical iteration.
    """

    _submit_collective(
        _CollectiveRequest(
            tensors=(tensor,),
            operation="all_reduce",
            reduction=reduction,
            scope=scope,
        )
    )


@compiled_operator_entry
def all_reduce_many_(
    tensors: Sequence[torch.Tensor],
    *,
    reduction: Reduction = "sum",
    scope: ParallelAxis = "spatial",
) -> None:
    """All-reduce a batch behind one handshake and one coalescing group."""

    _submit_collective(
        _CollectiveRequest(
            tensors=tensors,
            operation="all_reduce",
            reduction=reduction,
            scope=scope,
        )
    )


@compiled_operator_entry
def reduce_(
    tensor: torch.Tensor,
    *,
    destination: int,
    reduction: Reduction = "sum",
    scope: ParallelAxis = "spatial",
) -> None:
    """Reduce one tensor to ``destination`` through the managed-step protocol."""

    _submit_collective(
        _CollectiveRequest(
            tensors=(tensor,),
            operation="reduce",
            reduction=reduction,
            destination=destination,
            scope=scope,
        )
    )


@compiled_operator_entry
def reduce_many_(
    tensors: Sequence[torch.Tensor],
    *,
    destination: int,
    reduction: Reduction = "sum",
    scope: ParallelAxis = "spatial",
) -> None:
    """Reduce a batch to ``destination`` behind one handshake.

    All tensors must share a device, and every rank must supply the same batch
    length, order and ABI; a mismatch raises at the handshake, not in NCCL.
    """

    _submit_collective(
        _CollectiveRequest(
            tensors=tensors,
            operation="reduce",
            reduction=reduction,
            destination=destination,
            scope=scope,
        )
    )

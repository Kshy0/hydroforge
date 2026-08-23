"""Explicit distributed operators for compiled substeps.

Every collective goes through one batched path: a batch costs a single
managed-step handshake and, when the communication backend supports it, one
coalescing group rather than one of each per tensor. ``reduce_`` and
``all_reduce_`` are its one-tensor spellings.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal

import torch
import torch.distributed as dist
from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.context import (
    active_operator_recorder, compiled_operator_entry,
)


Reduction = Literal["min", "max", "sum"]

_DTYPE_CODES = {
    dtype: index for index, dtype in enumerate((
        torch.uint8, torch.int8, torch.int32, torch.int64,
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
    ), start=1)
}
# MPS has an ABI code so a Metal recorder can reject collectives with its
# backend-specific compile error before any process group exists. XPU uses the
# formal Torch path and communicates through XCCL when PyTorch provides it.
_ABI_DEVICE_CODES = {"cpu": 1, "cuda": 2, "mps": 3, "xpu": 4}
_COLLECTIVE_DEVICES = frozenset({"cpu", "cuda", "xpu"})
_REDUCTIONS = {
    "min": (0, dist.ReduceOp.MIN),
    "max": (1, dist.ReduceOp.MAX),
    "sum": (2, dist.ReduceOp.SUM),
}

# 63-bit FNV-1a parameters: the folded batch signature travels in one int64
# slot of the managed-step vector, so it must stay non-negative.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_SIGNATURE_MASK = (1 << 63) - 1


class _CollectiveRequest(HydroForgeModel):
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor]
    operation: Literal["all_reduce", "reduce"]
    reduction: Reduction
    destination: int | None = None

    _abis: tuple[tuple[int, int, int], ...] = PrivateAttr()

    @model_validator(mode="after")
    def _validate_collective(self):
        batch = tuple(self.tensors)
        if self.operation == "all_reduce" and self.destination is not None:
            raise ValueError("all_reduce does not accept a destination")
        if self.operation == "reduce" and self.destination is None:
            raise ValueError("reduce requires a destination")
        if self.destination is not None and self.destination < 0:
            raise ValueError("reduce destination must be non-negative")
        devices = {tensor.device for tensor in batch}
        if len(devices) > 1:
            raise ValueError(
                f"{self.operation} tensors must share one device, got "
                f"{sorted(map(str, devices))}"
            )
        self._abis = tuple(
            _tensor_abi(tensor, operation=self.operation)
            for tensor in batch
        )
        object.__setattr__(self, "tensors", batch)
        return self

    @property
    def abis(self) -> tuple[tuple[int, int, int], ...]:
        return self._abis


def _require_distributed(operation: str) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{operation} requires an initialized torch.distributed process group"
        )


def _tensor_abi(
    tensor: torch.Tensor, *, operation: str,
) -> tuple[int, int, int]:
    """Validate and encode the process-group-independent tensor ABI."""

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{operation} tensor must be a torch.Tensor")
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
    if tensor.device.type not in _COLLECTIVE_DEVICES:
        raise ValueError(
            f"{operation} does not support device {tensor.device.type!r}"
        )
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

    digest = _FNV_OFFSET
    for value in (
        _REDUCTIONS[reduction][0],
        -1 if destination is None else destination,
        *(field for abi in abis for field in abi),
    ):
        digest = ((digest ^ (value & 0xFFFFFFFFFFFFFFFF)) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return len(abis), digest & _SIGNATURE_MASK, sum(abi[1] for abi in abis)


def _validate_collective_environment(
    tensor: torch.Tensor | None, *, operation: str,
    destination: int | None = None,
) -> None:
    """Validate batch-invariant process-group state exactly once."""

    _require_distributed(operation)
    if destination is not None and not 0 <= destination < dist.get_world_size():
        raise ValueError(f"{operation} destination is outside the process group")
    if tensor is None:
        return
    backend = str(dist.get_backend()).lower()
    required_device = {"nccl": "cuda", "xccl": "xpu"}
    for backend_name, device_type in required_device.items():
        if backend_name in backend and tensor.device.type != device_type:
            raise ValueError(
                f"{operation} with {backend_name.upper()} requires a "
                f"{device_type.upper()} tensor"
            )
    required_backend = {"cuda": "nccl", "xpu": "xccl"}.get(
        tensor.device.type,
    )
    if required_backend is not None and required_backend not in backend:
        raise ValueError(
            f"{operation} of a {tensor.device.type.upper()} tensor requires "
            f"the {required_backend.upper()} process-group backend, got "
            f"{dist.get_backend()!s}"
        )


def _event_kind(
    operation: str, reduction: Reduction, destination: int | None = None,
) -> int:
    reduction_code = _REDUCTIONS[reduction][0]
    if operation == "all_reduce":
        return 10 + reduction_code
    return 100 + destination * 3 + reduction_code


def _coalescing_group(device: torch.device):
    """Group the batch into one NCCL submission when the backend allows it."""

    manager = getattr(dist, "_coalescing_manager", None)
    if manager is None or device.type != "cuda":
        return nullcontext()
    if "nccl" not in str(dist.get_backend()).lower():
        return nullcontext()
    return manager(device=device, async_ops=False)


def _run_validated_batch(
    tensors: tuple[torch.Tensor, ...],
    abis: tuple[tuple[int, int, int], ...],
    *,
    operation: str,
    reduction: Reduction,
    destination: int | None,
) -> None:
    """Synchronize once and launch one already validated batch."""

    from hydroforge.execution.step import _managed_step_active

    if not _managed_step_active():
        raise RuntimeError(
            "HydroForge collectives may be called only inside a managed step "
            "or an operator recorder"
        )
    _code, op = _REDUCTIONS[reduction]

    from hydroforge.execution.step import synchronize_collective

    _validate_collective_environment(
        tensors[0] if tensors else None,
        operation=operation,
        destination=destination,
    )
    # The handshake runs even for an empty batch: a rank that contributes no
    # tensors must still be seen to disagree with one that does.
    synchronize_collective(
        _event_kind(operation, reduction, destination),
        _batch_signature(abis, reduction, destination),
    )
    if not tensors:
        return
    with _coalescing_group(tensors[0].device):
        for tensor in tensors:
            if destination is None:
                dist.all_reduce(tensor, op=op)
            else:
                dist.reduce(tensor, dst=destination, op=op)


def _submit_collective(request: _CollectiveRequest) -> None:
    recorder = active_operator_recorder()
    if recorder is not None:
        recorder.record_collective_batch(
            request.tensors,
            request.abis,
            request.reduction,
            operation=request.operation,
            destination=request.destination,
        )
        return
    from hydroforge.execution.step import _managed_step_active

    if not _managed_step_active():
        raise RuntimeError(
            "HydroForge collectives may be called only inside a managed step "
            "or an operator recorder"
        )
    _run_validated_batch(
        request.tensors,
        request.abis,
        operation=request.operation,
        reduction=request.reduction,
        destination=request.destination,
    )


def launch_recorded_collective_batch(
    tensors: tuple[torch.Tensor, ...],
    abis: tuple[tuple[int, int, int], ...],
    *,
    operation: str,
    reduction: Reduction,
    destination: int | None,
) -> None:
    """Replay one compiled batch without repeating its tensor ABI checks."""

    _run_validated_batch(
        tensors, abis, operation=operation, reduction=reduction,
        destination=destination,
    )


@compiled_operator_entry
def all_reduce_(tensor: torch.Tensor, *, reduction: Reduction) -> None:
    """Apply an in-place distributed reduction as an explicit IR operator.

    Unlike calling ``torch.distributed`` directly inside a lexical substep,
    this operation is recorded once and replayed on every physical iteration.
    """

    _submit_collective(_CollectiveRequest(
        tensors=(tensor,), operation="all_reduce", reduction=reduction,
    ))


@compiled_operator_entry
def all_reduce_many_(
    tensors: Sequence[torch.Tensor], *, reduction: Reduction = "sum",
) -> None:
    """All-reduce a batch behind one handshake and one coalescing group."""

    _submit_collective(_CollectiveRequest(
        tensors=tensors, operation="all_reduce", reduction=reduction,
    ))


@compiled_operator_entry
def reduce_(
    tensor: torch.Tensor, *, destination: int, reduction: Reduction = "sum",
) -> None:
    """Reduce one tensor to ``destination`` through the managed-step protocol."""

    _submit_collective(_CollectiveRequest(
        tensors=(tensor,), operation="reduce", reduction=reduction,
        destination=destination,
    ))


@compiled_operator_entry
def reduce_many_(
    tensors: Sequence[torch.Tensor], *, destination: int,
    reduction: Reduction = "sum",
) -> None:
    """Reduce a batch to ``destination`` behind one handshake.

    All tensors must share a device, and every rank must supply the same batch
    length, order and ABI; a mismatch raises at the handshake, not in NCCL.
    """

    _submit_collective(_CollectiveRequest(
        tensors=tensors, operation="reduce", reduction=reduction,
        destination=destination,
    ))

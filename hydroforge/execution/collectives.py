"""Explicit distributed operators for compiled substeps.

Every collective goes through one batched path: a batch costs a single
managed-step handshake and one NCCL coalescing group rather than one of each
per tensor.  ``reduce_`` and ``all_reduce_`` are its one-tensor spellings.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal

import torch
import torch.distributed as dist

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
# backend-specific compile error before any process group exists. Eager
# communication remains limited to CPU/CUDA below.
_ABI_DEVICE_CODES = {"cpu": 1, "cuda": 2, "mps": 3}
_COLLECTIVE_DEVICES = frozenset({"cpu", "cuda"})
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


def _reduction_spec(reduction: Reduction):
    try:
        return _REDUCTIONS[reduction]
    except KeyError as error:
        raise ValueError("reduction must be 'min', 'max', or 'sum'") from error


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
        raise TypeError(f"{operation} tensor must be a torch.Tensor")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"{operation} tensor must be contiguous and strided")
    if tensor.numel() < 1:
        raise ValueError(f"{operation} tensor must be non-empty")
    try:
        dtype_code = _DTYPE_CODES[tensor.dtype]
    except KeyError as error:
        raise TypeError(
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

    digest = _FNV_OFFSET
    for value in (
        _reduction_spec(reduction)[0],
        -1 if destination is None else destination,
        *(field for abi in abis for field in abi),
    ):
        digest = ((digest ^ (value & 0xFFFFFFFFFFFFFFFF)) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return len(abis), digest & _SIGNATURE_MASK, sum(abi[1] for abi in abis)


def _validate_collective_runtime(
    tensor: torch.Tensor | None, *, operation: str,
    destination: int | None = None,
) -> None:
    """Validate batch-invariant process-group state exactly once."""

    _require_distributed(operation)
    if destination is not None and destination >= dist.get_world_size():
        raise ValueError(f"{operation} destination is outside the process group")
    if tensor is None:
        return
    if tensor.device.type not in _COLLECTIVE_DEVICES:
        raise ValueError(
            f"{operation} does not support device {tensor.device.type!r}"
        )
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend and tensor.device.type != "cuda":
        raise ValueError(f"{operation} with NCCL requires a CUDA tensor")


def _event_kind(
    operation: str, reduction: Reduction, destination: int | None = None,
) -> int:
    reduction_code = _reduction_spec(reduction)[0]
    if operation == "all_reduce":
        return 10 + reduction_code
    if operation == "reduce" and destination is not None:
        return 100 + destination * 3 + reduction_code
    raise ValueError(f"invalid collective operation {operation!r}")


def _coalescing_group(device: torch.device):
    """Group the batch into one NCCL submission when the backend allows it."""

    manager = getattr(dist, "_coalescing_manager", None)
    if manager is None or device.type != "cuda":
        return nullcontext()
    if "nccl" not in str(dist.get_backend()).lower():
        return nullcontext()
    return manager(device=device, async_ops=False)


def _normalize_batch(
    tensors: Sequence[torch.Tensor], *, operation: str,
) -> tuple[torch.Tensor, ...]:
    if isinstance(tensors, torch.Tensor):
        raise TypeError(
            f"{operation} takes a sequence of tensors; pass [tensor] for one"
        )
    batch = tuple(tensors)
    for tensor in batch:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{operation} tensors must all be torch.Tensor")
    devices = {tensor.device for tensor in batch}
    if len(devices) > 1:
        raise ValueError(
            f"{operation} tensors must share one device, got {sorted(map(str, devices))}"
        )
    return batch


def _run_batch(
    tensors: Sequence[torch.Tensor],
    *,
    operation: str,
    reduction: Reduction,
    destination: int | None,
) -> None:
    """Record, or synchronize once and launch the whole batch."""

    _code, op = _reduction_spec(reduction)
    if destination is not None and (
        type(destination) is not int or destination < 0
    ):
        raise ValueError(
            f"{operation} destination must be a non-negative exact int"
        )
    batch = _normalize_batch(tensors, operation=operation)
    abis = [_tensor_abi(tensor, operation=operation) for tensor in batch]

    recorder = active_operator_recorder()
    if recorder is not None:
        # Recording does not require an initialized process group.  Each
        # tensor stays its own IR operator so replay ordering is unchanged.
        if (
            destination is not None and dist.is_available()
            and dist.is_initialized()
            and destination >= dist.get_world_size()
        ):
            raise ValueError(
                f"{operation} destination is outside the process group"
            )
        for tensor in batch:
            recorder.record_collective(
                tensor, reduction, operation=operation,
                destination=destination,
            )
        return

    from hydroforge.execution.step import synchronize_collective

    _validate_collective_runtime(
        batch[0] if batch else None,
        operation=operation,
        destination=destination,
    )
    # The handshake runs even for an empty batch: a rank that contributes no
    # tensors must still be seen to disagree with one that does.
    synchronize_collective(
        _event_kind(operation, reduction, destination),
        _batch_signature(abis, reduction, destination),
    )
    if not batch:
        return
    with _coalescing_group(batch[0].device):
        for tensor in batch:
            if destination is None:
                dist.all_reduce(tensor, op=op)
            else:
                dist.reduce(tensor, dst=destination, op=op)


def launch_recorded_collective(
    tensor: torch.Tensor,
    *,
    operation: str,
    reduction: Reduction,
    destination: int | None,
) -> None:
    """Replay one recorded collective through the eager batch-of-one path,
    so its event kind and signature match the eager spelling."""

    _run_batch(
        (tensor,), operation=operation, reduction=reduction,
        destination=destination,
    )


@compiled_operator_entry
def all_reduce_(tensor: torch.Tensor, *, reduction: Reduction) -> None:
    """Apply an in-place distributed reduction as an explicit IR operator.

    Unlike calling ``torch.distributed`` directly inside a lexical substep,
    this operation is recorded once and replayed on every physical iteration.
    """

    _run_batch(
        (tensor,), operation="all_reduce", reduction=reduction,
        destination=None,
    )


@compiled_operator_entry
def all_reduce_many_(
    tensors: Sequence[torch.Tensor], *, reduction: Reduction = "sum",
) -> None:
    """All-reduce a batch behind one handshake and one coalescing group."""

    _run_batch(
        tensors, operation="all_reduce", reduction=reduction, destination=None,
    )


@compiled_operator_entry
def reduce_(
    tensor: torch.Tensor, *, destination: int, reduction: Reduction = "sum",
) -> None:
    """Reduce one tensor to ``destination`` through the managed-step protocol."""

    _run_batch(
        (tensor,), operation="reduce", reduction=reduction,
        destination=destination,
    )


@compiled_operator_entry
def reduce_many_(
    tensors: Sequence[torch.Tensor], *, destination: int,
    reduction: Reduction = "sum",
) -> None:
    """Reduce a batch to ``destination`` behind one handshake.

    All tensors must share a device, and every rank must supply the same batch
    length, order and ABI; a mismatch raises at the handshake, not in NCCL.
    """

    _run_batch(
        tensors, operation="reduce", reduction=reduction,
        destination=destination,
    )

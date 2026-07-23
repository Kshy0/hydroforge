"""Explicit distributed operators for compiled substeps."""

from __future__ import annotations

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
_DEVICE_CODES = {"cpu": 1, "cuda": 2}


def _reduce_op(reduction: Reduction):
    try:
        return {
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
            "sum": dist.ReduceOp.SUM,
        }[reduction]
    except KeyError as error:
        raise ValueError("reduction must be 'min', 'max', or 'sum'") from error


def _require_distributed(operation: str) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{operation} requires an initialized torch.distributed process group"
        )


def _collective_signature(
    tensor: torch.Tensor, *, operation: str,
) -> tuple[int, int, int]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{operation} tensor must be a torch.Tensor")
    _require_distributed(operation)
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
        device_code = _DEVICE_CODES[tensor.device.type]
    except KeyError as error:
        raise ValueError(
            f"{operation} does not support device {tensor.device.type!r}"
        ) from error
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend and tensor.device.type != "cuda":
        raise ValueError(f"{operation} with NCCL requires a CUDA tensor")
    return dtype_code, tensor.numel(), device_code


def _launch_all_reduce(tensor: torch.Tensor, reduction: Reduction) -> None:
    _require_distributed("all_reduce_")
    dist.all_reduce(tensor, op=_reduce_op(reduction))


def _launch_reduce(
    tensor: torch.Tensor, reduction: Reduction, *, destination: int,
) -> None:
    _require_distributed("reduce_")
    dist.reduce(tensor, dst=destination, op=_reduce_op(reduction))


def _event_kind(
    operation: str, reduction: Reduction, destination: int | None = None,
) -> int:
    reduction_code = {"min": 0, "max": 1, "sum": 2}[reduction]
    if operation == "all_reduce":
        return 10 + reduction_code
    if operation == "reduce" and destination is not None:
        return 100 + destination * 3 + reduction_code
    raise ValueError(f"invalid collective operation {operation!r}")


@compiled_operator_entry
def all_reduce_(tensor: torch.Tensor, *, reduction: Reduction) -> None:
    """Apply an in-place distributed reduction as an explicit IR operator.

    Unlike calling ``torch.distributed`` directly inside a lexical substep,
    this operation is recorded once and replayed on every physical iteration.
    """

    if reduction not in {"min", "max", "sum"}:
        raise ValueError("reduction must be 'min', 'max', or 'sum'")
    recorder = active_operator_recorder()
    if recorder is not None:
        recorder.record_collective(tensor, reduction)
        return
    from hydroforge.execution.step import synchronize_collective

    signature = _collective_signature(tensor, operation="all_reduce_")
    synchronize_collective(
        _event_kind("all_reduce", reduction), signature,
    )
    _launch_all_reduce(tensor, reduction)


@compiled_operator_entry
def reduce_(
    tensor: torch.Tensor, *, destination: int, reduction: Reduction = "sum",
) -> None:
    """Reduce one tensor to ``destination`` through the managed-step protocol."""

    if type(destination) is not int or destination < 0:
        raise ValueError("reduce_ destination must be a non-negative exact int")
    if reduction not in {"min", "max", "sum"}:
        raise ValueError("reduction must be 'min', 'max', or 'sum'")
    if dist.is_available() and dist.is_initialized() and (
        destination >= dist.get_world_size()
    ):
        raise ValueError("reduce_ destination is outside the process group")
    recorder = active_operator_recorder()
    if recorder is not None:
        recorder.record_collective(
            tensor, reduction, operation="reduce", destination=destination,
        )
        return
    from hydroforge.execution.step import synchronize_collective

    signature = _collective_signature(tensor, operation="reduce_")
    synchronize_collective(
        _event_kind("reduce", reduction, destination), signature,
    )
    _launch_reduce(tensor, reduction, destination=destination)

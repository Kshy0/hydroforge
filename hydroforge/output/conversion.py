"""Checked output dtype conversion shared by memory and file results."""

import numpy as np
import torch


def _narrowing_limit(source: torch.dtype, target: torch.dtype) -> float | None:
    """Return the finite magnitude bound of a lossy float narrowing, if any."""

    if not (source.is_floating_point and target.is_floating_point):
        raise TypeError(
            f"unsupported statistics output conversion from {source} to {target}"
        )
    limit = torch.finfo(target).max
    return limit if limit < torch.finfo(source).max else None


def _checked_narrowing(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    name: str,
    flags: list[tuple[torch.Tensor, str, str]] | None = None,
) -> torch.Tensor:
    """Convert on the tensor's device, rejecting finite values out of range.

    Tiny values may round to subnormals or zero.  With ``flags`` the
    device-side overflow flag is queued for a deferred host check; otherwise
    it is checked before returning.
    """

    source = tensor.detach()
    if source.dtype == target_dtype:
        return source
    limit = _narrowing_limit(source.dtype, target_dtype)
    converted = source.to(dtype=target_dtype)
    if limit is not None:
        if source.numel():
            magnitude = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
            outside = magnitude.abs_().amax() > limit
        else:
            outside = torch.zeros((), dtype=torch.bool, device=source.device)
        entry = (outside, name, str(target_dtype).removeprefix("torch."))
        if flags is None:
            _raise_narrowing_failures((entry,))
        else:
            flags.append(entry)
    return converted


def _raise_narrowing_failures(
    flags: tuple[tuple[torch.Tensor, str, str], ...] | list,
    observed: torch.Tensor | None = None,
) -> None:
    """Raise the first failed narrowing check from one host flag vector."""

    if not flags:
        return
    if observed is None:
        device = flags[0][0].device
        observed = torch.stack(
            [failure.to(device=device) for failure, _name, _label in flags]
        ).cpu()
    for outside, (_failure, name, label) in zip(observed.tolist(), flags, strict=True):
        if outside:
            raise OverflowError(
                f"statistics output {name!r} contains values outside {label} range"
            )


def _checked_output_array(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    name: str,
) -> np.ndarray:
    """Materialize one output without allowing a finite value to overflow."""

    converted = _checked_narrowing(tensor, target_dtype, name=name)
    return converted.cpu().numpy()


def _checked_output_tensor_copy(
    tensor: torch.Tensor,
    *,
    target_device: torch.device,
    target_dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Copy an in-memory result after validating a narrowing conversion."""

    converted = _checked_narrowing(tensor, target_dtype, name=name)
    return converted.to(device=target_device, copy=converted.dtype == tensor.dtype)

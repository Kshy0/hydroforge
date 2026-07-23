"""Address-stable streamed input bindings for model execution."""

from __future__ import annotations

import torch


def copy_input(
    target: torch.Tensor | None,
    source: torch.Tensor | None,
    *,
    when_none: str = "zero",
    name: str = "input",
    trial_broadcast: bool = False,
) -> torch.Tensor | None:
    """Explicit outer-step input copy with strict trial broadcasting.

    This operation is intentionally ordinary Torch outside a compiled substep.
    Its location in ``step_advance`` is therefore the exact, readable point at
    which caller-owned data enters address-stable model storage.
    """

    if target is None:
        if source is not None:
            raise ValueError(f"{name} was provided but its module is closed")
        return None
    if source is None:
        if when_none == "required":
            raise ValueError(f"step input {name!r} is required")
        if when_none == "zero":
            target.zero_()
        elif when_none != "keep":
            raise ValueError(f"unknown None policy {when_none!r}")
        return target
    if source.device != target.device:
        raise ValueError(f"{name} is on {source.device}, expected {target.device}")
    if source.dtype != target.dtype:
        raise ValueError(
            f"{name} has dtype {source.dtype}, expected {target.dtype}"
        )
    valid_shape = source.shape == target.shape
    trial_shape = target.shape[1:] if target.ndim > 0 else None
    broadcast_trial_axis = (
        trial_broadcast
        and
        trial_shape is not None
        and source.shape == trial_shape
        and target.shape != source.shape
    )
    if not (valid_shape or broadcast_trial_axis):
        raise ValueError(
            f"{name} shape {tuple(source.shape)} cannot copy into "
            f"{tuple(target.shape)}"
        )
    target.copy_(source.unsqueeze(0) if broadcast_trial_axis else source)
    return target

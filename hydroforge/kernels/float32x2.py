"""Explicit two-component FP32 buffers for Metal high-precision kernels.

The trailing pair is physical storage, not a model axis. Ordinary Torch
arithmetic must not operate on the encoded values. Unlike binary64 this
representation retains the FP32 exponent range.
"""

from functools import cache
from pathlib import Path

import torch


@cache
def source() -> str:
    """Return MSL primitives with explicit FMA only for product residuals."""
    return Path(__file__).with_suffix(".metal").read_text()


def encode(values: torch.Tensor, device: torch.device | str = "mps") -> torch.Tensor:
    """Encode CPU float64 values as independent, interleaved hi/lo floats.

    Refuse overflow and complete underflow rather than silently changing a
    finite input's range. NaNs and infinities use a zero low component.
    """
    if values.device.type != "cpu" or values.dtype != torch.float64:
        raise TypeError("float32x2 encoding requires a CPU float64 tensor")
    values = values.detach()
    hi = values.float()
    finite = torch.isfinite(values)
    if bool((finite & ~torch.isfinite(hi)).any()):
        raise OverflowError("float32x2 input exceeds the FP32 exponent range")
    if bool((finite & (values != 0) & (hi == 0)).any()):
        raise OverflowError("float32x2 input underflows the FP32 exponent range")
    lo = torch.where(finite, (values - hi.double()).float(), 0.0)
    return torch.stack((hi, lo), dim=-1).to(device, copy=True)


def decode(pairs: torch.Tensor) -> torch.Tensor:
    """Return independent CPU float64 values from normalized hi/lo pairs."""
    if pairs.dtype != torch.float32 or pairs.ndim == 0 or pairs.shape[-1] != 2:
        raise TypeError("float32x2 decoding requires float32 pairs in the last axis")
    values = pairs.detach().to(device="cpu", dtype=torch.float64, copy=True)
    return values[..., 0] + values[..., 1]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Explicit two-component FP32 buffers for Metal high-precision kernels.

The trailing pair is physical storage, not a model axis. Ordinary Torch
arithmetic must not operate on the encoded values. Unlike binary64 this
representation retains the FP32 exponent range. The low limb loses precision
near the FP32 subnormal range, and Metal may flush subnormals to zero; roughly
48 significand bits are therefore not guaranteed for very small values.
Arithmetic uses round-to-nearest with fast math and contraction disabled.
MSL hp-to-integer casts explicitly truncate toward zero, saturate outside the
integer range, and map NaN to zero; this is the encoded Metal cast policy.
All integers through absolute 2**48 are represented exactly, but arbitrary
64-bit integers may lose low bits on promotion.
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
    # Rounding the residual can reach half an ulp of an odd ``hi``; renormalize
    # with an exact FP32 sum so every pair is canonical (``hi == hi + lo``),
    # which the limb-wise Metal comparisons require.
    # A zero ``hi`` keeps its sign (its ``lo`` is zero).
    total = hi + lo
    lo = torch.where(finite, lo - (total - hi), 0.0)
    hi = torch.where(finite & (hi != 0), total, hi)
    return torch.stack((hi, lo), dim=-1).to(device, copy=True)


def decode(pairs: torch.Tensor) -> torch.Tensor:
    """Return independent CPU float64 values from normalized hi/lo pairs."""
    if pairs.dtype != torch.float32 or pairs.ndim == 0 or pairs.shape[-1] != 2:
        raise TypeError("float32x2 decoding requires float32 pairs in the last axis")
    values = pairs.detach().to(device="cpu", dtype=torch.float64, copy=True)
    hi, lo = values[..., 0], values[..., 1]
    # IEEE addition of -0 and +0 would erase a stored negative zero.
    return torch.where((hi == 0) & (lo == 0), hi, hi + lo)

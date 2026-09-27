"""Triton math for physics kernels, exact in the kernel's precision.

Two Triton lowerings make kernels deviate from the CPU reference models and
from the CUDA implementations of the same formulas:

- A Python float that is not folded straight into arithmetic with a tensor
  becomes an fp32 scalar: comparison operands (``x > 0.1``),
  ``tl.minimum``/``tl.maximum``/``tl.clamp`` operands, ``tl.where`` with two
  scalar branches, ``tl.load(other=...)`` and locals assigned from constants.
  FP64 kernels then use fp32-rounded constants. :func:`constant` builds the
  value in the kernel's precision; :func:`at_least`, :func:`at_most` and
  :func:`clamp` apply it to bounds.
- FP32 ``/`` lowers to an approximate division (up to 2 ulp, biased) and
  ``tl.sqrt``/``tl.exp``/``tl.log`` to approximate instructions, while the
  CPU models and CUDA use IEEE division and square root and libm-accurate
  functions. :func:`divide`, :func:`sqrt`, :func:`exp`, :func:`log`,
  :func:`pow` and :func:`cbrt` use the IEEE and libdevice forms (the
  functions CUDA calls), unless :func:`hydroforge.kernels.math_mode.fast_math`
  selects fast math.

:func:`to_compute` and :func:`to_index` mark the precision and index
boundaries of mixed-precision kernels: a storage-precision value narrowed to
the compute precision, and an integral non-negative count quantized to an
index.

Operands may be tensors or Python numbers, but at least one operand of each
call must be a tensor, whose dtype types the numbers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language import core
from triton.language.extra import libdevice

from hydroforge.kernels.math_mode import fast_math

FAST_MATH = tl.constexpr(fast_math())
_HIP = tl.constexpr(torch.version.hip is not None)


@triton.jit
def constant(value, like):
    """``value`` in the dtype of tensor ``like``; a tensor passes unchanged."""
    if isinstance(value, tl.tensor):
        return value
    else:
        return tl.full((), value, like.dtype)


@triton.jit
def divide(numerator, denominator):
    """IEEE ``numerator / denominator`` (approximate under fast math)."""
    numerator = constant(numerator, denominator)
    denominator = constant(denominator, numerator)
    if FAST_MATH:
        return numerator / denominator
    elif numerator.dtype == tl.float32:
        if denominator.dtype == tl.float32:
            return tl.div_rn(numerator, denominator)
        else:
            return numerator / denominator
    else:
        # FP64 division is already IEEE.
        return numerator / denominator


@triton.jit
def sqrt(value):
    """IEEE square root (approximate under fast math)."""
    if FAST_MATH:
        return tl.sqrt(value)
    elif value.dtype == tl.float32:
        if _HIP:
            # ROCm Triton's sqrt_rn returns NaN for subnormal inputs; OCML's
            # sqrt is IEEE.
            return libdevice.sqrt(value)
        else:
            return tl.sqrt_rn(value)
    else:
        return tl.sqrt(value)


@triton.jit
def exp(value):
    """libdevice exponential (``tl.exp`` under fast math)."""
    if FAST_MATH:
        return tl.exp(value)
    else:
        return libdevice.exp(value)


@triton.jit
def log(value):
    """libdevice natural logarithm (``tl.log`` under fast math)."""
    if FAST_MATH:
        return tl.log(value)
    else:
        return libdevice.log(value)


@triton.jit
def pow(base, exponent):
    """libdevice ``base ** exponent`` (``exp2(exponent * log2(base))`` under
    fast math); mixed FP32/FP64 operands are promoted to FP64 like ``/``."""
    base = constant(base, exponent)
    exponent = constant(exponent, base)
    if base.dtype == tl.float64:
        exponent = exponent.to(tl.float64)
    elif exponent.dtype == tl.float64:
        base = base.to(tl.float64)
    if FAST_MATH:
        return tl.exp2(exponent * tl.log2(base))
    else:
        return libdevice.pow(base, exponent)


@core.extern
def _ocml_cbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0],
        {
            (core.dtype("fp32"),): ("__ocml_cbrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__ocml_cbrt_f64", core.dtype("fp64")),
        },
        is_pure=True, _semantic=_semantic,
    )


@triton.jit
def cbrt(value):
    """Cube root: libdevice on NVIDIA, OCML on AMD, as CUDA/HIP ``cbrt``.

    ROCm Triton's libdevice has no ``cbrt``; ``pow(x, 1/3)`` would round
    worse and return NaN for negative ``x``.
    """
    if _HIP:
        return _ocml_cbrt(value)
    else:
        return libdevice.cbrt(value)


@triton.jit
def to_compute(hp_value, like):
    """Storage-precision ``hp_value`` narrowed to ``like``'s compute dtype."""
    return hp_value.to(like.dtype)


@triton.jit
def to_index(value):
    """An integral, non-negative count quantized to an ``int32`` index."""
    return value.to(tl.int32)


@triton.jit
def at_least(value, bound):
    """``max(value, bound)`` with the bound in ``value``'s precision."""
    return tl.maximum(value, constant(bound, value))


@triton.jit
def at_most(value, bound):
    """``min(value, bound)`` with the bound in ``value``'s precision."""
    return tl.minimum(value, constant(bound, value))


@triton.jit
def clamp(value, lower, upper):
    """``min(max(value, lower), upper)`` with bounds in ``value``'s precision."""
    return at_most(at_least(value, lower), upper)


__all__ = [
    "FAST_MATH",
    "at_least",
    "at_most",
    "cbrt",
    "clamp",
    "constant",
    "divide",
    "exp",
    "log",
    "pow",
    "sqrt",
    "to_compute",
    "to_index",
]

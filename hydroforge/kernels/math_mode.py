"""Process-wide choice of physics math mode, read once on first use.

Backends translate this choice into their own launch or compiler options.
Importing the policy does not initialize a backend or modify its compiler.
"""

from __future__ import annotations

import os
from functools import cache

FAST_MATH_ENV = "HYDROFORGE_FAST_MATH"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


@cache
def fast_math() -> bool:
    """Whether physics kernels use each backend's fast math."""

    value = os.environ.get(FAST_MATH_ENV, "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{FAST_MATH_ENV} must be a boolean flag, got {value!r}")


__all__ = ["FAST_MATH_ENV", "fast_math"]

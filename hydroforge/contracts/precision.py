# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The two precision tiers of a model: compute and storage.

A model computes in its ``precision`` tier, the element type of its
``float`` tensors, and stores its ``hpfloat`` tensors (state accumulating
small increments, such as water storage) in its storage tier:

``float64``, ``float32``
    IEEE elements.
``tf32``
    float32 elements; matrix products and convolutions of torch operators
    inside model calls run on TF32 tensor cores (a compute tier only).
``bfloat16``, ``float16``
    16-bit elements, read and written by kernels that declare them
    (:class:`~hydroforge.contracts.runtime.BackendRequirement.precision`).

Host scalars, step fields and ``"precision"`` workspaces keep the *scalar
precision*: float64 for a float64 compute tier, float32 otherwise, so a
16-bit model still counts seconds and widths exactly enough.

``storage_precision`` names the storage tier; without it ``mixed_precision``
promotes it one level above the compute tier (16-bit to float32, float32
and tf32 to float64) or keeps it equal. The storage tier never holds less
than the compute tier.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

PrecisionTier = Literal["float64", "float32", "tf32", "bfloat16", "float16"]
StorageTier = Literal["float64", "float32", "bfloat16", "float16"]
ScalarPrecision = Literal["float32", "float64"]

TIERS: tuple[str, ...] = ("float64", "float32", "tf32", "bfloat16", "float16")
STORAGE_TIERS: tuple[str, ...] = ("float64", "float32", "bfloat16", "float16")
HALF_TIERS = frozenset(("bfloat16", "float16"))

ELEMENTS: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "tf32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
_PROMOTED = {
    "float64": "float64",
    "float32": "float64",
    "tf32": "float64",
    "bfloat16": "float32",
    "float16": "float32",
}
# Bits of significand: the order the storage tier may not fall below.
_DIGITS = {"float64": 53, "float32": 24, "tf32": 24, "bfloat16": 8, "float16": 11}


def tier_of(dtype: torch.dtype) -> str:
    """The storage tier whose elements are ``dtype``."""

    for name in STORAGE_TIERS:
        if ELEMENTS[name] == dtype:
            return name
    raise TypeError(f"{dtype} is not a floating precision tier")


@dataclass(frozen=True, slots=True)
class Precisions:
    """The resolved tiers of one model."""

    compute: str
    storage: str

    @property
    def field_dtype(self) -> torch.dtype:
        """Elements of ``float`` tensors."""

        return ELEMENTS[self.compute]

    @property
    def storage_dtype(self) -> torch.dtype:
        """Elements of ``hpfloat`` tensors."""

        return ELEMENTS[self.storage]

    @property
    def scalar(self) -> ScalarPrecision:
        """Host scalars, step fields and ``"precision"`` workspaces."""

        return "float64" if self.compute == "float64" else "float32"

    @property
    def scalar_dtype(self) -> torch.dtype:
        return ELEMENTS[self.scalar]

    @property
    def mixed(self) -> bool:
        """Whether ``hpfloat`` tensors hold more than ``float`` ones."""

        return self.storage_dtype != self.field_dtype

    @property
    def tf32(self) -> bool:
        return self.compute == "tf32"

    @property
    def half(self) -> bool:
        """Whether some tensors hold 16-bit elements."""

        return self.compute in HALF_TIERS or self.storage in HALF_TIERS

    def describe(self) -> str:
        return (
            self.compute
            if self.storage == tier_of(self.field_dtype)
            else f"{self.compute}/{self.storage}"
        )


def resolve_precisions(
    precision: str,
    storage_precision: str | None,
    mixed_precision: bool,
) -> Precisions:
    """The tiers of a declaration; ``mixed_precision`` is already resolved
    to a bool when ``storage_precision`` is omitted."""

    if precision not in TIERS:
        raise ValueError(f"precision must be one of {TIERS}, got {precision!r}")
    if storage_precision is None:
        storage = (
            _PROMOTED[precision] if mixed_precision else tier_of(ELEMENTS[precision])
        )
        return Precisions(precision, storage)
    if storage_precision not in STORAGE_TIERS:
        raise ValueError(
            f"storage_precision must be one of {STORAGE_TIERS}, got "
            f"{storage_precision!r} (tf32 is a compute tier)"
        )
    resolved = Precisions(precision, storage_precision)
    if _DIGITS[storage_precision] < _DIGITS[precision] or (
        resolved.storage_dtype != resolved.field_dtype
        and {precision, storage_precision} == HALF_TIERS
    ):
        raise ValueError(
            f"storage_precision {storage_precision!r} holds less than the "
            f"{precision!r} compute tier"
        )
    return resolved


def precisions_of(plan: object) -> Precisions:
    """The tiers of a model plan; a plan built without them (as some tests
    do) has a scalar ``dtype`` and a ``mixed_precision`` flag."""

    found = getattr(plan, "precisions", None)
    if found is not None:
        return found
    dtype = getattr(plan, "dtype", torch.float32)
    return resolve_precisions(
        tier_of(dtype), None, bool(getattr(plan, "mixed_precision", False))
    )


@contextmanager
def matmul_scope(tf32: bool) -> Iterator[None]:
    """Run torch float32 matrix products and convolutions on TF32 tensor
    cores (``tf32``) or in IEEE float32, restoring the process setting.

    Model calls enter it, so a float32 model computes in float32 whatever
    the process allowed, and a tf32 model uses TF32 only inside them. The
    ``allow_tf32`` switches are used: torch (2.9+) rejects reading them
    while its newer ``fp32_precision`` settings disagree, and model code
    may read either.
    """

    backends = torch.backends
    saved = (backends.cuda.matmul.allow_tf32, backends.cudnn.allow_tf32)
    try:
        backends.cuda.matmul.allow_tf32 = tf32
        backends.cudnn.allow_tf32 = tf32
        yield
    finally:
        backends.cuda.matmul.allow_tf32, backends.cudnn.allow_tf32 = saved


__all__ = [
    "ELEMENTS",
    "HALF_TIERS",
    "STORAGE_TIERS",
    "TIERS",
    "matmul_scope",
    "precisions_of",
    "PrecisionTier",
    "Precisions",
    "ScalarPrecision",
    "StorageTier",
    "resolve_precisions",
    "tier_of",
]

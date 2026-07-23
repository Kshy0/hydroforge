"""Declarative, initialization-cached model values exposed to kernel binding."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class KernelField(cached_property):
    """A model value evaluated once when its first kernel plan is compiled."""

    __hydroforge_kernel_field__ = True


def kernel_field(function: Callable[[Any], T]) -> KernelField:
    """Expose one exact-name, cached model value to automatic kernel binding."""

    return KernelField(function)


__all__ = ["KernelField", "kernel_field"]

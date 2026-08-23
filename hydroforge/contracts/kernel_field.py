"""Declarative, initialization-cached model values exposed to kernel binding."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, TypeVar

from hydroforge.contracts.validation import HydroForgeModel

T = TypeVar("T")


class _KernelField(cached_property):
    """A model value evaluated once when its first kernel plan is compiled."""

    __hydroforge_kernel_field__ = True


class _KernelFieldDeclaration(HydroForgeModel):
    function: Callable[[Any], Any]


def kernel_field(function: Callable[[Any], T]) -> _KernelField:
    """Expose one exact-name, cached model value to automatic kernel binding."""

    declaration = _KernelFieldDeclaration(function=function)
    return _KernelField(declaration.function)


__all__ = ["kernel_field"]

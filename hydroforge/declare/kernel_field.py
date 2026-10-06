# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Declarative, initialization-cached model values exposed to kernel binding."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import cached_property
from typing import Any, TypeVar

T = TypeVar("T")


class _KernelField(cached_property):
    """A model value evaluated once when its first kernel plan is compiled."""

    __hydroforge_kernel_field__ = True


def kernel_field(function: Callable[[Any], T]) -> _KernelField:
    """Expose one exact-name, cached model value to automatic kernel binding."""

    try:
        inspect.signature(function).bind(None)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "kernel_field requires a callable taking only its model or module"
        ) from error
    return _KernelField(function)


__all__ = ["kernel_field"]

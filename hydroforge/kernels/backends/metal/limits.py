# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Metal launch limits imposed by 32-bit MSL grid coordinates."""

from __future__ import annotations

from typing import Any


METAL_GRID_MAX_EXTENT = 2 ** 32 - 1


def validate_metal_launch_extent(name: str, extent: Any) -> int:
    """Reject a grid width that cannot be represented without MSL wraparound."""

    if type(extent) is not int:
        raise TypeError(
            f"{name} Metal launch extent must be an exact host int, got "
            f"{type(extent).__name__}"
        )
    if extent < 0:
        raise ValueError(f"{name} Metal launch extent must be non-negative")
    if extent > METAL_GRID_MAX_EXTENT:
        raise OverflowError(
            f"{name} Metal launch extent exceeds the uint32 grid range"
        )
    return extent


__all__ = ["METAL_GRID_MAX_EXTENT", "validate_metal_launch_extent"]

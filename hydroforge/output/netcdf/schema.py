"""Immutable NetCDF variable schema compiled before file creation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class NetCDFSchema:
    actual_shape: tuple[int, ...]
    tensor_shape: tuple[int, ...]
    coordinate_name: str | None
    dtype: str
    order: int
    metadata: Mapping[str, Any]

    @classmethod
    def compile(cls, metadata: Mapping[str, Any]) -> "NetCDFSchema":
        try:
            dtype = np.dtype(metadata.get("dtype", "f8")).str.lstrip("<>|")
        except TypeError as error:
            raise TypeError(
                f"invalid NetCDF variable dtype {metadata.get('dtype')!r}"
            ) from error
        return cls(
            actual_shape=tuple(metadata.get("actual_shape", ())),
            tensor_shape=tuple(metadata.get("tensor_shape", ())),
            coordinate_name=metadata.get("nc_coord_name"),
            dtype=dtype,
            order=int(metadata.get("k", 1)),
            metadata=MappingProxyType(dict(metadata)),
        )

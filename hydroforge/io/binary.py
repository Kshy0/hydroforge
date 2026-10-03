# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Fortran-ordered binary and spatial-map file reads."""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import AfterValidator, Field, PositiveInt, validate_call

from hydroforge.core.validation import HydroForgeModel
from hydroforge.io.files import FileIdentity


def _plain_dtype(value: Any) -> np.dtype:
    dtype = np.dtype(value)
    if (
        dtype.itemsize <= 0
        or dtype.hasobject
        or dtype.subdtype is not None
        or dtype.fields is not None
    ):
        raise ValueError("binary dtype must be a plain fixed-width scalar dtype")
    return dtype


_BinaryDType = Annotated[Any, AfterValidator(_plain_dtype)]
_FilePath = Annotated[Path, Field(strict=False)]


def _read_fortran(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    path = path.absolute()
    identity = FileIdentity.capture(path)
    expected_size = prod(shape) * dtype.itemsize
    if identity.size != expected_size:
        raise ValueError(
            f"binary file {path} has {identity.size} bytes; expected "
            f"{expected_size} bytes for shape {shape} and dtype {dtype.str!r}"
        )
    try:
        array = np.fromfile(path, dtype=dtype, count=prod(shape))
    finally:
        identity.verify(path, label="binary file")
    return array.reshape(shape, order="F")


@validate_call(config=HydroForgeModel.model_config)
def binread(
    filename: _FilePath,
    shape: Annotated[tuple[PositiveInt, ...], Field(min_length=1)],
    dtype_str: _BinaryDType,
) -> np.ndarray:
    """Read a Fortran-ordered binary file and reshape to *shape*."""

    return _read_fortran(filename, shape, dtype_str)


@validate_call(config=HydroForgeModel.model_config)
def read_map(
    filename: _FilePath,
    map_shape: Annotated[tuple[PositiveInt, ...], Field(min_length=2, max_length=3)],
    precision: _BinaryDType,
) -> np.ndarray:
    """Read a spatial map binary file (Fortran-ordered)."""

    return _read_fortran(filename, map_shape, precision)

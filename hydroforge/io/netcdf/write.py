# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Atomic creation and publication of complete NetCDF datasets."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

from netCDF4 import Dataset
from pydantic import Field, validate_call

from hydroforge.core.validation import HydroForgeModel
from hydroforge.io.files import atomic_output_path
from hydroforge.io.netcdf.options import ensure_hdf5_plugins

NetCDFFormat = Literal[
    "NETCDF4",
    "NETCDF4_CLASSIC",
    "NETCDF3_CLASSIC",
    "NETCDF3_64BIT_OFFSET",
    "NETCDF3_64BIT_DATA",
]


@contextmanager
@validate_call(config=HydroForgeModel.model_config)
def atomic_netcdf_dataset(
    file_path: Annotated[Path, Field(strict=False)],
    *,
    format: NetCDFFormat = "NETCDF4",
) -> Iterator[Dataset]:
    """Create and atomically publish one complete on-disk NetCDF dataset."""

    ensure_hdf5_plugins()
    with atomic_output_path(file_path) as temporary:
        with Dataset(temporary, "w", format=format) as dataset:
            yield dataset

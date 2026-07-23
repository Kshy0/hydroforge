"""Backend-neutral serialization contracts and atomic file primitives."""

from hydroforge.serialization.files import atomic_output_path, atomic_write_text
from hydroforge.serialization.netcdf import (
    DEFAULT_NETCDF_OPTIONS,
    atomic_netcdf_dataset,
    atomic_netcdf_output,
    default_netcdf_options,
    normalize_netcdf_variable_options,
)

__all__ = [
    "DEFAULT_NETCDF_OPTIONS",
    "atomic_output_path",
    "atomic_write_text",
    "atomic_netcdf_dataset",
    "atomic_netcdf_output",
    "default_netcdf_options",
    "normalize_netcdf_variable_options",
]

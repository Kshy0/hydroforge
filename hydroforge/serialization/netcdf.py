"""Shared NetCDF serialization contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from netCDF4 import Dataset

from hydroforge.serialization.files import atomic_output_path


NETCDF_VARIABLE_OPTION_NAMES = frozenset({
    "compression", "zlib", "complevel", "shuffle", "szip_coding",
    "szip_pixels_per_block", "blosc_shuffle", "fletcher32", "contiguous",
    "chunksizes", "endian", "least_significant_digit",
    "significant_digits", "quantize_mode", "fill_value", "chunk_cache",
})
DEFAULT_NETCDF_OPTIONS: Mapping[str, Any] = MappingProxyType({
    "compression": "zlib",
    "complevel": 4,
})


def default_netcdf_options() -> dict[str, Any]:
    """Return an independent mutable copy of the canonical encoding default."""

    return dict(DEFAULT_NETCDF_OPTIONS)


def normalize_netcdf_variable_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach NetCDF ``createVariable`` keyword options."""

    if not isinstance(options, Mapping):
        raise TypeError("output_netcdf_options must be a mapping")
    normalized = dict(options)
    unknown = sorted(set(normalized) - NETCDF_VARIABLE_OPTION_NAMES)
    if unknown:
        raise ValueError(
            "unsupported NetCDF createVariable option(s): " + ", ".join(unknown)
        )
    for name in ("zlib", "shuffle", "fletcher32", "contiguous"):
        if name in normalized and type(normalized[name]) is not bool:
            raise TypeError(f"output_netcdf_options[{name!r}] must be a bool")
    if "complevel" in normalized:
        value = normalized["complevel"]
        if type(value) is not int or not 0 <= value <= 9:
            raise ValueError("NetCDF complevel must be an exact int in [0, 9]")
        if normalized.get("compression") in (None, False) and not normalized.get(
            "zlib", False,
        ):
            raise ValueError(
                "NetCDF complevel requires an explicit compression filter"
            )
    if "chunksizes" in normalized and normalized["chunksizes"] is not None:
        chunks = normalized["chunksizes"]
        if not isinstance(chunks, (tuple, list)):
            raise TypeError("NetCDF chunksizes must be a tuple, list, or None")
        invalid = [value for value in chunks if type(value) is not int or value < 1]
        if invalid:
            raise ValueError(
                "NetCDF chunksizes values must be exact positive ints, "
                f"got {invalid}"
            )
        normalized["chunksizes"] = tuple(chunks)
    if normalized.get("contiguous") and normalized.get("chunksizes") is not None:
        raise ValueError("NetCDF contiguous=True cannot be combined with chunksizes")
    if normalized.get("contiguous") and (
        normalized.get("compression") not in (None, False)
        or normalized.get("zlib", False)
    ):
        raise ValueError("NetCDF contiguous=True cannot be combined with compression")
    return normalized


@contextmanager
def atomic_netcdf_output(file_path: str | Path) -> Iterator[Path]:
    """Publish one same-directory NetCDF temporary file atomically."""

    with atomic_output_path(file_path) as temporary:
        yield temporary


@contextmanager
def atomic_netcdf_dataset(
    file_path: str | Path,
    **dataset_options: Any,
) -> Iterator[Dataset]:
    """Create and atomically publish one complete NetCDF dataset."""

    with atomic_netcdf_output(file_path) as temporary:
        with Dataset(temporary, "w", **dataset_options) as dataset:
            yield dataset

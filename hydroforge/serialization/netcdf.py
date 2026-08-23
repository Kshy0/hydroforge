"""Shared NetCDF serialization contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from inspect import signature
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict, Unpack

import numpy as np
from netCDF4 import Dataset
from pydantic import model_validator

from hydroforge.contracts.validation import HydroForgeModel, _immutable_dict
from hydroforge.serialization.files import atomic_output_path


class _NetCDFDatasetOptions(TypedDict, total=False):
    format: str


class _AtomicNetCDFDeclaration(HydroForgeModel):
    file_path: str | Path
    dataset_options: Mapping[str, Any]

    @model_validator(mode="after")
    def _validate_declaration(self):
        object.__setattr__(
            self, "dataset_options", _immutable_dict(self.dataset_options),
        )
        return self


OUTPUT_FORMAT = "hydroforge.statistics"
OUTPUT_VERSION = 3
COMMITTED_STEPS_ATTR = "hydroforge_committed_steps"
RUN_ID_ATTR = "hydroforge_run_id"

DEFAULT_NETCDF_OPTIONS: Mapping[str, Any] = MappingProxyType({
    "compression": "blosc_zstd",
    "complevel": 5,
    "blosc_shuffle": 1,
})

_ZLIB_FALLBACK_OPTIONS: Mapping[str, Any] = MappingProxyType({
    "compression": "zlib",
    "complevel": 4,
})

DEFAULT_NETCDF_CHUNK_BYTES = 4 * 1024 * 1024
MIN_BLOSC_CHUNK_BYTES = 128

_NETCDF_CREATE_VARIABLE_SIGNATURE = signature(Dataset.createVariable)

_NETCDF_COMPRESSION_FILTERS = frozenset({
    "zlib",
    "szip",
    "zstd",
    "bzip2",
    "blosc_lz",
    "blosc_lz4",
    "blosc_lz4hc",
    "blosc_zlib",
    "blosc_zstd",
})

LOGICAL_DTYPE_ATTR = "hydroforge_dtype"
BOOL_LOGICAL_DTYPE = "bool"
BOOL_NETCDF_STORAGE_DTYPE = np.dtype("u1")
BOOL_NETCDF_READ_DTYPES = frozenset({np.dtype("i1"), np.dtype("u1")})


def netcdf_dtype_encoding(dtype: Any) -> tuple[np.dtype, str | None]:
    """Return the physical NetCDF dtype and optional logical dtype marker."""

    normalized = np.dtype(dtype)
    if normalized == np.dtype(np.bool_):
        return BOOL_NETCDF_STORAGE_DTYPE, BOOL_LOGICAL_DTYPE
    return normalized, None


def decode_netcdf_logical_array(
    variable: Any,
    values: Any,
    *,
    name: str,
) -> np.ndarray | Any:
    """Decode one explicitly marked logical array without implicit casting."""

    if getattr(variable, LOGICAL_DTYPE_ATTR, None) != BOOL_LOGICAL_DTYPE:
        return values
    storage_dtype = np.dtype(variable.dtype)
    if storage_dtype not in BOOL_NETCDF_READ_DTYPES:
        raise TypeError(
            f"boolean NetCDF variable {name!r} must use i1/u1 storage; "
            f"got {storage_dtype}"
        )
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"boolean NetCDF variable {name!r} contains missing values")
    array = np.asarray(values)
    if array.size and not np.isin(array, (0, 1)).all():
        raise ValueError(
            f"boolean NetCDF variable {name!r} contains values outside 0/1"
        )
    return array.astype(np.bool_, copy=False)


def default_netcdf_options() -> dict[str, Any]:
    """Return an independent mutable copy of the canonical encoding default."""

    return dict(DEFAULT_NETCDF_OPTIONS)


def _uses_default_blosc_zstd_profile(options: Mapping[str, Any]) -> bool:
    """Return whether options request the canonical preferred compressor."""

    return (
        options.get("compression") == "blosc_zstd"
        and options.get("complevel") == 5
        and options.get("blosc_shuffle", 1) == 1
    )


def _blosc_chunk_is_too_small(
    dataset: Dataset,
    *,
    dtype: Any,
    dimensions: Sequence[str],
    options: Mapping[str, Any],
) -> bool:
    """Detect Blosc chunks at or below the filter's unsafe boundary."""

    dims = tuple(dimensions)
    if not dims:
        # Scalar variables have no chunk extent to enlarge and cannot satisfy
        # the Blosc filter's minimum input size.
        return True
    chunks = options.get("chunksizes")
    if chunks is not None:
        return math.prod(chunks) * np.dtype(dtype).itemsize <= MIN_BLOSC_CHUNK_BYTES
    resolved = tuple(dataset.dimensions[name] for name in dims)
    if any(dimension.isunlimited() for dimension in resolved):
        return False
    elements = math.prod(len(dimension) for dimension in resolved)
    return elements * np.dtype(dtype).itemsize <= MIN_BLOSC_CHUNK_BYTES


def _resolve_netcdf_compression_options_trusted(
    dataset: Dataset,
    *,
    dtype: Any,
    dimensions: Sequence[str],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the default Blosc-Zstd profile with a portable zlib fallback."""

    resolved = dict(options)
    if not _uses_default_blosc_zstd_profile(resolved):
        return resolved
    has_filter = getattr(dataset, "has_blosc_filter", None)
    try:
        blosc_available = callable(has_filter) and bool(has_filter())
    except (OSError, RuntimeError):
        blosc_available = False
    if blosc_available and not _blosc_chunk_is_too_small(
        dataset,
        dtype=dtype,
        dimensions=dimensions,
        options=resolved,
    ):
        return resolved
    resolved.pop("blosc_shuffle", None)
    resolved.update(_ZLIB_FALLBACK_OPTIONS)
    return resolved


def _create_netcdf_variable_trusted(
    dataset: Dataset,
    name: str,
    dtype: Any,
    dimensions: Sequence[str],
    *,
    options: Mapping[str, Any],
):
    """Create a variable after resolving the default compression fallback."""

    dims = tuple(dimensions)
    create_options = _resolve_netcdf_compression_options_trusted(
        dataset,
        dtype=dtype,
        dimensions=dims,
        options=options,
    )
    return dataset.createVariable(name, dtype, dims, **create_options)


def normalize_netcdf_variable_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach ``Dataset.createVariable`` keyword options."""

    if not isinstance(options, Mapping):
        raise TypeError("NetCDF variable options must be a mapping")
    normalized = dict(options)
    try:
        _NETCDF_CREATE_VARIABLE_SIGNATURE.bind(
            None,
            "__hydroforge_variable__",
            "f4",
            (),
            **normalized,
        )
    except TypeError as error:
        raise ValueError(
            f"unsupported NetCDF variable options: {error}"
        ) from error

    for name in ("zlib", "shuffle", "fletcher32", "contiguous"):
        if name in normalized and type(normalized[name]) is not bool:
            raise TypeError(f"NetCDF option {name!r} must be an exact bool")

    compression = normalized.get("compression")
    if compression is not None and compression is not False:
        if type(compression) is not str:
            raise TypeError(
                "NetCDF option 'compression' must be an exact str, False, or None"
            )
        if compression not in _NETCDF_COMPRESSION_FILTERS:
            raise ValueError(
                f"unsupported NetCDF compression filter {compression!r}"
            )
    if (
        normalized.get("zlib") is True
        and compression not in {None, False, "zlib"}
    ):
        raise ValueError(
            "NetCDF zlib=True cannot be combined with a different compression filter"
        )

    if "complevel" in normalized:
        level = normalized["complevel"]
        if type(level) is not int or not 0 <= level <= 9:
            raise ValueError("NetCDF complevel must be an exact int in [0, 9]")
        if compression in {None, False} and normalized.get("zlib") is not True:
            raise ValueError("NetCDF complevel requires a compression filter")

    chunks = normalized.get("chunksizes")
    if chunks is not None:
        if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            raise TypeError("NetCDF chunksizes must be a sequence of integers")
        chunks = tuple(chunks)
        if any(type(extent) is not int or extent <= 0 for extent in chunks):
            raise ValueError("NetCDF chunksizes must contain positive exact integers")
        normalized["chunksizes"] = chunks

    if "blosc_shuffle" in normalized:
        value = normalized["blosc_shuffle"]
        if type(value) is not int or value not in {0, 1, 2}:
            raise ValueError("NetCDF blosc_shuffle must be exactly 0, 1, or 2")
    if "szip_coding" in normalized:
        value = normalized["szip_coding"]
        if type(value) is not str or value not in {"nn", "ec"}:
            raise ValueError("NetCDF szip_coding must be 'nn' or 'ec'")
    if "szip_pixels_per_block" in normalized:
        value = normalized["szip_pixels_per_block"]
        if type(value) is not int or value < 4 or value > 32 or value % 2:
            raise ValueError(
                "NetCDF szip_pixels_per_block must be an even exact int in [4, 32]"
            )
    if "endian" in normalized:
        value = normalized["endian"]
        if type(value) is not str or value not in {"native", "little", "big"}:
            raise ValueError("NetCDF endian must be 'native', 'little', or 'big'")
    for name, minimum in (
        ("least_significant_digit", 0),
        ("significant_digits", 1),
    ):
        if name in normalized and normalized[name] is not None:
            value = normalized[name]
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"NetCDF {name} must be an exact int >= {minimum} or None"
                )
    if "quantize_mode" in normalized:
        value = normalized["quantize_mode"]
        if type(value) is not str or value not in {
            "BitGroom", "GranularBitRound", "BitRound",
        }:
            raise ValueError(
                "NetCDF quantize_mode must be 'BitGroom', "
                "'GranularBitRound', or 'BitRound'"
            )
    if "chunk_cache" in normalized and normalized["chunk_cache"] is not None:
        value = normalized["chunk_cache"]
        if type(value) is not int or value <= 0:
            raise ValueError("NetCDF chunk_cache must be a positive exact int")

    if normalized.get("contiguous") is True:
        conflicting = []
        if chunks is not None:
            conflicting.append("chunksizes")
        if compression not in {None, False} or normalized.get("zlib") is True:
            conflicting.append("compression")
        if normalized.get("fletcher32") is True:
            conflicting.append("fletcher32")
        if conflicting:
            raise ValueError(
                "NetCDF contiguous=True cannot be combined with "
                + ", ".join(conflicting)
            )
    return normalized


def _largest_divisor_not_exceeding(value: int, limit: int) -> int:
    """Return the largest divisor of ``value`` no greater than ``limit``."""

    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _fit_spatial_chunks(
    shape: Sequence[int], *, max_elements: int,
) -> tuple[int, ...]:
    """Tile a row deterministically without exceeding an element budget."""

    chunks = [int(extent) for extent in shape]
    if any(extent <= 0 for extent in chunks):
        raise ValueError("NetCDF output dimensions must be positive")
    while math.prod(chunks) > max_elements:
        axis = max(range(len(chunks)), key=chunks.__getitem__)
        other = math.prod(chunks[:axis] + chunks[axis + 1:])
        fitted = max(1, max_elements // max(other, 1))
        if fitted >= chunks[axis]:
            fitted = max(1, chunks[axis] // 2)
        chunks[axis] = fitted
    return tuple(chunks)


def plan_streaming_netcdf_chunks(
    options: Mapping[str, Any],
    *,
    dtype: Any,
    row_shape: Sequence[int],
    write_batch_size: int,
    target_bytes: int = DEFAULT_NETCDF_CHUNK_BYTES,
) -> dict[str, Any]:
    """Add an aligned streaming chunk layout unless the caller chose one."""

    normalized = dict(options)
    if "chunksizes" in normalized or normalized.get("contiguous") is True:
        return normalized
    if type(write_batch_size) is not int or write_batch_size <= 0:
        raise ValueError("NetCDF write_batch_size must be a positive exact int")
    if type(target_bytes) is not int or target_bytes <= 0:
        raise ValueError("NetCDF target chunk bytes must be a positive exact int")

    storage = np.dtype(dtype)
    shape = tuple(int(extent) for extent in row_shape)
    row_elements = math.prod(shape) if shape else 1
    row_bytes = max(1, row_elements * storage.itemsize)
    max_time_chunk = max(1, min(write_batch_size, target_bytes // row_bytes))
    time_chunk = _largest_divisor_not_exceeding(
        write_batch_size, max_time_chunk,
    )
    spatial_budget = max(1, target_bytes // (time_chunk * storage.itemsize))
    spatial_chunks = _fit_spatial_chunks(
        shape, max_elements=spatial_budget,
    ) if shape else ()
    compression = normalized.get("compression")
    if type(compression) is str and compression.startswith("blosc_"):
        spatial_elements = math.prod(spatial_chunks) if spatial_chunks else 1
        minimum_time = math.ceil(
            MIN_BLOSC_CHUNK_BYTES / (spatial_elements * storage.itemsize)
        )
        time_chunk = max(time_chunk, minimum_time)
    normalized["chunksizes"] = (time_chunk, *spatial_chunks)
    return normalized


def prepare_netcdf_variable_options(
    options: Mapping[str, Any],
    *,
    dtype: Any,
    dimensions: Sequence[str],
    name: str,
    logical_dtype: str | None = None,
) -> dict[str, Any]:
    """Validate options against one concrete variable storage contract."""

    if logical_dtype not in {None, BOOL_LOGICAL_DTYPE}:
        raise ValueError(f"unsupported NetCDF logical dtype {logical_dtype!r}")
    if not isinstance(dimensions, Sequence) or isinstance(
        dimensions, (str, bytes),
    ):
        raise TypeError("NetCDF variable dimensions must be a sequence")
    dims = tuple(dimensions)
    if any(type(dimension) is not str or not dimension for dimension in dims):
        raise TypeError("NetCDF variable dimensions must be non-empty exact strings")
    storage = np.dtype(dtype)
    normalized = normalize_netcdf_variable_options(options)
    return _prepare_netcdf_variable_options_trusted(
        normalized,
        dtype=storage,
        dimensions=dims,
        name=name,
        logical_dtype=logical_dtype,
    )


def _prepare_netcdf_variable_options_trusted(
    options: Mapping[str, Any],
    *,
    dtype: Any,
    dimensions: Sequence[str],
    name: str,
    logical_dtype: str | None = None,
) -> dict[str, Any]:
    """Bind normalized options to an already validated variable schema."""

    dims = tuple(dimensions)
    storage = np.dtype(dtype)
    normalized = dict(options)
    chunks = normalized.get("chunksizes")
    if chunks is not None and len(chunks) != len(dims):
        raise ValueError(
            f"NetCDF chunksizes for {name!r} have rank {len(chunks)}, "
            f"expected rank {len(dims)}"
        )
    if normalized.get("contiguous") is True and chunks is not None:
        raise ValueError(
            f"NetCDF variable {name!r} cannot be contiguous and chunked"
        )
    if storage.kind != "f" and any(
        normalized.get(option) is not None
        for option in ("least_significant_digit", "significant_digits")
    ):
        raise TypeError(
            f"NetCDF quantization for {name!r} requires floating storage"
        )
    if "fill_value" not in normalized:
        return normalized
    fill = normalized["fill_value"]
    if fill is None or fill is False or (
        type(fill) is str and fill == "default"
    ):
        return normalized
    if logical_dtype == BOOL_LOGICAL_DTYPE:
        raise TypeError(
            f"boolean NetCDF variable {name!r} cannot use a numeric fill_value; "
            "0 and 1 are both valid logical values"
        )
    if storage.kind in "iu":
        if type(fill) is not int:
            raise TypeError(
                f"integer NetCDF variable {name!r} requires an exact int fill_value"
            )
        limits = np.iinfo(storage)
        if not limits.min <= fill <= limits.max:
            raise OverflowError(
                f"NetCDF fill_value for {name!r} is outside {storage} range"
            )
        return normalized
    if storage.kind == "f":
        if type(fill) is not float:
            raise TypeError(
                f"floating NetCDF variable {name!r} requires an exact float "
                "fill_value"
            )
        if math.isfinite(fill):
            limits = np.finfo(storage)
            if abs(fill) > limits.max:
                raise OverflowError(
                    f"NetCDF fill_value for {name!r} is outside {storage} range"
                )
            encoded = np.asarray(fill, dtype=storage).item()
            if fill != 0.0 and encoded == 0.0:
                raise OverflowError(
                    f"NetCDF fill_value for {name!r} underflows {storage}"
                )
        return normalized
    raise TypeError(
        f"NetCDF fill_value for {name!r} is unsupported for dtype {storage}"
    )


@contextmanager
def atomic_netcdf_output(file_path: str | Path) -> Iterator[Path]:
    """Publish one same-directory NetCDF temporary file atomically."""

    with atomic_output_path(file_path) as temporary:
        yield temporary


@contextmanager
def atomic_netcdf_dataset(
    file_path: str | Path,
    **dataset_options: Unpack[_NetCDFDatasetOptions],
) -> Iterator[Dataset]:
    """Create and atomically publish one complete NetCDF dataset."""

    declaration = _AtomicNetCDFDeclaration(
        file_path=file_path, dataset_options=dataset_options,
    )
    with _atomic_netcdf_dataset_trusted(
        declaration.file_path,
        **dict(declaration.dataset_options),
    ) as dataset:
        yield dataset


@contextmanager
def _atomic_netcdf_dataset_trusted(
    file_path: str | Path,
    **dataset_options: Unpack[_NetCDFDatasetOptions],
) -> Iterator[Dataset]:
    """Create an atomic dataset from a validated path and option mapping."""

    with atomic_netcdf_output(file_path) as temporary:
        with Dataset(
            temporary, "w", **dataset_options,
        ) as dataset:
            yield dataset

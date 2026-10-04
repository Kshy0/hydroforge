"""NetCDF variable options: compression profile, filter plugins, and chunk plans."""

from __future__ import annotations

import atexit
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from functools import cache, partial
from importlib.metadata import distributions
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import MappingProxyType
from typing import Annotated, Any

import netCDF4 as _netcdf4
import numpy as np
from netCDF4 import Dataset
from pydantic import BeforeValidator, Field, validate_call

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.core.validation import FrozenMapping, HydroForgeModel
from hydroforge.io.netcdf.encoding import BOOL_LOGICAL_DTYPE

DEFAULT_NETCDF_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {
        "compression": "blosc_zstd",
        "complevel": 5,
        "blosc_shuffle": 1,
    }
)

_ZLIB_FALLBACK_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {
        "compression": "zlib",
        "complevel": 4,
    }
)

DEFAULT_NETCDF_CHUNK_BYTES = 4 * 1024 * 1024
MIN_BLOSC_CHUNK_BYTES = 128

_NETCDF_CREATE_VARIABLE_SIGNATURE = signature(Dataset.createVariable)

_NETCDF_COMPRESSION_FILTERS = frozenset(
    {
        "zlib",
        "szip",
        "zstd",
        "bzip2",
        "blosc_lz",
        "blosc_lz4",
        "blosc_lz4hc",
        "blosc_zlib",
        "blosc_zstd",
    }
)


def _conda_hdf5_plugin_directories() -> tuple[Path, ...]:
    """Return standard Conda HDF5 filter-plugin directories that exist."""

    prefixes: list[Path] = []
    for raw_prefix in (sys.prefix, os.environ.get("CONDA_PREFIX")):
        if not raw_prefix:
            continue
        prefix = Path(raw_prefix)
        if prefix not in prefixes and (prefix / "conda-meta").is_dir():
            prefixes.append(prefix)
    candidates = tuple(
        candidate
        for prefix in prefixes
        for candidate in (
            prefix / "lib" / "hdf5" / "plugin",
            prefix / "lib" / "hdf5" / "plugins",
            prefix / "lib" / "plugin",
            prefix / "Library" / "hdf5" / "lib" / "plugin",
        )
        if candidate.is_dir()
        and any(
            path.is_file()
            and ("nch5blosc" in path.name or "nczhdf5filters" in path.name)
            for path in candidate.iterdir()
        )
    )
    return candidates


def _netcdf4_installer() -> str | None:
    """Return the installer recorded for the imported ``netCDF4`` package."""

    package_directory = Path(_netcdf4.__file__).resolve().parent
    matches: list[str] = []
    for distribution in distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower().replace("-", "_") != "netcdf4":
            continue
        try:
            distribution_package = Path(distribution.locate_file("netCDF4")).resolve()
        except (OSError, ValueError):
            continue
        if distribution_package != package_directory:
            continue
        installer = distribution.read_text("INSTALLER")
        if installer:
            matches.append(installer.strip().lower())

    if "pip" in matches:
        return "pip"
    return matches[0] if matches else None


@cache
def ensure_hdf5_plugins() -> None:
    """Use Conda's HDF5 filter plugins for a Conda-installed netCDF4, once.

    Called before HydroForge first opens or creates a NetCDF file and before
    the Blosc probe child starts, so importing HydroForge does not scan
    package metadata.  Best effort: unreadable package metadata or plugin
    directories leave ``HDF5_PLUGIN_PATH`` unchanged.
    """

    try:
        if _netcdf4_installer() != "conda":
            return
        candidates = _conda_hdf5_plugin_directories()
    except (OSError, ValueError):
        return
    if not candidates:
        return
    current = os.environ.get("HDF5_PLUGIN_PATH")
    package_plugins = Path(_netcdf4.__file__).resolve().parent / "plugins"
    if current and Path(current).resolve() != package_plugins:
        return
    os.environ["HDF5_PLUGIN_PATH"] = os.pathsep.join(
        str(candidate) for candidate in candidates
    )


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
    """Reject chunks whose size is unsafe or left to unlimited-axis heuristics."""

    dims = tuple(dimensions)
    if not dims:
        # Scalar variables have no chunk extent to enlarge and cannot satisfy
        # the Blosc filter's minimum input size.
        return True
    chunks = options.get("chunksizes")
    if chunks is not None:
        return math.prod(chunks) * np.dtype(dtype).itemsize < MIN_BLOSC_CHUNK_BYTES
    resolved = tuple(dataset.dimensions[name] for name in dims)
    if any(dimension.isunlimited() for dimension in resolved):
        return True
    elements = math.prod(len(dimension) for dimension in resolved)
    return elements * np.dtype(dtype).itemsize < MIN_BLOSC_CHUNK_BYTES


# netCDF-C registers filters as mandatory, and its Blosc plugin fails any chunk
# whose Blosc output would exceed the input (incompressible data) instead of
# storing it raw.  Such a failure also leaves the HDF5 file impossible to
# close, so the probe runs in a child process and writes one incompressible
# chunk beside a compressible one.
_BLOSC_PROBE_SCRIPT = """
import json, sys
import numpy as np
from netCDF4 import Dataset
path, options = sys.argv[1], json.loads(sys.argv[2])
arrays = (
    np.arange(1024, dtype=np.float32),
    np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8),
)
with Dataset(path, "w", format="NETCDF4") as dataset:
    for index, values in enumerate(arrays):
        dataset.createDimension(f"n{index}", values.size)
        variable = dataset.createVariable(
            f"v{index}", values.dtype, (f"n{index}",),
            chunksizes=(values.size,), **options,
        )
        if not variable.filters().get("blosc"):
            raise SystemExit(1)
        variable[:] = values
with Dataset(path, "r") as dataset:
    for index, values in enumerate(arrays):
        if not np.array_equal(np.asarray(dataset.variables[f"v{index}"][:]), values):
            raise SystemExit(1)
print("ok")
"""


_blosc_probe_lock = Lock()
# One probe per process: the started child until collected, then its verdict.
_blosc_probe: tuple[subprocess.Popen, TemporaryDirectory] | None = None
_blosc_probe_verdict: bool | None = None


def _start_blosc_probe_locked() -> None:
    global _blosc_probe
    if _blosc_probe is not None or _blosc_probe_verdict is not None:
        return
    ensure_hdf5_plugins()
    directory = TemporaryDirectory(prefix="hydroforge_netcdf_probe-")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _BLOSC_PROBE_SCRIPT,
                str(Path(directory.name) / "blosc_zstd.nc"),
                json.dumps(dict(DEFAULT_NETCDF_OPTIONS)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except BaseException:
        with cleanup_on_exit("Blosc capability probe startup", (directory.cleanup,)):
            raise
    _blosc_probe = (process, directory)


def start_blosc_zstd_probe(
    options: Iterable[Mapping[str, Any]] = (DEFAULT_NETCDF_OPTIONS,),
) -> None:
    """Start early only when a planned variable requests the preferred filter."""

    if not any(_uses_default_blosc_zstd_profile(value) for value in options):
        return
    with _blosc_probe_lock:
        _start_blosc_probe_locked()


def _probe_blosc_zstd_filter(*, start_if_needed: bool = True) -> bool:
    """Verify that the active NetCDF/HDF5 stack can store any Blosc chunk.

    The first caller collects the probe under the lock; concurrent callers
    wait for and share its verdict.
    """

    global _blosc_probe, _blosc_probe_verdict
    with _blosc_probe_lock:
        if _blosc_probe_verdict is not None:
            return _blosc_probe_verdict
        if _blosc_probe is None and not start_if_needed:
            return False
        try:
            _start_blosc_probe_locked()
        except (OSError, subprocess.SubprocessError):
            _blosc_probe_verdict = False
            return False
        process, directory = _blosc_probe
        _blosc_probe = None
        with cleanup_on_exit("Blosc capability probe", (directory.cleanup,)):
            try:
                stdout, _stderr = process.communicate(timeout=120)
            except BaseException as error:
                _blosc_probe_verdict = False
                with cleanup_on_exit(
                    "Blosc capability probe child", (process.kill, process.communicate)
                ):
                    if not isinstance(error, subprocess.TimeoutExpired):
                        raise
            else:
                _blosc_probe_verdict = (
                    process.returncode == 0 and stdout.strip() == "ok"
                )
        return _blosc_probe_verdict


atexit.register(partial(_probe_blosc_zstd_filter, start_if_needed=False))


def _resolve_compression_options(
    dataset: Dataset,
    *,
    dtype: Any,
    dimensions: Sequence[str],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the default profile to usable Blosc, native Zstd, then zlib."""

    resolved = dict(options)
    if not _uses_default_blosc_zstd_profile(resolved):
        return resolved
    has_filter = getattr(dataset, "has_blosc_filter", None)
    try:
        blosc_available = callable(has_filter) and bool(has_filter())
    except (OSError, RuntimeError):
        blosc_available = False
    if (
        blosc_available
        and not _blosc_chunk_is_too_small(
            dataset,
            dtype=dtype,
            dimensions=dimensions,
            options=resolved,
        )
        and _probe_blosc_zstd_filter()
    ):
        return resolved
    has_filter = getattr(dataset, "has_zstd_filter", None)
    try:
        zstd_available = (
            bool(getattr(_netcdf4, "__has_zstandard_support__", False))
            and callable(has_filter)
            and bool(has_filter())
        )
    except (OSError, RuntimeError):
        zstd_available = False
    resolved.pop("blosc_shuffle", None)
    resolved.update(
        {"compression": "zstd", "complevel": 1}
        if zstd_available
        else _ZLIB_FALLBACK_OPTIONS
    )
    return resolved


def create_netcdf_variable(
    dataset: Dataset,
    name: str,
    dtype: Any,
    dimensions: Sequence[str],
    *,
    options: Mapping[str, Any],
):
    """Create a variable from :func:`prepare_netcdf_variable_options` options.

    Where this stack cannot store every Blosc chunk, the default falls back
    to native Zstd level 1 when available, otherwise portable zlib level 4.
    """

    dims = tuple(dimensions)
    create_options = _resolve_compression_options(
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
        raise ValueError(f"unsupported NetCDF variable options: {error}") from error

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
            raise ValueError(f"unsupported NetCDF compression filter {compression!r}")
    if normalized.get("zlib") is True and compression not in {None, False, "zlib"}:
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

    for name, kind, choices, description in (
        ("blosc_shuffle", int, {0, 1, 2}, "exactly 0, 1, or 2"),
        ("szip_coding", str, {"nn", "ec"}, "'nn' or 'ec'"),
        ("endian", str, {"native", "little", "big"}, "'native', 'little', or 'big'"),
        (
            "quantize_mode",
            str,
            {"BitGroom", "GranularBitRound", "BitRound"},
            "'BitGroom', 'GranularBitRound', or 'BitRound'",
        ),
    ):
        if name in normalized:
            value = normalized[name]
            if type(value) is not kind or value not in choices:
                raise ValueError(f"NetCDF {name} must be {description}")
    if "szip_pixels_per_block" in normalized:
        value = normalized["szip_pixels_per_block"]
        if type(value) is not int or value < 4 or value > 32 or value % 2:
            raise ValueError(
                "NetCDF szip_pixels_per_block must be an even exact int in [4, 32]"
            )
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


NetCDFOptions = Annotated[
    FrozenMapping[str, Any], BeforeValidator(normalize_netcdf_variable_options)
]


def _largest_divisor_not_exceeding(value: int, limit: int) -> int:
    """Return the largest divisor of ``value`` no greater than ``limit``."""

    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _fit_spatial_chunks(
    shape: Sequence[int],
    *,
    max_elements: int,
) -> tuple[int, ...]:
    """Tile a row deterministically without exceeding an element budget.

    NetCDF represents a zero-sized dimension as an unlimited dimension.  Its
    chunk extent must still be positive, so an empty logical axis uses a
    physical chunk extent of one while retaining a current dimension length
    of zero.
    """

    chunks = [max(1, extent) for extent in shape]
    while math.prod(chunks) > max_elements:
        axis = max(range(len(chunks)), key=chunks.__getitem__)
        other = math.prod(chunks[:axis] + chunks[axis + 1 :])
        fitted = max(1, max_elements // max(other, 1))
        if fitted >= chunks[axis]:
            fitted = max(1, chunks[axis] // 2)
        chunks[axis] = fitted
    return tuple(chunks)


def _exact_chunk_count(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("NetCDF chunk counts must be exact integers")
    return value


_ChunkExtent = Annotated[int, BeforeValidator(_exact_chunk_count), Field(ge=0)]
_PositiveChunkCount = Annotated[_ChunkExtent, Field(gt=0)]


@validate_call(config=HydroForgeModel.model_config)
def plan_streaming_netcdf_chunks(
    options: Mapping[str, Any],
    *,
    dtype: Any,
    row_shape: Sequence[_ChunkExtent],
    write_batch_size: _PositiveChunkCount,
    target_bytes: _PositiveChunkCount = DEFAULT_NETCDF_CHUNK_BYTES,
) -> dict[str, Any]:
    """Plan aligned streaming chunks and their cache unless the caller chose one."""

    normalized = dict(options)
    if "chunksizes" in normalized or normalized.get("contiguous") is True:
        return normalized

    storage = np.dtype(dtype)
    shape = tuple(row_shape)
    row_elements = math.prod(shape)
    row_bytes = max(1, row_elements * storage.itemsize)
    max_time_chunk = max(1, min(write_batch_size, target_bytes // row_bytes))
    time_chunk = _largest_divisor_not_exceeding(
        write_batch_size,
        max_time_chunk,
    )
    spatial_budget = max(1, target_bytes // (time_chunk * storage.itemsize))
    spatial_chunks = (
        _fit_spatial_chunks(
            shape,
            max_elements=spatial_budget,
        )
        if shape
        else ()
    )
    compression = normalized.get("compression")
    if type(compression) is str and compression.startswith("blosc_"):
        spatial_elements = math.prod(spatial_chunks)
        minimum_time = math.ceil(
            MIN_BLOSC_CHUNK_BYTES / (spatial_elements * storage.itemsize)
        )
        time_chunk = max(time_chunk, minimum_time)
    normalized["chunksizes"] = (time_chunk, *spatial_chunks)
    # Appends fill spatial chunks and revisit at most the final time chunk
    # after a partial flush. A bounded cache avoids retaining many completed
    # chunks on every open output variable. Explicit layouts/budgets stay as-is.
    normalized.setdefault(
        "chunk_cache",
        max(target_bytes, math.prod(normalized["chunksizes"]) * storage.itemsize),
    )
    return normalized


def plan_fixed_netcdf_chunks(
    options: Mapping[str, Any],
    *,
    dtype: Any,
    shape: Sequence[int],
    target_bytes: int = DEFAULT_NETCDF_CHUNK_BYTES,
) -> dict[str, Any]:
    """Add a bounded chunk layout for one fixed-shape array unless chosen."""

    normalized = dict(options)
    if not shape or "chunksizes" in normalized or normalized.get("contiguous") is True:
        return normalized
    normalized["chunksizes"] = _fit_spatial_chunks(
        tuple(shape),
        max_elements=max(1, target_bytes // np.dtype(dtype).itemsize),
    )
    return normalized


def prepare_netcdf_variable_options(
    options: Mapping[str, Any],
    *,
    dtype: Any,
    dimensions: Sequence[str],
    name: str,
    logical_dtype: str | None = None,
    shape: Sequence[int | None] | None = None,
) -> dict[str, Any]:
    """Bind :func:`normalize_netcdf_variable_options` output to one variable."""

    dims = tuple(dimensions)
    storage = np.dtype(dtype)
    normalized = dict(options)
    chunks = normalized.get("chunksizes")
    if chunks is not None and len(chunks) != len(dims):
        raise ValueError(
            f"NetCDF chunksizes for {name!r} have rank {len(chunks)}, "
            f"expected rank {len(dims)}"
        )
    if shape is not None:
        if len(shape) != len(dims):
            raise ValueError(f"NetCDF shape for {name!r} has the wrong rank")
        if chunks is not None and any(
            extent is not None and extent > 0 and chunk > extent
            for chunk, extent in zip(chunks, shape, strict=True)
        ):
            raise ValueError(
                f"NetCDF chunksizes for {name!r} exceed fixed dimension extents"
            )
    if storage.kind != "f" and any(
        normalized.get(option) is not None
        for option in ("least_significant_digit", "significant_digits")
    ):
        raise TypeError(f"NetCDF quantization for {name!r} requires floating storage")
    if "fill_value" not in normalized:
        return normalized
    fill = normalized["fill_value"]
    if fill is None or fill is False or (type(fill) is str and fill == "default"):
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
                f"floating NetCDF variable {name!r} requires an exact float fill_value"
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

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Lossless array canonicalization, integer index lookup, and dtype tables."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
from pydantic import AfterValidator, BeforeValidator, validate_call

from hydroforge.core.devices import devices_match
from hydroforge.core.validation import HydroForgeModel, _ImmutableDict

NumericScalar: TypeAlias = int | float | np.integer | np.floating
NumericValue: TypeAlias = NumericScalar | np.ndarray


def immutable_array(
    value: Any,
    *,
    dtype: Any = None,
    order: Literal["C", "F", "K"] = "C",
) -> np.ndarray:
    """Return an owned ndarray backed by an immutable Python buffer.

    ``array.setflags(write=False)`` is only advisory when an ndarray owns its
    allocation: callers can normally turn the write flag back on.  Arrays
    exposed by frozen public models therefore use ``bytes`` as their ultimate
    storage owner, which makes ``setflags(write=True)`` fail as well.

    Object arrays cannot safely be reconstructed from their raw pointer bytes
    and are deliberately rejected at the public data boundary.
    """

    source = np.asarray(value)
    target_dtype = source.dtype if dtype is None else np.dtype(dtype)
    if target_dtype.hasobject:
        raise ValueError("immutable arrays must not use object dtype")
    canonical = np.asarray(
        source,
        dtype=target_dtype,
        order=order,
    )
    if not canonical.flags.c_contiguous and not canonical.flags.f_contiguous:
        canonical = np.array(canonical, order=order, copy=True)
    storage_order: Literal["C", "F"] = (
        "F"
        if canonical.flags.f_contiguous and not canonical.flags.c_contiguous
        else "C"
    )
    payload = canonical.tobytes(order=storage_order)
    return np.ndarray(
        canonical.shape,
        dtype=canonical.dtype,
        buffer=payload,
        order=storage_order,
    )


def immutable_metadata(value: Any, *, label: str) -> Any:
    """Detach recursively nested metadata and seal numerical arrays."""

    if isinstance(value, Mapping):
        return _ImmutableDict(
            (
                immutable_metadata(key, label=f"{label} key"),
                immutable_metadata(item, label=f"{label}[{key!r}]"),
            )
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            immutable_metadata(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(
            immutable_metadata(item, label=f"{label} item") for item in value
        )
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{label} must not contain masked arrays")
    if isinstance(value, np.ndarray):
        return immutable_array(value, order="K")
    if isinstance(value, torch.Tensor):
        raise ValueError(
            f"{label} must not contain torch.Tensor values; use immutable "
            "Python or NumPy metadata"
        )
    return deepcopy(value)


def canonical_ids(values: np.ndarray, *, label: str) -> np.ndarray:
    """Return one canonical int64 identifier vector without changing shape."""

    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{label} contains missing values")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must contain integers")
    if (
        array.dtype.kind == "u"
        and array.size
        and int(array.max()) > np.iinfo(np.int64).max
    ):
        raise ValueError(f"{label} contains a value outside int64 range")
    return np.array(array, dtype=np.int64, order="C", copy=True)


def _unique_ids(value: Any) -> np.ndarray:
    ids = canonical_ids(value, label="ids")
    ordered = np.sort(ids)
    if np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("ids must be unique")
    return ids


UniqueIds = Annotated[np.ndarray, BeforeValidator(_unique_ids)]
"""A one-dimensional vector of unique integer IDs, canonicalized to int64."""


def finite_float64(value: NumericScalar, *, label: str) -> float:
    """Return one finite float64 scalar without lossy coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    if isinstance(value, Integral):
        exact = int(result) == int(value)
    elif isinstance(value, np.floating) and value.dtype.itemsize > 8:
        exact = bool(np.longdouble(result) == value)
    else:
        exact = bool(value == result)
    if not exact:
        raise ValueError(f"{label} is not exactly representable as float64")
    return result


def positive_finite_float64(value: NumericScalar, *, label: str) -> float:
    """Return one positive float64 scalar without lossy coercion."""

    try:
        result = finite_float64(value, label=label)
    except ValueError as error:
        if "not exactly representable as float64" in str(error):
            raise
        raise ValueError(f"{label} must be a finite positive real number") from error
    if result <= 0.0:
        raise ValueError(f"{label} must be a finite positive real number")
    return result


def canonical_floating_array(
    value: NumericValue,
    *,
    dtype: str,
    label: str,
    allow_nan: bool = False,
) -> np.ndarray:
    """Materialize finite float data without lossy integer conversion.

    Float narrowing rejects finite values outside the target range but lets
    tiny magnitudes round to subnormal values or zero.
    """

    if np.ma.isMaskedArray(value):
        raise ValueError(f"{label} must not be a masked array")
    source = np.asarray(value)
    if source.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must contain real numeric values")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    target = np.dtype(dtype)
    if source.dtype.kind in {"i", "u"}:
        result = np.asarray(source, dtype=target)
        exact_bits = np.finfo(target).nmant + 1
        value_bits = 8 * source.dtype.itemsize - (source.dtype.kind == "i")
        if value_bits > exact_bits and source.size:
            # Integers up to 2**exact_bits are exact; only larger magnitudes
            # need the exact (object) comparison.
            limit = 2**exact_bits
            large = (source > limit) | (source < -limit)
            if large.any() and not np.array_equal(
                source[large].astype(object),
                result[large].astype(object),
            ):
                raise ValueError(
                    f"{label} contains integers that are not exactly "
                    f"representable as {dtype}"
                )
        return result if result.ndim == 0 else np.ascontiguousarray(result)

    narrowing = source.dtype.itemsize > target.itemsize
    if source.size:
        # min/max reductions detect NaN/inf and the narrowing range in one
        # pass each without allocating boolean masks.
        if allow_nan:
            if np.isinf(source).any():
                raise ValueError(f"{label} contains infinite values")
            low = high = 0.0
            if narrowing:
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    low, high = np.nanmin(source), np.nanmax(source)
        else:
            low, high = source.min(), source.max()
            if not (np.isfinite(low) and np.isfinite(high)):
                raise ValueError(f"{label} contains non-finite values")
        if narrowing and max(-low, high) > np.finfo(target).max:
            if target == np.dtype(np.float32):
                raise ValueError(f"{label} contains values outside float32 range")
            raise ValueError(f"{label} overflowed {dtype}")
    result = np.asarray(source, dtype=target)
    return result if result.ndim == 0 else np.ascontiguousarray(result)


def canonical_float64(value: NumericValue, *, label: str) -> np.ndarray:
    """Return a contiguous float64 copy without changing any numeric value.

    Spatial coordinates and bounds are identities, not intermediate numerical
    work arrays.  Accepting an integer or extended-precision value that changes
    during float64 canonicalization can silently move a point, merge cells, or
    make two independently stored grids appear equal.
    """

    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{label} contains missing values")
    source = np.asarray(value)
    if source.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must contain real numbers")
    if not np.isfinite(source).all():
        raise ValueError(f"{label} must contain only finite values")

    result = np.array(source, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains values outside float64 range")

    if source.dtype.kind in {"i", "u"} or source.dtype.itemsize > 8:
        with np.errstate(invalid="ignore", over="ignore"):
            restored = result.astype(source.dtype)
        if not np.array_equal(restored, source):
            raise ValueError(
                f"{label} contains values that cannot be represented exactly as float64"
            )
    return result


def exact_numeric_array_equal(
    left: NumericValue,
    right: NumericValue,
) -> bool:
    """Compare real numeric arrays without NumPy's lossy mixed promotion."""

    if np.ma.isMaskedArray(left) or np.ma.isMaskedArray(right):
        return False
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind not in {"f", "i", "u"} or (
        right_array.dtype.kind not in {"f", "i", "u"}
    ):
        return False
    if left_array.dtype.kind == right_array.dtype.kind:
        return bool(np.array_equal(left_array, right_array))
    # In particular, int64/uint64 mixed with float64 can otherwise promote to
    # float64 and make adjacent large integers compare equal.  Object arrays
    # use Python's exact integer/float comparison rules.
    return bool(
        np.array_equal(
            left_array.astype(object),
            right_array.astype(object),
        )
    )


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------


def _integer_vector(value: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError("index lookup arrays must not be masked arrays")
    if value.ndim != 1:
        raise ValueError("index lookup arrays must be one-dimensional")
    if value.dtype.kind not in {"i", "u"}:
        raise ValueError("index lookup arrays must contain integers")
    return value


_IntegerVector = Annotated[np.ndarray, AfterValidator(_integer_vector)]


@validate_call(config=HydroForgeModel.model_config)
def find_indices_in(a: _IntegerVector, b: _IntegerVector) -> np.ndarray:
    """Return the index in unique *b* of each element of *a*, or ``-1``.

    Arrays of different integer dtypes are compared as int64.
    """

    if a.dtype != b.dtype:
        a = canonical_ids(a, label="index lookup query")
        b = canonical_ids(b, label="index lookup target")
    order = np.argsort(b)
    sorted_b = b[order]
    if np.any(sorted_b[1:] == sorted_b[:-1]):
        raise ValueError("index lookup target must contain unique values")
    pos_in_sorted = np.searchsorted(sorted_b, a)
    valid_mask = pos_in_sorted < len(sorted_b)
    hit_mask = np.zeros_like(a, dtype=bool)
    hit_mask[valid_mask] = sorted_b[pos_in_sorted[valid_mask]] == a[valid_mask]
    index = np.full_like(pos_in_sorted, -1, dtype=int)
    index[hit_mask] = order[pos_in_sorted[hit_mask]]
    return index


_TORCH_INDEX_DTYPES = frozenset(
    {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.uint16,
        torch.int32,
        torch.uint32,
        torch.int64,
    }
)


def _index_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError("torch index lookup tensors must be one-dimensional")
    if value.dtype not in _TORCH_INDEX_DTYPES:
        raise ValueError("torch index lookup tensors must contain integers")
    return value


_IndexTensor = Annotated[torch.Tensor, AfterValidator(_index_tensor)]


@validate_call(config=HydroForgeModel.model_config)
def find_indices_in_torch(a: _IndexTensor, b: _IndexTensor) -> torch.Tensor:
    """Return int32 indices in unique *b* of each element of *a*, or ``-1``.

    Tensors of different integer dtypes are compared as int64.
    """

    if not devices_match(a.device, b.device):
        raise ValueError("torch index lookup tensors must share one device")
    if b.numel() > torch.iinfo(torch.int32).max:
        raise ValueError(
            f"b has {b.numel()} elements, exceeding int32 range. "
            "find_indices_in_torch returns int32 indices."
        )
    if b.numel() == 0:
        return torch.full_like(a, -1, dtype=torch.int32)
    # bucketize does not support uint16/uint32.
    if a.dtype != b.dtype or b.dtype in {torch.uint16, torch.uint32}:
        query = a.to(torch.int64)
        target = b.to(torch.int64)
    else:
        query = a
        target = b
    sorted_b, order = torch.sort(target)
    if bool((sorted_b[1:] == sorted_b[:-1]).any()):
        raise ValueError("torch index lookup target must contain unique values")
    pos = torch.bucketize(query, sorted_b, right=False)
    # bucketize on MPS/some GPU backends may return len(sorted_b) for values
    # that equal the last element; clamp to keep indexing safe — the equality
    # check below still rejects true misses.
    pos = pos.clamp(max=len(sorted_b) - 1)
    hit_mask = sorted_b[pos] == query
    index = torch.full_like(a, -1, dtype=torch.int32)
    index[hit_mask] = order[pos[hit_mask]].to(torch.int32)
    return index


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


_TORCH_NUMPY_DTYPES = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}


def torch_to_numpy_dtype(torch_dtype: torch.dtype) -> type:
    try:
        return _TORCH_NUMPY_DTYPES[torch_dtype]
    except KeyError:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}") from None

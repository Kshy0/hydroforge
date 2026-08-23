# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Lossless numeric canonicalization for data-space identities."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

from hydroforge.contracts.validation import _ImmutableDict


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
    owned = np.array(
        source,
        dtype=target_dtype,
        order=order,
        copy=True,
        subok=False,
    )
    storage_order: Literal["C", "F"] = (
        "F"
        if owned.flags.f_contiguous and not owned.flags.c_contiguous
        else "C"
    )
    payload = owned.tobytes(order=storage_order)
    return np.ndarray(
        owned.shape,
        dtype=owned.dtype,
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
            immutable_metadata(item, label=f"{label} item")
            for item in value
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
        raise ValueError(
            f"{label} is not exactly representable as float64"
        )
    return result


def positive_finite_float64(value: NumericScalar, *, label: str) -> float:
    """Return one positive float64 scalar without lossy coercion."""

    try:
        result = finite_float64(value, label=label)
    except ValueError as error:
        if "not exactly representable as float64" in str(error):
            raise
        raise ValueError(
            f"{label} must be a finite positive real number"
        ) from error
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
    """Materialize finite float data without lossy integer conversion."""

    if np.ma.isMaskedArray(value):
        raise ValueError(f"{label} must not be a masked array")
    source = np.asarray(value)
    if source.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must contain real numeric values")
    finite = np.isfinite(source)
    if allow_nan:
        if np.isinf(source).any():
            raise ValueError(f"{label} contains infinite values")
    elif not finite.all():
        raise ValueError(f"{label} contains non-finite values")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    target = np.dtype(dtype)
    if target == np.dtype(np.float32) and source.size:
        as_float64 = np.asarray(source[finite], dtype=np.float64)
        if (
            not np.isfinite(as_float64).all()
            or np.any(np.abs(as_float64) > np.finfo(np.float32).max)
        ):
            raise ValueError(
                f"{label} contains values outside float32 range"
            )
    result = np.asarray(source, dtype=target)
    if np.isinf(result).any() or (not allow_nan and np.isnan(result).any()):
        raise ValueError(f"{label} overflowed {dtype}")
    if np.any(finite & (source != 0) & (result == 0)):
        raise ValueError(
            f"{label} contains nonzero values that underflow in {dtype}"
        )
    if source.dtype.kind in {"i", "u"} and not np.array_equal(
        source.astype(object), result.astype(object),
    ):
        raise ValueError(
            f"{label} contains integers that are not exactly representable "
            f"as {dtype}"
        )
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
                f"{label} contains values that cannot be represented exactly "
                "as float64"
            )
    return result


def exact_numeric_array_equal(
    left: NumericValue, right: NumericValue,
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
    if left_array.dtype == right_array.dtype:
        return bool(np.array_equal(left_array, right_array))
    if left_array.dtype.kind == right_array.dtype.kind:
        return bool(np.array_equal(left_array, right_array))
    # In particular, int64/uint64 mixed with float64 can otherwise promote to
    # float64 and make adjacent large integers compare equal.  Object arrays
    # use Python's exact integer/float comparison rules.
    return bool(np.array_equal(
        left_array.astype(object), right_array.astype(object),
    ))

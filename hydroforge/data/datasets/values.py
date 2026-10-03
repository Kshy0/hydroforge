"""The single read-side value pipeline of forcing datasets.

Every storage read passes ``ingest`` once: masked and NaN values follow the
dataset's missing-value policy, infinities are always rejected, and negative
values are clipped on request.  ``convert`` then aggregates, applies a unit
factor in float64 and narrows to the output dtype, range-checking only where a
step can leave the source range.  Values reaching a model were checked here,
so sharding on the device needs no second pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np

from hydroforge.core.arrays import canonical_floating_array

MissingPolicy = Literal["zero", "error"]
AggregationMethod = Literal["mean", "max", "min", "sum"]


def ingest(
    raw: np.ndarray,
    *,
    rows: int,
    missing: MissingPolicy,
    clip_negative: bool,
    label: str,
) -> np.ndarray:
    """Apply the missing-value policy and the non-finite check to one read.

    ``raw`` is never modified in place: extension readers may return views of
    storage they keep.
    """

    if np.ma.isMaskedArray(raw):
        mask = np.ma.getmaskarray(raw)
        if mask.any():
            if missing == "error":
                raise ValueError(f"{label} contains missing or non-finite values")
            # Integer storage has no NaN; masked cells become float64 zeros.
            raw = (raw if raw.dtype.kind == "f" else raw.astype(np.float64)).filled(0)
        else:
            raw = raw.data
    values = np.asarray(raw)
    if values.ndim < 1 or values.shape[0] != rows:
        raise ValueError(f"{label} must have {rows} rows on its time axis")
    if values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must contain real numeric values")
    if values.dtype.kind == "f" and values.size:
        # NaN propagates through min/max and an infinity becomes an extreme,
        # so two reductions detect both without allocating a mask.
        if not (np.isfinite(values.min()) and np.isfinite(values.max())):
            nan = np.isnan(values)
            if nan.any():
                if missing == "error":
                    raise ValueError(f"{label} contains missing or non-finite values")
                values = np.where(nan, values.dtype.type(0), values)
            if np.isinf(values).any():
                raise ValueError(f"{label} contains non-finite values")
    if clip_negative:
        values = np.maximum(values, 0)
    return values


def ingest_integer(raw: np.ndarray, *, rows: int, label: str) -> np.ndarray:
    """Own one declared integer output as int64 without float conversion."""

    if np.ma.isMaskedArray(raw):
        if np.ma.getmaskarray(raw).any():
            raise ValueError(f"{label} contains missing values")
        raw = raw.data
    values = np.asarray(raw)
    if values.ndim < 1 or values.shape[0] != rows:
        raise ValueError(f"{label} must have {rows} rows on its time axis")
    if values.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must contain integers")
    if (
        values.dtype.kind == "u"
        and values.size
        and int(values.max()) > np.iinfo(np.int64).max
    ):
        raise ValueError(f"{label} contains a value outside int64 range")
    return np.array(values, dtype=np.int64, order="C", copy=True)


def bounded(source: np.dtype, out_dtype: str) -> bool:
    """Whether every finite ``source`` value lies within the ``out_dtype`` range."""

    return source.kind in {"i", "u"} or source.itemsize <= np.dtype(out_dtype).itemsize


def finalize(
    values: np.ndarray,
    *,
    out_dtype: str,
    checked: bool,
    label: str,
) -> np.ndarray:
    """Cast finite values to ``out_dtype``; ``checked`` also verifies the range.

    Unchecked casts are only valid for values already known to be finite and
    representable, such as unscaled float storage no wider than the output.
    """

    if checked:
        return canonical_floating_array(values, dtype=out_dtype, label=label)
    return np.ascontiguousarray(values, dtype=out_dtype)


def direct_cast(source: np.dtype) -> bool:
    """Whether unscaled ``source`` values may be cast straight to the output.

    For floats and integers of at most 16 bits the direct cast equals the
    float64 route bit for bit.
    """

    return (source.kind == "f" and source.itemsize <= 8) or (
        source.kind in {"i", "u"} and source.itemsize <= 2
    )


def as_float64(values: np.ndarray, *, label: str) -> np.ndarray:
    """Promote finite values to a float64 calculation array.

    64-bit integers above 2**53 are rejected rather than rounded; the result
    may be ``values`` itself, which callers own.
    """

    if values.dtype.kind in {"i", "u"} and values.dtype.itemsize >= 8:
        return canonical_floating_array(values, dtype="float64", label=label)
    return np.ascontiguousarray(values, dtype=np.float64)


def _aggregate(
    values: np.ndarray, method: AggregationMethod, factor: int
) -> np.ndarray:
    if values.shape[0] % factor != 0:
        raise ValueError(
            f"Cannot aggregate {values.shape[0]} source frames into "
            f"windows of {factor} frames"
        )
    grouped = values.reshape((values.shape[0] // factor, factor) + values.shape[1:])
    if method == "mean":
        return grouped.mean(axis=1, dtype=np.float64)
    if method == "max":
        return grouped.max(axis=1)
    if method == "min":
        return grouped.min(axis=1)
    return grouped.sum(axis=1, dtype=np.float64)


def convert(
    values: np.ndarray,
    *,
    out_dtype: str,
    unit_factor: float = 1.0,
    aggregation: AggregationMethod | Mapping[str, AggregationMethod] | None = None,
    factor: int = 1,
    label: str,
) -> np.ndarray | dict[str, np.ndarray]:
    """Aggregate ``factor`` source rows per output row, divide, and narrow.

    ``label`` names the storage kind in errors (``"NetCDF dataset"``).
    Unscaled, unaggregated values that allow a :func:`direct_cast` skip the
    float64 calculation copy.
    """

    if aggregation is None and unit_factor == 1.0 and direct_cast(values.dtype):
        return finalize(
            values,
            out_dtype=out_dtype,
            checked=not bounded(values.dtype, out_dtype),
            label=f"{label} output",
        )
    calculation = as_float64(values, label=f"{label} input")
    if aggregation is None:
        converted: np.ndarray | dict[str, np.ndarray] = calculation
    elif isinstance(aggregation, str):
        converted = _aggregate(calculation, aggregation, factor)
    else:
        converted = {
            name: _aggregate(calculation, method, factor)
            for name, method in aggregation.items()
        }
    blocks = converted.values() if isinstance(converted, dict) else (converted,)
    if unit_factor != 1.0:
        for block in blocks:
            np.divide(block, unit_factor, out=block)
    reduces = aggregation in ("sum", "mean") or (
        isinstance(aggregation, Mapping)
        and any(method in {"sum", "mean"} for method in aggregation.values())
    )
    checked = not bounded(values.dtype, out_dtype) or unit_factor < 1.0 or reduces
    if isinstance(converted, dict):
        return {
            name: finalize(
                block,
                out_dtype=out_dtype,
                checked=checked,
                label=f"{label} output variable {name!r}",
            )
            for name, block in converted.items()
        }
    return finalize(
        converted, out_dtype=out_dtype, checked=checked, label=f"{label} output"
    )

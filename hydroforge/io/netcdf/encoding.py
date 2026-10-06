# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Value encoding for stored outputs: logical dtypes and checked narrowing."""

from typing import Any

import numpy as np
import torch

LOGICAL_DTYPE_ATTR = "hydroforge_dtype"
COMPLETE_DATA_ATTR = "hydroforge_complete_data"
BOOL_LOGICAL_DTYPE = "bool"
BOOL_NETCDF_STORAGE_DTYPE = np.dtype("u1")
BOOL_NETCDF_READ_DTYPES = frozenset({np.dtype("i1"), np.dtype("u1")})


def read_netcdf_values(variable: Any, selection: Any = Ellipsis) -> Any:
    """Read complete framework data without treating valid sentinels as missing.

    The marker declares every published element to be data; rank readers
    still restrict reads to committed rows. Unmarked external variables keep
    NetCDF's missing-value semantics. Automatic scale/offset decoding stays on.
    """

    if getattr(variable, COMPLETE_DATA_ATTR, None) == "true":
        variable.set_auto_mask(False)
    return variable[selection]


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


def saved_dtype(dtype: torch.dtype, save_precision: torch.dtype | None) -> torch.dtype:
    """Return the dtype an output of ``dtype`` is saved in.

    ``save_precision`` applies to floating-point outputs only.
    """

    if save_precision is not None and dtype.is_floating_point:
        return save_precision
    return dtype


def decoded_output_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detach ordinary storage, or snapshot encoded values as CPU float64.

    An encoded tensor's int64 carrier is never an integer output. Decode at
    the export boundary before narrowing, NumPy conversion or copying into
    ordinary host storage. ``copy=True`` is required even for a CPU-backed
    encoded tensor: a no-op ``cpu()`` would leave the wrapper undecoded.
    """

    from hydroforge.kernels.emulated import EmulatedTensor

    source = tensor.detach()
    if isinstance(source, EmulatedTensor):
        return source.to(device="cpu", copy=True)
    return source


def _narrowing_limit(source: torch.dtype, target: torch.dtype) -> float | None:
    """Return the finite magnitude bound of a lossy float narrowing, if any."""

    if not (source.is_floating_point and target.is_floating_point):
        raise TypeError(
            f"unsupported statistics output conversion from {source} to {target}"
        )
    limit = torch.finfo(target).max
    return limit if limit < torch.finfo(source).max else None


def narrowing_flag(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    name: str,
    converted: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str, str] | None:
    """Return the device flag of finite values a lossy narrowing overflows.

    ``None`` when ``target_dtype`` holds every value of the tensor's dtype.
    Encoded values are checked after decoding, so their flag is on the CPU.
    ``converted``, the tensor already narrowed to ``target_dtype``, lets the
    check count new infinities instead of materializing a magnitude copy.
    """

    source = tensor.detach()
    if source.dtype == target_dtype:
        return None
    limit = _narrowing_limit(source.dtype, target_dtype)
    if limit is None:
        return None
    source = decoded_output_tensor(source)
    if converted is not None:
        # Narrowing maps exactly the finite values beyond the target range
        # (after rounding) to infinity and keeps every other infinity.
        outside = torch.isinf(converted).sum() > torch.isinf(source).sum()
    elif source.numel():
        magnitude = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
        outside = magnitude.abs_().amax() > limit
    else:
        outside = torch.zeros((), dtype=torch.bool, device=source.device)
    return outside, name, str(target_dtype).removeprefix("torch.")


def checked_narrowing(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    name: str,
    flags: list[tuple[torch.Tensor, str, str]] | None = None,
) -> torch.Tensor:
    """Convert logical values, rejecting finite values out of range.

    Tiny values may round to subnormals or zero.  With ``flags`` the
    device-side overflow flag is queued for a deferred host check; otherwise
    it is checked before returning. Encoded Metal values are snapshotted and
    decoded on the CPU first; ordinary tensors stay on their device.
    """

    source = decoded_output_tensor(tensor)
    if source.dtype == target_dtype:
        return source
    converted = source.to(dtype=target_dtype)
    entry = narrowing_flag(source, target_dtype, name=name, converted=converted)
    if entry is not None:
        if flags is None:
            raise_narrowing_failures((entry,))
        else:
            flags.append(entry)
    return converted


def raise_narrowing_failures(
    flags: tuple[tuple[torch.Tensor, str, str], ...] | list,
    observed: list[bool] | None = None,
) -> None:
    """Raise the first failed narrowing check; ``observed`` holds host values.

    Without ``observed`` the flags are read back in one device transfer.
    """

    if not flags:
        return
    if observed is None:
        device = flags[0][0].device
        observed = (
            torch.stack([failure.to(device=device) for failure, _name, _label in flags])
            .cpu()
            .tolist()
        )
    for outside, (_failure, name, label) in zip(observed, flags, strict=True):
        if outside:
            raise OverflowError(
                f"statistics output {name!r} contains values outside {label} range"
            )

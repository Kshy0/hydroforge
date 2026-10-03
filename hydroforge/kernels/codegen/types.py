"""The one table of native spellings of tensor elements and scalar kinds.

Tensor elements and canonical scalar kinds each have one :class:`NativeType`:
its spelling in CUDA C++, MSL and Triton annotations (``None`` where the
dialect cannot represent it), its host ``ctypes`` representation and its
Itanium-mangled builtin codes, which recover a compiled kernel's declared
parameter types.
"""

from __future__ import annotations

import ctypes
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch

Dialect = Literal["cuda", "msl", "triton"]


@dataclass(frozen=True, slots=True)
class NativeType:
    dtype: torch.dtype
    cuda: str | None
    msl: str | None
    triton: str | None
    ctype: type | None
    mangled: tuple[str, ...]


def _native(dtype, cuda, msl, triton, ctype, *mangled) -> NativeType:
    return NativeType(dtype, cuda, msl, triton, ctype, mangled)


# Bool tensors are byte arrays in MSL; the other rows spell the same type.
ELEMENTS: Mapping[torch.dtype, NativeType] = MappingProxyType(
    {
        native.dtype: native
        for native in (
            _native(torch.float32, "float", "float", "fp32", ctypes.c_float, "f"),
            _native(torch.float64, "double", None, "fp64", ctypes.c_double, "d"),
            _native(torch.float16, None, None, None, None, "Dh"),
            _native(torch.bool, "bool", "uchar", None, ctypes.c_bool, "b"),
            _native(torch.int8, "int8_t", None, None, ctypes.c_int8, "a", "c"),
            _native(torch.uint8, "uint8_t", None, None, ctypes.c_uint8, "h"),
            _native(torch.int16, None, None, None, ctypes.c_int16, "s"),
            _native(torch.uint16, None, None, None, ctypes.c_uint16, "t"),
            _native(torch.int32, "int32_t", "int", None, ctypes.c_int32, "i"),
            _native(torch.uint32, None, None, None, ctypes.c_uint32, "j"),
            _native(torch.int64, "int64_t", "long", None, ctypes.c_int64, "l", "x"),
            _native(torch.uint64, "uint64_t", None, None, ctypes.c_uint64, "m", "y"),
        )
    }
)

SCALARS: Mapping[str, NativeType] = MappingProxyType(
    {
        "bool": _native(torch.bool, "bool", "bool", None, ctypes.c_bool, "b"),
        "int32": ELEMENTS[torch.int32],
        "uint32": _native(torch.uint32, "uint32_t", "uint", None, ctypes.c_uint32, "j"),
        "index": ELEMENTS[torch.int64],
        "float32": ELEMENTS[torch.float32],
        "float64": ELEMENTS[torch.float64],
    }
)

MANGLED_ELEMENTS: Mapping[str, torch.dtype] = MappingProxyType(
    {code: native.dtype for native in ELEMENTS.values() for code in native.mangled}
)


_SCALAR_KINDS: Mapping[torch.dtype, str] = MappingProxyType(
    {native.dtype: kind for kind, native in SCALARS.items()}
)


def scalar_kind(dtype: torch.dtype) -> str:
    """The canonical scalar kind of a by-value ``dtype`` (int64 is ``index``)."""

    kind = _SCALAR_KINDS.get(dtype)
    if kind is None:
        raise TypeError(f"no scalar kind for {dtype}")
    return kind


def element(dtype: torch.dtype, dialect: Dialect) -> str:
    """Spelling of a tensor element type, or ``TypeError`` if absent."""

    native = ELEMENTS.get(dtype)
    spelling = None if native is None else getattr(native, dialect)
    if spelling is None:
        raise TypeError(f"no {dialect} element type for {dtype}")
    return spelling


def scalar(kind: str, dialect: Dialect) -> str:
    """Spelling of a canonical scalar kind, or ``TypeError`` if absent."""

    native = SCALARS.get(kind)
    spelling = None if native is None else getattr(native, dialect)
    if spelling is None:
        raise TypeError(f"no {dialect} scalar type for kind {kind!r}")
    return spelling

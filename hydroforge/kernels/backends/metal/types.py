"""Canonical PyTorch and KernelSpec scalar representations in Metal ABI."""

from __future__ import annotations

from types import MappingProxyType

import torch


TENSOR_TYPES = MappingProxyType({
    torch.bool: "uchar",
    torch.float32: "float",
    torch.int32: "int",
    torch.int64: "long",
})
COMPILE_SCALAR_TYPES = MappingProxyType({
    "bool": "bool", "int32": "int", "float32": "float",
})
RUNTIME_SCALAR_TYPES = MappingProxyType({
    "bool": "bool", "int32": "int", "index": "long", "float32": "float",
})
NATIVE_BUFFER_DTYPES = MappingProxyType({
    "float": frozenset({torch.float32}),
    "atomic_float": frozenset({torch.float32}),
    "int": frozenset({torch.int32}),
    "atomic_int": frozenset({torch.int32}),
    "long": frozenset({torch.int64}),
    "uchar": frozenset({torch.bool}),
    "uint": frozenset({torch.int32}),
    # Some reductions use an atomic uint view of float32 bits.
    "atomic_uint": frozenset({torch.float32, torch.int32}),
})


def tensor_type(dtype: torch.dtype) -> str:
    """Return the one canonical MSL storage type for a PyTorch tensor."""

    try:
        return TENSOR_TYPES[dtype]
    except KeyError as error:
        raise TypeError(f"Metal does not support tensor dtype {dtype}") from error


__all__ = [
    "COMPILE_SCALAR_TYPES", "NATIVE_BUFFER_DTYPES", "RUNTIME_SCALAR_TYPES",
    "TENSOR_TYPES", "tensor_type",
]

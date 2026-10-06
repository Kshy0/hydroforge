# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Kernel argument layouts recorded in AMDGPU code objects.

HIP has no counterpart to ``cuFuncGetParamInfo``. The compiler instead records
each kernel's argument layout in the code object's ``NT_AMDGPU_METADATA`` ELF
note as a MessagePack document (``amdhsa.kernels[].args``), which this module
reads without third-party dependencies.
"""

from __future__ import annotations

import struct
from typing import Any

_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
_SHT_NOTE = 7
_NT_AMDGPU_METADATA = 32

ParameterTable = tuple[tuple[int, int], ...]


def kernel_parameters(image: bytes) -> dict[str, ParameterTable]:
    """Map each kernel's symbol to the ``(offset, size)`` of its explicit arguments.

    Compiler-appended ``hidden_*`` arguments are omitted. An image without the
    metadata note (or in an unknown container) yields an empty mapping.
    """

    elf = _code_object(image)
    if elf is None:
        return {}
    tables: dict[str, ParameterTable] = {}
    for metadata in _metadata_notes(elf):
        kernels = metadata.get("amdhsa.kernels", [])
        if not isinstance(kernels, list) or any(
            not isinstance(kernel, dict) for kernel in kernels
        ):
            raise ValueError("AMDGPU kernels metadata must be a list of maps")
        for kernel in kernels:
            args = kernel.get(".args", [])
            if not isinstance(args, list) or any(
                not isinstance(argument, dict) for argument in args
            ):
                raise ValueError("AMDGPU arguments metadata must be a list of maps")
            previous_end = 0
            for argument in args:
                offset, size = argument.get(".offset"), argument.get(".size")
                if (
                    type(offset) is not int
                    or type(size) is not int
                    or offset < previous_end
                    or size < 0
                    or offset + size >= 2**64
                ):
                    raise ValueError("AMDGPU argument layout is invalid or overlapping")
                previous_end = offset + size
            table = tuple(
                (argument[".offset"], argument[".size"])
                for argument in args
                if not str(argument.get(".value_kind", "")).startswith("hidden_")
            )
            for key in (kernel.get(".name"), kernel.get(".symbol")):
                if isinstance(key, str):
                    name = key.removesuffix(".kd")
                    if not name or name in tables and tables[name] != table:
                        raise ValueError("conflicting AMDGPU kernel symbols")
                    tables[name] = table
    return tables


def _code_object(image: bytes) -> bytes | None:
    """The AMDGPU ELF inside ``image``, unwrapping an uncompressed offload bundle."""

    if image[:4] == b"\x7fELF":
        return image
    if not image.startswith(_BUNDLE_MAGIC):
        return None
    position = len(_BUNDLE_MAGIC)
    _require_bytes(image, position, 8)
    (count,) = struct.unpack_from("<Q", image, position)
    position += 8
    if count > (len(image) - position) // 24:
        raise ValueError("truncated offload bundle table")
    for _ in range(count):
        _require_bytes(image, position, 24)
        offset, size, length = struct.unpack_from("<QQQ", image, position)
        position += 24
        _require_bytes(image, position, length)
        _require_bytes(image, offset, size)
        triple = image[position : position + length]
        position += length
        if b"amdgcn" in triple:
            return image[offset : offset + size]
    return None


def _metadata_notes(elf: bytes) -> list[dict[str, Any]]:
    _require_bytes(elf, 0, 64)
    if elf[4] != 2 or elf[5] != 1:  # ELFCLASS64, little endian
        return []
    (section_offset,) = struct.unpack_from("<Q", elf, 0x28)
    entry_size, count = struct.unpack_from("<HH", elf, 0x3A)
    if count and entry_size < 64:
        raise ValueError("invalid ELF section entry size")
    _require_bytes(elf, section_offset, entry_size * count)
    documents = []
    for index in range(count):
        header = section_offset + index * entry_size
        if struct.unpack_from("<I", elf, header + 4)[0] != _SHT_NOTE:
            continue
        start, size = struct.unpack_from("<QQ", elf, header + 0x18)
        _require_bytes(elf, start, size)
        position = start
        while position + 12 <= start + size:
            name_size, description_size, kind = struct.unpack_from(
                "<III", elf, position
            )
            position += 12
            if (
                position + ((name_size + 3) & ~3) + ((description_size + 3) & ~3)
                > start + size
            ):
                raise ValueError("truncated ELF note")
            name = elf[position : position + name_size].rstrip(b"\0")
            position += (name_size + 3) & ~3
            description = elf[position : position + description_size]
            position += (description_size + 3) & ~3
            if name == b"AMDGPU" and kind == _NT_AMDGPU_METADATA:
                document, stop = _unpack(description, 0)
                if stop != len(description):
                    raise ValueError("trailing MessagePack metadata")
                if isinstance(document, dict):
                    documents.append(document)
    return documents


# MessagePack subset used by AMDGPU metadata: nil, booleans, integers, floats,
# strings, binaries, arrays and maps (no extension types).
_SIZED = {
    0xC4: ("B", "bin"), 0xC5: (">H", "bin"), 0xC6: (">I", "bin"),
    0xCA: (">f", "value"), 0xCB: (">d", "value"),
    0xCC: ("B", "value"), 0xCD: (">H", "value"),
    0xCE: (">I", "value"), 0xCF: (">Q", "value"),
    0xD0: ("b", "value"), 0xD1: (">h", "value"),
    0xD2: (">i", "value"), 0xD3: (">q", "value"),
    0xD9: ("B", "str"), 0xDA: (">H", "str"), 0xDB: (">I", "str"),
    0xDC: (">H", "array"), 0xDD: (">I", "array"),
    0xDE: (">H", "map"), 0xDF: (">I", "map"),
}  # fmt: skip
_CONSTANTS = {0xC0: None, 0xC2: False, 0xC3: True}


def _require_bytes(data: bytes, position: int, size: int) -> None:
    if position < 0 or size < 0 or position > len(data) or size > len(data) - position:
        raise ValueError("truncated binary metadata")


def _unpack(data: bytes, position: int, depth: int = 0) -> tuple[Any, int]:
    if depth > 64:
        raise ValueError("MessagePack nesting exceeds 64 levels")
    _require_bytes(data, position, 1)
    code = data[position]
    position += 1
    if code <= 0x7F:
        return code, position
    if code >= 0xE0:
        return code - 0x100, position
    if code <= 0x8F:
        return _container(data, position, code & 0x0F, "map", depth)
    if code <= 0x9F:
        return _container(data, position, code & 0x0F, "array", depth)
    if code <= 0xBF:
        return _bytes(data, position, code & 0x1F, "str")
    if code in _CONSTANTS:
        return _CONSTANTS[code], position
    if code not in _SIZED:
        raise ValueError(f"unsupported MessagePack type 0x{code:02x}")
    layout, kind = _SIZED[code]
    _require_bytes(data, position, struct.calcsize(layout))
    (value,) = struct.unpack_from(layout, data, position)
    position += struct.calcsize(layout)
    if kind == "value":
        return value, position
    if kind in ("bin", "str"):
        return _bytes(data, position, value, kind)
    return _container(data, position, value, kind, depth)


def _bytes(data: bytes, position: int, length: int, kind: str) -> tuple[Any, int]:
    _require_bytes(data, position, length)
    raw = data[position : position + length]
    return (raw.decode() if kind == "str" else bytes(raw)), position + length


def _container(
    data: bytes, position: int, count: int, kind: str, depth: int = 0
) -> tuple[Any, int]:
    if count > (len(data) - position) // (2 if kind == "map" else 1):
        raise ValueError("MessagePack container exceeds available bytes")
    if kind == "array":
        items = []
        for _ in range(count):
            item, position = _unpack(data, position, depth + 1)
            items.append(item)
        return items, position
    mapping = {}
    for _ in range(count):
        key, position = _unpack(data, position, depth + 1)
        value, position = _unpack(data, position, depth + 1)
        if (
            type(key) not in {str, int, bool, float, bytes, type(None)}
            or key in mapping
        ):
            raise ValueError("MessagePack map key is invalid or duplicated")
        mapping[key] = value
    return mapping, position

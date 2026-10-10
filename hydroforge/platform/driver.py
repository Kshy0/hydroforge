# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The device driver library matching PyTorch's build, loaded once."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence
from functools import cache

import torch

HIP_DRIVER_LIBRARIES = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")
CUDA_DRIVER_LIBRARY = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"


def load_first(names: Sequence[str], *, kind: str) -> ctypes.CDLL:
    """Load the first of ``names`` that exists, reporting every failure."""

    errors = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            errors.append(f"{name}: {error}")
    raise OSError(f"{kind} library unavailable: " + "; ".join(errors))


@cache
def driver_library() -> ctypes.CDLL:
    """The HIP or CUDA driver library of the installed PyTorch."""

    if torch.version.hip is not None:
        return load_first(HIP_DRIVER_LIBRARIES, kind="device runtime")
    return ctypes.CDLL(CUDA_DRIVER_LIBRARY)


@cache
def _error_string() -> ctypes._CFuncPtr:
    library = driver_library()
    if torch.version.hip is not None:
        function = library.hipGetErrorString
        function.argtypes, function.restype = (ctypes.c_int,), ctypes.c_char_p
    else:
        function = library.cuGetErrorString
        function.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
        function.restype = ctypes.c_int
    return function


def driver_error(result: int) -> str:
    """The driver's message for a failed status ``result``."""

    function = _error_string()
    if torch.version.hip is not None:
        text = function(result)
    else:
        pointer = ctypes.c_char_p()
        function(result, ctypes.byref(pointer))
        text = pointer.value
    return text.decode() if text else f"error {result}"


def bind_primary_context(device: torch.device | int) -> None:
    """Make the device's primary context current on this thread.

    Driver calls need it; any runtime call binds it, and a stream query does
    so without waiting for the device's queued work.  Call inside
    ``torch.cuda.device(device)``.
    """

    torch.cuda.current_stream(device).query()

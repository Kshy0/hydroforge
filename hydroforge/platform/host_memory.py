"""Page-locking of existing host memory through the device driver.

Registered memory is a valid target of asynchronous device-to-host copies.
The driver entry points (``cuMemHostRegister``; HIP clears its last error
after a failure) leave no pending runtime error behind, so a failed
registration cannot resurface as the error of a later, unrelated kernel.
"""

from __future__ import annotations

import ctypes
import sys
from functools import cache

import torch


class _HostRegistry:
    """ctypes view of the driver calls that page-lock host memory."""

    def __init__(self) -> None:
        self.hip = torch.version.hip is not None
        if self.hip:
            library = _load_first(
                ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")
            )
            register, unregister = library.hipHostRegister, library.hipHostUnregister
            self._message = library.hipGetErrorString
            self._message.argtypes, self._message.restype = (
                (ctypes.c_int,),
                ctypes.c_char_p,
            )
            self._clear = library.hipGetLastError
            self._clear.argtypes, self._clear.restype = (), ctypes.c_int
        else:
            library = ctypes.CDLL(
                "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
            )
            register, unregister = (
                library.cuMemHostRegister_v2,
                library.cuMemHostUnregister,
            )
            self._message = library.cuGetErrorString
            self._message.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
            self._message.restype = ctypes.c_int
            self._clear = None
        register.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
        unregister.argtypes = (ctypes.c_void_p,)
        register.restype = unregister.restype = ctypes.c_int
        self.register, self.unregister = register, unregister

    def failure(self, result: int) -> str | None:
        """Return the message of a failed call, clearing HIP's last error."""

        if result == 0:
            return None
        if self._clear is not None:
            self._clear()
            text = self._message(result)
        else:
            pointer = ctypes.c_char_p()
            self._message(result, ctypes.byref(pointer))
            text = pointer.value
        return text.decode() if text else f"error {result}"


def _load_first(names: tuple[str, ...]) -> ctypes.CDLL:
    errors = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            errors.append(f"{name}: {error}")
    raise OSError("device runtime library unavailable: " + "; ".join(errors))


@cache
def _registry() -> _HostRegistry:
    return _HostRegistry()


def register_host_memory(address: int, nbytes: int, device: torch.device) -> str | None:
    """Page-lock ``nbytes`` at ``address`` for ``device``; return why it failed.

    ``None`` means the memory is now pinned for the device's primary context.
    """

    try:
        registry = _registry()
    except (AttributeError, OSError) as error:
        return str(error)
    with torch.cuda.device(device):
        # The driver call needs the device's primary context to be current.
        torch.cuda.synchronize()
        return registry.failure(registry.register(address, nbytes, 0))


def unregister_host_memory(address: int, device: torch.device) -> None:
    """Make memory registered by :func:`register_host_memory` pageable again."""

    registry = _registry()
    with torch.cuda.device(device):
        failure = registry.failure(registry.unregister(address))
    if failure is not None:
        raise RuntimeError(f"cannot unregister output ring host memory: {failure}")

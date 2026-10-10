# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Page-locking of existing host memory through the device driver.

Registered memory is a valid target of asynchronous device-to-host copies.
The driver entry points (``cuMemHostRegister``; HIP clears its last error
after a failure) leave no pending runtime error behind, so a failed
registration cannot resurface as the error of a later, unrelated kernel.
"""

from __future__ import annotations

import ctypes
from functools import cache

import torch

from hydroforge.platform.driver import (
    bind_primary_context,
    driver_error,
    driver_library,
)


class _HostRegistry:
    """ctypes view of the driver calls that page-lock host memory."""

    def __init__(self) -> None:
        library = driver_library()
        if torch.version.hip is not None:
            register, unregister = library.hipHostRegister, library.hipHostUnregister
            self._clear = library.hipGetLastError
            self._clear.argtypes, self._clear.restype = (), ctypes.c_int
        else:
            register, unregister = (
                library.cuMemHostRegister_v2,
                library.cuMemHostUnregister,
            )
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
        return driver_error(result)


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
        bind_primary_context(device)
        return registry.failure(registry.register(address, nbytes, 0))


def unregister_host_memory(address: int, device: torch.device) -> None:
    """Make memory registered by :func:`register_host_memory` pageable again."""

    registry = _registry()
    with torch.cuda.device(device):
        failure = registry.failure(registry.unregister(address))
    if failure is not None:
        raise RuntimeError(f"cannot unregister output ring host memory: {failure}")

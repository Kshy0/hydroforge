# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""HydroForge environment variables and their one parser.

Every ``HYDROFORGE_*`` variable is named here and read through one of the
typed parsers below.  An unset or empty variable takes its default; any other
unrecognized value is an error.

``HYDROFORGE_BACKEND`` (torch, triton, cuda or metal)
    Kernel backend; by default the model device selects it.
``HYDROFORGE_FAST_MATH`` (flag, off)
    Physics kernels use each backend's fast math.
``HYDROFORGE_PRECOMPILE_JOBS`` (positive int)
    Concurrent compilations; 6 on local rank 0 and 1 elsewhere by default.
``HYDROFORGE_CUDA_REBUILD`` (flag, off)
    Ignore the on-disk cache of runtime-compiled CUDA programs.
``HYDROFORGE_COMPILE_LOCK_TIMEOUT_SECONDS`` (non-negative float, 1800)
    Wait for a cross-process compile lock; 0 waits forever.
``HYDROFORGE_COMPILE_LOCK_STALE_SECONDS`` (non-negative float, 1800)
    Age after which a lock is stale; 0 never treats a lock as stale.
``HYDROFORGE_COMPILE_LOCK_POLL_SECONDS`` (non-negative float, 0.25)
    Lock polling interval, at least 0.05.
``HYDROFORGE_COMPILE_LOCK_DEAD_PID_GRACE_SECONDS`` (non-negative float, 2)
    Age before the lock of a dead holder on this host is removed.
``HYDROFORGE_COMPILE_LOCK_STEAL`` (flag, off)
    Remove stale locks whose holder may still be alive.

Launcher variables (``LOCAL_RANK``, ``WORLD_SIZE``, ...) belong to
:mod:`hydroforge.parallel.launch`.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal

BACKEND = "HYDROFORGE_BACKEND"
FAST_MATH = "HYDROFORGE_FAST_MATH"
PRECOMPILE_JOBS = "HYDROFORGE_PRECOMPILE_JOBS"
CUDA_REBUILD = "HYDROFORGE_CUDA_REBUILD"
COMPILE_LOCK_TIMEOUT = "HYDROFORGE_COMPILE_LOCK_TIMEOUT_SECONDS"
COMPILE_LOCK_STALE = "HYDROFORGE_COMPILE_LOCK_STALE_SECONDS"
COMPILE_LOCK_POLL = "HYDROFORGE_COMPILE_LOCK_POLL_SECONDS"
COMPILE_LOCK_DEAD_PID_GRACE = "HYDROFORGE_COMPILE_LOCK_DEAD_PID_GRACE_SECONDS"
COMPILE_LOCK_STEAL = "HYDROFORGE_COMPILE_LOCK_STEAL"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _value(name: str) -> str | None:
    """Return the stripped value, or ``None`` when unset or empty."""

    value = os.environ.get(name, "").strip()
    return value or None


def flag(name: str, *, default: bool) -> bool:
    """Read a boolean flag: 1/true/yes/on or 0/false/no/off."""

    value = _value(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def positive_int(name: str) -> int | None:
    """Read a positive integer, or ``None`` when unset."""

    value = _value(name)
    if value is None:
        return None
    try:
        result = int(value)
    except ValueError:
        result = 0
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def nonnegative_float(name: str, *, default: float) -> float:
    """Read a finite non-negative float."""

    value = _value(name)
    if value is None:
        return default
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a floating-point number, got {value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    if result == 0.0 and not Decimal(value.lower().partition("e")[0]).is_zero():
        raise ValueError(f"{name} contains a nonzero value that underflows float64")
    return result


def choice(name: str, choices: frozenset[str]) -> str | None:
    """Read one case-insensitive choice, or ``None`` when unset."""

    value = _value(name)
    if value is None:
        return None
    lowered = value.lower()
    if lowered not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {lowered!r}")
    return lowered

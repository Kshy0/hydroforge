# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Triton side of a device error status (``hydroforge.model.ErrorField``)."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def report_error(status_ptr, failed, code, info=None):
    """Report the lanes where ``failed`` holds to an ``ErrorField``.

    Every failing lane counts itself in ``status[0]``; the one that finds the
    count zero is the first failure and stores ``code`` and its own ``info``
    value (one int32, e.g. the lane's cell index) after it. Lanes that do not
    fail touch nothing, so the call costs nothing on the healthy path.
    """

    zero = tl.zeros_like(failed.to(tl.int32))
    previous = tl.atomic_add(
        status_ptr + zero, 1 + zero, mask=failed, sem="relaxed", scope="gpu"
    )
    first = failed & (previous == 0)
    tl.store(status_ptr + 1 + zero, code + zero, mask=first)
    if info is not None:
        tl.store(status_ptr + 2 + zero, info.to(tl.int32) + zero, mask=first)


__all__ = ["report_error"]

"""Concurrent runtime compilation of the kernel variants a program uses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from hydroforge.kernels.backends.cuda import rtc
from hydroforge.kernels.backends.precompile import precompile_jobs


def precompile_cuda_requests(
    requests: Iterable[tuple[rtc.RtcRequest, int]],
    *,
    default_jobs: int = 6,
) -> None:
    """Compile the exact kernel variants of an operator program.

    Local rank 0 compiles concurrently; other local ranks compile serially and
    share binaries through the cross-process cache lock.
    """
    by_device: dict[int, list[rtc.RtcRequest]] = defaultdict(list)
    for request, device in dict.fromkeys(requests):
        by_device[device].append(request)
    for device, pending in by_device.items():
        rtc.compile_requests(
            pending, device, jobs=precompile_jobs(len(pending), default_jobs)
        )


__all__ = ["precompile_cuda_requests"]

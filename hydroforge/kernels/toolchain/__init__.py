# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Compilation of kernel programs: requests, caches and batched precompilation.

Every program HydroForge compiles (physics kernels, statistics programs,
step-field clocks and loop control kernels) can be described before its first
launch as a :class:`CompileRequest`.  :func:`precompile` builds a batch of them
concurrently within each toolchain, so a model's first recording compiles all
of its programs at once instead of one at a time on first launch.

Toolchains cache in layers: an in-process memory cache keyed by content, the
on-disk cache of runtime-compiled CUDA (Triton keeps its own disk cache), and a
cross-process lock around each compilation (:mod:`.cache`).

Lifetime contract
-----------------
Compiled artifacts -- runtime-compiled binaries with their loaded modules and
functions, Triton compiled kernels, Metal libraries and pipelines -- are
process-wide caches keyed by content.  Models share them, closing a model does
not release them, and they grow with the number of distinct programs a
process compiles (for example new statistics topologies); nothing evicts them.

Bound artifacts belong to the runtime object that created them and are
released when it closes, is invalidated or rolls back a failure:

- prepared launches (packed arguments, workspaces): the operator program,
  the kernel binder's eager launch cache, the statistics bindings;
- Metal argument bindings: their launch, command sequence or ICB, with a
  finalizer as the fallback;
- CUDA graphs, conditional WHILE graphs and ICBs: the loop executor runners;
- generated Python modules: the statistics runtime and compiled step fields.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from hydroforge.parallel.launch import get_local_process_rank
from hydroforge.platform import env
from hydroforge.platform.backend import Toolchain


@dataclass(frozen=True, slots=True)
class CompileRequest:
    """One program to compile before its first launch.

    ``key`` identifies the compiled content, ``cost`` orders large programs
    first, and ``build`` compiles into the toolchain's cache; it is
    thread-safe and returns at once when the program is already cached.
    """

    toolchain: Toolchain
    key: str
    cost: int
    build: Callable[[], object]


def precompile_jobs(count: int) -> int:
    """Concurrent compilations for ``count`` programs on this process.

    Local rank 0 compiles concurrently by default; other local ranks compile
    serially and share binaries through the cross-process cache lock.
    """

    jobs = env.positive_int(env.PRECOMPILE_JOBS)
    if jobs is None:
        jobs = 6 if get_local_process_rank() == 0 else 1
    return min(jobs, count)


def precompile(requests: Iterable[CompileRequest], *, jobs: int | None = None) -> None:
    """Compile distinct requests, concurrently within each toolchain.

    The runtime compiler releases the GIL, so CUDA programs build on a thread
    pool while Triton compiles under its asynchronous compile mode on another
    (serially when the mode is unavailable); Metal and generated Python build
    serially. Toolchain groups finish before errors are reported; Triton
    warmup scheduling stops if a request cannot be submitted.
    """

    if jobs is not None and (type(jobs) is not int or jobs < 1):
        raise ValueError("precompile jobs must be a positive exact integer or None")
    groups: dict[str, list[CompileRequest]] = defaultdict(list)
    for request in {request.key: request for request in requests}.values():
        groups[request.toolchain].append(request)
    if not groups:
        return
    errors: list[BaseException] = []
    rtc = sorted(groups.pop("rtc", ()), key=lambda request: request.cost, reverse=True)
    rtc_jobs = precompile_jobs(len(rtc)) if jobs is None else min(jobs, len(rtc))
    pool = ThreadPoolExecutor(max_workers=rtc_jobs) if rtc_jobs > 1 else None
    futures: list[Future] = []
    try:
        if pool is None:
            _build_all(rtc, errors)
        else:
            futures = [pool.submit(request.build) for request in rtc]
        triton = groups.pop("triton", ())
        if triton:
            try:
                _compile_triton(triton, jobs)
            except Exception as error:
                errors.append(error)
        for remaining in groups.values():
            _build_all(remaining, errors)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    for future in futures:
        if (error := future.exception()) is not None:
            errors.append(error)
    if errors:
        # Report every failed program, raising the first one's own type.
        for other in errors[1:]:
            errors[0].add_note(f"another program also failed: {other!r}")
        raise errors[0]


def _build_all(requests: Iterable[CompileRequest], errors: list[BaseException]) -> None:
    for request in requests:
        try:
            request.build()
        except Exception as error:
            errors.append(error)


def _compile_triton(requests: Sequence[CompileRequest], jobs: int | None) -> None:
    """Warm up Triton specializations, compiling distinct kernels concurrently.

    Each build calls ``kernel.warmup`` with the exact launch arguments, so the
    compiled variants are the ones the launches use.
    """

    count = precompile_jobs(len(requests)) if jobs is None else min(jobs, len(requests))
    if count > 1:
        try:
            from triton.runtime._async_compile import AsyncCompileMode
        except ImportError:
            count = 1
    if count == 1:
        for request in requests:
            request.build()
        return
    with ThreadPoolExecutor(max_workers=count) as pool, AsyncCompileMode(pool):
        for request in requests:
            request.build()


def compile_calls(
    calls: Sequence[Any], pending: Iterable[CompileRequest] = ()
) -> list[Callable[[], Any]]:
    """Compile deferred kernel calls with ``pending`` programs in one batch.

    Each :class:`~hydroforge.kernels.registry.KernelCall` provides
    ``requests()`` and ``compile()``; the returned launches follow the order
    of ``calls``.
    """

    precompile([*pending, *(request for call in calls for request in call.requests())])
    return [call.compile() for call in calls]

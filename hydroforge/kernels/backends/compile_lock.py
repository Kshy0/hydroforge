"""Cross-process locks for native compilation caches.

A lock file records ``host:pid:time``; waiters poll it, remove it when its local
holder has died, and optionally steal it once it is stale. Every create/remove
decision runs under an OS file lock on a sibling guard file.
"""

from __future__ import annotations

import math
import os
import socket
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from hydroforge.contracts.errors import cleanup_on_exit


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *, fallback: str | None = None) -> float:
    """Resolve a finite float without evaluating an unused fallback variable."""

    val = os.environ.get(name)
    if val is None and fallback is not None:
        name = fallback
        val = os.environ.get(name)
    if val is None:
        return default
    try:
        result = float(val)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a floating-point number, got {val!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {val!r}")
    if result == 0.0 and not Decimal(val.lower().partition("e")[0]).is_zero():
        raise ValueError(f"{name} contains a nonzero value that underflows float64")
    return result


def clear_abandoned_torch_lock(build_dir: Path, *, verbose: bool) -> None:
    """Remove a PyTorch extension build lock left by an interrupted compile.

    ``torch.utils.cpp_extension`` uses a lock file named ``lock`` in the build
    directory and waits indefinitely when it already exists.  The caller holds
    HydroForge's exclusive per-fingerprint compile lock for ``build_dir``, so
    no live compiler can own that file: any existing lock is abandoned.
    """
    lock_path = build_dir / "lock"
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    if verbose:
        print(
            f"[hydroforge] removed abandoned extension lock {lock_path}",
            file=sys.stderr,
        )


def _read_compile_lock_holder(lock_path: Path) -> tuple[str | None, int | None]:
    try:
        raw = lock_path.read_text(errors="replace").strip()
    except OSError:
        return None, None
    parts = raw.split(":", 2)
    if len(parts) < 2:
        return None, None
    try:
        pid = int(parts[1])
    except ValueError:
        pid = None
    return parts[0] or None, pid


def _process_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return True
    if os.name == "nt":
        # os.kill(pid, 0) terminates processes on Windows; query a handle instead.
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return ctypes.get_last_error() != 87  # invalid PID; deny/unknown stays live
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _compile_lock_guard(lock_path: Path):
    """Serialize lock-file create/remove decisions on one stable inode."""
    guard_path = lock_path.with_name(f"{lock_path.name}.guard")
    descriptor = os.open(str(guard_path), os.O_CREAT | os.O_RDWR, 0o644)
    cleanup = [lambda: os.close(descriptor)]
    with cleanup_on_exit("compile lock guard", cleanup):
        if os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            cleanup.insert(0, lambda: fcntl.flock(descriptor, fcntl.LOCK_UN))
        elif os.name == "nt":
            import errno
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            while True:
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
            cleanup.insert(0, lambda: msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1))
        yield


def _remove_abandoned_compile_lock_unlocked(
    lock_path: Path,
    *,
    stale_after: float,
    env_prefix: str,
    verbose: bool,
) -> bool:
    if stale_after <= 0.0:
        return False
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    holder_host, holder_pid = _read_compile_lock_holder(lock_path)
    local_host = socket.gethostname()
    dead_local_holder = holder_host == local_host and not _process_is_alive(holder_pid)
    if dead_local_holder:
        grace = max(
            0.0,
            _env_float(
                f"{env_prefix}_CUDA_COMPILE_LOCK_DEAD_PID_GRACE_SECONDS",
                2.0,
                fallback="HYDROFORGE_CUDA_COMPILE_LOCK_DEAD_PID_GRACE_SECONDS",
            ),
        )
        if age < min(stale_after, grace):
            return False
        reason = f"abandoned by local pid {holder_pid}"
    elif age >= stale_after and (
        _env_truthy(f"{env_prefix}_CUDA_COMPILE_LOCK_STEAL", False)
        or _env_truthy("HYDROFORGE_CUDA_COMPILE_LOCK_STEAL", False)
    ):
        reason = "explicit stale-lock steal enabled"
    else:
        return False

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    if verbose:
        print(
            f"[hydroforge] removed compile lock {lock_path} ({reason}, age {age:.0f}s)",
            file=sys.stderr,
        )
    return True


def acquire_compile_lock(
    lock_path: Path,
    *,
    env_prefix: str,
    verbose: bool,
    cache_probe: Callable[[], Any | None] | None = None,
) -> Any | None:
    stale_after = _env_float(
        f"{env_prefix}_CUDA_COMPILE_LOCK_STALE_SECONDS",
        1800.0,
        fallback="HYDROFORGE_CUDA_COMPILE_LOCK_STALE_SECONDS",
    )
    poll = max(0.05, _env_float("HYDROFORGE_CUDA_COMPILE_LOCK_POLL_SECONDS", 0.25))
    timeout = _env_float(
        f"{env_prefix}_CUDA_COMPILE_LOCK_TIMEOUT_SECONDS",
        1800.0,
        fallback="HYDROFORGE_CUDA_COMPILE_LOCK_TIMEOUT_SECONDS",
    )
    deadline = time.time() + timeout if timeout > 0.0 else None

    while True:
        if cache_probe is not None:
            cached = cache_probe()
            if cached is not None:
                return cached
        removed = False
        with _compile_lock_guard(lock_path):
            try:
                fd = os.open(
                    str(lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                removed = _remove_abandoned_compile_lock_unlocked(
                    lock_path,
                    stale_after=stale_after,
                    env_prefix=env_prefix,
                    verbose=verbose,
                )
            else:
                try:
                    with cleanup_on_exit(
                        "compile lock descriptor", (lambda: os.close(fd),)
                    ):
                        remaining = memoryview(
                            f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}".encode()
                        )
                        while remaining:
                            written = os.write(fd, remaining)
                            if written == 0:
                                raise OSError("compile lock write made no progress")
                            remaining = remaining[written:]
                except BaseException:
                    # Only the successful O_EXCL creator owns this path, and
                    # the guard still excludes other create/remove decisions.
                    with cleanup_on_exit(
                        "incomplete compile lock", (lock_path.unlink,)
                    ):
                        raise
                return None
        if removed:
            continue
        if lock_path.exists():
            if deadline is not None and time.time() > deadline:
                holder_host, holder_pid = _read_compile_lock_holder(lock_path)
                holder = (
                    f"{holder_host or 'unknown'}:{holder_pid}"
                    if holder_pid is not None
                    else (holder_host or "unknown")
                )
                raise TimeoutError(
                    f"Timed out waiting for compile lock {lock_path} "
                    f"held by {holder}. Remove the lock if the compiler process "
                    "has exited, or set HYDROFORGE_CUDA_COMPILE_LOCK_STEAL=1 "
                    "to override a stale lock."
                )
            time.sleep(poll)
            continue


def release_compile_lock(lock_path: Path) -> None:
    with _compile_lock_guard(lock_path):
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


__all__ = ["acquire_compile_lock", "clear_abandoned_torch_lock", "release_compile_lock"]

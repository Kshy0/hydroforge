"""Cross-process compile locks and the compiler's process environment.

A lock file records ``host:pid:time:token``; waiters poll it, remove it when its local
holder has died, and optionally steal it once it is stale. Every create/remove
decision runs under an OS file lock on a sibling guard file.  The lock timing
is configured by the ``HYDROFORGE_COMPILE_LOCK_*`` variables of
:mod:`hydroforge.platform.env`; the Metal extension build and the runtime
compiled CUDA cache share them.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.platform import env

_compiler_environment_lock = RLock()


@contextmanager
def serialized_compilation():
    """Keep compiler resolution, cache identity and environment changes together."""
    with _compiler_environment_lock:
        yield


def _before_compiler_fork() -> None:
    _compiler_environment_lock.acquire()


def _after_compiler_fork_parent() -> None:
    _compiler_environment_lock.release()


def _after_compiler_fork_child() -> None:
    global _compiler_environment_lock
    _compiler_environment_lock = RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_compiler_fork,
        after_in_parent=_after_compiler_fork_parent,
        after_in_child=_after_compiler_fork_child,
    )


def _restore_environment_value(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


@contextmanager
def temporary_environment(changes: Mapping[str, str]):
    """Restore every owned value, including partially activated changes."""
    with _compiler_environment_lock, ExitStack() as cleanup:
        for name, value in changes.items():
            cleanup.callback(_restore_environment_value, name, os.environ.get(name))
            os.environ[name] = value
        yield


def clear_abandoned_torch_lock(build_dir: Path) -> None:
    """Remove a PyTorch extension build lock left by an interrupted compile.

    ``torch.utils.cpp_extension`` uses a lock file named ``lock`` in the build
    directory and waits indefinitely when it already exists.  The caller holds
    HydroForge's exclusive per-fingerprint compile lock for ``build_dir``, so
    no live compiler can own that file: any existing lock is abandoned.
    """
    (build_dir / "lock").unlink(missing_ok=True)


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
    lock_path: Path, *, stale_after: float
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
        grace = env.nonnegative_float(env.COMPILE_LOCK_DEAD_PID_GRACE, default=2.0)
        if age < min(stale_after, grace):
            return False
    elif age < stale_after or not env.flag(env.COMPILE_LOCK_STEAL, default=False):
        return False
    lock_path.unlink(missing_ok=True)
    return True


def acquire_compile_lock(
    lock_path: Path, *, cache_probe: Callable[[], Any | None] | None = None
) -> tuple[str | None, Any | None]:
    """Return an acquisition token and no value, or no token and a cache hit.

    A zero timeout waits forever and a zero stale age never removes a lock
    whose holder may be alive.
    """
    stale_after = env.nonnegative_float(env.COMPILE_LOCK_STALE, default=1800.0)
    poll = max(0.05, env.nonnegative_float(env.COMPILE_LOCK_POLL, default=0.25))
    timeout = env.nonnegative_float(env.COMPILE_LOCK_TIMEOUT, default=1800.0)
    deadline = time.time() + timeout if timeout > 0.0 else None

    while True:
        if cache_probe is not None:
            cached = cache_probe()
            if cached is not None:
                return None, cached
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
                    lock_path, stale_after=stale_after
                )
            else:
                token = f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}:{uuid4().hex}"
                try:
                    with cleanup_on_exit(
                        "compile lock descriptor", (lambda: os.close(fd),)
                    ):
                        remaining = memoryview(token.encode())
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
                return token, None
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
                    "has exited, or set HYDROFORGE_COMPILE_LOCK_STEAL=1 "
                    "to override a stale lock."
                )
            time.sleep(poll)
            continue


def release_compile_lock(lock_path: Path, token: str) -> None:
    with _compile_lock_guard(lock_path):
        try:
            if lock_path.read_text() == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass

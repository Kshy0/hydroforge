"""Shared process-environment ownership for native extension compilation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from threading import RLock

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

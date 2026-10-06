# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Shared strict failure types that do not depend on runtime layers."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any


def _exception_message(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException:
        return "<exception message unavailable>"


def failure_description(error: BaseException) -> dict[str, str]:
    """Return one stable rank-transfer-safe exception description."""

    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": _exception_message(error),
    }


def distributed_failure_error(
    scope: str,
    failures: Iterable[dict[str, Any] | None],
) -> RuntimeError:
    """Build one deterministic summary of failures observed across ranks."""

    failed = [
        (rank, failure) for rank, failure in enumerate(failures) if failure is not None
    ]
    details = "; ".join(
        f"rank {rank}: {failure['type']}: {failure['message']}"
        for rank, failure in failed
    )
    return RuntimeError(f"{scope} failed: {details}")


_HYDROFORGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def user_stacklevel() -> int:
    """Return the ``warnings.warn`` stacklevel of the first caller frame
    outside HydroForge and Pydantic, so a warning points at user code.

    Call it from the function that calls ``warnings.warn``.
    """

    prefixes = [_HYDROFORGE_DIR + os.sep]
    pydantic = sys.modules.get("pydantic")
    if pydantic is not None and pydantic.__file__:
        prefixes.append(os.path.dirname(os.path.abspath(pydantic.__file__)) + os.sep)
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(tuple(prefixes)):
        frame = frame.f_back
        level += 1
    return level


class SubstepCompileError(RuntimeError):
    """Raised when a substep contains an operator without strict lowering."""


class UnknownFieldError(KeyError, ValueError):
    """Raised when a name does not identify a declared tensor field.

    It is both a ``KeyError`` and a ``ValueError`` so callers of either
    historical lookup convention keep working; ``str()`` is the plain message.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


def error_message(error: BaseException) -> str:
    """Return an exception's message without ``KeyError``'s repr quoting."""

    if type(error) is KeyError and len(error.args) == 1:
        return str(error.args[0])
    return _exception_message(error)


class ResourceCleanupError(RuntimeError):
    """Report every cleanup failure after all owned resources were attempted."""

    def __init__(self, scope: str, failures: Iterable[BaseException]) -> None:
        self.failures = tuple(failures)
        if not self.failures:
            raise ValueError("ResourceCleanupError requires at least one failure")
        detail = ", ".join(
            f"{type(error).__name__}: {_exception_message(error)}"
            for error in self.failures
        )
        super().__init__(
            f"failed to close {scope} ({len(self.failures)} error(s)): {detail}"
        )


@contextmanager
def cleanup_on_exit(scope: str, actions: Iterable[Callable[[], None]]):
    """Attempt every release and retain any operation failure alongside them."""
    failures = []
    try:
        yield
    except BaseException as error:
        failures.append(error)
    for action in actions:
        try:
            action()
        except BaseException as error:
            failures.append(error)
    if len(failures) > 1:
        raise ResourceCleanupError(scope, failures) from failures[0]
    if failures:
        raise failures[0]

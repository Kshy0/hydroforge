"""Shared strict failure types that do not depend on runtime layers."""

from __future__ import annotations

from typing import Iterable


class ResourceCleanupError(RuntimeError):
    """Report every cleanup failure after all owned resources were attempted."""

    def __init__(self, scope: str, failures: Iterable[BaseException]) -> None:
        self.failures = tuple(failures)
        if not self.failures:
            raise ValueError("ResourceCleanupError requires at least one failure")
        detail = ", ".join(
            f"{type(error).__name__}: {error}" for error in self.failures
        )
        super().__init__(
            f"failed to close {scope} ({len(self.failures)} error(s)): {detail}"
        )

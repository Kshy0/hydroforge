"""Shared strict failure types that do not depend on runtime layers."""

from __future__ import annotations

from typing import Any, Iterable


def failure_description(error: BaseException) -> dict[str, str]:
    """Return one stable rank-transfer-safe exception description."""

    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }


def distributed_failure_error(
    scope: str,
    failures: Iterable[dict[str, Any] | None],
) -> RuntimeError:
    """Build one deterministic summary of failures observed across ranks."""

    failed = [
        (rank, failure) for rank, failure in enumerate(failures)
        if failure is not None
    ]
    details = "; ".join(
        f"rank {rank}: {failure['type']}: {failure['message']}"
        for rank, failure in failed
    )
    return RuntimeError(f"{scope} failed: {details}")


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

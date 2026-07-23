"""Atomic publication primitives for complete filesystem artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_output_path(file_path: str | Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and publish it on success."""

    target = Path(file_path)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(
    file_path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Durably write and atomically publish one text artifact."""

    with atomic_output_path(file_path) as temporary:
        with temporary.open("w", encoding=encoding) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

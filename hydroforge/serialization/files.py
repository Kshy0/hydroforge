"""Atomic publication primitives for complete filesystem artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import stat


@contextmanager
def atomic_output_path(file_path: str | Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and publish it on success."""

    target = Path(file_path)
    existing_mode = None
    try:
        existing_mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        pass

    # tempfile.NamedTemporaryFile hard-codes 0600.  Create the unpredictable,
    # same-directory name ourselves with mode 0666 so the process umask governs
    # permissions of newly published artifacts.
    temporary = None
    for _ in range(100):
        candidate = target.parent / (
            f".{target.name}.{secrets.token_hex(8)}.tmp"
        )
        try:
            descriptor = os.open(
                candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666,
            )
        except FileExistsError:
            continue
        else:
            os.close(descriptor)
            temporary = candidate
            break
    if temporary is None:
        raise FileExistsError(
            f"could not allocate a temporary output beside {target}"
        )
    if existing_mode is not None:
        os.chmod(temporary, existing_mode)
    try:
        yield temporary

        # Writers using this primitive (including netCDF/HDF5) have closed the
        # temporary when control returns here.  Make its data durable before
        # publishing the name, then make the directory entry durable as well.
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary.replace(target)
        directory = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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

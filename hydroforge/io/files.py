# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""File identities, process-local NetCDF read handles, and atomic publication."""

from __future__ import annotations

import os
import secrets
import stat
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Self
from weakref import WeakSet

from netCDF4 import Dataset

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.io.netcdf.options import ensure_hdf5_plugins


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Device, inode, size and modification time of one validated file."""

    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: Path) -> Self:
        status = path.stat()
        return cls(
            device=status.st_dev,
            inode=status.st_ino,
            size=status.st_size,
            mtime_ns=status.st_mtime_ns,
        )

    def verify(self, path: Path, *, label: str) -> None:
        """Raise ``RuntimeError`` when ``path`` no longer is the captured file."""

        try:
            observed = type(self).capture(path)
        except OSError as error:
            raise RuntimeError(
                f"{label} {str(path)!r} changed after validation"
            ) from error
        if observed != self:
            raise RuntimeError(f"{label} {str(path)!r} changed after validation")


_HANDLE_POOLS: WeakSet[_NetCDFHandlePool] = WeakSet()
_HANDLE_POOLS_LOCK = RLock()


def _close_handle_pools_before_fork() -> None:
    """Prevent native NetCDF/HDF5 handles from crossing a process fork."""

    with _HANDLE_POOLS_LOCK:
        pools = tuple(_HANDLE_POOLS)
    for pool in pools:
        try:
            pool.close()
        except BaseException:
            # ``close`` clears ownership before reporting cleanup failures.
            # A fork hook cannot safely propagate such an exception.
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_close_handle_pools_before_fork)


def _close_if_open(dataset: Dataset) -> None:
    if dataset.isopen():
        dataset.close()


class _NetCDFHandlePool:
    """Bounded lazy NetCDF read-handle cache owned by exactly one process.

    Handles are opened only on the first read.  A pool copied into a
    DataLoader worker never reuses a handle created by its parent process,
    and spawn/forkserver pickling drops all live native resources.
    """

    def __init__(self, max_open_files: int = 8) -> None:
        if type(max_open_files) is not int or max_open_files < 1:
            raise ValueError("max_open_files must be a positive exact int")
        self.max_open_files = max_open_files
        self._pid = os.getpid()
        self._handles: OrderedDict[Path, Dataset] = OrderedDict()
        self._lock = RLock()
        with _HANDLE_POOLS_LOCK:
            _HANDLE_POOLS.add(self)

    def _reset_for_process(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        # Native HDF5 state inherited across fork must never be used in the
        # child.  Runtime pools are normally empty before workers start; clear
        # defensively if a caller did perform an earlier main-process read.
        self._handles = OrderedDict()
        self._lock = RLock()
        self._pid = pid

    def _open_locked(self, path: str | Path) -> Dataset:
        """Return one live handle while the process-local lock is held."""

        canonical = Path(path).absolute()
        dataset = self._handles.pop(canonical, None)
        if dataset is not None and dataset.isopen():
            self._handles[canonical] = dataset
            return dataset
        ensure_hdf5_plugins()
        dataset = Dataset(canonical, "r")
        self._handles[canonical] = dataset
        while len(self._handles) > self.max_open_files:
            _old_path, old_dataset = self._handles.popitem(last=False)
            old_dataset.close()
        return dataset

    @contextmanager
    def acquire(self, path: str | Path) -> Iterator[Dataset]:
        """Serialize use of one persistent process-local read handle."""

        self._reset_for_process()
        with self._lock:
            yield self._open_locked(path)

    def close(self) -> None:
        """Close all handles owned by the current process, idempotently."""

        self._reset_for_process()
        with self._lock:
            handles = tuple(self._handles.values())
            self._handles.clear()
            with cleanup_on_exit(
                "NetCDF read handles",
                tuple(partial(_close_if_open, dataset) for dataset in handles),
            ):
                pass

    def __getstate__(self) -> dict[str, int]:
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(max_open_files=state["max_open_files"])

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class FileInspection:
    """Construction-time identity capture of the files one declaration reads.

    The first use of a path captures its identity; every later use, and the end
    of each use, verifies that the path is still that file.
    """

    def __init__(self, *, label: str) -> None:
        self.label = label
        self._identities: dict[Path, FileIdentity] = {}

    @contextmanager
    def open(self, path: str | Path) -> Iterator[Path]:
        """Yield the absolute path of one file checked on both sides of its use."""

        canonical = Path(path).absolute()
        identity = self._identities.get(canonical)
        if identity is None:
            identity = self._identities[canonical] = FileIdentity.capture(canonical)
        else:
            identity.verify(canonical, label=self.label)
        yield canonical
        identity.verify(canonical, label=self.label)

    @contextmanager
    def open_netcdf(self, path: str | Path) -> Iterator[Dataset]:
        """Yield one inspected NetCDF file opened read-only for this use."""

        with self.open(path) as canonical:
            ensure_hdf5_plugins()
            with Dataset(canonical, "r") as dataset:
                yield dataset

    def files(self) -> SourceFiles:
        """Verify every inspected identity once more and freeze the file set."""

        for path, identity in self._identities.items():
            identity.verify(path, label=self.label)
        return SourceFiles(self._identities, label=self.label)


class SourceFiles:
    """Captured identities of one declared file set and its NetCDF read handles.

    Every read checks the file identity before and after using a pooled,
    process-local handle.  Pickling keeps only the identities; the receiving
    process opens its own handles on first use.
    """

    def __init__(
        self,
        identities: Mapping[Path, FileIdentity],
        *,
        label: str,
        max_open: int = 8,
    ) -> None:
        self.identities = MappingProxyType(dict(identities))
        self.label = label
        self._handles = _NetCDFHandlePool(max_open)

    @classmethod
    def inspect(cls, *, label: str) -> FileInspection:
        """Start a construction-time inspection whose result is ``files()``."""

        return FileInspection(label=label)

    @classmethod
    def capture(
        cls,
        paths: Iterable[Path],
        *,
        label: str,
        max_open: int = 8,
    ) -> Self:
        """Capture the identity of each absolute path now."""

        return cls(
            {path: FileIdentity.capture(path) for path in paths},
            label=label,
            max_open=max_open,
        )

    def checked(self, path: str | Path) -> Path:
        """Return the absolute path after confirming its captured identity."""

        canonical = Path(path).absolute()
        self.identities[canonical].verify(canonical, label=self.label)
        return canonical

    def verify(self, path: str | Path) -> None:
        self.checked(path)

    @contextmanager
    def open_netcdf(self, path: str | Path) -> Iterator[Dataset]:
        """Yield a pooled read handle of one captured file, checked on both sides."""

        canonical = self.checked(path)
        with self._handles.acquire(canonical) as dataset:
            yield dataset
        self.verify(canonical)

    def close(self) -> None:
        """Close this process's handles; later reads reopen them lazily."""

        self._handles.close()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "identities": dict(self.identities),
            "label": self.label,
            "max_open": self._handles.max_open_files,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["identities"],
            label=state["label"],
            max_open=state["max_open"],
        )


def fsync_file(path: str | Path) -> None:
    """Flush one closed or externally written file's data to stable storage."""

    # Windows _commit requires a writable file descriptor.
    descriptor = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: str | Path) -> None:
    """Make directory entries durable where Python can open directories."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_file(temporary: Path, target: Path) -> None:
    """Durably replace ``target`` with the closed file ``temporary`` beside it.

    An existing target's permissions carry over.  File data is synced before
    the rename; the directory entry is additionally synced on POSIX, where
    Python supports opening directory descriptors.
    """

    try:
        existing_mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        existing_mode = None
    if existing_mode is not None:
        os.chmod(temporary, existing_mode)
    fsync_file(temporary)
    temporary.replace(target)
    fsync_directory(target.parent)


@contextmanager
def atomic_output_path(
    file_path: str | Path,
    *,
    preserve_suffix: bool = False,
) -> Iterator[Path]:
    """Yield a same-directory temporary path and publish it on success.

    ``preserve_suffix`` keeps the target suffix at the end of the temporary
    name for writers that select their container format from the filename.
    """

    target = Path(file_path)
    # tempfile.NamedTemporaryFile hard-codes 0600.  Create the unpredictable,
    # same-directory name ourselves with mode 0666 so the process umask governs
    # permissions of newly published artifacts.
    temporary = None
    for _ in range(100):
        token = secrets.token_hex(8)
        if preserve_suffix and target.suffix:
            stem = target.name[: -len(target.suffix)]
            temporary_name = f".{stem}.{token}.tmp{target.suffix}"
        else:
            temporary_name = f".{target.name}.{token}.tmp"
        candidate = target.parent / temporary_name
        try:
            descriptor = os.open(
                candidate,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
        except FileExistsError:
            continue
        else:
            os.close(descriptor)
            temporary = candidate
            break
    if temporary is None:
        raise FileExistsError(f"could not allocate a temporary output beside {target}")
    try:
        yield temporary
        # Writers using this primitive (including netCDF/HDF5) have closed the
        # temporary when control returns here.
        publish_file(temporary, target)
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
        temporary.write_text(content, encoding=encoding)

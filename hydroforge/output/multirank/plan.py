"""Immutable storage identity compiled for one multi-rank output reader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import cftime
import numpy as np


@dataclass(frozen=True, slots=True)
class _ReaderFileIdentity:
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

    def verify(self, path: Path) -> None:
        try:
            observed = type(self).capture(path)
        except OSError as error:
            raise RuntimeError(
                f"Reader source file {str(path)!r} changed after validation"
            ) from error
        if observed != self:
            raise RuntimeError(
                f"Reader source file {str(path)!r} changed after validation"
            )


def _frozen_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    result = np.array(value, order="C", copy=True)
    result.setflags(write=False)
    return result


def _freeze_rank_info(info: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = dict(info)
    frozen["paths"] = tuple(Path(path).absolute() for path in info["paths"])
    frozen["years"] = tuple(info["years"])
    frozen["file_time_offsets"] = tuple(info["file_time_offsets"])
    for name in ("coord_raw", "x", "y"):
        frozen[name] = _frozen_array(info.get(name))
    frozen.pop("cache", None)
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class _ReaderStoragePlan:
    """Complete immutable schema and coordinate plan for trusted reads."""

    rank_files: tuple[Mapping[str, Any], ...]
    time_units: str
    time_calendar: str
    time_values_num: np.ndarray
    time_datetimes: tuple[datetime | cftime.datetime, ...]
    slice_start: int
    slice_end: int
    time_indices: np.ndarray
    map_shape: tuple[int, int] | None
    file_identities: Mapping[Path, _ReaderFileIdentity]

    @classmethod
    def compile(
        cls,
        *,
        rank_files: Sequence[Mapping[str, Any]],
        time_units: str,
        time_calendar: str,
        time_values_num: np.ndarray,
        time_datetimes: Sequence[datetime | cftime.datetime],
        slice_start: int,
        slice_end: int,
        time_indices: np.ndarray,
        map_shape: tuple[int, int] | None,
        file_identities: Mapping[Path, _ReaderFileIdentity],
    ) -> Self:
        values = _frozen_array(time_values_num)
        indices = _frozen_array(time_indices)
        return cls(
            rank_files=tuple(_freeze_rank_info(info) for info in rank_files),
            time_units=time_units,
            time_calendar=time_calendar,
            time_values_num=values,
            time_datetimes=tuple(time_datetimes),
            slice_start=slice_start,
            slice_end=slice_end,
            time_indices=indices,
            map_shape=map_shape,
            file_identities=MappingProxyType(dict(file_identities)),
        )

    @property
    def time_len(self) -> int:
        return len(self.time_datetimes)

    @staticmethod
    def canonical_path(path: str | Path) -> Path:
        return Path(path).absolute()

    def checked_path(self, path: str | Path) -> Path:
        canonical = self.canonical_path(path)
        self.file_identities[canonical].verify(canonical)
        return canonical

    def verify_path(self, path: str | Path) -> None:
        canonical = self.canonical_path(path)
        self.file_identities[canonical].verify(canonical)

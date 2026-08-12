# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Immutable source-chunk planning derived from one temporal contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from hydroforge.contracts.temporal import (
    DatasetTemporalContract,
    DateLike,
    timedelta_quotient,
)


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """One real, unpadded source read on a logical dataset timeline."""

    index: int
    phase: Literal["spinup", "main"]
    source_start: DateLike
    length: int
    phase_offset: int
    source_offset: int
    spinup_cycle: int | None = None

    def source_time(self, offset: int, interval: timedelta) -> DateLike:
        """Return one logical source timestamp inside this chunk."""

        if type(offset) is not int:
            raise TypeError("chunk source offset must be an exact int")
        if not 0 <= offset < self.length:
            raise IndexError(offset)
        return self.source_start + interval * offset


@dataclass(frozen=True, slots=True)
class SourceChunkPlan:
    """The complete storage-chunk layout for a dataset contract.

    ``DatasetTemporalContract`` uses half-open support, while dataset
    constructors retain their historical inclusive ``end_date`` input. The
    boundary conversion happens before this plan is built. Every consumer then
    sees the same real chunk lengths, including each short final chunk.
    """

    contract: DatasetTemporalContract
    chunk_len: int
    _chunks: tuple[SourceChunk, ...] = field(init=False, repr=False)
    _spinup_source_count: int = field(init=False, repr=False)
    _num_spinup_chunks: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DatasetTemporalContract):
            raise TypeError("chunk plan requires a DatasetTemporalContract")
        if type(self.chunk_len) is not int or self.chunk_len < 1:
            raise ValueError("chunk_len must be an exact positive int")

        chunks: list[SourceChunk] = []
        spinup_source_count = 0
        spinup = self.contract.spinup
        if spinup is not None:
            spinup_source_count = timedelta_quotient(
                spinup.source_end - spinup.source_start,
                self.contract.interval,
                duration_label="spin-up source duration",
                interval_label="dataset sample interval",
            )
            source_origin_offset = timedelta_quotient(
                spinup.source_start - self.contract.start,
                self.contract.interval,
                duration_label="spin-up source origin offset",
                interval_label="dataset sample interval",
            )
            for cycle in range(spinup.cycles):
                self._append_phase(
                    chunks,
                    phase="spinup",
                    start=spinup.source_start,
                    count=spinup_source_count,
                    source_origin_offset=source_origin_offset,
                    spinup_cycle=cycle,
                )

        num_spinup_chunks = len(chunks)
        self._append_phase(
            chunks,
            phase="main",
            start=self.contract.start,
            count=self.contract.count,
            source_origin_offset=0,
            spinup_cycle=None,
        )
        object.__setattr__(self, "_chunks", tuple(chunks))
        object.__setattr__(self, "_spinup_source_count", spinup_source_count)
        object.__setattr__(self, "_num_spinup_chunks", num_spinup_chunks)

    def _append_phase(
        self,
        chunks: list[SourceChunk],
        *,
        phase: Literal["spinup", "main"],
        start: DateLike,
        count: int,
        source_origin_offset: int,
        spinup_cycle: int | None,
    ) -> None:
        for phase_offset in range(0, count, self.chunk_len):
            length = min(self.chunk_len, count - phase_offset)
            chunks.append(SourceChunk(
                index=len(chunks),
                phase=phase,
                source_start=start + self.contract.interval * phase_offset,
                length=length,
                phase_offset=phase_offset,
                source_offset=source_origin_offset + phase_offset,
                spinup_cycle=spinup_cycle,
            ))

    @property
    def spinup_source_count_per_cycle(self) -> int:
        return self._spinup_source_count

    @property
    def num_spinup_chunks(self) -> int:
        return self._num_spinup_chunks

    def source_time(self, chunk_index: int, offset: int) -> DateLike:
        return self[chunk_index].source_time(offset, self.contract.interval)

    def __getitem__(self, index: int) -> SourceChunk:
        if type(index) is not int:
            raise TypeError("dataset index must be an exact int")
        try:
            return self._chunks[index]
        except IndexError:
            raise IndexError(index) from None

    def __iter__(self):
        return iter(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

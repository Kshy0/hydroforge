# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Immutable source-chunk planning derived from one temporal contract."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from hydroforge.contracts.temporal import (
    _DatasetTemporalDomain,
    DateLike,
    date_calendar,
    require_calendar,
    _timedelta_quotient_trusted,
)
from hydroforge.contracts.validation import HydroForgeModel


class _ChunkPlanIndexQuery(HydroForgeModel):
    index: int = Field(strict=True)
    length: int = Field(strict=True, ge=0, exclude=True)

    _resolved: int = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self):
        index = self.index
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise ValueError(
                f"chunk-plan index must satisfy -{self.length} <= index < "
                f"{self.length}; got {self.index}"
            )
        self._resolved = index
        return self

    @property
    def resolved(self) -> int:
        return self._resolved


class SourceChunk(HydroForgeModel):
    """One real, unpadded source read on a logical dataset timeline."""

    index: int
    phase: Literal["spinup", "main"]
    source_start: DateLike
    length: int
    phase_offset: int
    source_offset: int
    # The immutable contract is part of the request identity.  Two plans can
    # otherwise produce identical offsets for different cadences (for example
    # daily and hourly first chunks), allowing a foreign request to be accepted
    # and read silently against the wrong timeline.
    temporal_domain: _DatasetTemporalDomain
    spinup_cycle: int | None = None

    @model_validator(mode="after")
    def _validate_chunk(self):
        if not isinstance(self.temporal_domain, _DatasetTemporalDomain):
            raise ValueError("source chunk requires its Dataset timeline")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("source chunk index must be a non-negative int")
        if type(self.phase) is not str or self.phase not in {"spinup", "main"}:
            raise ValueError("source chunk phase must be 'spinup' or 'main'")
        if type(self.source_start) is not type(self.temporal_domain.start):
            raise ValueError(
                "source chunk and dataset contract must use the same datetime "
                "representation"
            )
        require_calendar(
            self.source_start,
            self.temporal_domain.calendar,
            label="source chunk start",
        )
        if date_calendar(self.source_start) != date_calendar(
            self.temporal_domain.start,
        ):
            raise ValueError(
                "source chunk and dataset contract use different calendars"
            )
        if type(self.length) is not int or self.length < 1:
            raise ValueError("source chunk length must be a positive int")
        if type(self.phase_offset) is not int or self.phase_offset < 0:
            raise ValueError("source chunk phase offset must be a non-negative int")
        if type(self.source_offset) is not int:
            raise ValueError("source chunk source offset must be an exact int")
        expected_start = (
            self.temporal_domain.start
            + self.temporal_domain.interval * self.source_offset
        )
        if self.source_start != expected_start:
            raise ValueError("source chunk start does not match its source offset")
        if self.phase == "main":
            if self.spinup_cycle is not None:
                raise ValueError("main source chunks cannot have a spinup cycle")
            if self.source_offset != self.phase_offset:
                raise ValueError(
                    "main source chunk offsets must refer to the same sample"
                )
            if self.phase_offset + self.length > self.temporal_domain.count:
                raise ValueError(
                    "main source chunk extends beyond the dataset contract"
                )
            return self

        spinup = self.temporal_domain.spinup
        if spinup is None:
            raise ValueError("spinup source chunks require a dataset spinup contract")
        if (
            type(self.spinup_cycle) is not int
            or not 0 <= self.spinup_cycle < spinup.cycles
        ):
            raise ValueError(
                "spinup source chunk cycle is outside the dataset contract"
            )
        spinup_count = _timedelta_quotient_trusted(
            spinup.source_end - spinup.source_start,
            self.temporal_domain.interval,
            duration_label="spin-up source duration",
            interval_label="dataset sample interval",
        )
        if self.phase_offset + self.length > spinup_count:
            raise ValueError("spinup source chunk extends beyond the spinup contract")
        expected_spinup_start = (
            spinup.source_start + self.temporal_domain.interval * self.phase_offset
        )
        if self.source_start != expected_spinup_start:
            raise ValueError(
                "spinup source chunk start does not match its phase offset"
            )
        return self

    def _source_time(self, offset: int) -> DateLike:
        """Return one logical source timestamp inside this chunk."""

        return self.source_start + self.temporal_domain.interval * offset

    def _source_times(self) -> tuple[DateLike, ...]:
        """Return every logical source timestamp in this request."""

        return tuple(self._source_time(offset) for offset in range(self.length))

    def _main_source_slice(self, source_count: int) -> slice:
        """Return a bounded slice into storage aligned to the main source axis."""

        if type(source_count) is not int or source_count < 1:
            raise ValueError("main source count must be a positive int")
        stop = self.source_offset + self.length
        if self.source_offset < 0 or stop > source_count:
            raise IndexError(
                f"{self.phase} source chunk [{self.source_offset}, {stop}) is "
                f"outside the main-aligned source table [0, {source_count})"
            )
        return slice(self.source_offset, stop)


class SourceChunkPlan(HydroForgeModel):
    """The complete storage-chunk layout for a dataset contract.

    The internal temporal domain uses half-open support, while Dataset
    constructors retain their historical inclusive ``end_date`` input. The
    boundary conversion happens before this plan is built. Every consumer then
    sees the same real chunk lengths, including each short final chunk.
    """

    temporal_domain: _DatasetTemporalDomain
    chunk_len: int = Field(strict=True, ge=1)
    _chunks: tuple[SourceChunk, ...] = PrivateAttr(default=())
    _spinup_source_count: int = PrivateAttr(default=0)
    _num_spinup_chunks: int = PrivateAttr(default=0)

    @model_validator(mode="after")
    def _compile_chunks(self):
        chunks: list[SourceChunk] = []
        spinup_source_count = 0
        spinup = self.temporal_domain.spinup
        if spinup is not None:
            spinup_source_count = _timedelta_quotient_trusted(
                spinup.source_end - spinup.source_start,
                self.temporal_domain.interval,
                duration_label="spin-up source duration",
                interval_label="dataset sample interval",
            )
            source_origin_offset = _timedelta_quotient_trusted(
                spinup.source_start - self.temporal_domain.start,
                self.temporal_domain.interval,
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
            start=self.temporal_domain.start,
            count=self.temporal_domain.count,
            source_origin_offset=0,
            spinup_cycle=None,
        )
        self._chunks = tuple(chunks)
        self._spinup_source_count = spinup_source_count
        self._num_spinup_chunks = num_spinup_chunks
        return self

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
            chunk = SourceChunk(
                index=len(chunks),
                phase=phase,
                source_start=(start + self.temporal_domain.interval * phase_offset),
                length=length,
                phase_offset=phase_offset,
                source_offset=source_origin_offset + phase_offset,
                spinup_cycle=spinup_cycle,
                temporal_domain=self.temporal_domain,
            )
            chunks.append(chunk)

    @property
    def spinup_source_count_per_cycle(self) -> int:
        return self._spinup_source_count

    @property
    def num_spinup_chunks(self) -> int:
        return self._num_spinup_chunks

    def _validate_main_source_count(self, source_count: int) -> None:
        """Validate every planned read against a main-aligned source table."""

        for chunk in self._chunks:
            try:
                chunk._main_source_slice(source_count)
            except IndexError as error:
                raise ValueError(
                    "chunk plan cannot be served by the main-aligned source "
                    f"table: {error}"
                ) from error

    def __getitem__(self, index: int) -> SourceChunk:
        query = _ChunkPlanIndexQuery(index=index, length=len(self._chunks))
        return self._at_trusted(query.resolved)

    def _at_trusted(self, index: int) -> SourceChunk:
        """Return one framework-resolved chunk without another query model."""

        return self._chunks[index]

    def __iter__(self):
        return iter(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

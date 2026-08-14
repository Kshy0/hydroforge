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
    # The immutable contract is part of the request identity.  Two plans can
    # otherwise produce identical offsets for different cadences (for example
    # daily and hourly first chunks), allowing a foreign request to be accepted
    # and read silently against the wrong timeline.
    contract: DatasetTemporalContract
    # A contract describes temporal values only; two independent datasets can
    # therefore still have equal contracts.  This opaque plan token keeps a
    # request tied to the exact source plan that issued it.
    provenance: object = field(repr=False)
    spinup_cycle: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DatasetTemporalContract):
            raise TypeError("source chunk requires its DatasetTemporalContract")

    def source_time(self, offset: int, interval: timedelta) -> DateLike:
        """Return one logical source timestamp inside this chunk."""

        if type(offset) is not int:
            raise TypeError("chunk source offset must be an exact int")
        if not 0 <= offset < self.length:
            raise IndexError(offset)
        if interval != self.contract.interval:
            raise ValueError(
                "chunk interval differs from its DatasetTemporalContract"
            )
        return self.source_start + interval * offset

    def source_times(self, interval: timedelta) -> tuple[DateLike, ...]:
        """Return every logical source timestamp in this request."""

        return tuple(
            self.source_time(offset, interval)
            for offset in range(self.length)
        )

    def main_source_slice(self, source_count: int) -> slice:
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
    _provenance: object = field(init=False, repr=False, compare=False)
    # Keep accepted tokens as an identity list rather than a set.  Provenance
    # is deliberately opaque and identity-owned; set membership would invoke
    # user-defined ``__hash__``/``__eq__`` methods on a forged token and could
    # make two independent plans appear equal by accident.
    _accepted_provenance: list[object] = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DatasetTemporalContract):
            raise TypeError("chunk plan requires a DatasetTemporalContract")
        if type(self.chunk_len) is not int or self.chunk_len < 1:
            raise ValueError("chunk_len must be an exact positive int")

        object.__setattr__(self, "_provenance", object())
        object.__setattr__(self, "_accepted_provenance", [])

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
                contract=self.contract,
                provenance=self._provenance,
            ))

    @property
    def spinup_source_count_per_cycle(self) -> int:
        return self._spinup_source_count

    @property
    def num_spinup_chunks(self) -> int:
        return self._num_spinup_chunks

    def validate_main_source_count(self, source_count: int) -> None:
        """Validate every planned read against a main-aligned source table."""

        for chunk in self._chunks:
            try:
                chunk.main_source_slice(source_count)
            except IndexError as error:
                raise ValueError(
                    "chunk plan cannot be served by the main-aligned source "
                    f"table: {error}"
                ) from error

    def validate_chunk(self, chunk: SourceChunk) -> None:
        """Require a request to be one exact member of this plan."""

        if not isinstance(chunk, SourceChunk):
            raise TypeError("read_chunk requires a SourceChunk")
        if chunk.contract != self.contract:
            raise ValueError(
                "source chunk belongs to a different DatasetTemporalContract"
            )
        accepted = self._has_accepted_provenance(chunk.provenance)
        if chunk.provenance is not self._provenance and not accepted:
            raise ValueError("source chunk belongs to a different SourceChunkPlan")
        try:
            expected = self[chunk.index]
        except (IndexError, TypeError):
            raise ValueError(
                "source chunk does not belong to this dataset"
            ) from None
        # Dataclass equality is intentionally not used here: Python considers
        # values such as ``True == 1`` and ``1.0 == 1`` equal.  A forged
        # request with one of those values could then pass validation and
        # reach a storage adapter with a malformed offset/length.  Composite
        # adoption still permits a different object, but every structural
        # field must have the exact same type and value as the plan member.
        same_request = self._same_request_fields(chunk, expected)
        if not same_request:
            raise ValueError("source chunk does not belong to this dataset")

    @staticmethod
    def _same_request_fields(
        actual: SourceChunk, expected: SourceChunk,
    ) -> bool:
        return all(
            type(actual_value) is type(expected_value)
            and actual_value == expected_value
            for name in (
                "index", "phase", "source_start", "length",
                "phase_offset", "source_offset", "contract",
                "spinup_cycle",
            )
            for actual_value, expected_value in [
                (getattr(actual, name), getattr(expected, name)),
            ]
        )

    def _accept_provenance(self, provenance: object) -> None:
        """Allow an explicit composite owner to forward its exact request."""
        if not any(token is provenance for token in self._accepted_provenance):
            self._accepted_provenance.append(provenance)

    def _has_accepted_provenance(self, provenance: object) -> bool:
        """Return whether a composite token was explicitly adopted."""

        return any(token is provenance for token in self._accepted_provenance)

    def __copy__(self) -> SourceChunkPlan:
        """Treat a shallow copy as an alias of this immutable request plan.

        Chunks retain the plan's opaque provenance token.  Letting
        ``copy.copy`` manufacture a second plan object while sharing that
        token would make the two plans indistinguishable to provenance checks.
        ``deepcopy`` remains available when an independent plan is required.
        """

        return self

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

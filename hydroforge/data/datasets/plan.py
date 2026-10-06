# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Immutable temporal plans compiled once for one forcing dataset."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, Self, cast

import numpy as np

from hydroforge.contracts.schedule import SimulationSchedule, SpinupSchedule
from hydroforge.core.time import (
    DateLike,
    normalize_calendar_dates,
    timedelta_microseconds,
    timedelta_quotient,
)

UpsamplingMethod = Literal["repeat", "distribute"]

# The standard (mixed Julian/Gregorian) and proleptic Gregorian calendars label
# every instant from the 1582-10-15 reform onwards identically.
_GREGORIAN_CALENDARS = frozenset({"standard", "proleptic_gregorian"})
_GREGORIAN_REFORM = (1582, 10, 15)


def _components(value: DateLike) -> tuple[int, ...]:
    return (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
    )


def _domain_dates(domain: TemporalDomain) -> tuple[DateLike, ...]:
    if domain.spinup is None:
        return (domain.start,)
    return (domain.start, domain.spinup.source_start, domain.spinup.source_end)


def _same_temporal_contract(mine: TemporalDomain, theirs: TemporalDomain) -> bool:
    """Whether two domains read the same instants on the same cadence."""

    if (mine.interval, mine.count, mine.spinup_count) != (
        theirs.interval,
        theirs.count,
        theirs.spinup_count,
    ):
        return False
    if mine.calendar == theirs.calendar:
        return type(theirs.start) is type(mine.start) and (
            mine.start == theirs.start and mine.spinup == theirs.spinup
        )
    if {mine.calendar, theirs.calendar} != _GREGORIAN_CALENDARS:
        return False
    if (mine.spinup is None) != (theirs.spinup is None):
        return False
    if mine.spinup is not None and mine.spinup.cycles != theirs.spinup.cycles:
        return False
    left, right = _domain_dates(mine), _domain_dates(theirs)
    return all(
        _components(a) == _components(b) and _components(a)[:3] >= _GREGORIAN_REFORM
        for a, b in zip(left, right, strict=True)
    )


def plan_index(index: int, length: int, *, label: str) -> int:
    """Resolve one integer (``__index__``), possibly negative, sequence index.

    Booleans are rejected; NumPy integers from custom samplers are accepted.
    """

    if isinstance(index, (bool, np.bool_)) or not hasattr(type(index), "__index__"):
        raise TypeError(f"{label} index must be an int; got {type(index).__name__}")
    index = operator.index(index)
    resolved = index + length if index < 0 else index
    if not 0 <= resolved < length:
        raise IndexError(
            f"{label} index must satisfy -{length} <= index < {length}; got {index}"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class TemporalDomain:
    """Source samples ``start + k * interval`` (``k < count``) in one calendar.

    ``calendar_declared`` is false only for the provisional ``standard``
    calendar of plain ``datetime`` bounds, which storage metadata may replace.
    """

    calendar: str
    calendar_declared: bool
    start: DateLike
    interval: timedelta
    count: int
    spinup: SpinupSchedule | None
    spinup_count: int

    @classmethod
    def declare(
        cls,
        *,
        start_date: DateLike,
        end_date: DateLike,
        time_interval: timedelta,
        calendar: str | None,
        spin_up_cycles: int,
        spin_up_start_date: DateLike | None,
        spin_up_end_date: DateLike | None,
    ) -> Self:
        """Compile the inclusive dataset declaration into a half-open domain."""

        resolved, dates, defaulted = normalize_calendar_dates(
            {
                "dataset start_date": start_date,
                "dataset end_date": end_date,
                "dataset spin_up_start_date": spin_up_start_date,
                "dataset spin_up_end_date": spin_up_end_date,
            },
            calendar=calendar,
        )
        start = cast(DateLike, dates["dataset start_date"])
        end = cast(DateLike, dates["dataset end_date"])
        if end < start:
            raise ValueError("dataset end_date must not precede start_date")
        if timedelta_microseconds(time_interval) <= 0:
            raise ValueError("dataset sample interval must be positive")
        count = (
            timedelta_quotient(
                end - start,
                time_interval,
                duration_label="dataset endpoint span",
                interval_label="dataset sample interval",
            )
            + 1
        )
        if (spin_up_start_date is None) != (spin_up_end_date is None):
            raise ValueError(
                "spin_up_start_date and spin_up_end_date must be provided together"
            )
        if spin_up_cycles > 0 and spin_up_start_date is None:
            raise ValueError(
                "spin-up dates are required when spin_up_cycles is positive"
            )
        if spin_up_cycles == 0 and spin_up_start_date is not None:
            raise ValueError("spin-up dates require a positive spin_up_cycles value")
        spinup = None
        spinup_count = 0
        if spin_up_cycles > 0:
            spinup = SpinupSchedule(
                source_start=dates["dataset spin_up_start_date"],
                source_end=dates["dataset spin_up_end_date"] + time_interval,
                cycles=spin_up_cycles,
            )
            spinup_count = timedelta_quotient(
                spinup.source_end - spinup.source_start,
                time_interval,
                duration_label="dataset spinup source duration",
                interval_label="dataset sample interval",
            )
        if spinup is not None:
            timedelta_quotient(
                spinup.source_start - start,
                time_interval,
                duration_label="spinup start offset",
                interval_label="dataset sample interval",
            )
        return cls(
            calendar=resolved,
            calendar_declared=not defaulted,
            start=start,
            interval=time_interval,
            count=count,
            spinup=spinup,
            spinup_count=spinup_count,
        )

    @property
    def end(self) -> DateLike:
        """Exclusive end of the main period."""

        return self.start + self.interval * self.count

    def times(self) -> tuple[DateLike, ...]:
        """Main-period sample times."""

        return tuple(self.start + self.interval * index for index in range(self.count))

    def spinup_times(self) -> tuple[DateLike, ...]:
        """Sample times of one spin-up cycle."""

        if self.spinup is None:
            return ()
        return tuple(
            self.spinup.source_start + self.interval * index
            for index in range(self.spinup_count)
        )


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """One real, unpadded source read on a dataset timeline."""

    index: int
    phase: Literal["spinup", "main"]
    source_start: DateLike
    length: int
    phase_offset: int
    source_offset: int
    spinup_cycle: int | None
    interval: timedelta

    def source_times(self) -> tuple[DateLike, ...]:
        """Every logical source time read by this chunk."""

        return tuple(
            self.source_start + self.interval * offset for offset in range(self.length)
        )


@dataclass(frozen=True, slots=True)
class SourceChunkPlan:
    """All chunks of one domain: spin-up cycles, then the main period.

    Every chunk keeps its real length, including each short final chunk.
    """

    chunks: tuple[SourceChunk, ...]
    chunk_len: int
    num_spinup_chunks: int
    spinup_source_count: int

    @classmethod
    def compile(cls, domain: TemporalDomain, chunk_len: int) -> Self:
        chunks: list[SourceChunk] = []

        def append_phase(
            phase: Literal["spinup", "main"],
            start: DateLike,
            count: int,
            origin: int,
            cycle: int | None,
        ) -> None:
            for offset in range(0, count, chunk_len):
                chunks.append(
                    SourceChunk(
                        index=len(chunks),
                        phase=phase,
                        source_start=start + domain.interval * offset,
                        length=min(chunk_len, count - offset),
                        phase_offset=offset,
                        source_offset=origin + offset,
                        spinup_cycle=cycle,
                        interval=domain.interval,
                    )
                )

        spinup = domain.spinup
        if spinup is not None:
            origin = timedelta_quotient(
                spinup.source_start - domain.start,
                domain.interval,
                duration_label="spin-up source origin offset",
                interval_label="dataset sample interval",
            )
            for cycle in range(spinup.cycles):
                append_phase(
                    "spinup", spinup.source_start, domain.spinup_count, origin, cycle
                )
        num_spinup_chunks = len(chunks)
        append_phase("main", domain.start, domain.count, 0, None)
        return cls(
            chunks=tuple(chunks),
            chunk_len=chunk_len,
            num_spinup_chunks=num_spinup_chunks,
            spinup_source_count=domain.spinup_count,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def __getitem__(self, index: int) -> SourceChunk:
        return self.chunks[plan_index(index, len(self.chunks), label="chunk-plan")]


def compile_cadence(
    time_interval: timedelta, model_step: timedelta, upsampling: UpsamplingMethod | None
) -> int:
    """Validate cadence without scanning storage or resolving its calendar."""
    step = timedelta_microseconds(model_step)
    if step <= 0:
        raise ValueError("model_step must be positive")
    if step > timedelta_microseconds(time_interval):
        raise ValueError("model_step must not exceed dataset time_interval")
    reuse_count = timedelta_quotient(
        time_interval,
        model_step,
        duration_label="dataset time_interval",
        interval_label="model_step",
    )
    if reuse_count > 1 and upsampling not in {"repeat", "distribute"}:
        raise ValueError(
            "upsampling must be explicitly 'repeat' or 'distribute' "
            "when model_step is shorter than dataset time_interval"
        )
    if reuse_count == 1 and upsampling is not None:
        raise ValueError("upsampling must be None when model_step equals time_interval")
    return reuse_count


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    """The temporal domain, its chunks and the model schedule of one dataset."""

    domain: TemporalDomain
    chunk_plan: SourceChunkPlan
    schedule: SimulationSchedule
    reuse_count: int

    @classmethod
    def compile(
        cls,
        domain: TemporalDomain,
        *,
        chunk_len: int,
        model_step: timedelta,
        upsampling: UpsamplingMethod | None,
    ) -> Self:
        reuse_count = compile_cadence(domain.interval, model_step, upsampling)
        schedule = SimulationSchedule._from_domain(
            calendar=domain.calendar,
            start=domain.start,
            end=domain.end,
            source_interval=domain.interval,
            source_count=domain.count,
            spinup=domain.spinup,
            spinup_source_count=domain.spinup_count,
            step=model_step,
            reuse_count=reuse_count,
        )
        return cls(
            domain=domain,
            chunk_plan=SourceChunkPlan.compile(domain, chunk_len),
            schedule=schedule,
            reuse_count=reuse_count,
        )

    def require_equivalent(self, other: DatasetPlan, *, label: str) -> None:
        """Reject a plan that reads different samples or chunks on another cadence."""

        mine, theirs = self.domain, other.domain
        if not _same_temporal_contract(mine, theirs):
            raise ValueError(f"{label} has a different temporal contract")
        if other.schedule.cadence != self.schedule.cadence:
            raise ValueError(f"{label} has a different model cadence")
        if other.chunk_plan.chunk_len != self.chunk_plan.chunk_len:
            raise ValueError(f"{label} has a different chunk length")

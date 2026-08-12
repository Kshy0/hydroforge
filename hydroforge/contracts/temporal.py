"""Immutable temporal contracts shared by drivers, datasets, and models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Literal, Mapping, TypeAlias, cast

import cftime


DateLike: TypeAlias = datetime | cftime.datetime
CalendarPeriod = Literal["day", "month", "year"]
SimulationPhase = Literal["spinup", "main"]
UpsamplingMethod = Literal["repeat", "distribute"]


_CALENDAR_ALIASES = {
    "gregorian": "standard",
    "standard": "standard",
    "365_day": "noleap",
    "366_day": "all_leap",
}

_MICROSECONDS_PER_SECOND = 1_000_000
_SECONDS_PER_DAY = 86_400

_CFTIME_DATETIME_TYPES = {
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "noleap": cftime.DatetimeNoLeap,
    "all_leap": cftime.DatetimeAllLeap,
    "360_day": cftime.Datetime360Day,
    "julian": cftime.DatetimeJulian,
}


def timedelta_microseconds(value: timedelta, *, label: str = "duration") -> int:
    """Return the exact integer duration represented by ``timedelta``.

    ``timedelta.total_seconds()`` is a float and loses microseconds for long
    spans.  Temporal alignment, counts, and persisted identities must use this
    definition; floats are reserved for values handed to numerical kernels.
    """

    if type(value) is not timedelta:
        raise TypeError(f"{label} must be a timedelta")
    return (
        (value.days * _SECONDS_PER_DAY + value.seconds)
        * _MICROSECONDS_PER_SECOND
        + value.microseconds
    )


def timedelta_quotient(
    duration: timedelta,
    interval: timedelta,
    *,
    duration_label: str = "duration",
    interval_label: str = "interval",
) -> int:
    """Return an exact integral duration/interval ratio or reject misalignment."""

    numerator = timedelta_microseconds(duration, label=duration_label)
    denominator = timedelta_microseconds(interval, label=interval_label)
    if denominator <= 0:
        raise ValueError(f"{interval_label} must be positive")
    quotient, remainder = divmod(numerator, denominator)
    if remainder:
        raise ValueError(
            f"{duration_label}={duration!r} is not an exact multiple of "
            f"{interval_label}={interval!r}"
        )
    return quotient


def canonical_calendar(calendar: str) -> str:
    """Normalize only aliases that cftime defines as equivalent."""
    normalized = str(calendar).strip().lower()
    return _CALENDAR_ALIASES.get(normalized, normalized)


def convert_calendar_date(value: DateLike, calendar: str) -> DateLike:
    """Rebuild one date in HydroForge's canonical calendar representation."""
    _require_date(value, label="calendar value")
    calendar = canonical_calendar(calendar)
    components = (
        value.year, value.month, value.day,
        value.hour, value.minute, value.second, value.microsecond,
    )
    if calendar == "standard":
        return datetime(*components)
    try:
        date_type = _CFTIME_DATETIME_TYPES[calendar]
    except KeyError as error:
        raise ValueError(f"unsupported simulation calendar {calendar!r}") from error
    return date_type(*components)


def date_calendar(value: Any) -> str | None:
    calendar = getattr(value, "calendar", None)
    if calendar is not None:
        return canonical_calendar(calendar)
    if isinstance(value, datetime):
        return "standard"
    return None


def require_calendar(value: Any, expected: str, *, label: str) -> None:
    observed = date_calendar(value)
    expected = canonical_calendar(expected)
    if observed is not None and observed != expected:
        raise ValueError(
            f"{label} uses calendar {observed!r}, expected {expected!r}"
        )


def _require_date(value: Any, *, label: str) -> None:
    if not isinstance(value, (datetime, cftime.datetime)):
        raise TypeError(f"{label} must be a datetime value")
    if isinstance(value, datetime) and value.tzinfo is not None:
        raise ValueError(
            f"{label} must be timezone-naive; simulation calendars cannot "
            "mix wall-clock offsets with calendar arithmetic"
        )


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """One half-open model interval ``[start, end)``."""

    index: int
    start: DateLike
    end: DateLike
    source_start: DateLike | None = None
    source_end: DateLike | None = None
    phase: SimulationPhase = "main"
    spinup_cycle: int | None = None
    source_index: int = 0
    reuse_index: int = 0
    reuse_count: int = 1

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("simulation step index must be a non-negative int")
        _require_date(self.start, label="simulation step start")
        _require_date(self.end, label="simulation step end")
        start_calendar = date_calendar(self.start)
        end_calendar = date_calendar(self.end)
        if start_calendar != end_calendar:
            raise ValueError(
                "simulation step bounds use different calendars: "
                f"{start_calendar!r} and {end_calendar!r}"
            )
        if type(self.start) is not type(self.end):
            raise TypeError(
                "simulation step bounds must use the same datetime "
                "representation"
            )
        if self.end <= self.start:
            raise ValueError("simulation step must have positive duration")
        source_start = self.start if self.source_start is None else self.source_start
        source_end = self.end if self.source_end is None else self.source_end
        _require_date(source_start, label="simulation step source start")
        _require_date(source_end, label="simulation step source end")
        if type(source_start) is not type(source_end):
            raise TypeError(
                "simulation step source bounds must use the same datetime "
                "representation"
            )
        if date_calendar(source_start) != start_calendar:
            raise ValueError(
                "simulation step execution and source bounds use different "
                "calendars"
            )
        if source_end <= source_start:
            raise ValueError("simulation step source interval must be positive")
        if self.phase not in {"spinup", "main"}:
            raise ValueError("simulation step phase must be 'spinup' or 'main'")
        if self.phase == "main" and self.spinup_cycle is not None:
            raise ValueError("main simulation steps cannot have a spinup cycle")
        if self.phase == "spinup" and (
            type(self.spinup_cycle) is not int or self.spinup_cycle < 0
        ):
            raise ValueError(
                "spinup simulation steps require a non-negative cycle index"
            )
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("simulation source index must be a non-negative int")
        if type(self.reuse_count) is not int or self.reuse_count < 1:
            raise ValueError("simulation reuse count must be a positive int")
        if (
            type(self.reuse_index) is not int
            or not 0 <= self.reuse_index < self.reuse_count
        ):
            raise ValueError(
                "simulation reuse index must be in [0, reuse_count)"
            )
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "source_end", source_end)

    @property
    def is_spin_up(self) -> bool:
        return self.phase == "spinup"


@dataclass(frozen=True, slots=True)
class SpinupSchedule:
    """A half-open source interval replayed before the main simulation."""

    source_start: DateLike
    source_end: DateLike
    cycles: int = 1

    def __post_init__(self) -> None:
        _require_date(self.source_start, label="spinup source start")
        _require_date(self.source_end, label="spinup source end")
        if type(self.source_start) is not type(self.source_end):
            raise TypeError(
                "spinup source bounds must use the same datetime representation"
            )
        if date_calendar(self.source_start) != date_calendar(self.source_end):
            raise ValueError("spinup source bounds use different calendars")
        if self.source_end <= self.source_start:
            raise ValueError("spinup source end must be after its start")
        if type(self.cycles) is not int or self.cycles < 1:
            raise ValueError("spinup cycles must be an exact positive int")


@dataclass(frozen=True, slots=True)
class SimulationSchedule:
    """Runtime-owned model call schedule with optional forcing spinup."""

    calendar: str
    _regular_start: DateLike | None = None
    _regular_end: DateLike | None = None
    _regular_step: timedelta | None = None
    _source_interval: timedelta | None = None
    spinup: SpinupSchedule | None = None
    _explicit_steps: tuple[SimulationStep, ...] = ()

    def __post_init__(self) -> None:
        calendar = canonical_calendar(self.calendar)
        object.__setattr__(self, "calendar", calendar)
        regular_fields = (
            self._regular_start, self._regular_end, self._regular_step,
        )
        present = tuple(value is not None for value in regular_fields)
        if any(present) and not all(present):
            raise ValueError("regular schedule requires start, end, and cadence")
        regular = all(present)
        if regular and self._explicit_steps:
            raise ValueError("schedule cannot be both regular and explicit")
        if self.spinup is not None and not isinstance(
            self.spinup, SpinupSchedule,
        ):
            raise TypeError("schedule spinup must be a SpinupSchedule or None")
        if self.spinup is not None and not regular:
            raise ValueError("spinup is currently supported only by regular schedules")
        if regular:
            regular_start = cast(DateLike, self._regular_start)
            regular_end = cast(DateLike, self._regular_end)
            regular_step = cast(timedelta, self._regular_step)
            _require_date(regular_start, label="schedule start")
            _require_date(regular_end, label="schedule end")
            if type(regular_start) is not type(regular_end):
                raise TypeError(
                    "schedule bounds must use the same datetime representation"
                )
            require_calendar(
                regular_start, calendar, label="schedule start",
            )
            require_calendar(regular_end, calendar, label="schedule end")
            if regular_end <= regular_start:
                raise ValueError("schedule end must be after start")
            if type(regular_step) is not timedelta:
                raise TypeError("simulation step must be a timedelta")
            if timedelta_microseconds(
                regular_step, label="simulation step",
            ) <= 0:
                raise ValueError("simulation step must be positive")
            source_interval = (
                regular_step
                if self._source_interval is None else self._source_interval
            )
            if type(source_interval) is not timedelta:
                raise TypeError("source interval must be a timedelta")
            if timedelta_microseconds(
                source_interval, label="source interval",
            ) <= 0:
                raise ValueError("source interval must be positive")
            reuse_count = timedelta_quotient(
                source_interval,
                regular_step,
                duration_label="source interval",
                interval_label="simulation step",
            )
            if reuse_count < 1:
                raise ValueError(
                    "source interval must not be shorter than simulation step"
                )
            object.__setattr__(self, "_source_interval", source_interval)
            timedelta_quotient(
                regular_end - regular_start,
                source_interval,
                duration_label="main simulation duration",
                interval_label="source interval",
            )
            if self.spinup is not None:
                require_calendar(
                    self.spinup.source_start, calendar,
                    label="spinup source start",
                )
                require_calendar(
                    self.spinup.source_end, calendar,
                    label="spinup source end",
                )
                if type(self.spinup.source_start) is not type(regular_start):
                    raise TypeError(
                        "spinup source bounds and main schedule must use the "
                        "same datetime representation"
                    )
                timedelta_quotient(
                    self.spinup.source_end - self.spinup.source_start,
                    source_interval,
                    duration_label="spinup source duration",
                    interval_label="source interval",
                )
            return
        if not self._explicit_steps:
            raise ValueError("schedule must contain model intervals")
        if not isinstance(self._explicit_steps, tuple) or any(
            not isinstance(step, SimulationStep)
            for step in self._explicit_steps
        ):
            raise TypeError(
                "explicit schedule steps must be a tuple of SimulationStep values"
            )
        previous_end: DateLike | None = None
        bound_type: type[Any] | None = None
        for expected_index, step in enumerate(self._explicit_steps):
            if step.index != expected_index:
                raise ValueError("simulation step indices must be contiguous")
            require_calendar(step.start, calendar, label="simulation step start")
            require_calendar(step.end, calendar, label="simulation step end")
            if step.end <= step.start:
                raise ValueError("simulation steps must have positive duration")
            if previous_end is not None and step.start != previous_end:
                raise ValueError(
                    "simulation steps must be contiguous without gaps or overlap"
                )
            if bound_type is None:
                bound_type = type(step.start)
            elif type(step.start) is not bound_type:
                raise TypeError(
                    "all simulation steps must use one datetime representation"
                )
            previous_end = step.end

    @classmethod
    def regular(
        cls,
        *,
        start: DateLike,
        end: DateLike,
        step: timedelta,
        source_interval: timedelta | None = None,
        calendar: str | None = None,
        spinup: SpinupSchedule | None = None,
    ) -> SimulationSchedule:
        resolved_calendar = canonical_calendar(
            calendar or date_calendar(start) or "standard"
        )
        return cls(
            resolved_calendar,
            _regular_start=start,
            _regular_end=end,
            _regular_step=step,
            _source_interval=source_interval,
            spinup=spinup,
        )

    @classmethod
    def from_contract(
        cls,
        contract: DatasetTemporalContract,
        *,
        step: timedelta,
    ) -> SimulationSchedule:
        """Subdivide each source support into exact model calls."""
        return cls.regular(
            start=contract.start,
            end=contract.end,
            step=step,
            source_interval=contract.interval,
            calendar=contract.calendar,
            spinup=contract.spinup,
        )

    @classmethod
    def explicit(
        cls,
        intervals: Iterable[tuple[DateLike, DateLike]],
        *,
        calendar: str | None = None,
    ) -> SimulationSchedule:
        pairs = tuple(intervals)
        if not pairs:
            raise ValueError("explicit schedule must contain at least one interval")
        resolved_calendar = canonical_calendar(
            calendar or date_calendar(pairs[0][0]) or "standard"
        )
        return cls(
            resolved_calendar,
            _explicit_steps=tuple(
                SimulationStep(index, start, end)
                for index, (start, end) in enumerate(pairs)
            ),
        )

    @property
    def is_regular(self) -> bool:
        return self._regular_start is not None

    @property
    def cadence(self) -> timedelta | None:
        """Fixed model cadence, or ``None`` for an explicit schedule."""
        return self._regular_step

    @property
    def source_interval(self) -> timedelta | None:
        """Physical support represented by one source sample."""
        return self._source_interval if self.is_regular else None

    @property
    def reuse_count(self) -> int:
        """Number of model calls made from each source sample."""
        if not self.is_regular:
            raise ValueError("explicit schedules have no uniform reuse count")
        return timedelta_quotient(
            cast(timedelta, self._source_interval),
            cast(timedelta, self._regular_step),
            duration_label="source interval",
            interval_label="simulation step",
        )

    @property
    def start(self) -> DateLike:
        """Start of the main simulation period."""
        if self.is_regular:
            return cast(DateLike, self._regular_start)
        return self._explicit_steps[0].start

    @property
    def end(self) -> DateLike:
        """End of the main simulation period and complete execution."""
        if self.is_regular:
            return cast(DateLike, self._regular_end)
        return self._explicit_steps[-1].end

    @property
    def execution_start(self) -> DateLike:
        """Monotonic model-clock start, including all spinup calls."""
        if not self.is_regular or self.spinup is None:
            return self.start
        cadence = cast(timedelta, self._regular_step)
        return self.start - cadence * self.num_spinup_steps

    @property
    def num_spinup_steps(self) -> int:
        if self.spinup is None:
            return 0
        source_samples = timedelta_quotient(
            self.spinup.source_end - self.spinup.source_start,
            cast(timedelta, self._source_interval),
            duration_label="spinup source duration",
            interval_label="source interval",
        )
        return source_samples * self.reuse_count * self.spinup.cycles

    @property
    def num_main_steps(self) -> int:
        if not self.is_regular:
            return len(self._explicit_steps)
        source_samples = timedelta_quotient(
            self.end - self.start,
            cast(timedelta, self._source_interval),
            duration_label="main simulation duration",
            interval_label="source interval",
        )
        return source_samples * self.reuse_count

    def step_at(self, index: int) -> SimulationStep:
        if type(index) is not int:
            raise TypeError("simulation step index must be an exact int")
        if not 0 <= index < len(self):
            raise IndexError(index)
        if not self.is_regular:
            return self._explicit_steps[index]
        cadence = cast(timedelta, self._regular_step)
        start = self.execution_start + cadence * index
        spinup_steps = self.num_spinup_steps
        reuse_count = self.reuse_count
        source_interval = cast(timedelta, self._source_interval)
        if index < spinup_steps:
            spinup = cast(SpinupSchedule, self.spinup)
            per_cycle = spinup_steps // spinup.cycles
            cycle_model_index = index % per_cycle
            source_index, reuse_index = divmod(
                cycle_model_index, reuse_count,
            )
            source_start = spinup.source_start + source_interval * source_index
            return SimulationStep(
                index, start, start + cadence,
                source_start=source_start,
                source_end=source_start + source_interval,
                phase="spinup",
                spinup_cycle=index // per_cycle,
                source_index=source_index,
                reuse_index=reuse_index,
                reuse_count=reuse_count,
            )
        main_model_index = index - spinup_steps
        source_index, reuse_index = divmod(main_model_index, reuse_count)
        source_start = self.start + source_interval * source_index
        return SimulationStep(
            index, start, start + cadence,
            source_start=source_start,
            source_end=source_start + source_interval,
            source_index=source_index,
            reuse_index=reuse_index,
            reuse_count=reuse_count,
        )

    def index_at(self, start: DateLike) -> int:
        _require_date(start, label="model current_time")
        require_calendar(start, self.calendar, label="model current_time")
        if type(start) is not type(self.start):
            raise TypeError(
                "model current_time and schedule must use the same datetime "
                "representation"
            )
        if not self.is_regular:
            lower = 0
            upper = len(self._explicit_steps)
            while lower < upper:
                middle = (lower + upper) // 2
                if self._explicit_steps[middle].start < start:
                    lower = middle + 1
                else:
                    upper = middle
            if (
                lower < len(self._explicit_steps)
                and self._explicit_steps[lower].start == start
            ):
                return lower
            raise KeyError(start)
        regular_start = self.execution_start
        regular_step = cast(timedelta, self._regular_step)
        offset = timedelta_microseconds(
            start - regular_start, label="model schedule offset",
        )
        cadence = timedelta_microseconds(
            regular_step, label="simulation step",
        )
        index, remainder = divmod(offset, cadence)
        if index < 0 or index >= len(self) or remainder != 0:
            raise KeyError(start)
        return index

    def summary(self) -> str:
        """Return a stable, human-readable view of the compiled schedule."""

        if self.is_regular:
            schedule_type = "regular"
            cadence = str(self.cadence)
            source_interval = str(self.source_interval)
            reuse_count = self.reuse_count
            reuse_unit = "step" if reuse_count == 1 else "steps"
            source_reuse = (
                f"{reuse_count} model {reuse_unit}/source sample"
            )
        else:
            schedule_type = "explicit"
            cadence = source_interval = source_reuse = "per-step"

        if self.spinup is None:
            spinup_source = "none"
            spinup_cycles = 0
        else:
            spinup_source = (
                f"[{self.spinup.source_start}, {self.spinup.source_end})"
            )
            spinup_cycles = self.spinup.cycles

        fields = (
            ("Schedule type", schedule_type),
            ("Calendar", self.calendar),
            (
                "Execution period",
                f"[{self.execution_start}, {self.end})",
            ),
            ("Main period", f"[{self.start}, {self.end})"),
            ("Model cadence", cadence),
            ("Source interval", source_interval),
            ("Source reuse", source_reuse),
            ("Spinup source", spinup_source),
            ("Spinup cycles", spinup_cycles),
            ("Spinup steps", self.num_spinup_steps),
            ("Main steps", self.num_main_steps),
            ("Total steps", len(self)),
        )
        return "\n".join(f"{label:<17}: {value}" for label, value in fields)

    def __iter__(self):
        if not self.is_regular:
            yield from self._explicit_steps
            return
        for index in range(len(self)):
            yield self.step_at(index)

    def __len__(self) -> int:
        if not self.is_regular:
            return len(self._explicit_steps)
        return self.num_spinup_steps + self.num_main_steps


@dataclass(frozen=True, slots=True)
class DatasetTemporalContract:
    """Source sample support without any model-execution assumptions."""

    calendar: str
    start: DateLike
    interval: timedelta
    count: int
    spinup: SpinupSchedule | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "calendar", canonical_calendar(self.calendar))
        _require_date(self.start, label="dataset start")
        require_calendar(self.start, self.calendar, label="dataset start")
        if type(self.interval) is not timedelta:
            raise TypeError("dataset sample interval must be a timedelta")
        if timedelta_microseconds(
            self.interval, label="dataset sample interval",
        ) <= 0:
            raise ValueError("dataset sample interval must be positive")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("dataset temporal contract must contain samples")
        if self.spinup is not None:
            if not isinstance(self.spinup, SpinupSchedule):
                raise TypeError("dataset spinup must be a SpinupSchedule or None")
            require_calendar(
                self.spinup.source_start, self.calendar,
                label="dataset spinup source start",
            )
            require_calendar(
                self.spinup.source_end, self.calendar,
                label="dataset spinup source end",
            )
            if type(self.spinup.source_start) is not type(self.start):
                raise TypeError(
                    "dataset spinup and main bounds must use the same datetime "
                    "representation"
                )
            timedelta_quotient(
                self.spinup.source_end - self.spinup.source_start,
                self.interval,
                duration_label="dataset spinup source duration",
                interval_label="dataset sample interval",
            )

    def support(self, index: int) -> tuple[DateLike, DateLike]:
        if type(index) is not int:
            raise TypeError("dataset sample index must be an exact int")
        if not 0 <= index < self.count:
            raise IndexError(index)
        start = self.start + self.interval * index
        return start, start + self.interval

    @property
    def end(self) -> DateLike:
        return self.start + self.interval * self.count

    @classmethod
    def combine(
        cls, contracts: Mapping[str, DatasetTemporalContract],
    ) -> DatasetTemporalContract:
        if not contracts:
            raise ValueError("at least one dataset temporal contract is required")
        invalid_names = [
            name for name in contracts
            if not isinstance(name, str) or not name
        ]
        if invalid_names:
            raise ValueError(
                "dataset timeline names must be non-empty strings: "
                f"{invalid_names!r}"
            )
        invalid_contracts = {
            name: type(contract).__name__
            for name, contract in contracts.items()
            if not isinstance(contract, DatasetTemporalContract)
        }
        if invalid_contracts:
            raise TypeError(
                "dataset timelines must be DatasetTemporalContract values: "
                f"{invalid_contracts}"
            )
        name, reference = next(iter(contracts.items()))
        for other_name, other in tuple(contracts.items())[1:]:
            if type(other.start) is not type(reference.start):
                raise TypeError(
                    f"dataset timelines {name!r} and {other_name!r} use "
                    "different datetime representations"
                )
            for attribute in (
                "calendar", "start", "interval", "count", "spinup",
            ):
                if getattr(other, attribute) != getattr(reference, attribute):
                    raise ValueError(
                        f"dataset timelines {name!r} and {other_name!r} "
                        f"differ in {attribute}"
                    )
        return reference


@dataclass(frozen=True, slots=True)
class EveryStep:
    """Every model call is a complete inner statistics window."""


@dataclass(frozen=True, slots=True)
class CalendarWindow:
    period: CalendarPeriod
    start_month: int = 1
    start_day: int = 1

    def __post_init__(self) -> None:
        if self.period not in {"day", "month", "year"}:
            raise ValueError(
                "calendar window period must be 'day', 'month', or 'year'"
            )
        if not 1 <= self.start_month <= 12:
            raise ValueError("start_month must be in 1..12")
        if not 1 <= self.start_day <= 31:
            raise ValueError("start_day must be in 1..31")
        if self.period != "year" and (
            self.start_month != 1 or self.start_day != 1
        ):
            raise ValueError("custom origins are supported only for year windows")


@dataclass(frozen=True, slots=True)
class ExplicitWindow:
    name: str
    start: DateLike
    end: DateLike

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("explicit window name must be a non-empty string")
        _require_date(self.start, label=f"explicit window {self.name!r} start")
        _require_date(self.end, label=f"explicit window {self.name!r} end")
        if date_calendar(self.start) != date_calendar(self.end):
            raise ValueError("explicit window bounds use different calendars")
        if type(self.start) is not type(self.end):
            raise TypeError(
                "explicit window bounds must use the same datetime representation"
            )
        if self.end <= self.start:
            raise ValueError(f"explicit window {self.name!r} is empty")


@dataclass(frozen=True, slots=True)
class ExplicitWindows:
    windows: tuple[ExplicitWindow, ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("explicit windows must not be empty")
        if not isinstance(self.windows, tuple) or any(
            not isinstance(window, ExplicitWindow) for window in self.windows
        ):
            raise TypeError(
                "explicit windows must be a tuple of ExplicitWindow values"
            )
        names = tuple(window.name for window in self.windows)
        if len(set(names)) != len(names):
            raise ValueError("explicit statistics window names must be unique")
        previous_end: DateLike | None = None
        bound_type = type(self.windows[0].start)
        calendar = date_calendar(self.windows[0].start)
        for window in self.windows:
            if type(window.start) is not bound_type:
                raise TypeError(
                    "explicit statistics windows must use one datetime "
                    "representation"
                )
            if date_calendar(window.start) != calendar:
                raise ValueError(
                    "explicit statistics windows must use one calendar"
                )
            if previous_end is not None and window.start < previous_end:
                raise ValueError("explicit statistics windows must not overlap")
            previous_end = window.end


WindowRule = EveryStep | CalendarWindow | ExplicitWindows


@dataclass(frozen=True, slots=True)
class StatisticsPlan:
    """Declarative statistics-window rules independent of execution time."""

    inner: WindowRule
    outer: WindowRule | None = None
    partial_period: Literal["close", "drop"] = "close"

    def __post_init__(self) -> None:
        valid_rules = (EveryStep, CalendarWindow, ExplicitWindows)
        if not isinstance(self.inner, valid_rules):
            raise TypeError("statistics inner must be a WindowRule")
        if self.outer is not None and not isinstance(self.outer, valid_rules):
            raise TypeError("statistics outer must be a WindowRule or None")
        if self.partial_period not in {"close", "drop"}:
            raise ValueError("partial_period must be 'close' or 'drop'")
        if self.outer is None and not isinstance(self.inner, EveryStep):
            object.__setattr__(self, "outer", self.inner)

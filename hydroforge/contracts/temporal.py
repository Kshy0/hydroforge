"""Immutable temporal contracts shared by drivers, datasets, and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, TypeAlias, cast

import cftime


DateLike: TypeAlias = datetime | cftime.datetime
CalendarPeriod = Literal["day", "month", "year"]
ForcingSemantics = Literal["mean_rate", "accumulated", "instantaneous"]
ResamplingMethod = Literal["hold", "conservative", "linear"]


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


def date_signature(value: DateLike) -> dict[str, Any]:
    """Return the one stable serialized identity for a calendar instant."""

    _require_date(value, label="temporal value")
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "calendar": date_calendar(value),
        "parts": [
            value.year, value.month, value.day,
            value.hour, value.minute, value.second, value.microsecond,
        ],
        "has_year_zero": getattr(value, "has_year_zero", None),
    }


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """One half-open model interval ``[start, end)``."""

    index: int
    start: DateLike
    end: DateLike

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

    @property
    def duration_seconds(self) -> float:
        return float((self.end - self.start).total_seconds())


@dataclass(frozen=True, slots=True)
class SimulationSchedule:
    """Driver-owned model call schedule, independent of forcing cadence."""

    calendar: str
    _regular_start: DateLike | None = None
    _regular_end: DateLike | None = None
    _regular_step: timedelta | None = None
    _explicit_steps: tuple[SimulationStep, ...] = ()
    _explicit_index: Mapping[DateLike, int] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

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
            object.__setattr__(self, "_explicit_index", MappingProxyType({}))
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
            if previous_end is not None and step.start < previous_end:
                raise ValueError("simulation steps must not overlap or move backward")
            if bound_type is None:
                bound_type = type(step.start)
            elif type(step.start) is not bound_type:
                raise TypeError(
                    "all simulation steps must use one datetime representation"
                )
            previous_end = step.end
        object.__setattr__(self, "_explicit_index", MappingProxyType({
            step.start: step.index for step in self._explicit_steps
        }))

    @classmethod
    def regular(
        cls,
        *,
        start: DateLike,
        end: DateLike,
        step: timedelta,
        calendar: str | None = None,
    ) -> SimulationSchedule:
        if type(step) is not timedelta:
            raise TypeError("simulation step must be a timedelta")
        if timedelta_microseconds(step, label="simulation step") <= 0:
            raise ValueError("simulation step must be positive")
        resolved_calendar = canonical_calendar(
            calendar or date_calendar(start) or "standard"
        )
        require_calendar(start, resolved_calendar, label="schedule start")
        require_calendar(end, resolved_calendar, label="schedule end")
        if end <= start:
            raise ValueError("schedule end must be after start")
        return cls(
            resolved_calendar,
            _regular_start=start,
            _regular_end=end,
            _regular_step=step,
        )

    @classmethod
    def from_contract(
        cls,
        contract: DatasetTemporalContract,
        *,
        step: timedelta,
    ) -> SimulationSchedule:
        """Cover every source support interval at a chosen model cadence."""
        return cls.regular(
            start=contract.start,
            end=contract.end,
            step=step,
            calendar=contract.calendar,
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
    def start(self) -> DateLike:
        if self.is_regular:
            return cast(DateLike, self._regular_start)
        return self._explicit_steps[0].start

    @property
    def end(self) -> DateLike:
        if self.is_regular:
            return cast(DateLike, self._regular_end)
        return self._explicit_steps[-1].end

    @property
    def fingerprint(self) -> str:
        """Stable identity shared by statistics, forcing and checkpoints."""

        if self.is_regular:
            definition: dict[str, Any] = {
                "kind": "regular",
                "calendar": self.calendar,
                "start": date_signature(self.start),
                "end": date_signature(self.end),
                "step_microseconds": timedelta_microseconds(
                    cast(timedelta, self.cadence), label="simulation step",
                ),
            }
        else:
            definition = {
                "kind": "explicit",
                "calendar": self.calendar,
                "steps": [
                    {
                        "start": date_signature(step.start),
                        "end": date_signature(step.end),
                    }
                    for step in self._explicit_steps
                ],
            }
        encoded = json.dumps(
            definition, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def steps(self) -> tuple[SimulationStep, ...]:
        """Materialize intervals only for inspection and small control tasks."""
        return tuple(self)

    def step_at(self, index: int) -> SimulationStep:
        if type(index) is not int:
            raise TypeError("simulation step index must be an exact int")
        if not 0 <= index < len(self):
            raise IndexError(index)
        if not self.is_regular:
            return self._explicit_steps[index]
        cadence = cast(timedelta, self._regular_step)
        regular_end = cast(DateLike, self._regular_end)
        start = cast(DateLike, self._regular_start) + cadence * index
        return SimulationStep(
            index, start, min(start + cadence, regular_end),
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
            try:
                return self._explicit_index[start]
            except KeyError:
                raise KeyError(start) from None
        regular_start = cast(DateLike, self._regular_start)
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

    def __iter__(self):
        if not self.is_regular:
            yield from self._explicit_steps
            return
        for index in range(len(self)):
            yield self.step_at(index)

    def __len__(self) -> int:
        if not self.is_regular:
            return len(self._explicit_steps)
        regular_start = cast(DateLike, self._regular_start)
        regular_end = cast(DateLike, self._regular_end)
        regular_step = cast(timedelta, self._regular_step)
        duration = timedelta_microseconds(
            regular_end - regular_start, label="simulation duration",
        )
        cadence = timedelta_microseconds(
            regular_step, label="simulation step",
        )
        return (duration + cadence - 1) // cadence


@dataclass(frozen=True, slots=True)
class DatasetTemporalContract:
    """Source sample support without any model-execution assumptions."""

    calendar: str
    start: DateLike
    interval: timedelta
    count: int
    timestamp_position: Literal["start"] = "start"

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
        if self.timestamp_position != "start":
            raise ValueError("only start-stamped source support is implemented")

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
                "calendar", "start", "interval", "count",
                "timestamp_position",
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


def window_rule_signature(rule: WindowRule | None) -> dict[str, Any] | None:
    """Return a stable, explicit identity for one statistics window rule."""

    if rule is None:
        return None
    if isinstance(rule, EveryStep):
        return {"kind": "every_step"}
    if isinstance(rule, CalendarWindow):
        return {
            "kind": "calendar",
            "period": rule.period,
            "start_month": rule.start_month,
            "start_day": rule.start_day,
        }
    if isinstance(rule, ExplicitWindows):
        return {
            "kind": "explicit",
            "windows": [
                {
                    "name": window.name,
                    "start": date_signature(window.start),
                    "end": date_signature(window.end),
                }
                for window in rule.windows
            ],
        }
    raise TypeError(f"statistics window rule has unsupported type {type(rule)!r}")


@dataclass(frozen=True, slots=True)
class StatisticsPlan:
    """Declarative statistics windows over a model execution schedule."""

    schedule: SimulationSchedule
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
        for rule in (self.inner, self.outer):
            if isinstance(rule, ExplicitWindows):
                for window in rule.windows:
                    require_calendar(
                        window.start, self.schedule.calendar,
                        label=f"explicit window {window.name!r} start",
                    )
                    require_calendar(
                        window.end, self.schedule.calendar,
                        label=f"explicit window {window.name!r} end",
                    )
                    if type(window.start) is not type(self.schedule.start):
                        raise TypeError(
                            f"explicit window {window.name!r} and simulation "
                            "schedule must use the same datetime representation"
                        )
            if isinstance(rule, CalendarWindow) and rule.period == "year":
                try:
                    cftime.datetime(
                        2001, rule.start_month, rule.start_day,
                        calendar=self.schedule.calendar,
                    )
                except ValueError as exc:
                    raise ValueError(
                        "annual statistics origin must exist in every year of "
                        f"calendar {self.schedule.calendar!r}"
                    ) from exc


@dataclass(frozen=True, slots=True)
class StatisticsFlags:
    first: bool
    last: bool
    outer_first: bool
    outer_last: bool

    @property
    def bits(self) -> int:
        return (
            int(self.first) | (int(self.last) << 1)
            | (int(self.outer_first) << 2) | (int(self.outer_last) << 3)
        )

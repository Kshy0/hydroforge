"""Immutable temporal contracts shared by drivers, datasets, and models."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal, Mapping, Self, TypeAlias, cast

import cftime
from pydantic import (
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.contracts.validation import (
    HydroForgeModel,
)


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

    del label
    return (
        value.days * _SECONDS_PER_DAY + value.seconds
    ) * _MICROSECONDS_PER_SECOND + value.microseconds


def _timedelta_quotient_trusted(
    duration: timedelta,
    interval: timedelta,
    *,
    duration_label: str = "duration",
    interval_label: str = "interval",
) -> int:
    """Compute an exact ratio from already validated temporal values."""

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


class _TimedeltaQuotientRequest(HydroForgeModel):
    """One public exact-duration quotient request."""

    duration: timedelta
    interval: timedelta
    duration_label: str = "duration"
    interval_label: str = "interval"

    _quotient: int = PrivateAttr()

    @field_validator("duration_label", "interval_label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        if not value:
            raise ValueError("duration labels must be non-empty strings")
        return value

    @model_validator(mode="after")
    def _resolve(self):
        self._quotient = _timedelta_quotient_trusted(
            self.duration,
            self.interval,
            duration_label=self.duration_label,
            interval_label=self.interval_label,
        )
        return self

    @property
    def quotient(self) -> int:
        return self._quotient


def timedelta_quotient(
    duration: timedelta,
    interval: timedelta,
    *,
    duration_label: str = "duration",
    interval_label: str = "interval",
) -> int:
    """Validate and return an exact integral duration/interval ratio."""

    return _TimedeltaQuotientRequest(
        duration=duration,
        interval=interval,
        duration_label=duration_label,
        interval_label=interval_label,
    ).quotient


def canonical_calendar(calendar: str) -> str:
    """Normalize only aliases that cftime defines as equivalent."""
    if not isinstance(calendar, str) or not calendar.strip():
        raise ValueError("calendar must be a non-empty string")
    normalized = calendar.strip().lower()
    return _CALENDAR_ALIASES.get(normalized, normalized)


def convert_calendar_date(value: DateLike, calendar: str) -> DateLike:
    """Rebuild one date in HydroForge's canonical calendar representation."""
    _require_date(value, label="calendar value")
    calendar = canonical_calendar(calendar)
    components = (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
    )
    if calendar == "standard":
        return datetime(*components)
    try:
        date_type = _CFTIME_DATETIME_TYPES[calendar]
    except KeyError as error:
        raise ValueError(f"unsupported simulation calendar {calendar!r}") from error
    try:
        return date_type(*components)
    except ValueError as error:
        raise ValueError(
            f"date {value!r} cannot be represented in calendar {calendar!r}"
        ) from error


def normalize_calendar_dates(
    values: Mapping[str, DateLike | None],
    *,
    calendar: str | None,
    preserve_cftime_declaration: bool = False,
) -> tuple[str, dict[str, DateLike | None], bool]:
    """Infer one calendar and bind plain ``datetime`` values to it.

    Python ``datetime`` carries no CF-calendar declaration and is therefore
    treated as a civil date whose intended calendar may be supplied by an
    explicit ``calendar`` argument, another cftime bound, or storage metadata.
    A cftime value does carry a declaration; conflicting cftime calendars are
    never reinterpreted.

    The returned boolean is true only when neither the caller nor any cftime
    value selected the calendar, so a storage-backed owner may still replace
    the provisional ``standard`` default after inspecting its time axis.
    Nested declarations may request preservation of a standard-calendar
    cftime representation until an enclosing schedule or file binds it.
    """

    observed: dict[str, list[str]] = {}
    for label, value in values.items():
        if value is None:
            continue
        _require_date(value, label=label)
        if isinstance(value, cftime.datetime):
            observed.setdefault(date_calendar(value), []).append(label)
    if len(observed) > 1:
        detail = ", ".join(
            f"{name!r} from {labels}" for name, labels in sorted(observed.items())
        )
        raise ValueError(f"datetime values use conflicting calendars: {detail}")

    configured = None if calendar is None else canonical_calendar(calendar)
    inferred = next(iter(observed), None)
    if configured is not None and inferred is not None and configured != inferred:
        labels = observed[inferred]
        raise ValueError(
            f"calendar {configured!r} conflicts with cftime calendar "
            f"{inferred!r} from {labels}"
        )
    resolved = configured or inferred or "standard"
    standard_template = next(
        (
            value
            for value in values.values()
            if (
                isinstance(value, cftime.datetime)
                and date_calendar(value) == "standard"
            )
        ),
        None,
    )
    normalized: dict[str, DateLike | None] = {}
    for label, value in values.items():
        if value is None:
            normalized[label] = None
            continue
        try:
            if (
                preserve_cftime_declaration
                and resolved == "standard"
                and standard_template is not None
            ):
                normalized[label] = cftime.DatetimeGregorian(
                    value.year,
                    value.month,
                    value.day,
                    value.hour,
                    value.minute,
                    value.second,
                    value.microsecond,
                    has_year_zero=standard_template.has_year_zero,
                )
            else:
                normalized[label] = convert_calendar_date(value, resolved)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{label} cannot be represented in calendar {resolved!r}: "
                f"{value!r}"
            ) from error
    return resolved, normalized, configured is None and inferred is None


def date_calendar(value: DateLike) -> str:
    calendar = getattr(value, "calendar", None)
    if calendar is not None:
        return canonical_calendar(calendar)
    if isinstance(value, datetime):
        return "standard"
    raise ValueError("calendar value must be datetime or cftime.datetime")


def require_calendar(value: DateLike, expected: str, *, label: str) -> None:
    observed = date_calendar(value)
    expected = canonical_calendar(expected)
    if observed != expected:
        raise ValueError(f"{label} uses calendar {observed!r}, expected {expected!r}")


def _require_date(value: Any, *, label: str) -> None:
    if not isinstance(value, (datetime, cftime.datetime)):
        raise ValueError(f"{label} must be a datetime value")
    if isinstance(value, datetime) and value.tzinfo is not None:
        raise ValueError(
            f"{label} must be timezone-naive; simulation calendars cannot "
            "mix wall-clock offsets with calendar arithmetic"
        )


class SimulationStep(HydroForgeModel):
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

    @model_validator(mode="after")
    def _validate_step(self) -> Self:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("simulation step index must be a non-negative int")
        source_start = self.start if self.source_start is None else self.source_start
        source_end = self.end if self.source_end is None else self.source_end
        date_values = {
            "simulation step start": self.start,
            "simulation step end": self.end,
            "simulation step source start": source_start,
            "simulation step source end": source_end,
        }
        _calendar, normalized, _defaulted = normalize_calendar_dates(
            date_values,
            calendar=None,
            preserve_cftime_declaration=True,
        )
        start = cast(DateLike, normalized["simulation step start"])
        end = cast(DateLike, normalized["simulation step end"])
        source_start = cast(
            DateLike, normalized["simulation step source start"],
        )
        source_end = cast(DateLike, normalized["simulation step source end"])
        if end <= start:
            raise ValueError("simulation step must have positive duration")
        if source_end <= source_start:
            raise ValueError("simulation step source interval must be positive")
        if type(self.phase) is not str or self.phase not in {"spinup", "main"}:
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
            raise ValueError("simulation reuse index must be in [0, reuse_count)")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "source_end", source_end)
        return self

    @classmethod
    def _from_schedule_trusted(
        cls,
        *,
        index: int,
        start: DateLike,
        end: DateLike,
        source_start: DateLike,
        source_end: DateLike,
        phase: SimulationPhase = "main",
        spinup_cycle: int | None = None,
        source_index: int,
        reuse_index: int,
        reuse_count: int,
    ) -> "SimulationStep":
        """Materialize values already proved by a validated schedule."""

        return cls.model_construct(
            index=index,
            start=start,
            end=end,
            source_start=source_start,
            source_end=source_end,
            phase=phase,
            spinup_cycle=spinup_cycle,
            source_index=source_index,
            reuse_index=reuse_index,
            reuse_count=reuse_count,
        )

    @property
    def is_spin_up(self) -> bool:
        return self.phase == "spinup"


class SpinupSchedule(HydroForgeModel):
    """A half-open source interval replayed before the main simulation."""

    source_start: DateLike
    source_end: DateLike
    cycles: int = 1

    @model_validator(mode="after")
    def _validate_spinup(self) -> Self:
        date_values = {
            "spinup source start": self.source_start,
            "spinup source end": self.source_end,
        }
        _calendar, normalized, _defaulted = normalize_calendar_dates(
            date_values,
            calendar=None,
            preserve_cftime_declaration=True,
        )
        source_start = cast(DateLike, normalized["spinup source start"])
        source_end = cast(DateLike, normalized["spinup source end"])
        if source_end <= source_start:
            raise ValueError("spinup source end must be after its start")
        if type(self.cycles) is not int or self.cycles < 1:
            raise ValueError("spinup cycles must be an exact positive int")
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "source_end", source_end)
        return self


class SimulationSchedule(HydroForgeModel):
    """Runtime-owned model call schedule with optional forcing spinup."""

    calendar: str | None = None
    regular_start: DateLike | None = None
    regular_end: DateLike | None = None
    regular_step: timedelta | None = None
    source_interval: timedelta | None = None
    spinup: SpinupSchedule | None = None
    explicit_steps: tuple[SimulationStep, ...] = ()
    _compiled_reuse_count: int | None = PrivateAttr(default=None)
    _compiled_spinup_steps: int = PrivateAttr(default=0)
    _compiled_main_steps: int = PrivateAttr(default=0)

    @model_validator(mode="after")
    def _validate_schedule(self) -> Self:
        date_values: dict[str, DateLike | None] = {
            "schedule start": self.regular_start,
            "schedule end": self.regular_end,
        }
        if self.spinup is not None:
            date_values["spinup source start"] = self.spinup.source_start
            date_values["spinup source end"] = self.spinup.source_end
        for index, step in enumerate(self.explicit_steps):
            date_values[f"simulation step {index} start"] = step.start
            date_values[f"simulation step {index} end"] = step.end
            date_values[f"simulation step {index} source start"] = step.source_start
            date_values[f"simulation step {index} source end"] = step.source_end
        calendar, normalized, _defaulted = normalize_calendar_dates(
            date_values,
            calendar=self.calendar,
        )
        object.__setattr__(self, "calendar", calendar)
        regular_fields = (
            self.regular_start,
            self.regular_end,
            self.regular_step,
        )
        present = tuple(value is not None for value in regular_fields)
        if any(present) and not all(present):
            raise ValueError("regular schedule requires start, end, and cadence")
        regular = all(present)
        if regular and self.explicit_steps:
            raise ValueError("schedule cannot be both regular and explicit")
        if self.spinup is not None and not isinstance(
            self.spinup,
            SpinupSchedule,
        ):
            raise ValueError("schedule spinup must be a SpinupSchedule or None")
        if self.spinup is not None and not regular:
            raise ValueError("spinup is currently supported only by regular schedules")
        if regular:
            regular_start = cast(DateLike, normalized["schedule start"])
            regular_end = cast(DateLike, normalized["schedule end"])
            regular_step = cast(timedelta, self.regular_step)
            object.__setattr__(self, "regular_start", regular_start)
            object.__setattr__(self, "regular_end", regular_end)
            if regular_end <= regular_start:
                raise ValueError("schedule end must be after start")
            if type(regular_step) is not timedelta:
                raise ValueError("simulation step must be a timedelta")
            if (
                timedelta_microseconds(
                    regular_step,
                    label="simulation step",
                )
                <= 0
            ):
                raise ValueError("simulation step must be positive")
            source_interval = (
                regular_step if self.source_interval is None else self.source_interval
            )
            if type(source_interval) is not timedelta:
                raise ValueError("source interval must be a timedelta")
            if (
                timedelta_microseconds(
                    source_interval,
                    label="source interval",
                )
                <= 0
            ):
                raise ValueError("source interval must be positive")
            reuse_count = _timedelta_quotient_trusted(
                source_interval,
                regular_step,
                duration_label="source interval",
                interval_label="simulation step",
            )
            if reuse_count < 1:
                raise ValueError(
                    "source interval must not be shorter than simulation step"
                )
            object.__setattr__(self, "source_interval", source_interval)
            main_source_samples = _timedelta_quotient_trusted(
                regular_end - regular_start,
                source_interval,
                duration_label="main simulation duration",
                interval_label="source interval",
            )
            spinup_source_samples = 0
            if self.spinup is not None:
                spinup = SpinupSchedule(
                    source_start=cast(
                        DateLike, normalized["spinup source start"],
                    ),
                    source_end=cast(
                        DateLike, normalized["spinup source end"],
                    ),
                    cycles=self.spinup.cycles,
                )
                object.__setattr__(self, "spinup", spinup)
                spinup_source_samples = _timedelta_quotient_trusted(
                    spinup.source_end - spinup.source_start,
                    source_interval,
                    duration_label="spinup source duration",
                    interval_label="source interval",
                )
            self._compiled_reuse_count = reuse_count
            self._compiled_main_steps = main_source_samples * reuse_count
            self._compiled_spinup_steps = (
                spinup_source_samples
                * reuse_count
                * (1 if self.spinup is None else self.spinup.cycles)
            )
            return self
        if not self.explicit_steps:
            raise ValueError("schedule must contain model intervals")
        if not isinstance(self.explicit_steps, tuple) or any(
            not isinstance(step, SimulationStep) for step in self.explicit_steps
        ):
            raise ValueError(
                "explicit schedule steps must be a tuple of SimulationStep values"
            )
        normalized_steps = []
        for index, step in enumerate(self.explicit_steps):
            normalized_step = SimulationStep(
                index=step.index,
                start=cast(
                    DateLike, normalized[f"simulation step {index} start"],
                ),
                end=cast(
                    DateLike, normalized[f"simulation step {index} end"],
                ),
                source_start=cast(
                    DateLike,
                    normalized[f"simulation step {index} source start"],
                ),
                source_end=cast(
                    DateLike,
                    normalized[f"simulation step {index} source end"],
                ),
                phase=step.phase,
                spinup_cycle=step.spinup_cycle,
                source_index=step.source_index,
                reuse_index=step.reuse_index,
                reuse_count=step.reuse_count,
            )
            normalized_steps.append(normalized_step)
        normalized_steps = tuple(normalized_steps)
        object.__setattr__(self, "explicit_steps", normalized_steps)
        previous_end: DateLike | None = None
        for expected_index, step in enumerate(normalized_steps):
            if step.index != expected_index:
                raise ValueError("simulation step indices must be contiguous")
            if step.phase != "main" or step.spinup_cycle is not None:
                raise ValueError(
                    "explicit schedules cannot contain spinup steps; use a "
                    "regular schedule with SpinupSchedule"
                )
            if step.end <= step.start:
                raise ValueError("simulation steps must have positive duration")
            if previous_end is not None and step.start != previous_end:
                raise ValueError(
                    "simulation steps must be contiguous without gaps or overlap"
                )
            previous_end = step.end
        self._compiled_main_steps = len(self.explicit_steps)
        return self

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
    ) -> Self:
        return cls(
            calendar=calendar,
            regular_start=start,
            regular_end=end,
            regular_step=step,
            source_interval=source_interval,
            spinup=spinup,
        )

    @classmethod
    def _from_domain(
        cls,
        contract: _DatasetTemporalDomain,
        *,
        step: timedelta,
        reuse_count: int,
    ) -> Self:
        """Subdivide a validated source domain without revalidating it."""

        schedule = cls.model_construct(
            calendar=contract.calendar,
            regular_start=contract.start,
            regular_end=contract.end,
            regular_step=step,
            source_interval=contract.interval,
            spinup=contract.spinup,
            explicit_steps=(),
        )
        schedule._compiled_reuse_count = reuse_count
        schedule._compiled_main_steps = contract.count * reuse_count
        schedule._compiled_spinup_steps = (
            contract.spinup_count * reuse_count
            * (0 if contract.spinup is None else contract.spinup.cycles)
        )
        return schedule

    @property
    def _is_regular(self) -> bool:
        return self.regular_start is not None

    @property
    def cadence(self) -> timedelta | None:
        """Fixed model cadence, or ``None`` for an explicit schedule."""
        return self.regular_step

    @property
    def _reuse_count(self) -> int:
        """Number of model calls made from each source sample."""
        if self._compiled_reuse_count is None:
            raise ValueError("explicit schedules have no uniform reuse count")
        return self._compiled_reuse_count

    @property
    def _start(self) -> DateLike:
        """Start of the main simulation period."""
        if self._is_regular:
            return cast(DateLike, self.regular_start)
        return self.explicit_steps[0].start

    @property
    def _end(self) -> DateLike:
        """End of the main simulation period and complete execution."""
        if self._is_regular:
            return cast(DateLike, self.regular_end)
        return self.explicit_steps[-1].end

    @property
    def execution_start(self) -> DateLike:
        """Monotonic model-clock start, including all spinup calls."""
        if not self._is_regular or self.spinup is None:
            return self._start
        cadence = cast(timedelta, self.regular_step)
        return self._start - cadence * self._num_spinup_steps

    @property
    def _num_spinup_steps(self) -> int:
        return self._compiled_spinup_steps

    @property
    def num_main_steps(self) -> int:
        return self._compiled_main_steps

    def _step_at(self, index: int) -> SimulationStep:
        if type(index) is not int:
            raise TypeError("simulation step index must be an exact int")
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._step_at_trusted(index)

    def _step_at_trusted(self, index: int) -> SimulationStep:
        """Materialize a schedule index whose bounds were already proved."""

        if not self._is_regular:
            return self.explicit_steps[index]
        cadence = cast(timedelta, self.regular_step)
        start = self.execution_start + cadence * index
        spinup_steps = self._num_spinup_steps
        reuse_count = self._reuse_count
        source_interval = cast(timedelta, self.source_interval)
        if index < spinup_steps:
            spinup = cast(SpinupSchedule, self.spinup)
            per_cycle = spinup_steps // spinup.cycles
            cycle_model_index = index % per_cycle
            source_index, reuse_index = divmod(
                cycle_model_index,
                reuse_count,
            )
            source_start = spinup.source_start + source_interval * source_index
            return SimulationStep._from_schedule_trusted(
                index=index,
                start=start,
                end=start + cadence,
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
        source_start = self._start + source_interval * source_index
        return SimulationStep._from_schedule_trusted(
            index=index,
            start=start,
            end=start + cadence,
            source_start=source_start,
            source_end=source_start + source_interval,
            source_index=source_index,
            reuse_index=reuse_index,
            reuse_count=reuse_count,
        )

    def _index_at(self, start: DateLike) -> int:
        _require_date(start, label="model current_time")
        require_calendar(start, self.calendar, label="model current_time")
        if type(start) is not type(self._start):
            raise TypeError(
                "model current_time and schedule must use the same datetime "
                "representation"
            )
        if not self._is_regular:
            lower = 0
            upper = len(self.explicit_steps)
            while lower < upper:
                middle = (lower + upper) // 2
                if self.explicit_steps[middle].start < start:
                    lower = middle + 1
                else:
                    upper = middle
            if (
                lower < len(self.explicit_steps)
                and self.explicit_steps[lower].start == start
            ):
                return lower
            raise KeyError(start)
        regular_start = self.execution_start
        regular_step = cast(timedelta, self.regular_step)
        offset = timedelta_microseconds(
            start - regular_start,
            label="model schedule offset",
        )
        cadence = timedelta_microseconds(
            regular_step,
            label="simulation step",
        )
        index, remainder = divmod(offset, cadence)
        if index < 0 or index >= len(self) or remainder != 0:
            raise KeyError(start)
        return index

    def _main_index_at(self, start: DateLike) -> int:
        """Resolve a main-simulation boundary independently of spinup."""

        _require_date(start, label="parameter change start")
        require_calendar(
            start,
            self.calendar,
            label="parameter change start",
        )
        if type(start) is not type(self._start):
            raise TypeError(
                "parameter change start and schedule must use the same "
                "datetime representation"
            )
        if not self._is_regular:
            return self._index_at(start)
        regular_step = cast(timedelta, self.regular_step)
        offset = timedelta_microseconds(
            start - self._start,
            label="parameter change schedule offset",
        )
        cadence = timedelta_microseconds(
            regular_step,
            label="simulation step",
        )
        index, remainder = divmod(offset, cadence)
        if index < 0 or index >= self.num_main_steps or remainder != 0:
            raise KeyError(start)
        return index

    def _summary(self) -> str:
        """Return a stable, human-readable view of the compiled schedule."""

        if self._is_regular:
            schedule_type = "regular"
            cadence = str(self.cadence)
            source_interval = str(self.source_interval)
            reuse_count = self._reuse_count
            reuse_unit = "step" if reuse_count == 1 else "steps"
            source_reuse = f"{reuse_count} model {reuse_unit}/source sample"
        else:
            schedule_type = "explicit"
            cadence = source_interval = source_reuse = "per-step"

        if self.spinup is None:
            spinup_source = "none"
            spinup_cycles = 0
        else:
            spinup_source = f"[{self.spinup.source_start}, {self.spinup.source_end})"
            spinup_cycles = self.spinup.cycles

        fields = (
            ("Schedule type", schedule_type),
            ("Calendar", self.calendar),
            (
                "Execution period",
                f"[{self.execution_start}, {self._end})",
            ),
            ("Main period", f"[{self._start}, {self._end})"),
            ("Model cadence", cadence),
            ("Source interval", source_interval),
            ("Source reuse", source_reuse),
            ("Spinup source", spinup_source),
            ("Spinup cycles", spinup_cycles),
            ("Spinup steps", self._num_spinup_steps),
            ("Main steps", self.num_main_steps),
            ("Total steps", len(self)),
        )
        return "\n".join(f"{label:<17}: {value}" for label, value in fields)

    def __iter__(self):
        if not self._is_regular:
            yield from self.explicit_steps
            return
        for index in range(len(self)):
            yield self._step_at_trusted(index)

    def __len__(self) -> int:
        if not self._is_regular:
            return len(self.explicit_steps)
        return self._num_spinup_steps + self.num_main_steps


_DATASET_SUPPORT_COUNT_CONTEXT = "hydroforge_dataset_support_count"


class _DatasetSupportQuery(HydroForgeModel):
    """One bounded public lookup into a validated temporal domain."""

    index: int

    _resolved_index: int = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self, info: ValidationInfo) -> Self:
        count = (
            info.context.get(_DATASET_SUPPORT_COUNT_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if type(count) is not int or count < 1:
            raise ValueError("dataset support query requires domain context")
        index = self.index
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise ValueError(
                f"dataset sample index must satisfy -{count} <= index < "
                f"{count}; got {self.index}"
            )
        self._resolved_index = index
        return self

    @property
    def resolved_index(self) -> int:
        return self._resolved_index


class _DatasetTemporalDomain(HydroForgeModel):
    """Internal canonical timeline compiled from public Dataset fields."""

    start_date: DateLike
    end_date: DateLike
    time_interval: timedelta
    calendar: str | None = None
    spin_up_cycles: int = Field(default=0, strict=True, ge=0)
    spin_up_start_date: DateLike | None = None
    spin_up_end_date: DateLike | None = None

    _count: int = PrivateAttr()
    _spinup_count: int = PrivateAttr(default=0)
    _spinup: SpinupSchedule | None = PrivateAttr(default=None)
    _calendar_defaulted: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate_domain(self) -> Self:
        calendar, normalized, defaulted = normalize_calendar_dates(
            {
                "dataset start_date": self.start_date,
                "dataset end_date": self.end_date,
                "dataset spin_up_start_date": self.spin_up_start_date,
                "dataset spin_up_end_date": self.spin_up_end_date,
            },
            calendar=self.calendar,
        )
        start = cast(DateLike, normalized["dataset start_date"])
        end = cast(DateLike, normalized["dataset end_date"])
        if end < start:
            raise ValueError("dataset end_date must not precede start_date")
        if type(self.time_interval) is not timedelta:
            raise ValueError("dataset sample interval must be a timedelta")
        if (
            timedelta_microseconds(
                self.time_interval,
                label="dataset sample interval",
            )
            <= 0
        ):
            raise ValueError("dataset sample interval must be positive")
        count = (
            _timedelta_quotient_trusted(
                end - start,
                self.time_interval,
                duration_label="dataset endpoint span",
                interval_label="dataset sample interval",
            )
            + 1
        )
        if (self.spin_up_start_date is None) != (self.spin_up_end_date is None):
            raise ValueError(
                "spin_up_start_date and spin_up_end_date must be provided together"
            )
        if self.spin_up_cycles > 0 and self.spin_up_start_date is None:
            raise ValueError(
                "spin-up dates are required when spin_up_cycles is positive"
            )
        if self.spin_up_cycles == 0 and self.spin_up_start_date is not None:
            raise ValueError(
                "spin-up dates require a positive spin_up_cycles value"
            )
        spin_start = normalized["dataset spin_up_start_date"]
        spin_end = normalized["dataset spin_up_end_date"]
        if spin_start is not None and spin_end is not None:
            spin_start = cast(DateLike, spin_start)
            spin_end = cast(DateLike, spin_end)
        spinup = None
        spinup_count = 0
        if self.spin_up_cycles > 0 and spin_start is not None and spin_end is not None:
            spinup = SpinupSchedule(
                source_start=spin_start,
                source_end=spin_end + self.time_interval,
                cycles=self.spin_up_cycles,
            )
            require_calendar(
                spinup.source_start,
                calendar,
                label="dataset spinup source start",
            )
            require_calendar(
                spinup.source_end,
                calendar,
                label="dataset spinup source end",
            )
            if type(spinup.source_start) is not type(start):
                raise ValueError(
                    "dataset spinup and main bounds must use the same datetime "
                    "representation"
                )
            spinup_count = _timedelta_quotient_trusted(
                spinup.source_end - spinup.source_start,
                self.time_interval,
                duration_label="dataset spinup source duration",
                interval_label="dataset sample interval",
            )
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "spin_up_start_date", spin_start)
        object.__setattr__(self, "spin_up_end_date", spin_end)
        self._count = count
        self._spinup_count = spinup_count
        self._spinup = spinup
        self._calendar_defaulted = defaulted
        return self

    @property
    def calendar_defaulted(self) -> bool:
        """Whether storage metadata may replace the provisional calendar."""

        return self._calendar_defaulted

    @property
    def start(self) -> DateLike:
        return self.start_date

    @property
    def interval(self) -> timedelta:
        return self.time_interval

    @property
    def count(self) -> int:
        return self._count

    @property
    def spinup(self) -> SpinupSchedule | None:
        return self._spinup

    @property
    def spinup_count(self) -> int:
        return self._spinup_count

    def support(self, index: int) -> tuple[DateLike, DateLike]:
        query = _DatasetSupportQuery.model_validate(
            {"index": index},
            context={_DATASET_SUPPORT_COUNT_CONTEXT: self.count},
        )
        return self._support_trusted(query.resolved_index)

    def _support_trusted(self, index: int) -> tuple[DateLike, DateLike]:
        """Resolve one compiler-owned, already bounded support index."""

        start = self.start + self.interval * index
        return start, start + self.interval

    @property
    def end(self) -> DateLike:
        return self.end_date + self.time_interval

    @classmethod
    def combine(
        cls,
        contracts: Mapping[str, _DatasetTemporalDomain],
    ) -> Self:
        request = _DatasetTemporalCombineRequest(contracts=contracts)
        reference = request.reference
        if cls is _DatasetTemporalDomain:
            return reference
        return cls.model_validate(reference.model_dump())


def _combine_temporal_domains_trusted(
    contracts: Mapping[str, _DatasetTemporalDomain],
) -> _DatasetTemporalDomain:
    """Combine validated named domains inside a composite validator."""

    name, reference = next(iter(contracts.items()))
    for other_name, other in tuple(contracts.items())[1:]:
        if type(other.start) is not type(reference.start):
            raise ValueError(
                f"dataset timelines {name!r} and {other_name!r} use "
                "different datetime representations"
            )
        for attribute in (
            "calendar",
            "start",
            "interval",
            "count",
            "spinup",
        ):
            if getattr(other, attribute) != getattr(reference, attribute):
                raise ValueError(
                    f"dataset timelines {name!r} and {other_name!r} "
                    f"differ in {attribute}"
                )
    return reference


class _DatasetTemporalCombineRequest(HydroForgeModel):
    """Validated internal request for combining named Dataset domains."""

    contracts: Mapping[str, _DatasetTemporalDomain]

    _reference: _DatasetTemporalDomain = PrivateAttr()

    @model_validator(mode="after")
    def _combine(self) -> Self:
        if not self.contracts:
            raise ValueError("at least one dataset temporal contract is required")
        invalid_names = [name for name in self.contracts if not name]
        if invalid_names:
            raise ValueError("dataset timeline names must be non-empty strings")
        self._reference = _combine_temporal_domains_trusted(self.contracts)
        return self

    @property
    def reference(self) -> _DatasetTemporalDomain:
        return self._reference


class EveryStep(HydroForgeModel):
    """Every model call is a complete inner statistics window."""


class CalendarWindow(HydroForgeModel):
    period: CalendarPeriod
    start_month: int = 1
    start_day: int = 1

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if type(self.period) is not str or self.period not in {
            "day",
            "month",
            "year",
        }:
            raise ValueError("calendar window period must be 'day', 'month', or 'year'")
        if type(self.start_month) is not int:
            raise ValueError("start_month must be an exact int")
        if type(self.start_day) is not int:
            raise ValueError("start_day must be an exact int")
        if not 1 <= self.start_month <= 12:
            raise ValueError("start_month must be in 1..12")
        if not 1 <= self.start_day <= 31:
            raise ValueError("start_day must be in 1..31")
        if self.period != "year" and (self.start_month != 1 or self.start_day != 1):
            raise ValueError("custom origins are supported only for year windows")
        return self


class ExplicitWindow(HydroForgeModel):
    name: str
    start: DateLike
    end: DateLike

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("explicit window name must be a non-empty string")
        date_values = {
            f"explicit window {self.name!r} start": self.start,
            f"explicit window {self.name!r} end": self.end,
        }
        _calendar, normalized, _defaulted = normalize_calendar_dates(
            date_values,
            calendar=None,
            preserve_cftime_declaration=True,
        )
        start = cast(
            DateLike, normalized[f"explicit window {self.name!r} start"],
        )
        end = cast(
            DateLike, normalized[f"explicit window {self.name!r} end"],
        )
        if end <= start:
            raise ValueError(f"explicit window {self.name!r} is empty")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        return self


class ExplicitWindows(HydroForgeModel):
    windows: tuple[ExplicitWindow, ...]

    @model_validator(mode="after")
    def _validate_windows(self) -> Self:
        if not self.windows:
            raise ValueError("explicit windows must not be empty")
        if not isinstance(self.windows, tuple) or any(
            not isinstance(window, ExplicitWindow) for window in self.windows
        ):
            raise ValueError(
                "explicit windows must be a tuple of ExplicitWindow values"
            )
        names = tuple(window.name for window in self.windows)
        if len(set(names)) != len(names):
            raise ValueError("explicit statistics window names must be unique")
        values = {
            f"explicit window {index} start": window.start
            for index, window in enumerate(self.windows)
        } | {
            f"explicit window {index} end": window.end
            for index, window in enumerate(self.windows)
        }
        _calendar, normalized, _defaulted = normalize_calendar_dates(
            values,
            calendar=None,
            preserve_cftime_declaration=True,
        )
        windows = []
        for index, window in enumerate(self.windows):
            normalized_window = ExplicitWindow(
                name=window.name,
                start=cast(
                    DateLike, normalized[f"explicit window {index} start"],
                ),
                end=cast(
                    DateLike, normalized[f"explicit window {index} end"],
                ),
            )
            windows.append(normalized_window)
        windows = tuple(windows)
        object.__setattr__(self, "windows", windows)
        previous_end: DateLike | None = None
        for window in windows:
            if previous_end is not None and window.start < previous_end:
                raise ValueError("explicit statistics windows must not overlap")
            previous_end = window.end
        return self


WindowRule = EveryStep | CalendarWindow | ExplicitWindows


class _StatisticsOutput(HydroForgeModel):
    """One canonical output compiled from ``variables_to_save``."""

    name: str
    operation: str
    expression: str | None = None

    @model_validator(mode="after")
    def _validate_output(self) -> Self:
        if not self.name:
            raise ValueError("statistics output name must be non-empty")
        if not self.operation:
            raise ValueError("statistics output operation must be non-empty")
        if self.operation == "static":
            if self.expression is not None:
                raise ValueError("static statistics outputs must name a declared field")
            return self
        from hydroforge.statistics.ir import parse_operation

        parse_operation(self.operation)
        if self.expression is not None and not self.expression:
            raise ValueError("statistics output expression must be non-empty")
        return self


class StatisticsPlan(HydroForgeModel):
    """User-defined temporal windows for model statistics."""

    inner: WindowRule = EveryStep()
    outer: WindowRule | None = None
    partial_period: Literal["close", "drop"] = "close"

    @property
    def _effective_outer(self) -> WindowRule:
        """Return the resolved outer rule without rewriting caller input."""

        return self.inner if self.outer is None else self.outer

    @model_validator(mode="after")
    def _validate_plan(self) -> Self:
        valid_rules = (EveryStep, CalendarWindow, ExplicitWindows)
        if not isinstance(self.inner, valid_rules):
            raise ValueError("statistics inner must be a WindowRule")
        if self.outer is not None and not isinstance(self.outer, valid_rules):
            raise ValueError("statistics outer must be a WindowRule or None")
        if type(self.partial_period) is not str or self.partial_period not in {
            "close",
            "drop",
        }:
            raise ValueError("partial_period must be 'close' or 'drop'")
        return self

"""Immutable simulation call schedules with optional forcing spin-up."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal, Self, cast

from pydantic import Field, PrivateAttr, model_validator

from hydroforge.core.time import (
    DateLike,
    normalize_calendar_dates,
    require_calendar,
    require_date,
    timedelta_microseconds,
    timedelta_quotient,
)
from hydroforge.core.validation import HydroForgeModel

SimulationPhase = Literal["spinup", "main"]


class SimulationStep(HydroForgeModel):
    """One physical-time interval, identified by its execution index."""

    index: int = Field(ge=0)
    start: DateLike
    end: DateLike
    source_start: DateLike | None = None
    source_end: DateLike | None = None
    phase: SimulationPhase = "main"
    spinup_cycle: int | None = Field(default=None, ge=0)
    source_index: int = Field(default=0, ge=0)
    reuse_index: int = Field(default=0, ge=0)
    reuse_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_step(self) -> Self:
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
            DateLike,
            normalized["simulation step source start"],
        )
        source_end = cast(DateLike, normalized["simulation step source end"])
        if end <= start:
            raise ValueError("simulation step must have positive duration")
        if source_end <= source_start:
            raise ValueError("simulation step source interval must be positive")
        if self.phase == "main" and self.spinup_cycle is not None:
            raise ValueError("main simulation steps cannot have a spinup cycle")
        if self.phase == "spinup" and self.spinup_cycle is None:
            raise ValueError(
                "spinup simulation steps require a non-negative cycle index"
            )
        if self.reuse_index >= self.reuse_count:
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
    ) -> SimulationStep:
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
    cycles: int = Field(default=1, ge=1)

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
            if timedelta_microseconds(regular_step) <= 0:
                raise ValueError("simulation step must be positive")
            source_interval = (
                regular_step if self.source_interval is None else self.source_interval
            )
            if type(source_interval) is not timedelta:
                raise ValueError("source interval must be a timedelta")
            if timedelta_microseconds(source_interval) <= 0:
                raise ValueError("source interval must be positive")
            if source_interval < regular_step:
                raise ValueError(
                    "source interval must not be shorter than simulation step"
                )
            reuse_count = timedelta_quotient(
                source_interval,
                regular_step,
                duration_label="source interval",
                interval_label="simulation step",
            )
            object.__setattr__(self, "source_interval", source_interval)
            main_source_samples = timedelta_quotient(
                regular_end - regular_start,
                source_interval,
                duration_label="main simulation duration",
                interval_label="source interval",
            )
            spinup_source_samples = 0
            if self.spinup is not None:
                spinup = SpinupSchedule(
                    source_start=cast(
                        DateLike,
                        normalized["spinup source start"],
                    ),
                    source_end=cast(
                        DateLike,
                        normalized["spinup source end"],
                    ),
                    cycles=self.spinup.cycles,
                )
                object.__setattr__(self, "spinup", spinup)
                spinup_source_samples = timedelta_quotient(
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
        normalized_steps = []
        for index, step in enumerate(self.explicit_steps):
            normalized_step = SimulationStep(
                index=step.index,
                start=cast(
                    DateLike,
                    normalized[f"simulation step {index} start"],
                ),
                end=cast(
                    DateLike,
                    normalized[f"simulation step {index} end"],
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
        *,
        calendar: str,
        start: DateLike,
        end: DateLike,
        source_interval: timedelta,
        source_count: int,
        spinup: SpinupSchedule | None,
        spinup_source_count: int,
        step: timedelta,
        reuse_count: int,
    ) -> Self:
        """Subdivide a validated dataset source domain without revalidating it."""

        schedule = cls.model_construct(
            calendar=calendar,
            regular_start=start,
            regular_end=end,
            regular_step=step,
            source_interval=source_interval,
            spinup=spinup,
            explicit_steps=(),
        )
        schedule._compiled_reuse_count = reuse_count
        schedule._compiled_main_steps = source_count * reuse_count
        schedule._compiled_spinup_steps = (
            spinup_source_count * reuse_count * (0 if spinup is None else spinup.cycles)
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
            start = source_start + cadence * reuse_index
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
        start = self._start + cadence * main_model_index
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

    def _main_index_at(self, start: DateLike) -> int:
        require_date(start, label="model current_time")
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
        regular_start = self._start
        regular_step = cast(timedelta, self.regular_step)
        offset = timedelta_microseconds(start - regular_start)
        cadence = timedelta_microseconds(regular_step)
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

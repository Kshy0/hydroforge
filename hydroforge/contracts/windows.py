"""Declarative statistics windows and their canonical output requests."""

from __future__ import annotations

from typing import Literal, Self, cast

from pydantic import Field, model_validator

from hydroforge.core.expr import parse_operation
from hydroforge.core.time import DateLike, normalize_calendar_dates
from hydroforge.core.validation import HydroForgeModel

CalendarPeriod = Literal["day", "month", "year"]


class EveryStep(HydroForgeModel):
    """Every model call is a complete inner statistics window."""


class CalendarWindow(HydroForgeModel):
    period: CalendarPeriod
    start_month: int = Field(default=1, ge=1, le=12)
    start_day: int = Field(default=1, ge=1, le=31)

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if self.period != "year" and (self.start_month != 1 or self.start_day != 1):
            raise ValueError("custom origins are supported only for year windows")
        return self


class ExplicitWindow(HydroForgeModel):
    name: str = Field(min_length=1)
    start: DateLike
    end: DateLike

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
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
            DateLike,
            normalized[f"explicit window {self.name!r} start"],
        )
        end = cast(
            DateLike,
            normalized[f"explicit window {self.name!r} end"],
        )
        if end <= start:
            raise ValueError(f"explicit window {self.name!r} is empty")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        return self


class ExplicitWindows(HydroForgeModel):
    windows: tuple[ExplicitWindow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_windows(self) -> Self:
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
                    DateLike,
                    normalized[f"explicit window {index} start"],
                ),
                end=cast(
                    DateLike,
                    normalized[f"explicit window {index} end"],
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


class StatisticsOutput(HydroForgeModel):
    """One canonical output compiled from ``OutputConfig.variables``."""

    name: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    expression: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_output(self) -> Self:
        if self.operation == "static":
            if self.expression is not None:
                raise ValueError("static statistics outputs must name a declared field")
            return self
        parse_operation(self.operation)
        return self


class StatisticsPlan(HydroForgeModel):
    """User-defined temporal windows for model statistics.

    Simple statistics aggregate the samples of each ``inner`` window;
    compound statistics fold complete inner results over ``outer`` windows
    (the inner rule when omitted).  Windows follow the schedule whether or
    not a step collects output: a step called with ``output_enabled=False``
    contributes no sample, but windows keep their boundaries and still close
    there.  A window without samples publishes nothing and takes no part in
    the outer fold, and means weight only the sampled time.  Spin-up steps
    belong to no window; the main timeline starts new windows.

    ``partial_period="close"`` publishes the incomplete windows at both ends
    of the schedule; ``"drop"`` keeps only complete outer periods at both
    ends, including the inner windows of a dropped period, and therefore
    requires ``simulation_schedule``.
    """

    inner: WindowRule = EveryStep()
    outer: WindowRule | None = None
    partial_period: Literal["close", "drop"] = "close"

    @property
    def _effective_outer(self) -> WindowRule:
        """Return the resolved outer rule without rewriting caller input."""

        return self.inner if self.outer is None else self.outer

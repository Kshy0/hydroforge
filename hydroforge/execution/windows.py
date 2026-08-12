"""Compiled statistics-window control over model call intervals."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import cftime

from hydroforge.contracts.temporal import (
    CalendarWindow,
    EveryStep,
    ExplicitWindows,
    SimulationSchedule,
    SimulationStep,
    StatisticsPlan,
    WindowRule,
    require_calendar,
)


@dataclass(frozen=True, slots=True)
class WindowDecision:
    output_enabled: bool
    first: bool
    last: bool
    outer_first: bool
    outer_last: bool


class StatisticsWindowController:
    """O(1) regular/calendar cursor with explicit-window lookup support."""

    def __init__(
        self, plan: StatisticsPlan, schedule: SimulationSchedule,
    ) -> None:
        if not isinstance(plan, StatisticsPlan):
            raise TypeError("statistics plan must be a StatisticsPlan")
        if not isinstance(schedule, SimulationSchedule):
            raise TypeError("statistics schedule must be a SimulationSchedule")
        self.plan = plan
        self.schedule = schedule
        self._validate_schedule_contract()
        self._explicit_starts = {
            id(rule): tuple(window.start for window in rule.windows)
            for rule in (plan.inner, plan.outer)
            if isinstance(rule, ExplicitWindows)
        }
        self._last_inner_key: Any = None
        self._last_outer_key: Any = None

    def _validate_schedule_contract(self) -> None:
        """Bind schedule-dependent rule invariants once at runtime setup."""

        schedule = self.schedule
        for rule in (self.plan.inner, self.plan.outer):
            if isinstance(rule, ExplicitWindows):
                for window in rule.windows:
                    require_calendar(
                        window.start, schedule.calendar,
                        label=f"explicit window {window.name!r} start",
                    )
                    require_calendar(
                        window.end, schedule.calendar,
                        label=f"explicit window {window.name!r} end",
                    )
                    if type(window.start) is not type(schedule.start):
                        raise TypeError(
                            f"explicit window {window.name!r} and simulation "
                            "schedule must use the same datetime representation"
                        )
            if isinstance(rule, CalendarWindow) and rule.period == "year":
                try:
                    cftime.datetime(
                        2001, rule.start_month, rule.start_day,
                        calendar=schedule.calendar,
                    )
                except ValueError as exc:
                    raise ValueError(
                        "annual statistics origin must exist in every year of "
                        f"calendar {schedule.calendar!r}"
                    ) from exc

    @staticmethod
    def _calendar_key(rule: CalendarWindow, value: Any) -> tuple[Any, ...]:
        if rule.period == "day":
            return (value.year, value.month, value.day)
        if rule.period == "month":
            return (value.year, value.month)
        origin = (rule.start_month, rule.start_day)
        year = value.year if (value.month, value.day) >= origin else value.year - 1
        return (year, origin)

    @staticmethod
    def _is_calendar_boundary(rule: CalendarWindow, value: Any) -> bool:
        at_midnight = all(
            getattr(value, field, 0) == 0
            for field in ("hour", "minute", "second", "microsecond")
        )
        if not at_midnight:
            return False
        if rule.period == "day":
            return True
        if rule.period == "month":
            return value.day == 1
        return (value.month, value.day) == (
            rule.start_month, rule.start_day,
        )

    def _locate(
        self, rule: WindowRule, value: Any,
    ) -> tuple[Any, Any] | None:
        if isinstance(rule, EveryStep):
            return value, None
        if isinstance(rule, CalendarWindow):
            return self._calendar_key(rule, value), None
        starts = self._explicit_starts[id(rule)]
        index = bisect_right(starts, value) - 1
        if index < 0:
            return None
        window = rule.windows[index]
        if not window.start <= value < window.end:
            return None
        return index, window

    def _rule_position(
        self,
        rule: WindowRule,
        *,
        start: Any,
        end: Any,
        previous_key: Any,
        final_step: bool,
    ) -> tuple[Any, bool, bool] | None:
        located = self._locate(rule, start)
        if located is None:
            if isinstance(rule, ExplicitWindows):
                starts = self._explicit_starts[id(rule)]
                next_index = bisect_right(starts, start)
                if (
                    next_index < len(rule.windows)
                    and end > rule.windows[next_index].start
                ):
                    window = rule.windows[next_index]
                    raise ValueError(
                        f"model step [{start!r}, {end!r}) crosses explicit "
                        f"window {window.name!r} boundary"
                    )
            return None
        key, window = located
        if isinstance(rule, EveryStep):
            return key, True, True
        if isinstance(rule, CalendarWindow):
            end_key = self._calendar_key(rule, end)
            changed = end_key != key
            if changed:
                if not self._is_calendar_boundary(rule, end):
                    raise ValueError(
                        f"model step [{start!r}, {end!r}) crosses a "
                        f"{rule.period} statistics boundary"
                    )
                # The instant immediately before an exact end boundary must
                # still belong to the start window.  A midpoint probe is not
                # sufficient for unequal month/year lengths (Jan 1 -> Mar 1
                # has a midpoint that is still in January).
                preceding_key = self._calendar_key(
                    rule, end - timedelta(microseconds=1),
                )
                if preceding_key != key:
                    raise ValueError("model step crosses multiple statistics windows")
            last = changed or (
                final_step and self.plan.partial_period == "close"
            )
            return key, previous_key != key, last
        if end > window.end:
            raise ValueError(
                f"model step [{start!r}, {end!r}) crosses explicit window "
                f"{window.name!r} boundary"
            )
        return key, previous_key != key, (
            end == window.end
            or final_step and self.plan.partial_period == "close"
        )

    def resolve(
        self,
        *,
        step: SimulationStep,
        output_enabled: bool,
    ) -> WindowDecision:
        """Advance windows from the managed step's validated schedule record."""

        if not isinstance(step, SimulationStep):
            raise TypeError("statistics step must be a SimulationStep")
        if not output_enabled:
            self._last_inner_key = None
            self._last_outer_key = None
            return WindowDecision(False, False, False, False, False)
        final_step = step.index == len(self.schedule) - 1
        inner_position = self._rule_position(
            self.plan.inner,
            start=step.start,
            end=step.end,
            previous_key=self._last_inner_key,
            final_step=final_step,
        )
        outer_rule = self.plan.outer or self.plan.inner
        outer_position = self._rule_position(
            outer_rule,
            start=step.start,
            end=step.end,
            previous_key=self._last_outer_key,
            final_step=final_step,
        )
        if inner_position is None or outer_position is None:
            self._last_inner_key = None
            self._last_outer_key = None
            return WindowDecision(False, False, False, False, False)
        inner_key, inner_first, inner_last = inner_position
        outer_key, outer_first, outer_last = outer_position
        if self._last_inner_key is None:
            inner_first = True
            outer_first = True
        self._last_inner_key = inner_key
        self._last_outer_key = outer_key
        return WindowDecision(
            True,
            inner_first, inner_last, outer_first, outer_last,
        )

    def snapshot_state(self) -> tuple[Any, Any]:
        """Capture only mutable window state for transactional rollback."""

        return self._last_inner_key, self._last_outer_key

    def restore_snapshot_state(self, state: tuple[Any, Any]) -> None:
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("statistics rollback snapshot is invalid")
        self._last_inner_key, self._last_outer_key = state

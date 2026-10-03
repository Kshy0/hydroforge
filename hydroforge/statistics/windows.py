"""Statistics windows over model steps and the host state of open windows."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import cftime
import torch

from hydroforge.contracts.schedule import SimulationSchedule, SimulationStep
from hydroforge.contracts.windows import (
    CalendarWindow,
    EveryStep,
    ExplicitWindow,
    ExplicitWindows,
    StatisticsPlan,
    WindowRule,
)
from hydroforge.core.time import normalize_calendar_dates, require_calendar
from hydroforge.statistics.phases import SampleFlags

_INNER_FIRST = int(SampleFlags.INNER_FIRST)
_INNER_LAST = int(SampleFlags.INNER_LAST)
_OUTER_FIRST = int(SampleFlags.OUTER_FIRST)
_OUTER_LAST = int(SampleFlags.OUTER_LAST)
_SETTLE = int(SampleFlags.SETTLE)
_NO_WINDOW = object()


@dataclass(frozen=True, slots=True)
class WindowEvents:
    """Inner and outer window boundaries one managed step carries."""

    inner_first: bool
    inner_last: bool
    outer_first: bool
    outer_last: bool


# A step without a schedule, or of an every-step plan, is a whole window.
WHOLE_WINDOW = WindowEvents(True, True, True, True)


class StatisticsWindowController:
    """O(1) regular/calendar cursor with explicit-window lookup support.

    The cursor follows every main-timeline step, whether or not the step
    collects output; only an explicit-window gap or a dropped partial period
    leaves it outside any window.
    """

    def __init__(
        self,
        plan: StatisticsPlan,
        schedule: SimulationSchedule | None,
    ) -> None:
        self.plan = plan
        self.schedule = schedule
        self._explicit_starts = {
            id(rule): tuple(window.start for window in rule.windows)
            for rule in (plan.inner, plan._effective_outer)
            if isinstance(rule, ExplicitWindows)
        }
        self._last_inner_key: Any = None
        self._last_outer_key: Any = None
        self._dropped_outer_key = self._final_partial_outer_key()

    def _final_partial_outer_key(self) -> Any:
        """Return the incomplete outer window a ``drop`` plan skips at the end."""

        schedule = self.schedule
        if self.plan.partial_period != "drop" or schedule is None or not len(schedule):
            return _NO_WINDOW
        final = schedule._step_at_trusted(len(schedule) - 1)
        if final.is_spin_up:
            return _NO_WINDOW
        position = self._rule_position(
            self.plan._effective_outer,
            start=final.start,
            end=final.end,
            previous_key=None,
            final_step=False,
        )
        if position is None or position[2]:
            return _NO_WINDOW
        return position[0]

    def _validate_schedule_contract(self) -> None:
        """Bind schedule-dependent rule invariants during model validation."""

        schedule = self.schedule
        if schedule is None:
            return
        for rule in (self.plan.inner, self.plan._effective_outer):
            if isinstance(rule, ExplicitWindows):
                for window in rule.windows:
                    require_calendar(
                        window.start,
                        schedule.calendar,
                        label=f"explicit window {window.name!r} start",
                    )
                    require_calendar(
                        window.end,
                        schedule.calendar,
                        label=f"explicit window {window.name!r} end",
                    )
                    if type(window.start) is not type(schedule._start):
                        raise ValueError(
                            f"explicit window {window.name!r} and simulation "
                            "schedule must use the same datetime representation"
                        )
            if isinstance(rule, CalendarWindow) and rule.period == "year":
                try:
                    cftime.datetime(
                        2001,
                        rule.start_month,
                        rule.start_day,
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
            rule.start_month,
            rule.start_day,
        )

    def _locate(
        self,
        rule: WindowRule,
        value: Any,
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

    def _starts_complete_period(self, rule: WindowRule, value: Any) -> bool:
        """Return whether ``value`` is the exact start of one rule window."""

        if isinstance(rule, EveryStep):
            return True
        if isinstance(rule, CalendarWindow):
            return self._is_calendar_boundary(rule, value)
        located = self._locate(rule, value)
        return located is not None and value == located[1].start

    def _rule_position(
        self,
        rule: WindowRule,
        *,
        start: Any,
        end: Any,
        previous_key: Any,
        final_step: bool,
        validate: bool = False,
    ) -> tuple[Any, bool, bool] | None:
        located = self._locate(rule, start)
        if located is None:
            if validate and isinstance(rule, ExplicitWindows):
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
            if validate and changed:
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
                    rule,
                    end - timedelta(microseconds=1),
                )
                if preceding_key != key:
                    raise ValueError("model step crosses multiple statistics windows")
            last = changed or (final_step and self.plan.partial_period == "close")
            return key, previous_key != key, last
        if validate and end > window.end:
            raise ValueError(
                f"model step [{start!r}, {end!r}) crosses explicit window "
                f"{window.name!r} boundary"
            )
        return (
            key,
            previous_key != key,
            (end == window.end or final_step and self.plan.partial_period == "close"),
        )

    def resolve(
        self,
        step: SimulationStep,
        *,
        _validate: bool = False,
    ) -> WindowEvents | None:
        """Advance the cursor; return ``None`` for a step outside every window.

        Model construction alone enables the schedule checks.
        """

        final_step = step.index == len(self.schedule) - 1
        outer_rule = self.plan._effective_outer
        inner_position = self._rule_position(
            self.plan.inner,
            start=step.start,
            end=step.end,
            previous_key=self._last_inner_key,
            final_step=final_step,
            validate=_validate,
        )
        outer_position = self._rule_position(
            outer_rule,
            start=step.start,
            end=step.end,
            previous_key=self._last_outer_key,
            final_step=final_step,
            validate=_validate,
        )
        if _validate and (inner_position is None) != (outer_position is None):
            raise ValueError(
                "inner and outer statistics windows must cover the same "
                f"model steps; coverage differs for [{step.start!r}, "
                f"{step.end!r})"
            )
        if inner_position is None:
            self._last_inner_key = None
            self._last_outer_key = None
            return None
        inner_key, inner_first, inner_last = inner_position
        outer_key, outer_first, outer_last = outer_position
        if self.plan.partial_period == "drop" and (
            outer_key == self._dropped_outer_key
            or self._last_inner_key is None
            and (
                not self._starts_complete_period(self.plan.inner, step.start)
                or not self._starts_complete_period(outer_rule, step.start)
            )
        ):
            return None
        if _validate and outer_first and not inner_first:
            raise ValueError(
                "outer statistics windows must start on an inner window "
                f"boundary; [{step.start!r}, {step.end!r}) starts only the "
                "outer window"
            )
        if _validate and outer_last and not inner_last:
            raise ValueError(
                "outer statistics windows must end on an inner window "
                f"boundary; [{step.start!r}, {step.end!r}) ends only the "
                "outer window"
            )
        if self._last_inner_key is None:
            inner_first = True
            outer_first = True
        self._last_inner_key = inner_key
        self._last_outer_key = outer_key
        return WindowEvents(inner_first, inner_last, outer_first, outer_last)

    def snapshot_state(self) -> tuple[Any, Any]:
        """Capture only mutable window state for transactional rollback."""

        return self._last_inner_key, self._last_outer_key

    def restore_snapshot_state(self, state: tuple[Any, Any]) -> None:
        self._last_inner_key, self._last_outer_key = state


class WindowState:
    """Host state of the open statistics windows of one runtime.

    It is the only owner of the macro-step counters (inner folds of the open
    outer window), the outputs the next close publishes, and the state that
    lets windows continue across steps collecting no output: whether the
    open inner window has samples, whether the open outer window has folds,
    and whether the next fold restarts the outer accumulators.  A window
    without samples publishes nothing and takes no part in the outer fold.

    ``begin`` only records the step; counters change at the fold (``claim``)
    and window state at ``close``, so an unentered step needs no rollback.
    """

    __slots__ = (
        "inner_outputs",
        "outer_outputs",
        "mean_count_limit",
        "macro_index",
        "macro_count",
        "dirty",
        "start_time",
        "inner_samples",
        "outer_folded",
        "pending_outer_first",
        "events",
        "sampling",
        "time",
    )

    def __init__(
        self,
        inner_outputs: tuple[str, ...],
        outer_outputs: tuple[str, ...],
        mean_count_limit: int | None,
    ) -> None:
        self.inner_outputs = inner_outputs
        self.outer_outputs = outer_outputs
        self.mean_count_limit = mean_count_limit
        self.macro_index = 0
        self.macro_count = 0
        self.dirty: set[str] = set()
        self.start_time: Any = None
        self.inner_samples = False
        self.outer_folded = False
        self.pending_outer_first = False
        self.events: WindowEvents | None = None
        self.sampling = False
        self.time: Any = None

    def begin(self, events: WindowEvents | None, *, sampling: bool, time: Any) -> int:
        """Record one step and return the step bits its samples carry."""

        self.events = events
        self.sampling = sampling = sampling and events is not None
        self.time = time
        if not sampling:
            return 0
        continuing = not events.inner_first and self.inner_samples
        flags = 0 if continuing else _INNER_FIRST
        if events.inner_last:
            flags |= _INNER_LAST
            if events.outer_first or self.pending_outer_first:
                flags |= _OUTER_FIRST
            if events.outer_last:
                flags |= _OUTER_LAST
        return flags

    def settle_phase(self) -> int:
        """Return the phase of a close without a sample, or 0 if none is due."""

        events = self.events
        if (
            events is None
            or not events.inner_last
            or self.sampling
            or events.inner_first
            or not self.inner_samples
        ):
            return 0
        phase = _SETTLE | _INNER_LAST
        if events.outer_first or self.pending_outer_first:
            phase |= _OUTER_FIRST
        if events.outer_last:
            phase |= _OUTER_LAST
        return phase

    def claim(self, phase: int) -> tuple[int, int]:
        """Account one inner-window fold; return its outer count and index."""

        restart = phase & _OUTER_FIRST
        index = 0 if restart else self.macro_index
        count = (0 if restart else self.macro_count) + 1
        limit = torch.iinfo(torch.int64).max
        if index >= limit or count > limit:
            raise OverflowError("statistics macro-step accounting exceeds int64 range")
        mean_limit = self.mean_count_limit
        if mean_limit is not None and count > mean_limit:
            raise OverflowError(
                "statistics compound mean macro-step count exceeds the "
                f"largest consecutive integer exactly representable by its "
                f"accumulator dtype ({mean_limit})"
            )
        self.macro_index = index + 1
        self.macro_count = count
        self.dirty.update(self.inner_outputs)
        return count, index

    def close(self) -> Any:
        """Finish the recorded step; return the output time if it publishes."""

        events, self.events = self.events, None
        if events is None:
            return None
        if events.inner_first:
            self.start_time = self.time
            self.inner_samples = False
        if events.outer_first:
            self.pending_outer_first = True
            self.outer_folded = False
        self.inner_samples = self.inner_samples or self.sampling
        if not events.inner_last:
            return None
        if self.inner_samples:
            self.pending_outer_first = False
            self.outer_folded = True
        if events.outer_last and self.outer_folded:
            self.dirty.update(self.outer_outputs)
            self.outer_folded = False
        label, self.start_time = self.start_time, None
        self.inner_samples = False
        return label if self.dirty else None


def bind_statistics_plan_schedule(
    plan: StatisticsPlan,
    schedule: SimulationSchedule,
) -> StatisticsPlan:
    """Bind plain-datetime explicit windows to the schedule calendar."""

    def bind(rule: WindowRule | None) -> WindowRule | None:
        if not isinstance(rule, ExplicitWindows):
            return rule
        values = {
            f"explicit window {index} start": window.start
            for index, window in enumerate(rule.windows)
        } | {
            f"explicit window {index} end": window.end
            for index, window in enumerate(rule.windows)
        }
        _calendar, normalized, _defaulted = normalize_calendar_dates(
            {**values, "schedule start": schedule._start},
            calendar=schedule.calendar,
            preserve_cftime_declaration=isinstance(schedule._start, cftime.datetime),
        )
        return ExplicitWindows(
            windows=tuple(
                ExplicitWindow(
                    name=window.name,
                    start=normalized[f"explicit window {index} start"],
                    end=normalized[f"explicit window {index} end"],
                )
                for index, window in enumerate(rule.windows)
            )
        )

    return StatisticsPlan(
        inner=bind(plan.inner),
        outer=bind(plan.outer),
        partial_period=plan.partial_period,
    )


def validate_statistics_window_schedule(
    plan: StatisticsPlan,
    schedule: SimulationSchedule,
) -> None:
    """Validate every schedule/window interaction before runtime starts."""

    controller = StatisticsWindowController(plan, schedule)
    controller._validate_schedule_contract()
    if isinstance(plan.inner, EveryStep) and isinstance(
        plan._effective_outer,
        EveryStep,
    ):
        return
    for step in schedule:
        if step.is_spin_up:
            continue
        controller.resolve(step, _validate=True)

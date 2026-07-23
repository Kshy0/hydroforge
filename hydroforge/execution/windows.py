"""Compiled statistics-window control over model call intervals."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from hydroforge.contracts.temporal import (
    CalendarWindow,
    EveryStep,
    ExplicitWindows,
    StatisticsFlags,
    StatisticsPlan,
    WindowRule,
    require_calendar,
    window_rule_signature,
)


@dataclass(frozen=True, slots=True)
class WindowDecision:
    output_enabled: bool
    flags: StatisticsFlags


class StatisticsWindowController:
    """O(1) regular/calendar cursor with explicit-window lookup support."""

    def __init__(self, plan: StatisticsPlan) -> None:
        self.plan = plan
        self.schedule = plan.schedule
        self._explicit_starts = {
            id(rule): tuple(window.start for window in rule.windows)
            for rule in (plan.inner, plan.outer)
            if isinstance(rule, ExplicitWindows)
        }
        self._output_active = False
        self._last_inner_key: Any = None
        self._last_outer_key: Any = None
        self._last_step_index: int | None = None
        self._inner_open = False
        self._outer_open = False
        self.fingerprint = self._fingerprint()

    @property
    def open_windows(self) -> tuple[bool, bool]:
        """Whether inner/outer accumulators are incomplete after the last step."""
        return self._inner_open, self._outer_open

    def _fingerprint(self) -> str:
        definition = {
            "schedule": self.schedule.fingerprint,
            "inner": window_rule_signature(self.plan.inner),
            "outer": window_rule_signature(self.plan.outer),
            "partial_period": self.plan.partial_period,
        }
        encoded = json.dumps(
            definition, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _validate_step(
        self, current_time: Any, time_step: float, *, allow_spinup: bool = False,
    ):
        require_calendar(
            current_time, self.schedule.calendar, label="model current_time",
        )
        try:
            step = self.schedule.step_at(self.schedule.index_at(current_time))
        except KeyError as exc:
            if allow_spinup and current_time < self.schedule.start:
                return None
            raise ValueError(
                f"current_time {current_time!r} is not a model schedule boundary"
            ) from exc
        if abs(step.duration_seconds - float(time_step)) > 1e-9:
            raise ValueError(
                f"time_step {time_step} differs from scheduled duration "
                f"{step.duration_seconds} at {current_time!r}"
            )
        return step

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
                # Crossing more than one calendar window cannot be represented
                # by a single pair of first/last flags.
                midpoint = start + (end - start) / 2
                if self._calendar_key(rule, midpoint) not in {key, end_key}:
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
        current_time: Any,
        time_step: float,
        output_enabled: bool,
        override: StatisticsFlags | None = None,
    ) -> WindowDecision:
        if not output_enabled:
            step = self._validate_step(
                current_time, time_step,
                allow_spinup=self._last_step_index is None,
            )
            if step is None:
                # Pre-main spin-up is intentionally allowed while statistics
                # are disabled. Once the model schedule has started, however,
                # disabling output may not become a calendar-validation bypass.
                self._output_active = False
                self._last_inner_key = None
                self._last_outer_key = None
                self._inner_open = False
                self._outer_open = False
                return WindowDecision(
                    False, StatisticsFlags(False, False, False, False),
                )
            if (
                self._last_step_index is not None
                and step.index != self._last_step_index + 1
            ):
                raise ValueError(
                    f"model schedule moved from step {self._last_step_index} "
                    f"to {step.index}; expected {self._last_step_index + 1}"
                )
            self._output_active = False
            self._last_inner_key = None
            self._last_outer_key = None
            self._last_step_index = step.index
            self._inner_open = False
            self._outer_open = False
            return WindowDecision(False, StatisticsFlags(False, False, False, False))
        step = self._validate_step(current_time, time_step)
        final_step = step.index == len(self.schedule) - 1
        if (
            self._last_step_index is not None
            and step.index != self._last_step_index + 1
        ):
            raise ValueError(
                f"model schedule moved from step {self._last_step_index} "
                f"to {step.index}; expected {self._last_step_index + 1}"
            )
        if override is not None:
            self._validate_override(override)
            self._output_active = True
            inner_location = self._locate(self.plan.inner, step.start)
            outer_location = self._locate(
                self.plan.outer or self.plan.inner, step.start,
            )
            self._last_inner_key = (
                None if inner_location is None else inner_location[0]
            )
            self._last_outer_key = (
                None if outer_location is None else outer_location[0]
            )
            self._last_step_index = step.index
            self._inner_open = not override.last
            self._outer_open = not override.outer_last
            return WindowDecision(True, override)

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
            self._output_active = False
            self._last_inner_key = None
            self._last_outer_key = None
            # A statistics gap disables output, not schedule validation.  Keep
            # advancing the execution cursor so a caller cannot silently jump
            # over model steps while it happens to be outside output windows.
            self._last_step_index = step.index
            self._inner_open = False
            self._outer_open = False
            return WindowDecision(False, StatisticsFlags(False, False, False, False))
        inner_key, inner_first, inner_last = inner_position
        outer_key, outer_first, outer_last = outer_position
        if not self._output_active:
            inner_first = True
            outer_first = True
        self._output_active = True
        self._last_inner_key = inner_key
        self._last_outer_key = outer_key
        self._last_step_index = step.index
        flags = StatisticsFlags(
            inner_first, inner_last, outer_first, outer_last,
        )
        self._inner_open = not flags.last
        self._outer_open = not flags.outer_last
        return WindowDecision(True, flags)

    @staticmethod
    def _validate_override(flags: StatisticsFlags) -> None:
        if (flags.outer_first or flags.outer_last) and not flags.last:
            raise ValueError(
                "manual outer statistics flags require stat_is_last=True"
            )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "last_step_index": -1 if self._last_step_index is None
            else self._last_step_index,
            "output_active": int(self._output_active),
            "inner_open": int(self._inner_open),
            "outer_open": int(self._outer_open),
        }

    def _checkpoint_position(
        self, state: dict[str, Any],
    ) -> tuple[int | None, bool, Any, Any, bool, bool]:
        expected = {
            "fingerprint", "last_step_index", "output_active",
            "inner_open", "outer_open",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("checkpoint statistics cursor has an invalid schema")
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("checkpoint statistics plan does not match the model plan")
        index = state["last_step_index"]
        if type(index) is not int:
            raise TypeError("checkpoint statistics step index must be an exact int")
        if index < -1 or index >= len(self.schedule):
            raise ValueError("checkpoint statistics step index is outside the schedule")
        bits = {
            name: state[name]
            for name in ("output_active", "inner_open", "outer_open")
        }
        if any(type(value) is not int or value not in {0, 1} for value in bits.values()):
            raise TypeError(
                "checkpoint statistics flags must be exact integer bits"
            )
        active = bool(bits["output_active"])
        inner_open = bool(bits["inner_open"])
        outer_open = bool(bits["outer_open"])
        if (inner_open or outer_open) and not active:
            raise ValueError("checkpoint has open statistics but inactive output")
        if index == -1:
            if active:
                raise ValueError(
                    "checkpoint cannot have active statistics without a step"
                )
            return None, False, None, None, False, False
        if not active:
            # Inactive can mean an explicit statistics gap.  Retain the model
            # schedule cursor even though there is no open output window.
            return index, False, None, None, False, False
        step = self.schedule.step_at(index)
        inner = self._locate(self.plan.inner, step.start)
        outer = self._locate(self.plan.outer or self.plan.inner, step.start)
        if inner is None or outer is None:
            raise ValueError("checkpoint points outside configured statistics windows")
        return index, True, inner[0], outer[0], inner_open, outer_open

    def validate_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Validate persisted cursor state without mutating this controller."""
        self._checkpoint_position(state)

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        (
            index, active, inner_key, outer_key, inner_open, outer_open,
        ) = self._checkpoint_position(state)
        self._last_step_index = index
        self._output_active = active
        self._last_inner_key = inner_key
        self._last_outer_key = outer_key
        self._inner_open = inner_open
        self._outer_open = outer_open

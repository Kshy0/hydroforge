# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Explicit progress state and service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ProgressState:
    current_step: int = 0
    phase: str | None = None
    _wall_start: float = 0.0
    _last_emit: float = 0.0
    _refresh_interval: float = 0.1
    _schedule_start_fraction: float | None = None

    def start(
        self,
        phase: str,
        *,
        schedule_fraction: float | None = None,
    ) -> None:
        self.phase = phase
        self.current_step = 0
        self._wall_start = time.perf_counter()
        self._last_emit = self._wall_start - self._refresh_interval
        self._schedule_start_fraction = schedule_fraction

    def begin_step(
        self,
        phase: str,
        *,
        schedule_fraction: float | None = None,
    ) -> None:
        if phase != self.phase:
            self.start(phase, schedule_fraction=schedule_fraction)
        elif self._schedule_start_fraction is None and schedule_fraction is not None:
            self._schedule_start_fraction = schedule_fraction

    def tick(self, phase: str, *, force_emit: bool = False) -> bool:
        if phase != self.phase:
            self.start(phase)
        now = time.perf_counter()
        self.current_step += 1
        if not force_emit and now - self._last_emit < self._refresh_interval:
            return False
        self._last_emit = now
        return True

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._wall_start

    @property
    def speed(self) -> float:
        elapsed = self.elapsed
        return self.current_step / elapsed if elapsed > 0 else 0.0

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}min"
        return f"{seconds / 3600:.1f}h"

    def format_schedule(self, *, fraction: float, total_steps: int) -> str:
        start_fraction = (
            0.0
            if self._schedule_start_fraction is None
            else self._schedule_start_fraction
        )
        completed = fraction - start_fraction
        elapsed = self.elapsed
        completed_steps = completed * total_steps
        speed = completed_steps / elapsed if elapsed > 0.0 else 0.0
        eta = (
            elapsed * (1.0 - fraction) / completed if completed > 0.0 else float("inf")
        )
        return (
            f"[{fraction * 100:5.1f}%] "
            f"{speed:.2f} steps/s ETA {self._fmt_duration(eta)}"
        )

    def format_unbounded(self) -> str:
        # Only unscheduled runs are unbounded, and they have no spin-up phase.
        unit = "step" if self.current_step == 1 else "steps"
        return f"[running {self.current_step} {unit}] {self.speed:.2f} steps/s"


class ProgressRuntime:
    """Progress of one runtime along its schedule position.

    Each call receives the executing scheduled step (``None`` without a
    schedule); the runtime clock remains the only committed position.
    """

    def __init__(self, runtime: Any) -> None:
        self.schedule = runtime.plan.schedule
        self.state = ProgressState()

    @staticmethod
    def _fraction(schedule: Any, step: Any, *, completed: bool) -> float:
        if schedule._is_regular:
            return (step.index + int(completed)) / len(schedule)
        date = step.end if completed else step.start
        return (date - schedule.start) / (schedule.end - schedule.start)

    def _phase(self, step: Any) -> str:
        return "unbounded" if self.schedule is None or step is None else step.phase

    def begin_step(self, step: Any) -> None:
        schedule = self.schedule
        phase = self._phase(step)
        fraction = None
        if (
            schedule is not None
            and step is not None
            and (
                phase != self.state.phase or self.state._schedule_start_fraction is None
            )
        ):
            fraction = self._fraction(schedule, step, completed=False)
        self.state.begin_step(
            phase,
            schedule_fraction=fraction,
        )

    def progress_tick(self, step: Any) -> bool:
        schedule = self.schedule
        final_step = (
            schedule is not None
            and step is not None
            and step.index + 1 == len(schedule)
        )
        return self.state.tick(self._phase(step), force_emit=final_step)

    def format_progress(self, step: Any) -> str:
        schedule = self.schedule
        if schedule is None or step is None:
            return self.state.format_unbounded()
        return self.state.format_schedule(
            fraction=self._fraction(schedule, step, completed=True),
            total_steps=len(schedule),
        )

"""Explicit progress state and service."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressState:
    current_step: int = 0
    phase: str | None = None
    _wall_start: float = 0.0
    _step_start: float = 0.0
    _recent_dts: list[float] = field(default_factory=list)
    _window_size: int = 50
    _schedule_start_fraction: float | None = None

    def start(
        self, phase: str, *, schedule_fraction: float | None = None,
    ) -> None:
        self.phase = phase
        self.current_step = 0
        self._wall_start = time.perf_counter()
        self._step_start = self._wall_start
        self._recent_dts.clear()
        self._schedule_start_fraction = schedule_fraction

    def begin_step(
        self, phase: str, *, schedule_fraction: float | None = None,
    ) -> None:
        if phase != self.phase:
            self.start(phase, schedule_fraction=schedule_fraction)
        elif (
            self._schedule_start_fraction is None
            and schedule_fraction is not None
        ):
            self._schedule_start_fraction = schedule_fraction
        self._step_start = time.perf_counter()

    def tick(self, phase: str) -> None:
        if phase != self.phase:
            self.start(phase)
        now = time.perf_counter()
        self._recent_dts.append(now - self._step_start)
        if len(self._recent_dts) > self._window_size:
            self._recent_dts.pop(0)
        self.current_step += 1

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._wall_start

    @property
    def speed(self) -> float:
        elapsed = self.elapsed
        return self.current_step / elapsed if elapsed > 0 else 0.0

    @property
    def recent_speed(self) -> float:
        if not self._recent_dts:
            return self.speed
        return len(self._recent_dts) / sum(self._recent_dts)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}min"
        return f"{seconds / 3600:.1f}h"

    def format_schedule(self, *, fraction: float) -> str:
        speed = self.recent_speed
        fraction = min(max(fraction, 0.0), 1.0)
        start_fraction = self._schedule_start_fraction or 0.0
        completed = fraction - start_fraction
        eta = (
            self.elapsed * (1.0 - fraction) / completed
            if completed > 0.0 else float("inf")
        )
        return (
            f"[{fraction * 100:5.1f}%] "
            f"{speed:.2f} steps/s ETA {self._fmt_duration(eta)}"
        )

    def format_unbounded(self) -> str:
        label = "spin-up" if self.phase == "spinup" else "running"
        unit = "step" if self.current_step == 1 else "steps"
        return (
            f"[{label} {self.current_step} {unit}] "
            f"{self.recent_speed:.2f} steps/s"
        )


class ProgressRuntime:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.state = ProgressState()

    def _schedule_position(self) -> tuple[Any, Any]:
        runtime = self.owner._execution.step
        step = None if runtime is None else runtime.scheduled_step
        return self.owner.simulation_schedule, step

    def _phase(self) -> str:
        schedule, step = self._schedule_position()
        if schedule is None or step is None:
            return "unbounded"
        return step.phase

    def begin_step(self) -> None:
        schedule, step = self._schedule_position()
        fraction = None
        if schedule is not None and step is not None:
            elapsed = (step.start - schedule.execution_start).total_seconds()
            duration = (
                schedule.end - schedule.execution_start
            ).total_seconds()
            fraction = elapsed / duration
        self.state.begin_step(
            self._phase(), schedule_fraction=fraction,
        )

    def progress_tick(self) -> None:
        self.state.tick(self._phase())

    def format_progress(self) -> str:
        schedule, step = self._schedule_position()
        if schedule is None or step is None:
            return self.state.format_unbounded()
        elapsed = (step.end - schedule.execution_start).total_seconds()
        duration = (schedule.end - schedule.execution_start).total_seconds()
        return self.state.format_schedule(
            fraction=elapsed / duration,
        )

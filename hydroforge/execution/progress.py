"""Explicit progress state and service."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressState:
    total_steps: int = 0
    current_step: int = 0
    _wall_start: float = 0.0
    _last_tick: float = 0.0
    _recent_dts: list[float] = field(default_factory=list)
    _window_size: int = 50

    def start(self, total: int) -> None:
        self.total_steps = total
        self.current_step = 0
        self._wall_start = time.perf_counter()
        self._last_tick = self._wall_start
        self._recent_dts.clear()

    def tick(self) -> None:
        now = time.perf_counter()
        if self.current_step >= self.total_steps > 0:
            self.start(self.total_steps)
            now = time.perf_counter()
        if self.current_step > 0:
            self._recent_dts.append(now - self._last_tick)
            if len(self._recent_dts) > self._window_size:
                self._recent_dts.pop(0)
        self._last_tick = now
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

    def format_progress(self) -> str:
        speed = self.recent_speed
        remaining = max(0, self.total_steps - self.current_step)
        eta = remaining / speed if speed > 0 else float("inf")
        percent = min(
            self.current_step / max(self.total_steps, 1) * 100,
            100.0,
        )
        return (
            f"[{percent:5.1f}% {self.current_step}/{self.total_steps}] "
            f"{speed:.2f} steps/s ETA {self._fmt_duration(eta)}"
        )


class ProgressRuntime:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def set_total_steps(self, total: int) -> None:
        if type(total) is not int or total < 1:
            raise ValueError("total steps must be an exact positive int")
        self.owner._execution.total_steps = total
        self.owner._progress = ProgressState()
        self.owner._progress.start(total)

    def progress_tick(self) -> None:
        if self.owner._progress is not None:
            self.owner._progress.tick()

    def format_progress(self) -> str:
        if self.owner._progress is not None:
            return self.owner._progress.format_progress()
        return ""

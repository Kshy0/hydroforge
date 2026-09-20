"""Address-stable time bindings driven by a compiled device clock aggregator."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
from torch.utils._python_dispatch import _disable_current_modes

from hydroforge.compiler.step_fields import (
    CompiledStepFields,
    StepFieldCompileContext,
    StepFieldOutput,
)
from hydroforge.contracts.step_fields import (
    _BUILTIN_STEP_FIELDS,
    StepField,
    StepTime,
    _StepFieldValues,
)


@dataclass
class _StepFieldStorage:
    host: torch.Tensor
    device: torch.Tensor


class StepFieldRuntime:
    """Advance continuous time on device; upload only changed external controls.

    Builtins and expression dependencies are fused into one scalar aggregator.
    Host callbacks are an explicit escape hatch and never supply builtins.
    """

    def __init__(self, model: Any, *, execution: Any = None) -> None:
        self.device = torch.device(model.device)
        self.dtype = model.dtype
        self.backend = getattr(model, "_backend", "torch")
        self.execution = execution
        self.providers = dict(model._step_field_providers)
        self.expressions = getattr(model, "_step_field_expressions", {})
        sources = (*_BUILTIN_STEP_FIELDS, *self.expressions, *self.providers)
        self.slots = {name: index for index, name in enumerate(sources)}
        self.storage: dict[torch.dtype, _StepFieldStorage] = {}
        self.buffers: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.identities: set[int] = set()
        self.time: StepTime | None = None
        self.values: dict[str, int | float] = {}
        self.clock: torch.Tensor | None = None
        self.duration: torch.Tensor | None = None
        self.program: CompiledStepFields | None = None
        self._expected: Any = None
        self._duration_value: tuple[int, int] | None = None
        self._calendar_key: tuple[str, bool] | None = None
        self._calendar_active = False
        self._device_demand = False

    def concrete_dtype(self, field: StepField) -> torch.dtype:
        return self.dtype if field.dtype == "precision" else getattr(torch, field.dtype)

    def bind_many(self, fields: Iterable[StepField]) -> None:
        with _disable_current_modes(), torch.inference_mode(False):
            changed = False
            for field in fields:
                changed = self._allocate(field) or changed
            if changed and self.time is not None:
                self._refresh()

    def bind(self, field: StepField) -> torch.Tensor:
        self.bind_many((field,))
        return self.buffers[(field.source, self.concrete_dtype(field))]

    def _allocate(self, field: StepField) -> bool:
        if field.source not in self.slots:
            raise ValueError(f"unknown step field source {field.source!r}")
        dtype = self.concrete_dtype(field)
        key = (field.source, dtype)
        if key in self.buffers:
            return False
        if dtype not in self.storage:
            host = torch.zeros(len(self.slots), dtype=dtype)
            device = torch.zeros_like(host, device=self.device)
            self.storage[dtype] = _StepFieldStorage(host, device)
        slot = self.slots[field.source]
        buffer = self.storage[dtype].device[slot : slot + 1]
        self.buffers[key] = buffer
        self.identities.add(id(buffer))
        self._device_demand = self._device_demand or field.source not in self.providers
        self.invalidate_program()
        return True

    def _load_values(self) -> None:
        custom = {
            source: self.providers[source](self.time)
            for source in dict.fromkeys(source for source, _dtype in self.buffers)
            if source in self.providers and source not in self.values
        }
        if custom:
            self.values.update(_StepFieldValues(values=custom).values)

    def _upload_custom(self) -> None:
        dtypes = set()
        for source, dtype in self.buffers:
            if source in self.providers:
                self.storage[dtype].host[self.slots[source]] = self.values[source]
                dtypes.add(dtype)
        first = len(self.slots) - len(self.providers)
        for dtype in dtypes:
            storage = self.storage[dtype]
            storage.device[first:].copy_(storage.host[first:])

    def _date_key(self) -> tuple[str, bool]:
        date = self.time.current_time
        calendar = getattr(date, "calendar", "proleptic_gregorian")
        calendar = {
            "gregorian": "standard",
            "365_day": "noleap",
            "366_day": "all_leap",
        }.get(calendar, calendar)
        return calendar, getattr(date, "has_year_zero", True)

    def _context(self) -> StepFieldCompileContext:
        calendar, has_year_zero = self._date_key()
        return StepFieldCompileContext(
            calendar=calendar,
            has_year_zero=has_year_zero,
            outputs=tuple(
                StepFieldOutput(source, dtype, self.slots[source])
                for source, dtype in self.buffers
                if source not in self.providers
            ),
            expressions=self.expressions,
        )

    def _anchor(self, context: StepFieldCompileContext) -> None:
        date = self.time.current_time
        if date is None:
            year, ordinal, micros = 1, 0, 0
        else:
            year = date.year + (not context.has_year_zero and date.year < 0)
            start = date.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            ordinal = (date - start).days
            micros = (
                (date.hour * 60 + date.minute) * 60 + date.second
            ) * 1_000_000 + date.microsecond
        values = torch.tensor(
            (year, ordinal, micros, *self._duration_value), dtype=torch.int64
        )
        if self.clock is None:
            self.clock = values.to(self.device)
        else:
            self.clock.copy_(values)

    def _update_duration(self) -> None:
        delta = timedelta(seconds=self.time.step_seconds)
        value = (delta.days, delta.seconds * 1_000_000 + delta.microseconds)
        if value == self._duration_value:
            return
        host = torch.tensor(value, dtype=torch.int64)
        if self.duration is None:
            self.duration = host.to(self.device)
        else:
            self.duration.copy_(host)
        self._duration_value = value

    def _ensure_program(self, *, anchor: bool = False) -> CompiledStepFields | None:
        if not self._device_demand:
            return None
        key = self._date_key()
        if self.program is not None and key == self._calendar_key:
            self._update_duration()
            if anchor and self._calendar_active:
                self._anchor(self.program.context)
            return self.program
        context = self._context()
        if not context.outputs:
            return None
        calendar_needed = context.requires_calendar
        if calendar_needed:
            self.time.date
        if key != self._calendar_key:
            self.invalidate_program()
            anchor = True
        self._update_duration()
        if self.clock is None or (
            calendar_needed and (anchor or not self._calendar_active)
        ):
            self._anchor(context)
        self._calendar_key = key
        self._calendar_active = calendar_needed
        if self.program is None:
            capture = (
                self.execution.capture
                if self.execution is not None
                and self.execution.capture_mode == "cuda_graph"
                else None
            )
            self.program = CompiledStepFields(
                context,
                backend=self.backend,
                clock=self.clock,
                duration=self.duration,
                storage={dtype: slab.device for dtype, slab in self.storage.items()},
                capture=capture,
            )
        return self.program

    def _refresh(self) -> None:
        self._load_values()
        program = self._ensure_program()
        self._upload_custom()
        if program is not None:
            program.run(advance=False)

    def prepare(self, current_time: Any, step_seconds: float) -> None:
        continuous = (
            self.clock is not None
            and self._expected is not None
            and current_time == self._expected
        )
        self.time = StepTime(current_time, step_seconds)
        continuous = continuous and self._date_key() == self._calendar_key
        self.values.clear()
        if self.buffers:
            with _disable_current_modes(), torch.inference_mode(False):
                self._load_values()
                program = self._ensure_program(anchor=not continuous)
                self._upload_custom()
                if program is not None:
                    program.run(advance=continuous)
        self._expected = (
            None
            if current_time is None
            else current_time + timedelta(seconds=step_seconds)
        )

    def invalidate_program(self) -> None:
        if self.program is not None:
            program, self.program = self.program, None
            program.close()

    def close(self) -> None:
        try:
            self.invalidate_program()
        finally:
            self.buffers.clear()
            self.storage.clear()
            self.identities.clear()
            self.values.clear()
            self.time = None
            self.clock = None
            self.duration = None
            self._expected = None
            self._duration_value = None
            self._calendar_key = None
            self._calendar_active = False
            self._device_demand = False

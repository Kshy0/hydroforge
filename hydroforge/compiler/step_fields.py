# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Demand-driven scalar time aggregation as one-lane kernel IR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from hydroforge.core.expr import Expression
from hydroforge.core.graph import dependency_order
from hydroforge.kernels.codegen.expr import lower_expression
from hydroforge.kernels.codegen.ir import (
    Assign,
    Binary,
    Cast,
    Compare,
    Const,
    Expr,
    If,
    KernelFunction,
    Let,
    Load,
    Logical,
    Param,
    Select,
    Stmt,
    Store,
    Var,
    While,
    cast,
    type_of,
)

DAY_MICROSECONDS = 86_400_000_000

# ``(years, days)`` of each calendar's repeating cycle.
_CYCLES = {
    "proleptic_gregorian": (400, 146097),
    "julian": (4, 1461),
    "noleap": (1, 365),
    "all_leap": (1, 366),
    "360_day": (1, 360),
}
_MONTH_OFFSETS = (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


@dataclass(frozen=True)
class StepFieldOutput:
    source: str
    dtype: torch.dtype
    slot: int


@dataclass(frozen=True)
class StepFieldCompileContext:
    calendar: str
    has_year_zero: bool
    outputs: tuple[StepFieldOutput, ...]
    expressions: Mapping[str, Expression]

    def dependencies(self) -> tuple[str, ...]:
        expressions = self.expressions
        return dependency_order(
            dict.fromkeys(output.source for output in self.outputs),
            lambda name: expressions[name].dependencies if name in expressions else (),
            cycle_message=lambda cycle: (
                f"cyclic device step field expression: {cycle[0]!r}"
            ),
        )

    @property
    def requires_calendar(self) -> bool:
        return any(
            source not in self.expressions and source != "step_seconds"
            for source in self.dependencies()
        )

    @property
    def dtypes(self) -> tuple[torch.dtype, ...]:
        """Output dtypes in slab order: ``slab_<i>`` holds ``dtypes[i]``."""

        return tuple(dict.fromkeys(output.dtype for output in self.outputs))


_I64 = torch.int64
_CLOCK = Param("clock", _I64, "read_write")
_DURATION = Param("duration", _I64, "read")
_ADVANCE = Var("advance", torch.bool)


def _integer(value: int) -> Expr:
    # A converted literal stays an int64 value in every dialect, also where
    # a bare literal would be a compile-time constant (Triton).
    return Cast(Const(value), _I64)


def _load(param: Param, index: int) -> Load:
    return Load(param.name, Const(index), param.type)


class _Calendar:
    """Year rules of one calendar over int64 years."""

    def __init__(self, calendar: str) -> None:
        self.calendar = calendar

    def leap(self, year: Var) -> Expr:
        """The leap days of ``year`` (0 or 1)."""

        def divisible(divisor: int, holds: bool = True) -> Expr:
            remainder = Binary("%", year, Const(divisor))
            return Compare("==" if holds else "!=", remainder, Const(0))

        if self.calendar == "all_leap":
            return _integer(1)
        if self.calendar in {"noleap", "360_day"}:
            return _integer(0)
        julian = divisible(4)
        gregorian = Logical(
            "and",
            (julian, Logical("or", (divisible(100, False), divisible(400)))),
        )
        if self.calendar == "julian":
            return Cast(julian, _I64)
        if self.calendar == "standard":
            before = Compare("<", year, Const(1582))
            return Cast(Select(before, julian, gregorian), _I64)
        return Cast(gregorian, _I64)

    def year_length(self, year: Var) -> Expr:
        if self.calendar == "360_day":
            return _integer(360)
        length = Binary("+", Const(365), self.leap(year))
        if self.calendar == "standard":
            # 1582 skips ten days of October.
            switch = Compare("==", year, Const(1582))
            skipped = Select(switch, Const(10), Const(0))
            length = Binary("-", length, skipped)
        return length


def step_field_program(
    context: StepFieldCompileContext, real: torch.dtype
) -> KernelFunction:
    """One aggregator of the demanded fields, with floating values in
    ``real``; ``advance`` moves the clock one step first.

    Buffers: ``clock`` holds ``(year, ordinal day, microseconds of the day,
    step days, step microseconds)``, ``duration`` the step's ``(days,
    microseconds)``, and ``slab_<i>`` the outputs of ``context.dtypes[i]``.
    """

    body: list[Stmt] = []

    def let(name: str, value: Expr) -> Var:
        var = Var(name, _I64)
        body.append(Let(var, value))
        return var

    duration_days = let("duration_days", _load(_DURATION, 0))
    duration_micros = let("duration_micros", _load(_DURATION, 1))
    values: dict[str, Expr] = {
        "step_seconds": Binary(
            "+",
            Binary("*", Cast(duration_days, real), Const(86400.0)),
            Binary("/", Cast(duration_micros, real), Const(1000000.0)),
        )
    }
    if context.requires_calendar:
        calendar = _Calendar(context.calendar)
        year = let("clock_year", _load(_CLOCK, 0))
        ordinal = let("ordinal", _load(_CLOCK, 1))
        micros = let("micros", _load(_CLOCK, 2))
        year_length = let("year_length", calendar.year_length(year))
        advance: list[Stmt] = [
            Assign(micros, Binary("+", micros, _load(_CLOCK, 4))),
            Assign(
                ordinal,
                Binary(
                    "+",
                    Binary("+", ordinal, _load(_CLOCK, 3)),
                    Binary("/", micros, Const(DAY_MICROSECONDS)),
                ),
            ),
            Assign(micros, Binary("%", micros, Const(DAY_MICROSECONDS))),
        ]
        cycle = _CYCLES.get(context.calendar)
        if cycle is not None:
            cycle_years, cycle_days = cycle
            cycles = Var("cycles", _I64)
            advance += [
                Let(cycles, Binary("/", ordinal, Const(cycle_days))),
                Assign(
                    ordinal,
                    Binary("-", ordinal, Binary("*", cycles, Const(cycle_days))),
                ),
                Assign(
                    year,
                    Binary("+", year, Binary("*", cycles, Const(cycle_years))),
                ),
            ]
        advance += [
            Assign(year_length, calendar.year_length(year)),
            While(
                Compare(">=", ordinal, year_length),
                (
                    Assign(ordinal, Binary("-", ordinal, year_length)),
                    Assign(year, Binary("+", year, Const(1))),
                    Assign(year_length, calendar.year_length(year)),
                ),
            ),
        ]
        body.append(If(_ADVANCE, tuple(advance)))
        body.append(Assign(year_length, calendar.year_length(year)))
        nominal = ordinal
        if context.calendar == "standard":
            late = Logical(
                "and",
                (
                    Compare("==", year, Const(1582)),
                    Compare(">=", ordinal, Const(277)),
                ),
            )
            nominal = Binary("+", ordinal, Select(late, Const(10), Const(0)))
        nominal_day = let("nominal_day", nominal)
        leap_day = let("leap_day", calendar.leap(year))
        month = let("calendar_month", _integer(0))
        month_start = let("month_start", _integer(0))
        for number, offset in enumerate(_MONTH_OFFSETS, 1):
            if context.calendar == "360_day":
                start = Const(number * 30)
            elif number == 1:
                start = Const(offset)
            else:
                start = Binary("+", Const(offset), leap_day)
            reached = Compare(">=", nominal_day, start)
            body.append(Assign(month, Select(reached, Const(number), month)))
            body.append(Assign(month_start, Select(reached, start, month_start)))
        for index, value in enumerate(
            (year, ordinal, micros, duration_days, duration_micros)
        ):
            body.append(Store(_CLOCK.name, Const(index), value))
        one = Const(1)
        values |= {
            "doy": Binary("+", nominal_day, one),
            "month_idx": month,
            "days_in_year": year_length,
            "day_seconds": Binary("/", Cast(micros, real), Const(1000000.0)),
            "julian": Binary(
                "+",
                Cast(nominal_day, real),
                Binary(
                    "/",
                    Binary("/", Cast(micros, real), Const(1000000.0)),
                    Const(86400.0),
                ),
            ),
            "year": year
            if context.has_year_zero
            else Binary(
                "-",
                year,
                Select(
                    Compare("<=", year, Const(0)),
                    Const(1),
                    Const(0),
                ),
            ),
            "month": Binary("+", month, one),
            "day": Binary("+", Binary("-", nominal_day, month_start), one),
        }
    fields: dict[str, Var] = {}
    for index, source in enumerate(context.dependencies()):
        expression = context.expressions.get(source)
        value = (
            values[source]
            if expression is None
            else lower_expression(expression, fields, real)
        )
        field = fields[source] = Var(f"field_{index}", type_of(value))
        body.append(Let(field, value))
    dtypes = context.dtypes
    for output in context.outputs:
        body.append(
            Store(
                f"slab_{dtypes.index(output.dtype)}",
                Const(output.slot),
                cast(fields[output.source], output.dtype),
            )
        )
    slabs = tuple(
        Param(f"slab_{index}", dtype, "write") for index, dtype in enumerate(dtypes)
    )
    return KernelFunction(
        "aggregate_time",
        (_CLOCK, _DURATION, *slabs, Param(_ADVANCE.name, torch.bool)),
        tuple(body),
    )

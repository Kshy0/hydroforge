"""Declarative, read-only calendar inputs for model-bound kernel buffers."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, FiniteFloat, PrivateAttr, field_validator, model_validator

from hydroforge.contracts.naming import Identifier
from hydroforge.contracts.temporal import DateLike
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.statistics.ir import Expression


@dataclass(frozen=True)
class StepTime:
    """Trusted outer-step time passed to pure model-specific scalar providers."""

    current_time: DateLike | None
    step_seconds: float

    @cached_property
    def date(self) -> DateLike:
        if self.current_time is None:
            raise ValueError(
                "calendar step fields require initial_time or a simulation_schedule"
            )
        return self.current_time

    @cached_property
    def doy(self) -> int:
        return self.date.timetuple().tm_yday

    @cached_property
    def day_seconds(self) -> float:
        date = self.date
        return (
            date.hour * 3600
            + date.minute * 60
            + date.second
            + date.microsecond / 1_000_000
        )

    @cached_property
    def days_in_year(self) -> int:
        date = self.date
        last_day = 30 if getattr(date, "calendar", None) == "360_day" else 31
        return (
            date.replace(month=12, day=last_day) - date.replace(month=1, day=1)
        ).days + 1


StepFieldProvider = Callable[[StepTime], int | float]
StepFieldDType = Literal["int32", "int64", "float32", "float64", "precision"]
_BUILTIN_STEP_FIELDS = (
    "doy",
    "month_idx",
    "day_seconds",
    "days_in_year",
    "julian",
    "step_seconds",
    "year",
    "month",
    "day",
)


class StepField(HydroForgeModel):
    """Bind one canonical read-only buffer to a framework-owned step value."""

    source: Identifier
    dtype: StepFieldDType = "precision"


def step_field(source: str, *, dtype: StepFieldDType = "precision") -> StepField:
    """Declare demand for an outer-step calendar value, not a host scalar ABI."""

    return StepField(source=source, dtype=dtype)


class _StepFieldProviders(HydroForgeModel):
    providers: Mapping[str, StepFieldProvider]

    @model_validator(mode="after")
    def _validate_providers(self) -> Self:
        for name, provider in self.providers.items():
            if not name.isidentifier() or name in _BUILTIN_STEP_FIELDS:
                raise ValueError(f"invalid or reserved step field provider: {name!r}")
            try:
                inspect.signature(provider).bind(None)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"step field provider {name!r} must accept StepTime"
                ) from error
        object.__setattr__(self, "providers", MappingProxyType(dict(self.providers)))
        return self


class _StepFieldValues(HydroForgeModel):
    """Validate results once at the user-supplied provider boundary."""

    values: dict[str, int | FiniteFloat]

    @field_validator("values", mode="before")
    @classmethod
    def _host_scalars(cls, values: dict) -> dict:
        for name, value in values.items():
            if type(value) not in (int, float):
                raise ValueError(
                    f"step field provider {name!r} must return a Python int or float"
                )
        return values


class _StepFieldExpressions(HydroForgeModel):
    """Compile custom device expressions once at the declaration boundary."""

    expressions: Mapping[str, str]
    host_sources: frozenset[str] = Field(default=frozenset(), exclude=True)
    _compiled: Mapping[str, Expression] = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self) -> Self:
        from hydroforge.statistics.ir import ExpressionSource, parse_value_source

        known = set(_BUILTIN_STEP_FIELDS) | set(self.expressions)
        compiled = {}
        for name, expression in self.expressions.items():
            if (
                not name.isidentifier()
                or name in _BUILTIN_STEP_FIELDS
                or name in self.host_sources
            ):
                raise ValueError(f"invalid or reserved device step field: {name!r}")
            try:
                source = parse_value_source(expression, known)
            except ValueError as error:
                raise ValueError(f"device step field {name!r}: {error}") from error
            if not isinstance(source, ExpressionSource):
                raise ValueError("device step fields require scalar expressions")
            compiled[name] = source.expression
        ordered = {}
        visiting = set()

        def visit(name: str) -> None:
            if name in ordered or name not in compiled:
                return
            if name in visiting:
                raise ValueError(f"cyclic device step field expression: {name!r}")
            visiting.add(name)
            for dependency in compiled[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            ordered[name] = compiled[name]

        for name in compiled:
            visit(name)
        self._compiled = MappingProxyType(ordered)
        object.__setattr__(
            self, "expressions", MappingProxyType(dict(self.expressions))
        )
        return self

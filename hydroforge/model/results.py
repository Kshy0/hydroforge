# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Statistics results of one model, read through ``model.results``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import ConfigDict, Field, validate_call

from hydroforge.core.expr import parse_operation
from hydroforge.execution.session import ModelRuntime
from hydroforge.statistics.storage import StoragePlan

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel
    from hydroforge.statistics.runtime import StatisticsRuntime
    from hydroforge.statistics.sinks import MemorySink

_QUERY = validate_call(config=ConfigDict(strict=True))


def _operation(op: str) -> str:
    """Spell an operation as ``OutputConfig.variables`` canonicalizes it."""

    canonical = op.lower()
    return canonical if canonical == "static" else parse_operation(canonical).spelling


class ModelResults:
    """Query the statistics a model declared; results need ``sink="memory"``.

    Declaration errors are reported from the compiled plan, so they appear
    before materialization; reading any value requires a materialized,
    healthy runtime.
    """

    __slots__ = ("_model",)

    def __init__(self, model: AbstractModel) -> None:
        self._model = model

    def _statistics(
        self, key: tuple[str, str] | None = None, *, retained: bool = False
    ) -> StatisticsRuntime | None:
        model = self._model
        output = model.plan.output
        if output.windows is None and (key is not None or retained):
            raise ValueError(f"{type(model).__name__} has no statistics declaration")
        if key is not None:
            declared = [
                (item.name, item.operation)
                for item in output.outputs
                if item.operation != "static"
            ]
            if key not in declared:
                listed = ", ".join(f"{name}/{op}" for name, op in declared)
                raise ValueError(
                    f"statistics output {key[0]!r}/{key[1]!r} is not declared; "
                    f"declared outputs: {listed}"
                )
        if retained and output.config.sink != "memory":
            raise ValueError(
                'in-memory results require OutputConfig(sink="memory"); this '
                f"model writes NetCDF output under {output.directory}"
            )
        runtime = ModelRuntime.of(model)
        what = f"{type(model).__name__}.results"
        runtime.require_materialized(what)
        runtime.require_healthy(what)
        return runtime.statistics

    def _retained(self, key: tuple[str, str] | None = None) -> MemorySink:
        return self._statistics(key, retained=True).sink

    @_QUERY
    def get(
        self,
        name: str,
        op: str = "mean",
        *,
        as_stacked: bool = True,
        start: int | None = None,
        stop: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return isolated copies of one retained output, stacked over time."""

        op = _operation(op)
        return self._retained((name, op)).get(
            StoragePlan.output(name, op), as_stacked=as_stacked, start=start, stop=stop
        )

    @_QUERY
    def all(
        self,
        *,
        as_stacked: bool = True,
        start: int | None = None,
        stop: int | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Return isolated copies of every retained output by output name."""

        return self._retained().all(as_stacked=as_stacked, start=start, stop=stop)

    @_QUERY
    def accumulator(self, name: str, op: str = "mean") -> torch.Tensor:
        """Return a differentiable snapshot without exposing captured storage."""

        op = _operation(op)
        return self._statistics((name, op)).accumulator(name, op)

    @_QUERY
    def pop(self, name: str, op: str = "mean") -> torch.Tensor | None:
        """Pop the newest retained result without keeping its history."""

        op = _operation(op)
        return self._retained((name, op)).pop(StoragePlan.output(name, op))

    @_QUERY
    def drain(
        self,
        max_steps: Annotated[int, Field(ge=0)] | None = None,
        *,
        as_stacked: bool = True,
    ) -> dict[str, Any]:
        """Copy and release the oldest retained samples; keep the timeline."""

        return self._retained().drain(max_steps, as_stacked=as_stacked)

    @property
    def time_index(self) -> int:
        """Number of finalized output windows; 0 without statistics."""

        statistics = self._statistics()
        return 0 if statistics is None else statistics.get_time_index()

    def reset(self) -> None:
        """Restart the output timeline and drop retained results."""

        statistics = self._statistics()
        if statistics is not None:
            statistics.reset_time_index()

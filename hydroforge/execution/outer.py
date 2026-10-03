"""Cached compiled operator scopes outside the physical substep clock."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from hydroforge.execution.context import (
    InvocationScope,
    close_runner,
    specialization_key,
)
from hydroforge.execution.operators import record_operator_scope
from hydroforge.execution.substeps import Recorded, cached_program

if TYPE_CHECKING:
    from hydroforge.execution.runtime import ModelExecution


class _OuterProgram:
    def __init__(self, execution: ModelExecution, operators: Any) -> None:
        if operators is None or not operators.operators:
            raise RuntimeError("outer operator scope produced an empty program")
        self.executor = execution.executor
        self.operators = operators
        self.runner = self.executor.outer(operators)

    def launch(self) -> None:
        self.runner.run()

    def close(self) -> None:
        runner, self.runner = self.runner, None
        operators, self.operators = self.operators, None
        close_runner(
            self.executor, runner, (operators,), scope="outer operator program"
        )


class _OnceScope(InvocationScope):
    def __init__(self, context: Any, *, key: tuple[Any, ...]) -> None:
        super().__init__(context)
        self.key = key

    def _iterate(self) -> Generator[None, None, None]:
        context = self.context
        program = yield from cached_program(
            context.execution,
            self.key,
            (context.time_step, context.requested_sub_steps),
            self._record,
        )
        program.launch()
        context.scopes.pop()

    def _record(self) -> Generator[None, None, Any]:
        execution = self.context.execution
        with record_operator_scope(execution, scope_kind="outer") as recording:
            yield None
        program = recording.program
        return Recorded(
            (program.fingerprint({}),),
            (program,),
            lambda: _OuterProgram(execution, program),
        )


def outer_scope(
    context: Any, *, site: tuple[Any, int], specialization: Any
) -> _OnceScope:
    """Declare one cached once-per-outer-step operator sequence."""

    key = context.claim_outer_scope(site, specialization_key(specialization))
    return _OnceScope(context, key=key)

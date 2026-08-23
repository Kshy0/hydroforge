"""Cached compiled operator scopes outside the physical substep clock."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Iterator

from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.validation import HydroForgeModel

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


_MISSING = object()


class _OuterScopeRequest(HydroForgeModel):
    specialization: Any = None

    _key: Any = PrivateAttr()

    @model_validator(mode="after")
    def _validate_specialization(self):
        from hydroforge.execution.substeps import _specialization_key

        self._key = _specialization_key(self.specialization)
        return self

    @property
    def specialization_key(self) -> Any:
        return self._key


class _OuterProgram:
    def __init__(self, model: AbstractModel, operators: Any) -> None:
        if operators is None or not operators.operators:
            raise RuntimeError("outer operator scope produced an empty program")
        self.capture = model._execution.capture
        self.capture_mode = model._execution.capture_mode
        self.operators = operators
        self.graph = None
        if self.capture_mode == "metal_icb":
            self.operators.prepare_metal(self.capture)

    def launch(self) -> None:
        if (
            self.capture_mode == "cuda_graph"
            and self.operators.cuda_graph_capture_safe
        ):
            if self.graph is None:
                self.graph = self.capture.capture_cuda(
                    self.operators.launch,
                    mutated_state=self.operators.mutated_tensors,
                )
            self.graph.replay()
            return
        if self.capture_mode == "metal_icb":
            self.operators.reset_metal_errors()
        self.operators.launch()
        self.operators.check_metal_errors()

    def close(self) -> None:
        if self.graph is not None:
            self.capture.release(self.graph)
            self.graph = None
        operators, self.operators = self.operators, None
        if operators is not None:
            operators.close(self.capture)


class _OnceScope:
    def __init__(self, runtime: "OuterRuntime", *, key: tuple[Any, ...]) -> None:
        self.runtime = runtime
        self.key = key

    def __iter__(self) -> Iterator[None]:
        from hydroforge.execution.operators import record_operator_scope

        execution = self.runtime.model._execution
        step = self.runtime.step
        step.begin_outer_scope_execution()
        programs = execution.programs
        program = programs.get(self.key, _MISSING)
        if program is _MISSING:
            with record_operator_scope(
                self.runtime.model, scope_kind="outer",
            ) as recording:
                yield None
            program = _OuterProgram(self.runtime.model, recording.program)
            programs[self.key] = program
        program.launch()
        step.complete_outer_scope_execution()


class OuterRuntime:
    """Declare cached once-per-outer-step operator sequences."""

    def __init__(self, model: Any, step: Any) -> None:
        self.model = model
        self.step = step

    def once(self, *, specialization: Any = None) -> _OnceScope:
        request = _OuterScopeRequest(specialization=specialization)
        caller = sys._getframe(1)
        lexical_site = (caller.f_code, caller.f_lasti)
        key = self.step.claim_outer_scope(
            site=lexical_site,
            specialization=request.specialization_key,
        )
        return _OnceScope(self, key=key)

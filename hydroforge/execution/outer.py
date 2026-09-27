"""Cached compiled operator scopes outside the physical substep clock."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.execution.program import _close_program_resources

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


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
        if self.capture_mode == "cuda_graph" and self.operators.cuda_graph_capture_safe:
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
        graph, self.graph = self.graph, None
        operators, self.operators = self.operators, None
        _close_program_resources(
            self.capture, (graph,), (operators,), scope="outer operator program"
        )


class _OnceScope:
    def __init__(self, runtime: OuterRuntime, *, key: tuple[Any, ...]) -> None:
        self.runtime = runtime
        self.key = key

    def __iter__(self) -> Iterator[None]:
        from hydroforge.execution.substeps import _cached_program

        step = self.runtime.step
        step.begin_outer_scope_execution()
        program = yield from _cached_program(
            self.runtime.model._execution,
            self.key,
            (step.time_step, step.requested_sub_steps),
            self._record,
        )
        program.launch()
        step.complete_outer_scope_execution()

    def _record(self) -> Generator[None, None, Any]:
        from hydroforge.execution.operators import record_operator_scope
        from hydroforge.execution.substeps import _Recorded

        model = self.runtime.model
        with record_operator_scope(model, scope_kind="outer") as recording:
            yield None
        program = recording.program
        return _Recorded(
            (program.fingerprint({}),),
            (program,),
            lambda: _OuterProgram(model, program),
        )


class OuterRuntime:
    """Declare cached once-per-outer-step operator sequences."""

    def __init__(self, model: Any, step: Any) -> None:
        self.model = model
        self.step = step

    def once(
        self,
        *,
        site: tuple[Any, int],
        specialization: Any = None,
    ) -> _OnceScope:
        request = _OuterScopeRequest(specialization=specialization)
        key = self.step.claim_outer_scope(
            site=site,
            specialization=request.specialization_key,
        )
        return _OnceScope(self, key=key)

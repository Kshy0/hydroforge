"""Cached compiled operator scopes outside the physical substep clock."""
from __future__ import annotations

from typing import Any, Iterator


_MISSING = object()


class _OuterProgram:
    def __init__(self, model: Any, operators: Any) -> None:
        if operators is None or not operators.operators:
            raise RuntimeError("outer operator scope produced an empty program")
        self.capture = model._execution.capture
        self.capture_mode = model._execution.capture_mode
        self.operators = operators
        self.graph = None
        if self.capture_mode == "metal_icb":
            self.operators.prepare_metal(self.capture)

    def launch(self) -> None:
        self.operators.require_stable_bindings()
        if self.capture_mode == "cuda_graph":
            if self.graph is None:
                self.graph = self.capture.capture_cuda(
                    self.operators.launch,
                    mutated_state=self.operators.mutated_tensors,
                )
            self.graph.replay()
            return
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

        programs = self.runtime.model._execution.programs
        program = programs.get(self.key, _MISSING)
        if program is _MISSING:
            with record_operator_scope(self.runtime.model) as recording:
                yield None
            program = _OuterProgram(self.runtime.model, recording.program)
            programs[self.key] = program
        elif not isinstance(program, _OuterProgram):
            raise RuntimeError("cached outer operator program has the wrong kind")
        program.launch()


class OuterRuntime:
    """Declare cached once-per-outer-step operator sequences."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def once(self, *, specialization: Any = None) -> _OnceScope:
        from hydroforge.execution.substeps import _specialization_key

        step = self.model._execution.active_step
        if step is None:
            raise RuntimeError("outer operator scopes require @managed_step")
        key = step.claim_outer_scope(
            specialization=_specialization_key(specialization),
        )
        return _OnceScope(self, key=key)

    def cached(self, *, name: str, specialization: Any = None) -> _OnceScope:
        """Return a named program that may be replayed repeatedly in one step.

        This is intended for transactional algorithms that inspect a device
        result on the host, restore state, and replay the same compiled pass
        with another explicit specialization.
        """

        from hydroforge.execution.substeps import _specialization_key

        if not isinstance(name, str) or not name:
            raise ValueError("cached outer program name must be a non-empty string")
        step = self.model._execution.active_step
        if step is None:
            raise RuntimeError("cached outer programs require @managed_step")
        key = (
            step.program_owner, "outer_cached", name,
            _specialization_key(specialization),
        )
        return _OnceScope(self, key=key)

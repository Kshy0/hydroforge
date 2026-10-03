"""Backend-neutral compiled loops over recorded operator programs.

A loop owns its device control scalars and recorded bodies.  ``bind`` asks
the model's executor for the runner that launches the loop on the model
backend; the loop itself never branches on the backend.  The torch
iterations here are the tensor implementation of the control rules in
:mod:`hydroforge.kernels.backends.loop_control`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils._python_dispatch import _disable_current_modes

from hydroforge.core.errors import SubstepCompileError
from hydroforge.execution.context import SubstepFrame, close_runner
from hydroforge.execution.operators import PredicateLoopOperator

if TYPE_CHECKING:
    from hydroforge.execution.runtime import ModelExecution


def distributed_capture_safe(*programs: Any) -> bool:
    """Return whether recorded loop programs may be device-captured multi-rank.

    Collectives must pass the host rank handshake before every launch, so a
    program containing any non-capturable operator other than a nested
    predicate loop (whose body is inspected instead) stays eager.
    """

    pending = [program for program in programs if program is not None]
    while pending:
        program = pending.pop()
        for operator in program.operators:
            if isinstance(operator, PredicateLoopOperator):
                body = operator.loop.body
                if body is None:
                    return False
                pending.append(body)
            elif not getattr(operator, "cuda_graph_capture_safe", True):
                return False
    return True


class FixedLoop:
    """Fixed-width loop over one recorded body and an optional final tail."""

    def __init__(self, execution: ModelExecution) -> None:
        self.executor = execution.executor
        with _disable_current_modes(), torch.inference_mode(False):
            options = {"device": execution.device, "dtype": torch.int32}
            # ``(counter, continue, count)``: one copy from ``initial`` resets
            # a launch, and ``initial`` changes only with the count.
            self.controls = torch.zeros(3, **options)
            self.initial = torch.tensor((0, 1, 1), **options)
            self.weight = torch.zeros(1, device=execution.device, dtype=execution.dtype)
            self.counter = self.controls[0:1]
            self.continue_flag = self.controls[1:2]
            self.count = self.controls[2:3]
            self._initial_count = self.initial[2:3]
        self.frame = SubstepFrame(index=self.counter, dt=self.weight)
        self.body: Any = None
        self.final: Any = None
        self.controlled = False
        self.weighted = False
        self.runner: Any = None
        # Host copies of the count in ``initial`` and of the width in
        # ``weight``; a body that writes its width is refilled every launch.
        self._count = 1
        self._width: float | None = None
        self._keeps_width = True

    @property
    def programs(self) -> tuple[Any, ...]:
        return tuple(
            program for program in (self.body, self.final) if program is not None
        )

    def bind(self, body: Any, final: Any | None = None) -> FixedLoop:
        """Attach the recorded body and build its backend runner."""

        if not body.operators:
            raise SubstepCompileError(
                "fixed substep produced an empty operator IR; backend kernels "
                "must be registered through BackendRegistry + KernelSpec"
            )
        self.body = body
        self.final = final
        programs = self.programs
        self.controlled = any(
            program.references_tensor(self.counter) for program in programs
        )
        self.weighted = any(
            program.references_tensor(self.weight) for program in programs
        )
        self._keeps_width = not any(
            tensor is self.weight
            for program in programs
            for tensor in program.mutated_tensors
        )
        self.runner = self.executor.fixed(self)
        return self

    def prepare(
        self, count: int, width: float, *, controls: bool, weight: bool
    ) -> None:
        """Write the controls of one launch; values already held stay."""

        if controls:
            if count != self._count:
                self._initial_count.fill_(count)
                self._count = count
            self.controls.copy_(self.initial)
        if weight and width != self._width:
            self.weight.fill_(width)
            if self._keeps_width:
                self._width = width

    def reset(self) -> None:
        self.controls.copy_(self.initial)

    def execute(self, count: int, duration: float, step: Any) -> int:
        return self.runner.run(count, duration, step)

    def close(self) -> None:
        runner, self.runner = self.runner, None
        programs = self.programs
        self.body = self.final = None
        close_runner(self.executor, runner, programs, scope="fixed substep loop")


class PredicateLoop:
    """Nested loop controlled by a body-authored device predicate."""

    def __init__(self, execution: ModelExecution, *, maximum_steps: int) -> None:
        self.executor = execution.executor
        self.maximum_steps = maximum_steps
        with _disable_current_modes(), torch.inference_mode(False):
            options = {"device": execution.device, "dtype": torch.int32}
            # One slab lets every launch reset (predicate, counter, continue)
            # with a single device copy.
            self.controls = torch.zeros(3, **options)
            self.initial = torch.tensor((0, 0, 1), **options)
            self.maximum_count = torch.full((1,), maximum_steps, **options)
            self.zero_count = torch.zeros(1, **options)
            self.one_count = torch.ones(1, **options)
            self.has_more = torch.zeros(1, device=execution.device, dtype=torch.bool)
            self.under_limit = torch.zeros_like(self.has_more)
            self.predicate = self.controls[0:1]
            self.counter = self.controls[1:2]
            self.continue_flag = self.controls[2:3]
        self.body: Any = None
        self.runner: Any = None

    @property
    def programs(self) -> tuple[Any, ...]:
        return () if self.body is None else (self.body,)

    @property
    def control_state(self) -> tuple[torch.Tensor, ...]:
        """Control tensors one torch iteration writes."""
        return (
            self.predicate,
            self.counter,
            self.continue_flag,
            self.has_more,
            self.under_limit,
        )

    def bind(self, body: Any) -> PredicateLoop:
        """Attach the recorded loop body and build its backend runner."""

        if not body.operators:
            raise SubstepCompileError("predicate loop produced an empty operator IR")
        self.body = body
        self.runner = self.executor.predicate(self)
        return self

    def reset(self) -> None:
        self.controls.copy_(self.initial)

    def iterate(self) -> None:
        self.predicate.zero_()
        self.body.launch()
        self.counter.add_(self.one_count)
        torch.ne(self.predicate, self.zero_count, out=self.has_more)
        torch.lt(self.counter, self.maximum_count, out=self.under_limit)
        torch.logical_and(self.has_more, self.under_limit, out=self.has_more)
        self.continue_flag.copy_(self.has_more)

    def execute(self) -> None:
        self.runner.run()

    def close(self) -> None:
        runner, self.runner = self.runner, None
        programs = self.programs
        self.body = None
        close_runner(self.executor, runner, programs, scope="predicate loop")


class AdaptiveLoop:
    """Adaptive loop over one recorded width proposal and physics body."""

    def __init__(
        self,
        execution: ModelExecution,
        *,
        candidate_dt: torch.Tensor,
        dt: torch.Tensor,
        maximum_dt: float,
        maximum_steps: int,
    ) -> None:
        self.executor = execution.executor
        self.candidate = candidate_dt.view(1)
        self.time_step = dt.view(1)
        self.maximum = maximum_dt
        self.maximum_steps = maximum_steps
        # The first loop build may happen under ``torch.inference_mode``.
        # Runtime-owned controls must remain ordinary tensors because they are
        # also mutated by capture setup and cleanup outside inference mode.
        with torch.inference_mode(False):
            options = dict(device=candidate_dt.device, dtype=candidate_dt.dtype)
            counts = dict(device=candidate_dt.device, dtype=torch.int32)
            self.duration = torch.zeros(1, **options)
            self.elapsed = torch.zeros(1, **options)
            self.counter = torch.zeros(1, **counts)
            self.continue_flag = torch.zeros_like(self.counter)
            self.error_flag = torch.zeros_like(self.counter)
            # Stable scalar scratch belongs to the loop, not to an iteration.
            # CUDA captures these addresses and eager execution performs no
            # per-substep tensor allocation.
            self.remaining = torch.zeros(1, **options)
            self.accepted = torch.zeros(1, **options)
            self.predicate_a = torch.zeros(
                1, device=candidate_dt.device, dtype=torch.bool
            )
            self.predicate_b = torch.zeros_like(self.predicate_a)
            self.predicate_c = torch.zeros_like(self.predicate_a)
            self.maximum_value = torch.full((1,), maximum_dt, **options)
            self.zero_value = torch.zeros(1, **options)
            self.maximum_count = torch.full((1,), maximum_steps, **counts)
            self.one_count = torch.ones_like(self.maximum_count)
            # Host-read loop status: ``(error, continue, dt)`` per host-driven
            # substep and ``(error, counter)`` per device loop, each fetched
            # with one small transfer into reused (pinned on CUDA) storage.
            self.status = torch.zeros(3, **options)
            self.completion = torch.zeros(2, **counts)
            pinned = candidate_dt.device.type == "cuda"
            self.status_host = torch.zeros(3, dtype=dt.dtype, pin_memory=pinned)
            self.completion_host = torch.zeros(2, dtype=torch.int32, pin_memory=pinned)
        self.status_sources = (self.error_flag, self.continue_flag, dt.view(1))
        self.completion_sources = (self.error_flag, self.counter)
        self.frame = SubstepFrame(index=self.counter, dt=dt)
        self.proposal: Any = None
        self.body: Any = None
        self.runner: Any = None

    @property
    def programs(self) -> tuple[Any, ...]:
        return tuple(
            program for program in (self.proposal, self.body) if program is not None
        )

    @property
    def control_state(self) -> tuple[torch.Tensor, ...]:
        """Control tensors of one torch iteration; capture rollback restores them."""
        return (
            self.duration,
            self.elapsed,
            self.counter,
            self.continue_flag,
            self.error_flag,
            self.remaining,
            self.accepted,
            self.predicate_a,
            self.predicate_b,
            self.predicate_c,
            self.maximum_value,
            self.zero_value,
            self.maximum_count,
            self.one_count,
        )

    def bind(self, proposal: Any, body: Any) -> AdaptiveLoop:
        """Attach the recorded proposal and physics body; build the runner."""

        if not proposal.operators:
            raise SubstepCompileError(
                "adaptive dt proposal produced an empty operator IR"
            )
        if not body.operators:
            raise SubstepCompileError(
                "adaptive physics body produced an empty operator IR"
            )
        self.proposal = proposal
        self.body = body
        self.runner = self.executor.adaptive(self)
        return self

    def reset(self) -> None:
        self.elapsed.zero_()
        self.counter.zero_()
        self.continue_flag.fill_(1)
        self.error_flag.zero_()

    def iterate(self) -> None:
        self.candidate.copy_(self.maximum_value)
        self.proposal.launch()
        self.remaining.copy_(self.duration).sub_(self.elapsed)
        torch.minimum(self.candidate, self.remaining, out=self.accepted)
        # ``accepted != accepted`` is exactly the NaN test needed here.  A
        # positive infinity cannot survive minimum(candidate, finite
        # remaining), while negative infinity is caught by <= 0.
        torch.eq(self.accepted, self.accepted, out=self.predicate_a)
        torch.logical_not(self.predicate_a, out=self.predicate_a)
        torch.le(self.accepted, self.zero_value, out=self.predicate_b)
        torch.logical_or(self.predicate_a, self.predicate_b, out=self.predicate_a)
        # A bad proposal must terminate the device WHILE node.  Substitute the
        # finite positive remainder so the already-captured physics tail does
        # not receive zero/NaN before the host reports the strict error.
        torch.where(self.predicate_a, self.remaining, self.accepted, out=self.time_step)
        self.body.launch()
        self.elapsed.add_(self.time_step)
        self.counter.add_(self.one_count)
        torch.ge(self.counter, self.maximum_count, out=self.predicate_b)
        torch.lt(self.elapsed, self.duration, out=self.predicate_c)
        torch.logical_and(self.predicate_b, self.predicate_c, out=self.predicate_b)
        torch.logical_or(self.predicate_a, self.predicate_b, out=self.predicate_a)
        self.error_flag.copy_(self.predicate_a)
        torch.logical_not(self.predicate_a, out=self.predicate_b)
        torch.logical_and(self.predicate_b, self.predicate_c, out=self.predicate_c)
        self.continue_flag.copy_(self.predicate_c)

    def check_completion(self, failed: int) -> None:
        if failed:
            raise ValueError(
                "adaptive substep proposal must be finite and positive and "
                "the interval must complete within "
                f"maximum_sub_steps={self.maximum_steps}"
            )

    @staticmethod
    def fetch(
        sources: tuple[torch.Tensor, ...],
        device: torch.Tensor,
        host: torch.Tensor,
    ) -> list[Any]:
        """Pack loop scalars on device and read them with one transfer."""
        if sources:
            torch.cat(sources, out=device)
        host.copy_(device)
        return host.tolist()

    def execute(self, duration: float, step: Any) -> int:
        self.duration.fill_(duration)
        return self.runner.run(duration, step)

    def close(self) -> None:
        runner, self.runner = self.runner, None
        programs = self.programs
        self.proposal = self.body = None
        close_runner(self.executor, runner, programs, scope="adaptive substep loop")

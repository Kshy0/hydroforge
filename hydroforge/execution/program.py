"""Cached execution programs for lexical fixed and adaptive substeps."""

from __future__ import annotations

import math
from typing import Any

import torch


class FixedSubstepProgram:
    """Cached fixed-width loop whose control state is owned by HydroForge."""

    def __init__(self, model: Any) -> None:
        self.execution = model._execution
        self.capture = self.execution.capture
        self.statistics = self.execution.statistics
        dtype = model.dtype
        if dtype not in {torch.float32, torch.float64}:
            raise TypeError(
                "fixed substep control requires model.dtype to be float32 or float64"
            )
        with torch.inference_mode(False):
            self.count = torch.ones(
                1, device=self.execution.device, dtype=torch.int32,
            )
            self.counter = torch.zeros_like(self.count)
            self.continue_flag = torch.zeros_like(self.count)
            self.duration = torch.zeros(
                1, device=self.execution.device, dtype=dtype,
            )
            self.weight = torch.zeros_like(self.duration)
            self.midpoint = torch.zeros_like(self.duration)
            self.half = torch.full_like(self.duration, 0.5)
            self.position = torch.zeros_like(self.duration)
            self.count_value = torch.ones_like(self.duration)
            self.one_value = torch.ones_like(self.duration)
            self.one_count = torch.ones_like(self.count)
        from hydroforge.execution.substeps import SubstepFrame

        self.frame = SubstepFrame(
            index=self.counter, dt=self.weight, midpoint=self.midpoint,
        )
        self.operators = None
        self.metal_iteration = None
        self.metal_fold_iteration = None
        self._metal_fold_aggregator = None
        self.iteration_graph = None
        self.statistics_graph = None
        self.mode = self.execution.loop_mode(
            world_size=model.world_size, allow_distributed=False,
        )

    def install(self, operators: Any) -> None:
        """Atomically install one recorded fixed-step program."""
        if self.operators is not None:
            raise RuntimeError("fixed substep program is already installed")
        if not operators.operators:
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError(
                "fixed substep produced an empty operator IR; backend kernels "
                "must be registered through BackendRegistry + KernelSpec"
            )
        metal_iteration = (
            self._build_metal_iteration(operators)
            if self.execution.capture_mode == "metal_icb" else None
        )
        # Commit only after every backend compilation/capture step succeeds.
        # A failed Metal build must remain an uninstalled program, never a
        # partially initialized object that later executes operator-by-operator.
        self.operators = operators
        self.metal_iteration = metal_iteration

    def _build_metal_iteration(self, operators: Any) -> Any:
        from hydroforge.execution.metal_control import fixed_control_commands
        from hydroforge.execution.operators import capture_metal_commands

        begin, end = fixed_control_commands(
            count=self.count, counter=self.counter,
            midpoint=self.midpoint, continue_flag=self.continue_flag,
        )
        return capture_metal_commands(
            self.capture,
            (begin, *operators.metal_commands(), end),
            cyclic=True,
        )

    def close(self) -> None:
        graphs = ()
        if self.iteration_graph is not None:
            graphs = (*graphs, self.iteration_graph)
            self.iteration_graph = None
        if self.statistics_graph is not None:
            graphs = (*graphs, self.statistics_graph)
            self.statistics_graph = None
        operators, self.operators = self.operators, None
        metal_iteration, self.metal_iteration = self.metal_iteration, None
        folded_iteration = self.metal_fold_iteration
        self.metal_fold_iteration = None
        self._metal_fold_aggregator = None
        for resource in graphs:
            self.capture.release(resource)
        # Loop ICBs can reference online-ATen scratch owned by ``operators``.
        # Release every consumer before allowing the producer to drop it.
        for iteration in (metal_iteration, folded_iteration):
            if iteration is None:
                continue
            self.capture.release(iteration.icb)
        if operators is not None:
            operators.close(self.capture)

    def _folded_metal_iteration(self):
        aggregator = self.statistics.aggregator
        if (
            self.metal_fold_iteration is not None
            and self._metal_fold_aggregator is aggregator
        ):
            return self.metal_fold_iteration
        previous = self.metal_fold_iteration
        self.metal_fold_iteration = None
        self._metal_fold_aggregator = None
        if previous is not None:
            self.capture.release(previous.icb)
        from hydroforge.execution.metal_control import (
            fixed_control_commands, statistics_control_command,
        )
        from hydroforge.execution.operators import capture_metal_commands

        begin, end = fixed_control_commands(
            count=self.count, counter=self.counter,
            midpoint=self.midpoint, continue_flag=self.continue_flag,
        )
        states = aggregator._kernel_states
        control = statistics_control_command(
            weight_source=self.weight,
            continue_flag=self.continue_flag,
            counter=self.counter,
            weight=states["__weight"],
            sub_step=states["__sub_step"],
            num_sub_steps=states["__num_sub_steps"],
        )
        replacement = capture_metal_commands(
            self.capture,
            (
                begin, *self.operators.metal_commands(), end,
                control, self.statistics.metal_operator(),
            ),
            cyclic=True,
        )
        self.metal_fold_iteration = replacement
        self._metal_fold_aggregator = aggregator
        return replacement

    def _reset(self) -> None:
        self.counter.zero_()
        if self.metal_iteration is None:
            self.position.zero_()
        self.continue_flag.fill_(1)

    def _iteration(self) -> None:
        if self.metal_iteration is not None:
            self.metal_iteration.launch()
            return
        if self.execution.capture_mode == "cuda_graph":
            from hydroforge.execution.cuda_graph import (
                fixed_control_begin, fixed_control_end,
            )

            stream = torch.cuda.current_stream(
                self.execution.device,
            ).cuda_stream
            fixed_control_begin(
                self.count, self.counter, self.midpoint, stream,
            )
            self.operators.launch()
            fixed_control_end(
                self.count, self.counter, self.continue_flag, stream,
            )
            return
        # Keep loop control allocation-free as well as the recorded physics.
        # These scalar operations are captured once on CUDA and execute eager
        # without constructing a temporary tensor on CPU/Torch backends.
        self.midpoint.copy_(self.position).add_(self.half).div_(self.count_value)
        self.operators.launch()
        self.counter.add_(self.one_count)
        self.position.add_(self.one_value)
        torch.lt(self.counter, self.count, out=self.continue_flag)

    def _fixed_iteration_graph(self) -> Any:
        graph = self.iteration_graph
        if graph is None:
            controlled = (
                self.operators.references_tensor(self.counter)
                or self.operators.references_tensor(self.midpoint)
            )
            body = self._iteration if controlled else self.operators.launch
            control_state = (
                (self.counter, self.continue_flag, self.midpoint, self.position)
                if controlled else ()
            )
            graph = self.capture.capture_cuda(
                body,
                mutated_state=(
                    *control_state, *self.operators.mutated_tensors,
                ),
            )
            self.iteration_graph = graph
        return graph

    def _fixed_statistics_graph(self) -> Any:
        graph = self.statistics_graph
        if graph is not None:
            return graph
        aggregator = self.statistics.aggregator
        states = aggregator._kernel_states

        def body() -> None:
            from hydroforge.execution.cuda_graph import (
                fixed_control_begin, fixed_statistics_end,
            )

            stream = torch.cuda.current_stream(
                self.execution.device,
            ).cuda_stream
            if self.operators.references_tensor(self.midpoint):
                fixed_control_begin(
                    self.count, self.counter, self.midpoint, stream,
                )
            self.operators.launch()
            fixed_statistics_end(
                count=self.count,
                counter=self.counter,
                continue_flag=self.continue_flag,
                weight_src=self.weight,
                weight=states["__weight"],
                sub_step=states["__sub_step"],
                num_sub_steps=states["__num_sub_steps"],
                stream_ptr=stream,
            )
            aggregator._aggregator_function(
                states, self.statistics.model.BLOCK_SIZE,
            )

        graph = self.capture.capture_cuda(
            body,
            mutated_state=(
                self.counter, self.continue_flag, self.midpoint,
                self.position, *self.operators.mutated_tensors,
                *(value for value in states.values()
                  if isinstance(value, torch.Tensor)),
            ),
        )
        self.statistics_graph = graph
        return graph

    def execute(self, count: int, duration: float) -> int:
        if type(count) is not int:
            raise TypeError("fixed substep count must be an int")
        if count < 1:
            raise ValueError("fixed substep count must be positive")
        if type(duration) not in {int, float}:
            raise TypeError("fixed substep duration must be an int or float")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("fixed substep duration must be finite and positive")
        if self.operators is None:
            raise RuntimeError("fixed substep scope has not been recorded")
        self.operators.require_stable_bindings()
        if self.execution.capture_mode == "metal_icb":
            self.operators.reset_metal_errors()
        step = self.execution.active_step
        if step is None:
            raise RuntimeError("fixed substeps require @managed_step")
        if self.mode != "eager" and not step.run_statistics:
            controlled = (
                self.operators.references_tensor(self.counter)
                or self.operators.references_tensor(self.midpoint)
            )
            if controlled:
                self.count.fill_(count)
                self.count_value.fill_(count)
                self._reset()
            if self.operators.references_tensor(self.weight):
                self.weight.fill_(duration / count)
            graph = self._fixed_iteration_graph()
            for _ in range(count):
                graph.replay()
            step.advance_device(duration)
            return count
        self.count.fill_(count)
        if self.metal_iteration is None:
            self.count_value.fill_(count)
        self.duration.fill_(duration)
        self.weight.fill_(duration / count)
        fold = False
        if self.metal_iteration is not None:
            fold = step.run_statistics and self.statistics.should_fold()
        if self.metal_iteration is not None and fold:
            if not self.statistics.device_compatible():
                raise RuntimeError(
                    "fixed Metal execution requires device-compatible statistics"
                )
            self.statistics.prelaunch(step.flags, step.total_weight)
            self._reset()
            self._folded_metal_iteration().replay(count)
            step.advance_device(duration)
            self.operators.check_metal_errors()
            return count
        if self.metal_iteration is not None:
            self._reset()
            self.metal_iteration.replay(count)
            if step.run_statistics:
                self.statistics.sample(
                    sub_step=count - 1, num_sub_steps=count,
                    flags=step.flags, weight=duration,
                    total_weight=step.total_weight,
                )
            step.advance_device(duration)
            self.operators.check_metal_errors()
            return count
        if self.mode == "eager":
            self._reset()
            width = duration / count
            for index in range(count):
                self._iteration()
                step.sample_fixed(
                    sub_step=index, num_sub_steps=count, weight=width,
                )
            self.operators.check_metal_errors()
            return count
        if step.run_statistics and not self.statistics.device_compatible():
            raise RuntimeError(
                "fixed device execution requires device-compatible statistics"
            )
        fold = step.run_statistics and self.statistics.should_fold()
        if fold:
            self.statistics.prelaunch(step.flags, step.total_weight)
            self._reset()
            graph = self._fixed_statistics_graph()
            for _ in range(count):
                graph.replay()
        else:
            controlled = (
                self.operators.references_tensor(self.counter)
                or self.operators.references_tensor(self.midpoint)
            )
            if controlled:
                self._reset()
            graph = self._fixed_iteration_graph()
            for _ in range(count):
                graph.replay()
        if step.run_statistics and not fold:
            self.statistics.sample(
                sub_step=count - 1, num_sub_steps=count, flags=step.flags,
                weight=duration, total_weight=step.total_weight,
            )
        step.advance_device(duration)
        return count


class AdaptiveSubstepProgram:
    """Cached adaptive loop whose control state is owned by HydroForge."""

    def __init__(
        self, model: Any, *, candidate_dt: torch.Tensor, dt: torch.Tensor,
        maximum_dt: float, maximum_steps: int,
    ) -> None:
        self.execution = model._execution
        self.capture = self.execution.capture
        self.statistics = self.execution.statistics
        if not isinstance(candidate_dt, torch.Tensor) or candidate_dt.numel() != 1:
            raise TypeError("adaptive candidate_dt must be a one-element tensor")
        if not isinstance(dt, torch.Tensor) or dt.numel() != 1:
            raise TypeError("adaptive dt must be a one-element tensor")
        if (
            candidate_dt.layout is not torch.strided
            or not candidate_dt.is_contiguous()
        ):
            raise ValueError(
                "adaptive candidate_dt must be a contiguous strided tensor"
            )
        if dt.layout is not torch.strided or not dt.is_contiguous():
            raise ValueError("adaptive dt must be a contiguous strided tensor")
        if candidate_dt.dtype not in {torch.float32, torch.float64}:
            raise TypeError(
                "adaptive candidate_dt must have float32 or float64 dtype"
            )
        if dt.dtype != candidate_dt.dtype:
            raise TypeError(
                "adaptive dt and candidate_dt must have identical dtype"
            )
        if dt.device != candidate_dt.device:
            raise ValueError("adaptive dt and candidate_dt must share one device")
        if type(maximum_dt) not in {int, float}:
            raise TypeError("adaptive maximum_dt must be an exact int or float")
        if not math.isfinite(maximum_dt) or maximum_dt <= 0:
            raise ValueError("adaptive maximum_dt must be finite and positive")
        if type(maximum_steps) is not int:
            raise TypeError("adaptive maximum_steps must be an exact int")
        if maximum_steps < 1:
            raise ValueError("adaptive maximum_steps must be positive")
        self.candidate = candidate_dt
        self.time_step = dt
        self.maximum = float(maximum_dt)
        self.maximum_steps = maximum_steps
        # The first program build commonly happens under a model's
        # ``torch.inference_mode`` step. Conditional capture mutation tracing
        # requires ordinary tensors with version counters, so runtime-owned
        # state is deliberately allocated outside that mode.
        with torch.inference_mode(False):
            options = dict(device=candidate_dt.device, dtype=candidate_dt.dtype)
            self.duration = torch.zeros(1, **options)
            self.elapsed = torch.zeros(1, **options)
            self.fraction = torch.zeros(1, **options)
            self.counter = torch.zeros(
                1, device=candidate_dt.device, dtype=torch.int32,
            )
            self.continue_flag = torch.zeros_like(self.counter)
            self.error_flag = torch.zeros_like(self.counter)
            # Stable scalar scratch belongs to the loop program, not to an
            # iteration.  CUDA captures these addresses and eager execution
            # performs no per-substep tensor allocation.
            self.remaining = torch.zeros(1, **options)
            self.accepted = torch.zeros(1, **options)
            self.predicate_a = torch.zeros(
                1, device=candidate_dt.device, dtype=torch.bool,
            )
            self.predicate_b = torch.zeros_like(self.predicate_a)
            self.predicate_c = torch.zeros_like(self.predicate_a)
            self.maximum_value = torch.full(
                (1,), self.maximum, **options,
            )
            self.zero_value = torch.zeros(1, **options)
            self.half = torch.full((1,), 0.5, **options)
            self.maximum_count = torch.full(
                (1,), self.maximum_steps,
                device=candidate_dt.device, dtype=torch.int32,
            )
            self.one_count = torch.ones_like(self.maximum_count)
        from hydroforge.execution.substeps import SubstepFrame

        self.frame = SubstepFrame(
            index=self.counter, dt=self.time_step, midpoint=self.fraction,
        )
        self.graphs: dict[bool, Any] = {}
        self.proposal_operators = None
        self.body_operators = None
        self.metal_iteration = None
        self.mode = self.execution.loop_mode(
            world_size=model.world_size, allow_distributed=False,
        )

    def require_binding(
        self, *, candidate_dt: torch.Tensor, dt: torch.Tensor,
        maximum_dt: float, maximum_steps: int,
    ) -> None:
        """Reject address or control drift at a cached lexical scope."""

        if candidate_dt is not self.candidate or dt is not self.time_step:
            raise RuntimeError(
                "adaptive substep tensors changed at a cached lexical scope; "
                "bind stable model tensors or select a distinct specialization"
            )
        if (
            type(maximum_dt) not in {int, float}
            or float(maximum_dt) != self.maximum
            or type(maximum_steps) is not int
            or maximum_steps != self.maximum_steps
        ):
            raise RuntimeError(
                "adaptive substep controls changed at a cached lexical scope; "
                "include changing host controls in specialization"
            )

    def install(self, proposal: Any, body: Any) -> None:
        """Atomically install the proposal and physics operator regions."""
        if self.proposal_operators is not None or self.body_operators is not None:
            raise RuntimeError("adaptive substep program is already installed")
        if not proposal.operators:
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError(
                "adaptive dt proposal produced an empty operator IR"
            )
        if not body.operators:
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError(
                "adaptive physics body produced an empty operator IR"
            )
        metal_iteration = (
            self._build_metal_iteration(proposal, body)
            if self.execution.capture_mode == "metal_icb" else None
        )
        self.proposal_operators = proposal
        self.body_operators = body
        self.metal_iteration = metal_iteration

    def _build_metal_iteration(self, proposal: Any, body: Any) -> Any:
        from hydroforge.execution.metal_control import adaptive_control_commands
        from hydroforge.execution.operators import capture_metal_commands

        begin, accept, end = adaptive_control_commands(
            candidate=self.candidate, maximum=self.maximum,
            duration=self.duration, elapsed=self.elapsed,
            dt=self.time_step, midpoint=self.fraction,
            counter=self.counter, continue_flag=self.continue_flag,
            error_flag=self.error_flag,
            maximum_steps=self.maximum_steps,
        )
        return capture_metal_commands(
            self.capture,
            (
                begin, *proposal.metal_commands(), accept,
                *body.metal_commands(), end,
            ),
            cyclic=True,
        )

    def close(self) -> None:
        graphs, self.graphs = tuple(self.graphs.values()), {}
        proposal, self.proposal_operators = self.proposal_operators, None
        body, self.body_operators = self.body_operators, None
        metal_iteration, self.metal_iteration = self.metal_iteration, None
        for resource in graphs:
            self.capture.release(resource)
        if metal_iteration is not None:
            self.capture.release(metal_iteration.icb)
        for operators in (proposal, body):
            if operators is None:
                continue
            operators.close(self.capture)

    def _reset(self) -> None:
        self.elapsed.zero_()
        self.counter.zero_()
        self.continue_flag.fill_(1)
        self.error_flag.zero_()

    def _iteration(self) -> None:
        if self.metal_iteration is not None:
            self.metal_iteration.launch()
            return
        self.candidate.copy_(self.maximum_value)
        self.proposal_operators.launch()
        self.remaining.copy_(self.duration).sub_(self.elapsed)
        torch.minimum(
            self.candidate, self.remaining, out=self.accepted,
        )
        # ``accepted != accepted`` is exactly the NaN test needed here.  A
        # positive infinity cannot survive minimum(candidate, finite
        # remaining), while negative infinity is caught by <= 0.
        torch.eq(self.accepted, self.accepted, out=self.predicate_a)
        torch.logical_not(self.predicate_a, out=self.predicate_a)
        torch.le(self.accepted, self.zero_value, out=self.predicate_b)
        torch.logical_or(
            self.predicate_a, self.predicate_b, out=self.predicate_a,
        )
        # A bad proposal must terminate the device WHILE node.  Substitute the
        # finite positive remainder so the already-captured physics tail does
        # not receive zero/NaN before the host reports the strict error.
        torch.where(
            self.predicate_a, self.remaining, self.accepted,
            out=self.time_step,
        )
        self.fraction.copy_(self.time_step).mul_(self.half)
        self.fraction.add_(self.elapsed).div_(self.duration)
        self.body_operators.launch()
        self.elapsed.add_(self.time_step)
        self.counter.add_(self.one_count)
        torch.ge(
            self.counter, self.maximum_count, out=self.predicate_b,
        )
        torch.lt(self.elapsed, self.duration, out=self.predicate_c)
        torch.logical_and(
            self.predicate_b, self.predicate_c, out=self.predicate_b,
        )
        torch.logical_or(
            self.predicate_a, self.predicate_b, out=self.predicate_a,
        )
        self.error_flag.copy_(self.predicate_a)
        torch.logical_not(self.predicate_a, out=self.predicate_b)
        torch.lt(self.elapsed, self.duration, out=self.predicate_c)
        torch.logical_and(
            self.predicate_b, self.predicate_c, out=self.predicate_c,
        )
        self.continue_flag.copy_(self.predicate_c)

    def _graph(self, fold: bool) -> Any:
        graph = self.graphs.get(fold)
        if graph is not None:
            return graph
        extra = self.statistics.accumulators() if fold else ()

        def captured_body(graph: Any, _set_cond: bool, stream: int) -> None:
            self._iteration()
            if fold:
                self.statistics.captured_body(
                    graph=graph, weight_src=self.time_step,
                    counter=self.counter,
                    continue_flag=self.continue_flag, stream_ptr=stream,
                )

        graph = self.capture.build_conditional_graph(
            body=captured_body,
            reset=self._reset,
            continue_flag=self.continue_flag,
            extra_state=(
                self.duration, self.elapsed, self.fraction, self.counter,
                self.continue_flag, self.error_flag,
                self.remaining, self.accepted,
                self.predicate_a, self.predicate_b, self.predicate_c,
                self.maximum_value, self.zero_value, self.half,
                self.maximum_count, self.one_count,
                *self.proposal_operators.mutated_tensors,
                *self.body_operators.mutated_tensors, *(extra or ()),
            ),
        )
        self.graphs[fold] = graph
        return graph

    def execute(self, duration: float) -> int:
        if type(duration) not in {int, float}:
            raise TypeError("adaptive duration must be an int or float")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("adaptive duration must be finite and positive")
        self.duration.fill_(duration)
        if self.proposal_operators is None or self.body_operators is None:
            raise RuntimeError("adaptive substep scope has not been recorded")
        self.proposal_operators.require_stable_bindings()
        self.body_operators.require_stable_bindings()
        if self.execution.capture_mode == "metal_icb":
            self.proposal_operators.reset_metal_errors()
            self.body_operators.reset_metal_errors()
        step = self.execution.active_step
        if step is None:
            raise RuntimeError("adaptive substeps require @managed_step")
        if self.mode == "eager":
            self._reset()
            elapsed = 0.0
            count = 0
            while elapsed < duration:
                self._iteration()
                if int(self.error_flag.item()) != 0:
                    raise ValueError(
                        "adaptive substep proposal must be finite and positive "
                        "and the interval must complete within "
                        f"maximum_sub_steps={self.maximum_steps}"
                    )
                weight = float(self.time_step.item())
                if not 0.0 < weight <= duration - elapsed:
                    raise ValueError(
                        f"adaptive substep width {weight} is invalid at {elapsed}"
                    )
                elapsed += weight
                count += 1
                step.sample_adaptive(
                    weight=weight, first_event=count == 1,
                    last_event=elapsed >= duration,
                )
            self.proposal_operators.check_metal_errors()
            self.body_operators.check_metal_errors()
            return count
        if step.run_statistics and not self.statistics.device_compatible():
            raise RuntimeError(
                "adaptive device execution requires device-compatible statistics"
            )
        fold = step.run_statistics and self.statistics.should_fold()
        if fold:
            self.statistics.prelaunch(step.flags, step.total_weight)
        self._reset()
        self.execution.launch_conditional(self._graph(fold))
        if int(self.error_flag.item()) != 0:
            raise ValueError(
                "adaptive substep proposal must be finite and positive and "
                "the interval must complete within "
                f"maximum_sub_steps={self.maximum_steps}"
            )
        count = int(self.counter.item())
        if step.run_statistics and not fold:
            self.statistics.sample(
                sub_step=0, num_sub_steps=1, flags=step.flags,
                weight=duration, total_weight=step.total_weight,
            )
        step.advance_device(duration)
        return count

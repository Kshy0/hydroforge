"""Cached execution programs for lexical fixed and adaptive substeps."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from hydroforge.contracts.errors import ResourceCleanupError

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


def _close_program_resources(capture, graphs, operators, *, scope: str) -> None:
    failures: list[BaseException] = []
    for resource in {id(graph): graph for graph in graphs}.values():
        if resource is None:
            continue
        try:
            capture.release(resource)
        except BaseException as error:
            failures.append(error)
    for program in operators:
        if program is None:
            continue
        try:
            program.close(capture)
        except BaseException as error:
            failures.append(error)
    if failures:
        error = ResourceCleanupError(scope, failures)
        raise error from failures[0]


class _FixedSubstepDraft:
    """Recording-only controls that never enter the runtime program cache."""

    def __init__(self, model: AbstractModel) -> None:
        dtype = model.dtype
        with torch.inference_mode(False):
            self.count = torch.ones(
                1,
                device=model._execution.device,
                dtype=torch.int32,
            )
            self.counter = torch.zeros_like(self.count)
            self.continue_flag = torch.zeros_like(self.count)
            self.duration = torch.zeros(
                1,
                device=model._execution.device,
                dtype=dtype,
            )
            self.weight = torch.zeros_like(self.duration)
            self.one_count = torch.ones_like(self.count)
        from hydroforge.execution.substeps import SubstepFrame

        self.frame = SubstepFrame(
            index=self.counter,
            dt=self.weight,
        )


class FixedSubstepProgram:
    """Complete cached fixed-width loop compiled from one recording draft."""

    def __init__(
        self,
        model: AbstractModel,
        draft: _FixedSubstepDraft,
        operators: Any,
        final_operators: Any | None = None,
    ) -> None:
        if not operators.operators:
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError(
                "fixed substep produced an empty operator IR; backend kernels "
                "must be registered through BackendRegistry + KernelSpec"
            )
        self.execution = model._execution
        self.capture = self.execution.capture
        self.statistics = self.execution.statistics
        self.count = draft.count
        self.counter = draft.counter
        self.continue_flag = draft.continue_flag
        self.duration = draft.duration
        self.weight = draft.weight
        self.one_count = draft.one_count
        self.frame = draft.frame
        self.operators = operators
        self.final_operators = final_operators
        self.metal_iteration = None
        self.metal_final_iteration = None
        self.metal_fold_iteration = None
        self.metal_fold_final_iteration = None
        self._metal_fold_aggregator = None
        self.iteration_graph = None
        self.final_iteration_graph = None
        self.statistics_graph = None
        self.final_statistics_graph = None
        self._unrolled_graphs = OrderedDict()
        self._unrolled_hits = OrderedDict()
        self._unroll_fixed_loops = (
            os.environ.get("HYDROFORGE_UNROLL_FIXED_LOOPS", "0") == "1"
        )
        self.mode = self.execution.loop_mode(
            world_size=model.world_size,
            allow_distributed=False,
        )
        metal_iterations = (
            self._build_metal_iterations()
            if self.execution.capture_mode == "metal_icb"
            else None
        )
        if metal_iterations is not None:
            self.metal_iteration, self.metal_final_iteration = metal_iterations

    @staticmethod
    def recording_draft(model: AbstractModel) -> _FixedSubstepDraft:
        return _FixedSubstepDraft(model)

    def _build_metal_iterations(self) -> tuple[Any, Any | None]:
        from hydroforge.execution.metal_control import fixed_control_command

        control = fixed_control_command(
            count=self.count,
            counter=self.counter,
            continue_flag=self.continue_flag,
        )
        return self._capture_metal_iterations(
            (control,), scope="fixed Metal final capture"
        )

    def _capture_metal_iterations(
        self, tail: tuple[Any, ...], *, scope: str
    ) -> tuple[Any, Any | None]:
        """Capture regular/final bodies and roll back partial pair creation."""
        from hydroforge.execution.operators import capture_metal_commands

        iteration = capture_metal_commands(
            self.capture,
            (*self.operators.metal_commands(), *tail),
            cyclic=True,
        )
        try:
            final_iteration = None
            if self.final_operators is not None:
                final_iteration = capture_metal_commands(
                    self.capture,
                    (
                        *self.operators.metal_commands(),
                        *self.final_operators.metal_commands(),
                        *tail,
                    ),
                    cyclic=True,
                )
        except BaseException as primary:
            try:
                self.capture.release(iteration.icb)
            except BaseException as cleanup_error:
                error = ResourceCleanupError(
                    scope,
                    (primary, cleanup_error),
                )
                raise error from primary
            raise
        return iteration, final_iteration

    def close(self) -> None:
        graphs = tuple(self._unrolled_graphs.values())
        self._unrolled_graphs.clear()
        self._unrolled_hits.clear()
        if self.iteration_graph is not None:
            graphs = (*graphs, self.iteration_graph)
            self.iteration_graph = None
        if self.final_iteration_graph is not None:
            graphs = (*graphs, self.final_iteration_graph)
            self.final_iteration_graph = None
        if self.statistics_graph is not None:
            graphs = (*graphs, self.statistics_graph)
            self.statistics_graph = None
        if self.final_statistics_graph is not None:
            graphs = (*graphs, self.final_statistics_graph)
            self.final_statistics_graph = None
        operators, self.operators = self.operators, None
        final_operators, self.final_operators = self.final_operators, None
        metal_iteration, self.metal_iteration = self.metal_iteration, None
        metal_final_iteration, self.metal_final_iteration = (
            self.metal_final_iteration,
            None,
        )
        folded_iteration = self.metal_fold_iteration
        self.metal_fold_iteration = None
        folded_final_iteration = self.metal_fold_final_iteration
        self.metal_fold_final_iteration = None
        self._metal_fold_aggregator = None
        # Loop ICBs can reference online-ATen scratch owned by ``operators``.
        # Release every consumer before allowing the producer to drop it.
        _close_program_resources(
            self.capture,
            (
                *graphs,
                *(
                    iteration.icb
                    for iteration in (
                        metal_iteration,
                        metal_final_iteration,
                        folded_iteration,
                        folded_final_iteration,
                    )
                    if iteration is not None
                ),
            ),
            (operators, final_operators),
            scope="fixed substep program",
        )

    def invalidate_statistics(self, aggregator: Any) -> None:
        """Release captures that retain one statistics specialization."""
        graphs = tuple(
            self._unrolled_graphs.pop(key)
            for key in tuple(self._unrolled_graphs)
            if key[0]
        )
        graph, self.statistics_graph = self.statistics_graph, None
        final_graph, self.final_statistics_graph = self.final_statistics_graph, None
        metal_graphs = ()
        if self._metal_fold_aggregator is aggregator:
            iteration, self.metal_fold_iteration = (self.metal_fold_iteration, None)
            final_iteration, self.metal_fold_final_iteration = (
                self.metal_fold_final_iteration,
                None,
            )
            self._metal_fold_aggregator = None
            metal_graphs = tuple(
                resource.icb
                for resource in (iteration, final_iteration)
                if resource is not None
            )
        _close_program_resources(
            self.capture,
            (*graphs, graph, final_graph, *metal_graphs),
            (),
            scope="fixed substep statistics caches",
        )

    def _folded_metal_iterations(self):
        aggregator = self.statistics.aggregator
        if (
            self.metal_fold_iteration is not None
            and self._metal_fold_aggregator is aggregator
        ):
            return self.metal_fold_iteration, self.metal_fold_final_iteration
        previous = self.metal_fold_iteration
        previous_final = self.metal_fold_final_iteration
        self.metal_fold_iteration = None
        self.metal_fold_final_iteration = None
        self._metal_fold_aggregator = None
        _close_program_resources(
            self.capture,
            tuple(
                resource.icb
                for resource in (previous, previous_final)
                if resource is not None
            ),
            (),
            scope="fixed Metal statistics replacement",
        )
        from hydroforge.execution.metal_control import (
            fixed_control_command,
            statistics_control_command,
        )

        fixed_control = fixed_control_command(
            count=self.count,
            counter=self.counter,
            continue_flag=self.continue_flag,
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
        replacement, final_replacement = self._capture_metal_iterations(
            (fixed_control, control, self.statistics.metal_operator()),
            scope="fixed Metal statistics final capture",
        )
        self.metal_fold_iteration = replacement
        self.metal_fold_final_iteration = final_replacement
        self._metal_fold_aggregator = aggregator
        return replacement, final_replacement

    def _reset(self) -> None:
        self.counter.zero_()
        self.continue_flag.fill_(1)

    def _iteration(self, *, final: bool = False) -> None:
        if self.metal_iteration is not None:
            iteration = (
                self.metal_final_iteration
                if final and self.metal_final_iteration is not None
                else self.metal_iteration
            )
            iteration.launch()
            return
        if self.execution.capture_mode == "cuda_graph":
            from hydroforge.execution.cuda_graph import fixed_control_end

            stream = torch.cuda.current_stream(
                self.execution.device,
            ).cuda_stream
            self.operators.launch()
            if final and self.final_operators is not None:
                self.final_operators.launch()
            fixed_control_end(
                self.count,
                self.counter,
                self.continue_flag,
                stream,
            )
            return
        self.operators.launch()
        if final and self.final_operators is not None:
            self.final_operators.launch()
        self.counter.add_(self.one_count)
        torch.lt(self.counter, self.count, out=self.continue_flag)

    def _references_counter(self) -> bool:
        return self.operators.references_tensor(self.counter) or (
            self.final_operators is not None
            and self.final_operators.references_tensor(self.counter)
        )

    def _fixed_iteration_graph(self, *, final: bool = False) -> Any:
        graph = self.final_iteration_graph if final else self.iteration_graph
        if graph is None:
            controlled = self._references_counter()
            if controlled:

                def body() -> None:
                    self._iteration(final=final)
            elif final and self.final_operators is not None:

                def body() -> None:
                    self.operators.launch()
                    self.final_operators.launch()
            else:
                body = self.operators.launch
            control_state = (self.counter, self.continue_flag) if controlled else ()
            final_state = (
                self.final_operators.mutated_tensors
                if final and self.final_operators is not None
                else ()
            )
            graph = self.capture.capture_cuda(
                body,
                mutated_state=(
                    *control_state,
                    *self.operators.mutated_tensors,
                    *final_state,
                ),
            )
            if final:
                self.final_iteration_graph = graph
            else:
                self.iteration_graph = graph
        return graph

    def _fixed_statistics_graph(self, *, final: bool = False) -> Any:
        graph = self.final_statistics_graph if final else self.statistics_graph
        if graph is not None:
            return graph
        aggregator = self.statistics.aggregator
        states = aggregator._kernel_states

        def body() -> None:
            self._statistics_iteration(aggregator, final=final)

        graph = self.capture.capture_cuda(
            body,
            mutated_state=(
                self.counter,
                self.continue_flag,
                *self.operators.mutated_tensors,
                *(
                    self.final_operators.mutated_tensors
                    if final and self.final_operators is not None
                    else ()
                ),
                *(
                    value
                    for value in states.values()
                    if isinstance(value, torch.Tensor)
                ),
            ),
        )
        if final:
            self.final_statistics_graph = graph
        else:
            self.statistics_graph = graph
        return graph

    def _statistics_iteration(self, aggregator: Any, *, final: bool) -> None:
        """Capture physics, loop control, and statistics in their common order."""
        from hydroforge.execution.cuda_graph import fixed_statistics_end

        states = aggregator._kernel_states
        self.operators.launch()
        if final and self.final_operators is not None:
            self.final_operators.launch()
        fixed_statistics_end(
            count=self.count,
            counter=self.counter,
            continue_flag=self.continue_flag,
            weight_src=self.weight,
            weight=states["__weight"],
            sub_step=states["__sub_step"],
            num_sub_steps=states["__num_sub_steps"],
            stream_ptr=torch.cuda.current_stream(self.execution.device).cuda_stream,
        )
        aggregator._aggregator_function(states, aggregator.block_size)

    @staticmethod
    def _replay_with_final(
        regular: Any,
        final: Any | None,
        count: int,
    ) -> None:
        if final is None:
            for _ in range(count):
                regular.replay()
            return
        for _ in range(count - 1):
            regular.replay()
        final.replay()

    def _replay_fixed_graphs(self, regular, final, count: int, *, fold: bool) -> None:
        """Amortize repeated small fixed loops with a bounded whole-loop cache."""

        if not self._unroll_fixed_loops:
            self._replay_with_final(regular, final, count)
            return
        key = (fold, count)
        graph = self._unrolled_graphs.get(key)
        if graph is not None:
            self._unrolled_graphs.move_to_end(key)
            graph.replay()
            return
        operator_count = len(self.operators.operators)
        if (
            self.execution.capture_mode != "cuda_graph"
            or not 4 <= count <= 64
            or count * operator_count > 512
        ):
            self._replay_with_final(regular, final, count)
            return
        hits = self._unrolled_hits.pop(key, 0) + 1
        self._unrolled_hits[key] = hits
        if len(self._unrolled_hits) > 16:
            self._unrolled_hits.popitem(last=False)
        if hits < 8:
            self._replay_with_final(regular, final, count)
            return
        aggregator = self.statistics.aggregator if fold else None
        states = aggregator._kernel_states if fold else {}
        controlled = self._references_counter()

        def body():
            for index in range(count):
                is_final = index == count - 1 and self.final_operators is not None
                if fold:
                    self._statistics_iteration(aggregator, final=is_final)
                elif controlled:
                    self._iteration(final=is_final)
                else:
                    self.operators.launch()
                    if is_final:
                        self.final_operators.launch()

        graph = self.capture.capture_cuda(
            body,
            mutated_state=(
                self.counter,
                self.continue_flag,
                *self.operators.mutated_tensors,
                *(
                    self.final_operators.mutated_tensors
                    if self.final_operators is not None
                    else ()
                ),
                *(
                    value
                    for value in states.values()
                    if isinstance(value, torch.Tensor)
                ),
            ),
        )
        if len(self._unrolled_graphs) >= 4:
            _old_key, old_graph = self._unrolled_graphs.popitem(last=False)
            try:
                self.capture.release(old_graph)
            except BaseException as primary:
                try:
                    self.capture.release(graph)
                except BaseException as cleanup_error:
                    error = ResourceCleanupError(
                        "fixed unrolled graph replacement",
                        (primary, cleanup_error),
                    )
                    raise error from primary
                raise
        self._unrolled_graphs[key] = graph
        graph.replay()

    def execute(self, count: int, duration: float, step: Any) -> int:
        if self.execution.capture_mode == "metal_icb":
            self.operators.reset_metal_errors()
            if self.final_operators is not None:
                self.final_operators.reset_metal_errors()
        capture_safe = self.operators.cuda_graph_capture_safe and (
            self.final_operators is None or self.final_operators.cuda_graph_capture_safe
        )
        if self.mode != "eager" and not capture_safe:
            # A conditional-WHILE graph cannot be launched while an enclosing
            # CUDA stream capture is active.  Keep the predicate loop as one
            # device graph launch and execute its surrounding operators in
            # lexical order without wrapping them in a second CUDA graph.
            self.count.fill_(count)
            self.duration.fill_(duration)
            self.weight.fill_(duration / count)
            controlled = self._references_counter()
            if controlled:
                self._reset()
            width = duration / count
            for index in range(count):
                final = index == count - 1 and self.final_operators is not None
                if controlled:
                    self._iteration(final=final)
                else:
                    self.operators.launch()
                    if final:
                        self.final_operators.launch()
                # Nested predicate graphs cannot themselves be captured in an
                # enclosing fixed-loop graph.  Preserve fixed-loop semantics
                # by sampling after every host-scheduled physical substep.
                step.sample_fixed(
                    sub_step=index,
                    num_sub_steps=count,
                    weight=width,
                )
            return count
        if self.mode != "eager" and not step.run_statistics:
            controlled = self._references_counter()
            if controlled:
                self.count.fill_(count)
                self._reset()
            if self.operators.references_tensor(self.weight) or (
                self.final_operators is not None
                and self.final_operators.references_tensor(self.weight)
            ):
                self.weight.fill_(duration / count)
            graph = self._fixed_iteration_graph()
            final_graph = (
                self._fixed_iteration_graph(final=True)
                if self.final_operators is not None
                else None
            )
            self._replay_fixed_graphs(graph, final_graph, count, fold=False)
            step.advance_device(duration)
            return count
        self.count.fill_(count)
        self.duration.fill_(duration)
        self.weight.fill_(duration / count)
        fold = False
        if self.metal_iteration is not None:
            fold = step.run_statistics and self.statistics.should_fold()
        if self.metal_iteration is not None and fold:
            self.statistics.prelaunch(step.flags, step.total_weight)
            self._reset()
            regular, final = self._folded_metal_iterations()
            if final is None:
                regular.replay(count)
            else:
                if count > 1:
                    regular.replay(count - 1)
                final.replay()
            step.advance_device(duration)
            self.operators.check_metal_errors()
            if self.final_operators is not None:
                self.final_operators.check_metal_errors()
            return count
        if self.metal_iteration is not None:
            self._reset()
            if self.metal_final_iteration is None:
                self.metal_iteration.replay(count)
            else:
                if count > 1:
                    self.metal_iteration.replay(count - 1)
                self.metal_final_iteration.replay()
            if step.run_statistics:
                self.statistics.sample(
                    sub_step=count - 1,
                    num_sub_steps=count,
                    flags=step.flags,
                    weight=duration,
                    total_weight=step.total_weight,
                )
            step.advance_device(duration)
            self.operators.check_metal_errors()
            if self.final_operators is not None:
                self.final_operators.check_metal_errors()
            return count
        if self.mode == "eager":
            self._reset()
            width = duration / count
            for index in range(count):
                self._iteration(
                    final=(index == count - 1 and self.final_operators is not None),
                )
                step.sample_fixed(
                    sub_step=index,
                    num_sub_steps=count,
                    weight=width,
                )
            self.operators.check_metal_errors()
            if self.final_operators is not None:
                self.final_operators.check_metal_errors()
            return count
        fold = step.run_statistics and self.statistics.should_fold()
        if fold:
            self.statistics.prelaunch(step.flags, step.total_weight)
            self._reset()
            graph = self._fixed_statistics_graph()
            final_graph = (
                self._fixed_statistics_graph(final=True)
                if self.final_operators is not None
                else None
            )
            self._replay_fixed_graphs(graph, final_graph, count, fold=True)
        else:
            controlled = self._references_counter()
            if controlled:
                self._reset()
            graph = self._fixed_iteration_graph()
            final_graph = (
                self._fixed_iteration_graph(final=True)
                if self.final_operators is not None
                else None
            )
            self._replay_fixed_graphs(graph, final_graph, count, fold=False)
        if step.run_statistics and not fold:
            self.statistics.sample(
                sub_step=count - 1,
                num_sub_steps=count,
                flags=step.flags,
                weight=duration,
                total_weight=step.total_weight,
            )
        step.advance_device(duration)
        return count


class _PredicateLoopDraft:
    """Recording-only predicate controls."""

    def __init__(self, model: AbstractModel, *, maximum_steps: int) -> None:
        from torch.utils._python_dispatch import _disable_current_modes

        with _disable_current_modes(), torch.inference_mode(False):
            options = {"device": model._execution.device, "dtype": torch.int32}
            self.predicate = torch.zeros(1, **options)
            self.counter = torch.zeros(1, **options)
            self.continue_flag = torch.zeros(1, **options)
            self.maximum_count = torch.full((1,), maximum_steps, **options)
            self.zero_count = torch.zeros(1, **options)
            self.one_count = torch.ones(1, **options)
            self.has_more = torch.zeros(
                1,
                device=model._execution.device,
                dtype=torch.bool,
            )
            self.under_limit = torch.zeros_like(self.has_more)


class PredicateLoopProgram:
    """Complete nested loop controlled by a body-authored device predicate."""

    def __init__(
        self,
        model: AbstractModel,
        *,
        maximum_steps: int,
        draft: _PredicateLoopDraft,
        body: Any,
    ) -> None:
        if not body.operators:
            from hydroforge.execution.operators import SubstepCompileError

            raise SubstepCompileError("predicate loop produced an empty operator IR")
        self.execution = model._execution
        self.capture = self.execution.capture
        self.maximum_steps = maximum_steps
        self.predicate = draft.predicate
        self.counter = draft.counter
        self.continue_flag = draft.continue_flag
        self.maximum_count = draft.maximum_count
        self.zero_count = draft.zero_count
        self.one_count = draft.one_count
        self.has_more = draft.has_more
        self.under_limit = draft.under_limit
        self.body_operators = body
        self.graph = None
        self.mode = self.execution.loop_mode(
            world_size=model.world_size,
            allow_distributed=False,
        )

    @staticmethod
    def recording_draft(
        model: AbstractModel,
        *,
        maximum_steps: int,
    ) -> _PredicateLoopDraft:
        return _PredicateLoopDraft(model, maximum_steps=maximum_steps)

    def _reset(self) -> None:
        self.predicate.zero_()
        self.counter.zero_()
        self.continue_flag.fill_(1)

    def _iteration(self) -> None:
        self.predicate.zero_()
        self.body_operators.launch()
        self.counter.add_(self.one_count)
        torch.ne(self.predicate, self.zero_count, out=self.has_more)
        torch.lt(self.counter, self.maximum_count, out=self.under_limit)
        torch.logical_and(
            self.has_more,
            self.under_limit,
            out=self.has_more,
        )
        self.continue_flag.copy_(self.has_more)

    def _graph(self) -> Any:
        if self.graph is not None:
            return self.graph

        def body(graph: Any, _set_cond: bool, stream: int) -> None:
            self._iteration()

        self.graph = self.capture.build_conditional_graph(
            body=body,
            reset=self._reset,
            continue_flag=self.continue_flag,
            extra_state=(
                self.predicate,
                self.counter,
                self.continue_flag,
                self.has_more,
                self.under_limit,
                *self.body_operators.mutated_tensors,
            ),
        )
        return self.graph

    def execute(self) -> None:
        self._reset()
        if self.mode == "eager":
            while True:
                self._iteration()
                if int(self.continue_flag.item()) == 0:
                    break
            return
        self.execution.launch_conditional(self._graph())

    def close(self) -> None:
        graph, self.graph = self.graph, None
        body, self.body_operators = self.body_operators, None
        _close_program_resources(
            self.capture,
            (graph,),
            (body,),
            scope="predicate loop program",
        )


class _AdaptiveSubstepDraft:
    """Recording-only adaptive controls."""

    def __init__(
        self,
        *,
        candidate_dt: torch.Tensor,
        dt: torch.Tensor,
        maximum_dt: float,
        maximum_steps: int,
    ) -> None:
        self.candidate = candidate_dt
        self.time_step = dt
        self.maximum = maximum_dt
        self.maximum_steps = maximum_steps
        # The first program build may happen under ``torch.inference_mode``.
        # Runtime-owned controls must remain ordinary tensors because they are
        # also mutated by capture setup and cleanup outside inference mode.
        with torch.inference_mode(False):
            options = dict(device=candidate_dt.device, dtype=candidate_dt.dtype)
            self.duration = torch.zeros(1, **options)
            self.elapsed = torch.zeros(1, **options)
            self.counter = torch.zeros(
                1,
                device=candidate_dt.device,
                dtype=torch.int32,
            )
            self.continue_flag = torch.zeros_like(self.counter)
            self.error_flag = torch.zeros_like(self.counter)
            # Stable scalar scratch belongs to the loop program, not to an
            # iteration.  CUDA captures these addresses and eager execution
            # performs no per-substep tensor allocation.
            self.remaining = torch.zeros(1, **options)
            self.accepted = torch.zeros(1, **options)
            self.predicate_a = torch.zeros(
                1,
                device=candidate_dt.device,
                dtype=torch.bool,
            )
            self.predicate_b = torch.zeros_like(self.predicate_a)
            self.predicate_c = torch.zeros_like(self.predicate_a)
            self.maximum_value = torch.full(
                (1,),
                self.maximum,
                **options,
            )
            self.zero_value = torch.zeros(1, **options)
            self.maximum_count = torch.full(
                (1,),
                self.maximum_steps,
                device=candidate_dt.device,
                dtype=torch.int32,
            )
            self.one_count = torch.ones_like(self.maximum_count)
        from hydroforge.execution.substeps import SubstepFrame

        self.frame = SubstepFrame(
            index=self.counter,
            dt=self.time_step,
        )


class AdaptiveSubstepProgram:
    """Complete cached adaptive loop compiled from one recording draft."""

    def __init__(
        self,
        model: AbstractModel,
        *,
        draft: _AdaptiveSubstepDraft,
        proposal: Any,
        body: Any,
    ) -> None:
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
        self.execution = model._execution
        self.capture = self.execution.capture
        self.statistics = self.execution.statistics
        self.candidate = draft.candidate
        self.time_step = draft.time_step
        self.maximum = draft.maximum
        self.maximum_steps = draft.maximum_steps
        self.duration = draft.duration
        self.elapsed = draft.elapsed
        self.counter = draft.counter
        self.continue_flag = draft.continue_flag
        self.error_flag = draft.error_flag
        self.remaining = draft.remaining
        self.accepted = draft.accepted
        self.predicate_a = draft.predicate_a
        self.predicate_b = draft.predicate_b
        self.predicate_c = draft.predicate_c
        self.maximum_value = draft.maximum_value
        self.zero_value = draft.zero_value
        self.maximum_count = draft.maximum_count
        self.one_count = draft.one_count
        self.frame = draft.frame
        self.graphs: dict[bool, Any] = {}
        self.proposal_operators = proposal
        self.body_operators = body
        self.metal_iteration = None
        self.mode = self.execution.loop_mode(
            world_size=model.world_size,
            allow_distributed=False,
        )
        metal_iteration = (
            self._build_metal_iteration(proposal, body)
            if self.execution.capture_mode == "metal_icb"
            else None
        )
        self.metal_iteration = metal_iteration

    @staticmethod
    def recording_draft(
        *,
        candidate_dt: torch.Tensor,
        dt: torch.Tensor,
        maximum_dt: float,
        maximum_steps: int,
    ) -> _AdaptiveSubstepDraft:
        return _AdaptiveSubstepDraft(
            candidate_dt=candidate_dt,
            dt=dt,
            maximum_dt=maximum_dt,
            maximum_steps=maximum_steps,
        )

    def _build_metal_iteration(self, proposal: Any, body: Any) -> Any:
        from hydroforge.execution.metal_control import adaptive_control_commands
        from hydroforge.execution.operators import capture_metal_commands

        begin, accept, end = adaptive_control_commands(
            candidate=self.candidate,
            maximum=self.maximum,
            duration=self.duration,
            elapsed=self.elapsed,
            dt=self.time_step,
            counter=self.counter,
            continue_flag=self.continue_flag,
            error_flag=self.error_flag,
            maximum_steps=self.maximum_steps,
        )
        return capture_metal_commands(
            self.capture,
            (
                begin,
                *proposal.metal_commands(),
                accept,
                *body.metal_commands(),
                end,
            ),
            cyclic=True,
        )

    def close(self) -> None:
        graphs, self.graphs = tuple(self.graphs.values()), {}
        proposal, self.proposal_operators = self.proposal_operators, None
        body, self.body_operators = self.body_operators, None
        metal_iteration, self.metal_iteration = self.metal_iteration, None
        if metal_iteration is not None:
            graphs = (*graphs, metal_iteration.icb)
        _close_program_resources(
            self.capture,
            graphs,
            (proposal, body),
            scope="adaptive substep program",
        )

    def invalidate_statistics(self, aggregator: Any) -> None:
        """Release only the adaptive graph that folded statistics."""
        del aggregator
        graph = self.graphs.pop(True, None)
        if graph is not None:
            self.capture.release(graph)

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
            self.candidate,
            self.remaining,
            out=self.accepted,
        )
        # ``accepted != accepted`` is exactly the NaN test needed here.  A
        # positive infinity cannot survive minimum(candidate, finite
        # remaining), while negative infinity is caught by <= 0.
        torch.eq(self.accepted, self.accepted, out=self.predicate_a)
        torch.logical_not(self.predicate_a, out=self.predicate_a)
        torch.le(self.accepted, self.zero_value, out=self.predicate_b)
        torch.logical_or(
            self.predicate_a,
            self.predicate_b,
            out=self.predicate_a,
        )
        # A bad proposal must terminate the device WHILE node.  Substitute the
        # finite positive remainder so the already-captured physics tail does
        # not receive zero/NaN before the host reports the strict error.
        torch.where(
            self.predicate_a,
            self.remaining,
            self.accepted,
            out=self.time_step,
        )
        self.body_operators.launch()
        self.elapsed.add_(self.time_step)
        self.counter.add_(self.one_count)
        torch.ge(
            self.counter,
            self.maximum_count,
            out=self.predicate_b,
        )
        torch.lt(self.elapsed, self.duration, out=self.predicate_c)
        torch.logical_and(
            self.predicate_b,
            self.predicate_c,
            out=self.predicate_b,
        )
        torch.logical_or(
            self.predicate_a,
            self.predicate_b,
            out=self.predicate_a,
        )
        self.error_flag.copy_(self.predicate_a)
        torch.logical_not(self.predicate_a, out=self.predicate_b)
        torch.logical_and(
            self.predicate_b,
            self.predicate_c,
            out=self.predicate_c,
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
                    graph=graph,
                    weight_src=self.time_step,
                    counter=self.counter,
                    continue_flag=self.continue_flag,
                    stream_ptr=stream,
                )

        graph = self.capture.build_conditional_graph(
            body=captured_body,
            reset=self._reset,
            continue_flag=self.continue_flag,
            extra_state=(
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
                *self.proposal_operators.mutated_tensors,
                *self.body_operators.mutated_tensors,
                *(extra or ()),
            ),
        )
        self.graphs[fold] = graph
        return graph

    def _check_completion(self, failed: int) -> None:
        if failed:
            raise ValueError(
                "adaptive substep proposal must be finite and positive and "
                "the interval must complete within "
                f"maximum_sub_steps={self.maximum_steps}"
            )

    def execute(self, duration: float, step: Any) -> int:
        self.duration.fill_(duration)
        if self.execution.capture_mode == "metal_icb":
            self.proposal_operators.reset_metal_errors()
            self.body_operators.reset_metal_errors()
        if self.mode == "eager":
            self._reset()
            count = 0
            while int(self.continue_flag.item()) != 0:
                self._iteration()
                self._check_completion(int(self.error_flag.item()))
                weight = float(self.time_step.item())
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(
                        "adaptive substep proposal produced an invalid accepted "
                        f"width {weight}"
                    )
                count += 1
                continuing = int(self.continue_flag.item()) != 0
                step.sample_adaptive(
                    weight=weight,
                    first_event=count == 1,
                    last_event=not continuing,
                )
            self.proposal_operators.check_metal_errors()
            self.body_operators.check_metal_errors()
            return count
        fold = step.run_statistics and self.statistics.should_fold()
        if fold:
            self.statistics.prelaunch(step.flags, step.total_weight)
        self._reset()
        self.execution.launch_conditional(self._graph(fold))
        status = (
            torch.cat((self.error_flag.reshape(1), self.counter.reshape(1)))
            .cpu()
            .tolist()
        )
        self._check_completion(status[0])
        count = status[1]
        if step.run_statistics and not fold:
            self.statistics.sample(
                sub_step=0,
                num_sub_steps=1,
                flags=step.flags,
                weight=duration,
                total_weight=step.total_weight,
            )
        step.advance_device(duration)
        return count

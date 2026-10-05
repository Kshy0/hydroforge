"""CUDA graph captures and device-side conditional loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cached_property, partial
from itertools import groupby
from typing import Any

import torch
from torch.utils._python_dispatch import _disable_current_modes

from hydroforge.core.errors import ResourceCleanupError, cleanup_on_exit
from hydroforge.execution.executors.base import DirectRunner, LoopExecutor
from hydroforge.execution.executors.eager import (
    EagerAdaptive,
    EagerFixed,
    EagerPredicate,
)
from hydroforge.execution.loops import distributed_capture_safe
from hydroforge.execution.operators import launch_operators, rollback_tensors
from hydroforge.kernels.backends.cuda_control import (
    StatisticsControls,
    control_requests,
    fixed_end,
    set_conditional,
)
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain.cuda import ConditionalWhileGraph
from hydroforge.statistics.phases import (
    CONTROL_FLAGS,
    CONTROL_PHASE,
    CONTROL_WEIGHT,
    sample_phase_expr,
)


def _launch_all(launches: tuple[Callable[[], None], ...]) -> None:
    for launch in launches:
        launch()


def _capture_runs(
    program: Any,
) -> tuple[tuple[bool, Callable[[], None], tuple[torch.Tensor, ...]], ...]:
    """Split launch order into CUDA-capturable runs and host-only operators."""

    values = program.output_values()
    runs = []
    for safe, group in groupby(
        program.operators,
        key=lambda operator: getattr(operator, "cuda_graph_capture_safe", True),
    ):
        group = tuple(group)
        if safe:
            writes = rollback_tensors(group)
            runs.append((True, partial(launch_operators, group, values), writes))
        else:
            runs.extend((False, operator.launch, operator.writes) for operator in group)
    return tuple(runs)


class CudaGraphExecutor(LoopExecutor):
    """Capture CUDA graphs: fixed loops replay bounded batches of iterations,
    and conditional WHILE graphs run adaptive and predicate loops
    whole on the device.

    Captures run on one side stream and allocate from one graph pool owned by
    this executor, on the model device, while callers keep their streams.
    """

    name = "cuda_graph"

    def __init__(
        self,
        device: torch.device,
        *,
        world_size: int = 1,
        warmup_iterations: int = 3,
    ) -> None:
        if type(warmup_iterations) is not int or warmup_iterations < 0:
            raise ValueError("warmup_iterations must be a non-negative exact int")
        super().__init__(device, world_size=world_size)
        self.warmup_iterations = warmup_iterations
        self._graph_pool: Any = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._statistics_graph: tuple[Any, torch.cuda.CUDAGraph] | None = None
        self._requested: set[str] = set()

    def compile_requests(self, statistics: bool) -> tuple[CompileRequest, ...]:
        """The loop control programs not yet requested; statistics folding
        adds the sample control program."""

        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        requests = tuple(
            request
            for request in control_requests(
                index, sample_phase_expr if statistics else None
            )
            if request.key not in self._requested
        )
        self._requested.update(request.key for request in requests)
        return requests

    @property
    def graph_pool(self) -> Any:
        if self._graph_pool is None:
            with torch.cuda.device(self.device):
                self._graph_pool = torch.cuda.graph_pool_handle()
        return self._graph_pool

    @property
    def capture_stream(self) -> torch.cuda.Stream:
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=self.device)
        return self._capture_stream

    @cached_property
    def statistics_controls(self) -> StatisticsControls:
        return StatisticsControls(sample_phase_expr)

    @staticmethod
    def _snapshot(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        """Keep cold-path rollback state off the accelerator."""
        return [tensor.detach().to(device="cpu", copy=True) for tensor in tensors]

    @staticmethod
    def _restore(tensors: tuple[torch.Tensor, ...], saved: list[torch.Tensor]) -> None:
        with cleanup_on_exit(
            "captured tensor restoration",
            (
                partial(live.copy_, value)
                for live, value in zip(tensors, saved, strict=True)
            ),
        ):
            pass

    def capture(
        self,
        body: Callable[[], None],
        *,
        mutated_state: Iterable[torch.Tensor],
    ) -> torch.cuda.CUDAGraph:
        """Capture on the model device while preserving state and caller streams."""

        with torch.cuda.device(self.device):
            state = tuple(dict.fromkeys(mutated_state))
            snapshot = self._snapshot(state)
            current = torch.cuda.current_stream(self.device)
            side = self.capture_stream
            side.wait_stream(current)
            graph = None
            try:
                with torch.cuda.stream(side):
                    for _ in range(self.warmup_iterations):
                        self._restore(state, snapshot)
                        body()
                    # Every run must start with valid loop controls and data;
                    # restoring only afterwards cannot undo an invalid access.
                    self._restore(state, snapshot)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.graph_pool, stream=side):
                        body()
                current.wait_stream(side)
                self._restore(state, snapshot)
            except BaseException as primary:
                failures: list[BaseException] = [primary]
                try:
                    side.synchronize()
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
                try:
                    self._restore(state, snapshot)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
                if graph is not None:
                    try:
                        self._finalizer(graph)()
                    except BaseException as cleanup_error:
                        failures.append(cleanup_error)
                if len(failures) > 1:
                    error = ResourceCleanupError(
                        "CUDA graph capture transaction", failures
                    )
                    raise error from primary
                raise
            return self.register(graph)

    def conditional_graph(
        self,
        body: Callable[[Any, bool], None],
        *,
        reset: Callable[[], None],
        state: Iterable[torch.Tensor],
    ) -> ConditionalWhileGraph:
        """Capture one conditional WHILE graph under this owner.

        ``body(graph, capturing)`` launches one iteration and sets the
        condition through ``graph.handle`` when ``capturing``; ``state`` is
        everything the warmup iterations may change.
        """

        with torch.cuda.device(self.device):
            return self._conditional_graph(body, reset, tuple(dict.fromkeys(state)))

    def _conditional_graph(
        self,
        body: Callable[[Any, bool], None],
        reset: Callable[[], None],
        state: tuple[torch.Tensor, ...],
    ) -> ConditionalWhileGraph:
        device = self.device
        graph = ConditionalWhileGraph()
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        current = torch.cuda.current_stream(device)
        side = self.capture_stream
        side.wait_stream(current)
        snapshot: list[torch.Tensor] | None = None
        try:
            with torch.cuda.stream(side):
                stream = side.cuda_stream
                snapshot = self._snapshot(state)
                for _ in range(self.warmup_iterations):
                    self._restore(state, snapshot)
                    reset()
                    body(graph, False)
                # Each warmup and capture starts from the same physical state.
                # Enqueue rollback on the side stream before recording operations.
                self._restore(state, snapshot)
                reset()
                torch._C._cuda_beginAllocateToPool(device_index, self.graph_pool)
                try:
                    graph.begin_capture(stream)
                    try:
                        body(graph, True)
                    except BaseException as primary:
                        try:
                            graph.end_capture(stream)
                        except BaseException as cleanup_error:
                            error = ResourceCleanupError(
                                "conditional CUDA stream capture",
                                [primary, cleanup_error],
                            )
                            raise error from primary
                        raise
                    else:
                        graph.end_capture(stream)
                finally:
                    torch._C._cuda_endAllocateToPool(device_index, self.graph_pool)
                graph.instantiate()
            current.wait_stream(side)
            self._restore(state, snapshot)
        except BaseException as primary:
            failures: list[BaseException] = [primary]
            # The body and warmups run on ``side``.  Exiting the stream context
            # does not wait for queued work, so restoring tensors or destroying
            # the graph immediately can race an in-flight failed capture.
            # Synchronization is cold-path cleanup only.
            try:
                side.synchronize()
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
            if snapshot is not None:
                try:
                    self._restore(state, snapshot)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            try:
                graph.destroy()
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
            if len(failures) > 1:
                error = ResourceCleanupError(
                    "conditional CUDA graph transaction", failures
                )
                raise error from primary
            raise
        self.register(graph)
        return graph

    def fixed_end(self, loop: Any) -> None:
        """Advance a fixed loop's counter."""

        fixed_end(
            counter=loop.counter, continue_flag=loop.continue_flag, count=loop.count
        )

    def fixed_statistics_end(self, loop: Any, launch: Any, width: torch.Tensor) -> None:
        """Advance a fixed loop and write its folded sample's controls."""

        states = launch.states
        self.statistics_controls.fixed_end(
            counter=loop.counter,
            continue_flag=loop.continue_flag,
            count=loop.count,
            weight_src=width,
            flags=states[CONTROL_FLAGS],
            weight=states[CONTROL_WEIGHT],
            phase=states[CONTROL_PHASE],
        )

    def _eager_fallback(self, *programs: Any) -> bool:
        # Multi-rank collectives need the host handshake before every launch.
        return self.world_size > 1 and not distributed_capture_safe(*programs)

    def fixed(self, loop: Any) -> Any:
        if self._eager_fallback(*loop.programs):
            return EagerFixed(loop)
        if not all(program.cuda_graph_capture_safe for program in loop.programs):
            return _SegmentedFixed(self, loop)
        return _Fixed(self, loop)

    def adaptive(self, loop: Any) -> Any:
        if self._eager_fallback(*loop.programs):
            return EagerAdaptive(loop)
        return _Adaptive(self, loop)

    def predicate(self, loop: Any) -> Any:
        if self._eager_fallback(loop.body):
            return EagerPredicate(loop)
        return _Predicate(self, loop)

    def outer(self, program: Any) -> Any:
        if not program.cuda_graph_capture_safe:
            return DirectRunner(program.launch)
        return self.repeated(program.launch, mutated_state=program.rollback_tensors)

    def repeated(
        self, body: Callable[[], None], *, mutated_state: tuple[Any, ...]
    ) -> _Repeated:
        return _Repeated(self, body, mutated_state)

    def sample(self, launch: Any, phase: int) -> None:
        """Replay the statistics program captured once per installed launch.

        The captured launch gates every kernel on the device phase control
        that ``phase`` was written to.
        """

        del phase
        with torch.cuda.device(self.device):
            cached = self._statistics_graph
            if cached is None or cached[0] is not launch:
                if cached is not None:
                    self._statistics_graph = None
                    self.release(cached[1])
                graph = self.capture(partial(launch, -1), mutated_state=launch.mutated)
                self._statistics_graph = cached = (launch, graph)
            cached[1].replay()

    def invalidate(self) -> None:
        self._statistics_graph = None
        super().invalidate()

    def close(self) -> None:
        try:
            self.invalidate()
        finally:
            self._graph_pool = None
            self._capture_stream = None


class _Repeated:
    """A body captured once and replayed as one graph."""

    __slots__ = ("executor", "body", "mutated_state", "graph")

    def __init__(
        self,
        executor: CudaGraphExecutor,
        body: Callable[[], None],
        mutated_state: tuple[Any, ...],
    ) -> None:
        self.executor = executor
        self.body = body
        self.mutated_state = mutated_state
        self.graph: Any = None

    def run(self) -> None:
        if self.graph is None:
            self.graph = self.executor.capture(
                self.body, mutated_state=self.mutated_state
            )
        self.graph.replay()

    def close(self) -> None:
        graph, self.graph = self.graph, None
        if graph is not None:
            self.executor.release(graph)


class _Fixed:
    """Replay a fixed loop in batches, with a separate final-tail graph.

    Eight iterations amortize host launches without capturing a graph for
    every dynamic loop count. Each statistics binding has at most three
    graphs: one iteration, one batch, and the optional final iteration.
    """

    batch_size = 8

    def __init__(self, executor: CudaGraphExecutor, loop: Any) -> None:
        self.executor = executor
        self.loop = loop
        # (folds statistics, final, repetitions) -> (statistics launch, graph)
        self.iterations: dict[tuple[bool, bool, int], tuple[Any, Any]] = {}
        # A folded sample weighs the exact width, like a host sample; the
        # loop's own width has the model dtype.
        with _disable_current_modes(), torch.inference_mode(False):
            self.width = torch.zeros(1, dtype=torch.float64, device=executor.device)
        self._width: float | None = None

    def run(self, count: int, duration: float, step: Any) -> int:
        loop = self.loop
        fold = step.fold
        launch = step.statistics.launch if fold else None
        loop.prepare(
            count,
            duration / count,
            controls=fold or loop.controlled,
            weight=loop.weighted,
        )
        if fold:
            width = duration / count
            if width != self._width:
                self.width.fill_(width)
                self._width = width
            step.statistics.prelaunch()
        batches, remainder = divmod(
            count - int(loop.final is not None), self.batch_size
        )
        regular = self._iteration(launch, final=False) if remainder else None
        batch = (
            self._iteration(launch, final=False, repetitions=self.batch_size)
            if batches
            else None
        )
        final = self._iteration(launch, final=True) if loop.final is not None else None
        for _ in range(batches):
            batch.replay()
        for _ in range(remainder):
            regular.replay()
        if final is not None:
            final.replay()
        if step.sampling and not fold:
            step.sample(first=count == 1, last=True, weight=duration)
        return count

    def _iteration(self, launch: Any, *, final: bool, repetitions: int = 1) -> Any:
        """Capture ordered iterations with their optional statistics samples."""

        key = (launch is not None, final, repetitions)
        cached = self.iterations.get(key)
        if cached is not None:
            if cached[0] is launch:
                return cached[1]
            del self.iterations[key]
            self.executor.release(cached[1])
        loop = self.loop
        executor = self.executor
        programs = loop.programs if final else (loop.body,)

        def body() -> None:
            for _ in range(repetitions):
                for program in programs:
                    program.launch()
                if launch is not None:
                    executor.fixed_statistics_end(loop, launch, self.width)
                    launch(-1)
                elif loop.controlled:
                    executor.fixed_end(loop)

        graph = executor.capture(
            body,
            mutated_state=(
                *((loop.controls,) if launch is not None or loop.controlled else ()),
                *(
                    tensor
                    for program in programs
                    for tensor in program.rollback_tensors
                ),
                *(() if launch is None else launch.mutated),
            ),
        )
        self.iterations[key] = (launch, graph)
        return graph

    def close(self) -> None:
        graphs = tuple(graph for _launch, graph in self.iterations.values())
        self.iterations.clear()
        self.executor.release_all(graphs, scope="fixed loop graphs")


class _SegmentedFixed:
    """A fixed loop whose body launches host-only operators (nested predicate
    loops, collectives): the runs between them replay as separate graphs in
    lexical order, and every iteration samples from the host."""

    def __init__(self, executor: CudaGraphExecutor, loop: Any) -> None:
        self.executor = executor
        self.loop = loop
        self.plans: dict[bool, tuple[Callable[[], None], ...]] = {}
        self.graphs: list[Any] = []

    def run(self, count: int, duration: float, step: Any) -> int:
        loop = self.loop
        width = duration / count
        loop.prepare(count, width, controls=loop.controlled, weight=loop.weighted)
        regular = self._plan(final=False)
        last = self._plan(final=True) if loop.final is not None else regular
        for index in range(count):
            for launch in last if index == count - 1 else regular:
                launch()
            step.sample(first=index == 0, last=index == count - 1, weight=width)
        return count

    def _plan(self, *, final: bool) -> tuple[Callable[[], None], ...]:
        """Capture operator runs between host-launched operators exactly once."""

        plan = self.plans.get(final)
        if plan is not None:
            return plan
        loop = self.loop
        executor = self.executor
        runs = list(_capture_runs(loop.body))
        if final:
            runs.extend(_capture_runs(loop.final))
        if loop.controlled:
            runs.append(
                (
                    True,
                    partial(executor.fixed_end, loop),
                    (loop.controls,),
                )
            )
        groups: list[tuple[bool, list[Any], list[torch.Tensor]]] = []
        for safe, launch, writes in runs:
            if safe and groups and groups[-1][0]:
                groups[-1][1].append(launch)
                groups[-1][2].extend(writes)
            else:
                groups.append((safe, [launch], list(writes)))
        captured: list[Any] = []
        steps: list[Callable[[], None]] = []
        try:
            for safe, launches, writes in groups:
                if not safe:
                    steps.append(launches[0])
                    continue
                graph = executor.capture(
                    partial(_launch_all, tuple(launches)), mutated_state=writes
                )
                captured.append(graph)
                steps.append(graph.replay)
        except BaseException:
            with cleanup_on_exit(
                "fixed segment capture",
                (
                    partial(
                        executor.release_all, captured, scope="fixed segment capture"
                    ),
                ),
            ):
                raise
        self.graphs.extend(captured)
        plan = self.plans[final] = tuple(steps)
        return plan

    def close(self) -> None:
        graphs, self.graphs = self.graphs, []
        self.plans.clear()
        self.executor.release_all(graphs, scope="fixed segment graphs")


class _Adaptive:
    """One conditional WHILE launch per adaptive loop, then one status read."""

    def __init__(self, executor: CudaGraphExecutor, loop: Any) -> None:
        self.executor = executor
        self.loop = loop
        # folds statistics -> (statistics launch, conditional graph)
        self.graphs: dict[bool, tuple[Any, Any]] = {}

    def run(self, duration: float, step: Any) -> int:
        loop = self.loop
        launch = None
        if step.fold:
            launch = step.statistics.launch
            step.statistics.prelaunch()
        loop.reset()
        self._graph(launch).launch()
        # The error flag must fail this step before statistics, outputs or the
        # committed clock can observe it, so this read stays synchronous.
        failed, count = loop.fetch(
            loop.completion_sources, loop.completion, loop.completion_host
        )
        loop.check_completion(failed)
        if launch is None and step.sampling:
            step.sample(first=True, last=True, weight=duration)
        return count

    def _graph(self, launch: Any) -> Any:
        """Return the device loop, folding ``launch`` into every iteration."""

        key = launch is not None
        cached = self.graphs.get(key)
        if cached is not None:
            if cached[0] is launch:
                return cached[1]
            del self.graphs[key]
            self.executor.release(cached[1])
        loop = self.loop
        executor = self.executor

        def body(graph: Any, capturing: bool) -> None:
            loop.iterate()
            if launch is not None:
                states = launch.states
                executor.statistics_controls.adaptive(
                    weight_src=loop.time_step,
                    continue_flag=loop.continue_flag,
                    counter=loop.counter,
                    flags=states[CONTROL_FLAGS],
                    weight=states[CONTROL_WEIGHT],
                    phase=states[CONTROL_PHASE],
                )
                launch(-1)
            set_conditional(
                continue_flag=loop.continue_flag,
                handle=graph.handle,
                set_cond=capturing,
            )

        graph = executor.conditional_graph(
            body,
            reset=loop.reset,
            state=(
                *loop.control_state,
                *loop.proposal.rollback_tensors,
                *loop.body.rollback_tensors,
                *(() if launch is None else launch.mutated),
            ),
        )
        self.graphs[key] = (launch, graph)
        return graph

    def close(self) -> None:
        graphs = tuple(graph for _launch, graph in self.graphs.values())
        self.graphs.clear()
        self.executor.release_all(graphs, scope="adaptive loop graphs")


class _Predicate:
    """One conditional WHILE launch per predicate loop."""

    def __init__(self, executor: CudaGraphExecutor, loop: Any) -> None:
        self.executor = executor
        self.loop = loop
        self.graph: Any = None

    def run(self) -> None:
        self.loop.reset()
        self._graph().launch()

    def _graph(self) -> Any:
        if self.graph is not None:
            return self.graph
        loop = self.loop
        executor = self.executor

        def body(graph: Any, capturing: bool) -> None:
            loop.iterate()
            set_conditional(
                continue_flag=loop.continue_flag,
                handle=graph.handle,
                set_cond=capturing,
            )

        self.graph = executor.conditional_graph(
            body,
            reset=loop.reset,
            state=(*loop.control_state, *loop.body.rollback_tensors),
        )
        return self.graph

    def close(self) -> None:
        graph, self.graph = self.graph, None
        if graph is not None:
            self.executor.release(graph)

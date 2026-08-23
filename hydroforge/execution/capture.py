"""Unified ownership and exception-safe lifecycle for backend captures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.contracts.naming import RESERVED_CONTROL_STATE

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class CaptureRuntime:
    """Own every CUDA Graph and Metal ICB created for one model instance."""

    def __init__(
        self, model: AbstractModel, *, warmup_iterations: int = 3,
    ) -> None:
        self.model = model
        self.warmup_iterations = warmup_iterations
        self._graph_pool: Any = None
        self._statistics_graphs: dict[
            int,
            tuple[Any, torch.cuda.CUDAGraph],
        ] = {}
        self._resources: list[tuple[Any, Callable[[], None]]] = []
        self._closed = False

    @property
    def graph_pool(self) -> Any:
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        return self._graph_pool

    def register(self, resource: Any) -> Any:
        """Register a closeable backend resource under model ownership."""

        finalizer = self._resource_finalizer(resource)
        self._resources.append((resource, finalizer))
        return resource

    @staticmethod
    def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
        """Keep cold-path rollback state off the accelerator."""
        return tensor.detach().to(device="cpu", copy=True)

    @staticmethod
    def _resource_finalizer(resource: Any) -> Callable[[], None]:
        for method_name in ("close", "destroy", "reset"):
            finalizer = getattr(resource, method_name, None)
            if callable(finalizer):
                return finalizer
        raise TypeError(
            "backend capture resources must define close(), destroy(), or reset()"
        )

    @staticmethod
    def _close_resource(resource: Any) -> None:
        CaptureRuntime._resource_finalizer(resource)()

    def release(self, resource: Any) -> None:
        """Release one owned resource and remove every retained reference."""

        index = next(
            index for index, (owned, _finalizer) in enumerate(self._resources)
            if owned is resource
        )
        _owned, finalizer = self._resources.pop(index)
        finalizer()

    @staticmethod
    def _save_extra(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        return [CaptureRuntime._snapshot(tensor) for tensor in tensors]

    @staticmethod
    def _restore_extra(
        tensors: tuple[torch.Tensor, ...], saved: list[torch.Tensor],
    ) -> None:
        for live, value in zip(tensors, saved, strict=True):
            live.copy_(value)

    def capture_cuda(
        self,
        body: Callable[[], None],
        *,
        mutated_state: Iterable[torch.Tensor],
    ) -> torch.cuda.CUDAGraph:
        """Warm and capture while restoring the declared write set exactly."""

        state = tuple(dict.fromkeys(mutated_state))
        snapshot = self._save_extra(state)

        def restore() -> None:
            self._restore_extra(state, snapshot)

        graph = None
        try:
            for _ in range(self.warmup_iterations):
                body()
            restore()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                body()
            restore()
        except BaseException as primary:
            failures: list[BaseException] = [primary]
            try:
                restore()
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
            if graph is not None:
                try:
                    self._close_resource(graph)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            if len(failures) > 1:
                error = ResourceCleanupError(
                    "CUDA graph capture transaction", failures,
                )
                raise error from primary
            raise
        return self.register(graph)

    def run_statistics(self, aggregator: Any, block_size: int) -> None:
        """Execute one statistics kernel through this model's shared capture pool."""

        key = id(aggregator)
        cached = self._statistics_graphs.get(key)
        if cached is None:
            states = aggregator._kernel_states
            extras = tuple(
                value for name, value in states.items()
                if isinstance(value, torch.Tensor)
                and name not in RESERVED_CONTROL_STATE
            )
            graph = self.capture_cuda(
                lambda: aggregator._aggregator_function(states, block_size),
                mutated_state=extras,
            )
            self._statistics_graphs[key] = (aggregator, graph)
        else:
            owner, graph = cached
            if owner is not aggregator:
                raise RuntimeError(
                    "statistics capture identity was reused while still active"
                )
        graph.replay()

    def invalidate_statistics(self, aggregator: Any) -> None:
        """Release one cached statistics graph before its bindings change."""

        cached = self._statistics_graphs.pop(id(aggregator), None)
        if cached is not None:
            owner, graph = cached
            if owner is not aggregator:
                raise RuntimeError(
                    "statistics capture identity was reused while still active"
                )
            self.release(graph)

    def build_conditional_graph(
        self,
        *,
        body: Callable[[Any, bool, int], None],
        reset: Callable[[], None],
        continue_flag: torch.Tensor,
        extra_state: Iterable[torch.Tensor] = (),
    ) -> Any:
        """Capture one CUDA conditional-WHILE graph under this owner."""
        from hydroforge.execution.cuda_graph import ConditionalWhileGraph

        device = torch.device(self.model.device)
        extras = tuple(extra_state)
        state = tuple(dict.fromkeys((*extras, continue_flag)))

        graph = ConditionalWhileGraph()
        device_index = (
            torch.cuda.current_device() if device.index is None
            else device.index
        )
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device)
        side.wait_stream(current)
        snapshot: list[torch.Tensor] | None = None
        try:
            with torch.cuda.stream(side):
                stream = side.cuda_stream
                saved = self._save_extra(state)
                snapshot = saved

                def restore() -> None:
                    self._restore_extra(state, saved)

                for _ in range(self.warmup_iterations):
                    reset()
                    body(graph, False, stream)
                    graph.set_conditional(continue_flag, False, stream)
                    restore()
                reset()
                torch._C._cuda_beginAllocateToPool(
                    device_index, self.graph_pool,
                )
                try:
                    graph.begin_capture(stream)
                    body(graph, True, stream)
                    graph.set_conditional(continue_flag, True, stream)
                    graph.end_capture(stream)
                finally:
                    torch._C._cuda_endAllocateToPool(
                        device_index, self.graph_pool,
                    )
                graph.instantiate()
            current.wait_stream(side)
            restore()
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
                    self._restore_extra(state, snapshot)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            try:
                graph.destroy()
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
            if len(failures) > 1:
                error = ResourceCleanupError(
                    "conditional CUDA graph transaction", failures,
                )
                raise error from primary
            raise
        self.register(graph)
        return graph

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.invalidate()
        finally:
            self._graph_pool = None

    def invalidate(self) -> None:
        """Release captures whose fixed bindings may have become stale."""

        resources, self._resources = self._resources, []
        self._statistics_graphs.clear()
        failures: list[BaseException] = []
        for _resource, finalizer in reversed(resources):
            try:
                finalizer()
            except BaseException as error:
                failures.append(error)
        if failures:
            error = ResourceCleanupError("backend capture resources", failures)
            raise error from failures[0]

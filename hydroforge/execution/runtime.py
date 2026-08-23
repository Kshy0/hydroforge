"""Backend-neutral cached execution for explicit compiled substeps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.kernels.binding import KernelBinder
from hydroforge.execution.capture import CaptureRuntime
from hydroforge.statistics.observer import DisabledStatisticsObserver

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class ModelExecution:
    """The single explicit owner of one model's runtime plans and resources."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self.device = torch.device(model.device)
        self.backend = model._backend
        if model.execution_mode == "eager":
            self.capture_mode = "eager"
        elif self.backend in {"cuda", "triton"} and self.device.type == "cuda":
            self.capture_mode = "cuda_graph"
        elif self.backend == "metal" and self.device.type == "mps":
            self.capture_mode = "metal_icb"
        else:
            self.capture_mode = "eager"
        self.capture = CaptureRuntime(model)
        self.statistics = DisabledStatisticsObserver(model)
        self.kernel_binding = KernelBinder(model)
        self.step_policies: dict[Any, Any] = {}
        self.programs: dict[Any, Any] = {}
        self._model_tensor_ids: frozenset[int] = frozenset()
        self._tensor_index_valid = False
        self.step: Any = None
        self._failure: tuple[str, str, str] | None = None
        self.closed = False

    @property
    def failure(self) -> tuple[str, str, str] | None:
        """Return the recorded external mutation failure, if any."""

        return self._failure

    @staticmethod
    def poisoned_error(failure: tuple[str, str, str]) -> RuntimeError:
        phase, error_type, message = failure
        return RuntimeError(
            "model execution is poisoned by a prior mutation failure "
            f"during {phase}: {error_type}: {message}; close this model "
            "and rebuild or restore a fresh instance from checkpoint"
        )

    def precompile_cuda_catalogs(
        self, catalogs: Any, opened_modules: Any,
    ) -> dict[str, Any]:
        """Materialize CUDA extensions required by the opened model modules."""
        from hydroforge.kernels.backends.cuda.precompile import (
            precompile_cuda_modules,
        )

        return precompile_cuda_modules(
            catalogs, opened_modules=opened_modules,
        )

    def poison(self, error: BaseException, *, phase: str) -> None:
        """Permanently reject further stepping after unprovable mutation."""

        if self.closed:
            return
        if self._failure is None:
            self._failure = (
                phase, type(error).__name__, str(error),
            )

    def is_model_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is address-stable declared model state.

        The ownership index is a cold-path validation aid for compiled
        substeps.  It is derived from the compiler namespace directly, so
        recording never walks modules through ``get_module`` and never caches
        module handles that the model body did not reference.
        """

        if not self._tensor_index_valid:
            self._refresh_model_tensor_index()
        return id(tensor) in self._model_tensor_ids

    def _refresh_model_tensor_index(self) -> None:
        fields = self.model._field_namespace
        identities: set[int] = set()
        for field_name, owners in fields.items():
            for owner in owners:
                schema_getter = getattr(owner.owner, "_get_tensor_schema", None)
                schema = (
                    None if schema_getter is None else schema_getter(field_name)
                )
                if (
                    schema is not None
                    and schema.tensor is not None
                    and schema.tensor.category == "virtual"
                    and field_name not in owner.owner.__dict__
                ):
                    # Optional buffer virtuals stay descriptors until an
                    # output request or initialize_model_state explicitly
                    # materializes them. Index construction must not turn
                    # every declared diagnostic into resident model state.
                    continue
                value = getattr(owner.owner, field_name)
                if isinstance(value, torch.Tensor):
                    identities.add(id(value))
        self._model_tensor_ids = frozenset(identities)
        self._tensor_index_valid = True

    def loop_mode(
        self, *, world_size: int, allow_distributed: bool,
    ) -> str:
        supported = self.capture_mode == "cuda_graph" and (
            world_size == 1 or allow_distributed
        )
        return "conditional" if supported else "eager"

    def launch_conditional(self, graph: Any) -> None:
        graph.launch(torch.cuda.current_stream(self.device).cuda_stream)

    def run_statistics(self, statistics: Any, block_size: int) -> None:
        """Execute cached statistics without leaking backend policy outward."""

        if self.capture_mode == "cuda_graph":
            self.capture.run_statistics(statistics, block_size)
        else:
            statistics._aggregator_function(
                statistics._kernel_states, block_size,
            )

    def invalidate_statistics(self, aggregator: Any) -> None:
        """Release every cache that retains a statistics specialization."""
        failures: list[BaseException] = []
        try:
            self.statistics.invalidate()
        except BaseException as error:
            failures.append(error)
        seen_programs: set[int] = set()
        for program in tuple(self.programs.values()):
            identity = id(program)
            if identity in seen_programs:
                continue
            seen_programs.add(identity)
            invalidate = getattr(program, "invalidate_statistics", None)
            if invalidate is None:
                continue
            try:
                invalidate(aggregator)
            except BaseException as error:
                failures.append(error)
        try:
            self.capture.invalidate_statistics(aggregator)
        except BaseException as error:
            failures.append(error)
        if failures:
            error = ResourceCleanupError(
                "statistics execution caches", failures,
            )
            self.poison(error, phase="statistics invalidation")
            raise error from failures[0]

    def invalidate(self) -> None:
        if self.closed:
            return
        self.kernel_binding.invalidate()
        self._tensor_index_valid = False
        programs, self.programs = self.programs, {}
        failures: list[BaseException] = []
        for program in programs.values():
            try:
                program.close()
            except BaseException as error:
                failures.append(error)
        try:
            self.capture.invalidate()
        except BaseException as error:
            failures.append(error)
        if failures:
            error = ResourceCleanupError("model execution resources", failures)
            self.poison(error, phase="execution-plan invalidation")
            raise error from failures[0]

    def close(self) -> None:
        if self.closed:
            return
        failures: list[BaseException] = []
        try:
            self.invalidate()
        except BaseException as error:
            failures.append(error)
        try:
            self.capture.close()
        except BaseException as error:
            failures.append(error)
        self.step_policies.clear()
        self.closed = True
        if len(failures) == 1:
            raise failures[0]
        if failures:
            error = ResourceCleanupError("model execution close", failures)
            raise error from failures[0]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Backend-neutral cached execution for explicit compiled substeps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from hydroforge.core.errors import (
    cleanup_on_exit,
    failure_description,
)
from hydroforge.core.events import emit
from hydroforge.execution.binding import KernelBinder
from hydroforge.execution.executors import EagerExecutor, select_executor
from hydroforge.execution.step_fields import StepFieldRuntime
from hydroforge.kernels.toolchain import CompileRequest

if TYPE_CHECKING:
    from hydroforge.execution.session import ModelRuntime


class ModelExecution:
    """The single explicit owner of one model's runtime plans and resources."""

    def __init__(self, runtime: ModelRuntime) -> None:
        plan = runtime.plan
        self.runtime = runtime
        self.device = plan.device
        self.backend = plan.backend
        self.dtype = plan.dtype
        self.executor = select_executor(plan)
        if (
            plan.capture
            and plan.backend.capture == "cuda_graph"
            and plan.device.type == "cuda"
            and isinstance(self.executor, EagerExecutor)
        ):
            # An eager fallback costs a host launch per kernel; say why.
            emit(
                runtime,
                "warning",
                "execution.cuda_graphs_unavailable",
                "CUDA graph capture is unavailable, so managed steps run eagerly; "
                "it needs a CUDA toolkit of 12.4 or newer (set CUDA_HOME) and "
                "a matching driver",
                device=str(plan.device),
            )
        self.kernel_binding = KernelBinder(runtime)
        self.step_fields = StepFieldRuntime(plan, executor=self.executor)
        self.step_policies: dict[Any, Any] = {}
        self.programs: dict[Any, Any] = {}
        self.host_variants: dict[Any, Any] = {}
        # Validated adaptive-scope limits keyed by their exact inputs.
        self.adaptive_requests: dict[Any, Any] = {}
        # All-reduce transports of each process group.
        self.reducers: dict[str, Any] = {}
        self._model_tensor_ids: frozenset[int] = frozenset()
        self._tensor_index_valid = False
        self.structural_revision = 0
        self.step: Any = None
        self._failure: tuple[str, str, str] | None = None
        self.closed = False

    def take_pending(self) -> list[CompileRequest]:
        """Programs the model will launch that no compile batch has taken.

        A recording hands them to the batch that compiles its kernel calls,
        so statistics, step-field and loop-control programs compile together
        with the physics.
        """

        statistics = getattr(self.runtime, "statistics", None)
        return [
            *self.executor.compile_requests(statistics is not None),
            *self.step_fields.take_requests(),
            *(() if statistics is None else statistics.take_requests()),
        ]

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

    def poison(self, error: BaseException, *, phase: str) -> None:
        """Permanently reject further stepping after unprovable mutation."""

        if self.closed:
            return
        if self._failure is None:
            self._failure = (
                phase,
                type(error).__name__,
                failure_description(error)["message"],
            )

    def is_model_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is address-stable declared model state.

        The ownership index is a cold-path validation aid for compiled
        substeps.  It is derived from the bound field owners directly, so
        recording never walks modules and never caches module handles that
        the model body did not reference.
        """

        if id(tensor) in self.step_fields.identities:
            return True
        if not self._tensor_index_valid:
            self._refresh_model_tensor_index()
        return id(tensor) in self._model_tensor_ids

    def _refresh_model_tensor_index(self) -> None:
        fields = self.runtime.field_owners
        identities: set[int] = set()
        for field_name, owners in fields.items():
            for owner in owners:
                metadata_getter = getattr(owner.owner, "_tensor_metadata", None)
                metadata = (
                    None if metadata_getter is None else metadata_getter(field_name)
                )
                if (
                    metadata is not None
                    and metadata.category == "virtual"
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
        halo = getattr(self.runtime, "halo", None)
        if halo is not None:
            identities.update(id(tensor) for tensor in halo.tensors())
        for reducer in self.reducers.values():
            identities.update(id(tensor) for tensor in reducer.tensors())
        self._model_tensor_ids = frozenset(identities)
        self._tensor_index_valid = True

    def invalidate(self) -> None:
        if self.closed:
            return
        self.kernel_binding.invalidate()
        for name in self.runtime.plan.spec.kernel_fields:
            self.runtime.owner.__dict__.pop(name, None)
        for module in self.runtime.modules.values():
            for name in module.spec().kernel_fields:
                module.__dict__.pop(name, None)
        self._tensor_index_valid = False
        programs, self.programs = self.programs, {}
        self.host_variants = {}
        self.adaptive_requests = {}
        try:
            with cleanup_on_exit(
                "model execution resources",
                (
                    *(program.close for program in programs.values()),
                    self.step_fields.invalidate_program,
                    self.executor.invalidate,
                ),
            ):
                pass
        except BaseException as error:
            self.poison(error, phase="execution-plan invalidation")
            raise

    def reducer(self, group: Any, device: torch.device) -> Any:
        """The all-reduce transports of ``group``, chosen collectively on
        first use (see :mod:`hydroforge.parallel.reduce`)."""

        key = group.group_name
        if key not in self.reducers:
            from torch.utils._python_dispatch import _disable_current_modes

            from hydroforge.parallel.reduce import ReduceRuntime

            # First use may come while a substep is being recorded; the
            # collective setup runs outside the recorder.
            with _disable_current_modes():
                reducer = ReduceRuntime(group, device, self.backend.name)
            self.reducers[key] = reducer
            self._tensor_index_valid = False
            emit(
                self.runtime,
                "info",
                "model.reduce",
                "All-reduce transports of a process group",
                rank=self.runtime.plan.rank,
                group=key,
                transports=reducer.transports,
                skipped=dict(reducer.skipped),
            )
        return self.reducers[key]

    def close(self) -> None:
        if self.closed:
            return
        try:
            with cleanup_on_exit(
                "model execution close",
                (
                    self.executor.close,
                    self.step_policies.clear,
                    self.step_fields.close,
                    self.reducers.clear,
                ),
            ):
                self.invalidate()
        finally:
            self.closed = True

"""Backend-neutral cached execution for explicit compiled substeps."""

from __future__ import annotations

from typing import Any

import torch

from hydroforge.contracts import ResourceCleanupError
from hydroforge.kernels.binding import KernelBinder
from hydroforge.execution.capture import CaptureRuntime
from hydroforge.statistics.observer import StatisticsObserver
from hydroforge.kernels.registry import resolve_model_backend


class ModelExecution:
    """The single explicit owner of one model's runtime plans and resources."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.device = torch.device(model.device)
        self.backend = resolve_model_backend(self.device)
        required_device = {
            "cuda": "cuda", "triton": "cuda", "metal": "mps",
        }.get(self.backend)
        if required_device is not None and self.device.type != required_device:
            raise RuntimeError(
                f"HydroForge backend {self.backend!r} requires a "
                f"{required_device!r} model device, got {str(self.device)!r}; "
                "select the intended backend explicitly instead of running "
                "through eager fallback"
            )
        if model.execution_mode == "eager":
            self.capture_mode = "eager"
        elif self.backend in {"cuda", "triton"} and self.device.type == "cuda":
            self.capture_mode = "cuda_graph"
        elif self.backend == "metal" and self.device.type == "mps":
            self.capture_mode = "metal_icb"
        else:
            self.capture_mode = "eager"
        self.capture = CaptureRuntime(model)
        self.statistics = StatisticsObserver(model)
        self.kernel_binding = KernelBinder(model)
        self.step_policies: dict[Any, Any] = {}
        self.programs: dict[Any, Any] = {}
        self._model_tensor_ids: frozenset[int] = frozenset()
        self._field_namespace: Any = None
        self._tensor_index_valid = False
        # Execution semantics are independent from the optional progress UI.
        # Drivers may configure this after model initialization through
        # ``set_total_steps``; managed-step policies read only this value.
        self.total_steps = 0
        self.step: Any = None
        self.active_step: Any = None
        self._failure: tuple[str, str, str] | None = None
        self.closed = False

    def require_open(self) -> None:
        if self.closed:
            raise RuntimeError("model execution runtime is closed")
        if self._failure is not None:
            phase, error_type, message = self._failure
            raise RuntimeError(
                "model execution is poisoned by a prior managed-step failure "
                f"during {phase}: {error_type}: {message}; close this model "
                "and rebuild or restore a fresh instance from checkpoint"
            )

    def precompile_cuda_catalogs(
        self, catalogs: Any, opened_modules: Any,
    ) -> dict[str, Any]:
        """Materialize CUDA extensions required by the opened model modules."""
        if self.backend != "cuda":
            raise RuntimeError(
                "CUDA catalog precompilation requires the CUDA backend"
            )
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

    def require_between_steps(self, operation: str) -> None:
        """Require a healthy stable boundary outside a managed step."""

        self.require_open()
        if not isinstance(operation, str) or not operation:
            raise TypeError("between-step operation must be a non-empty str")
        if self.active_step is not None:
            raise RuntimeError(
                f"{operation} is forbidden during an active managed step; "
                "apply host-side state changes between step_advance calls"
            )

    def is_model_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is address-stable declared model state.

        The ownership index is a cold-path validation aid for compiled
        substeps.  It is derived from the compiler namespace directly, so
        recording never walks modules through ``get_module`` and never caches
        module handles that the model body did not reference.
        """

        if not self._tensor_index_valid:
            if self._field_namespace is None:
                plan = getattr(self.model, "_plan", None)
                if plan is None:
                    raise RuntimeError(
                        "compiled substep recording requires a completed model plan"
                    )
                self._field_namespace = plan.kernels.fields
            self._refresh_model_tensor_index()
        return id(tensor) in self._model_tensor_ids

    def install_model_plan(self, plan: Any) -> None:
        """Install the compiled namespace and materialize its tensor index."""

        self._field_namespace = plan.kernels.fields
        self._refresh_model_tensor_index()

    def _refresh_model_tensor_index(self) -> None:
        fields = self._field_namespace
        if fields is None:
            raise RuntimeError("model tensor indexing requires a compiled model plan")
        identities: set[int] = set()
        for field_name, owners in fields.owners.items():
            for owner in owners:
                try:
                    value = getattr(owner.owner, field_name)
                except AttributeError as error:
                    raise RuntimeError(
                        "compiled field owner no longer exposes "
                        f"{owner.module_name}.{field_name}"
                    ) from error
                if isinstance(value, torch.Tensor):
                    identities.add(id(value))
        self._model_tensor_ids = frozenset(identities)
        self._tensor_index_valid = True

    def mark_dependency(self) -> None:
        if self.capture_mode != "metal_icb":
            return
        from hydroforge.kernels.backends.metal.runtime import recording_metal_sequence

        sequence = recording_metal_sequence()
        if sequence is not None and sequence.prepared_commands:
            sequence.mark_barrier()

    def nested_capture_active(self) -> bool:
        return bool(
            self.capture_mode == "cuda_graph"
            and torch.cuda.is_current_stream_capturing()
        )

    def loop_mode(
        self, *, world_size: int, allow_distributed: bool,
    ) -> str:
        supported = self.capture_mode == "cuda_graph" and (
            world_size == 1 or allow_distributed
        )
        return "conditional" if supported else "eager"

    def launch_conditional(self, graph: Any) -> None:
        if self.capture_mode != "cuda_graph":
            raise RuntimeError("conditional device loop requires CUDA graph mode")
        graph.launch(torch.cuda.current_stream(self.device).cuda_stream)

    def run_statistics(self, statistics: Any, block_size: int) -> None:
        """Execute cached statistics without leaking backend policy outward."""

        if self.capture_mode == "cuda_graph":
            self.capture.run_statistics(statistics, block_size)
        else:
            statistics._aggregator_function(
                statistics._kernel_states, block_size,
            )

    def _step_policy_index(self) -> dict[str, Any]:
        policies: dict[str, Any] = {}
        for policy in self.step_policies.values():
            function = policy.descriptor.function
            name = f"{function.__module__}.{function.__qualname__}"
            if name in policies:
                raise RuntimeError(
                    f"managed-step checkpoint key {name!r} is not unique"
                )
            policies[name] = policy
        return policies

    def checkpoint_step_state(self) -> dict[str, Any]:
        """Return JSON-safe managed-step counters for exact continuation."""

        step_runtime = self.step
        return {
            "total_steps": self.total_steps,
            "schedule_step": (
                None if step_runtime is None
                else step_runtime.state.schedule_step
            ),
            "completed": {
                name: policy.completed_steps
                for name, policy in sorted(self._step_policy_index().items())
            },
        }

    def validate_checkpoint_step_state(
        self, state: Any,
    ) -> tuple[int, tuple[tuple[Any, int], ...]]:
        """Validate persisted counters without mutating live execution state."""

        if not isinstance(state, dict) or set(state) != {
            "total_steps", "schedule_step", "completed",
        }:
            raise ValueError(
                "checkpoint managed-step state must contain exactly "
                "'total_steps', 'schedule_step', and 'completed'"
            )
        total = state["total_steps"]
        if type(total) is not int or total < 0:
            raise ValueError("checkpoint total_steps must be a non-negative int")
        completed = state["completed"]
        if not isinstance(completed, dict):
            raise TypeError("checkpoint completed-step state must be a mapping")
        schedule_step = state["schedule_step"]
        if schedule_step is not None and (
            type(schedule_step) is not int or schedule_step < 0
        ):
            raise ValueError(
                "checkpoint schedule_step must be None or a non-negative int"
            )
        schedule = None if self.step is None else self.step.schedule
        if (
            schedule_step is not None
            and schedule is not None
            and schedule_step >= len(schedule)
        ):
            raise ValueError("checkpoint schedule_step is outside the model schedule")
        policies = self._step_policy_index()
        if set(completed) != set(policies):
            raise ValueError(
                "checkpoint managed-step methods do not match the model: "
                f"missing={sorted(set(policies).difference(completed))}, "
                f"extra={sorted(set(completed).difference(policies))}"
            )
        staged = []
        for name, policy in sorted(policies.items()):
            count = completed[name]
            if type(count) is not int or count < 0:
                raise ValueError(
                    f"checkpoint completed count for {name!r} must be a "
                    "non-negative int"
                )
            if total > 0 and count > total:
                raise ValueError(
                    f"checkpoint completed count for {name!r} exceeds "
                    f"total_steps={total}"
                )
            staged.append((policy, count))
        if self.total_steps not in {0, total}:
            raise ValueError(
                f"checkpoint total_steps={total} conflicts with configured "
                f"total_steps={self.total_steps}"
            )
        return total, schedule_step, tuple(staged)

    def restore_checkpoint_step_state(
        self, state: tuple[int, int | None, tuple[tuple[Any, int], ...]],
    ) -> None:
        """Commit a state returned by :meth:`validate_checkpoint_step_state`."""

        total, schedule_step, staged = state
        self.total_steps = total
        step_runtime = self.step
        if step_runtime is None and schedule_step is not None:
            raise RuntimeError(
                "cannot restore a model schedule cursor before step compilation"
            )
        if step_runtime is not None:
            step_runtime.state.schedule_step = schedule_step
        for policy, count in staged:
            policy.completed_steps = count

    def invalidate(self) -> None:
        if self.closed:
            raise RuntimeError("model execution runtime is closed")
        if self.active_step is not None:
            raise RuntimeError(
                "execution plan invalidation is forbidden during an active "
                "managed step"
            )
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
        if self.active_step is not None:
            raise RuntimeError(
                "model execution close is forbidden during an active managed step"
            )
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

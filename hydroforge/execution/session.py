"""Single owner of one model's runtime state and external resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import torch

from hydroforge.core.errors import (
    ResourceCleanupError,
    distributed_failure_error,
    failure_description,
)
from hydroforge.core.events import emit
from hydroforge.core.time import DateLike
from hydroforge.declare.module import AbstractModule, construct_module
from hydroforge.declare.spec import ModuleBinding
from hydroforge.execution.channel import LocalChannel, ProcessGroupChannel
from hydroforge.execution.checkpoint import CheckpointRuntime
from hydroforge.execution.inputs import InputBinding, prepare_payloads, shard_inputs
from hydroforge.execution.namespace import bind_field_owners, bind_namespace
from hydroforge.execution.progress import ProgressRuntime
from hydroforge.execution.runtime import ModelExecution
from hydroforge.execution.signatures import compiled_signature, declaration_signature
from hydroforge.execution.step import compile_step_policies
from hydroforge.execution.structure import (
    StructuralUpdateContext,
    StructuralUpdateResult,
)
from hydroforge.io.manifest import write_model_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from hydroforge.compiler.plan import ModelPlan
    from hydroforge.contracts.schedule import SimulationSchedule
    from hydroforge.declare.tensors import ModulePayload
    from hydroforge.execution.parameters import ParameterPlanRuntime
    from hydroforge.execution.partition import PartitionRuntime
    from hydroforge.statistics.runtime import StatisticsRuntime

_EMPTY_STRUCTURE_HOOK = AbstractModule.update_structure


class RuntimeClock:
    """Next execution index and physical date of one runtime.

    ``time`` is the unscheduled clock; a schedule derives the date from
    ``schedule_index`` instead.
    """

    __slots__ = ("schedule", "schedule_index", "time")

    def __init__(
        self, schedule: SimulationSchedule | None, initial_time: DateLike | None
    ) -> None:
        self.schedule = schedule
        self.schedule_index = 0
        self.time = initial_time if schedule is None else None

    @property
    def current_time(self) -> DateLike | None:
        """Physical date of the next step, or the end of a completed schedule."""

        schedule = self.schedule
        if schedule is None:
            return self.time
        if self.schedule_index == len(schedule):
            return schedule._end
        return schedule._step_at_trusted(self.schedule_index).start


class ModelRuntime:
    """Single owner of one model's runtime state and external resources.

    Created empty with the model: it holds the plan and the rank channel
    (whose agreement sequence spans every materialization), and performs no
    I/O or allocation until ``materialize()``. One release routine serves both a failed
    materialization and ``close()``.

    Only operations that advance or rewrite the run (managed steps,
    between-step APIs, checkpoint saves and structure updates) materialize a
    declared runtime; a closed runtime is rebuilt only by an explicit
    ``materialize()``.
    """

    state: Literal["declared", "materializing", "materialized", "closed"]
    input: InputBinding | None
    partition: PartitionRuntime | None
    modules: dict[str, AbstractModule]
    structure_hooks: tuple[Callable[[StructuralUpdateContext], None], ...]
    module_links: Mapping[str, AbstractModule | None] | None
    namespace: Mapping[str, Any]
    field_owners: Mapping[str, Any]
    execution: ModelExecution | None
    statistics: StatisticsRuntime | None
    checkpoint: CheckpointRuntime | None
    parameters: ParameterPlanRuntime | None
    progress: ProgressRuntime | None

    def __init__(self, owner: Any, plan: ModelPlan) -> None:
        self.owner = owner
        self.plan = plan
        self.event_sink = owner.event_sink
        self.channel = (
            LocalChannel()
            if plan.world_size == 1
            else ProcessGroupChannel(plan.world_size, plan.device, plan.parallel)
        )
        self.state = "declared"
        self._discard()

    @staticmethod
    def of(model: Any) -> ModelRuntime:
        """Return the runtime owned by ``model``."""

        return model.__pydantic_private__["_runtime"]

    def _discard(self) -> None:
        """Drop every object derived from one materialization."""

        for name in self.plan.spec.kernel_fields:
            self.owner.__dict__.pop(name, None)
        self.clock = RuntimeClock(self.plan.schedule, self.plan.initial_time)
        self.input = None
        self.partition = None
        self.modules = {}
        self.structure_hooks = ()
        self.module_links = None
        self.namespace = MappingProxyType({})
        self.field_owners = MappingProxyType({})
        self.execution = None
        self.statistics = None
        self.checkpoint = None
        self.parameters = None
        self.progress = None
        self.model_state_entered = False

    def _release(self) -> list[BaseException]:
        """Attempt every release of one materialization; return the failures.

        Captured programs may reference author-owned buffers, so execution
        closes before the model releases the state it initialized.  The input
        proxy's read handles close last; a later read reopens them.
        """

        releases = [
            resource.close
            for resource in (self.statistics, self.execution)
            if resource is not None
        ]
        if self.model_state_entered:
            releases.append(self.owner.release_model_state)
        releases.append(self.owner.input_proxy.close)
        failures: list[BaseException] = []
        for release in releases:
            try:
                release()
            except BaseException as error:
                failures.append(error)
        self._discard()
        return failures

    def unavailable(self, what: str) -> RuntimeError:
        """Explain why ``what`` needs a materialized runtime in this state."""

        if self.state == "materializing":
            return RuntimeError(
                f"{what} is unavailable while the model is materializing"
            )
        if self.state == "closed":
            reason = (
                "the model runtime was closed. Call model.materialize() to "
                "build a new runtime from the declared initial state."
            )
        else:
            reason = (
                "the model has not been materialized. Call model.materialize() "
                "(or enter `with model:`) to load inputs and construct modules; "
                "the first managed step also materializes it."
            )
        return RuntimeError(f"{what} is unavailable: {reason}")

    def require_materialized(self, what: str) -> None:
        """Reject a query of ``what`` unless the runtime is materialized."""

        if self.state != "materialized":
            raise self.unavailable(what)

    def ensure_materialized(self, what: str) -> None:
        """Materialize a declared runtime before ``what`` advances the run."""

        if self.state == "declared":
            self.materialize()
        elif self.state != "materialized":
            raise self.unavailable(what)

    def materialize(self) -> None:
        """Bind inputs and build every runtime service as one transaction.

        Multi-rank materialization is collective: every rank first proves
        the same declaration and inputs, then commits only after every rank
        built its services. A closed runtime is rebuilt from the declared
        initial state.
        """

        if self.state == "materialized":
            return
        if self.state == "materializing":
            raise self.unavailable(f"{type(self.owner).__name__}.materialize")
        self.state = "materializing"
        distributed = self.channel.distributed
        if distributed:
            try:
                self._preflight()
            except BaseException as error:
                failures = self._release()
                self.state = "declared"
                if failures:
                    raise ResourceCleanupError(
                        "model after preflight failure", (error, *failures)
                    ) from error
                raise
        error: BaseException | None = None
        try:
            from hydroforge.kernels.emulated import factories

            with factories(self.plan.metal_emulation):
                self._build()
        except BaseException as caught:
            error = caught
        if distributed:
            self._commit(error)
            return
        if error is not None:
            failures = self._release()
            self.state = "declared"
            if failures:
                raise ResourceCleanupError(
                    "model after initialization failure", (error, *failures)
                ) from error
            raise error
        self.state = "materialized"

    def _preflight(self) -> None:
        """Prove every rank will materialize the same declaration and inputs."""

        signature: tuple[Any, ...] | None = None
        local_error: BaseException | None = None
        try:
            self.input = InputBinding(
                self.plan, self.owner.input_proxy, self.event_sink
            )
            signature = declaration_signature(self)
        except BaseException as error:
            local_error = error
        failures = self.channel.gather(
            local_error, phase="runtime.materialization.preflight", signature=signature
        )
        if not any(failure is not None for failure in failures):
            return
        if local_error is not None:
            raise local_error
        raise distributed_failure_error(
            "distributed model runtime declaration validation", failures
        )

    def _build(self) -> None:
        # Declaration-only imports do not need partition compilation or Numba.
        from hydroforge.execution.parameters import (
            LocalParameterCompiler,
            ParameterPlanRuntime,
        )
        from hydroforge.execution.partition import PartitionRuntime

        plan = self.plan
        owner = self.owner
        if self.input is None:
            self.input = InputBinding(plan, owner.input_proxy, self.event_sink)
        self.partition = PartitionRuntime(plan, self.input)
        # Every input-dependent check completes before any output side effect.
        payloads = prepare_payloads(self, owner.prepare_model_input(shard_inputs(self)))
        changes = LocalParameterCompiler(self, payloads).compile(plan.parameters)
        directory = plan.output.directory
        if directory is not None:
            self._prepare_output_directory(directory)
        execution = ModelExecution(self)
        self.execution = execution
        self.progress = ProgressRuntime(self)
        if directory is not None and plan.rank == 0:
            write_model_manifest(
                directory,
                model=plan.model,
                experiment_name=plan.output.config.experiment,
                backend=execution.backend.name,
                device=str(plan.device),
                precision=plan.precision,
                mixed_precision=plan.mixed_precision,
                metal_emulation=plan.metal_emulation,
                opened_modules=plan.modules,
                options=plan.options.resolved_dict(),
                options_specialization=plan.options.specialization_key(),
            )
        emit(
            self,
            "info",
            "model.initializing",
            "Initializing model",
            rank=plan.rank,
            modules=plan.modules,
        )
        emit(
            self,
            "info",
            "model.partition",
            "Using partition root",
            key=plan.spec.partition_key,
            group=plan.spec.partition_group,
        )
        self._construct_modules(payloads)
        for name in plan.modules:
            self.modules[name]._tensors._apply_modes()
        self.model_state_entered = True
        owner.initialize_model_state()
        self.checkpoint = CheckpointRuntime(self)
        if plan.output.declaration is not None:
            from hydroforge.execution.outputs import bind_statistics

            self.statistics = bind_statistics(self)
        self.field_owners = bind_field_owners(plan.fields.binding, self.modules, owner)
        execution._refresh_model_tensor_index()
        self.parameters = ParameterPlanRuntime(self, changes)
        compile_step_policies(self)
        self._emit_memory_summary()
        emit(self, "info", "model.initialized", "Model initialized")

    def _emit_memory_summary(self) -> None:
        """Report resident tensor memory by module; shared storage counts once."""

        plan = self.plan
        if plan.rank != 0:
            return
        seen: set[int] = set()
        modules: dict[str, float] = {}
        total = 0
        for name in plan.modules:
            module = self.modules[name]
            size = 0
            for field in module.spec().tensor_fields.values():
                # Unmaterialized computed fields stay lazy.
                if field.computed and field.name not in module.__dict__:
                    continue
                value = getattr(module, field.name, None)
                if (
                    isinstance(value, torch.Tensor)
                    and value.device.type == module.device.type
                    and value.data_ptr() not in seen
                ):
                    seen.add(value.data_ptr())
                    size += value.element_size() * value.nelement()
            total += size
            modules[name] = size / (1024 * 1024)
        statistics = (
            0 if self.statistics is None else self.statistics.get_memory_usage()
        )
        total += statistics
        if statistics:
            modules["StatisticsAggregator"] = statistics / (1024 * 1024)
        emit(
            self,
            "info",
            "model.memory",
            "Model memory summary",
            rank=plan.rank,
            modules=modules,
            total_mb=total / (1024 * 1024),
        )

    def _prepare_output_directory(self, directory: Path) -> None:
        """Acquire the output directory only after the inputs validated."""

        if self.plan.spatial_rank != 0:
            return
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            return
        emit(
            self,
            "warning",
            "output.directory_exists",
            "Output directory already exists; contents may be overwritten",
            directory=directory,
        )

    def _construct_modules(self, payloads: Mapping[str, ModulePayload]) -> None:
        specs = self.plan.spec.modules
        modules = self.modules
        for name in self.plan.module_order:
            spec = specs[name]
            view = payloads[name]
            modules[name] = construct_module(
                spec.module_type,
                view._input_values,
                ModuleBinding(
                    plan=self.plan.fields.modules[name],
                    references=MappingProxyType(
                        {
                            reference: modules.get(reference)
                            for reference in spec.references
                        }
                    ),
                    event_sink=self.event_sink,
                    prepared=True,
                    defaults=MappingProxyType(view._default_values),
                ),
            )
        self.input.clear_cache()
        self.structure_hooks = tuple(
            hook
            for name in self.plan.module_order
            if getattr(hook := modules[name].update_structure, "__func__", None)
            is not _EMPTY_STRUCTURE_HOOK
        )
        self.module_links = MappingProxyType(
            {name: modules.get(name) for name in specs}
        )
        self.namespace = bind_namespace(self.plan.fields.names, modules)

    def _commit(self, initialization_error: BaseException | None) -> None:
        """Commit materialization only after every rank reports success."""

        world_size = self.plan.world_size
        signature: tuple[Any, ...] | None = None
        if initialization_error is None:
            try:
                signature = compiled_signature(self)
            except BaseException as error:
                initialization_error = error
        run_id_candidate: str | None = None
        if initialization_error is None and self.plan.rank == 0:
            try:
                run_id_candidate = str(uuid4())
            except BaseException as error:
                initialization_error = error
        try:
            initialization_failures, run_id_payloads = self.channel.exchange(
                initialization_error,
                phase="runtime.materialization",
                signature=signature,
                payload=run_id_candidate,
            )
        except BaseException as coordination_error:
            cleanup_error = self._release_error()
            failures = tuple(
                error
                for error in (initialization_error, coordination_error, cleanup_error)
                if error is not None
            )
            if len(failures) == 1:
                raise failures[0]
            error = ResourceCleanupError(
                "distributed model initialization coordination", failures
            )
            raise error from coordination_error
        if not any(failure is not None for failure in initialization_failures):
            try:
                self._install_run_id(run_id_payloads)
            except BaseException as error:
                initialization_error = error
                description = failure_description(error)
                initialization_failures = tuple(description for _ in range(world_size))
            else:
                self.state = "materialized"
                return
        cleanup_error = self._release_error()
        primary = (
            initialization_error
            if initialization_error is not None
            else distributed_failure_error(
                "distributed model initialization", initialization_failures
            )
        )
        try:
            cleanup_failures = self.channel.gather(
                cleanup_error, phase="runtime.materialization.cleanup"
            )
        except BaseException as coordination_error:
            failures = [primary, coordination_error]
            if cleanup_error is not None:
                failures.append(cleanup_error)
            error = ResourceCleanupError(
                "distributed model initialization cleanup coordination", failures
            )
            raise error from primary
        if any(failure is not None for failure in cleanup_failures):
            cleanup_failure = (
                cleanup_error
                if cleanup_error is not None
                else distributed_failure_error(
                    "distributed model initialization cleanup", cleanup_failures
                )
            )
            error = ResourceCleanupError(
                "distributed model initialization rollback", (primary, cleanup_failure)
            )
            raise error from primary
        raise primary

    def _release_error(self) -> BaseException | None:
        """Release a failed materialization and return its cleanup failure."""

        failures = self._release()
        self.state = "declared"
        return ResourceCleanupError("model resources", failures) if failures else None

    def _install_run_id(self, payloads: tuple[Any, ...]) -> None:
        """Install the rank-zero output identity exchanged at runtime commit."""

        if len(payloads) != self.plan.world_size:
            raise RuntimeError(
                "distributed runtime materialization returned an invalid output run-ID payload count"
            )
        run_id = payloads[0]
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError(
                "distributed runtime materialization did not publish a non-empty output run ID from rank zero"
            )
        if any(payload is not None for payload in payloads[1:]):
            raise RuntimeError(
                "distributed runtime materialization received output run IDs from nonzero ranks"
            )
        if self.statistics is not None:
            self.statistics.sink.set_run_id(run_id)

    def require_healthy(self, what: str) -> None:
        """Materialize a declared runtime; reject a poisoned or detached one."""

        parallel = self.plan.parallel
        if parallel is not None:
            parallel.validate_live()
        self.ensure_materialized(what)
        execution = self.execution
        failure = execution.failure
        if failure is not None:
            raise execution.poisoned_error(failure)

    def update_structure(self) -> StructuralUpdateResult | None:
        """Run the ordered module structure pass and commit one staged update."""

        self.require_healthy(f"{type(self.owner).__name__}.update_structure")
        if not self.structure_hooks:
            return None
        context = StructuralUpdateContext(self)
        for hook in self.structure_hooks:
            hook(context)
        return context.commit()

    def close(self) -> None:
        """Release every resource of a materialized runtime; mark it closed."""

        if self.state != "materialized":
            return
        failures = self._release()
        self.state = "closed"
        if failures:
            error = ResourceCleanupError("model resources", failures)
            raise error from failures[0]

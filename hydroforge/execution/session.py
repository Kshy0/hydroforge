# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Single owner of one model's runtime state and external resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np
import torch

from hydroforge.contracts.precision import precisions_of
from hydroforge.core.errors import (
    ResourceCleanupError,
    distributed_failure_error,
    failure_description,
)
from hydroforge.core.events import emit
from hydroforge.core.identity import canonical
from hydroforge.core.time import (
    DateLike,
    normalize_calendar_dates,
    timedelta_microseconds,
)
from hydroforge.declare.initial import evaluate_after_init, materialize_declared
from hydroforge.declare.module import AbstractModule, construct_module
from hydroforge.declare.spec import ModuleBinding
from hydroforge.execution.channel import LocalChannel, ProcessGroupChannel
from hydroforge.execution.checkpoint import CheckpointRuntime
from hydroforge.execution.context import managed_step_active, model_call
from hydroforge.execution.inputs import InputBinding, prepare_payloads, shard_inputs
from hydroforge.execution.namespace import bind_field_owners, bind_namespace
from hydroforge.execution.progress import ProgressRuntime, ProgressState
from hydroforge.execution.runtime import ModelExecution
from hydroforge.execution.signatures import compiled_signature, declaration_signature
from hydroforge.execution.step import compile_step_policies
from hydroforge.execution.structure import (
    StructuralUpdateContext,
    StructuralUpdateResult,
)
from hydroforge.io.construction_input import check_options_fingerprint
from hydroforge.io.manifest import write_model_manifest
from hydroforge.kernels.spec import HaloArgument

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

    __slots__ = ("schedule", "schedule_index", "time", "elapsed_microseconds")

    def __init__(
        self, schedule: SimulationSchedule | None, initial_time: DateLike | None
    ) -> None:
        self.schedule = schedule
        self.schedule_index = 0
        self.time = initial_time if schedule is None else None
        self.elapsed_microseconds = 0

    @property
    def current_time(self) -> DateLike | None:
        """Physical date of the next step, or the end of a completed schedule."""

        schedule = self.schedule
        if schedule is None:
            return self.time
        if self.schedule_index == len(schedule):
            return schedule.end
        return schedule._step_at_trusted(self.schedule_index).start

    @property
    def next_step_seconds(self) -> float | None:
        """Width of the next scheduled step; unknown without a schedule."""

        schedule = self.schedule
        if schedule is None or self.schedule_index == len(schedule):
            return None
        step = schedule._step_at_trusted(self.schedule_index)
        return timedelta_microseconds(step.end - step.start) / 1_000_000


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
        self.halo = None
        self.slab = None
        self.topology = None
        self._halo_layouts: dict[int, tuple[Any, Any]] = {}
        # The halo argument of calls with no halo work.
        self.halo_quiet: HaloArgument | None = None
        self.fixed_parameters: Mapping[str, frozenset[str]] = MappingProxyType({})
        self.missing_inputs: frozenset[str] = frozenset()
        self.progress = None
        self.model_state_entered = False

    def halo_layout(self, tensor: torch.Tensor):
        """Where an exchanged field's decomposed axis is: a field declared
        ``halo=True`` whose shape names the slab axis (cells or faces), or
        whose leading axis holds the partition key's owned and ghost rows."""

        from hydroforge.parallel.halo import CELL, FACE, KEY, FieldLayout

        found = self._halo_layouts.get(id(tensor))
        if found is not None and found[0] is tensor:
            return found[1]
        entry = next(
            (
                entry
                for entry in self.plan.fields.entries.values()
                if (module := self.modules.get(entry.module)) is not None
                and module.__dict__.get(entry.name) is tensor
            ),
            None,
        )
        if entry is None or not entry.tensor.halo:
            what = (
                "a tensor that is no module field" if entry is None else entry.qualified
            )
            raise ValueError(
                f"{what} is exchanged across ranks (a kernel's halo, publishes "
                "or contributes); declare the field halo=True"
            )
        info = self.plan.fields.slab.get(entry.qualified)
        halo = self.halo
        if info is not None and info.kind != "interior":
            axis = len(entry.tensor.shape) - 1 - info.position
            layout = FieldLayout(FACE if info.face else CELL, axis)
        elif (
            halo is not None
            and KEY in halo.plan.kinds
            and tensor.dim() >= 1
            and tensor.shape[0] == halo.plan.extent(KEY)
        ):
            layout = FieldLayout(KEY, tensor.dim() - 1)
        else:
            raise ValueError(
                f"{entry.qualified} is declared halo=True but neither names the "
                "slab axis nor holds the partition key's rows"
            )
        self._halo_layouts[id(tensor)] = (tensor, layout)
        return layout

    def spatial_group(self) -> Any:
        """The process group of this rank's spatial partitions: the mesh's
        spatial group, or WORLD when the spatial ranks are every rank.
        Halo and topology ranks are ranks of this group."""

        import torch.distributed as dist

        parallel = self.plan.parallel
        group = None if parallel is None else parallel.group("spatial")
        return dist.group.WORLD if group is None else group

    def halo_depth(self, layout: Any) -> int:
        """Ghost rows on each side of a slab field's rows; 0 for partition
        key fields (their ghosts are whole rows, read or not)."""

        from hydroforge.parallel.halo import KEY

        return 0 if layout.kind == KEY or self.slab is None else self.slab.halo

    def _build_halo(self) -> None:
        """Create the halo of the partition's ghost rows and the slab's
        (collective over the spatial group), its message slots sized for the
        widest rows of the fields declared ``halo=True``."""

        from hydroforge.declare.tensors import ModuleTensors
        from hydroforge.kernels.halo import attach
        from hydroforge.parallel.halo import (
            CELL,
            FACE,
            KEY,
            HaloPlan,
            HaloRuntime,
            partition_layouts,
        )
        from hydroforge.parallel.slab import slab_layouts

        plan, slab = self.plan, self.slab
        layouts = {}
        row_bytes = {}
        rows = self.partition.halo_rows()
        if rows is not None:
            layouts[KEY] = partition_layouts(*rows)
            extent = layouts[KEY][plan.spatial_rank].extent
            # Rows of the fields declared halo=True, as materialized or
            # declared (computed ones not yet materialized, at the widest
            # element).
            widths = []
            for entry in plan.fields.entries.values():
                module = self.modules.get(entry.module)
                if not entry.tensor.halo or module is None:
                    continue
                value = module.__dict__.get(entry.name)
                if isinstance(value, torch.Tensor) and value.dim() >= 1:
                    if value.shape[0] == extent:
                        widths.append(value.numel() // extent * value.element_size())
                try:
                    shape = ModuleTensors(module)._expected_shape(entry.name)
                except ValueError:
                    continue
                if shape and shape[0] == extent:
                    widths.append(8 * int(np.prod(shape[1:], dtype=np.int64)))
            row_bytes[KEY] = max(widths or [8])
        if slab is not None and slab.world > 1:
            positions = self.topology.positions()
            slabs = [
                type(slab).build(
                    rank,
                    slab.world,
                    slab.total,
                    slab.halo,
                    slab.periodic,
                    positions[rank],
                )
                for rank in range(slab.world)
            ]
            layouts[CELL] = slab_layouts(slabs, CELL)
            layouts[FACE] = slab_layouts(slabs, FACE)
            for qualified, info in plan.fields.slab.items():
                entry = plan.fields.entries[qualified]
                module = self.modules.get(entry.module)
                if module is None or info.kind == "interior" or not info.halo:
                    continue
                shape = ModuleTensors(module)._expected_shape(entry.name)
                if shape is None:
                    continue
                kind = FACE if info.face else CELL
                width = 8 * (
                    int(np.prod(shape, dtype=np.int64)) // max(shape[info.position], 1)
                )
                row_bytes[kind] = max(row_bytes.get(kind, 8), width)
        if not layouts:
            return
        self.halo = attach(
            HaloRuntime(
                HaloPlan(plan.spatial_rank, layouts),
                plan.device,
                plan.backend.name,
                self.spatial_group(),
                row_bytes,
            )
        )
        summary = self.halo.summary()
        emit(
            self,
            "info",
            "model.halo",
            "Ranks exchange ghost rows of the decomposed axis",
            rank=plan.rank,
            kinds=summary["kinds"],
            ghosts=summary["ghosts"],
            ghosts_total=summary["ghosts_total"],
            ghosts_across_nodes=summary["ghosts_across_nodes"],
            links=summary["links"],
            skipped=summary["skipped"],
            **({} if slab is None else {"slab_rows": (slab.start, slab.stop)}),
        )

    def _release(self) -> list[BaseException]:
        """Attempt every release of one materialization; return the failures.

        Captured programs may reference author-owned buffers, so execution
        closes before the model releases the state it initialized.  The input
        proxy's read handles close last; a later read reopens them.
        """

        releases = [
            resource.close
            for resource in (self.statistics, self.execution, self.halo)
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

        # Resolve rank agreement without constructing an InputBinding.
        from hydroforge.execution.signatures import field_selection_signature

        failures = self.channel.gather(
            None,
            phase="runtime.materialization.field_selection",
            signature=field_selection_signature(self),
        )
        if any(failure is not None for failure in failures):
            raise distributed_failure_error("distributed field selection", failures)
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
            fixed_derived_parameters,
        )
        from hydroforge.execution.partition import PartitionRuntime

        plan = self.plan
        owner = self.owner
        check_options_fingerprint(
            owner.input_proxy.attrs, plan.options, opened_modules=plan.modules
        )
        if self.input is None:
            self.input = InputBinding(plan, owner.input_proxy, self.event_sink)
        from hydroforge.parallel.topology import Topology, gather_topology

        # Placement prefers links within a node; every rank knows every node.
        self.topology = (
            gather_topology(self.spatial_group())
            if plan.spatial_world_size > 1
            else Topology.single(plan.spatial_world_size)
        )
        self.partition = PartitionRuntime(plan, self.input, self.topology)
        # The halo argument of calls with no halo work.
        self.halo_quiet = HaloArgument.quiet()
        from hydroforge.execution.slab import build_slab

        self.slab = build_slab(plan, self.input, self.topology)
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
                precision=precisions_of(plan).compute,
                storage_precision=precisions_of(plan).storage,
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
        balance = {}
        if plan.spatial_world_size > 1 and plan.spec.partition_key is not None:
            pieces = self.partition.pieces
            balance = {
                "split": pieces.split,
                "pieces": pieces.pieces,
                "imbalance": round(pieces.imbalance, 4),
                "cut": pieces.cut,
            }
        emit(
            self,
            "info",
            "model.partition",
            "Using partition root",
            key=plan.spec.partition_key,
            group=plan.spec.partition_group,
            **balance,
        )
        missing = {
            name: frozenset(
                field
                for field, spec in plan.spec.modules[name].tensor_fields.items()
                if not spec.computed
                and field in plan.fields.modules[name].active
                and field not in payloads[name]._input_values
            )
            for name in plan.module_order
        }
        self.missing_inputs = frozenset(
            f"{name}.{field}" for name, fields in missing.items() for field in fields
        )
        self._construct_modules(payloads, missing)
        self._build_halo()
        modules = tuple(self.modules[name] for name in plan.module_order)
        for module in modules:
            module._tensors.constructed = True
        materialize_declared(modules)
        self._check_field_values()
        # What derived parameters read is recorded before inputs are discarded.
        self.fixed_parameters = fixed_derived_parameters(self)
        for name in plan.modules:
            self.modules[name]._tensors._apply_modes()
        self.model_state_entered = True
        # Registered kernels called by the hook bind like inside a step,
        # with step fields describing the initial clock.
        execution.step_fields.prepare_initial(
            self.clock.current_time, self.clock.next_step_seconds
        )
        with model_call(execution):
            owner.initialize_model_state()
        evaluate_after_init(modules)
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
        seen: set[tuple[torch.device, int]] = set()
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
                    not isinstance(value, torch.Tensor)
                    or value.device.type != module.device.type
                ):
                    continue
                # Views share one storage; count each storage once in full.
                storage = value.untyped_storage()
                identity = (value.device, storage._cdata)
                if identity not in seen:
                    seen.add(identity)
                    size += storage.nbytes()
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

    def _construct_modules(
        self,
        payloads: Mapping[str, ModulePayload],
        missing: Mapping[str, frozenset[str]],
    ) -> None:
        specs = self.plan.spec.modules
        modules = self.modules
        key = self.plan.spec.partition_key
        count = 0 if key is None else self.partition.ghost_indices(key).size
        ghosts = MappingProxyType({key: count} if count else {})
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
                    missing=missing[name],
                    ghosts=ghosts,
                    slab=self.slab,
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

    def _check_field_values(self) -> None:
        """Check declared ``finite`` and bounds of stored fields in one pass.

        Runs on each rank's local values before tensor modes discard any
        construction-only field.
        """

        problems: list[str] = []
        for name in self.plan.module_order:
            problems.extend(self.modules[name]._tensors.constraint_violations())
        if problems:
            raise ValueError(
                "model input violates declared field constraints:\n  - "
                + "\n  - ".join(problems)
            )

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
        return self.run_structure_hooks()

    def reset_clock(self, time: DateLike | None) -> None:
        """Validate collectively before discarding the current timeline."""

        if managed_step_active():
            raise RuntimeError("reset_clock is allowed only between managed steps")
        plan = self.plan
        target = plan.initial_time
        error: BaseException | None = None
        signature = None
        try:
            if plan.schedule is not None and time is not None:
                raise ValueError("reset_clock with a schedule does not accept time")
            if plan.parameters:
                raise ValueError(
                    "reset_clock does not support scheduled parameter changes"
                )
            if (
                plan.output.declaration is not None
                and plan.output.config.sink != "memory"
            ):
                raise ValueError(
                    "reset_clock requires in-memory statistics or no statistics"
                )
            if time is not None:
                _, normalized, _ = normalize_calendar_dates(
                    {"reset time": time}, calendar=plan.calendar
                )
                target = normalized["reset time"]
                if plan.initial_time is None:
                    raise ValueError(
                        "reset_clock(time) requires a declared initial_time"
                    )
            signature = (self.state, canonical(target))
        except BaseException as caught:
            error = caught
        self.channel.preflight(
            error,
            phase="clock.reset.validation",
            scope="distributed clock reset validation",
            signature=signature,
        )
        error = None
        try:
            self.require_healthy(f"{type(self.owner).__name__}.reset_clock")
        except BaseException as caught:
            error = caught
        self.channel.preflight(
            error,
            phase="clock.reset.health",
            scope="distributed clock reset runtime health",
        )
        # Preserve the clock/progress owners cached by managed-step policies.
        error = None
        try:
            if self.statistics is not None:
                self.statistics.restart()
            self.clock.time = target
            self.clock.schedule_index = 0
            self.clock.elapsed_microseconds = 0
            self.progress.state = ProgressState()
        except BaseException as caught:
            error = caught
        try:
            failures = self.channel.gather(error, phase="clock.reset.commit")
        except BaseException as coordination_error:
            failure = (
                coordination_error
                if error is None
                else ResourceCleanupError(
                    "clock reset coordination", (error, coordination_error)
                )
            )
            self.execution.poison(failure, phase="clock reset")
            raise failure
        if any(failure is not None for failure in failures):
            failure = error or distributed_failure_error(
                "distributed clock reset", failures
            )
            self.execution.poison(failure, phase="clock reset")
            raise failure

    def run_structure_hooks(self) -> StructuralUpdateResult | None:
        """Run the structure pass of a runtime already proven healthy."""

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

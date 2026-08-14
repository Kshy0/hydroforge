"""Ordered, exception-safe model initialization pipeline."""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Any

from hydroforge.output.checkpoint import CheckpointRuntime
from hydroforge.compiler.data import ModelDataCompiler
from hydroforge.contracts.events import emit
from hydroforge.compiler.namespace import NamespaceCompiler
from hydroforge.compiler.partition import PartitionCompiler
from hydroforge.compiler.statistics_binding import StatisticsBindingCompiler
from hydroforge.compiler.model import FieldNamespaceCompiler
from hydroforge.execution.parameters import ParameterPlanRuntime
from hydroforge.execution.progress import ProgressRuntime
from hydroforge.execution.runtime import ModelExecution
from hydroforge.contracts.temporal import canonical_calendar
from hydroforge.contracts.runtime import (
    DEFAULT_BACKEND_REQUIREMENT, RUNTIME_BACKEND_REQUIREMENTS,
)
from hydroforge.contracts import ResourceCleanupError

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class ModelInitializer:
    """Execute the ordered, exception-safe model initialization pipeline."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self._execution: ModelExecution | None = None
        self._statistics: StatisticsBindingCompiler | None = None

    def run(self) -> None:
        model = self.model
        try:
            self._runtime_services()
            self._validate_runtime_config()
            self._validate_schema()
            module_data = model.shard_param()
            self._construct_modules(module_data)
            self._initialize_modules()
            self._precompile_backend()
            self._apply_tensor_modes()
            hook_result = model.initialize_model_state()
            if hook_result is not None:
                raise TypeError(
                    "initialize_model_state() must mutate registered model "
                    f"state and return None, got {type(hook_result).__name__}"
                )
            self._initialize_output()
            self._compile_field_namespace()
            self._compile_execution()
            self._seal_tensor_bindings()
            model.print_memory_summary()
            emit(model, "info", "model.initialized", "Model initialized")
        except BaseException as initialization_error:
            cleanup_failures: list[BaseException] = []
            if self._statistics is not None:
                try:
                    self._statistics.close()
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            if self._execution is not None:
                try:
                    self._execution.close()
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            if cleanup_failures:
                error = ResourceCleanupError(
                    "model after initialization failure",
                    (initialization_error, *cleanup_failures),
                )
                raise error from initialization_error
            raise

    def _runtime_services(self) -> None:
        model = self.model
        execution = ModelExecution(model)
        self._execution = execution
        model._execution = execution
        from hydroforge.execution.substeps import SubstepRuntime
        from hydroforge.execution.outer import OuterRuntime

        model._substeps = SubstepRuntime(model)
        model._outer = OuterRuntime(model)
        model._namespace = NamespaceCompiler(model)
        model._partition = PartitionCompiler(model)
        statistics = StatisticsBindingCompiler(model)
        self._statistics = statistics
        model._statistics = statistics
        model._checkpoint = CheckpointRuntime(model)
        model._data = ModelDataCompiler(model)
        model._parameters = ParameterPlanRuntime(model)
        model._progress_service = ProgressRuntime(model)
        emit(
            model, "info", "model.initializing", "Initializing model",
            rank=model.rank, modules=tuple(model.opened_modules),
        )

    def _validate_schema(self) -> None:
        model = self.model
        model.check_namespace_conflicts()
        emit(
            model, "info", "model.partition", "Using partition root",
            key=model.partition_key, group=model.partition_group,
        )

    def _validate_runtime_config(self) -> None:
        """Resolve backend-owned defaults and validate its declarative contract."""
        model = self.model
        runtime = model._execution
        backend = runtime.backend
        plan = model.statistics_plan
        schedule = model.simulation_schedule
        if plan is not None and schedule is None:
            raise ValueError("statistics_plan requires simulation_schedule")
        if schedule is not None:
            if model.calendar is not None:
                configured = canonical_calendar(model.calendar)
                if configured != schedule.calendar:
                    raise ValueError(
                        f"model calendar {configured!r} differs from simulation "
                        f"schedule calendar {schedule.calendar!r}"
                    )
            model.calendar = schedule.calendar
            if model.current_time is not None:
                raise ValueError(
                    "current_time must not be configured together with "
                    "simulation_schedule; the schedule initializes the "
                    "runtime clock"
                )
            model.current_time = schedule.execution_start
        else:
            model.calendar = canonical_calendar(model.calendar or "standard")
        if model.mixed_precision is None:
            model.mixed_precision = bool(
                model.device.type == "cuda" and backend in {"cuda", "triton"}
            )

        runtime_rule = RUNTIME_BACKEND_REQUIREMENTS.get(
            backend, DEFAULT_BACKEND_REQUIREMENT,
        )
        runtime_rule.validate_precision(
            model.precision, model.mixed_precision, backend=backend,
        )
        rule = model.backend_requirements.get(
            backend, DEFAULT_BACKEND_REQUIREMENT,
        )
        rule.validate_precision(
            model.precision, model.mixed_precision, backend=backend,
        )
        if model.BLOCK_SIZE is not None:
            rule.validate_block_size(model.BLOCK_SIZE, backend=backend)
        if not rule.trials and model.num_trials is not None:
            raise ValueError(
                f"backend {backend!r} does not support ensemble trials"
            )

    def _construct_modules(self, module_data: dict[str, Any]) -> None:
        model = self.model
        module_types = model.module_types()
        opened = frozenset(model.opened_modules)
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for name in model.opened_modules:
            references = module_types[name].get_module_reference_fields().values()
            sorter.add(
                name,
                *(reference.module_name for reference in references
                  if reference.module_name in opened),
            )
        try:
            construction_order = tuple(sorter.static_order())
        except CycleError as error:
            raise ValueError(
                "opened module references must form an acyclic construction "
                f"graph: {error.args[1]}"
            ) from error
        for name in construction_order:
            module_class = module_types[name]
            module = module_class.model_validate({
                **module_data,
                "opened_modules": model.opened_modules,
                "rank": model.rank,
                "device": model.device,
                "precision": model.dtype,
                "mixed_precision": model.mixed_precision,
                "num_trials": model.num_trials,
            }, context={
                "hydroforge_model_initialization": True,
                "hydroforge_module_references": model._modules,
            })
            module._bind_event_sink(model.event_sink)
            model._modules[name] = module

    def _initialize_modules(self) -> None:
        """Expose the complete model graph, then validate cross-module state."""

        model = self.model
        model._model_modules_bound = True
        for name in model.opened_modules:
            model._modules[name].validate_linked_state()

    def _apply_tensor_modes(self) -> None:
        model = self.model
        for name in model.opened_modules:
            model._modules[name]._tensors.apply_modes()

    def _precompile_backend(self) -> None:
        """Materialize only backend extensions reachable by opened modules."""
        model = self.model
        execution = self._execution
        if execution is None or getattr(execution, "backend", None) != "cuda":
            return
        catalogs = tuple(getattr(model, "cuda_extension_modules", ()))
        if not catalogs:
            return
        execution.precompile_cuda_catalogs(
            catalogs, model.opened_modules,
        )

    def _compile_field_namespace(self) -> None:
        namespace = FieldNamespaceCompiler(self.model).compile()
        self.model._field_namespace = namespace
        self.model._execution._refresh_model_tensor_index()

    def _seal_tensor_bindings(self) -> None:
        """Make compiled storage identities immutable without hot-path scans."""

        for module in self.model._modules.values():
            module._seal_declared_tensor_bindings()
        self.model._seal_model_configuration()

    def _initialize_output(self) -> None:
        model = self.model
        if model.variables_to_save:
            model._statistics.initialize(model.variables_to_save)

    def _compile_execution(self) -> None:
        from hydroforge.execution.step import compile_step_policies

        model = self.model
        compile_step_policies(model)

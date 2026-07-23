"""Ordered, exception-safe model initialization pipeline."""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Any

from hydroforge.output.checkpoint import CheckpointRuntime
from hydroforge.compiler.data import ModelDataCompiler
from hydroforge.contracts.events import emit
from hydroforge.compiler.namespace import NamespaceCompiler
from hydroforge.compiler.partition import PartitionCompiler
from hydroforge.compiler.statistics_binding import StatisticsBindingCompiler
from hydroforge.compiler.model import ModelCompiler
from hydroforge.execution.parameters import ParameterPlanRuntime
from hydroforge.execution.progress import ProgressRuntime
from hydroforge.execution.runtime import ModelExecution
from hydroforge.contracts.temporal import canonical_calendar, require_calendar
from hydroforge.contracts.runtime import (
    DEFAULT_BACKEND_REQUIREMENT, RUNTIME_BACKEND_REQUIREMENTS,
)
from hydroforge.contracts import ResourceCleanupError


class ModelInitializer:
    """Execute cold-path model setup in explicit dependency order."""

    def __init__(self, model: Any) -> None:
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
            self._bind_module_attributes()
            self._specialize_capabilities()
            self._precompile_backend()
            self._apply_tensor_modes()
            hook_result = model.initialize_model_state()
            if hook_result is not None:
                raise TypeError(
                    "initialize_model_state() must mutate registered model "
                    f"state and return None, got {type(hook_result).__name__}"
                )
            self._initialize_output()
            self._compile_execution()
            self._compile_model_plan()
            self._seal_tensor_bindings()
            model.print_memory_summary()
            emit(model, "info", "model.initialized", "All modules initialized")
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

        object.__setattr__(model, "substeps", SubstepRuntime(model))
        object.__setattr__(model, "outer", OuterRuntime(model))
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
        if plan is not None:
            conflicting = {
                name for name in (
                    "statistics_interval", "statistics_outer_interval",
                )
                if name in model.model_fields_set and getattr(model, name) is not None
            }
            if conflicting:
                raise ValueError(
                    "statistics_plan cannot be combined with legacy interval "
                    f"fields: {sorted(conflicting)}"
                )
            if schedule is not None and schedule != plan.schedule:
                raise ValueError(
                    "statistics_plan and simulation_schedule use different schedules"
                )
            schedule = plan.schedule
            model.simulation_schedule = schedule
        if schedule is not None:
            configured = canonical_calendar(model.calendar)
            if "calendar" in model.model_fields_set and configured != schedule.calendar:
                raise ValueError(
                    f"model calendar {configured!r} differs from simulation "
                    f"schedule calendar {schedule.calendar!r}"
                )
            model.calendar = schedule.calendar
            if model.output_start_time is not None:
                require_calendar(
                    model.output_start_time, schedule.calendar,
                    label="output_start_time",
                )
                if model.output_start_time >= schedule.end:
                    raise ValueError(
                        "output_start_time is outside the simulation schedule"
                    )
        if "mixed_precision" not in model.model_fields_set:
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
        rule.validate_block_size(model.BLOCK_SIZE, backend=backend)
        if not rule.trials and model.num_trials is not None:
            raise ValueError(
                f"backend {backend!r} does not support ensemble trials"
            )

    def _construct_modules(self, module_data: dict[str, Any]) -> None:
        model = self.model
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for name in model.opened_modules:
            module_class = model.module_list[name]
            sorter.add(name, *module_class.dependencies)
        for name in sorter.static_order():
            if name not in model.opened_modules:
                continue
            module_class = model.module_list[name]
            module = module_class(
                opened_modules=model.opened_modules,
                rank=model.rank,
                device=model.device,
                world_size=model.world_size,
                precision=model.dtype,
                mixed_precision=model.mixed_precision,
                num_trials=model.num_trials,
                **model._modules,
                **module_data,
            )
            module._event_sink = model.event_sink
            model._modules[name] = module
        missing = set(model.opened_modules).difference(model._modules)
        if missing:
            raise RuntimeError(
                "module construction did not produce opened modules: "
                + ", ".join(sorted(missing))
            )

    def _bind_module_attributes(self) -> None:
        """Install every module slot once for zero-reflection model access.

        Downstream orchestration uses ``self.base`` / ``self.reservoir``
        directly.  Open modules resolve to their instance and closed optional
        modules resolve to ``None``; model authors never need cached
        ``get_module`` forwarding properties.
        """

        model = self.model
        for name in model.module_list:
            object.__setattr__(model, name, model._modules.get(name))

    def _specialize_capabilities(self) -> None:
        model = self.model
        # Module membership must be available while composite feature rules run.
        model._capabilities = frozenset(model.opened_modules)
        capabilities = set(model.opened_modules)
        for name, rule in model.feature_rules.items():
            if bool(rule(model) if callable(rule) else rule):
                capabilities.add(name)
        model._capabilities = frozenset(capabilities)

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

    def _compile_model_plan(self) -> None:
        plan = ModelCompiler(self.model).compile()
        self.model._plan = plan
        self.model._execution.install_model_plan(plan)

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

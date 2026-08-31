"""Ordered, exception-safe model initialization pipeline."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from hydroforge.output.checkpoint import CheckpointRuntime
from hydroforge.compiler.data import ModelDataCompiler
from hydroforge.contracts.events import emit
from hydroforge.compiler.namespace import NamespaceCompiler
from hydroforge.compiler.partition import PartitionCompiler
from hydroforge.compiler.statistics_binding import (
    DisabledStatisticsBinding,
    StatisticsBindingCompiler,
)
from hydroforge.compiler.model import FieldNamespaceCompiler
from hydroforge.execution.parameters import ParameterPlanRuntime
from hydroforge.execution.progress import ProgressRuntime
from hydroforge.execution.runtime import ModelExecution
from hydroforge.contracts.errors import ResourceCleanupError

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class ModelInitializer:
    """Execute the ordered, exception-safe model initialization pipeline."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self._execution: ModelExecution | None = None
        self._statistics: (
            DisabledStatisticsBinding | StatisticsBindingCompiler | None
        ) = None

    def run(self) -> None:
        model = self.model
        try:
            self._runtime_services()
            module_data = model.shard_param()
            self._construct_modules(module_data)
            self._precompile_backend()
            self._apply_tensor_modes()
            model.initialize_model_state()
            self._compile_checkpoint()
            self._initialize_output()
            self._compile_field_namespace()
            self._construct_parameter_runtime()
            self._compile_execution()
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
        model._namespace = NamespaceCompiler(model)
        semantic_plan = model._semantic_plan
        model._partition = PartitionCompiler(
            model,
            schema=semantic_plan.partition_schema,
            variable_groups=semantic_plan.variable_groups,
        )
        model._data = ModelDataCompiler(model)
        model._progress_service = ProgressRuntime(model)
        emit(
            model,
            "info",
            "model.initializing",
            "Initializing model",
            rank=model.rank,
            modules=tuple(model.opened_modules),
        )
        emit(
            model,
            "info",
            "model.partition",
            "Using partition root",
            key=model.partition_key,
            group=model.partition_group,
        )

    def _construct_modules(self, module_data: dict[str, Any]) -> None:
        model = self.model
        module_types = model._module_types()
        for name in model._module_order:
            module_class = module_types[name]
            declared = module_class.model_fields
            payload = {
                field_name: value
                for field_name, value in module_data.items()
                if field_name in declared
            }
            payload.update(
                {
                    "opened_modules": model.opened_modules,
                    "rank": model.rank,
                    "device": model.device,
                    "precision": model.dtype,
                    "mixed_precision": model.mixed_precision,
                    "num_trials": model.num_trials,
                }
            )
            module = module_class.model_validate(
                payload,
                context={
                    "hydroforge_model_initialization": True,
                    "hydroforge_module_references": model._modules,
                    "hydroforge_module_event_sink": model.event_sink,
                    "hydroforge_module_reference_targets": (
                        model._semantic_plan.reference_targets[name]
                    ),
                    "hydroforge_model_initial_time": model.initial_time,
                    "hydroforge_model_simulation_schedule": (
                        model.simulation_schedule
                    ),
                    "hydroforge_trial_forcing_fields": (
                        model._semantic_plan.trial_forcing_fields.get(name, ())
                    ),
                    "hydroforge_field_demand_plan": (
                        model._semantic_plan.field_demand
                    ),
                },
            )
            model._modules[name] = module
        model._module_links = MappingProxyType(
            {
                name: model._modules.get(reference.module_name)
                for name, reference in model._module_reference_fields().items()
            }
        )

    def _apply_tensor_modes(self) -> None:
        model = self.model
        for name in model.opened_modules:
            model._modules[name]._tensors._apply_modes()

    def _precompile_backend(self) -> None:
        """Materialize only backend extensions reachable by opened modules."""
        model = self.model
        execution = model._execution
        if execution.backend != "cuda":
            return
        catalogs = tuple(getattr(model, "cuda_extension_modules", ()))
        if not catalogs:
            return
        execution.precompile_cuda_catalogs(
            catalogs,
            model.opened_modules,
        )

    def _compile_field_namespace(self) -> None:
        namespace = FieldNamespaceCompiler(self.model).compile()
        self.model._field_namespace = namespace
        self.model._execution._refresh_model_tensor_index()

    def _compile_checkpoint(self) -> None:
        self.model._checkpoint = CheckpointRuntime(self.model)

    def _construct_parameter_runtime(self) -> None:
        """Bind already compiled rank-local plans to materialized tensors."""

        self.model._parameters = ParameterPlanRuntime(
            self.model,
            self.model._semantic_plan.parameter_changes,
        )

    def _initialize_output(self) -> None:
        model = self.model
        declaration = model._semantic_plan.statistics
        statistics = (
            DisabledStatisticsBinding()
            if declaration is None
            else StatisticsBindingCompiler(model, declaration)
        )
        self._statistics = statistics
        model._statistics = statistics

    def _compile_execution(self) -> None:
        from hydroforge.execution.step import compile_step_policies

        model = self.model
        compile_step_policies(model)

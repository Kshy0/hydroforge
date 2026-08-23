"""Bind model fields and output requests to the statistics runtime."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, cast

import torch

from hydroforge.statistics.runtime import (
    StatisticsInstallation,
    StatisticsRuntime,
    StatisticsStaticBinding,
)
from hydroforge.statistics.observer import StatisticsObserver
from hydroforge.statistics.ir import (
    _StatisticsDeclaration, ExpressionSource, ScatterSource,
    StatisticsProgram, TensorSource,
)
from hydroforge.contracts.events import emit
from hydroforge.contracts.fields import RuntimeTensorMetadata, TensorMetadata
from hydroforge.contracts.runtime import (
    DEFAULT_BLOCK_SIZE,
    _effective_block_size,
)

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class DisabledStatisticsBinding:
    """Resource-free statistics binding for a model without a declaration."""

    def close(self) -> None:
        pass

    def memory_usage(self) -> int:
        return 0

    def time_index(self) -> int:
        return 0

    def reset_time_index(self) -> None:
        pass


class StatisticsBindingCompiler:
    """Compile and own one complete statistics runtime."""

    def __init__(
        self,
        model: AbstractModel,
        declaration: _StatisticsDeclaration,
    ) -> None:
        self.model = model
        adhoc = self.prepare_virtuals(declaration.program)
        installation = self._compile_installation(
            declaration.variable_ops,
            declaration.program,
            adhoc,
            declaration.static_names,
            declaration.netcdf_options,
        )
        self._aggregator = self._create(installation)
        model._execution.statistics = StatisticsObserver(
            model, self._aggregator,
        )

    @property
    def variable_map(self):
        return self.model._namespace.build()

    @property
    def aggregator(self) -> StatisticsRuntime:
        return self._aggregator

    def _create(
        self, installation: StatisticsInstallation,
    ) -> StatisticsRuntime:
        model = self.model
        aggregator = StatisticsRuntime(
            device=model.device,
            backend=model._execution.backend,
            installation=installation,
            execution=model._execution,
            base_dtype=model.dtype,
            mixed_precision=model.mixed_precision,
            output_dir=model.output_full_dir,
            rank=model.rank,
            world_size=model.world_size,
            num_workers=model.output_workers,
            output_split_by_year=model.output_split_by_year,
            num_trials=(1 if model.num_trials is None else model.num_trials),
            save_kernels=model.save_kernels,
            max_pending_steps=model.max_pending_steps,
            max_pending_output_bytes=model.max_pending_output_bytes,
            block_size=_effective_block_size(
                model.BLOCK_SIZE,
                backend=model._execution.backend,
                default=DEFAULT_BLOCK_SIZE,
            ),
            calendar=model.calendar,
            in_memory=model.in_memory_output,
            result_device=model.result_device,
            save_precision=torch.float32,
            output_netcdf_options=model.output_netcdf_options,
            event_sink=model.event_sink,
        )
        return aggregator

    def _compile_static(
        self, values: tuple[str, ...],
    ) -> tuple[StatisticsStaticBinding, ...]:
        model = self.model
        bindings: list[StatisticsStaticBinding] = []
        for name in values:
            entry = self.variable_map[name]
            tensor = getattr(entry.module, entry.field_name)
            field = entry.module._get_tensor_schema(entry.field_name)
            bound, tensors = model._partition.bind_output(field)
            coordinate = bound.output_coord
            if name == coordinate:
                continue
            output_index = tensors.get(bound.output_index)
            bindings.append(StatisticsStaticBinding(
                name=name,
                tensor=tensor,
                output_index=output_index,
                coordinate=coordinate,
            ))
        return tuple(bindings)

    def prepare_virtuals(
        self,
        program: StatisticsProgram,
    ) -> dict[str, Any]:
        """Construct virtual metadata from the validated model declaration."""
        adhoc: dict[str, Any] = {}
        for name, source in program.sources.items():
            if name in self.variable_map:
                continue
            dependencies = (
                source.expression.dependencies
                if isinstance(source, ExpressionSource)
                else source.value.dependencies
                if isinstance(source, ScatterSource)
                else (cast(TensorSource, source).name,)
            )
            expression = (
                source.expression.source
                if isinstance(source, ExpressionSource)
                else source.value.source
                if isinstance(source, ScatterSource)
                else source.name
            )
            tensor_shape, output, coordinate = self._field_metadata(
                dependencies[0],
            )[:3]
            adhoc[name] = RuntimeTensorMetadata(
                tensor=TensorMetadata.compile({
                    "tensor_shape": tensor_shape, "category": "virtual",
                    "expr": expression, "dim_coords": coordinate,
                    "output": output,
                }),
                description=f"Ad-hoc expression: {expression}",
            )
        return adhoc

    def expand_dependencies(
        self,
        variable_ops: Mapping[str, tuple[str, ...]],
        program: StatisticsProgram,
    ) -> list[str]:
        """Return selected fields and all typed virtual dependencies."""
        ordered = list(variable_ops)
        seen = set(ordered)
        cursor = 0
        while cursor < len(ordered):
            name = ordered[cursor]
            cursor += 1
            source = program.sources.get(name, TensorSource(name))
            if isinstance(source, TensorSource):
                continue
            dependencies = (
                (*source.value.dependencies, source.index)
                if isinstance(source, ScatterSource)
                else source.expression.dependencies
            )
            for dependency in dependencies:
                if dependency not in seen:
                    seen.add(dependency)
                    ordered.append(dependency)
        return ordered

    def _compile_installation(
        self,
        variable_ops: Mapping[str, tuple[str, ...]],
        program: StatisticsProgram,
        adhoc: Mapping[str, Any],
        static_names: tuple[str, ...],
        netcdf_options: Mapping[str, Mapping[str, Any]],
    ) -> StatisticsInstallation:
        model = self.model
        by_shape: dict[tuple[int, ...], list[str]] = {}
        tensors: dict[str, torch.Tensor] = {}
        fields: dict[str, RuntimeTensorMetadata] = {}
        pending_bindings: list[tuple[str, torch.Tensor, bool]] = []

        def install_tensor(
            name: str,
            tensor: torch.Tensor,
            info: RuntimeTensorMetadata | None,
            *,
            output_coordinate: bool = False,
            output_index: bool = False,
        ) -> None:
            if output_coordinate:
                installed = tensor.detach().to(torch.int64).clone(
                    memory_format=torch.contiguous_format,
                )
            elif output_index:
                installed = tensor.detach().clone(
                    memory_format=torch.contiguous_format,
                )
            else:
                installed = tensor
            tensors[name] = installed
            if info is not None:
                fields[name] = info
            by_shape.setdefault(tuple(installed.shape), []).append(name)

        for name in self.expand_dependencies(variable_ops, program):
            if name not in self.variable_map:
                info = adhoc[name]
                fields[name] = info
                continue

            entry = self.variable_map[name]
            tensor = getattr(entry.module, entry.field_name)
            field = entry.module._get_tensor_schema(entry.field_name)
            info, bindings = model._partition.bind_output(field)
            category = info.tensor.category
            if category == "virtual" and info.tensor.expression:
                fields[name] = info
            else:
                install_tensor(
                    name, tensor, info,
                )

            for binding_name in (info.output_index, info.output_coord):
                if not binding_name:
                    continue
                binding = bindings[binding_name]
                pending_bindings.append((
                    binding_name, binding,
                    binding_name == info.output_coord,
                ))

        for binding_name, binding, output_coordinate in pending_bindings:
            existing = tensors.get(binding_name)
            if existing is not None:
                continue
            install_tensor(
                binding_name, binding, None,
                output_coordinate=output_coordinate,
                output_index=not output_coordinate,
            )

        for shape, names in by_shape.items():
            emit(
                model, "info", "statistics.tensors_registered",
                "Registered tensors for streaming statistics",
                rank=model.rank, variables=tuple(names), shape=str(shape),
            )

        return StatisticsInstallation(
            variable_ops=MappingProxyType({
                name: tuple(operations)
                for name, operations in variable_ops.items()
            }),
            program=program,
            tensors=MappingProxyType(tensors),
            fields=MappingProxyType(fields),
            statics=self._compile_static(static_names),
            netcdf_options=netcdf_options,
        )

    def _field_metadata(
        self, name: str,
    ) -> tuple[tuple[str | int, ...], str, str | None]:
        entry = self.variable_map[name]
        field = entry.module._get_tensor_schema(entry.field_name)
        coordinate = field.tensor.dim_coords
        if coordinate:
            coordinate = coordinate.split(".")[-1]
        return field.tensor.shape, field.tensor.output, coordinate

    def close(self) -> None:
        self._aggregator._shutdown()

    def memory_usage(self) -> int:
        return self._aggregator.get_memory_usage()

    def results(self, *, stacked: bool) -> dict[str, torch.Tensor]:
        return self._aggregator.get_results(as_stacked=stacked)

    def result(
        self, variable: str, operation: str, *, stacked: bool,
    ) -> torch.Tensor:
        return self._aggregator.get_result(
            variable, operation, as_stacked=stacked,
        )

    def time_index(self) -> int:
        return self._aggregator.get_time_index()

    def reset_time_index(self) -> None:
        self._aggregator.reset_time_index()

    def accumulator(self, variable: str, operation: str) -> torch.Tensor:
        """Return an ownership-isolated differentiable accumulator snapshot."""

        key = f"{variable}_{operation}"
        accumulator = self._aggregator._storage[key]
        return accumulator.clone(memory_format=torch.preserve_format)

    def pop_result(
        self, variable: str, operation: str,
    ) -> torch.Tensor | None:
        """Remove and return the newest finalized in-memory result."""

        key = f"{variable}_{operation}"
        values = self._aggregator._result_tensors[key]
        return values.pop() if values else None

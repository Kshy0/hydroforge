"""Bind the compiled statistics program to materialized model tensors."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch

from hydroforge.compiler.partition import bare
from hydroforge.compiler.selection import resolve_block_size
from hydroforge.contracts.fields import RuntimeTensorMetadata, TensorMetadata
from hydroforge.core.arrays import find_indices_in_torch
from hydroforge.core.events import emit
from hydroforge.core.expr import ExpressionSource, ScatterSource, TensorSource
from hydroforge.declare.tensors import ModuleTensors
from hydroforge.statistics.ir import StatisticsDeclaration, StatisticsProgram
from hydroforge.statistics.runtime import (
    StatisticsInstallation,
    StatisticsRuntime,
    StatisticsStaticBinding,
)

if TYPE_CHECKING:
    from hydroforge.compiler.fields import FieldEntry
    from hydroforge.execution.session import ModelRuntime


def bind_output(
    runtime: ModelRuntime,
    field: FieldEntry,
    *,
    selections: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    | None = None,
) -> tuple[RuntimeTensorMetadata, dict[str, torch.Tensor]]:
    """Attach output coordinate and default-selection tensors to one field."""

    policy = field.tensor.output
    coordinate = None if policy == "disabled" else bare(field.tensor.dim_coords)
    index_name = None
    indices = None
    coordinate_tensor = None
    variable_map = runtime.namespace
    if policy != "disabled" and coordinate:
        coordinate_entry = variable_map[coordinate]
        coordinate_tensor = getattr(
            coordinate_entry.module,
            coordinate_entry.field_name,
        )
        selection = (
            runtime.plan.fields.partition.selections.get(coordinate)
            if policy == "auto"
            else None
        )
        if selection:
            selection_entry = variable_map[selection]
            selected = getattr(
                selection_entry.module,
                selection_entry.field_name,
            )
            if selected is not None:
                device = runtime.plan.device
                cached = None if selections is None else selections.get(selection)
                if cached is None:
                    indices = (
                        torch.empty(0, dtype=torch.int32, device=device)
                        if selected.numel() == 0
                        else find_indices_in_torch(selected, coordinate_tensor).to(
                            device
                        )
                    )
                    if selections is not None:
                        selections[selection] = (coordinate_tensor, selected, indices)
                else:
                    indices = cached[2]
                index_name = f"__selection_idx__{selection}"
                coordinate = selection
                coordinate_tensor = selected
    resolved_shape = None
    if field.expression_virtual:
        entry = variable_map[field.qualified]
        resolved_shape = ModuleTensors(entry.module)._expected_shape(entry.field_name)
    bound = RuntimeTensorMetadata(
        tensor=field.tensor,
        description=field.description,
        resolved_shape=resolved_shape,
        output_index=index_name,
        output_coord=coordinate,
    )
    tensors: dict[str, torch.Tensor] = {}
    if index_name is not None and indices is not None:
        tensors[index_name] = indices
    if coordinate and coordinate_tensor is not None:
        tensors[coordinate] = coordinate_tensor
    return bound, tensors


def bind_statistics(runtime: ModelRuntime) -> StatisticsRuntime:
    """Install the compiled statistics program and start its output writers."""

    return _StatisticsBinder(runtime).bind(runtime.plan.output.declaration)


class _StatisticsBinder:
    """Resolve model fields and construct one statistics runtime."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self.variable_map = runtime.namespace
        self.fields = runtime.plan.fields.names
        self.selections: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def bind(self, declaration: StatisticsDeclaration) -> StatisticsRuntime:
        adhoc = self.prepare_virtuals(declaration.program)
        installation = self._compile_installation(
            declaration.variable_ops,
            declaration.program,
            adhoc,
            declaration.static_names,
            declaration.netcdf_options,
        )
        runtime = self.runtime
        plan = runtime.plan
        config = plan.output.config
        backend = plan.backend
        return StatisticsRuntime(
            device=plan.device,
            backend=backend,
            installation=installation,
            windows=plan.output.windows,
            schedule=plan.schedule,
            on_write_failure=partial(
                runtime.execution.poison, phase="statistics background write"
            ),
            base_dtype=plan.dtype,
            mixed_precision=plan.mixed_precision,
            output_dir=plan.output.directory,
            rank=plan.spatial_rank,
            world_size=plan.spatial_world_size,
            num_workers=config.workers,
            output_split_by_year=config.split_by_year,
            ensemble_size=(
                1 if plan.local_ensemble_size is None else plan.local_ensemble_size
            ),
            ensemble_member_ids=plan.member_ids,
            max_pending_steps=config.max_pending_steps,
            block_size=resolve_block_size(
                backend, plan.backend_requirement, plan.block_size
            ),
            calendar=plan.calendar,
            in_memory=config.sink == "memory",
            result_device=config.result_device,
            save_precision=plan.output.save_dtype,
            event_sink=runtime.event_sink,
        )

    def _compile_static(
        self,
        values: tuple[str, ...],
    ) -> tuple[StatisticsStaticBinding, ...]:
        bindings: list[StatisticsStaticBinding] = []
        for name in values:
            entry = self.variable_map[name]
            tensor = getattr(entry.module, entry.field_name)
            bound, tensors = bind_output(
                self.runtime, self.fields[name], selections=self.selections
            )
            coordinate = bound.output_coord
            if name == coordinate:
                continue
            output_index = tensors.get(bound.output_index)
            bindings.append(
                StatisticsStaticBinding(
                    name=name,
                    tensor=tensor,
                    output_index=output_index,
                    coordinate=coordinate,
                )
            )
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
            references = tuple(self._field_metadata(item) for item in dependencies)
            reference = next(
                (item for item in references if item.tensor.dtype != "bool"),
                references[0],
            )
            coordinate = reference.tensor.dim_coords
            if coordinate:
                coordinate = coordinate.rsplit(".", 1)[-1]
            adhoc[name] = RuntimeTensorMetadata(
                tensor=TensorMetadata(
                    shape=reference.tensor.shape,
                    category="virtual",
                    expression=expression,
                    dim_coords=coordinate,
                    output=reference.tensor.output,
                    dtype=reference.tensor.dtype,
                ),
                description=f"Ad-hoc expression: {expression}",
                output_index=reference.output_index,
                output_coord=reference.output_coord,
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
            source = program.sources.get(name)
            if source is None or isinstance(source, TensorSource):
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
        by_shape: dict[tuple[int, ...], list[str]] = {}
        tensors: dict[str, torch.Tensor] = {}
        fields: dict[str, RuntimeTensorMetadata] = {}
        scatter_extents: dict[str, int] = {}
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
                installed = tensor.detach().to(
                    torch.int64, copy=True, memory_format=torch.contiguous_format
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
            info, bindings = bind_output(
                self.runtime, self.fields[name], selections=self.selections
            )
            category = info.tensor.category
            if category == "virtual" and info.tensor.expression:
                fields[name] = info
                if isinstance(program.sources.get(name), ScatterSource):
                    # Scatter buffers span the declared target domain.
                    scatter_extents[name] = cast(tuple[int, ...], info.resolved_shape)[
                        0
                    ]
            else:
                install_tensor(
                    name,
                    getattr(entry.module, entry.field_name),
                    info,
                )

            for binding_name in (info.output_index, info.output_coord):
                if not binding_name:
                    continue
                binding = bindings[binding_name]
                pending_bindings.append(
                    (
                        binding_name,
                        binding,
                        binding_name == info.output_coord,
                    )
                )

        for binding_name, binding, output_coordinate in pending_bindings:
            existing = tensors.get(binding_name)
            if existing is not None:
                continue
            install_tensor(
                binding_name,
                binding,
                None,
                output_coordinate=output_coordinate,
                output_index=not output_coordinate,
            )

        for shape, names in by_shape.items():
            emit(
                self.runtime,
                "info",
                "statistics.tensors_registered",
                "Registered tensors for streaming statistics",
                rank=self.runtime.plan.rank,
                variables=tuple(names),
                shape=str(shape),
            )

        statics = self._compile_static(static_names)
        return StatisticsInstallation(
            variable_ops=MappingProxyType(
                {name: tuple(operations) for name, operations in variable_ops.items()}
            ),
            program=program,
            tensors=MappingProxyType(tensors),
            fields=MappingProxyType(fields),
            scatter_extents=MappingProxyType(scatter_extents),
            statics=statics,
            netcdf_options=netcdf_options,
            selection_sources=tuple(
                (coordinate, selected, tensors.get(f"__selection_idx__{name}", index))
                for name, (coordinate, selected, index) in self.selections.items()
            ),
        )

    def _field_metadata(
        self,
        name: str,
    ) -> RuntimeTensorMetadata:
        metadata, _bindings = bind_output(
            self.runtime, self.fields[name], selections=self.selections
        )
        return metadata

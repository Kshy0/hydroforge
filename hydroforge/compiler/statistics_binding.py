"""Bind model fields and output requests to the statistics runtime."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch

from hydroforge.contracts.events import emit
from hydroforge.contracts.fields import RuntimeTensorMetadata, TensorMetadata
from hydroforge.contracts.runtime import (
    DEFAULT_BLOCK_SIZE,
    _effective_block_size,
)
from hydroforge.statistics.ir import (
    ExpressionSource,
    ScatterSource,
    StatisticsProgram,
    TensorSource,
    _StatisticsDeclaration,
)
from hydroforge.statistics.runtime import (
    StatisticsInstallation,
    StatisticsRuntime,
    StatisticsStaticBinding,
)

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class StatisticsBindingCompiler:
    """Resolve model fields and construct one statistics runtime."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self.variable_map = model._namespace.build()

    def compile(self, declaration: _StatisticsDeclaration) -> StatisticsRuntime:
        adhoc = self.prepare_virtuals(declaration.program)
        installation = self._compile_installation(
            declaration.variable_ops,
            declaration.program,
            adhoc,
            declaration.static_names,
            declaration.netcdf_options,
        )
        model = self.model
        return StatisticsRuntime(
            device=model.device,
            backend=model._execution.backend,
            installation=installation,
            execution=model._execution,
            base_dtype=model.dtype,
            mixed_precision=model.mixed_precision,
            output_dir=model.output_full_dir,
            rank=model.spatial_rank,
            world_size=model.spatial_world_size,
            num_workers=model.output_workers,
            output_split_by_year=model.output_split_by_year,
            ensemble_size=(
                1 if model.local_ensemble_size is None else model.local_ensemble_size
            ),
            ensemble_member_ids=(
                None if model.parallel is None else model.parallel.member_ids
            ),
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
            save_precision=(
                None
                if model.statistics_save_precision is None
                else {
                    "float32": torch.float32,
                    "float64": torch.float64,
                }[model.statistics_save_precision]
            ),
            event_sink=model.event_sink,
        )

    def _compile_static(
        self,
        values: tuple[str, ...],
    ) -> tuple[StatisticsStaticBinding, ...]:
        model = self.model
        bindings: list[StatisticsStaticBinding] = []
        for name in values:
            entry = self.variable_map[name]
            tensor = getattr(entry.module, entry.field_name)
            field = entry.module._get_tensor_schema(
                entry.field_name,
                opened_modules=model.opened_modules,
                field_demand=model._field_demand,
            )
            bound, tensors = model._partition.bind_output(field)
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
                tensor=TensorMetadata.compile(
                    {
                        "tensor_shape": reference.tensor.shape,
                        "category": "virtual",
                        "expr": expression,
                        "dim_coords": coordinate,
                        "output": reference.tensor.output,
                        "tensor_dtype": reference.tensor.dtype,
                    }
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
        model = self.model
        by_shape: dict[tuple[int, ...], list[str]] = {}
        tensors: dict[str, torch.Tensor] = {}
        fields: dict[str, RuntimeTensorMetadata] = {}
        pending_bindings: list[tuple[str, torch.Tensor, bool]] = []
        source_copies: list[tuple[str, torch.Tensor, torch.Tensor]] = []

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
            if output_index:
                source_copies.append((name, tensor, installed))
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
            field = entry.module._get_tensor_schema(
                entry.field_name,
                opened_modules=model.opened_modules,
                field_demand=model._field_demand,
            )
            info, bindings = model._partition.bind_output(field)
            category = info.tensor.category
            if category == "virtual" and info.tensor.expression:
                fields[name] = info
            else:
                install_tensor(
                    name,
                    tensor,
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
                model,
                "info",
                "statistics.tensors_registered",
                "Registered tensors for streaming statistics",
                rank=model.rank,
                variables=tuple(names),
                shape=str(shape),
            )

        return StatisticsInstallation(
            variable_ops=MappingProxyType(
                {name: tuple(operations) for name, operations in variable_ops.items()}
            ),
            program=program,
            tensors=MappingProxyType(tensors),
            fields=MappingProxyType(fields),
            statics=self._compile_static(static_names),
            netcdf_options=netcdf_options,
            source_copies=tuple(source_copies),
        )

    def _field_metadata(
        self,
        name: str,
    ) -> RuntimeTensorMetadata:
        entry = self.variable_map[name]
        field = entry.module._get_tensor_schema(
            entry.field_name,
            opened_modules=self.model.opened_modules,
            field_demand=self.model._field_demand,
        )
        metadata, _bindings = self.model._partition.bind_output(field)
        return metadata

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Bind the compiled statistics program to materialized model tensors."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from hydroforge.compiler.partition import bare
from hydroforge.compiler.selection import resolve_block_size
from hydroforge.contracts.fields import (
    OutputAxis,
    RuntimeTensorMetadata,
    TensorMetadata,
)
from hydroforge.contracts.views import OutputUnion, OutputView, view_indices
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


def _regular_step(indices: tuple[int, ...]) -> int | None:
    """The common step of evenly spaced increasing indices, else ``None``."""

    if len(indices) == 1:
        return 1
    step = indices[1] - indices[0]
    if any(b - a != step for a, b in zip(indices, indices[1:])):
        return None
    return step


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
        installation = self._compile_installation(declaration, adhoc)
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
            precisions=plan.precisions,
            status_ticket=runtime.execution.status.ticket,
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
            save_kernels=config.save_kernels,
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
            if output_index is None and tensor is not None and tensor.ndim:
                tensor = tensor[: self._owned_rows(coordinate, tensor.shape[0])]
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
        self, declaration: StatisticsDeclaration, adhoc: Mapping[str, Any]
    ) -> StatisticsInstallation:
        variable_ops = declaration.variable_ops
        program = declaration.program
        by_shape: dict[tuple[int, ...], list[str]] = {}
        tensors: dict[str, torch.Tensor] = {}
        fields: dict[str, RuntimeTensorMetadata] = {}
        scatter_extents: dict[str, int] = {}
        pending_bindings: list[tuple[str, torch.Tensor, bool]] = []
        # Output coordinates that are selections (their IDs never change).
        selected_ids: set[str] = set()

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

            if info.output_index is not None:
                selected_ids.add(info.output_coord)
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

        coordinate_sources: dict[str, torch.Tensor] = {}
        for binding_name, binding, output_coordinate in pending_bindings:
            existing = tensors.get(binding_name)
            if existing is not None:
                continue
            if output_coordinate and binding_name not in selected_ids:
                coordinate_sources[binding_name] = binding
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

        for name in variable_ops:
            fields[name] = self._bind_axes(name, declaration, program, tensors, fields)
        statics = self._compile_static(declaration.static_names)
        return StatisticsInstallation(
            variable_ops=MappingProxyType(
                {name: tuple(operations) for name, operations in variable_ops.items()}
            ),
            program=program,
            tensors=MappingProxyType(tensors),
            fields=MappingProxyType(fields),
            scatter_extents=MappingProxyType(scatter_extents),
            statics=statics,
            netcdf_options=declaration.netcdf_options,
            selection_sources=tuple(
                (coordinate, selected, tensors.get(f"__selection_idx__{name}", index))
                for name, (coordinate, selected, index) in self.selections.items()
            ),
            coordinate_sources=tuple(coordinate_sources.items()),
        )

    def _slab_axis(self, tensor: TensorMetadata) -> tuple[int, str] | None:
        """Position and kind of the slab axis in ``tensor``'s shape."""

        slab = self.runtime.slab
        if slab is None:
            return None
        from hydroforge.compiler.slab import slab_tokens

        plan = self.runtime.plan
        tokens = slab_tokens(plan.spec.slab, plan.spec.grid_axes)
        for position, token in enumerate(tensor.shape):
            if isinstance(token, str) and bare(token) in tokens:
                return position, tokens[bare(token)]
        return None

    def _slab_output_axis(
        self,
        name: str,
        axis: str,
        indices: tuple[int, ...] | None,
        kind: str,
        tensors: dict[str, torch.Tensor],
    ) -> OutputAxis:
        """The owned rows of the slab axis, selected by the view's global rows.

        A rank writes only the rows it owns (never ghost rows); the output
        coordinate holds their global indices, so ranks' files join along the
        axis.
        """

        slab = self.runtime.slab
        owned = slab.global_rows(kind)
        stored = slab.owned_slice(kind).start
        if indices is not None:
            owned = owned[np.isin(owned, np.asarray(indices, dtype=np.int64))]
        local = stored + (owned - slab.global_rows(kind)[0] if owned.size else owned)
        values = tuple(int(row) for row in owned)
        if local.size == 0:
            return OutputAxis(name=axis, extent=0, start=0, step=1, indices=())
        step = _regular_step(tuple(int(row) for row in local))
        if step is not None:
            return OutputAxis(
                name=axis,
                extent=int(local.size),
                start=int(local[0]),
                step=step,
                indices=values,
            )
        table = f"__view__{name}__slab"
        tensors[table] = torch.as_tensor(
            local, dtype=torch.int64, device=self.runtime.plan.device
        )
        return OutputAxis(
            name=axis, extent=int(local.size), table=table, indices=values
        )

    def _located_axis(self, tensor: TensorMetadata) -> tuple[int, Any] | None:
        """Position of a located entity dimension in ``tensor``'s shape and
        this rank's entities of it (``None``: none is cut by rank)."""

        located = self.runtime.located
        for position, token in enumerate(tensor.shape):
            if isinstance(token, str) and bare(token) in located:
                return position, located[bare(token)]
        return None

    def _located_output_axis(
        self,
        name: str,
        axis: str,
        indices: tuple[int, ...] | None,
        located: Any,
        tensors: dict[str, torch.Tensor],
    ) -> OutputAxis:
        """This rank's entities, selected by the view's global positions.

        The output coordinate holds each entity's place in the complete
        table, so ranks' files join along the axis.
        """

        places = located.positions
        local = np.arange(places.size, dtype=np.int64)
        if indices is not None:
            chosen = np.isin(places, np.asarray(indices, dtype=np.int64))
            places, local = places[chosen], local[chosen]
        values = tuple(int(place) for place in places)
        if local.size == 0:
            return OutputAxis(name=axis, extent=0, start=0, step=1, indices=())
        step = _regular_step(tuple(int(row) for row in local))
        if step is not None:
            return OutputAxis(
                name=axis,
                extent=int(local.size),
                start=int(local[0]),
                step=step,
                indices=values,
            )
        table = f"__view__{name}__located"
        tensors[table] = torch.as_tensor(
            local, dtype=torch.int64, device=self.runtime.plan.device
        )
        return OutputAxis(
            name=axis, extent=int(local.size), table=table, indices=values
        )

    def _view_points(
        self,
        name: str,
        view: OutputView,
        names: tuple[str, ...],
        extents: tuple[int, ...],
        slab_axis: tuple[int, str] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """This rank's paired points of ``view`` (global indices, one column
        per point) and their positions in the view's point list."""

        points = np.asarray([view.points[axis] for axis in names], dtype=np.int64)
        if any(np.any(points[i] >= extent) for i, extent in enumerate(extents)):
            raise ValueError(f"point view for {name!r} exceeds grid extents {extents}")
        positions = np.arange(points.shape[1], dtype=np.int64)
        if slab_axis is not None:
            position, kind = slab_axis
            keep = np.isin(points[position], self.runtime.slab.global_rows(kind))
            points, positions = points[:, keep], positions[keep]
        return points, positions

    def _union_cells(
        self,
        view: OutputUnion,
        names: tuple[str, ...],
        extents: tuple[int, ...],
        slab_axis: tuple[int, str] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """This rank's distinct cells of ``view`` (global indices, one column
        per cell) in row-major order, and their flat row-major indices on the
        global grid.

        Each part selects only the rows of the slab axis this rank owns, so
        no rank enumerates another's cells; a cell has one owner, so ranks'
        cells are distinct across ranks too.
        """

        owned = None
        if slab_axis is not None:
            owned = self.runtime.slab.global_rows(slab_axis[1])
        grid_axes = self.runtime.plan.spec.grid_axes
        found = [np.empty((0, len(names)), dtype=np.int64)]
        for part in view.parts:
            if part.points:
                points = {
                    axis: np.asarray(indices, dtype=np.int64)
                    for axis, indices in part.points.items()
                }
                for axis, values in points.items():
                    extent = extents[names.index(axis)]
                    if int(values.max()) >= extent:
                        raise ValueError(
                            f"point index {int(values.max())} exceeds the axis "
                            f"{axis!r} extent {extent}"
                        )
                keep = np.ones(len(next(iter(points.values()))), dtype=bool)
                if owned is not None and names[slab_axis[0]] in points:
                    keep = np.isin(points[names[slab_axis[0]]], owned)
                # Each point, repeated over the axes it leaves complete.
                ranges = [np.flatnonzero(keep)]
                free = [i for i, axis in enumerate(names) if axis not in points]
                for i in free:
                    ranges.append(
                        owned
                        if owned is not None and i == slab_axis[0]
                        else np.arange(extents[i], dtype=np.int64)
                    )
                grid = [item.reshape(-1) for item in np.meshgrid(*ranges, indexing="ij")]
                cells = np.empty((grid[0].size, len(names)), dtype=np.int64)
                for i, axis in enumerate(names):
                    cells[:, i] = (
                        points[axis][grid[0]]
                        if axis in points
                        else grid[1 + free.index(i)]
                    )
            else:
                selected = view_indices(names, extents, part, grid_axes)
                ranges = [
                    np.arange(extent, dtype=np.int64)
                    if indices is None
                    else np.asarray(indices, dtype=np.int64)
                    for indices, extent in zip(selected, extents, strict=True)
                ]
                if owned is not None:
                    position = slab_axis[0]
                    ranges[position] = np.intersect1d(ranges[position], owned)
                grid = np.meshgrid(*ranges, indexing="ij")
                cells = np.stack([item.reshape(-1) for item in grid], axis=1)
            found.append(cells)
        cells = np.concatenate(found)
        numbers = np.unique(np.ravel_multi_index(tuple(cells.T), extents))
        return np.stack(np.unravel_index(numbers, extents)).astype(np.int64), numbers

    def _owned_rows(self, coordinate: str | None, rows: int) -> int:
        """Rows of ``coordinate`` this rank owns: halo ghost rows follow them."""

        partition = self.runtime.partition
        if coordinate is None or coordinate != partition.partition_key:
            return rows
        return rows - partition.ghost_indices(coordinate).size

    def _logical_shape(
        self,
        name: str,
        program: StatisticsProgram,
        tensors: Mapping[str, torch.Tensor],
        fields: Mapping[str, RuntimeTensorMetadata],
    ) -> tuple[int, ...]:
        info = fields[name]
        if info.resolved_shape is not None:
            return tuple(info.resolved_shape)
        tensor = tensors.get(name)
        if tensor is not None:
            return tuple(tensor.shape[tensor.ndim - len(info.tensor.shape) :])
        source = program.sources[name]
        dependencies = (
            source.value.dependencies
            if isinstance(source, ScatterSource)
            else cast(ExpressionSource, source).expression.dependencies
        )
        return self._logical_shape(dependencies[0], program, tensors, fields)

    def _bind_axes(
        self,
        name: str,
        declaration: StatisticsDeclaration,
        program: StatisticsProgram,
        tensors: dict[str, torch.Tensor],
        fields: Mapping[str, RuntimeTensorMetadata],
    ) -> RuntimeTensorMetadata:
        """Resolve one output variable's axes on this rank.

        The view's global indices select local source rows; a partition-key
        coordinate without a selection keeps the owned rows only, so halo
        ghost rows are never written.
        """

        info = fields[name]
        shape = self._logical_shape(name, program, tensors, fields)
        names = declaration.axes[name]
        slab_axis = self._slab_axis(info.tensor)
        coordinate = bare(info.tensor.dim_coords)
        located_axis = self._located_axis(info.tensor)
        if located_axis is not None and (
            (located_axis[0] == 0 and (coordinate or info.output_index))
            or (slab_axis is not None and slab_axis[0] == located_axis[0])
        ):
            # Rows identified by coordinate IDs join ranks by those IDs.
            located_axis = None
        # Views select by global row or entity, so these axes have their
        # global extents.
        extents = list(shape)
        if slab_axis is not None:
            extents[slab_axis[0]] = self.runtime.slab.total_of(slab_axis[1])
        if located_axis is not None:
            extents[located_axis[0]] = located_axis[1].total
        view = declaration.views.get(name)
        union = isinstance(view, OutputUnion)
        if view is not None and (union or view.points):
            if located_axis is not None:
                raise ValueError("point views require grid axes, not located entity axes")
            points, numbers = (
                self._union_cells(view, names, tuple(extents), slab_axis)
                if union
                else self._view_points(name, view, names, tuple(extents), slab_axis)
            )
            # Where each written point lies, by global index of every axis.
            coordinates = tuple(
                (axis, tuple(int(value) for value in points[i]))
                for i, axis in enumerate(names)
            )
            if slab_axis is not None and numbers.size:
                position, kind = slab_axis
                slab = self.runtime.slab
                points[position] += (
                    slab.owned_slice(kind).start - slab.global_rows(kind)[0]
                )
            flat = (
                np.ravel_multi_index(tuple(points), shape)
                if numbers.size
                else np.empty(0, dtype=np.int64)
            )
            table = f"__view__{name}__points"
            tensors[table] = torch.as_tensor(
                flat, dtype=torch.int64, device=self.runtime.plan.device
            )
            return RuntimeTensorMetadata(
                tensor=info.tensor,
                description=info.description,
                output_index=info.output_index,
                output_coord=info.output_coord,
                resolved_shape=info.resolved_shape,
                output_axes=(
                    OutputAxis(
                        name=view.point_dimension,
                        extent=int(numbers.size),
                        indices=tuple(int(number) for number in numbers),
                        coordinates=coordinates,
                    ),
                ),
                output_gather=table,
            )
        selected = view_indices(
            names,
            tuple(extents),
            declaration.views.get(name),
            self.runtime.plan.spec.grid_axes,
        )
        device = self.runtime.plan.device
        axes = []
        for position, (axis, extent, indices) in enumerate(
            zip(names, shape, selected, strict=True)
        ):
            if slab_axis is not None and position == slab_axis[0]:
                axes.append(
                    self._slab_output_axis(name, axis, indices, slab_axis[1], tensors)
                )
            elif located_axis is not None and position == located_axis[0]:
                axes.append(
                    self._located_output_axis(
                        name, axis, indices, located_axis[1], tensors
                    )
                )
            elif position == 0 and info.output_index is not None:
                count = tensors[info.output_index].numel()
                axes.append(
                    OutputAxis(name=axis, extent=count, table=info.output_index)
                )
            elif position == 0 and coordinate:
                owned = self._owned_rows(coordinate, extent)
                axes.append(OutputAxis(name=axis, extent=owned))
                if owned != extent and info.output_coord in tensors:
                    tensors[info.output_coord] = tensors[info.output_coord][:owned]
            elif indices is None:
                axes.append(OutputAxis(name=axis, extent=extent))
            elif (step := _regular_step(indices)) is not None:
                # A regular range addresses its rows without a table.
                axes.append(
                    OutputAxis(
                        name=axis,
                        extent=len(indices),
                        start=indices[0],
                        step=step,
                        indices=indices,
                    )
                )
            else:
                table = f"__view__{name}__{position}"
                tensors[table] = torch.tensor(indices, dtype=torch.int64, device=device)
                axes.append(
                    OutputAxis(
                        name=axis, extent=len(indices), table=table, indices=indices
                    )
                )
        return RuntimeTensorMetadata(
            tensor=info.tensor,
            description=info.description,
            output_index=info.output_index,
            output_coord=info.output_coord,
            resolved_shape=info.resolved_shape,
            output_axes=tuple(axes),
        )

    def _field_metadata(
        self,
        name: str,
    ) -> RuntimeTensorMetadata:
        metadata, _bindings = bind_output(
            self.runtime, self.fields[name], selections=self.selections
        )
        return metadata

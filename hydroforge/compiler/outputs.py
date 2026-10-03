"""Output stage: statistics declaration, windows and the output directory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from hydroforge.compiler.fields import (
    FieldEntry,
    FieldNameResolver,
    FieldPlan,
    source_dependencies,
)
from hydroforge.compiler.partition import coordinate_identity
from hydroforge.compiler.selection import Selection
from hydroforge.contracts.fields import concrete_tensor_dtype
from hydroforge.contracts.windows import EveryStep, StatisticsOutput, StatisticsPlan
from hydroforge.core.arrays import torch_to_numpy_dtype
from hydroforge.core.expr import (
    ExpressionSource,
    Reduction,
    ScatterSource,
    TensorSource,
    ValueSource,
    parse_operation,
    parse_value_source,
    validate_expression_constants,
)
from hydroforge.core.naming import sanitize_symbol
from hydroforge.io.netcdf.encoding import netcdf_dtype_encoding
from hydroforge.io.netcdf.options import prepare_netcdf_variable_options
from hydroforge.statistics.ir import StatisticsDeclaration, StatisticsProgram
from hydroforge.statistics.phases import CONTROL_STATE
from hydroforge.statistics.storage import StoragePlan, variable_slots
from hydroforge.statistics.windows import (
    bind_statistics_plan_schedule,
    validate_statistics_window_schedule,
)

if TYPE_CHECKING:
    from hydroforge.declare.model import ModelDeclaration, OutputConfig

_SAVE_DTYPES = MappingProxyType({"float32": torch.float32, "float64": torch.float64})


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Resolved output of one model: directory, windows and statistics program.

    ``directory`` is ``None`` exactly when the model writes no file.
    ``windows`` is bound to the simulation schedule and is ``None`` exactly
    when the model declares no statistics.
    """

    config: OutputConfig
    directory: Path | None
    windows: StatisticsPlan | None
    outputs: tuple[StatisticsOutput, ...]
    declaration: StatisticsDeclaration | None
    save_dtype: torch.dtype | None


class StatisticsDeclarationCompiler:
    """Compile fields, expressions, operations and output storage without I/O."""

    def __init__(
        self,
        selection: Selection,
        fields: FieldPlan,
        config: OutputConfig,
    ) -> None:
        self.selection = selection
        self.field_plan = fields
        self.config = config
        self._parsed_sources: dict[str, ValueSource] = {}

    def parse_source(self, expression: str) -> ValueSource:
        source = self._parsed_sources.get(expression)
        if source is None:
            source = parse_value_source(expression, self.known)
            self._parsed_sources[expression] = source
        return source

    def active_declared_field(self, name: str) -> bool:
        field = self.fields.get(name)
        return field is not None and field.active

    def metadata(self, name: str) -> tuple[Any, ...]:
        tensor = self.fields[name].tensor
        if self.selection.metal_emulation != "native" and tensor.dtype == "hpfloat":
            raise ValueError(
                "statistics of emulated hpfloat fields are not implemented; CPU checkpoint export remains available"
            )
        coordinate = tensor.dim_coords
        if coordinate:
            coordinate = coordinate.split(".")[-1]
        return (
            tuple(
                dimension.rsplit(".", 1)[-1]
                if isinstance(dimension, str)
                else dimension
                for dimension in tensor.shape
            ),
            tensor.output,
            coordinate,
            tensor.dtype,
            tensor.category,
        )

    def schema_identity(self, name: str) -> tuple[tuple[Any, ...], str | None]:
        entry = self.fields[name]
        return (
            tuple(
                token.rsplit(".", 1)[-1] if isinstance(token, str) else token
                for token in entry.tensor.shape
            ),
            None
            if entry.tensor.dim_coords is None
            else coordinate_identity(
                entry.tensor.dim_coords, self.field_plan.entries.values()
            ),
        )

    def validate_expression(
        self,
        *,
        name: str,
        expression: str,
        target_metadata: tuple[Any, ...] | None,
        allow_scatter: bool,
    ) -> tuple[Any, ...]:
        source = self.parse_source(expression)
        if isinstance(source, ScatterSource) and (not allow_scatter):
            raise ValueError(
                "ad-hoc scatter statistics must be declared as a computed tensor field"
            )
        names = (
            source.value.dependencies
            if isinstance(source, ScatterSource)
            else source_dependencies(source)
        )
        if not names:
            raise ValueError(f"statistics expression {name!r} has no field dependency")
        forcing_dependencies = tuple(
            dependency
            for dependency in names
            if self.metadata(dependency)[4] == "forcing"
        )
        if forcing_dependencies:
            raise ValueError(
                f"statistics expression {name!r} depends on forcing fields {forcing_dependencies}; forcing layout is run-specific and cannot define persistent output storage"
            )
        reference_name = next(
            (
                dependency
                for dependency in names
                if self.metadata(dependency)[3] != "bool"
            ),
            names[0],
        )
        reference = self.metadata(reference_name)
        definite_ensemble_layouts: set[str] = set()
        for dependency in names:
            observed = self.metadata(dependency)
            incompatible = self.schema_identity(dependency) != self.schema_identity(
                reference_name
            ) or (observed[3] != "bool" and observed[3] != reference[3])
            if incompatible:
                raise ValueError(
                    f"statistics expression {name!r} mixes incompatible field metadata: {reference_name!r} has {reference}, but {dependency!r} has {observed}"
                )
            if self.selection.ensemble_size is not None:
                if observed[4] in {"state", "init_state"}:
                    definite_ensemble_layouts.add("batched")
                elif observed[4] in {"topology", "shared_state"}:
                    definite_ensemble_layouts.add("shared")
        if len(definite_ensemble_layouts) > 1:
            raise ValueError(
                f"statistics expression {name!r} mixes shared and member-batched fields"
            )
        coordinates = {
            self.schema_identity(dependency)[1]
            for dependency in names
            if self.schema_identity(dependency)[1] is not None
        }
        if len(coordinates) > 1:
            raise ValueError(
                f"statistics expression {name!r} mixes coordinate axes {sorted(coordinates)}"
            )
        target = None if target_metadata is None else self.schema_identity(name)[1]
        if (
            not isinstance(source, ScatterSource)
            and coordinates
            and (target is not None)
            and (target not in coordinates)
        ):
            raise ValueError(
                f"statistics field {name!r} declares dim_coords={target!r}, but its expression uses {next(iter(coordinates))!r}"
            )
        if target_metadata is not None:
            # A scatter declares its target domain; an expression its operands'.
            declared = self.schema_identity(name)[0]
            if isinstance(source, ScatterSource) and len(declared) != 1:
                raise ValueError(
                    f"scatter statistics field {name!r} declares shape {declared}; "
                    "a scatter target must be one-dimensional"
                )
            if (
                not isinstance(source, ScatterSource)
                and declared != self.schema_identity(reference_name)[0]
            ):
                raise ValueError(
                    f"statistics field {name!r} declares shape {declared}, but its "
                    f"expression has shape {reference[0]}"
                )
        if len(reference[0]) < 1:
            raise ValueError(
                f"statistics expression {name!r} must have at least one logical dimension"
            )
        if len(reference[0]) > 2:
            raise ValueError(
                f"statistics expression {name!r} has logical rank {len(reference[0])}; only rank <= 2 is supported"
            )
        shared_categories = {
            self.metadata(dependency)[4]
            for dependency in names
            if self.metadata(dependency)[4] in {"topology", "shared_state"}
        }
        if self.selection.ensemble_size is not None and shared_categories:
            raise ValueError(
                f"dynamic statistics expression {name!r} uses shared field categories {sorted(shared_categories)!r} in a multi-member model"
            )
        if isinstance(source, ScatterSource):
            index_tensor = self.fields[source.index].tensor
            index_coordinate = self.schema_identity(source.index)[1]
            if coordinates and index_coordinate not in coordinates:
                raise ValueError(
                    f"statistics scatter index {source.index!r} uses coordinate {index_coordinate!r}, but its values use {next(iter(coordinates))!r}"
                )
            if (
                len(index_tensor.shape) != 1
                or index_tensor.dtype not in {"idx", "int"}
                or index_tensor.category != "topology"
            ):
                raise ValueError(
                    f"statistics scatter index {source.index!r} must be a shared one-dimensional topology integer field"
                )
            if len(reference[0]) != 1:
                raise ValueError(
                    f"statistics scatter value {name!r} must be one-dimensional"
                )
            target_field = self.fields.get(name)
            target_tensor = None if target_field is None else target_field.tensor
            target_coord = None if target_tensor is None else target_tensor.dim_coords
            bare_target = None if target_coord is None else target_coord.split(".")[-1]
            if (
                target_tensor is None
                or target_tensor.output != "auto"
                or bare_target not in self.selection_targets
            ):
                raise ValueError(
                    f"scatter statistics field {name!r} requires an output selection on its declared coordinate"
                )
        value_expression = (
            source.value
            if isinstance(source, ScatterSource)
            else source.expression
            if isinstance(source, ExpressionSource)
            else None
        )
        if value_expression is not None:
            validate_expression_constants(
                name,
                value_expression,
                concrete_tensor_dtype(
                    reference[3] if target_metadata is None else target_metadata[3],
                    self.selection.dtype,
                    self.selection.mixed_precision,
                ),
            )
        return reference if target_metadata is None else target_metadata

    def compile_operation(
        self, output_name: str, operation: str, field_metadata: tuple[Any, ...]
    ) -> Any:
        parsed = parse_operation(operation)
        shape, output, coordinate, dtype, _category = field_metadata
        if dtype not in {"float", "hpfloat"}:
            unsupported = (
                parsed.inner is None
                and parsed.outer in {Reduction.MEAN, Reduction.SUM}
                or (
                    parsed.inner is not None
                    and (
                        parsed.inner
                        in {Reduction.MEAN, Reduction.SUM, Reduction.MAX, Reduction.MIN}
                        or parsed.outer in {Reduction.MEAN, Reduction.SUM}
                        or parsed.k > 1
                    )
                )
            )
            if unsupported:
                raise ValueError(
                    f"statistics operation {operation!r} for non-floating field {output_name!r} is unsupported"
                )
        selected = output == "auto" and coordinate in self.selection_targets
        if not selected and (parsed.k > 1 or parsed.stores_index):
            raise ValueError(
                f"full-output statistics field {output_name!r} does not support top-k or arg operation {operation!r}"
            )
        if (
            selected
            and len(shape) == 2
            and (parsed.compound or parsed.k > 1 or parsed.stores_index)
        ):
            raise ValueError(
                f"indexed-level statistics field {output_name!r} does not support compound, top-k, or arg operation {operation!r}"
            )
        return parsed

    def compile_output_options(
        self, output_name: str, field_metadata: tuple[Any, ...]
    ) -> Mapping[str, Any]:
        shape, _output, _coordinate, dtype, category = field_metadata
        batched = bool(
            self.selection.ensemble_size is not None
            and category in {"state", "init_state"}
        )
        netcdf_options = self.config.netcdf
        chunks = netcdf_options.get("chunksizes")
        if (
            chunks is not None
            and self.selection.ensemble_size is not None
            and (category in {"param", "derived_param", "virtual"})
        ):
            raise ValueError(
                f"NetCDF chunksizes for statistics output {output_name!r} cannot be fixed because its member batching depends on materialized parameter/expression storage"
            )
        dimensions = tuple(
            f"axis_{index}" for index in range(1 + len(shape) + int(batched))
        )
        tensor_dtype = concrete_tensor_dtype(
            dtype, self.selection.dtype, self.selection.mixed_precision
        )
        saved_dtype = tensor_dtype
        precision = self.config.save_precision
        if tensor_dtype.is_floating_point and precision:
            saved_dtype = _SAVE_DTYPES[precision]
        storage_dtype, logical_dtype = netcdf_dtype_encoding(
            torch_to_numpy_dtype(saved_dtype)
        )
        options = prepare_netcdf_variable_options(
            netcdf_options,
            dtype=storage_dtype,
            dimensions=dimensions,
            name=output_name,
            logical_dtype=logical_dtype,
        )
        return MappingProxyType(dict(options))

    def visit(self, name: str) -> None:
        if name in self.visited or name not in self.virtual_graph:
            return
        if name in self.visiting:
            raise ValueError(f"cyclic statistics dependency involving {name!r}")
        self.visiting.add(name)
        for dependency in self.virtual_graph[name]:
            self.visit(dependency)
        self.visiting.remove(name)
        self.visited.add(name)

    def resolve_declared_source(self, name: str) -> Any:
        existing = self.compiled_sources.get(name)
        if existing is not None:
            return existing
        tensor = self.fields[name].tensor
        expression = (
            tensor.expression
            if tensor.category == "virtual" and tensor.expression
            else None
        )
        source = (
            TensorSource(name) if expression is None else self.parse_source(expression)
        )
        if expression is not None:
            self.validate_expression(
                name=name,
                expression=expression,
                target_metadata=self.metadata(name),
                allow_scatter=True,
            )
        self.compiled_sources[name] = source
        self.resolve_value_dependencies(source)
        return source

    def resolve_value_dependencies(self, source: ValueSource) -> None:
        """Materialize virtual value dependencies; scatter indices are topology."""
        source_dependencies = (
            source.value.dependencies
            if isinstance(source, ScatterSource)
            else source.expression.dependencies
            if isinstance(source, ExpressionSource)
            else ()
        )
        for dependency in source_dependencies:
            dependency_tensor = self.fields[dependency].tensor
            if dependency_tensor.category == "virtual" and dependency_tensor.expression:
                self.resolve_declared_source(dependency)

    def compile_dynamic_output(
        self, output: StatisticsOutput, metadata: tuple[Any, ...]
    ) -> None:
        key = (output.name, output.operation)
        self.compiled_operations[key] = self.compile_operation(
            output.name, output.operation, metadata
        )
        self.compiled_output_metadata[key] = metadata
        name = f"{output.name}_{output.operation}"
        self.compiled_netcdf_options[name] = self.compile_output_options(name, metadata)

    def collect_fields(self) -> None:
        self.outputs: list[StatisticsOutput] = []
        expressions: dict[str, str | None] = {}
        pairs: set[tuple[str, str]] = set()
        for operation, items in self.config.variables.items():
            for item in items:
                if isinstance(item, str):
                    name = item
                    expression = None
                else:
                    name, expression = next(iter(item.items()))
                output = StatisticsOutput(
                    name=name, operation=operation, expression=expression
                )
                pair = (output.name, output.operation)
                if pair in pairs:
                    raise ValueError("variables must not repeat a field/operation")
                pairs.add(pair)
                previous = expressions.setdefault(output.name, output.expression)
                if previous != output.expression:
                    raise ValueError(
                        f"statistics output {output.name!r} has conflicting expressions"
                    )
                self.outputs.append(output)
        if not any(output.operation != "static" for output in self.outputs):
            raise ValueError("variables requires at least one dynamic output")
        # Inactive output-only fields stay visible so an alias can shadow them.
        field_names: FieldNameResolver[FieldEntry] = FieldNameResolver()
        for entry in self.field_plan.entries.values():
            if entry.active or entry.tensor.output_only:
                field_names.install(
                    entry.module,
                    entry.name,
                    entry,
                    expression_virtual=entry.expression_virtual,
                )
        self.fields = field_names.entries
        self.known = set(self.fields)
        self.selection_targets = {
            field.tensor.selects.split(".")[-1]
            for field in self.fields.values()
            if field.tensor.selects
        }

    def compile_outputs(self) -> None:
        self.compiled_operations: dict[tuple[str, str], Any] = {}
        self.compiled_output_metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
        self.compiled_netcdf_options: dict[str, Mapping[str, Any]] = {}
        for output in self.outputs:
            if output.operation == "static":
                operation = None
            else:
                operation = output.operation
            if output.expression is not None and (
                not self.active_declared_field(output.name)
            ):
                expression_metadata = self.validate_expression(
                    name=output.name,
                    expression=output.expression,
                    target_metadata=None,
                    allow_scatter=False,
                )
                self.compile_dynamic_output(output, expression_metadata)
                continue
            field = self.fields.get(output.name)
            if field is None:
                raise ValueError(
                    f"statistics output {output.name!r} is unknown, ambiguous, or inactive"
                )
            tensor = field.tensor
            if output.expression is not None and (
                not (tensor.depends_on or tensor.required_by or tensor.output_only)
            ):
                raise ValueError(
                    f"statistics alias {output.name!r} shadows an unconditional model field"
                )
            if tensor.output == "disabled" and output.expression is None:
                raise ValueError(
                    f"statistics field {output.name!r} is disabled for output"
                )
            if output.operation == "static" and (
                len(tensor.shape) != 1 or tensor.dim_coords is None
            ):
                raise ValueError(
                    f"static statistics field {output.name!r} must be one-dimensional and declare dim_coords"
                )
            if output.operation != "static" and tensor.category not in {
                "state",
                "shared_state",
                "init_state",
                "param",
                "virtual",
            }:
                raise ValueError(
                    f"statistics field {output.name!r} has unsupported category {tensor.category!r}"
                )
            if output.operation != "static" and (not tensor.shape):
                raise ValueError(
                    f"statistics field {output.name!r} must have at least one logical dimension"
                )
            if output.operation != "static" and len(tensor.shape) > 2:
                raise ValueError(
                    f"statistics field {output.name!r} has logical rank {len(tensor.shape)}; only rank <= 2 is supported"
                )
            if (
                output.operation != "static"
                and self.selection.ensemble_size is not None
                and (tensor.category in {"topology", "shared_state"})
            ):
                raise ValueError(
                    f"dynamic statistics field {output.name!r} is shared in a multi-member model"
                )
            if operation is not None:
                self.compile_dynamic_output(output, self.metadata(output.name))

    def validate_dependencies(self) -> None:
        self.virtual_graph: dict[str, tuple[str, ...]] = {}
        for name, field in self.fields.items():
            tensor = field.tensor
            if tensor.category != "virtual" or not tensor.expression:
                continue
            self.virtual_graph[name] = source_dependencies(
                self.parse_source(tensor.expression)
            )
        self.visiting: set[str] = set()
        self.visited: set[str] = set()
        for name in tuple(self.virtual_graph):
            self.visit(name)

    def compile_program(self) -> StatisticsDeclaration:
        self.grouped_operations: dict[str, list[Any]] = {}
        self.compiled_sources: dict[str, Any] = {}
        self.kernel_inputs: set[str] = set()
        self.safe_outputs: dict[str, tuple[str, str]] = {}
        self.generated_output_names: set[str] = set()
        for output in self.outputs:
            if output.operation == "static":
                continue
            self.grouped_operations.setdefault(output.name, []).append(
                self.compiled_operations[output.name, output.operation]
            )
            if output.expression is None or self.active_declared_field(output.name):
                source = self.resolve_declared_source(output.name)
            else:
                source = self.parse_source(output.expression)
                self.compiled_sources[output.name] = source
                self.resolve_value_dependencies(source)
            self.kernel_inputs.update(source_dependencies(source))
            output_name = f"{output.name}_{output.operation}"
            safe_name = sanitize_symbol(output_name)
            if not safe_name:
                raise ValueError(
                    f"statistics output {output_name!r} has no valid NetCDF characters"
                )
            # Output file names must also stay distinct on case-insensitive
            # file systems; field/operation pairs are already unique.
            file_key = safe_name.casefold()
            previous = self.safe_outputs.get(file_key)
            if previous is not None:
                raise ValueError(
                    f"statistics outputs {previous[1]!r} of {previous[0]!r} and "
                    f"{output.operation!r} of {output.name!r} both map to NetCDF "
                    f"variable {safe_name!r} (compared case-insensitively)"
                )
            self.safe_outputs[file_key] = (output.name, output.operation)
            operation = self.compiled_operations[output.name, output.operation]
            if operation.k > 1:
                output_storage_names = {
                    f"{safe_name}_{index}" for index in range(operation.k)
                }
            else:
                output_storage_names = {safe_name}
            self.generated_output_names.update(output_storage_names)
            output_metadata = self.compiled_output_metadata[
                output.name, output.operation
            ]
            coordinate = output_metadata[2]
            if (
                coordinate is not None
                and sanitize_symbol(coordinate) in output_storage_names
            ):
                raise ValueError(
                    f"statistics output {output_name!r} conflicts with its NetCDF coordinate {coordinate!r}"
                )
        for static_name in (
            output.name for output in self.outputs if output.operation == "static"
        ):
            safe_static = sanitize_symbol(static_name)
            if safe_static == "time":
                raise ValueError(
                    f"static statistics field {static_name!r} conflicts with the reserved NetCDF time variable"
                )
            if safe_static in self.generated_output_names:
                raise ValueError(
                    f"static statistics field {static_name!r} conflicts with a generated NetCDF output variable"
                )
        storage_names: set[str] = set()
        for name, source in self.compiled_sources.items():
            self.kernel_inputs.update(source_dependencies(source))
            if isinstance(source, ScatterSource):
                storage_names.add(StoragePlan.scatter_buffer(name))
                if source.reduction is Reduction.MEAN:
                    storage_names.add(StoragePlan.scatter_count(name))
        for name, operations in self.grouped_operations.items():
            storage = variable_slots(name, operations, (), torch.float32)
            storage_names.update(slot.name for slot in storage)
        reserved_collision = self.kernel_inputs.intersection(CONTROL_STATE)
        if reserved_collision:
            raise ValueError(
                f"statistics input names collide with reserved control state: {sorted(reserved_collision)}"
            )
        storage_collision = self.kernel_inputs.intersection(storage_names)
        if storage_collision:
            raise ValueError(
                f"statistics inputs collide with generated accumulator state: {sorted(storage_collision)}"
            )
        symbols: dict[str, str] = {}
        for name in sorted(
            self.kernel_inputs | storage_names | set(self.grouped_operations)
        ):
            symbol = sanitize_symbol(name)
            previous = symbols.get(symbol)
            if previous is not None and previous != name:
                raise ValueError(
                    f"statistics names {previous!r} and {name!r} both map to generated symbol {symbol!r}"
                )
            symbols[symbol] = name
        return StatisticsDeclaration(
            program=StatisticsProgram(
                operations=MappingProxyType(
                    {
                        name: tuple(operations)
                        for name, operations in self.grouped_operations.items()
                    }
                ),
                sources=MappingProxyType(dict(self.compiled_sources)),
            ),
            static_names=tuple(
                output.name for output in self.outputs if output.operation == "static"
            ),
            netcdf_options=MappingProxyType(dict(self.compiled_netcdf_options)),
        )

    def compile(self) -> StatisticsDeclaration:
        self.collect_fields()
        self.compile_outputs()
        self.validate_dependencies()
        return self.compile_program()


def _windows(declaration: ModelDeclaration) -> StatisticsPlan | None:
    """Resolve statistics windows and bind them to the schedule calendar."""

    config = declaration.output
    if not config.variables:
        return None
    configured = config.statistics_plan
    plan = StatisticsPlan() if configured is None else configured
    schedule = declaration.simulation_schedule
    if schedule is None:
        if not (
            isinstance(plan.inner, EveryStep)
            and isinstance(plan._effective_outer, EveryStep)
        ):
            raise ValueError(
                "calendar or explicit statistics windows require simulation_schedule"
            )
        if plan.partial_period == "drop":
            raise ValueError(
                'partial_period="drop" requires simulation_schedule to tell '
                "whether the last outer window is complete"
            )
        return plan
    plan = bind_statistics_plan_schedule(plan, schedule)
    validate_statistics_window_schedule(plan, schedule)
    return plan


def plan_output(
    selection: Selection, fields: FieldPlan, declaration: ModelDeclaration
) -> OutputPlan:
    """Compile the statistics program and resolve where output is written."""

    config = declaration.output
    windows = _windows(declaration)
    compiler = StatisticsDeclarationCompiler(selection, fields, config)
    statistics = None if windows is None else compiler.compile()
    directory = None
    if config.dir is not None:
        directory = config.dir / config.experiment
        parallel = selection.parallel
        if parallel is not None and parallel.ensemble_partitions > 1:
            directory /= f"ensemble_{parallel.ensemble_rank:04d}"
    precision = config.save_precision
    return OutputPlan(
        config=config,
        directory=directory,
        windows=windows,
        outputs=() if statistics is None else tuple(compiler.outputs),
        declaration=statistics,
        save_dtype=None if precision is None else _SAVE_DTYPES[precision],
    )

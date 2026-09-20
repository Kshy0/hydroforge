from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from hydroforge.compiler.namespace import FieldNameResolver
from hydroforge.contracts.fields import ModuleFieldSchema
from hydroforge.contracts.naming import RESERVED_CONTROL_STATE, sanitize_symbol
from hydroforge.contracts.temporal import _StatisticsOutput
from hydroforge.statistics.ir import (
    ExpressionSource,
    Reduction,
    ScatterSource,
    StatisticsProgram,
    TensorSource,
    ValueSource,
    _StatisticsDeclaration,
    build_variable_storage_plan,
    parse_operation,
    parse_value_source,
    validate_expression_constants,
)

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


class StatisticsDeclarationCompiler:
    """Compile fields, expressions, operations and output storage without I/O."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self._parsed_sources: dict[str, ValueSource] = {}

    def parse_source(self, expression: str) -> ValueSource:
        source = self._parsed_sources.get(expression)
        if source is None:
            source = parse_value_source(expression, self.known)
            self._parsed_sources[expression] = source
        return source

    def install_field(self, module_name: str, field: Any) -> None:
        tensor = field.tensor
        expression_virtual = bool(
            tensor is not None and tensor.category == "virtual" and tensor.expression
        )
        self.field_names.install(
            module_name, field.name, field, expression_virtual=expression_virtual
        )

    def active_declared_field(self, name: str) -> bool:
        field = self.fields.get(name)
        if field is None or field.tensor is None:
            return False
        return self.model._is_tensor_field_active(field.module_name, field)

    def metadata(self, name: str) -> tuple[Any, ...]:
        tensor = self.fields[name].tensor
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

    def dependencies(self, source: Any) -> tuple[str, ...]:
        if isinstance(source, TensorSource):
            return (source.name,)
        if isinstance(source, ExpressionSource):
            return source.expression.dependencies
        return (*source.value.dependencies, source.index)

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
            else self.dependencies(source)
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
            incompatible = (
                observed[0] != reference[0]
                or observed[2] != reference[2]
                or (observed[3] != "bool" and observed[3] != reference[3])
            )
            if incompatible:
                raise ValueError(
                    f"statistics expression {name!r} mixes incompatible field metadata: {reference_name!r} has {reference}, but {dependency!r} has {observed}"
                )
            if self.model.ensemble_size is not None:
                if observed[4] in {"state", "init_state"}:
                    definite_ensemble_layouts.add("batched")
                elif observed[4] in {"topology", "shared_state"}:
                    definite_ensemble_layouts.add("shared")
        if len(definite_ensemble_layouts) > 1:
            raise ValueError(
                f"statistics expression {name!r} mixes shared and member-batched fields"
            )
        coordinates = {
            self.metadata(dependency)[2]
            for dependency in names
            if self.metadata(dependency)[2] is not None
        }
        if len(coordinates) > 1:
            raise ValueError(
                f"statistics expression {name!r} mixes coordinate axes {sorted(coordinates)}"
            )
        target = None if target_metadata is None else target_metadata[2]
        if (
            not isinstance(source, ScatterSource)
            and coordinates
            and (target is not None)
            and (target not in coordinates)
        ):
            raise ValueError(
                f"statistics field {name!r} declares dim_coords={target!r}, but its expression uses {next(iter(coordinates))!r}"
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
        if self.model.ensemble_size is not None and shared_categories:
            raise ValueError(
                f"dynamic statistics expression {name!r} uses shared field categories {sorted(shared_categories)!r} in a multi-member model"
            )
        if isinstance(source, ScatterSource):
            index_tensor = self.fields[source.index].tensor
            index_coordinate = index_tensor.dim_coords
            if index_coordinate:
                index_coordinate = index_coordinate.rsplit(".", 1)[-1]
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
            from hydroforge.contracts.fields import concrete_tensor_dtype

            validate_expression_constants(
                name,
                value_expression,
                concrete_tensor_dtype(
                    reference[3] if target_metadata is None else target_metadata[3],
                    self.model.dtype,
                    self.model.mixed_precision,
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
        from hydroforge.contracts.fields import concrete_tensor_dtype
        from hydroforge.data.distributed import torch_to_numpy_dtype
        from hydroforge.serialization.netcdf import (
            _prepare_netcdf_variable_options_trusted,
            netcdf_dtype_encoding,
        )

        shape, _output, _coordinate, dtype, category = field_metadata
        batched = bool(
            self.model.ensemble_size is not None and category in {"state", "init_state"}
        )
        chunks = self.model.output_netcdf_options.get("chunksizes")
        if (
            chunks is not None
            and self.model.ensemble_size is not None
            and (category in {"param", "derived_param", "virtual"})
        ):
            raise ValueError(
                f"NetCDF chunksizes for statistics output {output_name!r} cannot be fixed because its member batching depends on materialized parameter/expression storage"
            )
        dimensions = tuple(
            f"axis_{index}" for index in range(1 + len(shape) + int(batched))
        )
        tensor_dtype = concrete_tensor_dtype(
            dtype, self.model.dtype, self.model.mixed_precision
        )
        saved_dtype = tensor_dtype
        if tensor_dtype.is_floating_point and self.model.statistics_save_precision:
            saved_dtype = {"float32": torch.float32, "float64": torch.float64}[
                self.model.statistics_save_precision
            ]
        storage_dtype, logical_dtype = netcdf_dtype_encoding(
            torch_to_numpy_dtype(saved_dtype)
        )
        options = _prepare_netcdf_variable_options_trusted(
            self.model.output_netcdf_options,
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
        self, output: _StatisticsOutput, metadata: tuple[Any, ...]
    ) -> None:
        key = (output.name, output.operation)
        self.compiled_operations[key] = self.compile_operation(
            output.name, output.operation, metadata
        )
        self.compiled_output_metadata[key] = metadata
        name = f"{output.name}_{output.operation}"
        self.compiled_netcdf_options[name] = self.compile_output_options(name, metadata)

    def collect_fields(self) -> None:
        cls = type(self.model)
        self.outputs: list[_StatisticsOutput] = []
        expressions: dict[str, str | None] = {}
        pairs: set[tuple[str, str]] = set()
        for operation, items in self.model.variables_to_save.items():
            for item in items:
                if isinstance(item, str):
                    name = item
                    expression = None
                else:
                    name, expression = next(iter(item.items()))
                output = _StatisticsOutput(
                    name=name, operation=operation, expression=expression
                )
                pair = (output.name, output.operation)
                if pair in pairs:
                    raise ValueError(
                        "variables_to_save must not repeat a field/operation"
                    )
                pairs.add(pair)
                previous = expressions.setdefault(output.name, output.expression)
                if previous != output.expression:
                    raise ValueError(
                        f"statistics output {output.name!r} has conflicting expressions"
                    )
                self.outputs.append(output)
        if not any(output.operation != "static" for output in self.outputs):
            raise ValueError("variables_to_save requires at least one dynamic output")
        self.model._statistics_outputs = tuple(self.outputs)
        opened_modules = self.model.opened_modules
        self.field_names: FieldNameResolver[ModuleFieldSchema] = FieldNameResolver()
        self.fields = self.field_names.entries
        schema = cls._compiled_schema()
        module_types = cls._module_types()
        for module_name in opened_modules:
            for field in schema.fields(module_name):
                tensor = field.tensor
                if tensor is None:
                    continue
                if not self.model._is_tensor_field_active(module_name, field) and (
                    not tensor.output_only
                ):
                    continue
                self.install_field(module_name, field)
            module_type = module_types[module_name]
            for field_name in module_type._reference_index_fields(
                opened_modules=opened_modules,
                field_demand=self.model._field_demand,
            ):
                field = module_type._get_tensor_schema(
                    field_name,
                    opened_modules=opened_modules,
                    field_demand=self.model._field_demand,
                )
                if field is None:
                    raise ValueError(
                        f"ReferenceIndexField {module_name}.{field_name} has no tensor schema"
                    )
                self.install_field(module_name, field)
        self.known = set(self.fields)
        self.selection_targets = {
            field.tensor.selects.split(".")[-1]
            for field in self.fields.values()
            if field.tensor is not None and field.tensor.selects
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
                and self.model.ensemble_size is not None
                and (tensor.category in {"topology", "shared_state"})
            ):
                raise ValueError(
                    f"dynamic statistics field {output.name!r} is shared in a multi-member model"
                )
            if tensor.category == "virtual" and tensor.expression:
                self.validate_expression(
                    name=output.name,
                    expression=tensor.expression,
                    target_metadata=self.metadata(output.name),
                    allow_scatter=True,
                )
            if operation is not None:
                self.compile_dynamic_output(output, self.metadata(output.name))

    def validate_dependencies(self) -> None:
        self.virtual_graph: dict[str, tuple[str, ...]] = {}
        for name, field in self.fields.items():
            tensor = field.tensor
            if tensor.category != "virtual" or not tensor.expression:
                continue
            self.virtual_graph[name] = self.dependencies(
                self.parse_source(tensor.expression)
            )
        self.visiting: set[str] = set()
        self.visited: set[str] = set()
        for name in tuple(self.virtual_graph):
            self.visit(name)

    def compile_program(self) -> None:
        self.grouped_operations: dict[str, list[Any]] = {}
        self.compiled_sources: dict[str, Any] = {}
        self.kernel_inputs: set[str] = set()
        self.safe_outputs: dict[str, str] = {}
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
            self.kernel_inputs.update(self.dependencies(source))
            output_name = f"{output.name}_{output.operation}"
            safe_name = sanitize_symbol(output_name)
            if not safe_name:
                raise ValueError(
                    f"statistics output {output_name!r} has no valid NetCDF characters"
                )
            previous = self.safe_outputs.get(safe_name)
            if previous is not None and previous != output_name:
                raise ValueError(
                    f"statistics outputs {previous!r} and {output_name!r} both map to NetCDF variable {safe_name!r}"
                )
            self.safe_outputs[safe_name] = output_name
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
        for name, operations in self.grouped_operations.items():
            storage = build_variable_storage_plan(name, (), tuple(operations))
            storage_names.update(slot.name for slot in storage.slots)
        reserved_collision = self.kernel_inputs.intersection(RESERVED_CONTROL_STATE)
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
        self.model._statistics_declaration = _StatisticsDeclaration(
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

    def compile(self) -> AbstractModel:
        if self.model._statistics_plan is None:
            return self.model
        self.collect_fields()
        self.compile_outputs()
        self.validate_dependencies()
        self.compile_program()
        return self.model

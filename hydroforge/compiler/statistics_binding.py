"""Bind model fields and output requests to the statistics runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from hydroforge.statistics.runtime import StatisticsRuntime, StatisticsConfig
from hydroforge.statistics.ir import (
    ExpressionSource, ScatterSource, parse_operation, parse_value_source,
)
from hydroforge.statistics.layout import compile_statistics
from hydroforge.contracts.events import emit
from hydroforge.contracts.fields import RuntimeTensorMetadata, TensorMetadata


@dataclass(frozen=True, slots=True)
class _AttributeTensorBinding:
    owner: Any
    attribute: str

    def resolve(self, name: str) -> torch.Tensor:
        value = getattr(self.owner, self.attribute, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Statistics field {name!r} resolved to "
                f"{type(value).__name__}, expected Tensor"
            )
        return value


@dataclass(frozen=True, slots=True)
class _OutputTensorBinding:
    partition: Any
    field: Any
    binding_name: str

    def resolve(self, name: str) -> torch.Tensor:
        _, tensors = self.partition.bind_output(self.field)
        try:
            value = tensors[self.binding_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Statistics output binding {name!r} is no longer resolved"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Statistics output binding {name!r} resolved to "
                f"{type(value).__name__}, expected Tensor"
            )
        return value


def _tensor_abi(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


class StatisticsBindingCompiler:
    """Own aggregator construction, request normalization and result access."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._aggregator: StatisticsRuntime | None = None
        self._tensor_bindings: dict[
            str, _AttributeTensorBinding | _OutputTensorBinding
        ] = {}

    @property
    def variable_map(self):
        return self.model._namespace.build()

    def _field(self, name: str):
        module, attribute, _ = self.variable_map[name]
        field = module.get_tensor_schema(attribute)
        if field is None:
            raise ValueError(f"Model field {name!r} has no tensor schema")
        return field

    @property
    def aggregator(self) -> StatisticsRuntime | None:
        return self._aggregator

    def create(self) -> StatisticsRuntime:
        model = self.model
        aggregator = StatisticsRuntime(StatisticsConfig(
            device=model.device,
            backend=model._execution.backend,
            output_dir=model.output_full_dir,
            rank=model.rank,
            world_size=model.world_size,
            num_workers=model.output_workers,
            output_split_by_year=model.output_split_by_year,
            num_trials=model.num_trials or 1,
            save_kernels=model.save_kernels,
            max_pending_steps=model.max_pending_steps,
            calendar=model.calendar,
            in_memory=model.in_memory_output,
            result_device=model.result_device,
            save_precision=torch.float32,
            output_netcdf_options=model.output_netcdf_options,
            event_sink=model.event_sink,
        ))
        aggregator._execution = model._execution
        self._aggregator = aggregator
        model._execution.statistics.attach(aggregator)
        return aggregator

    def initialize(self, requests: Mapping[str, Any]) -> None:
        aggregator = self.create()
        variable_ops, expressions = self.normalize_requests(requests)
        if not variable_ops:
            raise ValueError(
                "variables_to_save must contain at least one dynamic output"
            )
        adhoc = self.prepare_virtuals(variable_ops, expressions)
        self._register_dynamic(variable_ops, adhoc)
        aggregator.initialize(compile_statistics(aggregator, variable_ops))

    def normalize_requests(
        self,
        requests: Mapping[str, Any],
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Compile public output requests and materialize static selections."""
        variable_ops: dict[str, list[str]] = {}
        expressions: dict[str, str] = {}
        for spelling, values in requests.items():
            if spelling == "static":
                self._register_static(values)
                continue
            normalized = str(spelling).lower()
            parse_operation(normalized)
            if isinstance(values, (str, tuple)):
                items = [values]
            elif isinstance(values, list):
                items = values
            else:
                raise ValueError(
                    f"variables_to_save[{spelling!r}] must be a string, "
                    "(name, expression), or a list"
                )
            for item in items:
                name, expression = self._normalize_item(spelling, item)
                if expression is not None:
                    previous = expressions.get(name)
                    if previous is not None and previous != expression:
                        raise ValueError(
                            f"Conflicting expressions for alias {name!r}: "
                            f"{previous!r} vs {expression!r}"
                        )
                    expressions[name] = expression
                operations = variable_ops.setdefault(name, [])
                if normalized not in operations:
                    operations.append(normalized)
        return variable_ops, expressions

    @staticmethod
    def _normalize_item(
        spelling: str,
        item: Any,
    ) -> tuple[str, str | None]:
        if isinstance(item, str):
            return item, None
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return str(item[0]), str(item[1])
        if isinstance(item, dict) and len(item) == 1:
            name, expression = next(iter(item.items()))
            return str(name), str(expression)
        raise ValueError(
            f"Invalid item in variables_to_save[{spelling!r}]: {item!r}; "
            "expected a field, {alias: expression}, or (alias, expression)"
        )

    def _register_static(self, values: Any) -> None:
        model = self.model
        aggregator = self.aggregator
        if aggregator is None:
            raise RuntimeError("statistics aggregator has not been created")
        items = [values] if isinstance(values, str) else list(values)
        for name in items:
            if not isinstance(name, str):
                raise ValueError(
                    "variables_to_save['static'] entries must be field names"
                )
            try:
                module, attribute, _ = self.variable_map[name]
            except KeyError as exc:
                raise ValueError(
                    f"Static variable {name!r} was not found in an opened module"
                ) from exc
            tensor = getattr(module, attribute)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                raise ValueError(
                    f"Static variable {name!r} must be a one-dimensional tensor"
                )
            field = module.get_tensor_schema(attribute)
            if field is None:
                raise ValueError(f"Static variable {name!r} has no field metadata")
            if field.tensor.output == "disabled":
                raise ValueError(f"Variable {name!r} is disabled for output")
            bound, tensors = model._partition.bind_output(field)
            coordinate = bound.output_coord
            if coordinate is None:
                raise ValueError(
                    f"Static variable {name!r} must declare dim_coords"
                )
            if name == coordinate:
                continue
            output_index = tensors.get(bound.output_index)
            aggregator.register_static(
                name, tensor, output_index=output_index, coordinate=coordinate,
            )

    def prepare_virtuals(
        self,
        variable_ops: Mapping[str, list[str]],
        explicit_expressions: Mapping[str, str],
    ) -> dict[str, Any]:
        """Validate declared virtuals and construct explicit ad-hoc fields."""
        known = set(self.variable_map)
        adhoc: dict[str, Any] = {}
        for name in variable_ops:
            if name in self.variable_map:
                module, attribute, _ = self.variable_map[name]
                field = module.get_tensor_schema(attribute)
                metadata = field.tensor
                expression = (
                    metadata.expression
                    if metadata.category == "virtual" else None
                )
                if expression:
                    source = parse_value_source(expression, known)
                    if isinstance(source, ExpressionSource):
                        self._validate_coordinate_dependencies(
                            name, source.expression.dependencies,
                            metadata.dim_coords,
                        )
                continue

            expression = explicit_expressions.get(name, name)
            source = parse_value_source(expression, known)
            if isinstance(source, ScatterSource):
                raise ValueError(
                    f"Scatter expression {expression!r} for ad-hoc variable "
                    f"{name!r} must be declared as a module virtual field"
                )
            dependencies = source.expression.dependencies
            if not dependencies:
                raise ValueError(
                    f"Ad-hoc output {name!r} has no registered field dependency"
                )
            output, coordinate = self._common_field_metadata(
                name, expression, dependencies,
            )
            adhoc[name] = RuntimeTensorMetadata(
                tensor=TensorMetadata.compile({
                    "tensor_shape": (), "category": "virtual",
                    "expr": expression, "dim_coords": coordinate,
                    "output": output,
                }),
                description=f"Ad-hoc expression: {expression}",
            )
        return adhoc

    def expand_dependencies(
        self,
        variable_ops: Mapping[str, list[str]],
        adhoc: Mapping[str, Any],
    ) -> list[str]:
        """Return selected fields and all typed virtual dependencies."""
        known = set(self.variable_map) | set(adhoc)
        ordered = list(variable_ops)
        seen = set(ordered)
        cursor = 0
        while cursor < len(ordered):
            name = ordered[cursor]
            cursor += 1
            info = adhoc.get(name)
            if info is None and name in self.variable_map:
                module, attribute, _ = self.variable_map[name]
                info = module.get_tensor_schema(attribute)
            metadata = info.tensor
            expression = (
                metadata.expression if metadata.category == "virtual" else None
            )
            if not expression:
                continue
            source = parse_value_source(expression, known)
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

    def _register_dynamic(
        self,
        variable_ops: Mapping[str, list[str]],
        adhoc: Mapping[str, Any],
    ) -> None:
        model = self.model
        aggregator = self.aggregator
        if aggregator is None:
            raise RuntimeError("statistics aggregator has not been created")
        by_shape: dict[tuple[int, ...], list[str]] = {}
        registered: set[str] = set()

        def register_tensor(
            name: str,
            tensor: torch.Tensor,
            info: Any,
            binding: _AttributeTensorBinding | _OutputTensorBinding,
            *,
            output_coordinate: bool = False,
        ) -> None:
            if output_coordinate:
                aggregator.register_output_coordinate(name, tensor)
            else:
                aggregator.register_tensor(name, tensor, info)
            self._tensor_bindings[name] = binding
            registered.add(name)
            by_shape.setdefault(tuple(tensor.shape), []).append(name)

        for name in self.expand_dependencies(variable_ops, adhoc):
            if name in registered:
                continue
            if name not in self.variable_map:
                info = adhoc.get(name)
                if info is None:
                    raise ValueError(
                        f"Output dependency {name!r} is not a registered model field"
                    )
                aggregator.register_virtual_tensor(name, info)
                registered.add(name)
                continue

            module, attribute, _ = self.variable_map[name]
            if not hasattr(module, attribute):
                raise ValueError(f"Output field {name!r} has no runtime value")
            tensor = getattr(module, attribute)
            field = module.get_tensor_schema(attribute)
            if field is None:
                raise ValueError(f"Output field {name!r} has no tensor metadata")
            if name in variable_ops and field.tensor.output == "disabled":
                raise ValueError(f"Variable {name!r} is disabled for output")
            info, bindings = model._partition.bind_output(field)
            category = info.tensor.category
            allowed = {"state", "shared_state", "init_state", "param", "virtual"}
            dependency_only = name not in variable_ops
            if category != "topology" or not dependency_only:
                if category not in allowed:
                    raise ValueError(
                        f"Output variable {name!r} has unsupported category "
                        f"{category!r}; allowed={sorted(allowed)}"
                    )
            if category == "virtual" and info.tensor.expression:
                aggregator.register_virtual_tensor(name, info)
                registered.add(name)
            elif isinstance(tensor, torch.Tensor):
                register_tensor(
                    name, tensor, info,
                    _AttributeTensorBinding(module, attribute),
                )
            else:
                raise TypeError(
                    f"Output field {name!r} is {type(tensor).__name__}, expected Tensor"
                )

            for binding_name in (info.output_index, info.output_coord):
                if not binding_name or binding_name in registered:
                    continue
                try:
                    binding = bindings[binding_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Runtime output binding {binding_name!r} was not "
                        f"resolved for {name!r}"
                    ) from exc
                register_tensor(
                    binding_name, binding, {},
                    _OutputTensorBinding(
                        model._partition, field, binding_name,
                    ),
                    output_coordinate=binding_name == info.output_coord,
                )

        for shape, names in by_shape.items():
            emit(
                model, "info", "statistics.tensors_registered",
                "Registered tensors for streaming statistics",
                rank=model.rank, variables=tuple(names), shape=str(shape),
            )

    def _field_metadata(self, name: str) -> tuple[str | None, str | None]:
        module, attribute, _ = self.variable_map[name]
        field = module.get_tensor_schema(attribute)
        if field is None:
            return None, None
        coordinate = field.tensor.dim_coords
        if coordinate:
            coordinate = coordinate.split(".")[-1]
        return field.tensor.output, coordinate

    def _validate_coordinate_dependencies(
        self,
        name: str,
        dependencies: tuple[str, ...],
        target_coordinate: str | None,
    ) -> None:
        coordinates = {
            self._field_metadata(dependency)[1]
            for dependency in dependencies
            if self._field_metadata(dependency)[1] is not None
        }
        if len(coordinates) > 1:
            raise ValueError(
                f"Virtual field {name!r} mixes coordinate axes "
                f"{sorted(coordinates)}"
            )
        target = target_coordinate.split(".")[-1] if target_coordinate else None
        if coordinates and target not in coordinates:
            raise ValueError(
                f"Virtual field {name!r} declares dim_coords={target!r}, "
                f"but its expression uses {next(iter(coordinates))!r}"
            )

    def _common_field_metadata(
        self,
        name: str,
        expression: str,
        dependencies: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        reference = self._field_metadata(dependencies[0])
        for dependency in dependencies[1:]:
            observed = self._field_metadata(dependency)
            if observed != reference:
                raise ValueError(
                    f"Inconsistent metadata in virtual variable {name!r} "
                    f"({expression!r}): {dependencies[0]!r} has {reference}, "
                    f"but {dependency!r} has {observed}"
                )
        return reference

    def close(self) -> None:
        aggregator = self._aggregator
        if aggregator is not None:
            self._aggregator = None
            failures: list[BaseException] = []
            try:
                self.model._execution.statistics.detach(aggregator)
            except BaseException as error:
                failures.append(error)
            try:
                aggregator._shutdown()
            except BaseException as error:
                failures.append(error)
            if failures:
                from hydroforge.contracts import ResourceCleanupError

                error = ResourceCleanupError("statistics resources", failures)
                raise error from failures[0]

    def refresh_bindings(self) -> bool:
        """Transactionally rebind exact-ABI tensor replacements.

        Statistics storage and generated kernels are specialized to tensor
        shape, dtype, device, layout and stride. Changing any of those while
        an aggregator is live has no unambiguous accumulator migration, so it
        is rejected instead of partially rebuilding against stale storage.
        """

        aggregator = self.aggregator
        if aggregator is None:
            return False
        registered = set(aggregator._tensor_registry)
        declared = set(self._tensor_bindings)
        if registered != declared:
            raise RuntimeError(
                "Statistics tensor binding registry is inconsistent: "
                f"missing_resolvers={sorted(registered - declared)}, "
                f"orphan_resolvers={sorted(declared - registered)}"
            )

        replacements: dict[str, torch.Tensor] = {}
        for name, binding in self._tensor_bindings.items():
            previous = aggregator._tensor_registry[name]
            current = binding.resolve(name)
            if current.layout is not torch.strided or not current.is_contiguous():
                raise ValueError(
                    f"Statistics tensor {name!r} replacement must retain the "
                    "canonical contiguous strided buffer ABI"
                )
            if current is previous:
                continue
            previous_abi = _tensor_abi(previous)
            current_abi = _tensor_abi(current)
            if current_abi != previous_abi:
                labels = ("dtype", "device", "layout", "shape", "stride")
                changes = {
                    label: (old, new)
                    for label, old, new in zip(
                        labels, previous_abi, current_abi, strict=True,
                    )
                    if old != new
                }
                raise ValueError(
                    f"Statistics tensor {name!r} changed compiled ABI: "
                    f"{changes}. Close and recreate the model/statistics "
                    "runtime instead of migrating live accumulators"
                )
            if (
                isinstance(binding, _OutputTensorBinding)
                and not torch.equal(current, previous)
            ):
                raise ValueError(
                    f"Statistics output topology binding {name!r} changed "
                    "values while accumulators are live; recreate the model/"
                    "statistics runtime at the topology boundary"
                )
            replacements[name] = current

        if replacements:
            previous = {
                name: aggregator._tensor_registry[name]
                for name in replacements
            }
            aggregator._tensor_registry.update(replacements)
            try:
                aggregator._prepare_kernel_states()
            except BaseException:
                aggregator._tensor_registry.update(previous)
                raise
        return bool(replacements)

    def results(self, *, stacked: bool) -> dict[str, torch.Tensor]:
        if self.aggregator is None:
            raise RuntimeError("Statistics aggregator is not initialized")
        return self.aggregator.get_results(as_stacked=stacked)

    def result(
        self, variable: str, operation: str, *, stacked: bool,
    ) -> torch.Tensor:
        if self.aggregator is None:
            raise RuntimeError("Statistics aggregator is not initialized")
        return self.aggregator.get_result(variable, operation, as_stacked=stacked)

    def time_index(self) -> int:
        if self.aggregator is None:
            return 0
        return self.aggregator.get_time_index()

    def reset_time_index(self) -> None:
        if self.aggregator is not None:
            self.aggregator.reset_time_index()

    def accumulator(self, variable: str, operation: str) -> torch.Tensor:
        """Return an ownership-isolated differentiable accumulator snapshot."""

        if self.aggregator is None:
            raise RuntimeError("Statistics aggregator is not initialized")
        key = f"{variable}_{operation}"
        try:
            accumulator = self.aggregator._storage[key]
        except KeyError as exc:
            raise KeyError(
                f"No live accumulator {key!r}; available="
                f"{sorted(self.aggregator._storage)}"
            ) from exc
        return accumulator.clone(memory_format=torch.preserve_format)

    def pop_result(self, variable: str, operation: str) -> torch.Tensor:
        """Remove and return the newest finalized in-memory result."""

        if self.aggregator is None:
            raise RuntimeError("Statistics aggregator is not initialized")
        key = f"{variable}_{operation}"
        values = self.aggregator._result_tensors.get(key)
        if values is None:
            raise KeyError(
                f"No in-memory result {key!r}; available="
                f"{sorted(self.aggregator._result_tensors)}"
            )
        if not values:
            raise RuntimeError(f"No finalized result is available for {key!r}")
        return values.pop()

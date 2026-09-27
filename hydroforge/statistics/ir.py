"""Typed, backend-neutral statistics intermediate representation."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from hydroforge.contracts.fields import RuntimeTensorMetadata
    from hydroforge.statistics.layout import StatisticsVariableLayout


class Reduction(StrEnum):
    __str__ = Enum.__str__
    __format__ = Enum.__format__

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    FIRST = "first"
    LAST = "last"


class StorageInitialization(StrEnum):
    __str__ = Enum.__str__
    __format__ = Enum.__format__

    ZERO = "zero"
    NEGATIVE_INFINITY = "negative_infinity"
    POSITIVE_INFINITY = "positive_infinity"


class StorageDType(StrEnum):
    __str__ = Enum.__str__
    __format__ = Enum.__format__

    VALUE = "value"
    INDEX = "index"


def storage_initialization(reduction: Reduction) -> StorageInitialization:
    if reduction is Reduction.MAX:
        return StorageInitialization.NEGATIVE_INFINITY
    if reduction is Reduction.MIN:
        return StorageInitialization.POSITIVE_INFINITY
    return StorageInitialization.ZERO


@dataclass(frozen=True)
class StatisticOperation:
    """One output operation, normalized independently of backend syntax."""

    spelling: str
    outer: Reduction
    inner: Reduction | None
    k: int
    stores_index: bool

    @property
    def compound(self) -> bool:
        return self.inner is not None


@dataclass(frozen=True)
class Expression:
    """Validated Python expression AST and its model-field dependencies."""

    source: str
    tree: ast.Expression
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class TensorSource:
    name: str


@dataclass(frozen=True)
class ExpressionSource:
    expression: Expression


@dataclass(frozen=True)
class ScatterSource:
    reduction: Reduction
    value: Expression
    index: str


def validate_expression_constants(
    name: str,
    expression: Expression,
    dtype: torch.dtype,
) -> None:
    """Validate literal meaning once at the public statistics boundary."""

    if not dtype.is_floating_point:
        raise ValueError(f"statistics expression {name!r} requires a floating dtype")
    for node in ast.walk(expression.tree):
        if (
            not isinstance(node, ast.Constant)
            or isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
        ):
            continue
        value = node.value
        if isinstance(value, float) and value == 0.0:
            literal = ast.get_source_segment(expression.source, node)
            if literal is not None:
                try:
                    intended = Decimal(literal.replace("_", ""))
                except InvalidOperation:
                    intended = Decimal(0)
                if intended != 0:
                    raise ValueError(
                        f"statistics expression {name!r} constant "
                        f"{literal!r} underflows Python float64"
                    )
        # Narrowing may round tiny constants to subnormals or zero; only
        # overflow and inexact integers change a constant's meaning.
        converted = float(torch.tensor(value, dtype=dtype).item())
        if not math.isfinite(converted):
            raise ValueError(
                f"statistics expression {name!r} constant {value!r} "
                f"exceeds {dtype} range"
            )
        if isinstance(value, int) and int(converted) != value:
            raise ValueError(
                f"statistics expression {name!r} integer constant "
                f"{value!r} cannot be represented exactly in {dtype}"
            )


ValueSource = TensorSource | ExpressionSource | ScatterSource


@dataclass(frozen=True)
class StatisticsProgram:
    """Shape-independent statistics semantics compiled before allocation."""

    operations: Mapping[str, tuple[StatisticOperation, ...]]
    sources: Mapping[str, ValueSource]

    def dependencies(self, name: str) -> tuple[str, ...]:
        source = self.sources.get(name) or TensorSource(name)
        if isinstance(source, TensorSource):
            return (source.name,)
        expression = (
            source.expression if isinstance(source, ExpressionSource) else source.value
        )
        return expression.dependencies

    def leaf_tensors(self, name: str) -> tuple[str, ...]:
        """Return concrete model tensors required to evaluate ``name``."""
        leaves: set[str] = set()
        visited: set[str] = set()

        def visit(field: str) -> None:
            if field in visited:
                return
            visited.add(field)
            source = self.sources.get(field) or TensorSource(field)
            if isinstance(source, TensorSource):
                leaves.add(source.name)
                return
            if isinstance(source, ScatterSource):
                leaves.add(source.index)
                dependencies = source.value.dependencies
            else:
                dependencies = source.expression.dependencies
            for dependency in dependencies:
                visit(dependency)

        visit(name)
        return tuple(sorted(leaves))


@dataclass(frozen=True, slots=True)
class _StatisticsDeclaration:
    """Construction-time statistics semantics owned by a validated model."""

    program: StatisticsProgram
    static_names: tuple[str, ...]
    netcdf_options: Mapping[str, Mapping[str, Any]]

    @property
    def variable_ops(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                name: tuple(operation.spelling for operation in operations)
                for name, operations in self.program.operations.items()
            }
        )


@dataclass(frozen=True)
class MaterializedScatter:
    """One reachable scatter source that must run before aggregation."""

    name: str
    source: ScatterSource


@dataclass(frozen=True)
class StorageSlot:
    name: str
    shape: tuple[int, ...]
    dtype: StorageDType
    initialization: StorageInitialization
    output: bool


@dataclass(frozen=True)
class VariableStoragePlan:
    """Backend-neutral allocation plan for one selected statistic field."""

    variable: str
    slots: tuple[StorageSlot, ...]


def build_variable_storage_plan(
    variable: str,
    actual_shape: tuple[int, ...],
    operations: tuple[StatisticOperation, ...],
) -> VariableStoragePlan:
    slots: list[StorageSlot] = []
    internal_names: set[str] = set()

    def add_internal(
        name: str,
        initialization: StorageInitialization,
    ) -> None:
        if name in internal_names:
            return
        internal_names.add(name)
        slots.append(
            StorageSlot(
                name,
                actual_shape,
                StorageDType.VALUE,
                initialization,
                False,
            )
        )

    for operation in operations:
        shape = actual_shape + (operation.k,) if operation.k > 1 else actual_shape
        initialization = storage_initialization(operation.outer)
        dtype = StorageDType.INDEX if operation.stores_index else StorageDType.VALUE
        slots.append(
            StorageSlot(
                f"{variable}_{operation.spelling}",
                shape,
                dtype,
                StorageInitialization.ZERO
                if operation.stores_index
                else initialization,
                True,
            )
        )
        if operation.stores_index:
            add_name = f"{variable}_{operation.spelling}_aux"
            if add_name not in internal_names:
                internal_names.add(add_name)
                slots.append(
                    StorageSlot(
                        add_name,
                        shape,
                        StorageDType.VALUE,
                        initialization,
                        False,
                    )
                )
        if operation.inner is None:
            if operation.outer is Reduction.MEAN:
                add_internal(
                    f"{variable}_mean_sample_weight_state",
                    StorageInitialization.ZERO,
                )
            continue
        if operation.inner is Reduction.LAST:
            continue
        add_internal(
            f"{variable}_{operation.inner.value}_inner_state",
            storage_initialization(operation.inner),
        )
        if operation.inner is Reduction.MEAN:
            add_internal(
                f"{variable}_{operation.inner.value}_weight_state",
                StorageInitialization.ZERO,
            )
    return VariableStoragePlan(variable, tuple(slots))


@dataclass(frozen=True)
class StatisticVariable:
    name: str
    safe_name: str
    source: ValueSource
    operations: tuple[StatisticOperation, ...]
    tensor_shape: tuple[Any, ...]
    actual_shape: tuple[int, ...]
    actual_ndim: int
    output_group: str


@dataclass(frozen=True)
class StatisticsIR:
    """Complete aggregation program consumed by backend syntax emitters."""

    variables: tuple[StatisticVariable, ...]
    by_name: Mapping[str, StatisticVariable]
    grouped_variables: Mapping[str, tuple[StatisticVariable, ...]]
    sources: Mapping[str, ValueSource]

    def materialized_inputs(self, name: str) -> tuple[str, ...]:
        """Return leaf buffers read by the main aggregation kernel."""
        inputs: set[str] = set()
        visited: set[str] = set()

        def visit(field: str) -> None:
            if field in visited:
                return
            visited.add(field)
            source = self.sources.get(field) or TensorSource(field)
            if isinstance(source, TensorSource):
                inputs.add(source.name)
            elif isinstance(source, ScatterSource):
                inputs.add(f"__scatter_buf_{field}")
            else:
                for dependency in source.expression.dependencies:
                    visit(dependency)

        visit(name)
        return tuple(sorted(inputs))

    def scatter_inputs(self, name: str) -> tuple[str, ...]:
        """Return source and index buffers for one scatter pre-kernel."""
        source = self.sources[name]
        inputs = {source.index}
        for dependency in source.value.dependencies:
            inputs.update(self.materialized_inputs(dependency))
        return tuple(sorted(inputs))

    def ordered_scatters(self) -> tuple[MaterializedScatter, ...]:
        """Topologically order scatter materializations by virtual dependency."""
        result: list[MaterializedScatter] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            source = self.sources.get(name) or TensorSource(name)
            if isinstance(source, TensorSource):
                return
            dependencies = (
                source.value.dependencies
                if isinstance(source, ScatterSource)
                else source.expression.dependencies
            )
            for dependency in dependencies:
                visit(dependency)
            if isinstance(source, ScatterSource):
                result.append(MaterializedScatter(name, source))

        for variable in self.variables:
            visit(variable.name)
        return tuple(result)


_OP_RE = re.compile(r"^(arg)?(mean|sum|max|min|first|last)(\d*)$")

_FUNCTION_ARITIES = {
    "abs": 1,
    "sqrt": 1,
    "exp": 1,
    "log": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "pow": 2,
    "maximum": 2,
    "minimum": 2,
    "where": 3,
}


def parse_operation(spelling: str) -> StatisticOperation:
    parts = spelling.split("_")
    if len(parts) > 2:
        raise ValueError(f"invalid statistics operation {spelling!r}")
    match = _OP_RE.fullmatch(parts[0])
    if match is None:
        raise ValueError(f"unsupported statistics operation {spelling!r}")
    stores_index = match.group(1) is not None
    outer = Reduction(match.group(2))
    if stores_index and outer not in {Reduction.MAX, Reduction.MIN}:
        raise ValueError(f"arg prefix is invalid for {spelling!r}")
    digits = match.group(3)
    if digits and outer not in {Reduction.MAX, Reduction.MIN}:
        raise ValueError(f"top-k suffix is invalid for {spelling!r}")
    k = int(digits or "1")
    if k < 1:
        raise ValueError(f"top-k must be positive in {spelling!r}")
    try:
        inner = Reduction(parts[1]) if len(parts) == 2 else None
    except ValueError as error:
        raise ValueError(f"unsupported inner reduction in {spelling!r}") from error
    if inner is None and (stores_index or k > 1):
        raise ValueError(
            f"{spelling!r} requires an inner statistics window; "
            "use a compound operation such as argmax_mean or max3_last"
        )
    return StatisticOperation(spelling, outer, inner, k, stores_index)


class _ExpressionValidator(ast.NodeVisitor):
    """Reject semantics that cannot be rendered identically by every backend."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.dependencies: set[str] = set()

    def _unsupported(self, node: ast.AST) -> ValueError:
        return ValueError(
            f"unsupported statistics expression node in {self.source!r}: "
            f"{type(node).__name__}"
        )

    def generic_visit(self, node: ast.AST) -> None:
        raise self._unsupported(node)

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow),
        ):
            raise self._unsupported(node.op)
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            raise self._unsupported(node.op)
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise self._unsupported(node.op)
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> None:
        supported = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
        for operator in node.ops:
            if not isinstance(operator, supported):
                raise self._unsupported(operator)
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (bool, int, float)):
            raise self._unsupported(node)
        if isinstance(node.value, bool):
            return
        try:
            finite = math.isfinite(float(node.value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(
                f"statistics expression {self.source!r} contains a "
                "non-finite numeric constant"
            )
        if isinstance(node.value, int) and int(float(node.value)) != node.value:
            raise ValueError(
                f"statistics expression {self.source!r} contains an integer "
                "constant that cannot be represented exactly"
            )

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in {"pi", "M_PI", "True", "False"}:
            self.dependencies.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.dependencies.add(
            _field_reference(
                node, message="statistics expressions only support dotted field names"
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ValueError("statistics expression functions must use bare names")
        function = node.func.id
        try:
            arity = _FUNCTION_ARITIES[function]
        except KeyError as error:
            raise ValueError(f"unsupported expression function {function!r}") from error
        if node.keywords:
            raise ValueError(
                f"statistics expression function {function!r} does not "
                "accept keyword arguments"
            )
        if len(node.args) != arity:
            raise ValueError(
                f"statistics expression function {function!r} expects "
                f"{arity} arguments, got {len(node.args)}"
            )
        for argument in node.args:
            self.visit(argument)


def _parse_expression(source: str) -> tuple[str, ast.Expression]:
    normalized = source.strip()
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid statistics expression {source!r}") from exc
    return normalized, tree


def _compile_expression(
    source: str,
    tree: ast.Expression,
    known_fields: set[str],
) -> Expression:
    visitor = _ExpressionValidator(source)
    visitor.visit(tree)
    unknown = visitor.dependencies.difference(known_fields)
    if unknown:
        raise ValueError(
            f"statistics expression {source!r} references unknown fields: "
            f"{sorted(unknown)}"
        )
    return Expression(source, tree, tuple(sorted(visitor.dependencies)))


def _field_reference(
    node: ast.AST, *, message: str = "scatter index must be a field name"
) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise ValueError(message)
    parts.append(node.id)
    return ".".join(reversed(parts))


def parse_value_source(source: str, known_fields: set[str]) -> ValueSource:
    """Compile one virtual field expression into its canonical typed source."""
    normalized, tree = _parse_expression(source)
    body = tree.body
    if not (
        isinstance(body, ast.Call)
        and isinstance(body.func, ast.Name)
        and body.func.id.startswith("scatter_")
    ):
        return ExpressionSource(
            _compile_expression(normalized, tree, known_fields),
        )
    if (
        body.func.id not in {"scatter_sum", "scatter_mean"}
        or len(body.args) != 2
        or body.keywords
    ):
        raise ValueError(
            "scatter expression must be scatter_sum(value, index) or "
            "scatter_mean(value, index)"
        )
    index = _field_reference(body.args[1])
    if index not in known_fields:
        raise ValueError(f"scatter index {index!r} is not a registered field")
    value_source = ast.get_source_segment(normalized, body.args[0])
    if value_source is None:
        raise ValueError("scatter value expression has no source location")
    value_source, value_tree = _parse_expression(f"({value_source}\n)")
    return ScatterSource(
        Reduction(body.func.id.removeprefix("scatter_")),
        _compile_expression(value_source, value_tree, known_fields),
        index,
    )


def build_statistics_ir(
    program: StatisticsProgram,
    *,
    fields: Mapping[str, RuntimeTensorMetadata],
    layouts: Mapping[str, StatisticsVariableLayout],
    symbol_names: Mapping[str, str],
) -> StatisticsIR:

    variables: list[StatisticVariable] = []
    groups: dict[str, list[StatisticVariable]] = {}
    for name in sorted(program.operations):
        info = fields[name]
        metadata = info.tensor
        layout = layouts[name]
        group = info.output_index or "__full__"
        variable = StatisticVariable(
            name=name,
            safe_name=symbol_names[name],
            source=program.sources.get(name) or TensorSource(name),
            operations=program.operations[name],
            tensor_shape=metadata.shape,
            actual_shape=layout.actual_shape,
            actual_ndim=layout.actual_ndim,
            output_group=group,
        )
        variables.append(variable)
        groups.setdefault(group, []).append(variable)

    by_name = MappingProxyType({variable.name: variable for variable in variables})
    grouped = MappingProxyType({key: tuple(value) for key, value in groups.items()})
    return StatisticsIR(
        tuple(variables),
        by_name,
        grouped,
        program.sources,
    )

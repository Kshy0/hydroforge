"""Typed, backend-neutral statistics intermediate representation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from hydroforge.statistics.expression import parse_scatter_expr


class Reduction(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    FIRST = "first"
    LAST = "last"
    MID = "mid"


class ExpressionDialect(str, Enum):
    CUDA = "cuda"
    TRITON = "triton"
    METAL = "metal"
    TORCH = "torch"


class StorageInitialization(str, Enum):
    ZERO = "zero"
    NEGATIVE_INFINITY = "negative_infinity"
    POSITIVE_INFINITY = "positive_infinity"


class StorageDType(str, Enum):
    VALUE = "value"
    INDEX = "index"


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


ValueSource = TensorSource | ExpressionSource | ScatterSource


@dataclass(frozen=True)
class StatisticsProgram:
    """Shape-independent statistics semantics compiled before allocation."""

    operations: Mapping[str, tuple[StatisticOperation, ...]]
    sources: Mapping[str, ValueSource]

    def dependencies(self, name: str) -> tuple[str, ...]:
        source = self.sources.get(name, TensorSource(name))
        if isinstance(source, TensorSource):
            return (source.name,)
        expression = (
            source.expression if isinstance(source, ExpressionSource)
            else source.value
        )
        return expression.dependencies

    def leaf_tensors(self, name: str) -> tuple[str, ...]:
        """Return concrete model tensors required to evaluate ``name``."""
        leaves: set[str] = set()
        visiting: set[str] = set()

        def visit(field: str) -> None:
            if field in visiting:
                raise ValueError(
                    f"cyclic statistics virtual expression involving {field!r}"
                )
            source = self.sources.get(field, TensorSource(field))
            if isinstance(source, TensorSource):
                leaves.add(source.name)
                return
            visiting.add(field)
            if isinstance(source, ScatterSource):
                leaves.add(source.index)
                dependencies = source.value.dependencies
            else:
                dependencies = source.expression.dependencies
            for dependency in dependencies:
                visit(dependency)
            visiting.remove(field)

        visit(name)
        return tuple(sorted(leaves))


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
        slots.append(StorageSlot(
            name, actual_shape, StorageDType.VALUE, initialization, False,
        ))

    for operation in operations:
        shape = (
            actual_shape + (operation.k,)
            if operation.k > 1 else actual_shape
        )
        initialization = (
            StorageInitialization.NEGATIVE_INFINITY
            if operation.outer is Reduction.MAX
            else StorageInitialization.POSITIVE_INFINITY
            if operation.outer is Reduction.MIN
            else StorageInitialization.ZERO
        )
        dtype = (
            StorageDType.INDEX if operation.stores_index
            else StorageDType.VALUE
        )
        slots.append(StorageSlot(
            f"{variable}_{operation.spelling}", shape, dtype,
            StorageInitialization.ZERO if operation.stores_index
            else initialization,
            True,
        ))
        if operation.stores_index:
            suffix = "" if operation.k == 1 else str(operation.k)
            add_name = f"{variable}_{operation.outer.value}{suffix}_aux"
            if add_name not in internal_names:
                internal_names.add(add_name)
                slots.append(StorageSlot(
                    add_name, shape, StorageDType.VALUE, initialization, False,
                ))
        if operation.inner is None or operation.inner is Reduction.LAST:
            continue
        inner_initialization = (
            StorageInitialization.NEGATIVE_INFINITY
            if operation.inner is Reduction.MAX
            else StorageInitialization.POSITIVE_INFINITY
            if operation.inner is Reduction.MIN
            else StorageInitialization.ZERO
        )
        add_internal(
            f"{variable}_{operation.inner.value}_inner_state",
            inner_initialization,
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
        source = self.sources.get(name, TensorSource(name))
        if isinstance(source, TensorSource):
            return (source.name,)
        if isinstance(source, ScatterSource):
            return (f"__scatter_buf_{name}",)
        inputs = {
            leaf
            for dependency in source.expression.dependencies
            for leaf in self.materialized_inputs(dependency)
        }
        return tuple(sorted(inputs))

    def scatter_inputs(self, name: str) -> tuple[str, ...]:
        """Return source and index buffers for one scatter pre-kernel."""
        source = self.sources[name]
        if not isinstance(source, ScatterSource):
            raise TypeError(f"{name!r} is not a scatter value")
        inputs = {source.index}
        for dependency in source.value.dependencies:
            inputs.update(self.materialized_inputs(dependency))
        return tuple(sorted(inputs))

    def ordered_scatters(self) -> tuple[StatisticVariable, ...]:
        """Topologically order scatter materializations by virtual dependency."""
        result: list[StatisticVariable] = []
        visited: set[str] = set()

        def visit(variable: StatisticVariable) -> None:
            if variable.name in visited:
                return
            source = variable.source
            if not isinstance(source, ScatterSource):
                return
            for dependency in source.value.dependencies:
                dependency_variable = self.by_name.get(dependency)
                if dependency_variable is not None:
                    visit(dependency_variable)
            visited.add(variable.name)
            result.append(variable)

        for variable in self.variables:
            visit(variable)
        return tuple(result)


_OP_RE = re.compile(r"^(arg)?(mean|sum|max|min|first|last|mid)(\d*)$")
_INNER = frozenset({Reduction.MEAN, Reduction.SUM, Reduction.MAX,
                    Reduction.MIN, Reduction.FIRST, Reduction.LAST})


def parse_operation(spelling: str) -> StatisticOperation:
    parts = spelling.lower().split("_")
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
    inner = Reduction(parts[1]) if len(parts) == 2 else None
    if inner is not None and inner not in _INNER:
        raise ValueError(f"unsupported inner reduction in {spelling!r}")
    if inner is None and (stores_index or k > 1):
        raise ValueError(
            f"{spelling!r} requires an inner statistics window; "
            "use a compound operation such as argmax_mean or max3_last"
        )
    if inner is not None and outer is Reduction.MID:
        raise ValueError("mid is valid only as a standalone reduction")
    return StatisticOperation(spelling, outer, inner, k, stores_index)


class _DependencyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.dependencies: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in {"pi", "M_PI", "True", "False"}:
            self.dependencies.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            raise ValueError("statistics expressions only support dotted field names")
        parts.append(current.id)
        self.dependencies.add(".".join(reversed(parts)))


def parse_expression(source: str, known_fields: set[str]) -> Expression:
    normalized = source.strip().replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid statistics expression {source!r}") from exc
    visitor = _DependencyVisitor()
    visitor.visit(tree)
    unknown = visitor.dependencies.difference(known_fields)
    if unknown:
        raise ValueError(
            f"statistics expression {source!r} references unknown fields: "
            f"{sorted(unknown)}"
        )
    return Expression(normalized, tree, tuple(sorted(visitor.dependencies)))


def parse_value_source(source: str, known_fields: set[str]) -> ValueSource:
    """Compile one virtual field expression into its canonical typed source."""
    scatter = parse_scatter_expr(source)
    if scatter is None:
        return ExpressionSource(parse_expression(source, known_fields))
    if scatter.index_var not in known_fields:
        raise ValueError(
            f"scatter index {scatter.index_var!r} is not a registered field"
        )
    return ScatterSource(
        Reduction(scatter.mode),
        parse_expression(scatter.value_expr, known_fields),
        scatter.index_var,
    )


_FUNCTIONS: dict[ExpressionDialect, dict[str, str]] = {
    ExpressionDialect.CUDA: {
        "abs": "fabs", "fabs": "fabs", "sqrt": "sqrt", "exp": "exp",
        "log": "log", "sin": "sin", "cos": "cos", "tan": "tan",
        "pow": "pow", "maximum": "fmax", "minimum": "fmin",
        "max": "fmax", "min": "fmin",
    },
    ExpressionDialect.TRITON: {
        "abs": "tl.abs", "fabs": "tl.abs", "sqrt": "tl.sqrt",
        "exp": "tl.exp", "log": "tl.log", "sin": "tl.sin",
        "cos": "tl.cos", "tan": "libdevice.tan", "pow": "libdevice.pow",
        "maximum": "tl.maximum", "minimum": "tl.minimum",
        "max": "tl.maximum", "min": "tl.minimum",
    },
    ExpressionDialect.METAL: {
        "abs": "fabs", "fabs": "fabs", "sqrt": "sqrt", "exp": "exp",
        "log": "log", "sin": "sin", "cos": "cos", "tan": "tan",
        "pow": "pow", "maximum": "fmax", "minimum": "fmin",
        "max": "fmax", "min": "fmin",
    },
    ExpressionDialect.TORCH: {
        "abs": "torch.abs", "fabs": "torch.abs", "sqrt": "torch.sqrt",
        "exp": "torch.exp", "log": "torch.log", "sin": "torch.sin",
        "cos": "torch.cos", "tan": "torch.tan", "pow": "torch.pow",
        "maximum": "torch.maximum", "minimum": "torch.minimum",
        "max": "torch.maximum", "min": "torch.minimum",
    },
}


class _ExpressionRenderer:
    def __init__(
        self, dialect: ExpressionDialect, names: Mapping[str, str],
    ) -> None:
        self.dialect = dialect
        self.names = names

    def render(self, expression: Expression) -> str:
        return self.visit(expression.tree.body)

    def visit(self, node: ast.AST) -> str:
        if isinstance(node, ast.BinOp):
            left, right = self.visit(node.left), self.visit(node.right)
            operators = {
                ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                ast.Mod: "%",
            }
            symbol = operators.get(type(node.op))
            if symbol is not None:
                return f"({left} {symbol} {right})"
            if isinstance(node.op, ast.Pow):
                function = _FUNCTIONS[self.dialect]["pow"]
                return f"{function}({left}, {right})"
        if isinstance(node, ast.UnaryOp):
            value = self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                return f"(-{value})"
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.Not):
                if self.dialect in {
                    ExpressionDialect.TRITON, ExpressionDialect.TORCH,
                }:
                    return f"(~({value}))"
                return f"(!({value}))"
        if isinstance(node, ast.BoolOp):
            if self.dialect in {
                ExpressionDialect.TRITON, ExpressionDialect.TORCH,
            }:
                symbol = "&" if isinstance(node.op, ast.And) else "|"
            else:
                symbol = "&&" if isinstance(node.op, ast.And) else "||"
            return f"({f' {symbol} '.join(self.visit(value) for value in node.values)})"
        if isinstance(node, ast.Compare):
            symbols = {
                ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
                ast.Eq: "==", ast.NotEq: "!=",
            }
            left = self.visit(node.left)
            pieces = []
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                right = self.visit(comparator)
                try:
                    symbol = symbols[type(operator)]
                except KeyError as exc:
                    raise ValueError(
                        f"unsupported comparison {type(operator).__name__}"
                    ) from exc
                pieces.append(f"({left} {symbol} {right})")
                left = right
            conjunction = " & " if self.dialect in {
                ExpressionDialect.TRITON, ExpressionDialect.TORCH,
            } else " && "
            return f"({conjunction.join(pieces)})"
        if isinstance(node, ast.IfExp):
            condition = self.visit(node.test)
            positive, negative = self.visit(node.body), self.visit(node.orelse)
            if self.dialect is ExpressionDialect.TRITON:
                return f"tl.where({condition}, {positive}, {negative})"
            if self.dialect is ExpressionDialect.TORCH:
                return f"torch.where({condition}, {positive}, {negative})"
            return f"(({condition}) ? ({positive}) : ({negative}))"
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
            if isinstance(node.value, bool):
                if self.dialect in {
                    ExpressionDialect.TRITON, ExpressionDialect.TORCH,
                }:
                    return "True" if node.value else "False"
                return "true" if node.value else "false"
            return repr(float(node.value))
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = self._name(node)
            if name in {"pi", "M_PI"}:
                return "torch.pi" if self.dialect is ExpressionDialect.TORCH else "M_PI"
            try:
                return self.names[name]
            except KeyError as exc:
                raise ValueError(f"unbound expression field {name!r}") from exc
        if isinstance(node, ast.Call):
            function = self._name(node.func).split(".")[-1]
            arguments = [self.visit(argument) for argument in node.args]
            if function == "where":
                if len(arguments) != 3:
                    raise ValueError("where expects three arguments")
                condition, positive, negative = arguments
                if self.dialect is ExpressionDialect.TRITON:
                    return f"tl.where({condition}, {positive}, {negative})"
                if self.dialect is ExpressionDialect.TORCH:
                    return f"torch.where({condition}, {positive}, {negative})"
                return f"(({condition}) ? ({positive}) : ({negative}))"
            try:
                rendered = _FUNCTIONS[self.dialect][function]
            except KeyError as exc:
                raise ValueError(f"unsupported expression function {function!r}") from exc
            return f"{rendered}({', '.join(arguments)})"
        raise ValueError(f"unsupported statistics expression node {ast.dump(node)}")

    @staticmethod
    def _name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        raise ValueError("expression callable/field must be a name")


def render_expression(
    expression: Expression,
    dialect: ExpressionDialect,
    names: Mapping[str, str],
) -> str:
    """Lower one validated expression; only syntax varies by dialect."""
    return _ExpressionRenderer(dialect, names).render(expression)


def compile_statistics_program(
    aggregator: Any,
    variable_ops: Mapping[str, list[str] | tuple[str, ...]],
) -> StatisticsProgram:
    """Parse all operations and virtual sources exactly once."""
    known = set(aggregator._field_registry) | set(aggregator._tensor_registry)
    resolving: set[str] = set()
    resolved: dict[str, ValueSource] = {}

    def resolve_source(name: str) -> ValueSource:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"cyclic statistics virtual expression involving {name!r}")
        resolving.add(name)
        info = aggregator._field_registry.get(name)
        metadata = None if info is None else info.tensor
        expr = (
            metadata.expression
            if metadata is not None and metadata.category == "virtual" else ""
        )
        if not expr:
            source: ValueSource = TensorSource(name)
        else:
            source = parse_value_source(expr, known)
            dependencies = (
                source.expression.dependencies
                if isinstance(source, ExpressionSource)
                else source.value.dependencies
                if isinstance(source, ScatterSource)
                else ()
            )
            for dependency in dependencies:
                dep_info = aggregator._field_registry.get(dependency)
                if dep_info is not None and dep_info.tensor.category == "virtual":
                    resolve_source(dependency)
        resolving.remove(name)
        resolved[name] = source
        return source

    for name in variable_ops:
        resolve_source(name)
    operations = MappingProxyType({
        name: tuple(parse_operation(operation) for operation in values)
        for name, values in variable_ops.items()
    })
    return StatisticsProgram(
        operations=operations,
        sources=MappingProxyType(dict(resolved)),
    )


def build_statistics_ir(aggregator: Any) -> StatisticsIR:
    program = getattr(aggregator, "_statistics_program", None)
    if program is None:
        program = compile_statistics_program(aggregator, aggregator._variable_ops)

    variables: list[StatisticVariable] = []
    groups: dict[str, list[StatisticVariable]] = {}
    for name in sorted(aggregator._variables):
        info = aggregator._field_registry[name]
        metadata = info.tensor
        tensor = aggregator._tensor_registry.get(name)
        if tensor is None:
            first_output = f"{name}_{aggregator._variable_ops[name][0]}"
            output_metadata = aggregator._metadata[first_output]
            actual_shape = tuple(output_metadata["actual_shape"])
            actual_ndim = int(output_metadata["actual_ndim"])
        else:
            actual_shape = tuple(int(size) for size in tensor.shape)
            actual_ndim = int(tensor.ndim)
        group = info.output_index or "__full__"
        variable = StatisticVariable(
            name=name,
            safe_name=aggregator._get_safe_name(name),
            source=program.sources.get(name, TensorSource(name)),
            operations=program.operations[name],
            tensor_shape=metadata.shape,
            actual_shape=actual_shape,
            actual_ndim=actual_ndim,
            output_group=group,
        )
        variables.append(variable)
        groups.setdefault(group, []).append(variable)

    by_name = MappingProxyType({variable.name: variable for variable in variables})
    grouped = MappingProxyType({key: tuple(value) for key, value in groups.items()})
    return StatisticsIR(
        tuple(variables), by_name, grouped, program.sources,
    )

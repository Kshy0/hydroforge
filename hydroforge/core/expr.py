# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Backend-neutral expression language for virtual fields and statistics.

Expressions are parsed from Python syntax into a validated AST with their
field dependencies; value sources and statistic operations are the typed
forms that contracts, compilers and code generators consume.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from functools import lru_cache
from typing import Any

import torch


class Reduction(StrEnum):
    __str__ = Enum.__str__
    __format__ = Enum.__format__

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    FIRST = "first"
    LAST = "last"


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
class FieldView:
    """A field read at other elements than the expression's own.

    ``field[1:-1, 1:-1]`` slices its leading axes (``slices`` holds each
    sliced axis's ``(start, stop, step)``, Python bounds; trailing axes are
    whole). ``take(field, index)`` gathers the field's elements at the flat
    row-major positions an integer ``index`` field holds; the result has the
    index's shape. ``key`` is the canonical spelling the lowering binds.
    """

    key: str
    field: str
    slices: tuple[tuple[int | None, int | None, int], ...] = ()
    index: str | None = None


@dataclass(frozen=True)
class Expression:
    """Validated Python expression AST and its model-field dependencies.

    ``dependencies`` names every field it reads; ``operands`` those read at
    the expression's own elements, ``views`` those read elsewhere.
    """

    source: str
    tree: ast.Expression
    dependencies: tuple[str, ...]
    operands: tuple[str, ...] = ()
    views: tuple[FieldView, ...] = ()
    # ``options.<path>`` constants, bound to the model's option values by
    # the compiler (bind_options) before any backend lowers the expression.
    options: tuple[str, ...] = ()


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


@lru_cache(maxsize=256)
def parse_operation(spelling: str) -> StatisticOperation:
    """Parse one statistics operation; ``.spelling`` is its canonical form."""

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
    # One canonical spelling per operation: ``max1_mean`` and ``max01_mean``
    # are ``max_mean``, so equivalent requests cannot become two outputs.
    canonical = (
        f"{'arg' if stores_index else ''}{outer.value}{k if k > 1 else ''}"
        f"{'' if inner is None else '_' + inner.value}"
    )
    return StatisticOperation(canonical, outer, inner, k, stores_index)


def _bound(node: ast.AST | None, what: str) -> int | None:
    """An omitted or integer-literal slice bound."""

    if node is None:
        return None
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign, node = -1, node.operand
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        raise ValueError(f"slice {what} must be an integer literal")
    return sign * node.value


def field_view(node: ast.AST) -> FieldView | None:
    """The view a ``field[...]`` or ``take(field, index)`` node reads
    (``None`` for any other node)."""

    if isinstance(node, ast.Subscript):
        field = field_reference(
            node.value, message="only a field name can be sliced: field[1:-1, ...]"
        )
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        slices = []
        for item in items:
            if not isinstance(item, ast.Slice):
                raise ValueError(
                    "field subscripts are slices (field[1:-1]); sample points "
                    "with take(field, index)"
                )
            step = _bound(item.step, "step")
            if step is not None and step < 1:
                raise ValueError("slice step must be a positive integer")
            slices.append(
                (_bound(item.lower, "start"), _bound(item.upper, "stop"), step or 1)
            )
        text = ",".join(
            f"{'' if start is None else start}:{'' if stop is None else stop}"
            + ("" if step == 1 else f":{step}")
            for start, stop, step in slices
        )
        return FieldView(f"{field}[{text}]", field, slices=tuple(slices))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "take"
    ):
        if len(node.args) != 2 or node.keywords:
            raise ValueError("take expects take(field, index)")
        message = "take(field, index) takes two field names"
        field = field_reference(node.args[0], message=message)
        index = field_reference(node.args[1], message=message)
        return FieldView(f"take({field},{index})", field, index=index)
    return None


class _ExpressionValidator(ast.NodeVisitor):
    """Reject semantics that cannot be rendered identically by every backend."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.dependencies: set[str] = set()
        self.operands: set[str] = set()
        self.views: dict[str, FieldView] = {}
        self.options: set[str] = set()

    def _view(self, node: ast.AST) -> None:
        view = field_view(node)
        self.views[view.key] = view
        self.dependencies.add(view.field)
        if view.index is not None:
            self.dependencies.add(view.index)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._view(node)

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
            self.operands.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = field_reference(
            node, message="statistics expressions only support dotted field names"
        )
        if name.startswith(OPTIONS_PREFIX):
            self.options.add(name)
            return
        self.dependencies.add(name)
        self.operands.add(name)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ValueError("statistics expression functions must use bare names")
        function = node.func.id
        if function == "take":
            self._view(node)
            return
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
    known_fields: set[str] | None,
) -> Expression:
    visitor = _ExpressionValidator(source)
    visitor.visit(tree)
    unknown = (
        visitor.dependencies.difference(known_fields)
        if known_fields is not None
        else set()
    )
    if unknown:
        raise ValueError(
            f"statistics expression {source!r} references unknown fields: "
            f"{sorted(unknown)}"
        )
    return Expression(
        source,
        tree,
        tuple(sorted(visitor.dependencies)),
        tuple(sorted(visitor.operands)),
        tuple(visitor.views[key] for key in sorted(visitor.views)),
        tuple(sorted(visitor.options)),
    )


OPTIONS_PREFIX = "options."


def bind_options(expression: Expression, values: Any) -> Expression:
    """``expression`` with every ``options.<path>`` replaced by its value.

    ``values`` resolves a path (``OptionsConfig.value``); an option an
    expression reads must hold a bool, int or float.
    """

    if not expression.options:
        return expression

    class _Bind(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            name = field_reference(node, message="")
            if not name.startswith(OPTIONS_PREFIX):
                return node
            value = values(name.removeprefix(OPTIONS_PREFIX))
            if type(value) not in (bool, int, float):
                raise ValueError(
                    f"expression {expression.source!r} reads {name}, which is "
                    f"not a number: {value!r}"
                )
            return ast.copy_location(ast.Constant(value), node)

    tree = ast.fix_missing_locations(_Bind().visit(copy.deepcopy(expression.tree)))
    return _compile_expression(ast.unparse(tree), tree, None)


def field_reference(
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


def parse_value_source(
    source: str, known_fields: set[str] | None = None
) -> ValueSource:
    parsed = _value_source_syntax(source)
    dependencies = (
        set(parsed.value.dependencies) | {parsed.index}
        if isinstance(parsed, ScatterSource)
        else set(parsed.expression.dependencies)
    )
    if known_fields is not None and (unknown := dependencies.difference(known_fields)):
        raise ValueError(
            f"statistics expression {source!r} references unknown fields: {sorted(unknown)}"
        )
    return parsed


@lru_cache(maxsize=512)
def _value_source_syntax(source: str) -> ValueSource:
    """Compile one virtual field expression into its canonical typed source."""
    normalized, tree = _parse_expression(source)
    body = tree.body
    if not (
        isinstance(body, ast.Call)
        and isinstance(body.func, ast.Name)
        and body.func.id.startswith("scatter_")
    ):
        return ExpressionSource(
            _compile_expression(normalized, tree, None),
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
    index = field_reference(body.args[1])
    value_source = ast.get_source_segment(normalized, body.args[0])
    if value_source is None:
        raise ValueError("scatter value expression has no source location")
    value_source, value_tree = _parse_expression(f"({value_source}\n)")
    return ScatterSource(
        Reduction(body.func.id.removeprefix("scatter_")),
        _compile_expression(value_source, value_tree, None),
        index,
    )

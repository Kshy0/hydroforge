"""Validated expressions (:mod:`hydroforge.core.expr`) lowered to kernel IR.

Every dialect evaluates an expression in one value type: names, operands
and the result are converted to it, numeric literals are typed with it, and
the truth of a number is ``value != 0``.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping

import torch

from hydroforge.core.expr import Expression, field_reference
from hydroforge.kernels.codegen.ir import (
    Binary,
    Call,
    Cast,
    Compare,
    Const,
    Expr,
    Logical,
    Select,
    Unary,
)

_ARITHMETIC = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_COMPARISONS = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}
_FUNCTIONS = {
    "abs": "abs",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "pow": "pow",
    "maximum": "nan_max",
    "minimum": "nan_min",
}


class _Lowering:
    def __init__(self, names: Mapping[str, Expr], dtype: torch.dtype) -> None:
        self.names = names
        self.dtype = dtype

    def operand(self, node: ast.AST) -> Expr:
        return Cast(self.visit(node), self.dtype)

    def truth(self, node: ast.AST) -> Expr:
        return Compare("!=", self.visit(node), Const(0.0))

    def visit(self, node: ast.AST) -> Expr:
        match node:
            case ast.BinOp(op=ast.Mod() | ast.Pow()):
                function = "py_mod" if isinstance(node.op, ast.Mod) else "pow"
                arguments = (self.operand(node.left), self.operand(node.right))
                return Call(function, arguments, self.dtype)
            case ast.BinOp():
                return Binary(
                    _ARITHMETIC[type(node.op)],
                    self.operand(node.left),
                    self.operand(node.right),
                )
            case ast.UnaryOp(op=ast.USub()):
                return Unary("neg", self.operand(node.operand))
            case ast.UnaryOp(op=ast.UAdd()):
                return self.operand(node.operand)
            case ast.UnaryOp(op=ast.Not()):
                return Compare("==", self.truth(node.operand), Const(0))
            case ast.BoolOp():
                op = "and" if isinstance(node.op, ast.And) else "or"
                return Logical(op, tuple(self.truth(value) for value in node.values))
            case ast.Compare():
                pieces = []
                left = self.visit(node.left)
                for operator, comparator in zip(
                    node.ops, node.comparators, strict=True
                ):
                    right = self.visit(comparator)
                    pieces.append(Compare(_COMPARISONS[type(operator)], left, right))
                    left = right
                return pieces[0] if len(pieces) == 1 else Logical("and", tuple(pieces))
            case ast.IfExp():
                return self.select(node.test, node.body, node.orelse)
            case ast.Constant(value=bool()):
                return Const(node.value, torch.bool)
            case ast.Constant():
                return Const(float(node.value), self.dtype)
            case ast.Call(func=ast.Name(id="where")):
                return self.select(*node.args)
            case ast.Call():
                return Call(
                    _FUNCTIONS[node.func.id],
                    tuple(self.operand(argument) for argument in node.args),
                    self.dtype,
                )
        name = field_reference(node)
        if name in {"pi", "M_PI"}:
            # pi is a literal of the value type: a C ``M_PI`` double would
            # promote a float32 expression.
            return Const(math.pi, self.dtype)
        return Cast(self.names[name], self.dtype)

    def select(self, test: ast.AST, body: ast.AST, orelse: ast.AST) -> Expr:
        return Select(self.truth(test), self.visit(body), self.visit(orelse))


def lower_expression(
    expression: Expression, names: Mapping[str, Expr], dtype: torch.dtype
) -> Expr:
    """Lower ``expression`` evaluated in ``dtype``; ``names`` bind its fields."""

    return Cast(_Lowering(names, dtype).visit(expression.tree.body), dtype)

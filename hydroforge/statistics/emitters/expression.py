"""Backend syntax rendering for validated statistics expressions."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from enum import Enum, StrEnum

from hydroforge.statistics.ir import Expression, _field_reference


class ExpressionDialect(StrEnum):
    __str__ = Enum.__str__
    __format__ = Enum.__format__

    CUDA = "cuda"
    TRITON = "triton"
    METAL = "metal"
    TORCH = "torch"


_FUNCTIONS: dict[ExpressionDialect, dict[str, str]] = {
    ExpressionDialect.CUDA: {
        "abs": "fabs",
        "sqrt": "sqrt",
        "exp": "exp",
        "log": "log",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "pow": "pow",
        "maximum": "hf_max",
        "minimum": "hf_min",
    },
    ExpressionDialect.TRITON: {
        "abs": "tl.abs",
        "sqrt": "tl.sqrt",
        "exp": "tl.exp",
        "log": "tl.log",
        "sin": "tl.sin",
        "cos": "tl.cos",
        "tan": "libdevice.tan",
        "pow": "libdevice.pow",
        "maximum": "hydroforge_maximum",
        "minimum": "hydroforge_minimum",
    },
    ExpressionDialect.METAL: {
        "abs": "fabs",
        "sqrt": "sqrt",
        "exp": "exp",
        "log": "log",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "pow": "pow",
        "maximum": "hydroforge_maximum",
        "minimum": "hydroforge_minimum",
    },
    ExpressionDialect.TORCH: {
        "abs": "torch.abs",
        "sqrt": "torch.sqrt",
        "exp": "torch.exp",
        "log": "torch.log",
        "sin": "torch.sin",
        "cos": "torch.cos",
        "tan": "torch.tan",
        "pow": "torch.pow",
        "maximum": "hydroforge_maximum",
        "minimum": "hydroforge_minimum",
    },
}


class _ExpressionRenderer:
    def __init__(
        self,
        dialect: ExpressionDialect,
        names: Mapping[str, str],
        value_type: str | None,
    ) -> None:
        self.dialect = dialect
        self.names = names
        self.value_type = value_type

    def render(self, expression: Expression) -> str:
        rendered = self.visit(expression.tree.body)
        if self.dialect is ExpressionDialect.TORCH:
            return self._torch_tensor(rendered)
        return self._cast_tensor(rendered)

    def _torch_tensor(self, value: str) -> str:
        reference = next(iter(self.names.values()), None)
        arguments = [value]
        if self.value_type is not None:
            arguments.append(f"dtype=torch.{self.value_type}")
        elif reference is not None:
            arguments.append(f"dtype=({reference}).dtype")
        if reference is not None:
            arguments.append(f"device=({reference}).device")
        return f"torch.as_tensor({', '.join(arguments)})"

    def _cast_tensor(self, value: str) -> str:
        if self.value_type is None:
            return value
        native = {
            "float32": {
                ExpressionDialect.CUDA: "float",
                ExpressionDialect.METAL: "float",
                ExpressionDialect.TRITON: "tl.float32",
                ExpressionDialect.TORCH: "torch.float32",
            },
            "float64": {
                ExpressionDialect.CUDA: "double",
                ExpressionDialect.TRITON: "tl.float64",
                ExpressionDialect.TORCH: "torch.float64",
            },
        }[self.value_type][self.dialect]
        if self.dialect is ExpressionDialect.CUDA:
            return f"static_cast<{native}>({value})"
        if self.dialect is ExpressionDialect.METAL:
            return f"{native}({value})"
        if self.dialect is ExpressionDialect.TRITON:
            return f"tl.cast({value}, {native})"
        return f"({value}).to({native})"

    def _numeric_constant(self, value: int | float) -> str:
        rendered = repr(float(value))
        if self.dialect is ExpressionDialect.TORCH:
            return self._torch_tensor(rendered)
        if self.value_type is None:
            return rendered
        if self.dialect is ExpressionDialect.TRITON:
            return f"tl.full((), {rendered}, tl.{self.value_type})"
        if self.dialect in {
            ExpressionDialect.CUDA,
            ExpressionDialect.METAL,
        }:
            return self._cast_tensor(rendered)
        return rendered

    def _truth(self, node: ast.AST) -> str:
        """Render the backend-neutral numeric truth-value contract."""

        return f"(({self.visit(node)}) != 0.0)"

    def _numeric_operand(self, node: ast.AST) -> str:
        rendered = self.visit(node)
        if self.dialect is ExpressionDialect.TORCH:
            return self._torch_tensor(rendered)
        return self._cast_tensor(rendered)

    def _conditional(self, test: ast.AST, body: ast.AST, orelse: ast.AST) -> str:
        condition = self._truth(test)
        positive, negative = self.visit(body), self.visit(orelse)
        if self.dialect is ExpressionDialect.TRITON:
            return f"tl.where({condition}, {positive}, {negative})"
        if self.dialect is ExpressionDialect.TORCH:
            return f"hydroforge_where({condition}, {positive}, {negative})"
        return f"(({condition}) ? ({positive}) : ({negative}))"

    def visit(self, node: ast.AST) -> str:
        if isinstance(node, ast.BinOp):
            left = self._numeric_operand(node.left)
            right = self._numeric_operand(node.right)
            if isinstance(node.op, ast.Mod):
                if self.dialect is ExpressionDialect.TORCH:
                    return f"hydroforge_remainder({left}, {right})"
                if self.dialect is ExpressionDialect.TRITON:
                    left, right = self._triton_promote_binary(left, right)
                    remainder = f"libdevice.fmod({left}, {right})"
                    adjust = (
                        f"(({remainder} != 0.0) & "
                        f"(({remainder} < 0.0) != ({right} < 0.0)))"
                    )
                    adjusted = f"tl.where({adjust}, {remainder} + {right}, {remainder})"
                    return adjusted
                remainder = f"fmod({left}, {right})"
                adjust = (
                    f"(({remainder} != 0.0) && "
                    f"(({remainder} < 0.0) != ({right} < 0.0)))"
                )
                adjusted = f"(({adjust}) ? ({remainder} + {right}) : ({remainder}))"
                return adjusted
            operators = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }
            symbol = operators.get(type(node.op))
            if symbol is not None:
                return f"({left} {symbol} {right})"
            if isinstance(node.op, ast.Pow):
                function = _FUNCTIONS[self.dialect]["pow"]
                if self.dialect is ExpressionDialect.TRITON:
                    left, right = self._triton_promote_binary(left, right)
                return f"{function}({left}, {right})"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return f"(-{self._numeric_operand(node.operand)})"
            if isinstance(node.op, ast.UAdd):
                return self._numeric_operand(node.operand)
            if isinstance(node.op, ast.Not):
                return f"({self._truth(node.operand)} == 0)"
        if isinstance(node, ast.BoolOp):
            if self.dialect in {
                ExpressionDialect.TRITON,
                ExpressionDialect.TORCH,
            }:
                symbol = "&" if isinstance(node.op, ast.And) else "|"
            else:
                symbol = "&&" if isinstance(node.op, ast.And) else "||"
            return (
                f"({f' {symbol} '.join(self._truth(value) for value in node.values)})"
            )
        if isinstance(node, ast.Compare):
            symbols = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            left = self.visit(node.left)
            pieces = []
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                right = self.visit(comparator)
                symbol = symbols[type(operator)]
                pieces.append(f"({left} {symbol} {right})")
                left = right
            conjunction = (
                " & "
                if self.dialect
                in {
                    ExpressionDialect.TRITON,
                    ExpressionDialect.TORCH,
                }
                else " && "
            )
            return f"({conjunction.join(pieces)})"
        if isinstance(node, ast.IfExp):
            return self._conditional(node.test, node.body, node.orelse)
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (bool, int, float)
        ):
            if isinstance(node.value, bool):
                if self.dialect in {
                    ExpressionDialect.TRITON,
                    ExpressionDialect.TORCH,
                }:
                    return "True" if node.value else "False"
                return "true" if node.value else "false"
            return self._numeric_constant(node.value)
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = _field_reference(node)
            if name in {"pi", "M_PI"}:
                # Treat pi exactly like every other numeric literal.  Leaving
                # M_PI as a double in CUDA/Metal promotes an otherwise float32
                # expression to double, while Torch/Triton evaluate it in
                # float32, producing backend-dependent results.
                return self._numeric_constant(math.pi)
            return self._cast_tensor(self.names[name])
        if isinstance(node, ast.Call):
            function = node.func.id
            if function == "where":
                return self._conditional(*node.args)
            arguments = [self._numeric_operand(argument) for argument in node.args]
            rendered = _FUNCTIONS[self.dialect][function]
            if self.dialect is ExpressionDialect.TRITON and function == "pow":
                arguments = list(self._triton_promote_binary(*arguments))
            return f"{rendered}({', '.join(arguments)})"
        return ""

    def _triton_promote_binary(self, left: str, right: str) -> tuple[str, str]:
        """Unify libdevice operand types without evaluating extra arithmetic."""
        dtype = (
            f"tl.{self.value_type}"
            if self.value_type is not None
            else f"(tl.where(True, {left}, {right})).dtype"
        )
        return (
            f"tl.cast({left}, {dtype})",
            f"tl.cast({right}, {dtype})",
        )


def render_expression(
    expression: Expression,
    dialect: ExpressionDialect,
    names: Mapping[str, str],
    *,
    value_type: str | None = None,
) -> str:
    """Lower one validated expression; only syntax varies by dialect."""
    return _ExpressionRenderer(
        dialect,
        names,
        value_type,
    ).render(expression)

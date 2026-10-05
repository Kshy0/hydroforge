"""Triton spelling of kernel IR.

A Triton kernel runs a block of lanes per program: ``ThreadIndex`` is the
program's block of thread indices.  Values that depend on it are per-lane
tiles; the rest (loaded controls, phase tests, parameters, unrolled loop
counters) are uniform scalars.  Uniform ``If`` conditions are branches and a
uniform ``Guard`` opens a branch around the rest of the kernel; a per-lane
``Guard`` narrows the mask of every later load and store, and a per-lane
``If`` narrows them within its branches, where assignments select with
``tl.where``.  A ``While`` loops on a uniform condition outside per-lane
branches.  Loads of per-lane addresses always carry the mask in effect,
so both operands of ``tl.where`` stay in bounds.

FP32 division and square root are IEEE (``tl.div_rn``, ``tl.sqrt_rn``; ROCm
uses OCML's IEEE ``sqrt``), and transcendental functions call the libdevice
functions CUDA uses: Triton's defaults are approximate instructions.  The
prelude's helpers are the NaN rules: ``hydroforge_maximum``/``_minimum``
ignore one NaN operand and ``hydroforge_weighted_mean`` is the incremental
mean with the blended fallback for non-finite results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from hydroforge.kernels.codegen.c import identifier
from hydroforge.kernels.codegen.ir import (
    PHASE,
    Assign,
    AtomicAdd,
    Binary,
    Block,
    Call,
    Cast,
    Compare,
    Const,
    Expr,
    ForK,
    Guard,
    If,
    KernelFunction,
    Let,
    Load,
    Logical,
    Names,
    PhaseTest,
    Select,
    Stmt,
    Store,
    ThreadIndex,
    TileIndex,
    Unary,
    Var,
    While,
    type_of,
)
from hydroforge.kernels.codegen.passes import effects, subexpressions

_INDENT = "    "

PRELUDE = """\
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Ignore one-sided NaN and select like the C printers (``a > b ? a : b``).
# ``tl.maximum``/``tl.minimum`` lower float64 to ``max.f64``/``min.f64``,
# whose two halves ptxas may select by different predicates.
@triton.jit
def hydroforge_maximum(left, right):
    return tl.where(left != left, right, tl.where(right != right, left, tl.where(left > right, left, right)))


@triton.jit
def hydroforge_minimum(left, right):
    return tl.where(left != left, right, tl.where(right != right, left, tl.where(left < right, left, right)))


@triton.jit
def hydroforge_divide(numerator, denominator):
    # FP32 ``/`` is approximate in Triton; statistics divide exactly.
    if denominator.dtype == tl.float32:
        return tl.div_rn(numerator, denominator)
    else:
        return numerator / denominator


@triton.jit
def hydroforge_weighted_mean(old_value, old_weight, value, weight):
    # Incremental form bounds FP32 drift; non-finite results keep the
    # blended form's infinity/NaN propagation.
    new_weight = old_weight + weight
    ratio = hydroforge_divide(weight, new_weight)
    incremental = old_value + (value - old_value) * ratio
    blended = old_value * hydroforge_divide(old_weight, new_weight) + value * ratio
    return tl.where(tl.abs(incremental) < float('inf'), incremental, blended)
"""

# Names the prelude and kernel signatures bind; no parameter or local may
# take them.
_GLOBALS = frozenset(
    (
        "triton",
        "tl",
        "libdevice",
        "BLOCK_SIZE",
        "hydroforge_maximum",
        "hydroforge_minimum",
        "hydroforge_divide",
        "hydroforge_weighted_mean",
    )
)

_FUNCTIONS = {
    "abs": "tl.abs",
    "sqrt": "tl.sqrt",
    "exp": "libdevice.exp",
    "log": "libdevice.log",
    "sin": "libdevice.sin",
    "cos": "libdevice.cos",
    "tan": "libdevice.tan",
    "pow": "libdevice.pow",
    "nan_max": "hydroforge_maximum",
    "nan_min": "hydroforge_minimum",
    "weighted_mean": "hydroforge_weighted_mean",
}

_LANES = "tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)"


def _type(dtype: torch.dtype) -> str:
    names = {
        torch.bool: "int1",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.int16: "int16",
        torch.uint16: "uint16",
        torch.int32: "int32",
        torch.uint32: "uint32",
        torch.int64: "int64",
        torch.uint64: "uint64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    return f"tl.{names[dtype]}"


def _integral(node: Expr) -> bool:
    dtype = type_of(node)
    return dtype is not None and not dtype.is_floating_point


def _literal(value: bool | int | float) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return f"float('{value}')"
    return repr(value)


class TritonPrinter:
    """Spell kernel IR expressions; :meth:`kernel` spells whole kernels."""

    def expr(self, node: Expr) -> str:
        match node:
            case Var():
                return node.name
            case Const(value=bool()) | Const(type=None):
                return _literal(node.value)
            case Const() if not node.type.is_floating_point:
                # Integers take the width of the lanes or offsets they meet.
                return _literal(node.value)
            case Const():
                return f"tl.full((), {_literal(node.value)}, {_type(node.type)})"
            case Load():
                return self.load(node)
            case Unary(op="neg"):
                return f"(-{self.expr(node.operand)})"
            case Unary():
                return f"({self.expr(node.operand)} == 0)"
            case Binary(op="/") if type_of(node) == torch.float32:
                return f"tl.div_rn({self.expr(node.left)}, {self.expr(node.right)})"
            case Binary(op="/") if _integral(node):
                # Integer division truncates, as in C, on nonnegative operands.
                return f"({self.expr(node.left)} // {self.expr(node.right)})"
            case Binary() | Compare():
                return f"({self.expr(node.left)} {node.op} {self.expr(node.right)})"
            case Logical():
                symbol = " & " if node.op == "and" else " | "
                return f"({symbol.join(self.expr(item) if type_of(item) == torch.bool else f'({self.expr(item)} != 0)' for item in node.operands)})"
            case Select():
                return (
                    f"tl.where({self.expr(node.condition)}, "
                    f"{self.expr(node.positive)}, {self.expr(node.negative)})"
                )
            case Cast():
                return f"tl.cast({self.expr(node.operand)}, {_type(node.type)})"
            case Call(function="py_mod" | "pow"):
                # libdevice takes operands of one type.
                left, right = (
                    f"tl.cast({self.expr(item)}, {_type(node.type)})"
                    for item in node.arguments
                )
                if node.function == "pow":
                    return f"libdevice.pow({left}, {right})"
                remainder = f"libdevice.fmod({left}, {right})"
                adjust = (
                    f"(({remainder} != 0.0) & (({remainder} < 0.0) != ({right} < 0.0)))"
                )
                return f"tl.where({adjust}, {remainder} + {right}, {remainder})"
            case Call(function="isnan"):
                (operand,) = (self.expr(item) for item in node.arguments)
                return f"({operand} != {operand})"
            case Call():
                function = _FUNCTIONS[node.function]
                if node.function == "sqrt" and node.type == torch.float32:
                    # ROCm Triton's sqrt_rn returns NaN for subnormal inputs.
                    function = "libdevice.sqrt" if torch.version.hip else "tl.sqrt_rn"
                arguments = ", ".join(self.expr(item) for item in node.arguments)
                return f"{function}({arguments})"
            case PhaseTest():
                return f"(({PHASE.name} & {int(node.bits)}) != 0)"
            case ThreadIndex():
                return f"({_LANES})"
            case TileIndex(axis=0):
                return f"({_LANES})[:, None]"
            case TileIndex():
                return f"tl.arange(0, {node.extent})[None, :]"
        raise TypeError(f"not a Triton expression: {node!r}")

    def load(self, node: Load) -> str:
        raise TypeError(f"a load outside a kernel: {node!r}")

    def kernel(self, function: KernelFunction) -> str:
        """``@triton.jit`` source of ``function``; scalars are constexpr."""

        return "\n".join(_Kernel(function).lines())

    def program(self, functions: Sequence[KernelFunction]) -> str:
        """The prelude and every kernel of one module."""

        return (
            "\n\n\n".join((PRELUDE.rstrip("\n"), *map(self.kernel, functions))) + "\n"
        )


class _Kernel(TritonPrinter):
    """The printing state of one kernel: masks, predicates, lane values."""

    def __init__(self, function: KernelFunction) -> None:
        self.function = function
        params = [identifier(param) for param in function.params]
        touched = effects(function.body)
        taken = touched.uses | touched.defines
        clashes = (set(params) & touched.defines) | ((set(params) | taken) & _GLOBALS)
        if len(set(params)) != len(params) or clashes:
            raise ValueError(
                f"{function.name}: parameter names collide: {params}; "
                f"names taken by parameters or the prelude: {sorted(clashes)}"
            )
        self.names = Names((*params, *taken, *_GLOBALS))
        self.buffers = {param.name: identifier(param) for param in function.params}
        self.varying: set[str] = set()
        self.mask: str | None = None
        self.predicate: str | None = None

    def lines(self) -> list[str]:
        params = [
            identifier(param)
            if param.access is not None
            else f"{identifier(param)}: tl.constexpr"
            for param in self.function.params
        ]
        return [
            "@triton.jit",
            f"def {self.function.name}(",
            *(f"{_INDENT}{param}," for param in params),
            f"{_INDENT}BLOCK_SIZE: tl.constexpr,",
            "):",
            *(self.stmts(self.function.body, 1, top=True) or [f"{_INDENT}pass"]),
        ]

    def lanes(self, node: Expr) -> bool:
        """Whether ``node`` differs between the lanes of a program."""

        match node:
            case ThreadIndex() | TileIndex():
                return True
            case Var():
                return node.name in self.varying
        return any(self.lanes(item) for item in subexpressions(node))

    def load(self, node: Load) -> str:
        pointer = f"{self.buffers[node.buffer]} + {self.expr(node.index)}"
        if self.mask is None or not self.lanes(node.index):
            return f"tl.load({pointer})"
        return f"tl.load({pointer}, mask={self.mask})"

    def access(self, name: str, node: Store | AtomicAdd) -> str:
        pointer = f"{self.buffers[node.buffer]} + {self.expr(node.index)}"
        value = self.expr(node.value)
        if self.mask is None or not self.lanes(node.index):
            return f"{name}({pointer}, {value})"
        return f"{name}({pointer}, {value}, mask={self.mask})"

    def local(self, base: str, value: str, lines: list[str], indent: str) -> str:
        name = self.names.var(base, torch.bool).name
        self.varying.add(name)
        lines.append(f"{indent}{name} = {value}")
        return name

    def narrow(self, condition: str, lines: list[str], indent: str) -> None:
        value = condition if self.mask is None else f"{self.mask} & {condition}"
        self.mask = self.local("mask", value, lines, indent)

    def stmts(
        self, statements: Sequence[Stmt], depth: int, *, top: bool = False
    ) -> list[str]:
        """Spell ``statements``; ``Guard`` ends lanes only at the top level."""

        indent = _INDENT * depth
        lines: list[str] = []
        for position, node in enumerate(statements):
            match node:
                case Let() | Assign():
                    value = self.expr(node.value)
                    varying = self.lanes(node.value)
                    if isinstance(node, Assign) and self.predicate is not None:
                        value = f"tl.where({self.predicate}, {value}, {node.var.name})"
                        varying = True
                    if varying:
                        self.varying.add(node.var.name)
                    lines.append(f"{indent}{node.var.name} = {value}")
                case Store():
                    lines.append(f"{indent}{self.access('tl.store', node)}")
                case AtomicAdd():
                    lines.append(f"{indent}{self.access('tl.atomic_add', node)}")
                case Guard() if not top:
                    raise TypeError(f"a guard inside a branch or loop: {node!r}")
                case Guard() if self.lanes(node.condition):
                    self.narrow(self.expr(node.condition), lines, indent)
                case Guard():
                    # The rest of the kernel runs only where it holds.
                    lines.append(f"{indent}if {self.expr(node.condition)}:")
                    rest = self.stmts(statements[position + 1 :], depth + 1, top=True)
                    lines.extend(rest or [f"{indent}{_INDENT}pass"])
                    return lines
                case If() if self.lanes(node.condition):
                    lines.extend(self.masked(node, indent, depth))
                case If():
                    lines.append(f"{indent}if {self.expr(node.condition)}:")
                    lines.extend(self.branch(node.then, depth + 1))
                    if node.orelse:
                        lines.append(f"{indent}else:")
                        lines.extend(self.branch(node.orelse, depth + 1))
                case ForK():
                    lines.append(
                        f"{indent}for {node.var.name} in tl.static_range({node.count}):"
                    )
                    lines.extend(self.branch(node.body, depth + 1))
                case While(per_lane=True):
                    raise TypeError(
                        "Triton vectorized printer cannot lower per-lane While"
                    )
                case While() if self.lanes(node.condition) or self.predicate:
                    raise TypeError(f"a loop on a per-lane condition: {node!r}")
                case While():
                    lines.append(f"{indent}while {self.expr(node.condition)}:")
                    lines.extend(self.branch(node.body, depth + 1))
                case Block():
                    lines.extend(self.stmts(node.body, depth, top=top))
                case _:
                    raise TypeError(f"not a statement: {node!r}")
        return lines

    def branch(self, statements: Sequence[Stmt], depth: int) -> list[str]:
        state = (self.mask, self.predicate)
        try:
            return self.stmts(statements, depth) or [f"{_INDENT * depth}pass"]
        finally:
            self.mask, self.predicate = state

    def masked(self, node: If, indent: str, depth: int) -> list[str]:
        """A per-lane branch: both run for all lanes, each masked by its side."""

        lines: list[str] = []
        condition = self.local("cond", self.expr(node.condition), lines, indent)
        state = (self.mask, self.predicate)
        for body, taken in ((node.then, condition), (node.orelse, f"~{condition}")):
            if not body:
                continue
            try:
                self.narrow(taken, lines, indent)
                self.predicate = taken if state[1] is None else f"{state[1]} & {taken}"
                lines.extend(self.stmts(body, depth))
            finally:
                self.mask, self.predicate = state
        return lines


PRINTER = TritonPrinter()

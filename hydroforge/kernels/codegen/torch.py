"""PyTorch spelling of kernel IR.

Kernels (:meth:`TorchPrinter.kernel`) run every lane at once as tensor
operations on flat buffers ``states[name]``.  ``ThreadIndex`` is the lane
range: a ``Guard`` (or an ``If``) comparing it with a bound sets how many
lanes run, and an index ``base + lane`` addresses a slice, so loads and
stores of consecutive elements are views.  Strided lane indices and indices
that depend on loaded values gather and scatter through index tensors.  Parameters, loop counters,
integer arithmetic on them and the sample phase (argument ``phase``) are
Python values, so conditions on them are Python branches; conditions on
loaded values select per lane with ``torch.where``, and atomic additions
become ``index_add_``.  A ``While`` is a Python loop, so its condition is
one value.  Floating literals and converted operands are tensors on the
program's device.  A loaded view of a buffer the kernel writes is cloned
when a local keeps it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

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
    Unary,
    Var,
    While,
    type_of,
)
from hydroforge.kernels.codegen.passes import effects, subexpressions

_INDENT = "    "

PRELUDE = """\
import torch


def _hydroforge_tensor_operand(value, reference):
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device,
    )


def _hydroforge_binary_operands(left, right):
    if isinstance(left, torch.Tensor):
        right = _hydroforge_tensor_operand(right, left)
    elif isinstance(right, torch.Tensor):
        left = _hydroforge_tensor_operand(left, right)
    else:
        left = torch.as_tensor(left)
        right = torch.as_tensor(right)
    return left, right


def hydroforge_maximum(left, right):
    return torch.fmax(*_hydroforge_binary_operands(left, right))


def hydroforge_minimum(left, right):
    return torch.fmin(*_hydroforge_binary_operands(left, right))


# Incremental update: its FP32 error stays bounded as the window grows.
# Non-finite results fall back to the blended form so infinities and
# NaNs propagate exactly as before.
def hydroforge_weighted_mean(old, old_weight, value, weight):
    new_weight = old_weight + weight
    ratio = weight / new_weight
    incremental = old + (value - old) * ratio
    return torch.where(
        torch.isfinite(incremental),
        incremental,
        old * (old_weight / new_weight) + value * ratio,
    )


def hydroforge_remainder(left, right):
    if not isinstance(left, torch.Tensor) and not isinstance(right, torch.Tensor):
        return left % right
    return torch.remainder(*_hydroforge_binary_operands(left, right))
"""

# Names the prelude and kernel signatures bind; no local may take them.
_GLOBALS = frozenset(
    (
        "torch",
        "device",
        "states",
        "_hydroforge_tensor_operand",
        "_hydroforge_binary_operands",
        "hydroforge_maximum",
        "hydroforge_minimum",
        "hydroforge_weighted_mean",
        "hydroforge_remainder",
    )
)

_FUNCTIONS = {
    "abs": "torch.abs",
    "sqrt": "torch.sqrt",
    "exp": "torch.exp",
    "log": "torch.log",
    "sin": "torch.sin",
    "cos": "torch.cos",
    "tan": "torch.tan",
    "pow": "torch.pow",
    "py_mod": "hydroforge_remainder",
    "nan_max": "hydroforge_maximum",
    "nan_min": "hydroforge_minimum",
    "weighted_mean": "hydroforge_weighted_mean",
    "isnan": "torch.isnan",
}


def _literal(value: bool | int | float) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return f"float('{value}')"
    return repr(value)


class TorchPrinter:
    """Spell kernels whose tensors live on the device ``device`` spells."""

    def __init__(self, device: str) -> None:
        self.device = device

    def tensor(self, value: str, dtype) -> str:
        return f"torch.as_tensor({value}, dtype={dtype}, device={self.device})"

    def kernel(
        self,
        function: KernelFunction,
        *,
        sizes: Mapping[str, int],
        scalars: Mapping[str, int],
    ) -> str:
        """``def name(states, phase)`` running ``function`` on flat buffers.

        ``sizes`` are the buffers' element counts and ``scalars`` the values
        of its scalar parameters, which the source spells as literals.
        """

        return "\n".join(_Kernel(self.device, function, sizes, scalars).lines())


_PYTHON = "python"
_TENSOR = "tensor"
_ARITHMETIC = {
    "+": lambda left, right: left + right,
    "-": lambda left, right: left - right,
    "*": lambda left, right: left * right,
}


@dataclass(frozen=True, slots=True)
class _Lanes:
    """The lane index ``base + stride * lane`` of a Python ``base``."""

    base: str
    stride: int


def _integer(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def _add(left: str, right: str) -> str:
    known = _integer(left), _integer(right)
    if None not in known:
        return str(known[0] + known[1])
    if known[0] == 0:
        return right
    if known[1] == 0:
        return left
    return f"({left} + {right})"


def _scale(text: str, factor: int) -> str:
    known = _integer(text)
    if known is not None:
        return str(known * factor)
    return text if factor == 1 else f"({text} * {factor})"


def _python_const(node: Const) -> bool:
    return (
        isinstance(node.value, bool)
        or node.type is None
        or not node.type.is_floating_point
    )


class _Kernel(TorchPrinter):
    """The printing state of one kernel: lane count, predicate, value kinds."""

    def __init__(
        self,
        device: str | None,
        function: KernelFunction,
        sizes: Mapping[str, int],
        scalars: Mapping[str, int],
    ) -> None:
        super().__init__(device)
        self.function = function
        self.sizes = sizes
        self.dtypes = {param.name: param.type for param in function.params}
        self.scalars = {}
        for param in function.params:
            if param.access is not None:
                continue
            value = scalars[param.name]
            if param.type == torch.bool:
                if type(value) not in {bool, int} or value not in (0, 1):
                    raise TypeError(
                        f"Torch boolean specialization {param.name!r} requires bool or integer 0/1"
                    )
                self.scalars[param.name] = bool(value)
                continue
            if param.type not in {torch.int32, torch.int64} or type(value) is not int:
                raise TypeError(
                    f"Torch scalar specialization {param.name!r} requires an exact int32/int64 integer"
                )
            limits = torch.iinfo(param.type)
            if not limits.min <= value <= limits.max:
                raise ValueError(
                    f"Torch scalar specialization {param.name!r} is outside {param.type} range"
                )
            self.scalars[param.name] = value
        touched = effects(function.body)
        taken = touched.uses | touched.defines
        if taken & _GLOBALS:
            raise ValueError(
                f"{function.name}: locals take reserved names {sorted(taken & _GLOBALS)}"
            )
        self.names = Names((*taken, *_GLOBALS))
        self.written = touched.writes
        self.kinds: dict[str, str | _Lanes] = {PHASE.name: _PYTHON}
        self.count: str | None = None
        self.predicate: str | None = None

    def lines(self) -> list[str]:
        body = self.stmts(self.function.body, 1, top=True)
        return [
            f"def {self.function.name}(states, {PHASE.name}):",
            *(body or [f"{_INDENT}pass"]),
        ]

    # Index arithmetic: Python integers and lane indices.

    def index(self, node: Expr) -> str | _Lanes | None:
        """``node`` as a Python integer, a lane index, or ``None`` (a tensor)."""

        match node:
            case Var() if node.name in self.scalars:
                return str(self.scalars[node.name])
            case Var():
                kind = self.kinds.get(node.name, _PYTHON)
                if kind == _PYTHON:
                    return node.name
                return kind if isinstance(kind, _Lanes) else None
            case Const(value=int()) if _python_const(node) and not isinstance(
                node.value, bool
            ):
                return str(node.value)
            case ThreadIndex():
                return _Lanes("0", 1)
            case Cast() if not node.type.is_floating_point:
                operand = type_of(node.operand)
                if operand == node.type:
                    return self.index(node.operand)
                return None
            case Binary():
                left, right = self.index(node.left), self.index(node.right)
                if left is None or right is None:
                    return None
                if isinstance(left, str) and isinstance(right, str):
                    known = _integer(left), _integer(right)
                    if None not in known and node.op in _ARITHMETIC:
                        return str(_ARITHMETIC[node.op](*known))
                    op = "//" if node.op == "/" else node.op
                    return f"({left} {op} {right})"
                return self._lanes(node.op, left, right)
        return None

    @staticmethod
    def _lanes(op: str, left: str | _Lanes, right: str | _Lanes) -> _Lanes | None:
        if op == "+":
            if isinstance(left, str):
                left, right = right, left
            if isinstance(right, str):
                return _Lanes(_add(left.base, right), left.stride)
            return _Lanes(_add(left.base, right.base), left.stride + right.stride)
        if op == "-" and isinstance(right, str):
            return _Lanes(f"({left.base} - {right})", left.stride)
        if op == "*":
            if isinstance(left, str):
                left, right = right, left
            factor = _integer(right) if isinstance(right, str) else None
            if factor is not None:
                return _Lanes(_scale(left.base, factor), left.stride * factor)
        return None

    def lane_count(self) -> str:
        if self.count is None:
            raise TypeError(f"{self.function.name}: lanes used before their bound")
        return self.count

    def lanes_tensor(self, lanes: _Lanes) -> str:
        """A lane index as an int64 tensor."""

        arange = f"torch.arange({self.lane_count()}, device={self.device})"
        return _add(lanes.base, _scale(arange, lanes.stride))

    def target(self, buffer: str, index: Expr) -> tuple[str, str]:
        """What ``index`` addresses in ``buffer``: its spelling and form.

        The form is ``whole`` (the flat buffer, also for the element of a
        one-element buffer), ``view`` (consecutive lanes or one element) or
        ``gather`` (through an index tensor).
        """

        name = f"states[{buffer!r}]"
        address = self.index(index)
        if isinstance(address, str):
            if _integer(address) == 0 and self.sizes.get(buffer) == 1:
                return name, "whole"
            return f"{name}[{address}]", "view"
        if address is None:
            return f"{name}[{self.value(index)}]", "gather"
        if address.stride != 1:
            # A strided view store compiles (torch.compile) to another fusion
            # of its arithmetic, which rounds weighted means differently.
            return f"{name}[{self.lanes_tensor(address)}]", "gather"
        count = self.lane_count()
        if _integer(address.base) == 0 and _integer(count) == self.sizes.get(buffer):
            return name, "whole"
        return f"{name}[{address.base}:{_add(address.base, count)}]", "view"

    # Values.

    def kind(self, node: Expr) -> str:
        """Whether ``node`` is a Python value or a tensor."""

        if isinstance(self.index(node), str):
            return _PYTHON
        match node:
            case PhaseTest():
                return _PYTHON
            case Const():
                return _PYTHON if _python_const(node) else _TENSOR
            case Compare() | Logical() | Unary():
                if all(self.kind(item) == _PYTHON for item in subexpressions(node)):
                    return _PYTHON
        return _TENSOR

    def value(self, node: Expr, own: bool = False) -> str:
        """Spell ``node``; ``own`` clones a view of a buffer the kernel writes."""

        address = self.index(node)
        if isinstance(address, str):
            return address
        if isinstance(address, _Lanes):
            return self.lanes_tensor(address)
        match node:
            case Var():
                return node.name
            case Const() if _python_const(node):
                return _literal(node.value)
            case Const():
                return self.tensor(_literal(node.value), node.type)
            case Load():
                text, form = self.target(node.buffer, node.index)
                if own and form != "gather" and node.buffer in self.written:
                    return f"{text}.clone()"
                return text
            case Unary(op="not") if self.kind(node.operand) == _PYTHON:
                return f"(not {self.value(node.operand)})"
            case Unary(op="not"):
                return f"torch.logical_not({self.value(node.operand)})"
            case Unary():
                return f"(-{self.value(node.operand)})"
            case Binary(op="/") if not (
                type_of(node) or torch.float32
            ).is_floating_point:
                left, right = self.value(node.left), self.value(node.right)
                return f"torch.div({left}, {right}, rounding_mode='trunc')"
            case Binary() | Compare():
                return f"({self.value(node.left)} {node.op} {self.value(node.right)})"
            case Logical():
                python = self.kind(node) == _PYTHON
                symbol = {
                    ("and", True): " and ",
                    ("and", False): " & ",
                    ("or", True): " or ",
                    ("or", False): " | ",
                }[node.op, python]
                return f"({symbol.join(self.value(item) if type_of(item) == torch.bool else f'({self.value(item)} != 0)' for item in node.operands)})"
            case Select() if self.kind(node.condition) == _PYTHON:
                return (
                    f"({self.value(node.positive, own)} if "
                    f"{self.value(node.condition)} else "
                    f"{self.value(node.negative, own)})"
                )
            case Select():
                return (
                    f"torch.where({self.value(node.condition)}, "
                    f"{self.value(node.positive)}, {self.value(node.negative)})"
                )
            case Cast():
                source_type = type_of(node.operand)
                value = self.tensor(
                    self.value(
                        node.operand, own=own and type_of(node.operand) == node.type
                    ),
                    source_type,
                )
                if node.type == torch.bool:
                    return f"({value} != 0)"
                if (
                    node.type in (torch.uint8, torch.uint16, torch.uint32)
                    and source_type is not None
                    and source_type != torch.bool
                    and not source_type.is_floating_point
                    and torch.iinfo(source_type).bits > torch.iinfo(node.type).bits
                ):
                    # Preserve narrowing even when Inductor fuses an arange
                    # through a subsequent widening cast.
                    value = f"({value} & {torch.iinfo(node.type).max})"
                return f"({value}).to({node.type})"
            case Call():
                function = _FUNCTIONS[node.function]
                arguments = ", ".join(
                    self.tensor(self.value(item), type_of(item))
                    if function.startswith("torch.") and self.kind(item) == _PYTHON
                    else self.value(item)
                    for item in node.arguments
                )
                return f"{function}({arguments})"
            case PhaseTest():
                return f"(({PHASE.name} & {int(node.bits)}) != 0)"
        raise TypeError(f"not a PyTorch kernel expression: {node!r}")

    # Statements.

    def bound(self, condition: Expr) -> str | None:
        """The lane count ``lane < bound`` sets, if ``condition`` is one."""

        if not (isinstance(condition, Compare) and condition.op == "<"):
            return None
        lanes, bound = self.index(condition.left), self.index(condition.right)
        if (
            isinstance(lanes, _Lanes)
            and lanes.stride == 1
            and _integer(lanes.base) == 0
            and isinstance(bound, str)
        ):
            return bound
        return None

    def local(self, base: str, value: str, lines: list[str], indent: str) -> str:
        name = self.names.var(base, torch.bool).name
        self.kinds[name] = _TENSOR
        lines.append(f"{indent}{name} = {value}")
        return name

    def selected(self, value: str, current: str) -> str:
        if self.predicate is None:
            return value
        return f"torch.where({self.predicate}, {value}, {current})"

    def stmts(
        self, statements: Sequence[Stmt], depth: int, *, top: bool = False
    ) -> list[str]:
        """Spell ``statements``; ``Guard`` ends lanes only at the top level."""

        indent = _INDENT * depth
        lines: list[str] = []
        for node in statements:
            match node:
                case Let(var=var) if var.name == PHASE.name:
                    pass  # the sample phase is the ``phase`` argument
                case Let() | Assign():
                    lines.extend(self.assign(node, indent))
                case Store():
                    lines.append(f"{indent}{self.store(node)}")
                case AtomicAdd():
                    lines.append(f"{indent}{self.atomic_add(node)}")
                case Guard() if not top:
                    raise TypeError(f"a guard inside a branch or loop: {node!r}")
                case Guard():
                    lines.extend(self.guard(node, indent))
                case If():
                    lines.extend(self.branch(node, indent, depth))
                case ForK():
                    self.kinds[node.var.name] = _PYTHON
                    lines.append(f"{indent}for {node.var.name} in range({node.count}):")
                    lines.extend(self.block(node.body, depth + 1))
                case While() if self.predicate is not None:
                    raise TypeError(f"a loop under a per-lane condition: {node!r}")
                case While():
                    lines.append(f"{indent}while {self.value(node.condition)}:")
                    lines.extend(self.block(node.body, depth + 1))
                case Block():
                    lines.extend(self.stmts(node.body, depth, top=top))
                case _:
                    raise TypeError(f"not a statement: {node!r}")
        return lines

    def block(self, statements: Sequence[Stmt], depth: int) -> list[str]:
        state = (self.count, self.predicate)
        try:
            return self.stmts(statements, depth) or [f"{_INDENT * depth}pass"]
        finally:
            self.count, self.predicate = state

    def assign(self, node: Let | Assign, indent: str) -> list[str]:
        name = node.var.name
        address = self.index(node.value)
        if isinstance(node, Let) and isinstance(address, _Lanes):
            self.kinds[name] = address
            return []
        if isinstance(address, str):
            if isinstance(node, Assign) and self.predicate is not None:
                self.kinds[name] = _TENSOR
                value = self.selected(
                    self.tensor(address, node.var.type),
                    self.tensor(name, node.var.type),
                )
                return [f"{indent}{name} = {value}"]
            self.kinds[name] = _PYTHON
            return [f"{indent}{name} = {address}"]
        value = self.value(node.value, own=True)
        kind = self.kind(node.value)
        if isinstance(node, Assign) and self.predicate is not None:
            value, kind = self.selected(value, name), _TENSOR
        self.kinds[name] = kind
        return [f"{indent}{name} = {value}"]

    def store(self, node: Store) -> str:
        target, form = self.target(node.buffer, node.index)
        value = self.selected(self.value(node.value), target)
        if form != "whole":
            return f"{target} = {value}"
        if self.predicate is None and self.kind(node.value) == _PYTHON:
            return f"{target}.fill_({value})"
        return f"{target}.copy_({value})"

    def atomic_add(self, node: AtomicAdd) -> str:
        if self.sizes[node.buffer] == 0:
            return "pass"
        address = self.index(node.index)
        index = (
            self.lanes_tensor(address)
            if isinstance(address, _Lanes)
            else self.value(node.index)
        )
        index = f"({self.tensor(index, torch.int64)}).reshape(-1)"
        if self.count is not None and not isinstance(address, _Lanes):
            index = f"{index}.expand({self.count})"
        value = self.value(node.value)
        if self.predicate is not None:
            # Excluded lanes add zero to element 0, which no sum from a
            # zeroed buffer can hold as -0.0.
            index = f"torch.where({self.predicate}, {index}, 0)"
            value = f"torch.where({self.predicate}, {value}, 0)"
        dtype = self.dtypes[node.buffer]
        value = f"({self.tensor(value, dtype)}).expand_as({index})"
        return f"states[{node.buffer!r}].index_add_(0, {index}, {value})"

    def guard(self, node: Guard, indent: str) -> list[str]:
        bound = self.bound(node.condition)
        if bound is not None:
            self.count = bound
            return []
        condition = self.value(node.condition)
        if self.kind(node.condition) == _PYTHON:
            return [f"{indent}if not {condition}:", f"{indent}{_INDENT}return"]
        lines: list[str] = []
        if self.predicate is not None:
            condition = f"{self.predicate} & {condition}"
        self.predicate = self.local("valid", condition, lines, indent)
        return lines

    def branch(self, node: If, indent: str, depth: int) -> list[str]:
        bound = self.bound(node.condition)
        if bound is not None:
            if node.orelse:
                raise TypeError(f"a lane bound with an else branch: {node!r}")
            state = self.count
            try:
                self.count = bound
                return self.stmts(node.then, depth)
            finally:
                self.count = state
        if self.kind(node.condition) == _PYTHON:
            lines = [f"{indent}if {self.value(node.condition)}:"]
            lines.extend(self.block(node.then, depth + 1))
            if node.orelse:
                lines.append(f"{indent}else:")
                lines.extend(self.block(node.orelse, depth + 1))
            return lines
        lines: list[str] = []
        condition = self.local("cond", self.value(node.condition), lines, indent)
        outer = self.predicate
        for body, taken in ((node.then, condition), (node.orelse, f"~{condition}")):
            if not body:
                continue
            try:
                self.predicate = taken if outer is None else f"({outer} & {taken})"
                lines.extend(self.stmts(body, depth))
            finally:
                self.predicate = outer
        return lines

"""Element-wise kernel IR shared by the framework code generators.

Expressions and statements spell out every operation, operand order and
conversion, so a printer only chooses the dialect's spelling and the operation
structure is the same in every dialect.  Types are tensor element dtypes;
``None`` marks a value whose type the generator does not track (a caller's
symbol, or a literal of the dialect's default type).

A kernel reads its sample phase into the local :data:`PHASE`;
:class:`PhaseTest` tests its bits.  :class:`ThreadIndex` is the global index
of the executing thread; block-program dialects also address two-axis tiles
with :class:`TileIndex`.  Kernels without ``ThreadIndex`` run one lane (the
framework's clock and loop control kernels); only they loop with
:class:`While`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

import torch

Access = Literal["read", "write", "read_write", "atomic_add"]
Intrinsic = Literal[
    "abs",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "pow",
    "py_mod",
    "nan_max",
    "nan_min",
    "weighted_mean",
    "isnan",
    "set_conditional",
]


@dataclass(frozen=True, slots=True)
class Var:
    name: str
    type: torch.dtype | None = None


@dataclass(frozen=True, slots=True)
class Const:
    """A literal; float ``nan`` and infinities are allowed with a type."""

    value: bool | int | float
    type: torch.dtype | None = None

    def __post_init__(self) -> None:
        if type(self.value) not in {bool, int, float}:
            raise TypeError("IR literals must be exact bool, int or float")
        if self.type is None:
            if type(self.value) is float and not math.isfinite(self.value):
                raise ValueError("non-finite IR literals require a dtype")
            return
        if self.type == torch.bool:
            object.__setattr__(self, "value", bool(self.value))
        elif self.type.is_floating_point:
            if math.isfinite(self.value):
                if abs(self.value) > torch.finfo(self.type).max:
                    raise OverflowError("IR floating literal exceeds dtype range")
                if type(self.value) is int:
                    precision = 1 - int(math.log2(torch.finfo(self.type).eps))
                    magnitude = abs(self.value)
                    shift = max(0, magnitude.bit_length() - precision)
                    if (magnitude >> shift) << shift != magnitude:
                        raise ValueError(
                            "IR integer literal is not exactly representable"
                        )
        else:
            limits = torch.iinfo(self.type)
            if (
                type(self.value) is not int
                or not limits.min <= self.value <= limits.max
            ):
                raise ValueError("IR integer literal exceeds dtype range")


@dataclass(frozen=True, slots=True)
class Load:
    buffer: str
    index: Expr
    type: torch.dtype


@dataclass(frozen=True, slots=True)
class Unary:
    op: Literal["neg", "not"]
    operand: Expr

    def __post_init__(self) -> None:
        if self.op not in {"neg", "not"}:
            raise ValueError(f"unknown unary operator {self.op!r}")


@dataclass(frozen=True, slots=True)
class Binary:
    """Arithmetic on two operands of one type; ``/`` and ``%`` on integers
    are C's truncating operations and appear only on nonnegative operands,
    where they agree with floor division; ``&`` and ``|`` are bitwise on
    integers."""

    op: Literal["+", "-", "*", "/", "%", "&", "|"]
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in {"+", "-", "*", "/", "%", "&", "|"}:
            raise ValueError(f"unknown binary operator {self.op!r}")
        left, right = type_of(self.left), type_of(self.right)
        if left is not None and right is not None and left != right:
            raise TypeError(f"{self.op} operands differ in type: {left} and {right}")
        if (
            self.op in ("/", "%")
            and isinstance(self.right, Const)
            and self.right.value == 0
            and (right is None or not right.is_floating_point)
        ):
            raise ValueError("integer division or remainder by literal zero")
        if self.op in {"&", "|"} and any(
            dtype is not None and dtype.is_floating_point for dtype in (left, right)
        ):
            raise TypeError("bitwise operations require integer operands")
        if (
            self.op in {"/", "%"}
            and not (left or right or torch.int64).is_floating_point
        ):
            numerator, divisor = constant_value(self.left), constant_value(self.right)
            if divisor == 0:
                raise ValueError("integer division or remainder by constant zero")
            if (numerator is not None and numerator < 0) or (
                divisor is not None and divisor < 0
            ):
                raise ValueError(
                    "integer division and remainder require non-negative operands"
                )


@dataclass(frozen=True, slots=True)
class Compare:
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in {"<", "<=", ">", ">=", "==", "!="}:
            raise ValueError(f"unknown comparison operator {self.op!r}")


@dataclass(frozen=True, slots=True)
class Logical:
    op: Literal["and", "or"]
    operands: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if self.op not in {"and", "or"} or not self.operands:
            raise ValueError(
                "logical expressions require and/or and at least one operand"
            )


@dataclass(frozen=True, slots=True)
class Select:
    """Select values by condition; operand evaluation need not be lazy.

    Both operands must be valid even where unselected. Use uniform control flow
    for branches that require conditional execution.
    """

    condition: Expr
    positive: Expr
    negative: Expr

    def __post_init__(self) -> None:
        positive, negative = type_of(self.positive), type_of(self.negative)
        if positive is not None and negative is not None and positive != negative:
            raise TypeError("Select branches differ in type")
        type_of(self.condition)


@dataclass(frozen=True, slots=True)
class Cast:
    operand: Expr
    type: torch.dtype


@dataclass(frozen=True, slots=True)
class Call:
    """An intrinsic; ``py_mod`` is Python's remainder, ``nan_max`` and
    ``nan_min`` ignore one NaN operand.  ``set_conditional(handle, value)``
    sets a CUDA conditional graph node's condition (CUDA only)."""

    function: Intrinsic
    arguments: tuple[Expr, ...]
    type: torch.dtype | None

    def __post_init__(self) -> None:
        arities = {
            "abs": 1,
            "sqrt": 1,
            "exp": 1,
            "log": 1,
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "pow": 2,
            "py_mod": 2,
            "nan_max": 2,
            "nan_min": 2,
            "weighted_mean": 4,
            "isnan": 1,
            "set_conditional": 2,
        }
        if (
            self.function not in arities
            or len(self.arguments) != arities[self.function]
        ):
            raise ValueError(f"invalid intrinsic or arity: {self.function!r}")


@dataclass(frozen=True, slots=True)
class PhaseTest:
    """Whether the kernel's sample phase has any of ``bits``."""

    bits: int


@dataclass(frozen=True, slots=True)
class ThreadIndex:
    """The global index of the executing thread (int64)."""


@dataclass(frozen=True, slots=True)
class TileIndex:
    """Block programs only: the lanes of a two-axis tile.

    Axis 0 is :class:`ThreadIndex` as a column; axis 1 counts
    ``0 .. extent - 1`` (a power of two) as a row, so values indexed by both
    are ``[block, extent]`` tiles.
    """

    axis: Literal[0, 1]
    extent: int = 0

    def __post_init__(self) -> None:
        if type(self.axis) is not int or self.axis not in {0, 1}:
            raise ValueError("tile axis must be 0 or 1")
        if (
            type(self.extent) is not int
            or self.extent < 0
            or (
                self.axis == 1 and (self.extent == 0 or self.extent & (self.extent - 1))
            )
        ):
            raise ValueError("tile axis 1 requires a positive power-of-two extent")


Expr = (
    Var
    | Const
    | Load
    | Unary
    | Binary
    | Compare
    | Logical
    | Select
    | Cast
    | Call
    | PhaseTest
    | ThreadIndex
    | TileIndex
)

PHASE = Var("phase", torch.int32)


def type_of(expression: Expr) -> torch.dtype | None:
    """The element type of ``expression``, or ``None`` if untracked."""

    match expression:
        case Var() | Const() | Load() | Cast() | Call():
            return expression.type
        case Binary():
            return type_of(expression.left) or type_of(expression.right)
        case Unary(op="neg"):
            return type_of(expression.operand)
        case Select():
            return type_of(expression.positive) or type_of(expression.negative)
        case ThreadIndex() | TileIndex():
            return torch.int64
        case Unary(op="not") | Compare() | Logical() | PhaseTest():
            return torch.bool
    raise TypeError(f"unknown kernel expression {expression!r}")


def cast(expression: Expr, dtype: torch.dtype) -> Expr:
    """``expression`` converted to ``dtype``; no conversion if it has it."""

    return expression if type_of(expression) == dtype else Cast(expression, dtype)


@dataclass(frozen=True, slots=True)
class Let:
    """Declare and initialize a local."""

    var: Var
    value: Expr


@dataclass(frozen=True, slots=True)
class Assign:
    var: Var
    value: Expr


@dataclass(frozen=True, slots=True)
class Store:
    buffer: str
    index: Expr
    value: Expr


@dataclass(frozen=True, slots=True)
class AtomicAdd:
    buffer: str
    index: Expr
    value: Expr


@dataclass(frozen=True, slots=True)
class If:
    condition: Expr
    then: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...] = ()


@dataclass(frozen=True, slots=True)
class ForK:
    """``body`` for ``var`` = 0 .. ``count`` - 1, a compile-time count."""

    var: Var
    count: int
    body: tuple[Stmt, ...]

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("ForK count must be a non-negative exact int")
        if self.var.type not in {torch.int32, torch.int64}:
            raise TypeError("ForK loop variable must be an integer")


@dataclass(frozen=True, slots=True)
class While:
    """Repeat ``body`` while ``condition`` holds.

    By default this is a one-lane control loop. ``per_lane`` explicitly
    permits independent loops in scalar-thread C-family kernels (e.g. CSR).
    Vectorized printers must reject that form unless they implement masks.
    """

    condition: Expr
    body: tuple[Stmt, ...]
    per_lane: bool = False


@dataclass(frozen=True, slots=True)
class Evaluate:
    """Call an intrinsic for its effect."""

    call: Call


@dataclass(frozen=True, slots=True)
class Guard:
    """End this thread unless ``condition`` holds."""

    condition: Expr


@dataclass(frozen=True, slots=True)
class Block:
    """A scope for the locals ``body`` declares."""

    body: tuple[Stmt, ...]


Stmt = Let | Assign | Store | AtomicAdd | If | ForK | While | Evaluate | Guard | Block


@dataclass(frozen=True, slots=True)
class Param:
    """A kernel parameter: a buffer with its access, or a by-value scalar
    (``access`` is ``None``)."""

    name: str
    type: torch.dtype
    access: Access | None = None


@dataclass(frozen=True, slots=True)
class KernelFunction:
    name: str
    params: tuple[Param, ...]
    body: tuple[Stmt, ...]

    def __post_init__(self) -> None:
        validate_function(self)


def constant_value(expression: Expr) -> int | float | bool | None:
    """Fold only defined literal arithmetic for static domain checks."""
    if isinstance(expression, Const):
        return expression.value
    if isinstance(expression, Cast):
        value = constant_value(expression.operand)
        if value is None:
            return None
        return float(value) if expression.type.is_floating_point else int(value)
    if isinstance(expression, Unary) and expression.op == "neg":
        value = constant_value(expression.operand)
        return None if value is None else -value
    if isinstance(expression, Binary):
        left, right = constant_value(expression.left), constant_value(expression.right)
        if left is None or right is None:
            return None
        if expression.op == "/" and not type_of(expression).is_floating_point:
            if right == 0:
                return None
            quotient = abs(left) // abs(right)
            return -quotient if (left < 0) != (right < 0) else quotient
        operations = {
            "+": lambda: left + right,
            "-": lambda: left - right,
            "*": lambda: left * right,
            "/": lambda: left / right,
            "%": lambda: left % right,
            "&": lambda: left & right,
            "|": lambda: left | right,
        }
        try:
            return operations[expression.op]()
        except (ArithmeticError, TypeError):
            return None
    return None


def validate_function(function: KernelFunction) -> None:
    """Seal the lexical, typed buffer contract before optimization or printing."""
    if not function.name.isidentifier():
        raise ValueError("kernel name must be an identifier")
    params = {param.name: param for param in function.params}
    if len(params) != len(function.params):
        raise ValueError("kernel parameter names must be unique")
    for param in function.params:
        if not param.name or param.access not in {
            None,
            "read",
            "write",
            "read_write",
            "atomic_add",
        }:
            raise ValueError(f"invalid kernel parameter {param.name!r}")
    scalar_scope = {
        param.name: param.type for param in function.params if param.access is None
    }

    def children(node):
        if is_dataclass(node):
            for field in fields(node):
                value = getattr(node, field.name)
                for item in value if isinstance(value, tuple) else (value,):
                    if is_dataclass(item):
                        yield item

    def has_lanes(node):
        return isinstance(node, (ThreadIndex, TileIndex)) or any(
            has_lanes(item) for item in children(node)
        )

    lanes = any(has_lanes(statement) for statement in function.body)

    def expression(node, scope):
        type_of(node)
        if isinstance(node, Var):
            if node.name not in scope:
                raise ValueError(f"IR variable {node.name!r} is used before definition")
            if (
                node.type is not None
                and scope[node.name] is not None
                and node.type != scope[node.name]
            ):
                raise TypeError(f"IR variable {node.name!r} differs in type")
        elif isinstance(node, Load):
            buffer(node.buffer, node.index, node.type, scope, "read")
        for item in children(node):
            expression(item, scope)

    def buffer(name, index, dtype, scope, access):
        param = params.get(name)
        allowed = {
            "read": {"read", "read_write", "atomic_add"},
            "write": {"write", "read_write"},
            "atomic": {"atomic_add", "read_write"},
        }[access]
        if param is None or param.access not in allowed:
            raise ValueError(f"buffer {name!r} does not permit {access}")
        if dtype is not None and dtype != param.type:
            raise TypeError(f"buffer {name!r} element type mismatch")
        if type_of(index) is not None and (
            type_of(index).is_floating_point or type_of(index) == torch.bool
        ):
            raise TypeError("buffer indices must be integer")
        if (known := constant_value(index)) is not None and (
            type(known) not in {int} or known < 0
        ):
            raise ValueError("known buffer index must be a non-negative integer")
        expression(index, scope)

    def statements(body, scope, *, top=False):
        scope = dict(scope)
        for node in body:
            if isinstance(node, (Let, Assign)):
                expression(node.value, scope)
                if (
                    node.var.type is not None
                    and type_of(node.value) is not None
                    and node.var.type != type_of(node.value)
                ):
                    raise TypeError("IR assignment type mismatch")
                if isinstance(node, Let):
                    if (
                        not node.var.name.isidentifier()
                        or node.var.name in scope
                        or node.var.name in scalar_scope
                    ):
                        raise ValueError(
                            f"IR local shadowing or invalid name {node.var.name!r}"
                        )
                    scope[node.var.name] = node.var.type
                else:
                    expression(node.var, scope)
            elif isinstance(node, (Store, AtomicAdd)):
                buffer(
                    node.buffer,
                    node.index,
                    type_of(node.value),
                    scope,
                    "atomic" if isinstance(node, AtomicAdd) else "write",
                )
                expression(node.value, scope)
            elif isinstance(node, If):
                expression(node.condition, scope)
                statements(node.then, scope)
                statements(node.orelse, scope)
            elif isinstance(node, ForK):
                if node.var.name in scope:
                    raise ValueError("loop variable shadows an existing variable")
                statements(node.body, {**scope, node.var.name: node.var.type})
            elif isinstance(node, While):
                if lanes and not node.per_lane:
                    raise ValueError("While requires a one-lane kernel")
                expression(node.condition, scope)
                statements(node.body, scope)
            elif isinstance(node, Guard):
                if not top:
                    raise ValueError("Guard is only allowed at the top level")
                expression(node.condition, scope)
            elif isinstance(node, Block):
                statements(node.body, scope, top=top)
            elif isinstance(node, Evaluate):
                expression(node.call, scope)
            else:
                raise TypeError(f"unknown kernel statement {node!r}")

    statements(function.body, scalar_scope, top=True)


class Names:
    """Distinct local names of one kernel, avoiding ``reserved``."""

    def __init__(self, reserved: Iterable[str] = ()) -> None:
        self._used = set(reserved)

    def var(self, base: str, dtype: torch.dtype) -> Var:
        name, suffix = base, 2
        while name in self._used:
            name, suffix = f"{base}_{suffix}", suffix + 1
        self._used.add(name)
        return Var(name, dtype)

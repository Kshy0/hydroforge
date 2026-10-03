"""The C-family printer of kernel IR: CUDA C++ and the Metal Shading Language.

Both dialects share statement and expression spelling; a trait supplies what
differs: the entry and its argument ABI, type names and conversions, atomic
addition, the thread index, the intrinsics (``set_conditional`` is CUDA's
``cudaGraphSetConditional``) and the helper prelude.  Buffer parameters are
spelled ``p_<name>`` and scalar parameters by name, with non-word characters
replaced.

The prelude helpers are the dialects' NaN rules: ``nan_max``/``nan_min``
ignore one NaN operand and ``weighted_mean`` is the incremental mean with a
blended fallback for non-finite results.  CUDA tests NaN and finiteness with
the math library; MSL tests IEEE bits, which fast math cannot fold away.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
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
    Evaluate,
    Expr,
    ForK,
    Guard,
    If,
    KernelFunction,
    Let,
    Load,
    Logical,
    Param,
    PhaseTest,
    Select,
    Stmt,
    Store,
    ThreadIndex,
    Unary,
    Var,
    While,
)
from hydroforge.kernels.codegen.passes import effects
from hydroforge.kernels.codegen.types import element, scalar, scalar_kind

_INDENT = "    "

_CUDA_PRELUDE = """\
template <typename T> __device__ inline T hf_max(T a, T b) { return a > b ? a : b; }
template <typename T> __device__ inline T hf_min(T a, T b) { return a < b ? a : b; }
template <> __device__ inline float hf_max<float>(float a, float b) { if (isnan(a)) return b; if (isnan(b)) return a; return a > b ? a : b; }
template <> __device__ inline float hf_min<float>(float a, float b) { if (isnan(a)) return b; if (isnan(b)) return a; return a < b ? a : b; }
template <> __device__ inline double hf_max<double>(double a, double b) { if (isnan(a)) return b; if (isnan(b)) return a; return a > b ? a : b; }
template <> __device__ inline double hf_min<double>(double a, double b) { if (isnan(a)) return b; if (isnan(b)) return a; return a < b ? a : b; }
// Incremental form bounds FP32 drift; non-finite results keep the
// blended form's infinity/NaN propagation.
template <typename T> __device__ inline T hf_weighted_mean(T old_v, T old_w, T value, T weight) { T new_w = old_w + weight; T ratio = weight / new_w; T incremental = old_v + (value - old_v) * ratio; return isfinite(incremental) ? incremental : old_v * (old_w / new_w) + value * ratio; }
"""

_MSL_PRELUDE = """\
#include <metal_stdlib>
using namespace metal;

// Extrema treat one NaN as missing and preserve NaN when both are NaN.
// Inspecting the IEEE bits keeps this contract explicit under fast math.
inline bool hydroforge_isnan(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;
}
inline float hydroforge_nan() {
    return as_type<float>(0x7fc00000u);
}
inline float hydroforge_maximum(float left, float right) {
    if (hydroforge_isnan(left)) return right;
    if (hydroforge_isnan(right)) return left;
    return max(left, right);
}
inline float hydroforge_minimum(float left, float right) {
    if (hydroforge_isnan(left)) return right;
    if (hydroforge_isnan(right)) return left;
    return min(left, right);
}
// Incremental form bounds FP32 drift; non-finite results keep the
// blended form's infinity/NaN propagation (bit test survives fast math).
inline float hydroforge_weighted_mean(float old_value, float old_weight, float value, float weight) {
    float new_weight = old_weight + weight;
    float ratio = weight / new_weight;
    float incremental = old_value + (value - old_value) * ratio;
    if ((as_type<uint>(incremental) & 0x7f800000u) != 0x7f800000u) return incremental;
    return old_value * (old_weight / new_weight) + value * ratio;
}
inline int hydroforge_maximum(int left, int right) { return max(left, right); }
inline int hydroforge_minimum(int left, int right) { return min(left, right); }
inline long hydroforge_maximum(long left, long right) { return max(left, right); }
inline long hydroforge_minimum(long left, long right) { return min(left, right); }
inline uchar hydroforge_maximum(uchar left, uchar right) { return max(left, right); }
inline uchar hydroforge_minimum(uchar left, uchar right) { return min(left, right); }
"""

_MATH = {name: name for name in ("sqrt", "exp", "log", "sin", "cos", "tan", "pow")}


def _c_type(node: Expr) -> torch.dtype | None:
    """The C type of ``node`` where no overload can change it."""

    match node:
        case Var() | Const() | Load() | Cast():
            return node.type
        case Unary(op="neg"):
            return _c_type(node.operand)
        case Binary():
            left = _c_type(node.left)
            return left if left == _c_type(node.right) else None
        case Select():
            positive = _c_type(node.positive)
            return positive if positive == _c_type(node.negative) else None
    return None


def _word(name: str) -> str:
    word = re.sub(r"\W", "_", name)
    return f"_{word}" if not word or word[0].isdigit() else word


def identifier(param: Param) -> str:
    """The C spelling of a parameter: ``p_<name>`` for buffers."""

    return _word(param.name) if param.access is None else f"p_{_word(param.name)}"


@dataclass(frozen=True, slots=True)
class _Trait:
    dialect: str
    functions: Mapping[str, str]
    cast: str
    atomic_add: str
    thread_index: str
    prelude: str
    nan: Callable[[str], str]
    entry: Callable[[KernelFunction, Sequence[str]], list[str]]


def _cuda_entry(function: KernelFunction, body: Sequence[str]) -> list[str]:
    params = []
    for param in function.params:
        spelled = element(param.type, "cuda")
        if param.access is None:
            params.append(f"{spelled} {identifier(param)}")
        else:
            const = "const " if param.access == "read" else ""
            params.append(f"{const}{spelled}* {identifier(param)}")
    return [
        f"__global__ void {function.name}(",
        _INDENT + f",\n{_INDENT}".join(params),
        ") {",
        *body,
        "}",
    ]


def _msl_entry(function: KernelFunction, body: Sequence[str]) -> list[str]:
    fields, locals_ = [], []
    for index, (name, access, native) in enumerate(msl_arguments(function)):
        if access is None:
            spelled = scalar(native, "msl")
            fields.append(f"    constant {spelled}* {name} [[id({index})]];")
            locals_.append(f"    const {spelled} {name} = *args.{name};")
            continue
        qualifier = "device const" if access == "read" else "device"
        fields.append(f"    {qualifier} {native}* {name} [[id({index})]];")
        locals_.append(f"    {qualifier} {native}* {name} = args.{name};")
    return [
        f"struct {function.name}_args {{",
        *fields,
        "};",
        "",
        f"kernel void {function.name}(",
        f"    constant {function.name}_args& args [[buffer(0)]],",
        "    uint tid [[thread_position_in_grid]]",
        ") {",
        *locals_,
        *body,
        "}",
    ]


def msl_arguments(function: KernelFunction) -> tuple[tuple[str, str | None, str], ...]:
    """The MSL argument buffer of ``function``: ``(field, access, native)``.

    ``native`` is a buffer's pointee type, atomic for ``atomic_add``, or a
    scalar's kind.
    """

    fields = []
    for param in function.params:
        if param.access is None:
            fields.append((identifier(param), None, scalar_kind(param.type)))
            continue
        native = element(param.type, "msl")
        if param.access == "atomic_add":
            native = f"atomic_{native}"
        fields.append((identifier(param), param.access, native))
    return tuple(fields)


class CPrinter:
    """Spell kernel IR in one C-family dialect."""

    def __init__(self, trait: _Trait) -> None:
        self.trait = trait

    def type(self, dtype: torch.dtype) -> str:
        return element(dtype, self.trait.dialect)

    def cast(self, text: str, dtype: torch.dtype) -> str:
        # Metal bool buffers use bytes, but casts must normalize truth values.
        logical_type = "bool" if dtype == torch.bool else self.type(dtype)
        return self.trait.cast.format(type=logical_type, value=text)

    def const(self, node: Const) -> str:
        value = node.value
        if isinstance(value, bool):
            return "true" if value else "false"
        if node.type is None:
            return repr(value)
        if isinstance(value, float) and math.isnan(value):
            return self.trait.nan(self.type(node.type))
        if isinstance(value, float) and math.isinf(value):
            return self.cast("INFINITY" if value > 0 else "-INFINITY", node.type)
        return self.cast(repr(value), node.type)

    def buffer(self, name: str) -> str:
        return f"p_{_word(name)}"

    def expr(self, node: Expr) -> str:
        match node:
            case Var():
                return node.name
            case Const():
                return self.const(node)
            case Load():
                return f"{self.buffer(node.buffer)}[{self.expr(node.index)}]"
            case Unary(op="neg"):
                return f"(-{self.expr(node.operand)})"
            case Unary():
                return f"(!{self.expr(node.operand)})"
            case Binary() | Compare():
                return f"({self.expr(node.left)} {node.op} {self.expr(node.right)})"
            case Logical():
                symbol = " && " if node.op == "and" else " || "
                return f"({symbol.join(self.expr(item) for item in node.operands)})"
            case Select():
                return (
                    f"(({self.expr(node.condition)}) ? ({self.expr(node.positive)}) "
                    f": ({self.expr(node.negative)}))"
                )
            case Cast():
                operand = self.expr(node.operand)
                if _c_type(node.operand) == node.type:
                    return operand
                return self.cast(operand, node.type)
            case Call(function="py_mod"):
                left, right = (self.expr(item) for item in node.arguments)
                remainder = f"fmod({left}, {right})"
                adjust = (
                    f"(({remainder} != 0.0) && "
                    f"(({remainder} < 0.0) != ({right} < 0.0)))"
                )
                return f"(({adjust}) ? ({remainder} + {right}) : ({remainder}))"
            case Call():
                function = self.trait.functions.get(node.function)
                if function is None:
                    raise TypeError(
                        f"{self.trait.dialect} has no intrinsic {node.function!r}"
                    )
                arguments = ", ".join(self.expr(item) for item in node.arguments)
                return f"{function}({arguments})"
            case PhaseTest():
                return f"(({PHASE.name} & {int(node.bits)}) != 0)"
            case ThreadIndex():
                return self.trait.thread_index
        raise TypeError(f"not an expression: {node!r}")

    def stmts(self, statements: Sequence[Stmt], depth: int = 1) -> list[str]:
        indent = _INDENT * depth
        lines: list[str] = []
        for node in statements:
            match node:
                case Let():
                    lines.append(
                        f"{indent}{self.type(node.var.type)} {node.var.name} = "
                        f"{self.expr(node.value)};"
                    )
                case Assign():
                    lines.append(f"{indent}{node.var.name} = {self.expr(node.value)};")
                case Store():
                    lines.append(
                        f"{indent}{self.buffer(node.buffer)}[{self.expr(node.index)}] "
                        f"= {self.expr(node.value)};"
                    )
                case AtomicAdd():
                    pointer = f"{self.buffer(node.buffer)} + {self.expr(node.index)}"
                    atomic = self.trait.atomic_add.format(
                        pointer=pointer, value=self.expr(node.value)
                    )
                    lines.append(f"{indent}{atomic};")
                case If():
                    lines.append(f"{indent}if ({self.expr(node.condition)}) {{")
                    lines.extend(self.stmts(node.then, depth + 1))
                    if node.orelse:
                        lines.append(f"{indent}}} else {{")
                        lines.extend(self.stmts(node.orelse, depth + 1))
                    lines.append(f"{indent}}}")
                case ForK():
                    var = node.var.name
                    lines.append(
                        f"{indent}for ({self.type(node.var.type)} {var} = 0; "
                        f"{var} < {node.count}; ++{var}) {{"
                    )
                    lines.extend(self.stmts(node.body, depth + 1))
                    lines.append(f"{indent}}}")
                case While():
                    lines.append(f"{indent}while ({self.expr(node.condition)}) {{")
                    lines.extend(self.stmts(node.body, depth + 1))
                    lines.append(f"{indent}}}")
                case Evaluate():
                    lines.append(f"{indent}{self.expr(node.call)};")
                case Guard():
                    lines.append(f"{indent}if (!{self.expr(node.condition)}) return;")
                case Block():
                    lines.append(f"{indent}{{")
                    lines.extend(self.stmts(node.body, depth + 1))
                    lines.append(f"{indent}}}")
                case _:
                    raise TypeError(f"not a statement: {node!r}")
        return lines

    def kernel(self, function: KernelFunction) -> str:
        names = [identifier(param) for param in function.params]
        shadowed = set(names).intersection(effects(function.body).defines)
        if len(set(names)) != len(names) or shadowed:
            raise ValueError(
                f"{function.name}: parameter names collide: {names}; "
                f"locals shadowing parameters: {sorted(shadowed)}"
            )
        body = self.stmts(function.body)
        return "\n".join(self.trait.entry(function, body))

    def program(self, functions: Sequence[KernelFunction]) -> str:
        """The prelude and every kernel of one compilation unit."""

        return "\n\n".join(
            (self.trait.prelude, *(self.kernel(item) for item in functions))
        )


CUDA = CPrinter(
    _Trait(
        dialect="cuda",
        functions={
            **_MATH,
            "abs": "fabs",
            "nan_max": "hf_max",
            "nan_min": "hf_min",
            "weighted_mean": "hf_weighted_mean",
            "isnan": "isnan",
            "set_conditional": "cudaGraphSetConditional",
        },
        cast="static_cast<{type}>({value})",
        atomic_add="atomicAdd({pointer}, {value})",
        thread_index="int64_t(blockIdx.x) * blockDim.x + threadIdx.x",
        prelude=_CUDA_PRELUDE,
        nan=lambda spelled: f"static_cast<{spelled}>(NAN)",
        entry=_cuda_entry,
    )
)
MSL = CPrinter(
    _Trait(
        dialect="msl",
        functions={
            **_MATH,
            "abs": "fabs",
            "nan_max": "hydroforge_maximum",
            "nan_min": "hydroforge_minimum",
            "weighted_mean": "hydroforge_weighted_mean",
            "isnan": "hydroforge_isnan",
        },
        cast="{type}({value})",
        atomic_add="atomic_fetch_add_explicit({pointer}, {value}, memory_order_relaxed)",
        thread_index="tid",
        prelude=_MSL_PRELUDE,
        nan=lambda spelled: "hydroforge_nan()",
        entry=_msl_entry,
    )
)

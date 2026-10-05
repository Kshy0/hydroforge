"""Explicit double-FP32 spelling of logical float64 framework IR.

This is deliberately separate from native MSL: a logical float64 is hf_hp
only when a caller also supplies encoded storage. Native float64 continues
to fail rather than silently accepting binary64 bytes as an FP32 pair.
"""

import math
import struct
from dataclasses import replace
from functools import cache, partial

import torch

from hydroforge.kernels.codegen.c import MSL, CPrinter, _msl_entry
from hydroforge.kernels.codegen.ir import Call, Cast, Const, Expr, type_of
from hydroforge.kernels.emulated import source


def _float(value: float) -> str:
    if math.isnan(value):
        return "as_type<float>(0x7fc00000u)"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    # Hex constants carry the exact component, including signed zero.
    return value.hex() + "f"


def _pair(value: float) -> tuple[float, float]:
    try:
        high = struct.unpack("=f", struct.pack("=f", value))[0]
    except OverflowError as error:
        raise OverflowError(
            "Metal float32x2 literal exceeds FP32 exponent range"
        ) from error
    if math.isfinite(value):
        if not math.isfinite(high) or (value != 0 and high == 0):
            raise OverflowError("Metal float32x2 literal exceeds FP32 exponent range")
        low = struct.unpack("=f", struct.pack("=f", value - high))[0]
    else:
        low = 0.0
    return high, low


class EmulatedMSLPrinter(CPrinter):
    def type(self, dtype: torch.dtype) -> str:
        return "hf_hp" if dtype == torch.float64 else super().type(dtype)

    def const(self, node: Const) -> str:
        if node.type is None and type(node.value) is float:
            return _float(node.value)
        if node.type == torch.float64:
            high, low = _pair(float(node.value))
            return f"hf_hp({_float(high)}, {_float(low)})"
        return super().const(node)

    def expr(self, node: Expr) -> str:
        if isinstance(node, Cast) and type_of(node.operand) == torch.float64:
            text = self.expr(node.operand)
            if node.type == torch.float64:
                return text
            conversion = {
                torch.int64: "hf_hp_to_long",
                torch.int32: "hf_hp_to_int",
                torch.bool: "hf_hp_to_bool",
            }.get(node.type)
            if conversion is not None:
                return f"{conversion}({text})"
            return self.cast(text, node.type)
        if isinstance(node, Call) and any(
            type_of(argument) == torch.float64 for argument in node.arguments
        ):
            function = {
                "abs": "hf_hp_abs",
                "sqrt": "hf_hp_sqrt",
                "exp": "hf_hp_exp",
                "log": "hf_hp_log",
                "sin": "hf_hp_sin",
                "cos": "hf_hp_cos",
                "tan": "hf_hp_tan",
                "pow": "hf_hp_pow",
                "py_mod": "hf_hp_mod",
                "nan_max": "hydroforge_maximum",
                "nan_min": "hydroforge_minimum",
                "weighted_mean": "hydroforge_weighted_mean",
                "isnan": "hydroforge_isnan",
            }.get(node.function)
            if function is None:
                raise TypeError(f"no float32x2 intrinsic for {node.function!r}")
            return f"{function}({', '.join(self.expr(arg) for arg in node.arguments)})"
        return super().expr(node)


@cache
def emulated_msl() -> EmulatedMSLPrinter:
    return EmulatedMSLPrinter(
        replace(
            MSL.trait,
            prelude=MSL.trait.prelude + "\n" + source(),
            entry=partial(_msl_entry, emulated=True),
        )
    )

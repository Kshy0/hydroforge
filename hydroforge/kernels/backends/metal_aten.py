# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Online Metal kernels for the compiled ATen operators of substep programs.

They stand in for torch operators of a physics body, so they compile with the
``aten`` origin: safe math and Metal's default floating-point contraction.
"""

from __future__ import annotations

from functools import cache

import torch

from hydroforge.core.errors import SubstepCompileError
from hydroforge.kernels.codegen.types import element
from hydroforge.kernels.metal import MetalBuffer, MetalScalar, online_program


@cache
def copy_program(dtype: torch.dtype):
    native = element(dtype, "msl")
    return online_program(
        f"hf_aten_copy_{native}",
        buffers=(
            MetalBuffer("output_ptr", dtype, "write"),
            MetalBuffer("input_ptr", dtype, "read"),
        ),
        scalars=(MetalScalar("n", "index"),),
        size="n",
        origin="aten",
        body="    if ((long)i < *args.n) args.output_ptr[i] = args.input_ptr[i];",
    )


@cache
def lerp_program():
    return online_program(
        "hf_aten_lerp_float",
        buffers=(
            MetalBuffer("input_ptr", torch.float32, "read"),
            MetalBuffer("end_ptr", torch.float32, "read"),
            MetalBuffer("weight_ptr", torch.float32, "read"),
            MetalBuffer("output_ptr", torch.float32, "write"),
        ),
        scalars=(MetalScalar("n", "index"),),
        size="n",
        origin="aten",
        body="""    if ((long)i < *args.n) {
        float start = args.input_ptr[i];
        float end = args.end_ptr[i];
        float weight = *args.weight_ptr;
        args.output_ptr[i] = abs(weight) < 0.5f
            ? start + weight * (end - start)
            : end - (end - start) * (1.0f - weight);
    }""",
    )


@cache
def scatter_add_program(index_dtype: torch.dtype):
    index_native = element(index_dtype, "msl")
    return online_program(
        f"hf_aten_scatter_add_{index_native}",
        buffers=(
            MetalBuffer("output_ptr", torch.float32, "atomic_add"),
            MetalBuffer("index_ptr", index_dtype, "read"),
            MetalBuffer("source_ptr", torch.float32, "read"),
            MetalBuffer("error_ptr", torch.int32, "atomic_write"),
        ),
        scalars=(
            MetalScalar("alpha", "float32"),
            MetalScalar("n", "index"),
            MetalScalar("output_n", "index"),
        ),
        size="n",
        origin="aten",
        body=f"""    if ((long)i < *args.n) {{
        {index_native} target = args.index_ptr[i];
        if (target < 0 || target >= *args.output_n) {{
            atomic_store_explicit(args.error_ptr, 1, memory_order_relaxed);
        }} else {{
            atomic_fetch_add_explicit(
                args.output_ptr + target, args.source_ptr[i] * *args.alpha,
                memory_order_relaxed);
        }}
    }}""",
    )


@cache
def zero_program(dtype: torch.dtype):
    native = element(dtype, "msl")
    return online_program(
        f"hf_aten_zero_{native}",
        buffers=(MetalBuffer("output_ptr", dtype, "write"),),
        scalars=(MetalScalar("n", "index"),),
        size="n",
        origin="aten",
        body="    if ((long)i < *args.n) args.output_ptr[i] = 0;",
    )


@cache
def fill_program(dtype: torch.dtype):
    kinds = {
        torch.bool: "bool",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "index",
    }
    if dtype not in kinds:
        raise SubstepCompileError(
            f"Metal fill_ lowering does not support scalar dtype {dtype}"
        )
    native = element(dtype, "msl")
    return online_program(
        f"hf_aten_fill_{native}",
        buffers=(MetalBuffer("output_ptr", dtype, "write"),),
        scalars=(
            MetalScalar("value", kinds[dtype]),
            MetalScalar("n", "index"),
        ),
        size="n",
        origin="aten",
        body="    if ((long)i < *args.n) args.output_ptr[i] = *args.value;",
    )


BINARY_EXPRESSIONS = {
    "add": "left + right",
    "sub": "left - right",
    "mul": "left * right",
    "div": "left / right",
    # torch.minimum propagates NaN and preserves the left operand on an
    # equality tie (including signed zero). MSL min/fmin is not that contract
    # on every Metal implementation, so spell it explicitly.
    "minimum": (
        "(isnan(left) || isnan(right)) ? "
        "as_type<float>(0x7fc00000u) : "
        "((right < left) ? right : left)"
    ),
    "lt": "left < right",
}


@cache
def binary_program(
    name: str,
    rhs_kind: str,
    result_dtype: torch.dtype,
    scaled: bool,
):
    try:
        expression = BINARY_EXPRESSIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown Metal pointwise operation {name!r}") from error
    buffers = [
        MetalBuffer("input_ptr", torch.float32, "read"),
        MetalBuffer("output_ptr", result_dtype, "write"),
    ]
    scalars = [MetalScalar("n", "index")]
    right = {
        "tensor": "args.rhs_ptr[i]",
        "tensor_scalar": "*args.rhs_ptr",
        "scalar": "*args.rhs",
    }[rhs_kind]
    if rhs_kind != "scalar":
        buffers.insert(1, MetalBuffer("rhs_ptr", torch.float32, "read"))
    else:
        scalars.insert(0, MetalScalar("rhs", "float32"))
    if scaled:
        scalars.insert(0, MetalScalar("alpha", "float32"))
        right = f"(*args.alpha * ({right}))"
    kernel_name = f"hf_aten_{name}_float_{rhs_kind}_{element(result_dtype, 'msl')}"
    body = f"""    if ((long)i < *args.n) {{
        float left = args.input_ptr[i];
        float right = {right};
        args.output_ptr[i] = {expression};
    }}"""
    return online_program(
        kernel_name,
        buffers=tuple(buffers),
        scalars=tuple(scalars),
        size="n",
        origin="aten",
        body=body,
    )

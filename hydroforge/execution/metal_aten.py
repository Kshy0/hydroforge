"""Strict online Metal lowering for ATen nodes in compiled substeps."""

from __future__ import annotations

from functools import cache
from typing import Any

import torch

from hydroforge.execution.aten import (
    COMPILED_ATEN, COMPILED_ATEN_CONTRACTS,
    normalize_fill_scalar, normalize_float32_scalar,
)
from hydroforge.execution.operators import SubstepCompileError
from hydroforge.kernels.backends.metal.online import (
    MetalBuffer, MetalCommand, MetalScalar, make_online_metal_dispatcher,
)
from hydroforge.kernels.backends.metal.types import tensor_type


@cache
def _copy_dispatcher(dtype: torch.dtype):
    native = tensor_type(dtype)
    return make_online_metal_dispatcher(
        f"hf_aten_copy_{native}",
        buffers=(
            MetalBuffer("output_ptr", dtype, "write"),
            MetalBuffer("input_ptr", dtype, "read"),
        ),
        scalars=(MetalScalar("n", "index"),),
        size_key="n",
        body="    if ((long)i < *args.n) "
        "args.output_ptr[i] = args.input_ptr[i];",
    )


@cache
def _lerp_dispatcher():
    return make_online_metal_dispatcher(
        "hf_aten_lerp_float",
        buffers=(
            MetalBuffer("input_ptr", torch.float32, "read"),
            MetalBuffer("end_ptr", torch.float32, "read"),
            MetalBuffer("weight_ptr", torch.float32, "read"),
            MetalBuffer("output_ptr", torch.float32, "write"),
        ),
        scalars=(MetalScalar("n", "index"),),
        size_key="n",
        body="""    if ((long)i < *args.n) {
        float start = args.input_ptr[i];
        args.output_ptr[i] = start + *args.weight_ptr * (args.end_ptr[i] - start);
    }""",
    )


@cache
def _scatter_dispatcher(index_dtype: torch.dtype):
    index_native = tensor_type(index_dtype)
    return make_online_metal_dispatcher(
        f"hf_aten_scatter_add_{index_native}",
        buffers=(
            MetalBuffer("output_ptr", torch.float32, "atomic_add"),
            MetalBuffer("index_ptr", index_dtype, "read"),
            MetalBuffer("source_ptr", torch.float32, "read"),
            MetalBuffer("error_ptr", torch.int32, "atomic_write"),
        ),
        scalars=(
            MetalScalar("alpha", "float32"), MetalScalar("n", "index"),
            MetalScalar("output_n", "index"),
        ),
        size_key="n",
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
def _zero_dispatcher(dtype: torch.dtype):
    native = tensor_type(dtype)
    return make_online_metal_dispatcher(
        f"hf_aten_zero_{native}",
        buffers=(MetalBuffer("output_ptr", dtype, "write"),),
        scalars=(MetalScalar("n", "index"),),
        size_key="n",
        body="    if ((long)i < *args.n) args.output_ptr[i] = 0;",
    )


@cache
def _fill_dispatcher(dtype: torch.dtype):
    kinds = {
        torch.bool: "bool", torch.float32: "float32",
        torch.int32: "int32", torch.int64: "index",
    }
    if dtype not in kinds:
        raise SubstepCompileError(
            f"Metal fill_ lowering does not support scalar dtype {dtype}"
        )
    native = tensor_type(dtype)
    return make_online_metal_dispatcher(
        f"hf_aten_fill_{native}",
        buffers=(MetalBuffer("output_ptr", dtype, "write"),),
        scalars=(
            MetalScalar("value", kinds[dtype]),
            MetalScalar("n", "index"),
        ),
        size_key="n",
        body="    if ((long)i < *args.n) "
        "args.output_ptr[i] = *args.value;",
    )


_BINARY_EXPRESSIONS = {
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
def _binary_dispatcher(
    name: str, rhs_kind: str, result_dtype: torch.dtype, scaled: bool,
):
    try:
        expression = _BINARY_EXPRESSIONS[name]
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
    kernel_name = (
        f"hf_aten_{name}_float_"
        f"{rhs_kind}_{tensor_type(result_dtype)}"
    )
    body = f"""    if ((long)i < *args.n) {{
        float left = args.input_ptr[i];
        float right = {right};
        args.output_ptr[i] = {expression};
    }}"""
    return make_online_metal_dispatcher(
        kernel_name, buffers=tuple(buffers), scalars=tuple(scalars),
        size_key="n", body=body,
    )


def _copy(output: torch.Tensor, source: torch.Tensor) -> MetalCommand:
    if output.dtype == torch.float64:
        raise SubstepCompileError(
            "Metal copy_ lowering requires a Metal-supported tensor dtype"
        )
    return MetalCommand(
        _copy_dispatcher(output.dtype),
        {"output_ptr": output, "input_ptr": source, "n": output.numel()},
    )


def _lower_copy(operator: Any) -> tuple[MetalCommand, ...]:
    args, kwargs, output = operator.static_values()
    del kwargs, output
    return (_copy(args[0], args[1]),)


def _lower_lerp(operator: Any) -> tuple[MetalCommand, ...]:
    args, kwargs, output = operator.static_values()
    start, end, weight = args[:3]
    destination = kwargs.get("out", output)
    if start.dtype != torch.float32:
        raise SubstepCompileError(
            "Metal lerp lowering requires float32 model precision"
        )
    return (MetalCommand(
        _lerp_dispatcher(),
        {
            "input_ptr": start, "end_ptr": end, "weight_ptr": weight,
            "output_ptr": destination, "n": destination.numel(),
        },
    ),)


def _lower_scatter(operator: Any) -> tuple[MetalCommand, ...]:
    args, kwargs, output = operator.static_values()
    name = operator.function._schema.name
    destination, dim, index, source = args[:4]
    del dim
    alpha = 1.0
    if name == "aten::index_add_":
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        alpha = normalize_float32_scalar("index_add_ alpha", alpha)
    calls: list[MetalCommand] = []
    target = destination if name.endswith("_") else output
    if not isinstance(target, torch.Tensor):
        raise SubstepCompileError("Metal scatter_add output is not address-stable")
    if target is not destination:
        calls.append(_copy(target, destination))
    error = torch.zeros(1, dtype=torch.int32, device=index.device)
    calls.append(MetalCommand(
        _scatter_dispatcher(index.dtype),
        {
            "output_ptr": target, "index_ptr": index,
            "source_ptr": source, "n": source.numel(),
            "output_n": target.numel(), "error_ptr": error, "alpha": alpha,
        },
        (error,),
    ))
    return tuple(calls)


def _lower_zero(operator: Any) -> tuple[MetalCommand, ...]:
    args, _kwargs, _output = operator.static_values()
    destination = args[0]
    if destination.dtype == torch.float64:
        raise SubstepCompileError(
            "Metal zero_ lowering requires a Metal-supported tensor dtype"
        )
    return (MetalCommand(
        _zero_dispatcher(destination.dtype),
        {"output_ptr": destination, "n": destination.numel()},
    ),)


def _lower_fill(operator: Any) -> tuple[MetalCommand, ...]:
    args, _kwargs, _output = operator.static_values()
    destination, value = args[:2]
    if destination.dtype == torch.float64:
        raise SubstepCompileError(
            "Metal fill_ lowering requires float32 model precision"
        )
    value = normalize_fill_scalar(destination.dtype, value)
    return (MetalCommand(
        _fill_dispatcher(destination.dtype),
        {
            "output_ptr": destination, "value": value,
            "n": destination.numel(),
        },
    ),)


def _lower_binary(operator: Any) -> tuple[MetalCommand, ...]:
    args, kwargs, output = operator.static_values()
    schema_name = operator.function._schema.name
    name = schema_name.removeprefix("aten::").removesuffix("_")
    left, right = args[:2]
    if left.dtype != torch.float32:
        raise SubstepCompileError(
            f"Metal {name} lowering requires float32 model precision"
        )
    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
    scaled = name in {"add", "sub"}
    if scaled:
        alpha = normalize_float32_scalar(f"{name} alpha", alpha)
    destination = left if schema_name.endswith("_") else output
    if not isinstance(destination, torch.Tensor):
        raise SubstepCompileError(f"Metal {name} output is not address-stable")
    result_dtype = torch.bool if name == "lt" else torch.float32
    rhs_kind = "tensor" if isinstance(right, torch.Tensor) else "scalar"
    arguments: dict[str, Any] = {
        "input_ptr": left, "output_ptr": destination,
        "n": destination.numel(),
    }
    if isinstance(right, torch.Tensor):
        if right.numel() == 1:
            rhs_kind = "tensor_scalar"
        arguments["rhs_ptr"] = right
    else:
        arguments["rhs"] = normalize_float32_scalar(name, right)
    if scaled:
        arguments["alpha"] = alpha
    return (MetalCommand(
        _binary_dispatcher(name, rhs_kind, result_dtype, scaled),
        arguments,
    ),)


_METAL_SEMANTIC_LOWERERS = {
    "binary": _lower_binary,
    "copy": _lower_copy,
    "fill": _lower_fill,
    "lerp": _lower_lerp,
    "scatter": _lower_scatter,
    "zero": _lower_zero,
}
_contract_semantics = {
    contract.semantics for contract in COMPILED_ATEN_CONTRACTS.values()
}
if _contract_semantics != set(_METAL_SEMANTIC_LOWERERS):
    missing = sorted(_contract_semantics.difference(_METAL_SEMANTIC_LOWERERS))
    extra = sorted(set(_METAL_SEMANTIC_LOWERERS).difference(_contract_semantics))
    raise RuntimeError(
        "compiled ATen semantics differ between graph and Metal backends: "
        f"missing={missing}, extra={extra}"
    )
_contract_binary_names = {
    name.removeprefix("aten::").removesuffix("_")
    for name, contract in COMPILED_ATEN_CONTRACTS.items()
    if contract.semantics == "binary"
}
if _contract_binary_names != set(_BINARY_EXPRESSIONS):
    missing = sorted(_contract_binary_names.difference(_BINARY_EXPRESSIONS))
    extra = sorted(set(_BINARY_EXPRESSIONS).difference(_contract_binary_names))
    raise RuntimeError(
        "compiled binary ATen contract differs from Metal expressions: "
        f"missing={missing}, extra={extra}"
    )
_METAL_ATEN_LOWERERS = {
    name: _METAL_SEMANTIC_LOWERERS[contract.semantics]
    for name, contract in COMPILED_ATEN_CONTRACTS.items()
}


def supports_metal_aten(name: str, overload: str) -> bool:
    """Return whether one exact ATen schema overload has an ICB lowering."""

    return (name, overload) in COMPILED_ATEN and name in _METAL_ATEN_LOWERERS


def lower_metal_aten(operator: Any) -> tuple[MetalCommand, ...]:
    """Lower one recorded ATen node; never return an eager fallback."""

    name = operator.function._schema.name
    try:
        lower = _METAL_ATEN_LOWERERS[name]
    except KeyError as error:
        raise SubstepCompileError(
            f"Torch operator {name!r} has no native Metal ICB lowering"
        ) from error
    return lower(operator)

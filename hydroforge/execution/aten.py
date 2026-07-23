"""Backend-owned ATen contracts for explicit compiled substeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch


FORBIDDEN_SUBSTEP_CONVERSIONS = frozenset({
    "bfloat16", "cpu", "cuda", "double", "float", "half", "mps",
    "to", "type", "type_as",
})
FORBIDDEN_SUBSTEP_CONSTRUCTORS = frozenset({
    "arange", "as_tensor", "empty", "full", "ones", "tensor", "zeros",
})


@dataclass(frozen=True, slots=True)
class CompiledAtenContract:
    """Exact overloads and backend-neutral value semantics for one operator."""

    semantics: str
    overloads: frozenset[str]
    preallocated: tuple[tuple[str, str], ...] = ()


def _contract(
    semantics: str,
    *overloads: str,
    preallocated: tuple[tuple[str, str], ...] = (),
) -> CompiledAtenContract:
    if semantics not in {"binary", "copy", "fill", "lerp", "scatter", "zero"}:
        raise ValueError(f"unknown compiled ATen semantics {semantics!r}")
    if not overloads or len(overloads) != len(set(overloads)):
        raise ValueError("compiled ATen overloads must be non-empty and unique")
    sources = tuple(source for source, _target in preallocated)
    targets = tuple(target for _source, target in preallocated)
    if (
        len(sources) != len(set(sources))
        or len(targets) != len(set(targets))
        or not set((*sources, *targets)).issubset(overloads)
    ):
        raise ValueError(
            "compiled ATen preallocated source/target overloads must be unique "
            "and supported"
        )
    return CompiledAtenContract(
        semantics, frozenset(overloads), preallocated,
    )


# This is the single support definition for Torch execution, CUDA graph capture
# and online Metal lowering. Adding a name requires choosing explicit semantics;
# there is no default validator category.
COMPILED_ATEN_CONTRACTS = MappingProxyType({
    "aten::add": _contract(
        "binary", "Tensor", "Scalar", "out", "Scalar_out",
        preallocated=(("Tensor", "out"), ("Scalar", "Scalar_out")),
    ),
    "aten::add_": _contract("binary", "Tensor", "Scalar"),
    "aten::copy_": _contract("copy", ""),
    "aten::div": _contract(
        "binary", "Tensor", "Scalar", "out", "Scalar_out",
        preallocated=(("Tensor", "out"), ("Scalar", "Scalar_out")),
    ),
    "aten::div_": _contract("binary", "Tensor", "Scalar"),
    "aten::fill_": _contract("fill", "Scalar"),
    "aten::index_add_": _contract("scatter", ""),
    "aten::lerp": _contract(
        "lerp", "Tensor", "Tensor_out",
        preallocated=(("Tensor", "Tensor_out"),),
    ),
    "aten::lt": _contract(
        "binary", "Tensor", "Scalar", "Tensor_out", "Scalar_out",
        preallocated=(("Tensor", "Tensor_out"), ("Scalar", "Scalar_out")),
    ),
    "aten::minimum": _contract(
        "binary", "", "out", preallocated=(("", "out"),),
    ),
    "aten::mul": _contract(
        "binary", "Tensor", "Scalar", "out", "Scalar_out",
        preallocated=(("Tensor", "out"), ("Scalar", "Scalar_out")),
    ),
    "aten::mul_": _contract("binary", "Tensor", "Scalar"),
    "aten::scatter_add": _contract(
        "scatter", "", "out", preallocated=(("", "out"),),
    ),
    "aten::scatter_add_": _contract("scatter", ""),
    "aten::sub": _contract(
        "binary", "Tensor", "Scalar", "out", "Scalar_out",
        preallocated=(("Tensor", "out"), ("Scalar", "Scalar_out")),
    ),
    "aten::sub_": _contract("binary", "Tensor", "Scalar"),
    "aten::zero_": _contract("zero", ""),
})
COMPILED_ATEN = frozenset(
    (name, overload)
    for name, contract in COMPILED_ATEN_CONTRACTS.items()
    for overload in contract.overloads
)
for _name, _contract_value in COMPILED_ATEN_CONTRACTS.items():
    if _name.endswith("_"):
        continue
    _covered = {
        overload
        for pair in _contract_value.preallocated
        for overload in pair
    }
    if _covered != _contract_value.overloads:
        raise RuntimeError(
            f"out-of-place compiled ATen {_name} must define exact "
            f"preallocated replay coverage: supported="
            f"{sorted(_contract_value.overloads)}, covered={sorted(_covered)}"
        )

COMPILED_ATEN_DTYPES = frozenset({
    torch.bool, torch.float32, torch.float64, torch.int32, torch.int64,
})


def preallocated_replay_overload(function: Any) -> Any | None:
    """Return the explicitly declared ``out=`` replay overload, if any."""

    schema_name = function._schema.name
    overload = function._schema.overload_name
    contract = COMPILED_ATEN_CONTRACTS.get(schema_name)
    if contract is None:
        return None
    target = dict(contract.preallocated).get(overload)
    if target is None:
        return None
    packet_name = schema_name.removeprefix("aten::")
    packet = getattr(torch.ops.aten, packet_name)
    return getattr(packet, target)


def _error(message: str) -> None:
    from hydroforge.execution.operators import SubstepCompileError

    raise SubstepCompileError(message)


def _require_contiguous(name: str, *tensors: torch.Tensor) -> None:
    bad = [tuple(tensor.shape) for tensor in tensors if not tensor.is_contiguous()]
    if bad:
        _error(f"Compiled ATen {name} requires contiguous tensors; got {bad}")


def _require_same_shape(name: str, *tensors: torch.Tensor) -> None:
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        _error(
            f"Compiled ATen {name} does not permit implicit broadcasting: "
            f"{sorted(shapes)}"
        )


def _require_tensor_scalar_shape(
    name: str, reference: torch.Tensor, scalar: torch.Tensor,
) -> None:
    """Reject singleton tensors whose broadcasting would increase rank."""

    if scalar.numel() != 1:
        _require_same_shape(name, reference, scalar)
        return
    if scalar.ndim > reference.ndim:
        _error(
            f"Compiled ATen {name} one-element tensor has shape "
            f"{tuple(scalar.shape)}, which would increase output rank from "
            f"{reference.ndim}; use a scalar tensor with at most that rank"
        )


def normalize_float32_scalar(name: str, value: Any) -> float:
    """Return the one canonical host representation used by all backends."""

    if type(value) not in {bool, int, float}:
        _error(f"Compiled ATen {name} scalar must be bool, int, or float")
    try:
        result = float(value)
    except OverflowError:
        _error(f"Compiled ATen {name} scalar is outside float32 range")
    if not math.isfinite(result) or abs(result) > torch.finfo(torch.float32).max:
        _error(
            f"Compiled ATen {name} scalar must be finite and within float32 range"
        )
    return result


def normalize_floating_scalar(
    name: str, value: Any, dtype: torch.dtype,
) -> float:
    """Normalize a scalar for a declared float32/float64 tensor contract."""

    if dtype == torch.float32:
        return normalize_float32_scalar(name, value)
    if dtype != torch.float64:
        _error(f"Compiled ATen {name} requires a floating tensor dtype")
    if type(value) not in {bool, int, float}:
        _error(f"Compiled ATen {name} scalar must be bool, int, or float")
    try:
        result = float(value)
    except OverflowError:
        _error(f"Compiled ATen {name} scalar is outside float64 range")
    if not math.isfinite(result) or abs(result) > torch.finfo(torch.float64).max:
        _error(
            f"Compiled ATen {name} scalar must be finite and within float64 range"
        )
    return result


def normalize_fill_scalar(dtype: torch.dtype, value: Any) -> bool | int | float:
    """Normalize ``fill_`` exactly once before backend lowering."""

    if type(value) not in {bool, int, float}:
        _error("Compiled ATen fill_ scalar must be bool, int, or float")
    if dtype in {torch.float32, torch.float64}:
        label = "float32" if dtype == torch.float32 else "float64"
        try:
            result = float(value)
        except OverflowError:
            _error(f"Compiled ATen fill_ scalar is outside {label} range")
        if not math.isfinite(result) or abs(result) > torch.finfo(dtype).max:
            _error(
                "Compiled ATen fill_ scalar must be finite and within "
                f"{label} range"
            )
        return result
    if dtype == torch.bool:
        return bool(value)
    if dtype in {torch.int32, torch.int64}:
        if isinstance(value, float) and not math.isfinite(value):
            _error("Compiled ATen fill_ cannot convert a non-finite integer value")
        result = int(value)
        bits = 32 if dtype == torch.int32 else 64
        if not -(2 ** (bits - 1)) <= result < 2 ** (bits - 1):
            _error(f"Compiled ATen fill_ scalar is outside int{bits} range")
        return result
    _error(f"Compiled ATen fill_ does not support dtype {dtype}")


def _validate_copy(args: tuple[Any, ...]) -> None:
    destination, source = args[:2]
    _require_contiguous("copy_", destination, source)
    _require_same_shape("copy_", destination, source)
    if destination.dtype != source.dtype:
        _error("Compiled ATen copy_ requires identical source/destination dtype")
    if destination.dtype not in COMPILED_ATEN_DTYPES:
        _error(f"Compiled ATen copy_ does not support dtype {destination.dtype}")


def _validate_lerp(
    args: tuple[Any, ...], kwargs: dict[str, Any], result: Any,
) -> None:
    start, end, weight = args[:3]
    destination = kwargs.get("out", result)
    tensors = (start, end, weight, destination)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        _error("Compiled ATen lerp requires tensor weight and output")
    _require_contiguous("lerp", *tensors)
    _require_same_shape("lerp", start, end, destination)
    if start.dtype not in {torch.float32, torch.float64}:
        _error("Compiled ATen lerp requires float32 or float64 tensors")
    if any(value.dtype != start.dtype for value in tensors):
        _error("Compiled ATen lerp requires one identical floating dtype")
    if weight.numel() != 1:
        _error("Compiled ATen lerp weight must contain exactly one value")
    _require_tensor_scalar_shape("lerp weight", start, weight)


def _validate_scatter(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any,
) -> None:
    destination, dim, index, source = args[:4]
    if dim != 0 or any(value.ndim != 1 for value in (destination, index, source)):
        _error(f"Compiled ATen {name} requires 1-D tensors and dim=0")
    _require_contiguous(name, destination, index, source)
    _require_same_shape(f"{name} index/source", index, source)
    if destination.dtype != torch.float32 or source.dtype != torch.float32:
        _error(f"Compiled ATen {name} requires float32 values")
    if index.dtype not in {torch.int32, torch.int64}:
        _error(f"Compiled ATen {name} index must be int32 or int64")
    target = destination if name.endswith("_") else result
    if not isinstance(target, torch.Tensor) or target.dtype != torch.float32:
        _error(f"Compiled ATen {name} output must be an address-stable float32 tensor")
    _require_same_shape(f"{name} destination/output", destination, target)
    if name == "index_add_":
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        normalize_float32_scalar("index_add_ alpha", alpha)


def _validate_binary(
    name: str, schema_name: str, args: tuple[Any, ...],
    kwargs: dict[str, Any], result: Any,
) -> None:
    left, right = args[:2]
    destination = left if schema_name.endswith("_") else result
    if not isinstance(left, torch.Tensor) or not isinstance(destination, torch.Tensor):
        _error(f"Compiled ATen {name} requires tensor input and output")
    _require_contiguous(name, left, destination)
    _require_same_shape(name, left, destination)
    if left.dtype not in {torch.float32, torch.float64}:
        _error(f"Compiled ATen {name} requires float32 or float64 input")
    expected = torch.bool if name == "lt" else left.dtype
    if destination.dtype != expected:
        _error(f"Compiled ATen {name} output must be {expected}")
    if isinstance(right, torch.Tensor):
        _require_contiguous(name, right)
        _require_tensor_scalar_shape(name, left, right)
        if right.dtype != left.dtype:
            _error(f"Compiled ATen {name} tensor operands must have one dtype")
    else:
        normalize_floating_scalar(name, right, left.dtype)
    if name in {"add", "sub"}:
        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        normalize_floating_scalar(f"{name} alpha", alpha, left.dtype)
    if name == "div":
        rounding_mode = kwargs.get(
            "rounding_mode", args[2] if len(args) > 2 else None,
        )
        if rounding_mode is not None:
            _error("Compiled ATen div does not support rounding_mode")


def validate_compiled_aten(
    function: Any, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any,
) -> None:
    """Validate the backend-neutral value contract for one recorded node."""

    schema_name = function._schema.name
    overload = function._schema.overload_name
    contract = COMPILED_ATEN_CONTRACTS.get(schema_name)
    if contract is None or overload not in contract.overloads:
        _error(f"Torch operator {schema_name}.{overload} is not compiled")
    name = schema_name.removeprefix("aten::").removesuffix("_")
    if contract.semantics == "copy":
        _validate_copy(args)
    elif contract.semantics == "lerp":
        _validate_lerp(args, kwargs, result)
    elif contract.semantics == "scatter":
        _validate_scatter(
            schema_name.removeprefix("aten::"), args, kwargs, result,
        )
    elif contract.semantics == "zero":
        destination = args[0]
        _require_contiguous("zero_", destination)
        if destination.dtype not in COMPILED_ATEN_DTYPES:
            _error(f"Compiled ATen zero_ does not support dtype {destination.dtype}")
    elif contract.semantics == "fill":
        destination, value = args[:2]
        _require_contiguous("fill_", destination)
        normalize_fill_scalar(destination.dtype, value)
    elif contract.semantics == "binary":
        _validate_binary(name, schema_name, args, kwargs, result)
    else:
        raise RuntimeError(
            f"unhandled compiled ATen semantics {contract.semantics!r}"
        )


def supports_aten(execution: Any, name: str, overload: str) -> bool:
    """Query the selected substep compiler's explicit operator contract."""

    if execution.capture_mode == "metal_icb":
        from hydroforge.execution.metal_aten import supports_metal_aten

        return supports_metal_aten(name, overload)
    return (name, overload) in COMPILED_ATEN

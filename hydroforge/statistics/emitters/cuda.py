# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from hydroforge.kernels.backends.cuda import rtc
from hydroforge.serialization.files import atomic_write_text
from hydroforge.statistics.emitters.common import StatisticsEmitter
from hydroforge.statistics.emitters.expression import (
    ExpressionDialect,
    render_expression,
)
from hydroforge.statistics.ir import (
    Expression,
    ExpressionSource,
    ScatterSource,
    TensorSource,
)
from hydroforge.statistics.lowering import OutputLayout

_FLOAT_DTYPES = {
    torch.float32: ("float", "at::kFloat"),
    torch.float64: ("double", "at::kDouble"),
}
_INT_DTYPES = {
    torch.bool: ("bool", "at::kBool"),
    torch.int32: ("int32_t", "at::kInt"),
    torch.int64: ("int64_t", "at::kLong"),
}
_CONTROL_FLOAT_KEYS = (
    "__weight",
    "__total_weight",
)
_INTEGER_SCALAR_TYPES = {
    "__num_macro_steps": ("int64_t", "at::kLong"),
    "__sub_step": ("int32_t", "at::kInt"),
    "__num_sub_steps": ("int32_t", "at::kInt"),
    "__flags": ("int32_t", "at::kInt"),
    "__macro_step_index": ("int64_t", "at::kLong"),
}


def _scalar_types(dtype: torch.dtype) -> dict[str, tuple[str, str]]:
    floating = _FLOAT_DTYPES[dtype]
    return {
        **{key: floating for key in _CONTROL_FLOAT_KEYS},
        **_INTEGER_SCALAR_TYPES,
    }


def _kernel_start(name: str, params: Sequence[str]) -> list[str]:
    return [f"__global__ void {name}(", "    " + ",\n    ".join(params), ") {"]


_GATE_PARAMS = (
    "const int32_t* p___sub_step",
    "const int32_t* p___num_sub_steps",
    "const int32_t* p___flags",
)
_GATE_KEYS = ("__sub_step", "__num_sub_steps", "__flags")
_TORCH_DTYPES = {
    "at::kFloat": torch.float32,
    "at::kDouble": torch.float64,
    "at::kBool": torch.bool,
    "at::kInt": torch.int32,
    "at::kLong": torch.int64,
}


@dataclass(frozen=True, slots=True)
class _Points:
    """The saved-point count of an output index, known once buffers bind."""

    key: str


@dataclass(frozen=True, slots=True)
class _Launch:
    """One kernel launch; arguments name buffers, point counts or int64 values."""

    kernel: str
    args: tuple[Any, ...]
    total: tuple[_Points | None, int]
    mask: int | None


def _argument(item: Any, states: Mapping[str, torch.Tensor]) -> rtc.KernelArgument:
    if isinstance(item, str):
        return rtc.pointer(states[item])
    if isinstance(item, _Points):
        return rtc.int64(states[item.key].numel())
    return rtc.int64(item)


def _bind(
    request: rtc.RtcRequest,
    params: Sequence[dict[str, Any]],
    launches: Sequence[_Launch],
    states: Mapping[str, torch.Tensor],
    block_size: int,
) -> list[tuple[int | None, Any]]:
    """Validate the bound buffers once and pre-pack every launch."""
    if not 1 <= block_size <= 1024:
        raise RuntimeError("CUDA statistics block_size must be in [1, 1024]")
    owner = None
    for param in params:
        key = param["key"]
        tensor = states[key]
        if not tensor.is_cuda:
            raise RuntimeError(f"{key} must be a CUDA/HIP tensor")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{key} must be contiguous")
        if tensor.dtype != param["dtype"]:
            raise RuntimeError(f"{key} has unexpected dtype")
        if owner is None:
            owner = tensor.device
        elif tensor.device != owner:
            raise RuntimeError(
                f"{key} must be on the same CUDA/HIP device as all statistics buffers"
            )
    bound = []
    for launch in launches:
        points, factor = launch.total
        count = factor * (1 if points is None else states[points.key].numel())
        if count == 0:
            continue
        step = rtc.CudaLaunch(
            launch.kernel,
            rtc.blocks(count, block_size),
            block_size,
            tuple(_argument(item, states) for item in launch.args),
        )
        bound.append((launch.mask, rtc.prepare(request, (step,), owner.index)))
    return bound


def _device_gate(mask: int | None) -> list[str]:
    """Return early from a pre-kernel whose consumers skip this sample."""
    if mask is None:
        return []
    terms = []
    if mask & 1:
        terms.append("((p___flags[0] & 1) != 0 && p___sub_step[0] == 0)")
    if mask & 2:
        terms.append(
            "(((p___flags[0] >> 1) & 1) != 0 && "
            "p___sub_step[0] == p___num_sub_steps[0] - 1)"
        )
    return [f"    if (!({' || '.join(terms) or 'false'})) return;"]


def _c_ident(name: str) -> str:
    ident = re.sub(r"\W", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


class CudaStatisticsEmitter(StatisticsEmitter):
    """Runtime-compiled CUDA/HIP kernels for statistics aggregation.

    The generated device source is compiled by NVRTC (hiprtc under ROCm); the
    launches are described here, validated once per bound state mapping and
    replayed with pre-packed arguments.
    """

    def emit(self):
        self._generate_cuda_aggregator_function()
        return self.result()

    def _generate_cuda_aggregator_function(self) -> None:
        source, params, launches = self._generate_cuda_program()
        program = rtc.RtcProgram(source, (), "hydroforge_statistics")
        request = rtc.RtcRequest(program, tuple(launch.kernel for launch in launches))
        device = torch.device(self.device).index
        if device is None:
            device = torch.cuda.current_device()
        rtc.compile_request(request, device)
        host_phase = self._host_phase
        bound: list[Any] = []

        def internal_update_statistics(states, BLOCK_SIZE):
            # Rebinding follows a replaced state mapping or block size; the
            # mapping itself keeps every bound tensor alive.
            if not bound or bound[0] is not states or bound[1] != BLOCK_SIZE:
                bound[:] = [
                    states,
                    BLOCK_SIZE,
                    _bind(request, params, launches, states, BLOCK_SIZE),
                ]
            phase = -1 if host_phase is None else host_phase.bits
            for mask, launch in bound[2]:
                if mask is None or phase < 0 or phase & mask:
                    launch()

        self._aggregator_function = internal_update_statistics
        if self.save_kernels:
            self._save_cuda_kernel_file(source)

    def _generate_cuda_program(
        self,
    ) -> tuple[str, list[dict[str, Any]], list[_Launch]]:
        """Return device source, bound buffers in parameter order and launches."""
        params: dict[str, dict[str, Any]] = {}

        def add_param(key: str, ctype: str, scalar_type: str, *, const: bool) -> None:
            ident = _c_ident(key)
            old = params.get(key)
            if old is not None:
                old["const"] = old["const"] and const
                return
            params[key] = {
                "key": key,
                "ident": ident,
                "ptr": f"p_{ident}",
                "ctype": ctype,
                "dtype": _TORCH_DTYPES[scalar_type],
                "const": const,
            }

        for key, (ctype, scalar_type) in _scalar_types(
            self._control_dtype,
        ).items():
            add_param(key, ctype, scalar_type, const=True)

        specs = self._build_cuda_specs(add_param)
        lines: list[str] = [
            "template <typename T> __device__ inline T hf_max(T a, T b) { return a > b ? a : b; }",
            "template <typename T> __device__ inline T hf_min(T a, T b) { return a < b ? a : b; }",
            "template <> __device__ inline float hf_max<float>(float a, float b) { if (isnan(a)) return b; if (isnan(b)) return a; return a > b ? a : b; }",
            "template <> __device__ inline float hf_min<float>(float a, float b) { if (isnan(a)) return b; if (isnan(b)) return a; return a < b ? a : b; }",
            "template <> __device__ inline double hf_max<double>(double a, double b) { if (isnan(a)) return b; if (isnan(b)) return a; return a > b ? a : b; }",
            "template <> __device__ inline double hf_min<double>(double a, double b) { if (isnan(a)) return b; if (isnan(b)) return a; return a < b ? a : b; }",
            "// Incremental form bounds FP32 drift; non-finite results keep the",
            "// blended form's infinity/NaN propagation.",
            "template <typename T> __device__ inline T hf_weighted_mean(T old_v, T old_w, T value, T weight) { T new_w = old_w + weight; T ratio = weight / new_w; T incremental = old_v + (value - old_v) * ratio; return isfinite(incremental) ? incremental : old_v * (old_w / new_w) + value * ratio; }",
            "template <typename T> __device__ inline T hf_neg_inf() { return static_cast<T>(-INFINITY); }",
            "template <typename T> __device__ inline T hf_pos_inf() { return static_cast<T>(INFINITY); }",
            "",
        ]

        for scatter in specs["scatters"]:
            lines.extend(self._generate_scatter_kernel(scatter))
            lines.append("")

        for group in specs["groups"]:
            lines.extend(self._generate_group_kernel(group))
            lines.append("")

        ordered = sorted(params.values(), key=lambda param: param["key"])
        return "\n".join(lines), ordered, self._cuda_launches(specs)

    def _build_cuda_specs(
        self,
        add_param,
    ) -> dict[str, Any]:
        def add_tensor_param(key: str, *, const: bool) -> None:
            ctype, scalar_type = self._state_ctype(key)
            add_param(key, ctype, scalar_type, const=const)

        scatters = []
        for variable in self._statistics_ir.ordered_scatters():
            scatter = variable.source
            var_name = variable.name
            buf_key = f"__scatter_buf_{var_name}"
            add_tensor_param(buf_key, const=False)
            cnt_key = (
                f"__scatter_cnt_{var_name}"
                if scatter.reduction.value == "mean"
                else None
            )
            if cnt_key:
                add_tensor_param(cnt_key, const=False)
            add_tensor_param(scatter.index, const=True)
            for key in self._statistics_ir.scatter_inputs(var_name):
                if key != scatter.index:
                    add_tensor_param(key, const=True)

            scatters.append(
                {
                    "name": var_name,
                    "safe": self._get_safe_name(var_name),
                    "scatter": scatter,
                    "buf_key": buf_key,
                    "cnt_key": cnt_key,
                    "source_size": self._tensor_registry[scatter.index].numel(),
                    "target_size": self._storage[buf_key].shape[-1],
                    "ensemble_size": self.ensemble_size
                    if self._statistics_layouts[var_name].batched
                    else 1,
                    "ctype": self._state_ctype(buf_key)[0],
                    "mask": self._statistics_lowering.scatter_phase_mask(var_name),
                }
            )

        groups = []
        for output_index, var_list in self._statistics_lowering.groups.items():
            full_output = output_index == "__full__"
            if not full_output:
                add_tensor_param(output_index, const=True)
            group_vars = []
            max_levels = 1
            full_total = 0
            for var in var_list:
                variable = self._statistics_lowering.by_name[var]
                is_2d = variable.layout is OutputLayout.INDEXED_LEVEL
                n_levels = variable.variable.actual_shape[-1] if is_2d else 1
                max_levels = max(max_levels, n_levels)
                var_numel = (
                    prod(self._statistics_layouts[var].actual_shape)
                    if full_output
                    else 0
                )
                full_total = max(full_total, var_numel)
                for key in self._statistics_ir.materialized_inputs(var):
                    add_tensor_param(key, const=True)

                ops = []
                for operation in variable.operations:
                    op = operation.spelling
                    out_key = f"{var}_{op}"
                    add_tensor_param(out_key, const=False)
                    info = {
                        "base": operation.outer.value,
                        "is_arg": operation.stores_index,
                        "k": operation.k,
                    }
                    aux_key = None
                    if info["is_arg"]:
                        aux_key = f"{var}_{operation.spelling}_aux"
                        add_tensor_param(aux_key, const=False)
                    ops.append(
                        {
                            "op": op,
                            "out_key": out_key,
                            "outer": info,
                            "inner": (
                                operation.inner.value
                                if operation.inner is not None
                                else None
                            ),
                            "aux_key": aux_key,
                        }
                    )

                inner_ops = sorted(
                    {
                        operation.inner.value
                        for operation in variable.operations
                        if operation.inner is not None
                    }
                )
                inner_states = {}
                for inner in inner_ops:
                    if inner == "last":
                        continue
                    state_key = f"{var}_{inner}_inner_state"
                    add_tensor_param(state_key, const=False)
                    inner_states[inner] = {"state_key": state_key}
                    if inner == "mean":
                        weight_key = f"{var}_{inner}_weight_state"
                        add_tensor_param(weight_key, const=False)
                        inner_states[inner]["weight_key"] = weight_key

                sample_weight_key = None
                if any(
                    operation.inner is None and operation.outer.value == "mean"
                    for operation in variable.operations
                ):
                    sample_weight_key = f"{var}_mean_sample_weight_state"
                    add_tensor_param(sample_weight_key, const=False)

                stride_input = self._statistics_layouts[var].stride_input

                group_vars.append(
                    {
                        "name": var,
                        "safe": self._get_safe_name(var),
                        "ctype": self._value_ctype(var),
                        "is_2d": is_2d,
                        "n_levels": n_levels,
                        "stride_input": stride_input,
                        "ops": ops,
                        "inner_ops": inner_ops,
                        "inner_states": inner_states,
                        "sample_weight_key": sample_weight_key,
                        "numel": var_numel,
                    }
                )

            payload = json.dumps(
                {
                    "output_index": output_index,
                    "vars": [
                        {
                            "name": v["name"],
                            "ops": [op["op"] for op in v["ops"]],
                            "n_levels": v["n_levels"],
                        }
                        for v in group_vars
                    ],
                    "ensemble_size": self.ensemble_size,
                    "full_output": full_output,
                },
                sort_keys=True,
            )
            groups.append(
                {
                    "output_index": output_index,
                    "kernel_name": f"hf_aggr_{_c_ident(output_index)}_{hashlib.sha1(payload.encode()).hexdigest()[:10]}",
                    "vars": group_vars,
                    "max_levels": max_levels,
                    "ensemble_size": self.ensemble_size,
                    "full_output": full_output,
                    "full_total": full_total,
                    "mask": self._statistics_lowering.group_phase_mask(output_index),
                }
            )

        return {"groups": groups, "scatters": scatters}

    def _value_ctype(self, name: str) -> str:
        if name in self._tensor_registry:
            return self._state_ctype(name)[0]
        if f"__scatter_buf_{name}" in self._storage:
            return self._state_ctype(f"__scatter_buf_{name}")[0]
        dtype = self._statistics_layouts[name].dtype
        return (_FLOAT_DTYPES.get(dtype) or _INT_DTYPES[dtype])[0]

    def _state_ctype(self, key: str) -> tuple[str, str]:
        if key in _CONTROL_FLOAT_KEYS:
            return _FLOAT_DTYPES[self._control_dtype]
        if key in _INTEGER_SCALAR_TYPES:
            return _INTEGER_SCALAR_TYPES[key]
        tensor = self._tensor_registry.get(key)
        if tensor is None:
            tensor = self._storage[key]
        if tensor.dtype in _FLOAT_DTYPES:
            return _FLOAT_DTYPES[tensor.dtype]
        return _INT_DTYPES[tensor.dtype]

    def _value_expr(
        self,
        name: str,
        lines: list[str],
        emitted: dict[str, str],
        *,
        context: str,
        is_2d: bool,
        n_levels: int,
        ctype: str,
        full_variable: str | None = None,
    ) -> str:
        key = f"{context}:{name}:{is_2d}:{n_levels}"
        if key in emitted:
            return emitted[key]

        safe = _c_ident(f"{context}_{name}_val_{len(emitted)}")
        source = self._statistics_ir.sources.get(name) or TensorSource(name)

        if isinstance(source, (TensorSource, ScatterSource)):
            input_key = (
                source.name
                if isinstance(source, TensorSource)
                else f"__scatter_buf_{name}"
            )
            stride = self._source_stride(input_key, logical_rank=2 if is_2d else 1)
            if context == "scatter":
                offset = f"t * {stride} + src"
            elif context == "full":
                offset = self._full_source_offset(
                    input_key, full_variable or name, "linear"
                )
            elif is_2d:
                offset = f"(t * {stride} + idx) * {n_levels} + level"
            else:
                offset = f"t * {stride} + idx"
            lines.append(
                f"    {ctype} {safe} = static_cast<{ctype}>(p_{_c_ident(input_key)}[{offset}]);"
            )
        elif isinstance(source, ExpressionSource):
            names = {
                dependency: self._value_expr(
                    dependency,
                    lines,
                    emitted,
                    context=context,
                    is_2d=is_2d,
                    n_levels=n_levels,
                    ctype=ctype,
                    full_variable=full_variable,
                )
                for dependency in source.expression.dependencies
            }
            rendered = render_expression(
                source.expression,
                ExpressionDialect.CUDA,
                names,
                value_type=("float32" if ctype == "float" else "float64"),
            )
            lines.append(f"    {ctype} {safe} = static_cast<{ctype}>({rendered});")

        emitted[key] = safe
        return safe

    def _generate_scatter_kernel(self, scatter_spec: dict[str, Any]) -> list[str]:
        scatter = scatter_spec["scatter"]
        safe = scatter_spec["safe"]
        ctype = scatter_spec["ctype"]
        cnt_ctype = (
            self._state_ctype(scatter_spec["cnt_key"])[0]
            if scatter_spec["cnt_key"]
            else None
        )
        index_ctype = self._state_ctype(scatter.index)[0]
        zero_params = [f"{ctype}* p_{_c_ident(scatter_spec['buf_key'])}"]
        if scatter_spec["cnt_key"]:
            zero_params.append(f"{cnt_ctype}* p_{_c_ident(scatter_spec['cnt_key'])}")
        gated = scatter_spec["mask"] is not None
        gate = _device_gate(scatter_spec["mask"])
        gate_params = list(_GATE_PARAMS) if gated else []
        zero_params.extend([*gate_params, "int64_t total"])
        lines = _kernel_start(f"hf_scatter_zero_{safe}", zero_params)
        lines.extend(
            [
                *gate,
                "    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (linear >= total) return;",
                f"    p_{_c_ident(scatter_spec['buf_key'])}[linear] = static_cast<{ctype}>(0);",
            ]
        )
        if scatter_spec["cnt_key"]:
            lines.append(
                f"    p_{_c_ident(scatter_spec['cnt_key'])}[linear] = "
                f"static_cast<{cnt_ctype}>(0);"
            )
        lines.extend(["}", ""])

        add_params = [
            f"{ctype}* p_{_c_ident(scatter_spec['buf_key'])}",
            f"const {index_ctype}* p_{_c_ident(scatter.index)}",
        ]
        if scatter_spec["cnt_key"]:
            add_params.append(f"{cnt_ctype}* p_{_c_ident(scatter_spec['cnt_key'])}")
        for key in self._statistics_ir.scatter_inputs(scatter_spec["name"]):
            if key != scatter.index:
                pc, _ = self._state_ctype(key)
                add_params.append(f"const {pc}* p_{_c_ident(key)}")
        add_params = list(dict.fromkeys(add_params))
        add_params.extend(
            [
                *gate_params,
                "int64_t source_size",
                "int64_t target_size",
                "int64_t ensemble_size",
            ]
        )
        lines.extend(_kernel_start(f"hf_scatter_add_{safe}", add_params))
        lines.extend(
            [
                *gate,
                "    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;",
                "    int64_t source_total = source_size * ensemble_size;",
                "    if (linear >= source_total) return;",
                "    int64_t t = linear / source_size;",
                "    int64_t src = linear - t * source_size;",
                f"    int64_t dst = static_cast<int64_t>(p_{_c_ident(scatter.index)}[src]);",
                "    if (dst < 0 || dst >= target_size) return;",
            ]
        )
        emitted: dict[str, str] = {}
        val = self._scatter_value_expr(scatter.value, lines, emitted, ctype)
        lines.append(
            f"    atomicAdd(p_{_c_ident(scatter_spec['buf_key'])} + t * target_size + dst, {val});"
        )
        if scatter_spec["cnt_key"]:
            lines.append(
                f"    atomicAdd(p_{_c_ident(scatter_spec['cnt_key'])} + "
                f"t * target_size + dst, static_cast<{cnt_ctype}>(1));"
            )
        lines.append("}")

        if scatter_spec["cnt_key"]:
            div_params = [
                f"{ctype}* p_{_c_ident(scatter_spec['buf_key'])}",
                f"const {cnt_ctype}* p_{_c_ident(scatter_spec['cnt_key'])}",
                *gate_params,
                "int64_t total",
            ]
            lines.append("")
            lines.extend(_kernel_start(f"hf_scatter_divide_{safe}", div_params))
            lines.extend(
                [
                    *gate,
                    "    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;",
                    "    if (linear >= total) return;",
                    f"        {cnt_ctype} cnt = p_{_c_ident(scatter_spec['cnt_key'])}[linear];",
                    "        if (cnt > 0) {",
                    f"            p_{_c_ident(scatter_spec['buf_key'])}[linear] = p_{_c_ident(scatter_spec['buf_key'])}[linear] / static_cast<{ctype}>(cnt);",
                    "        } else {",
                    f"            p_{_c_ident(scatter_spec['buf_key'])}[linear] = static_cast<{ctype}>(NAN);",
                    "        }",
                    "    }",
                ]
            )
        return lines

    def _scatter_value_expr(
        self,
        expression: Expression,
        lines: list[str],
        emitted: dict[str, str],
        ctype: str,
    ) -> str:
        names = {
            dependency: self._value_expr(
                dependency,
                lines,
                emitted,
                context="scatter",
                is_2d=False,
                n_levels=1,
                ctype=ctype,
            )
            for dependency in expression.dependencies
        }
        return render_expression(
            expression,
            ExpressionDialect.CUDA,
            names,
            value_type=("float32" if ctype == "float" else "float64"),
        )

    def _group_parameters(self, group: dict[str, Any]) -> dict[str, bool]:
        """Share pointer order and mutability between kernels and their launchers."""
        params: dict[str, bool] = {}

        def add(key: str, *, const: bool) -> None:
            params[key] = params.get(key, True) and const

        if not group["full_output"]:
            add(group["output_index"], const=True)
        for var in group["vars"]:
            for key in self._statistics_ir.materialized_inputs(var["name"]):
                add(key, const=True)
            for op in var["ops"]:
                add(op["out_key"], const=False)
                if op["aux_key"]:
                    add(op["aux_key"], const=False)
            for state in var["inner_states"].values():
                for key in state.values():
                    add(key, const=False)
            if var["sample_weight_key"] is not None:
                add(var["sample_weight_key"], const=False)
        for key in _scalar_types(self._control_dtype):
            add(key, const=True)
        return params

    def _generate_group_kernel(self, group: dict[str, Any]) -> list[str]:
        full_output = group["full_output"]
        params = [
            f"{'const ' if const else ''}{self._state_ctype(key)[0]}* p_{_c_ident(key)}"
            for key, const in self._group_parameters(group).items()
        ]
        params.extend(
            ["int64_t n_elements"]
            if full_output
            else ["int64_t n_saved_points", "int64_t ensemble_size"]
        )
        lines = _kernel_start(group["kernel_name"], params)
        lines.append("    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;")
        if full_output:
            lines.append("    if (linear >= n_elements) return;")
        else:
            lines.extend(
                [
                    f"    int64_t max_levels = {group['max_levels']};",
                    "    int64_t total = n_saved_points * ensemble_size * max_levels;",
                    "    if (linear >= total) return;",
                ]
            )
        for key, (ctype, _) in _scalar_types(self._control_dtype).items():
            lines.append(
                f"    {ctype} {key.removeprefix('__')} = p_{_c_ident(key)}[0];"
            )
        lines.extend(
            [
                "    bool is_inner_first = ((flags & 1) != 0) && (sub_step == 0);",
                "    bool is_inner_last = (((flags >> 1) & 1) != 0) && (sub_step == num_sub_steps - 1);",
                "    bool is_outer_first = (((flags >> 2) & 1) != 0) && is_inner_last;",
                "    bool is_outer_last = (((flags >> 3) & 1) != 0) && is_inner_last;",
                "",
            ]
        )
        if full_output:
            for var in group["vars"]:
                self._emit_group_variable(
                    lines, var, f"linear < {var['numel']}", "linear", full=True
                )
            lines.append("}")
            return lines
        # Vector outputs use one thread per point and member; only level
        # outputs spread over the level axis, so no vector lane idles.
        index = f"p_{_c_ident(group['output_index'])}"
        for is_2d in (False, True):
            variables = [var for var in group["vars"] if var["is_2d"] == is_2d]
            if not variables:
                continue
            point = "linear / max_levels" if is_2d else "linear"
            lines.extend(
                [
                    "    {",
                    f"    int64_t point_linear = {point};",
                    "    int64_t t = point_linear / n_saved_points;",
                    "    int64_t offs = point_linear - t * n_saved_points;",
                    *(["    int64_t level = linear % max_levels;"] if is_2d else []),
                    "    if (t < ensemble_size) {",
                    f"    int64_t idx = static_cast<int64_t>({index}[offs]);",
                ]
            )
            for var in variables:
                condition = f"level < {var['n_levels']}" if is_2d else "true"
                if not self._statistics_layouts[var["name"]].batched:
                    condition = f"({condition}) && t == 0"
                self._emit_group_variable(
                    lines, var, condition, self._out_offset_expr(var), full=False
                )
            lines.extend(["    }", "    }"])
        lines.append("}")
        return lines

    def _emit_group_variable(
        self,
        lines: list[str],
        var: dict[str, Any],
        condition: str,
        out_offset: str,
        *,
        full: bool,
    ) -> None:
        lines.append(f"    if ({condition}) {{")
        lines.append(f"        int64_t out_off = {out_offset};")
        val = self._value_expr(
            var["name"],
            lines,
            {},
            context="full" if full else "group",
            is_2d=False if full else var["is_2d"],
            n_levels=1 if full else var["n_levels"],
            ctype=var["ctype"],
            full_variable=var["name"] if full else None,
        )
        lines.append(f"        {var['ctype']} val = {val};")
        lines.extend(self._generate_inner_updates(var))
        for op in var["ops"]:
            lines.extend(self._generate_op_update(var, op))
        lines.extend(["    }", ""])

    def _out_offset_expr(self, var: dict[str, Any]) -> str:
        if var["is_2d"]:
            return f"(t * n_saved_points + offs) * {var['n_levels']} + level"
        return "t * n_saved_points + offs"

    def _generate_inner_updates(self, var: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        ctype = var["ctype"]
        for inner in var["inner_ops"]:
            if inner == "last":
                lines.append(f"        {ctype} val_for_{inner} = val;")
                continue
            state = var["inner_states"][inner]
            ptr = f"p_{_c_ident(state['state_key'])}"
            acc = self._state_ctype(state["state_key"])[0]
            lines.append(f"        {acc} val_for_{inner} = static_cast<{acc}>(0);")
            if inner == "mean":
                wptr = f"p_{_c_ident(state['weight_key'])}"
                lines.extend(
                    [
                        "        {",
                        f"            {acc} old_v = is_inner_first ? static_cast<{acc}>(0) : {ptr}[out_off];",
                        f"            {acc} old_w = is_inner_first ? static_cast<{acc}>(0) : {wptr}[out_off];",
                        f"            {acc} new_w = old_w + static_cast<{acc}>(weight);",
                        f"            {acc} new_v = hf_weighted_mean(old_v, old_w, static_cast<{acc}>(val), static_cast<{acc}>(weight));",
                        f"            if (is_inner_last) {{ val_for_{inner} = new_v; {ptr}[out_off] = 0; {wptr}[out_off] = 0; }}",
                        f"            else {{ {ptr}[out_off] = new_v; {wptr}[out_off] = new_w; }}",
                        "        }",
                    ]
                )
            elif inner == "sum":
                lines.extend(
                    [
                        "        {",
                        f"            {acc} old_v = is_inner_first ? static_cast<{acc}>(0) : {ptr}[out_off];",
                        f"            {acc} new_v = old_v + static_cast<{acc}>(val) * static_cast<{acc}>(weight);",
                        f"            if (is_inner_last) {{ val_for_{inner} = new_v; {ptr}[out_off] = 0; }}",
                        f"            else {{ {ptr}[out_off] = new_v; }}",
                        "        }",
                    ]
                )
            elif inner in {"max", "min"}:
                fn = "hf_max" if inner == "max" else "hf_min"
                reset = (
                    f"hf_neg_inf<{ctype}>()"
                    if inner == "max"
                    else f"hf_pos_inf<{ctype}>()"
                )
                lines.extend(
                    [
                        "        {",
                        f"            {ctype} old_v = {ptr}[out_off];",
                        f"            {ctype} new_v = is_inner_first ? val : {fn}(old_v, val);",
                        f"            if (is_inner_last) {{ val_for_{inner} = new_v; {ptr}[out_off] = {reset}; }}",
                        f"            else {{ {ptr}[out_off] = new_v; }}",
                        "        }",
                    ]
                )
            elif inner == "first":
                lines.extend(
                    [
                        f"        if (is_inner_first) {{ {ptr}[out_off] = val; }}",
                        f"        if (is_inner_last) {{ val_for_{inner} = {ptr}[out_off]; }}",
                    ]
                )
        return lines

    def _generate_op_update(self, var: dict[str, Any], op: dict[str, Any]) -> list[str]:
        outer = op["outer"]
        compound = op["inner"] is not None
        out = f"p_{_c_ident(op['out_key'])}"
        ctype = var["ctype"]
        value = (
            "val" if not compound or op["inner"] == "last" else f"val_for_{op['inner']}"
        )
        # State, weight and control scalars all use the value's own dtype.
        acc = self._state_ctype(op["out_key"])[0]
        guard = "        if (is_inner_last) {\n" if compound else ""
        end_guard = "        }\n" if compound else ""
        indent = "            " if compound else "        "
        lines: list[str] = []

        if outer["k"] > 1:
            body = self._topk_update(var, op, value, indent)
            body = [f"{indent}{{"] + body + [f"{indent}}}"]
            if guard:
                lines.append(guard.rstrip())
                lines.extend(body)
                lines.append(end_guard.rstrip())
            else:
                lines.extend(body)
            return lines

        if outer["is_arg"]:
            aux = f"p_{_c_ident(op['aux_key'])}"
            cmp = ">" if outer["base"] == "max" else "<"
            reset = "is_outer_first" if compound else "is_inner_first"
            if self._statistics_layouts[var["name"]].dtype.is_floating_point:
                body = [
                    f"{indent}{ctype} candidate = {value};",
                    f"{indent}if ({reset}) {{ {out}[out_off] = -1; {aux}[out_off] = static_cast<{ctype}>(NAN); }}",
                    f"{indent}{ctype} old_v = {aux}[out_off];",
                    f"{indent}if (!isnan(candidate) && (isnan(old_v) || candidate {cmp} old_v)) {{ {aux}[out_off] = candidate; {out}[out_off] = macro_step_index; }}",
                ]
            else:
                body = [
                    f"{indent}if ({reset}) {{",
                    f"{indent}    {out}[out_off] = macro_step_index;",
                    f"{indent}    {aux}[out_off] = {value};",
                    f"{indent}}} else {{",
                    f"{indent}    {ctype} old_v = {aux}[out_off];",
                    f"{indent}    if ({value} {cmp} old_v) {{ {aux}[out_off] = {value}; {out}[out_off] = macro_step_index; }}",
                    f"{indent}}}",
                ]
        elif outer["base"] == "mean":
            if not compound:
                weight_ptr = f"p_{_c_ident(var['sample_weight_key'])}"
                body = [
                    f"{indent}{acc} old_v = is_inner_first ? static_cast<{acc}>(0) : {out}[out_off];",
                    f"{indent}{acc} old_w = is_inner_first ? static_cast<{acc}>(0) : {weight_ptr}[out_off];",
                    f"{indent}{acc} new_w = old_w + static_cast<{acc}>(weight);",
                    f"{indent}{out}[out_off] = hf_weighted_mean(old_v, old_w, static_cast<{acc}>({value}), static_cast<{acc}>(weight));",
                    f"{indent}{weight_ptr}[out_off] = is_inner_last ? static_cast<{acc}>(0) : new_w;",
                ]
            else:
                body = [
                    f"{indent}{acc} count = static_cast<{acc}>(num_macro_steps);",
                    f"{indent}{acc} candidate = static_cast<{acc}>({value});",
                    f"{indent}{out}[out_off] = is_outer_first ? candidate : hf_weighted_mean({out}[out_off], count - static_cast<{acc}>(1), candidate, static_cast<{acc}>(1));",
                ]
        elif outer["base"] == "sum":
            reset = "is_outer_first" if compound else "is_inner_first"
            weighted = (
                f"static_cast<{acc}>({value})"
                if compound
                else f"static_cast<{acc}>({value}) * static_cast<{acc}>(weight)"
            )
            body = [
                f"{indent}{acc} old_v = {reset} ? static_cast<{acc}>(0) : {out}[out_off];",
                f"{indent}{out}[out_off] = old_v + {weighted};",
            ]
        elif outer["base"] in {"max", "min"}:
            fn = "hf_max" if outer["base"] == "max" else "hf_min"
            reset = "is_outer_first" if compound else "is_inner_first"
            body = [
                f"{indent}if ({reset}) {{ {out}[out_off] = {value}; }}",
                f"{indent}else {{ {out}[out_off] = {fn}({out}[out_off], {value}); }}",
            ]
        elif outer["base"] == "first":
            cond = "is_outer_first" if compound else "is_inner_first"
            body = [f"{indent}if ({cond}) {{ {out}[out_off] = {value}; }}"]
        elif outer["base"] == "last":
            cond = "true" if compound else "is_inner_last"
            body = [f"{indent}if ({cond}) {{ {out}[out_off] = {value}; }}"]
        body = [f"{indent}{{"] + body + [f"{indent}}}"]
        if guard:
            lines.append(guard.rstrip())
            lines.extend(body)
            lines.append(end_guard.rstrip())
        else:
            lines.extend(body)
        return lines

    def _topk_update(
        self, var: dict[str, Any], op: dict[str, Any], value: str, indent: str
    ) -> list[str]:
        outer = op["outer"]
        k = outer["k"]
        ctype = var["ctype"]
        is_max = outer["base"] == "max"
        cmp = ">" if is_max else "<"
        missing = f"static_cast<{ctype}>(NAN)"
        reset = "is_inner_first" if "_" not in op["op"] else "is_outer_first"
        if outer["is_arg"]:
            out = f"p_{_c_ident(op['out_key'])}"
            aux = f"p_{_c_ident(op['aux_key'])}"
            return [
                f"{indent}int64_t k_base = out_off * {k};",
                f"{indent}{ctype} new_v = {value};",
                f"{indent}int64_t new_i = macro_step_index;",
                f"{indent}if ({reset}) {{",
                f"{indent}    for (int kk = 0; kk < {k}; ++kk) {{ {aux}[k_base + kk] = {missing}; {out}[k_base + kk] = -1; }}",
                f"{indent}}}",
                f"{indent}for (int kk = 0; kk < {k}; ++kk) {{",
                f"{indent}    {ctype} old_v = {aux}[k_base + kk];",
                f"{indent}    int64_t old_i = {out}[k_base + kk];",
                f"{indent}    if (!isnan(new_v) && (isnan(old_v) || new_v {cmp} old_v || (new_v == old_v && new_i < old_i))) {{ {aux}[k_base + kk] = new_v; {out}[k_base + kk] = new_i; new_v = old_v; new_i = old_i; }}",
                f"{indent}}}",
            ]
        out = f"p_{_c_ident(op['out_key'])}"
        return [
            f"{indent}int64_t k_base = out_off * {k};",
            f"{indent}{ctype} new_v = {value};",
            f"{indent}if ({reset}) {{",
            f"{indent}    for (int kk = 0; kk < {k}; ++kk) {{ {out}[k_base + kk] = {missing}; }}",
            f"{indent}}}",
            f"{indent}for (int kk = 0; kk < {k}; ++kk) {{",
            f"{indent}    {ctype} old_v = {out}[k_base + kk];",
            f"{indent}    if (!isnan(new_v) && (isnan(old_v) || new_v {cmp} old_v)) {{ {out}[k_base + kk] = new_v; new_v = old_v; }}",
            f"{indent}}}",
        ]

    def _cuda_launches(self, specs: dict[str, Any]) -> list[_Launch]:
        """Scatter zero/add/divide launches precede each output group's kernel."""
        launches = []
        for scatter in specs["scatters"]:
            gate = _GATE_KEYS if scatter["mask"] is not None else ()
            target = scatter["target_size"] * scatter["ensemble_size"]
            buffers = (
                scatter["buf_key"],
                *((scatter["cnt_key"],) if scatter["cnt_key"] else ()),
            )
            launches.append(
                _Launch(
                    f"hf_scatter_zero_{scatter['safe']}",
                    (*buffers, *gate, target),
                    (None, target),
                    scatter["mask"],
                )
            )
            inputs = [scatter["buf_key"], scatter["scatter"].index]
            if scatter["cnt_key"]:
                inputs.append(scatter["cnt_key"])
            inputs.extend(
                key
                for key in self._statistics_ir.scatter_inputs(scatter["name"])
                if key != scatter["scatter"].index
            )
            launches.append(
                _Launch(
                    f"hf_scatter_add_{scatter['safe']}",
                    (
                        *dict.fromkeys(inputs),
                        *gate,
                        scatter["source_size"],
                        scatter["target_size"],
                        scatter["ensemble_size"],
                    ),
                    (None, scatter["source_size"] * scatter["ensemble_size"]),
                    scatter["mask"],
                )
            )
            if scatter["cnt_key"]:
                launches.append(
                    _Launch(
                        f"hf_scatter_divide_{scatter['safe']}",
                        (*buffers, *gate, target),
                        (None, target),
                        scatter["mask"],
                    )
                )
        for group in specs["groups"]:
            args: tuple[Any, ...] = tuple(self._group_parameters(group))
            if group["full_output"]:
                args += (group["full_total"],)
                total = (None, group["full_total"])
            else:
                points = _Points(group["output_index"])
                args += (points, group["ensemble_size"])
                total = (points, group["ensemble_size"] * group["max_levels"])
            launches.append(_Launch(group["kernel_name"], args, total, group["mask"]))
        return launches

    def _save_cuda_kernel_file(self, source: str) -> None:
        unique_name = self._generate_unique_name()
        self._saved_kernel_file = self.kernels_dir / f"kern_cuda_{unique_name}.cu"
        atomic_write_text(self._saved_kernel_file, rtc.RTC_PRELUDE + source)

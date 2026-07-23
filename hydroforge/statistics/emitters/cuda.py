# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import torch

from hydroforge.serialization.files import atomic_write_text

from hydroforge.statistics.ir import (
    Expression, ExpressionDialect, ExpressionSource, ScatterSource,
    TensorSource, render_expression,
)
from hydroforge.statistics.lowering import OutputLayout
from hydroforge.statistics.emitters.common import StatisticsEmitter

if TYPE_CHECKING:
    from hydroforge.statistics.runtime import StatisticsRuntime


_FLOAT_DTYPES = {
    torch.float32: ("float", "at::kFloat"),
    torch.float64: ("double", "at::kDouble"),
}
_INT_DTYPES = {
    torch.bool: ("bool", "at::kBool"),
    torch.int32: ("int32_t", "at::kInt"),
    torch.int64: ("int64_t", "at::kLong"),
}
_SCALAR_TYPES = {
    "__weight": ("float", "at::kFloat"),
    "__total_weight": ("float", "at::kFloat"),
    "__num_macro_steps": ("float", "at::kFloat"),
    "__sub_step": ("int32_t", "at::kInt"),
    "__num_sub_steps": ("int32_t", "at::kInt"),
    "__flags": ("int32_t", "at::kInt"),
    "__macro_step_index": ("int32_t", "at::kInt"),
}


def _c_ident(name: str) -> str:
    ident = re.sub(r"\W", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


def _dtype_info(tensor: torch.Tensor, *, floating: bool) -> Tuple[str, str]:
    table = _FLOAT_DTYPES if floating else _INT_DTYPES
    if tensor.dtype not in table:
        kind = "floating" if floating else "integer"
        raise TypeError(f"unsupported {kind} dtype {tensor.dtype}")
    return table[tensor.dtype]


class CudaStatisticsEmitter(StatisticsEmitter):
    """CUDA C++ extension code generation for statistics aggregation.

    The generated ``.cu`` source is compiled with NVCC on NVIDIA GPUs and,
    when ``HYDROFORGE_BACKEND=cuda`` is selected under a ROCm PyTorch build,
    hipified and compiled with hipcc by :func:`load_inline_cu_module` — so AMD
    users who want a compiled aggregator get one without a separate HIP path.
    """

    def emit(self):
        self._generate_cuda_aggregator_function()
        return self.result()

    def _generate_cuda_aggregator_function(
        self: StatisticsRuntime,
    ) -> None:
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        cpp_sources, cuda_sources = self._generate_cuda_extension_sources()

        from hydroforge.kernels.backends.cuda.build import load_inline_cu_module

        digest = hashlib.sha256((cpp_sources + "\n" + cuda_sources).encode()).hexdigest()[:12]
        module_name = f"hydroforge_cuda_aggregator_r{self.rank}_{digest}"
        ext = load_inline_cu_module(
            module_name,
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            functions=["launch_update"],
            extra_cuda_cflags=("-O3", "--use_fast_math"),
        )

        def internal_update_statistics(states, BLOCK_SIZE):
            if type(BLOCK_SIZE) is not int or not 1 <= BLOCK_SIZE <= 1024:
                raise ValueError(
                    "CUDA statistics BLOCK_SIZE must be an exact int in "
                    f"[1, 1024], got {BLOCK_SIZE!r}"
                )
            ext.launch_update(states, BLOCK_SIZE)

        self._kernel_module = ext
        self._aggregator_function = internal_update_statistics
        self._aggregator_generated = True

        if self.save_kernels:
            self._save_cuda_kernel_file(cpp_sources, cuda_sources)

    def _generate_cuda_extension_sources(
        self: StatisticsRuntime,
    ) -> Tuple[str, str]:
        params: Dict[str, Dict[str, str]] = {}

        def add_param(key: str, ctype: str, scalar_type: str, *, const: bool) -> None:
            ident = _c_ident(key)
            old = params.get(key)
            if old is not None:
                if old["ctype"] != ctype:
                    raise TypeError(f"state '{key}' is used with incompatible dtypes")
                old["const"] = old["const"] and const
                return
            params[key] = {
                "key": key,
                "ident": ident,
                "ptr": f"p_{ident}",
                "ctype": ctype,
                "scalar_type": scalar_type,
                "const": const,
            }

        for key, (ctype, scalar_type) in _SCALAR_TYPES.items():
            add_param(key, ctype, scalar_type, const=True)

        specs = self._build_cuda_specs(add_param)

        cpp_sources = "\n".join(
            [
                "#include <torch/extension.h>",
                "#include <pybind11/pybind11.h>",
                "namespace py = pybind11;",
                "void launch_update(py::dict states, long block_size);",
                "",
            ]
        )
        lines: List[str] = [
            "#include <torch/extension.h>",
            "#include <pybind11/pybind11.h>",
            "#include <cuda_runtime.h>",
            "#include <c10/cuda/CUDAStream.h>",
            "#include <cmath>",
            "#include <cstdint>",
            "",
            "#ifndef M_PI",
            "#define M_PI 3.14159265358979323846",
            "#endif",
            "",
            "namespace py = pybind11;",
            "",
            "template <typename T> __device__ inline T hf_max(T a, T b) { return a > b ? a : b; }",
            "template <typename T> __device__ inline T hf_min(T a, T b) { return a < b ? a : b; }",
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

        lines.extend(self._generate_launcher(specs, list(params.values())))
        return cpp_sources, "\n".join(lines)

    def _build_cuda_specs(
        self: StatisticsRuntime,
        add_param,
    ) -> Dict[str, Any]:
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
                if scatter.reduction.value == "mean" else None
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
                    "source_size": int(self._tensor_registry[scatter.index].numel()),
                    "target_size": int(self._storage[buf_key].shape[-1]),
                    "num_trials": int(self.num_trials),
                    "ctype": self._state_ctype(buf_key)[0],
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
                n_levels = (
                    int(variable.variable.actual_shape[-1]) if is_2d else 1
                )
                max_levels = max(max_levels, n_levels)
                var_numel = int(self._tensor_registry[var].numel()) if full_output else 0
                full_total = max(full_total, var_numel)
                self._add_value_params(var, add_tensor_param)

                ops = []
                for operation in variable.operations:
                    op = operation.spelling
                    out_key = f"{var}_{op}"
                    add_tensor_param(out_key, const=False)
                    info = {
                        "outer": operation.outer.value,
                        "base": operation.outer.value,
                        "is_arg": operation.stores_index,
                        "k": operation.k,
                    }
                    aux_key = None
                    if info["is_arg"]:
                        aux_key = f"{var}_{info['base']}{info['k'] if info['k'] > 1 else ''}_aux"
                        add_tensor_param(aux_key, const=False)
                    ops.append(
                        {
                            "op": op,
                            "out_key": out_key,
                            "outer": info,
                            "inner": (
                                operation.inner.value
                                if operation.inner is not None else None
                            ),
                            "aux_key": aux_key,
                        }
                    )

                inner_ops = sorted({
                    operation.inner.value
                    for operation in variable.operations
                    if operation.inner is not None
                })
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

                stride_input = 0
                for meta in self._metadata.values():
                    if meta["original_variable"] == var:
                        stride_input = int(meta.get("stride_input", 0))
                        break

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
                    "num_trials": self.num_trials,
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
                    "num_trials": int(self.num_trials),
                    "full_output": full_output,
                    "full_total": full_total,
                }
            )

        return {"groups": groups, "scatters": scatters}

    def _value_ctype(self: StatisticsRuntime, name: str) -> str:
        if name in self._tensor_registry:
            return self._state_ctype(name)[0]
        if f"__scatter_buf_{name}" in self._storage:
            return self._state_ctype(f"__scatter_buf_{name}")[0]
        variable = self._statistics_lowering.by_name.get(name)
        for operation in variable.operations if variable is not None else ():
            if operation.stores_index:
                suffix = str(operation.k) if operation.k > 1 else ""
                aux_key = f"{name}_{operation.outer.value}{suffix}_aux"
                if aux_key in self._storage:
                    return self._state_ctype(aux_key)[0]
            out_key = f"{name}_{operation.spelling}"
            if out_key in self._storage and self._storage[out_key].dtype in _FLOAT_DTYPES:
                return self._state_ctype(out_key)[0]
        return "float"

    def _state_ctype(self: StatisticsRuntime, key: str) -> Tuple[str, str]:
        if key in _SCALAR_TYPES:
            return _SCALAR_TYPES[key]
        tensor = self._tensor_registry.get(key)
        if tensor is None:
            tensor = self._storage[key]
        if tensor.dtype in _FLOAT_DTYPES:
            return _FLOAT_DTYPES[tensor.dtype]
        return _INT_DTYPES[tensor.dtype]

    def _add_value_params(self: StatisticsRuntime, name: str, add_tensor_param) -> None:
        for key in self._statistics_ir.materialized_inputs(name):
            add_tensor_param(key, const=True)

    def _value_expr(
        self: StatisticsRuntime,
        name: str,
        lines: List[str],
        emitted: Dict[str, str],
        *,
        context: str,
        is_2d: bool,
        n_levels: int,
        ctype: str,
    ) -> str:
        key = f"{context}:{name}:{is_2d}:{n_levels}"
        if key in emitted:
            return emitted[key]

        safe = _c_ident(f"{context}_{name}_val_{len(emitted)}")
        source = self._statistics_ir.sources.get(name, TensorSource(name))

        if context == "scatter":
            offset = "src_off"
        elif is_2d:
            offset = f"(t * {self._stride_expr(name)} + idx) * {n_levels} + level"
        else:
            offset = f"t * {self._stride_expr(name)} + idx"

        if isinstance(source, TensorSource):
            lines.append(f"    {ctype} {safe} = static_cast<{ctype}>(p_{_c_ident(name)}[{offset}]);")
        elif isinstance(source, ScatterSource):
            buf = f"__scatter_buf_{name}"
            lines.append(f"    {ctype} {safe} = static_cast<{ctype}>(p_{_c_ident(buf)}[{offset}]);")
        elif isinstance(source, ExpressionSource):
            names = {
                dependency: self._value_expr(
                    dependency, lines, emitted, context=context, is_2d=is_2d,
                    n_levels=n_levels, ctype=ctype,
                )
                for dependency in source.expression.dependencies
            }
            rendered = render_expression(
                source.expression, ExpressionDialect.CUDA, names,
            )
            lines.append(f"    {ctype} {safe} = static_cast<{ctype}>({rendered});")

        emitted[key] = safe
        return safe

    def _stride_expr(self: StatisticsRuntime, name: str) -> str:
        if self.num_trials <= 1:
            return "0"
        source = self._statistics_ir.sources.get(name, TensorSource(name))
        if isinstance(source, ScatterSource):
            return str(int(self._storage[f"__scatter_buf_{name}"].shape[-1]))
        for key in self._statistics_ir.materialized_inputs(name):
            tensor = self._tensor_registry.get(key)
            if tensor is not None:
                return str(int(tensor.shape[1])) if tensor.ndim >= 2 else "0"
        return "0"

    def _generate_scatter_kernel(self: StatisticsRuntime, scatter_spec: Dict[str, Any]) -> List[str]:
        scatter = scatter_spec["scatter"]
        safe = scatter_spec["safe"]
        ctype = scatter_spec["ctype"]
        index_ctype = self._state_ctype(scatter.index)[0]
        zero_params = [f"{ctype}* p_{_c_ident(scatter_spec['buf_key'])}"]
        if scatter_spec["cnt_key"]:
            zero_params.append(f"{ctype}* p_{_c_ident(scatter_spec['cnt_key'])}")
        zero_params.append("long total")
        lines = [f"__global__ void hf_scatter_zero_{safe}("]
        for i, param in enumerate(zero_params):
            comma = "," if i < len(zero_params) - 1 else ""
            lines.append(f"    {param}{comma}")
        lines.extend(
            [
                ") {",
                "    long linear = blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (linear >= total) return;",
                f"    p_{_c_ident(scatter_spec['buf_key'])}[linear] = static_cast<{ctype}>(0);",
            ]
        )
        if scatter_spec["cnt_key"]:
            lines.append(f"    p_{_c_ident(scatter_spec['cnt_key'])}[linear] = static_cast<{ctype}>(0);")
        lines.extend(["}", ""])

        add_params = [
            f"{ctype}* p_{_c_ident(scatter_spec['buf_key'])}",
            f"const {index_ctype}* p_{_c_ident(scatter.index)}",
        ]
        if scatter_spec["cnt_key"]:
            add_params.append(f"{ctype}* p_{_c_ident(scatter_spec['cnt_key'])}")
        for key in self._statistics_ir.scatter_inputs(scatter_spec["name"]):
            if key != scatter.index:
                pc, _ = self._state_ctype(key)
                add_params.append(f"const {pc}* p_{_c_ident(key)}")
        add_params.extend(["long source_size", "long target_size", "long num_trials"])
        add_params = list(dict.fromkeys(add_params))
        lines.append(f"__global__ void hf_scatter_add_{safe}(")
        for i, param in enumerate(add_params):
            comma = "," if i < len(add_params) - 1 else ""
            lines.append(f"    {param}{comma}")
        lines.extend(
            [
                ") {",
                "    long linear = blockIdx.x * blockDim.x + threadIdx.x;",
                "    long source_total = source_size * num_trials;",
                "    if (linear >= source_total) return;",
                "    long t = linear / source_size;",
                "    long src = linear - t * source_size;",
                "    long src_off = num_trials > 1 ? t * source_size + src : src;",
                f"    long dst = static_cast<long>(p_{_c_ident(scatter.index)}[src]);",
            ]
        )
        emitted: Dict[str, str] = {}
        val = self._scatter_value_expr(scatter.value, lines, emitted, ctype)
        lines.append(f"    atomicAdd(p_{_c_ident(scatter_spec['buf_key'])} + t * target_size + dst, {val});")
        if scatter_spec["cnt_key"]:
            lines.append(f"    atomicAdd(p_{_c_ident(scatter_spec['cnt_key'])} + t * target_size + dst, static_cast<{ctype}>(1));")
        lines.append("}")

        if scatter_spec["cnt_key"]:
            div_params = [
                f"{ctype}* p_{_c_ident(scatter_spec['buf_key'])}",
                f"const {ctype}* p_{_c_ident(scatter_spec['cnt_key'])}",
                "long total",
            ]
            lines.extend(["", f"__global__ void hf_scatter_divide_{safe}("])
            for i, param in enumerate(div_params):
                comma = "," if i < len(div_params) - 1 else ""
                lines.append(f"    {param}{comma}")
            lines.extend(
                [
                    ") {",
                    "    long linear = blockIdx.x * blockDim.x + threadIdx.x;",
                    "    if (linear >= total) return;",
                    f"        {ctype} cnt = p_{_c_ident(scatter_spec['cnt_key'])}[linear];",
                    "        if (cnt > 0) {",
                    f"            p_{_c_ident(scatter_spec['buf_key'])}[linear] = p_{_c_ident(scatter_spec['buf_key'])}[linear] / cnt;",
                    "        }",
                    "    }",
                ]
            )
        return lines

    def _scatter_value_expr(
        self: StatisticsRuntime,
        expression: Expression,
        lines: List[str],
        emitted: Dict[str, str],
        ctype: str,
    ) -> str:
        names = {
            dependency: self._value_expr(
                dependency, lines, emitted, context="scatter", is_2d=False,
                n_levels=1, ctype=ctype,
            )
            for dependency in expression.dependencies
        }
        return render_expression(expression, ExpressionDialect.CUDA, names)

    def _collect_value_param_keys(self: StatisticsRuntime, name: str) -> List[str]:
        return list(self._statistics_ir.materialized_inputs(name))

    def _generate_group_kernel(self: StatisticsRuntime, group: Dict[str, Any]) -> List[str]:
        if group.get("full_output"):
            return self._generate_full_group_kernel(group)

        params = [f"const {self._state_ctype(group['output_index'])[0]}* p_{_c_ident(group['output_index'])}"]
        for var in group["vars"]:
            for key in self._collect_value_param_keys(var["name"]):
                ctype, _ = self._state_ctype(key)
                params.append(f"const {ctype}* p_{_c_ident(key)}")
            for op in var["ops"]:
                out_ctype, _ = self._state_ctype(op["out_key"])
                params.append(f"{out_ctype}* p_{_c_ident(op['out_key'])}")
                if op["aux_key"]:
                    aux_ctype, _ = self._state_ctype(op["aux_key"])
                    params.append(f"{aux_ctype}* p_{_c_ident(op['aux_key'])}")
            for state in var["inner_states"].values():
                ctype, _ = self._state_ctype(state["state_key"])
                params.append(f"{ctype}* p_{_c_ident(state['state_key'])}")
                if "weight_key" in state:
                    wc, _ = self._state_ctype(state["weight_key"])
                    params.append(f"{wc}* p_{_c_ident(state['weight_key'])}")
        for key, (ctype, _) in _SCALAR_TYPES.items():
            params.append(f"const {ctype}* p_{_c_ident(key)}")
        params.extend(["long n_saved_points", "long num_trials"])
        params = list(dict.fromkeys(params))

        lines = [f"__global__ void {group['kernel_name']}("]
        for i, param in enumerate(params):
            comma = "," if i < len(params) - 1 else ""
            lines.append(f"    {param}{comma}")
        lines.extend(
            [
                ") {",
                "    long linear = blockIdx.x * blockDim.x + threadIdx.x;",
                f"    long max_levels = {group['max_levels']};",
                "    long total = n_saved_points * num_trials * max_levels;",
                "    if (linear >= total) return;",
                "    long level = linear % max_levels;",
                "    long point_linear = linear / max_levels;",
                "    long t = point_linear / n_saved_points;",
                "    long offs = point_linear - t * n_saved_points;",
                f"    long idx = static_cast<long>(p_{_c_ident(group['output_index'])}[offs]);",
                "    float weight = p___weight[0];",
                "    float total_weight = p___total_weight[0];",
                "    float num_macro_steps = p___num_macro_steps[0];",
                "    int32_t sub_step = p___sub_step[0];",
                "    int32_t num_sub_steps = p___num_sub_steps[0];",
                "    int32_t flags = p___flags[0];",
                "    int32_t macro_step_index = p___macro_step_index[0];",
                "    bool is_inner_first = ((flags & 1) != 0) && (sub_step == 0);",
                "    bool is_inner_last = (((flags >> 1) & 1) != 0) && (sub_step == num_sub_steps - 1);",
                "    bool is_middle = sub_step == (num_sub_steps / 2);",
                "    bool is_outer_first = (((flags >> 2) & 1) != 0) && is_inner_last;",
                "    bool is_outer_last = (((flags >> 3) & 1) != 0) && is_inner_last;",
                "",
            ]
        )

        for var in group["vars"]:
            condition = f"level < {var['n_levels']}" if var["is_2d"] else "level == 0"
            lines.append(f"    if ({condition}) {{")
            lines.append(f"        long out_off = {self._out_offset_expr(var)};")
            emitted: Dict[str, str] = {}
            val = self._value_expr(
                var["name"],
                lines,
                emitted,
                context="group",
                is_2d=var["is_2d"],
                n_levels=var["n_levels"],
                ctype=var["ctype"],
            )
            lines.append(f"        {var['ctype']} val = {val};")
            lines.extend(self._generate_inner_updates(var))
            for op in var["ops"]:
                lines.extend(self._generate_op_update(var, op))
            lines.append("    }")
            lines.append("")
        lines.append("}")
        return lines

    def _generate_full_group_kernel(self: StatisticsRuntime, group: Dict[str, Any]) -> List[str]:
        params = []
        for var in group["vars"]:
            for key in self._collect_value_param_keys(var["name"]):
                ctype, _ = self._state_ctype(key)
                params.append(f"const {ctype}* p_{_c_ident(key)}")
            for op in var["ops"]:
                out_ctype, _ = self._state_ctype(op["out_key"])
                params.append(f"{out_ctype}* p_{_c_ident(op['out_key'])}")
                if op["aux_key"]:
                    aux_ctype, _ = self._state_ctype(op["aux_key"])
                    params.append(f"{aux_ctype}* p_{_c_ident(op['aux_key'])}")
            for state in var["inner_states"].values():
                ctype, _ = self._state_ctype(state["state_key"])
                params.append(f"{ctype}* p_{_c_ident(state['state_key'])}")
                if "weight_key" in state:
                    wc, _ = self._state_ctype(state["weight_key"])
                    params.append(f"{wc}* p_{_c_ident(state['weight_key'])}")
        for key, (ctype, _) in _SCALAR_TYPES.items():
            params.append(f"const {ctype}* p_{_c_ident(key)}")
        params.append("long n_elements")
        params = list(dict.fromkeys(params))

        lines = [f"__global__ void {group['kernel_name']}("]
        for i, param in enumerate(params):
            comma = "," if i < len(params) - 1 else ""
            lines.append(f"    {param}{comma}")
        lines.extend(
            [
                ") {",
                "    long linear = blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (linear >= n_elements) return;",
                "    float weight = p___weight[0];",
                "    float total_weight = p___total_weight[0];",
                "    float num_macro_steps = p___num_macro_steps[0];",
                "    int32_t sub_step = p___sub_step[0];",
                "    int32_t num_sub_steps = p___num_sub_steps[0];",
                "    int32_t flags = p___flags[0];",
                "    int32_t macro_step_index = p___macro_step_index[0];",
                "    bool is_inner_first = ((flags & 1) != 0) && (sub_step == 0);",
                "    bool is_inner_last = (((flags >> 1) & 1) != 0) && (sub_step == num_sub_steps - 1);",
                "    bool is_middle = sub_step == (num_sub_steps / 2);",
                "    bool is_outer_first = (((flags >> 2) & 1) != 0) && is_inner_last;",
                "    bool is_outer_last = (((flags >> 3) & 1) != 0) && is_inner_last;",
                "",
            ]
        )

        for var in group["vars"]:
            lines.append(f"    if (linear < {var['numel']}) {{")
            lines.append("        long out_off = linear;")
            ctype = var["ctype"]
            name_ident = _c_ident(var["name"])
            lines.append(f"        {ctype} val = static_cast<{ctype}>(p_{name_ident}[linear]);")
            lines.extend(self._generate_inner_updates(var))
            for op in var["ops"]:
                lines.extend(self._generate_op_update(var, op))
            lines.append("    }")
            lines.append("")
        lines.append("}")
        return lines

    def _out_offset_expr(self, var: Dict[str, Any]) -> str:
        if var["is_2d"]:
            return f"(t * n_saved_points + offs) * {var['n_levels']} + level"
        return "t * n_saved_points + offs"

    def _generate_inner_updates(self, var: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        ctype = var["ctype"]
        for inner in var["inner_ops"]:
            if inner == "last":
                lines.append(f"        {ctype} val_for_{inner} = val;")
                continue
            state = var["inner_states"][inner]
            ptr = f"p_{_c_ident(state['state_key'])}"
            lines.append(f"        {ctype} val_for_{inner} = static_cast<{ctype}>(0);")
            if inner == "mean":
                wptr = f"p_{_c_ident(state['weight_key'])}"
                lines.extend(
                    [
                        "        {",
                        f"            {ctype} old_v = {ptr}[out_off];",
                        f"            {ctype} old_w = {wptr}[out_off];",
                        f"            {ctype} new_v = old_v + val * static_cast<{ctype}>(weight);",
                        f"            {ctype} new_w = old_w + static_cast<{ctype}>(weight);",
                        f"            if (is_inner_last) {{ val_for_{inner} = new_v / new_w; {ptr}[out_off] = static_cast<{ctype}>(0); {wptr}[out_off] = static_cast<{ctype}>(0); }}",
                        f"            else {{ {ptr}[out_off] = new_v; {wptr}[out_off] = new_w; }}",
                        "        }",
                    ]
                )
            elif inner == "sum":
                lines.extend(
                    [
                        "        {",
                        f"            {ctype} new_v = {ptr}[out_off] + val * static_cast<{ctype}>(weight);",
                        f"            if (is_inner_last) {{ val_for_{inner} = new_v; {ptr}[out_off] = static_cast<{ctype}>(0); }}",
                        f"            else {{ {ptr}[out_off] = new_v; }}",
                        "        }",
                    ]
                )
            elif inner in {"max", "min"}:
                fn = "hf_max" if inner == "max" else "hf_min"
                reset = f"hf_neg_inf<{ctype}>()" if inner == "max" else f"hf_pos_inf<{ctype}>()"
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
            elif inner == "mid":
                lines.extend(
                    [
                        f"        if (is_middle) {{ {ptr}[out_off] = val; }}",
                        f"        if (is_inner_last) {{ val_for_{inner} = {ptr}[out_off]; }}",
                    ]
                )
        return lines

    def _generate_op_update(self, var: Dict[str, Any], op: Dict[str, Any]) -> List[str]:
        outer = op["outer"]
        compound = op["inner"] is not None
        out = f"p_{_c_ident(op['out_key'])}"
        ctype = var["ctype"]
        value = "val" if not compound or op["inner"] == "last" else f"val_for_{op['inner']}"
        guard = "        if (is_inner_last) {\n" if compound else ""
        end_guard = "        }\n" if compound else ""
        indent = "            " if compound else "        "
        lines: List[str] = []

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
            body = [
                f"{indent}if ({'is_outer_first' if compound else 'is_inner_first'}) {{",
                f"{indent}    {out}[out_off] = macro_step_index;",
                f"{indent}    {aux}[out_off] = {value};",
                f"{indent}}} else {{",
                f"{indent}    {ctype} old_v = {aux}[out_off];",
                f"{indent}    if ({value} {cmp} old_v) {{ {aux}[out_off] = {value}; {out}[out_off] = macro_step_index; }}",
                f"{indent}}}",
            ]
        elif outer["base"] == "mean":
            if not compound:
                body = [
                    f"{indent}{ctype} old_v = is_inner_first ? static_cast<{ctype}>(0) : {out}[out_off];",
                    f"{indent}{ctype} new_v = old_v + {value} * static_cast<{ctype}>(weight);",
                    f"{indent}{out}[out_off] = is_inner_last ? new_v / static_cast<{ctype}>(total_weight) : new_v;",
                ]
            else:
                body = [
                    f"{indent}{ctype} accum = is_outer_first ? {value} : ({out}[out_off] + {value});",
                    f"{indent}{out}[out_off] = is_outer_last ? accum / static_cast<{ctype}>(num_macro_steps) : accum;",
                ]
        elif outer["base"] == "sum":
            reset = "is_outer_first" if compound else "is_inner_first"
            weighted = value if compound else f"{value} * static_cast<{ctype}>(weight)"
            body = [
                f"{indent}{ctype} old_v = {reset} ? static_cast<{ctype}>(0) : {out}[out_off];",
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
        elif outer["base"] == "mid":
            body = [f"{indent}if (is_middle) {{ {out}[out_off] = {value}; }}"]
        else:
            raise ValueError(f"unsupported op '{op['op']}'")

        body = [f"{indent}{{"] + body + [f"{indent}}}"]
        if guard:
            lines.append(guard.rstrip())
            lines.extend(body)
            lines.append(end_guard.rstrip())
        else:
            lines.extend(body)
        return lines

    def _topk_update(self, var: Dict[str, Any], op: Dict[str, Any], value: str, indent: str) -> List[str]:
        outer = op["outer"]
        k = outer["k"]
        ctype = var["ctype"]
        is_max = outer["base"] == "max"
        cmp = ">" if is_max else "<"
        init = f"hf_neg_inf<{ctype}>()" if is_max else f"hf_pos_inf<{ctype}>()"
        reset = "is_inner_first" if "_" not in op["op"] else "is_outer_first"
        if outer["is_arg"]:
            out = f"p_{_c_ident(op['out_key'])}"
            aux = f"p_{_c_ident(op['aux_key'])}"
            return [
                f"{indent}long k_base = out_off * {k};",
                f"{indent}{ctype} new_v = {value};",
                f"{indent}int32_t new_i = macro_step_index;",
                f"{indent}if ({reset}) {{",
                f"{indent}    {aux}[k_base] = new_v; {out}[k_base] = new_i;",
                f"{indent}    for (int kk = 1; kk < {k}; ++kk) {{ {aux}[k_base + kk] = {init}; {out}[k_base + kk] = 0; }}",
                f"{indent}}} else {{",
                f"{indent}    for (int kk = 0; kk < {k}; ++kk) {{",
                f"{indent}        {ctype} old_v = {aux}[k_base + kk];",
                f"{indent}        int32_t old_i = {out}[k_base + kk];",
                f"{indent}        if (new_v {cmp} old_v) {{ {aux}[k_base + kk] = new_v; {out}[k_base + kk] = new_i; new_v = old_v; new_i = old_i; }}",
                f"{indent}    }}",
                f"{indent}}}",
            ]
        out = f"p_{_c_ident(op['out_key'])}"
        return [
            f"{indent}long k_base = out_off * {k};",
            f"{indent}{ctype} new_v = {value};",
            f"{indent}if ({reset}) {{",
            f"{indent}    {out}[k_base] = new_v;",
            f"{indent}    for (int kk = 1; kk < {k}; ++kk) {{ {out}[k_base + kk] = {init}; }}",
            f"{indent}}} else {{",
            f"{indent}    for (int kk = 0; kk < {k}; ++kk) {{",
            f"{indent}        {ctype} old_v = {out}[k_base + kk];",
            f"{indent}        if (new_v {cmp} old_v) {{ {out}[k_base + kk] = new_v; new_v = old_v; }}",
            f"{indent}    }}",
            f"{indent}}}",
        ]

    def _generate_launcher(self, specs: Dict[str, Any], params: Sequence[Dict[str, str]]) -> List[str]:
        lines = [
            "static void hf_check_tensor(const at::Tensor& t, const char* name, at::ScalarType dtype) {",
            "    TORCH_CHECK(t.is_cuda(), name, \" must be a CUDA/HIP tensor\");",
            "    TORCH_CHECK(t.is_contiguous(), name, \" must be contiguous\");",
            "    TORCH_CHECK(t.scalar_type() == dtype, name, \" has unexpected dtype\");",
            "}",
            "",
            "void launch_update(py::dict states, long block_size) {",
            "    int threads = block_size > 0 ? static_cast<int>(block_size) : 128;",
            "    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();",
        ]
        for param in sorted(params, key=lambda p: p["key"]):
            key = param["key"]
            key_lit = json.dumps(key)
            tname = f"t_{param['ident']}"
            lines.extend(
                [
                    f"    at::Tensor {tname} = states[{key_lit}].cast<at::Tensor>();",
                    f"    hf_check_tensor({tname}, {key_lit}, {param['scalar_type']});",
                    f"    {param['ctype']}* {param['ptr']} = {tname}.data_ptr<{param['ctype']}>();",
                ]
            )
        for scatter in specs["scatters"]:
            zero_args = [
                f"p_{_c_ident(scatter['buf_key'])}",
            ]
            if scatter["cnt_key"]:
                zero_args.append(f"p_{_c_ident(scatter['cnt_key'])}")
            zero_args.append(str(scatter["target_size"] * scatter["num_trials"]))
            lines.extend(
                [
                    "    {",
                    f"        long total = {scatter['target_size'] * scatter['num_trials']};",
                    "        int blocks = static_cast<int>((total + threads - 1) / threads);",
                    f"        if (blocks > 0) hf_scatter_zero_{scatter['safe']}<<<blocks, threads, 0, stream>>>({', '.join(zero_args)});",
                    "    }",
                ]
            )

            add_args = [
                f"p_{_c_ident(scatter['buf_key'])}",
                f"p_{_c_ident(scatter['scatter'].index)}",
            ]
            if scatter["cnt_key"]:
                add_args.append(f"p_{_c_ident(scatter['cnt_key'])}")
            for key in self._statistics_ir.scatter_inputs(scatter["name"]):
                if key != scatter["scatter"].index:
                    add_args.append(f"p_{_c_ident(key)}")
            add_args = list(dict.fromkeys(add_args))
            add_args.extend([str(scatter["source_size"]), str(scatter["target_size"]), str(scatter["num_trials"])])
            lines.extend(
                [
                    "    {",
                    f"        long total = {scatter['source_size'] * scatter['num_trials']};",
                    "        int blocks = static_cast<int>((total + threads - 1) / threads);",
                    f"        if (blocks > 0) hf_scatter_add_{scatter['safe']}<<<blocks, threads, 0, stream>>>({', '.join(add_args)});",
                    "    }",
                ]
            )
            if scatter["cnt_key"]:
                divide_args = [
                    f"p_{_c_ident(scatter['buf_key'])}",
                    f"p_{_c_ident(scatter['cnt_key'])}",
                    str(scatter["target_size"] * scatter["num_trials"]),
                ]
                lines.extend(
                    [
                        "    {",
                        f"        long total = {scatter['target_size'] * scatter['num_trials']};",
                        "        int blocks = static_cast<int>((total + threads - 1) / threads);",
                        f"        if (blocks > 0) hf_scatter_divide_{scatter['safe']}<<<blocks, threads, 0, stream>>>({', '.join(divide_args)});",
                        "    }",
                    ]
                )
        for group in specs["groups"]:
            args = [] if group.get("full_output") else [f"p_{_c_ident(group['output_index'])}"]
            for var in group["vars"]:
                for key in self._collect_value_param_keys(var["name"]):
                    args.append(f"p_{_c_ident(key)}")
                for op in var["ops"]:
                    args.append(f"p_{_c_ident(op['out_key'])}")
                    if op["aux_key"]:
                        args.append(f"p_{_c_ident(op['aux_key'])}")
                for state in var["inner_states"].values():
                    args.append(f"p_{_c_ident(state['state_key'])}")
                    if "weight_key" in state:
                        args.append(f"p_{_c_ident(state['weight_key'])}")
            for key in _SCALAR_TYPES:
                args.append(f"p_{_c_ident(key)}")
            args = list(dict.fromkeys(args))
            if group.get("full_output"):
                args.append(str(group["full_total"]))
                lines.extend(
                    [
                        "    {",
                        f"        long total = {group['full_total']};",
                        "        int blocks = static_cast<int>((total + threads - 1) / threads);",
                        f"        if (blocks > 0) {group['kernel_name']}<<<blocks, threads, 0, stream>>>({', '.join(args)});",
                        "    }",
                    ]
                )
            else:
                args.extend([f"t_{_c_ident(group['output_index'])}.numel()", str(group["num_trials"])])
                lines.extend(
                    [
                        "    {",
                        f"        long total = t_{_c_ident(group['output_index'])}.numel() * {group['num_trials']} * {group['max_levels']};",
                        "        int blocks = static_cast<int>((total + threads - 1) / threads);",
                        f"        if (blocks > 0) {group['kernel_name']}<<<blocks, threads, 0, stream>>>({', '.join(args)});",
                        "    }",
                    ]
                )
        lines.append("}")
        return lines

    def _save_cuda_kernel_file(self: StatisticsRuntime, cpp_sources: str, cuda_sources: str) -> None:
        unique_name = self._generate_unique_name()
        self._saved_kernel_file = self.kernels_dir / f"kern_cuda_{unique_name}.cu"
        atomic_write_text(
            self._saved_kernel_file,
            "// C++ declarations passed to torch.utils.cpp_extension.load_inline\n"
            + cpp_sources
            + "\n// CUDA source\n"
            + cuda_sources,
        )

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from math import prod

from hydroforge.statistics.emitters.common import StatisticsEmitter
from hydroforge.statistics.ir import (
    ExpressionDialect,
    ExpressionSource,
    ScatterSource,
    StatisticsIR,
    TensorSource,
    render_expression,
)

_FULL_OUTPUT_GROUP = "__full__"
# Full-output entry points use a namespace disjoint from indexed
# ``aggr_kernel_*`` and scatter helpers.
_FULL_OUTPUT_KERNEL = "hydroforge_full_output_metal_kernel"


_CONTROL_SCALARS = (
    ("__hf_weight", "float", "__weight"),
    ("__hf_total_weight", "float", "__total_weight"),
    ("__hf_num_macro_steps", "long", "__num_macro_steps"),
    ("__hf_sub_step", "int", "__sub_step"),
    ("__hf_num_sub_steps", "int", "__num_sub_steps"),
    ("__hf_flags", "int", "__flags"),
    ("__hf_macro_step_index", "long", "__macro_step_index"),
)


def _emit_argument_kernel_start(
    lines: list[str],
    kernel_name: str,
    fields: list[tuple[str, str, bool]],
) -> None:
    """Emit an argument-buffer struct and kernel entry point directly."""
    lines.append(f"struct {kernel_name}_args {{")
    for index, (type_decl, name, scalar) in enumerate(fields):
        field_type = f"constant {type_decl}*" if scalar else type_decl
        lines.append(f"    {field_type} arg_{index} [[id({index})]];")
    lines.append(f"    constant long* _grid_size [[id({len(fields)})]];")
    lines.extend(
        [
            "};",
            "",
            f"kernel void {kernel_name}(",
            f"    constant {kernel_name}_args& args [[buffer(0)]],",
            "    uint tid [[thread_position_in_grid]]",
            ") {",
        ]
    )
    for index, (type_decl, name, scalar) in enumerate(fields):
        if scalar:
            lines.append(f"    const {type_decl} {name} = *args.arg_{index};")
        else:
            lines.append(f"    {type_decl} {name} = args.arg_{index};")


class MetalStatisticsEmitter(StatisticsEmitter):
    """Metal MSL kernel code generation for statistics aggregation."""

    def emit(self):
        self._generate_metal_aggregator_function()
        return self.result()

    # ========================================================================
    # Metal MSL code generation
    # ========================================================================

    def _metal_dtype_str(self, var_name: str) -> str:
        """Return the Metal Shading Language scalar type string for a variable."""
        import torch

        from hydroforge.kernels.backends.metal.types import tensor_type

        tensor = self._tensor_registry.get(var_name)
        if tensor is not None:
            dt = tensor.dtype
        elif var_name in self._statistics_layouts:
            dt = self._statistics_layouts[var_name].dtype
        else:
            stored = self._storage.get(var_name)
            dt = stored.dtype if stored is not None else torch.float32
        return tensor_type(dt)

    def _metal_emit_val_load(
        self,
        var_name: str,
        lines: list,
        emitted: set,
        indent: str,
        idx_expr: str = "idx",
        *,
        n_levels: int | None = None,
        full_variable: str | None = None,
    ) -> str:
        """Emit MSL code to load a variable value (handling virtuals).

        Returns the MSL variable name for the loaded value.
        """
        safe_var = self._get_safe_name(var_name)
        val_name = f"{safe_var}_val"
        if safe_var in emitted:
            return val_name

        ctype = self._metal_dtype_str(var_name)
        source = self._statistics_ir.sources.get(var_name) or TensorSource(var_name)

        if isinstance(source, (TensorSource, ScatterSource)):
            key = (
                source.name
                if isinstance(source, TensorSource)
                else f"__scatter_buf_{var_name}"
            )
            if full_variable is not None:
                offset = self._full_source_offset(key, full_variable, idx_expr)
            else:
                stride = self._source_stride(
                    key, logical_rank=1 if n_levels is None else 2
                )
                offset = (
                    f"t * {stride} + {idx_expr}" if self.ensemble_size > 1 else idx_expr
                )
                if n_levels is not None:
                    offset = f"({offset}) * {n_levels} + level"
            buffer_name = self._get_safe_name(key)
            lines.append(f"{indent}{ctype} {val_name} = p_{buffer_name}[{offset}];")
        elif isinstance(source, ExpressionSource):
            names = {
                dependency: self._metal_emit_val_load(
                    dependency,
                    lines,
                    emitted,
                    indent,
                    idx_expr,
                    n_levels=n_levels,
                    full_variable=full_variable,
                )
                for dependency in source.expression.dependencies
            }
            expression = render_expression(
                source.expression,
                ExpressionDialect.METAL,
                names,
                value_type="float32",
            )
            lines.append(f"{indent}{ctype} {val_name} = ({ctype})({expression});")

        emitted.add(safe_var)
        return val_name

    def _emit_control_values(self, lines: list[str], indent: str) -> None:
        for sname, stype, key in _CONTROL_SCALARS:
            lines.append(f"{indent}{stype} {key.removeprefix('__')} = *p_{sname}_ptr;")
        expressions = {
            "is_inner_first": "((flags & 1) != 0) && (sub_step == 0)",
            "is_inner_last": "(((flags >> 1) & 1) != 0) && (sub_step == num_sub_steps - 1)",
            "is_outer_first": "(((flags >> 2) & 1) != 0) && is_inner_last",
            "is_outer_last": "(((flags >> 3) & 1) != 0) && is_inner_last",
        }
        for name, expression in expressions.items():
            if name in self._statistics_lowering.required_flags:
                lines.append(f"{indent}bool {name} = {expression};")
        lines.append("")

    def _output_tensors(self, variables: list[str]) -> list[tuple[str, str, str]]:
        outputs: dict[str, tuple[str, str, str]] = {}
        for var in variables:
            safe_var = self._get_safe_name(var)
            ctype = self._metal_dtype_str(var)

            def add(suffix: str, dtype: str = ctype) -> None:
                key = f"{var}_{suffix}"
                outputs[key] = (key, dtype, f"p_{safe_var}_{suffix}")

            for operation in self._statistics_lowering.operations(var):
                add(operation.spelling, "long" if operation.stores_index else ctype)
                if operation.stores_index:
                    add(f"{operation.spelling}_aux")
                if operation.inner is None:
                    if operation.outer.value == "mean":
                        add("mean_sample_weight_state")
                elif operation.inner.value != "last":
                    inner = operation.inner.value
                    add(f"{inner}_inner_state")
                    if inner == "mean":
                        add(f"{inner}_weight_state")
        return list(outputs.values())

    def _generate_metal_full_kernel_for_group(
        self, msl_lines: list, output_index: str, var_list: list
    ) -> dict:
        """Generate a Metal kernel for variables saved at full tensor shape."""
        kernel_name = _FULL_OUTPUT_KERNEL

        sorted_inputs = sorted(
            {
                name
                for var in var_list
                for name in self._statistics_ir.materialized_inputs(var)
            }
        )

        out_tensors = self._output_tensors(var_list)

        full_total = max(
            prod(self._statistics_layouts[var].actual_shape) for var in var_list
        )

        arg_order = []
        abi_fields = []
        indent = "    "

        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            ctype = self._metal_dtype_str(inp)
            abi_fields.append((f"device const {ctype}*", f"p_{safe_inp}", False))
            arg_order.append(("tensor", inp, "read"))

        for state_key, ctype, pname in out_tensors:
            abi_fields.append((f"device {ctype}*", pname, False))
            arg_order.append(("tensor", state_key, "read_write"))

        for sname, stype, state_key in _CONTROL_SCALARS:
            abi_fields.append((f"device const {stype}*", f"p_{sname}_ptr", False))
            arg_order.append(("tensor", state_key, "read"))

        abi_fields.append(("long", "n_elements", True))
        arg_order.append(("scalar", "n_elements", "long"))
        _emit_argument_kernel_start(msl_lines, kernel_name, abi_fields)
        msl_lines.append(f"{indent}if ((long)tid >= n_elements) return;")
        msl_lines.append(f"{indent}long out_idx = (long)tid;")
        msl_lines.append("")
        self._emit_control_values(msl_lines, indent)

        self._emit_full_output_updates(
            indent=indent, msl_lines=msl_lines, var_list=var_list
        )

        msl_lines.append("}")
        msl_lines.append("")

        return {
            "kernel_name": kernel_name,
            "output_index": output_index,
            "full_output": True,
            "n_elements_val": full_total,
            "arg_order": arg_order,
        }

    def _generate_metal_kernel_for_group(
        self, msl_lines: list, output_index: str, var_list: list
    ) -> dict:
        """Generate a Metal kernel function for one output_index group.

        Returns metadata dict used by the Python wrapper.
        """
        if output_index == _FULL_OUTPUT_GROUP:
            return self._generate_metal_full_kernel_for_group(
                msl_lines, output_index, var_list
            )

        ensemble_size = self.ensemble_size
        safe_save = self._get_safe_name(output_index)
        kernel_name = f"aggr_kernel_{safe_save}"

        dims_1d, dims_2d = self._statistics_lowering.split_indexed(var_list)

        sorted_inputs = sorted(
            {
                name
                for var in var_list
                for name in self._statistics_ir.materialized_inputs(var)
            }
        )

        out_tensors = self._output_tensors(var_list)

        # ---- Build MSL kernel signature with [[buffer(N)]] bindings ----
        arg_order = []  # track Python-side arg order: ('tensor', key) or ('scalar', name, msl_type)
        abi_fields = []

        indent = "    "

        # output_index buffer
        index_ctype = self._metal_dtype_str(output_index)
        abi_fields.append((f"device const {index_ctype}*", f"p_{safe_save}", False))
        arg_order.append(("tensor", output_index, "read"))

        # input var buffers
        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            if safe_inp == safe_save:
                continue
            ctype = self._metal_dtype_str(inp)
            abi_fields.append((f"device const {ctype}*", f"p_{safe_inp}", False))
            arg_order.append(("tensor", inp, "read"))

        # output/state buffers (read-write)
        for state_key, ctype, pname in out_tensors:
            abi_fields.append((f"device {ctype}*", pname, False))
            arg_order.append(("tensor", state_key, "read_write"))

        # varying scalar params → device buffer pointers (avoids host-device sync)
        for sname, stype, state_key in _CONTROL_SCALARS:
            abi_fields.append((f"device const {stype}*", f"p_{sname}_ptr", False))
            arg_order.append(("tensor", state_key, "read"))

        # fixed scalar params (truly constant per-capture)
        abi_fields.append(("long", "n_saved_points", True))
        arg_order.append(("scalar", "n_saved_points", "long"))

        _emit_argument_kernel_start(msl_lines, kernel_name, abi_fields)

        # Bounds check
        msl_lines.append(f"{indent}if ((long)tid >= n_saved_points) return;")
        msl_lines.append(f"{indent}long idx = (long)p_{safe_save}[tid];")
        msl_lines.append("")
        # Dereference varying scalar device pointers
        self._emit_control_values(msl_lines, indent)

        # Member loop
        if ensemble_size > 1:
            msl_lines.append(f"{indent}for (long t = 0; t < {ensemble_size}; t++) {{")
            indent2 = indent + "    "
        else:
            msl_lines.append(f"{indent}const long t = 0;")
            indent2 = indent

        for batched in (True, False):
            vectors = [
                name
                for name in dims_1d
                if self._statistics_layouts[name].batched == batched
            ]
            levels = [
                name
                for name in dims_2d
                if self._statistics_layouts[name].batched == batched
            ]
            if not vectors and not levels:
                continue
            body_indent = indent2
            if not batched:
                msl_lines.append(f"{indent2}if (t == 0) {{")
                body_indent += "    "
            self._emit_scalar_updates(
                variables=vectors,
                indent=body_indent,
                msl_lines=msl_lines,
            )
            self._emit_indexed_vector_updates(
                dims_2d=levels,
                indent2=body_indent,
                msl_lines=msl_lines,
            )
            if not batched:
                msl_lines.append(f"{indent2}}}")

        if ensemble_size > 1:
            msl_lines.append(f"{indent}}}")
        msl_lines.append("}")
        msl_lines.append("")

        return {
            "kernel_name": kernel_name,
            "output_index": output_index,
            "arg_order": arg_order,
        }

    def _generate_metal_scatter_kernels(
        self,
        msl_lines: list[str],
        ir: StatisticsIR,
    ) -> list[dict]:
        """Emit scatter materialization from typed IR before group kernels."""
        metas: list[dict] = []
        for variable in ir.ordered_scatters():
            source = variable.source
            name = variable.name
            safe = self._get_safe_name(name)
            buf_key = f"__scatter_buf_{name}"
            cnt_key = (
                f"__scatter_cnt_{name}" if source.reduction.value == "mean" else None
            )
            target_size = self._storage[buf_key].shape[-1]
            source_size = self._tensor_registry[source.index].numel()
            ensemble_size = (
                self.ensemble_size if self._statistics_layouts[name].batched else 1
            )
            total_target = target_size * ensemble_size
            total_source = source_size * ensemble_size

            zero_name = f"aggr_scatter_zero_{safe}"
            zero_fields = [("device float*", "p_buf", False)]
            zero_order = [("tensor", buf_key, "write")]
            if cnt_key is not None:
                zero_fields.append(("device int*", "p_cnt", False))
                zero_order.append(("tensor", cnt_key, "write"))
            zero_fields.append(("long", "total", True))
            zero_order.append(("scalar", "total", total_target))
            _emit_argument_kernel_start(msl_lines, zero_name, zero_fields)
            msl_lines.extend(
                [
                    "    if ((long)tid >= total) return;",
                    "    p_buf[tid] = 0.0f;",
                ]
            )
            if cnt_key is not None:
                msl_lines.append("    p_cnt[tid] = 0;")
            msl_lines.extend(["}", ""])
            metas.append(
                {
                    "kernel_name": zero_name,
                    "arg_order": zero_order,
                    "grid_size": total_target,
                }
            )

            add_name = f"aggr_scatter_add_{safe}"
            add_fields = [("device atomic_float*", "p_buf", False)]
            add_order = [("tensor", buf_key, "atomic_add")]
            if cnt_key is not None:
                add_fields.append(("device atomic_int*", "p_cnt", False))
                add_order.append(("tensor", cnt_key, "atomic_add"))
            index_ctype = self._metal_dtype_str(source.index)
            index_safe = self._get_safe_name(source.index)
            add_fields.append(
                (f"device const {index_ctype}*", f"p_{index_safe}", False)
            )
            add_order.append(("tensor", source.index, "read"))

            leaf_inputs = tuple(
                key for key in ir.scatter_inputs(name) if key != source.index
            )
            # Expressions are allowed to reference the (shared) scatter index
            # itself.  It is already bound above and always has zero member
            # stride, but must still participate in value-load addressing.
            strides: dict[str, int] = {source.index: 0}
            for key in leaf_inputs:
                ctype = self._metal_dtype_str(key)
                key_safe = self._get_safe_name(key)
                add_fields.append((f"device const {ctype}*", f"p_{key_safe}", False))
                add_order.append(("tensor", key, "read"))
                strides[key] = self._source_stride(key)
            for scalar, value in (
                ("source_size", source_size),
                ("target_size", target_size),
                ("total", total_source),
            ):
                add_fields.append(("long", scalar, True))
                add_order.append(("scalar", scalar, value))
            _emit_argument_kernel_start(msl_lines, add_name, add_fields)
            msl_lines.extend(
                [
                    "    if ((long)tid >= total) return;",
                    "    long t = (long)tid / source_size;",
                    "    long src = (long)tid - t * source_size;",
                    f"    long dst = (long)p_{index_safe}[src];",
                    "    if (dst < 0 || dst >= target_size) return;",
                ]
            )

            emitted: dict[str, str] = {}

            def emit_value(field: str) -> str:
                previous = emitted.get(field)
                if previous is not None:
                    return previous
                field_source = ir.sources.get(field) or TensorSource(field)
                value_name = f"v_{self._get_safe_name(field)}"
                if isinstance(field_source, ExpressionSource):
                    names = {
                        dependency: emit_value(dependency)
                        for dependency in field_source.expression.dependencies
                    }
                    expression = render_expression(
                        field_source.expression,
                        ExpressionDialect.METAL,
                        names,
                        value_type="float32",
                    )
                    msl_lines.append(f"    float {value_name} = (float)({expression});")
                else:
                    key = (
                        f"__scatter_buf_{field}"
                        if isinstance(field_source, ScatterSource)
                        else field_source.name
                    )
                    key_safe = self._get_safe_name(key)
                    stride = strides[key]
                    msl_lines.append(
                        f"    float {value_name} = (float)p_{key_safe}[t * {stride} + src];"
                    )
                emitted[field] = value_name
                return value_name

            names = {
                dependency: emit_value(dependency)
                for dependency in source.value.dependencies
            }
            expression = render_expression(
                source.value,
                ExpressionDialect.METAL,
                names,
                value_type="float32",
            )
            msl_lines.append(f"    float value = (float)({expression});")
            msl_lines.append(
                "    atomic_fetch_add_explicit(p_buf + t * target_size + dst, "
                "value, memory_order_relaxed);"
            )
            if cnt_key is not None:
                msl_lines.append(
                    "    atomic_fetch_add_explicit(p_cnt + t * target_size + dst, "
                    "1, memory_order_relaxed);"
                )
            msl_lines.extend(["}", ""])
            metas.append(
                {
                    "kernel_name": add_name,
                    "arg_order": add_order,
                    "grid_size": total_source,
                }
            )

            if cnt_key is not None:
                divide_name = f"aggr_scatter_divide_{safe}"
                divide_fields = [
                    ("device float*", "p_buf", False),
                    ("device const int*", "p_cnt", False),
                    ("long", "total", True),
                ]
                divide_order = [
                    ("tensor", buf_key, "read_write"),
                    ("tensor", cnt_key, "read"),
                    ("scalar", "total", total_target),
                ]
                _emit_argument_kernel_start(msl_lines, divide_name, divide_fields)
                msl_lines.extend(
                    [
                        "    if ((long)tid >= total) return;",
                        "    float count = float(p_cnt[tid]);",
                        "    p_buf[tid] = count > 0.0f ? p_buf[tid] / count : hydroforge_nan();",
                        "}",
                        "",
                    ]
                )
                metas.append(
                    {
                        "kernel_name": divide_name,
                        "arg_order": divide_order,
                        "grid_size": total_target,
                    }
                )
        return metas

    def _generate_metal_aggregator_function(
        self,
    ) -> None:
        """Generate and compile a Metal MSL aggregation kernel.

        Generates raw MSL kernels and compiles them through HydroForge's
        native Metal pipeline bridge.
        """

        grouped_by_output_index = self._statistics_lowering.groups

        msl_lines = [
            "// Auto-generated Metal aggregation kernels for hydroforge statistics",
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            "// Extrema treat one NaN as missing and preserve NaN when both are NaN.",
            "// Inspecting the IEEE bits keeps this contract explicit under fast math.",
            "inline bool hydroforge_isnan(float value) {",
            "    return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;",
            "}",
            "inline float hydroforge_nan() {",
            "    return as_type<float>(0x7fc00000u);",
            "}",
            "inline float hydroforge_maximum(float left, float right) {",
            "    if (hydroforge_isnan(left)) return right;",
            "    if (hydroforge_isnan(right)) return left;",
            "    return max(left, right);",
            "}",
            "inline float hydroforge_minimum(float left, float right) {",
            "    if (hydroforge_isnan(left)) return right;",
            "    if (hydroforge_isnan(right)) return left;",
            "    return min(left, right);",
            "}",
            "inline float hydroforge_weighted_mean(float old_value, float old_weight, float value, float weight) {",
            "    float new_weight = old_weight + weight;",
            "    return old_value * (old_weight / new_weight) + value * (weight / new_weight);",
            "}",
            "inline int hydroforge_maximum(int left, int right) { return max(left, right); }",
            "inline int hydroforge_minimum(int left, int right) { return min(left, right); }",
            "inline long hydroforge_maximum(long left, long right) { return max(left, right); }",
            "inline long hydroforge_minimum(long left, long right) { return min(left, right); }",
            "inline uchar hydroforge_maximum(uchar left, uchar right) { return max(left, right); }",
            "inline uchar hydroforge_minimum(uchar left, uchar right) { return min(left, right); }",
            "",
        ]

        scatter_metas = self._generate_metal_scatter_kernels(
            msl_lines,
            self._statistics_ir,
        )
        group_metas = []
        for output_index, var_list in grouped_by_output_index.items():
            meta = self._generate_metal_kernel_for_group(
                msl_lines, output_index, var_list
            )
            group_metas.append(meta)

        msl_src = "\n".join(msl_lines)

        from hydroforge.contracts import KernelSpec
        from hydroforge.kernels.registry import make_metal_dispatcher

        dispatchers = {}
        for meta in (*scatter_metas, *group_metas):
            arg_names = tuple(f"arg_{index}" for index in range(len(meta["arg_order"])))
            buffer_access = {
                arg_names[index]: rest[1]
                for index, (kind, *rest) in enumerate(meta["arg_order"])
                if kind == "tensor"
            }
            runtime_scalars = {
                arg_names[index]: (
                    "float32" if rest[-1] in {"float", "double"} else "index"
                )
                for index, (kind, *rest) in enumerate(meta["arg_order"])
                if kind == "scalar"
            }
            runtime_scalars["_grid_size"] = "index"
            parameters = (*arg_names, "_grid_size")
            spec = KernelSpec(
                name=meta["kernel_name"],
                parameters=parameters,
                size_key="_grid_size",
                buffers=buffer_access,
                runtime_scalars=runtime_scalars,
            )
            dispatchers[meta["kernel_name"]] = make_metal_dispatcher(
                msl_src,
                meta["kernel_name"],
                spec=spec,
            )

        # Build the Python wrapper
        def _make_wrapper(compiled, scatters, metas):
            from hydroforge.kernels.backends.metal.online import (
                launch_metal_dispatcher,
            )

            def internal_update_statistics(states, BLOCK_SIZE):
                for meta in scatters:
                    dispatcher = compiled[meta["kernel_name"]]
                    args = [
                        states[rest[0]] if kind == "tensor" else rest[1]
                        for kind, *rest in meta["arg_order"]
                    ]
                    launch_metal_dispatcher(
                        dispatcher,
                        {
                            **{f"arg_{i}": value for i, value in enumerate(args)},
                            "BLOCK_SIZE": BLOCK_SIZE,
                            "_grid_size": meta["grid_size"],
                        },
                    )
                for meta in metas:
                    dispatcher = compiled[meta["kernel_name"]]
                    si = meta["output_index"]
                    if meta.get("full_output"):
                        n_threads = meta["n_elements_val"]
                        n_saved = n_threads
                    else:
                        n_saved = len(states[si])
                        n_threads = n_saved

                    args = []

                    for kind, *rest in meta["arg_order"]:
                        if kind == "tensor":
                            key = rest[0]
                            args.append(states[key])
                        else:  # scalar (fixed constants only)
                            sname = rest[0]
                            if sname == "n_saved_points":
                                args.append(n_saved)
                            elif sname == "n_elements":
                                args.append(meta["n_elements_val"])

                    launch_metal_dispatcher(
                        dispatcher,
                        {
                            **{f"arg_{i}": value for i, value in enumerate(args)},
                            "BLOCK_SIZE": BLOCK_SIZE,
                            "_grid_size": n_threads,
                        },
                    )

            return internal_update_statistics

        self._aggregator_function = _make_wrapper(
            dispatchers,
            scatter_metas,
            group_metas,
        )

        # Save for debugging
        if self.save_kernels:
            self._save_kernel_file(msl_src)

    def _emit_scalar_updates(
        self,
        *,
        variables: list[str],
        indent: str,
        msl_lines: list[str],
        out_idx: str = "t * n_saved_points + tid",
        full_output: bool = False,
    ) -> None:
        if variables:
            msl_lines.append(f"{indent}// === 1D variables ===")
            emitted = set()
            for var in variables:
                self._metal_emit_val_load(
                    var,
                    msl_lines,
                    emitted,
                    indent,
                    idx_expr="out_idx" if full_output else "idx",
                    full_variable=var if full_output else None,
                )

            # Inner aggregation states for compound ops
            for reduction, inner_vars in self._statistics_lowering.variables_by_inner(
                variables
            ).items():
                inner_type = reduction.value
                for var in inner_vars:
                    safe_var = self._get_safe_name(var)
                    var_val = f"{safe_var}_val"
                    ctype = self._metal_dtype_str(var)
                    val_for = f"val_for_{safe_var}_{inner_type}"

                    if inner_type == "last":
                        pass
                    elif inner_type == "mean":
                        msl_lines.extend(
                            [
                                f"{indent}{ctype} {val_for} = ({ctype})0;",
                                f"{indent}{{",
                                f"{indent}    {ctype} inner_old = p_{safe_var}_mean_inner_state[{out_idx}];",
                                f"{indent}    {ctype} w_old = p_{safe_var}_mean_weight_state[{out_idx}];",
                                f"{indent}    {ctype} w_new = w_old + ({ctype})weight;",
                                f"{indent}    {ctype} inner_new = hydroforge_weighted_mean(inner_old, w_old, {var_val}, ({ctype})weight);",
                                f"{indent}    if (is_inner_last) {{",
                                f"{indent}        p_{safe_var}_mean_inner_state[{out_idx}] = ({ctype})0;",
                                f"{indent}        p_{safe_var}_mean_weight_state[{out_idx}] = ({ctype})0;",
                                f"{indent}        {val_for} = inner_new;",
                                f"{indent}    }} else {{",
                                f"{indent}        p_{safe_var}_mean_inner_state[{out_idx}] = inner_new;",
                                f"{indent}        p_{safe_var}_mean_weight_state[{out_idx}] = w_new;",
                                f"{indent}    }}",
                                f"{indent}}}",
                            ]
                        )
                    elif inner_type == "sum":
                        msl_lines.extend(
                            [
                                f"{indent}{ctype} {val_for} = ({ctype})0;",
                                f"{indent}{{",
                                f"{indent}    {ctype} inner_old = p_{safe_var}_sum_inner_state[{out_idx}];",
                                f"{indent}    {ctype} inner_new = inner_old + {var_val} * ({ctype})weight;",
                                f"{indent}    if (is_inner_last) {{",
                                f"{indent}        p_{safe_var}_sum_inner_state[{out_idx}] = ({ctype})0;",
                                f"{indent}        {val_for} = inner_new;",
                                f"{indent}    }} else {{",
                                f"{indent}        p_{safe_var}_sum_inner_state[{out_idx}] = inner_new;",
                                f"{indent}    }}",
                                f"{indent}}}",
                            ]
                        )
                    elif inner_type in {"max", "min"}:
                        fn = (
                            "hydroforge_maximum"
                            if inner_type == "max"
                            else "hydroforge_minimum"
                        )
                        reset = "-INFINITY" if inner_type == "max" else "INFINITY"
                        msl_lines.extend(
                            [
                                f"{indent}{ctype} {val_for} = ({ctype})0;",
                                f"{indent}{{",
                                f"{indent}    {ctype} inner_old = p_{safe_var}_{inner_type}_inner_state[{out_idx}];",
                                f"{indent}    {ctype} inner_new = is_inner_first ? {var_val} : {fn}(inner_old, {var_val});",
                                f"{indent}    if (is_inner_last) {{",
                                f"{indent}        p_{safe_var}_{inner_type}_inner_state[{out_idx}] = ({ctype})({reset});",
                                f"{indent}        {val_for} = inner_new;",
                                f"{indent}    }} else {{",
                                f"{indent}        p_{safe_var}_{inner_type}_inner_state[{out_idx}] = inner_new;",
                                f"{indent}    }}",
                                f"{indent}}}",
                            ]
                        )
                    elif inner_type == "first":
                        msl_lines.extend(
                            [
                                f"{indent}{ctype} {val_for} = ({ctype})0;",
                                f"{indent}if (is_inner_first) p_{safe_var}_first_inner_state[{out_idx}] = {var_val};",
                                f"{indent}if (is_inner_last) {val_for} = p_{safe_var}_first_inner_state[{out_idx}];",
                            ]
                        )
            # Emit actual ops
            for var in variables:
                safe_var = self._get_safe_name(var)
                var_val = f"{safe_var}_val"
                ctype = self._metal_dtype_str(var)
                operations = self._statistics_lowering.operations(var)

                for operation in operations:
                    op = operation.spelling

                    # ---- Compound ops ----
                    if operation.compound:
                        outer = operation.outer.value
                        inner = operation.inner.value
                        is_arg = operation.stores_index

                        if inner == "last":
                            val_var = var_val
                        else:
                            val_var = f"val_for_{safe_var}_{inner}"

                        msl_lines.append(f"{indent}// Compound {op} for {safe_var}")
                        if is_arg:
                            cmp_op = ">" if outer == "max" else "<"
                            aux_ptr = f"p_{safe_var}_{op}_aux"
                            out_ptr = f"p_{safe_var}_{op}"
                            if operation.k == 1:
                                if self._statistics_layouts[
                                    var
                                ].dtype.is_floating_point:
                                    msl_lines.extend(
                                        [
                                            f"{indent}if (is_inner_last) {{",
                                            f"{indent}    {ctype} candidate = {val_var};",
                                            f"{indent}    if (is_outer_first) {{",
                                            f"{indent}        {out_ptr}[{out_idx}] = -1;",
                                            f"{indent}        {aux_ptr}[{out_idx}] = hydroforge_nan();",
                                            f"{indent}    }}",
                                            f"{indent}    {ctype} old_aux = {aux_ptr}[{out_idx}];",
                                            f"{indent}    if (!hydroforge_isnan(candidate) && (hydroforge_isnan(old_aux) || candidate {cmp_op} old_aux)) {{",
                                            f"{indent}        {aux_ptr}[{out_idx}] = candidate;",
                                            f"{indent}        {out_ptr}[{out_idx}] = macro_step_index;",
                                            f"{indent}    }}",
                                            f"{indent}}}",
                                        ]
                                    )
                                else:
                                    msl_lines.extend(
                                        [
                                            f"{indent}if (is_inner_last) {{",
                                            f"{indent}    if (is_outer_first) {{",
                                            f"{indent}        {out_ptr}[{out_idx}] = macro_step_index;",
                                            f"{indent}        {aux_ptr}[{out_idx}] = {val_var};",
                                            f"{indent}    }} else {{",
                                            f"{indent}        {ctype} old_aux = {aux_ptr}[{out_idx}];",
                                            f"{indent}        if ({val_var} {cmp_op} old_aux) {{",
                                            f"{indent}            {aux_ptr}[{out_idx}] = {val_var};",
                                            f"{indent}            {out_ptr}[{out_idx}] = macro_step_index;",
                                            f"{indent}        }}",
                                            f"{indent}    }}",
                                            f"{indent}}}",
                                        ]
                                    )
                            else:
                                msl_lines.extend(
                                    [
                                        f"{indent}if (is_inner_last) {{",
                                        f"{indent}    long k_base = ({out_idx}) * {operation.k};",
                                        f"{indent}    {ctype} new_value = {val_var};",
                                        f"{indent}    long new_index = macro_step_index;",
                                        f"{indent}    if (is_outer_first) {{",
                                        f"{indent}        for (int rank = 0; rank < {operation.k}; ++rank) {{",
                                        f"{indent}            {aux_ptr}[k_base + rank] = hydroforge_nan();",
                                        f"{indent}            {out_ptr}[k_base + rank] = -1;",
                                        f"{indent}        }}",
                                        f"{indent}    }}",
                                        f"{indent}    for (int rank = 0; rank < {operation.k}; ++rank) {{",
                                        f"{indent}        {ctype} old_value = {aux_ptr}[k_base + rank];",
                                        f"{indent}        long old_index = {out_ptr}[k_base + rank];",
                                        f"{indent}        if (!hydroforge_isnan(new_value) && (hydroforge_isnan(old_value) || new_value {cmp_op} old_value || (new_value == old_value && new_index < old_index))) {{",
                                        f"{indent}            {aux_ptr}[k_base + rank] = new_value;",
                                        f"{indent}            {out_ptr}[k_base + rank] = new_index;",
                                        f"{indent}            new_value = old_value;",
                                        f"{indent}            new_index = old_index;",
                                        f"{indent}        }}",
                                        f"{indent}    }}",
                                        f"{indent}}}",
                                    ]
                                )
                        elif outer in ("max", "min"):
                            cmp_fn = (
                                "hydroforge_maximum"
                                if outer == "max"
                                else "hydroforge_minimum"
                            )
                            out_ptr = f"p_{safe_var}_{op}"
                            if operation.k == 1:
                                msl_lines.extend(
                                    [
                                        f"{indent}if (is_inner_last) {{",
                                        f"{indent}    if (is_outer_first) {{",
                                        f"{indent}        {out_ptr}[{out_idx}] = {val_var};",
                                        f"{indent}    }} else {{",
                                        f"{indent}        {out_ptr}[{out_idx}] = {cmp_fn}({out_ptr}[{out_idx}], {val_var});",
                                        f"{indent}    }}",
                                        f"{indent}}}",
                                    ]
                                )
                            else:
                                cmp_op = ">" if outer == "max" else "<"
                                msl_lines.extend(
                                    [
                                        f"{indent}if (is_inner_last) {{",
                                        f"{indent}    long k_base = ({out_idx}) * {operation.k};",
                                        f"{indent}    {ctype} new_value = {val_var};",
                                        f"{indent}    if (is_outer_first) {{",
                                        f"{indent}        for (int rank = 0; rank < {operation.k}; ++rank) {{",
                                        f"{indent}            {out_ptr}[k_base + rank] = hydroforge_nan();",
                                        f"{indent}        }}",
                                        f"{indent}    }}",
                                        f"{indent}    for (int rank = 0; rank < {operation.k}; ++rank) {{",
                                        f"{indent}        {ctype} old_value = {out_ptr}[k_base + rank];",
                                        f"{indent}        if (!hydroforge_isnan(new_value) && (hydroforge_isnan(old_value) || new_value {cmp_op} old_value)) {{",
                                        f"{indent}            {out_ptr}[k_base + rank] = new_value;",
                                        f"{indent}            new_value = old_value;",
                                        f"{indent}        }}",
                                        f"{indent}    }}",
                                        f"{indent}}}",
                                    ]
                                )
                        elif outer == "mean":
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend(
                                [
                                    f"{indent}if (is_inner_last) {{",
                                    f"{indent}    {ctype} count = ({ctype})num_macro_steps;",
                                    f"{indent}    {out_ptr}[{out_idx}] = is_outer_first ? {val_var} : hydroforge_weighted_mean({out_ptr}[{out_idx}], count - ({ctype})1, {val_var}, ({ctype})1);",
                                    f"{indent}}}",
                                ]
                            )
                        elif outer == "sum":
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend(
                                [
                                    f"{indent}if (is_inner_last) {{",
                                    f"{indent}    if (is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}",
                                    f"{indent}    else {{ {out_ptr}[{out_idx}] += {val_var}; }}",
                                    f"{indent}}}",
                                ]
                            )
                        elif outer == "last":
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend(
                                [
                                    f"{indent}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {val_var}; }}",
                                ]
                            )
                        elif outer == "first":
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend(
                                [
                                    f"{indent}if (is_inner_last && is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}",
                                ]
                            )
                        continue

                    # ---- Simple ops ----
                    out_ptr = f"p_{safe_var}_{op}"
                    msl_lines.append(f"{indent}// {op} for {safe_var}")

                    if op == "mean":
                        weight_ptr = f"p_{safe_var}_mean_sample_weight_state"
                        msl_lines.extend(
                            [
                                f"{indent}{{",
                                f"{indent}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[{out_idx}];",
                                f"{indent}    {ctype} old_weight = is_inner_first ? ({ctype})0 : {weight_ptr}[{out_idx}];",
                                f"{indent}    {ctype} new_weight = old_weight + ({ctype})weight;",
                                f"{indent}    {out_ptr}[{out_idx}] = hydroforge_weighted_mean(old_val, old_weight, {var_val}, ({ctype})weight);",
                                f"{indent}    {weight_ptr}[{out_idx}] = is_inner_last ? ({ctype})0 : new_weight;",
                                f"{indent}}}",
                            ]
                        )
                    elif op == "sum":
                        msl_lines.extend(
                            [
                                f"{indent}{{",
                                f"{indent}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[{out_idx}];",
                                f"{indent}    {out_ptr}[{out_idx}] = old_val + {var_val} * ({ctype})weight;",
                                f"{indent}}}",
                            ]
                        )
                    elif op in {"max", "min"}:
                        fn = (
                            "hydroforge_maximum"
                            if op == "max"
                            else "hydroforge_minimum"
                        )
                        msl_lines.extend(
                            [
                                f"{indent}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}",
                                f"{indent}else {{ {out_ptr}[{out_idx}] = {fn}({out_ptr}[{out_idx}], {var_val}); }}",
                            ]
                        )
                    elif op == "last":
                        msl_lines.append(
                            f"{indent}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {var_val}; }}"
                        )
                    elif op == "first":
                        msl_lines.append(
                            f"{indent}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}"
                        )

    def _emit_indexed_vector_updates(
        self, *, dims_2d: list[str], indent2: str, msl_lines: list[str]
    ) -> None:
        if dims_2d:
            msl_lines.append(f"{indent2}// === 2D variables ===")
            for var in dims_2d:
                safe_var = self._get_safe_name(var)
                ctype = self._metal_dtype_str(var)
                for operation in self._statistics_lowering.operations(var):
                    op = operation.spelling
                    out_ptr = f"p_{safe_var}_{op}"
                    n_levels = self._statistics_layouts[var].actual_shape[-1]
                    msl_lines.append(
                        f"{indent2}for (int level = 0; level < {n_levels}; level++) {{"
                    )
                    out_2d = f"(t * n_saved_points + tid) * {n_levels} + level"
                    val_2d = self._metal_emit_val_load(
                        var,
                        msl_lines,
                        set(),
                        indent2 + "    ",
                        n_levels=n_levels,
                    )
                    if op == "mean":
                        weight_ptr = f"p_{safe_var}_mean_sample_weight_state"
                        msl_lines.extend(
                            [
                                f"{indent2}    {ctype} old_2d = is_inner_first ? ({ctype})0 : {out_ptr}[{out_2d}];",
                                f"{indent2}    {ctype} old_weight_2d = is_inner_first ? ({ctype})0 : {weight_ptr}[{out_2d}];",
                                f"{indent2}    {ctype} new_weight_2d = old_weight_2d + ({ctype})weight;",
                                f"{indent2}    {out_ptr}[{out_2d}] = hydroforge_weighted_mean(old_2d, old_weight_2d, {val_2d}, ({ctype})weight);",
                                f"{indent2}    {weight_ptr}[{out_2d}] = is_inner_last ? ({ctype})0 : new_weight_2d;",
                            ]
                        )
                    elif op == "sum":
                        msl_lines.extend(
                            [
                                f"{indent2}    {ctype} old_2d = is_inner_first ? ({ctype})0 : {out_ptr}[{out_2d}];",
                                f"{indent2}    {out_ptr}[{out_2d}] = old_2d + {val_2d} * ({ctype})weight;",
                            ]
                        )
                    elif op in {"max", "min"}:
                        fn = (
                            "hydroforge_maximum"
                            if op == "max"
                            else "hydroforge_minimum"
                        )
                        msl_lines.extend(
                            [
                                f"{indent2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = {val_2d}; }}",
                                f"{indent2}    else {{ {out_ptr}[{out_2d}] = {fn}({out_ptr}[{out_2d}], {val_2d}); }}",
                            ]
                        )
                    elif op == "last":
                        msl_lines.append(
                            f"{indent2}    if (is_inner_last) {{ {out_ptr}[{out_2d}] = {val_2d}; }}"
                        )
                    elif op == "first":
                        msl_lines.append(
                            f"{indent2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = {val_2d}; }}"
                        )
                    msl_lines.append(f"{indent2}}}")

    def _emit_full_output_updates(
        self, *, indent: str, msl_lines: list[str], var_list: list[str]
    ) -> None:
        for var in var_list:
            var_numel = prod(self._statistics_layouts[var].actual_shape)
            msl_lines.append(f"{indent}if ((long)tid < {var_numel}) {{")
            self._emit_scalar_updates(
                variables=[var],
                indent=indent + "    ",
                msl_lines=msl_lines,
                out_idx="out_idx",
                full_output=True,
            )
            msl_lines.extend([f"{indent}}}", ""])

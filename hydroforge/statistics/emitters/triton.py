# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Set

from hydroforge.statistics.ir import (
    ExpressionDialect, ExpressionSource, Reduction, ScatterSource, TensorSource,
    render_expression,
)
from hydroforge.statistics.emitters.common import StatisticsEmitter

if TYPE_CHECKING:
    from hydroforge.statistics.runtime import StatisticsRuntime


class TritonStatisticsEmitter(StatisticsEmitter):
    """Triton JIT kernel code generation for statistics aggregation."""

    def emit(self):
        self._generate_triton_aggregator_function()
        return self.result()

    def _generate_triton_aggregator_function(self) -> None:
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        groups = self._statistics_lowering.groups
        lines = self._generate_kernel_header()
        self._generate_scatter_kernels(lines)
        for output_index, variables in groups.items():
            if output_index == "__full__":
                self._generate_full_kernel_for_group(
                    lines, output_index, variables,
                )
            else:
                self._generate_kernel_for_group(
                    lines, f"kernel_{output_index}", output_index, variables,
                )
        self._generate_main_function(lines, groups)
        source = "\n".join(lines)
        self._compile_generated_kernels(source)
        if self.save_kernels:
            self._save_kernel_file(source)

    def _generate_kernel_header(self: StatisticsRuntime) -> List[str]:
        """Generate the header for the kernel file with documentation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        var_list = sorted(list(self._variables))

        header = [
            '"""',
            'Auto-generated Triton kernels for hydroforge statistics aggregation (mean/max/min/last)',
            f'Generated at: {timestamp}',
            f'Rank: {self.rank}',
            f'Variables: {", ".join(var_list)}',
            f'Device: {self.device}',
            '',
            'Kernel Logic:',
            '- Load output_index values to get original grid indices',
            '- Use idx to access original data: data[idx]',
            '- Store outputs using sequential indexing: out[offs]',
            '- explicit argmax/argmin ops store the macro-step index',
            '- argmax/argmin indices are converted to datetime on NC file write',
            '- For mid: stores val when is_middle is True',
            '',
            'Optimizations Applied:',
            '- tl.static_range for compile-time loop unrolling (num_trials, bubble sort)',
            '- Base offset precomputation (shared across max/min/argmax/argmin for same var+K)',
            '- Merged maxK+minK bubble insert in single loop with shared offset',
            '- Precise mask for tl.store: mask & swap_mask to reduce write pressure',
            '"""',
            "",
            "import triton",
            "import triton.language as tl",
            "from triton.language.extra import libdevice",
            "",
            '# ============================================================================',
            f"# Generated Triton kernels for statistics aggregation - Rank {self.rank}",
            "# ============================================================================",
            "",
        ]
        return header



    def _compile_generated_kernels(
        self: StatisticsRuntime, kernel_code: str,
    ) -> None:
        """Compile generated kernels in memory and bind their entry point."""
        module = self._compile_generated_module(
            kernel_code, prefix="statistics",
        )
        self._kernel_module = module
        self._aggregator_function = getattr(module, 'internal_update_statistics')
        self._aggregator_generated = True


    def _generate_scatter_kernels(
        self: StatisticsRuntime,
        kernel_code_lines: List[str],
    ) -> None:
        """Generate Triton kernels for scatter virtual pre-steps.

        For each scatter virtual variable, two kernels are emitted:
          1. ``scatter_zero_{var}``  – fills the target buffer (and count buffer
             for scatter_mean) with zeros.
          2. ``scatter_add_{var}``   – computes the value expression per source
             element and atomically accumulates into the target buffer.
        For *scatter_mean* an additional kernel is emitted:
          3. ``scatter_divide_{var}`` – divides the sum buffer element-wise by the
             count buffer.
        """
        scatter_virtuals = {
            variable.name: variable.source
            for variable in self._statistics_ir.variables
            if isinstance(variable.source, ScatterSource)
        }
        if not scatter_virtuals:
            return

        kernel_code_lines.append(
            "# ======================================================================"
        )
        kernel_code_lines.append(
            "# Triton scatter pre-step kernels"
        )
        kernel_code_lines.append(
            "# ======================================================================"
        )
        kernel_code_lines.append("")

        for var_name, scatter in scatter_virtuals.items():
            safe_var = self._get_safe_name(var_name)
            buf_safe = self._get_safe_name(f"__scatter_buf_{var_name}")
            is_mean = scatter.reduction.value == 'mean'

            # ── 1. Zero kernel ──
            kernel_code_lines.append("@triton.jit")
            if is_mean:
                cnt_safe = self._get_safe_name(f"__scatter_cnt_{var_name}")
                kernel_code_lines.append(
                    f"def scatter_zero_{safe_var}("
                    f"{buf_safe}_ptr, {cnt_safe}_ptr, "
                    f"N, BLOCK_SIZE: tl.constexpr, num_trials: tl.constexpr):"
                )
            else:
                kernel_code_lines.append(
                    f"def scatter_zero_{safe_var}("
                    f"{buf_safe}_ptr, "
                    f"N, BLOCK_SIZE: tl.constexpr, num_trials: tl.constexpr):"
                )
            kernel_code_lines.extend([
                "    pid = tl.program_id(0)",
                "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
                "    mask = offs < N",
                "    for t in tl.static_range(num_trials):",
                f"        tl.store({buf_safe}_ptr + t * N + offs, 0.0, mask=mask)",
            ])
            if is_mean:
                kernel_code_lines.append(
                    f"        tl.store({cnt_safe}_ptr + t * N + offs, 0.0, mask=mask)"
                )
            kernel_code_lines.append("")

            # ── 2. Scatter-add kernel ──
            source_ptrs = set(self._statistics_ir.scatter_inputs(var_name))
            sorted_src = sorted(source_ptrs)

            kernel_code_lines.append("@triton.jit")
            sig_parts = [f"{buf_safe}_ptr"]
            if is_mean:
                sig_parts.append(f"{cnt_safe}_ptr")
            for tok in sorted_src:
                sig_parts.append(f"{self._get_safe_name(tok)}_ptr")
            sig_parts.extend([
                "M", "N",
                "BLOCK_SIZE: tl.constexpr",
                "num_trials: tl.constexpr",
            ])
            # Per-token stride constexprs
            stride_names = {}
            for tok in sorted_src:
                sname = f"stride_{self._get_safe_name(tok)}"
                sig_parts.append(f"{sname}: tl.constexpr")
                stride_names[tok] = sname

            kernel_code_lines.append(
                f"def scatter_add_{safe_var}({', '.join(sig_parts)}):"
            )
            idx_safe = self._get_safe_name(scatter.index)
            kernel_code_lines.extend([
                "    pid = tl.program_id(0)",
                "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
                "    mask = offs < M",
                f"    idx = tl.load({idx_safe}_ptr + offs, mask=mask, other=0).to(tl.int64)",
                "    for t in tl.static_range(num_trials):",
            ])
            emitted_values: dict[str, str] = {}

            def emit_scatter_value(name: str) -> str:
                previous = emitted_values.get(name)
                if previous is not None:
                    return previous
                source = self._statistics_ir.sources.get(name, TensorSource(name))
                safe_name = self._get_safe_name(name)
                value_name = f"{safe_name}_val"
                if isinstance(source, ExpressionSource):
                    names = {
                        dependency: emit_scatter_value(dependency)
                        for dependency in source.expression.dependencies
                    }
                    expression = render_expression(
                        source.expression, ExpressionDialect.TRITON, names,
                    )
                    kernel_code_lines.append(
                        f"        {value_name} = {expression}"
                    )
                else:
                    key = (
                        f"__scatter_buf_{name}"
                        if isinstance(source, ScatterSource) else source.name
                    )
                    pointer = self._get_safe_name(key)
                    kernel_code_lines.append(
                        f"        {value_name} = tl.load({pointer}_ptr + t * "
                        f"{stride_names[key]} + offs, mask=mask, other=0.0)"
                    )
                emitted_values[name] = value_name
                return value_name

            value_names = {
                dependency: emit_scatter_value(dependency)
                for dependency in scatter.value.dependencies
            }
            value_expression = render_expression(
                scatter.value, ExpressionDialect.TRITON, value_names,
            )
            kernel_code_lines.append(f"        _val = {value_expression}")
            kernel_code_lines.append(
                f"        tl.atomic_add({buf_safe}_ptr + t * N + idx, _val, mask=mask)"
            )
            if is_mean:
                kernel_code_lines.append(
                    f"        tl.atomic_add({cnt_safe}_ptr + t * N + idx, 1.0, mask=mask)"
                )
            kernel_code_lines.append("")

            # ── 3. Divide kernel (scatter_mean only) ──
            if is_mean:
                kernel_code_lines.append("@triton.jit")
                kernel_code_lines.append(
                    f"def scatter_divide_{safe_var}("
                    f"{buf_safe}_ptr, {cnt_safe}_ptr, "
                    f"N, BLOCK_SIZE: tl.constexpr, num_trials: tl.constexpr):"
                )
                kernel_code_lines.extend([
                    "    pid = tl.program_id(0)",
                    "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
                    "    mask = offs < N",
                    "    for t in tl.static_range(num_trials):",
                    f"        _cnt = tl.load({cnt_safe}_ptr + t * N + offs, mask=mask, other=1.0)",
                    "        _cnt = tl.where(_cnt > 0.0, _cnt, 1.0)",
                    f"        _val = tl.load({buf_safe}_ptr + t * N + offs, mask=mask, other=0.0)",
                    f"        tl.store({buf_safe}_ptr + t * N + offs, _val / _cnt, mask=mask)",
                ])
                kernel_code_lines.append("")
        kernel_code_lines.append("")


    def _emit_variable_load(self: StatisticsRuntime, var_name: str, code_lines: List[str], emitted: Set[str], is_2d: bool = False):
        """Helper to emit load instructions or expression evaluation recursively."""
        if var_name in emitted:
            return

        # Get safe name for this variable
        safe_var_name = self._get_safe_name(var_name)

        source = self._statistics_ir.sources.get(var_name, TensorSource(var_name))
        if isinstance(source, TensorSource):
            # Real data in tensor registry (includes virtual source buffers)
            indent = "        " if is_2d else "    "
            if is_2d:
                code_lines.append(f"{indent}{safe_var_name} = tl.load({safe_var_name}_ptr + idx * n_levels + level, mask=mask, other=0.0)")
            else:
                code_lines.append(f"{indent}{safe_var_name} = tl.load({safe_var_name}_ptr + idx, mask=mask, other=0.0)")
        elif isinstance(source, ScatterSource):
            # Scatter virtuals are pre-materialized; load like a real tensor
            buf_safe = self._get_safe_name(f"__scatter_buf_{var_name}")
            indent = "        " if is_2d else "    "
            if is_2d:
                code_lines.append(f"{indent}{safe_var_name} = tl.load({buf_safe}_ptr + idx * n_levels + level, mask=mask, other=0.0)")
            else:
                code_lines.append(f"{indent}{safe_var_name} = tl.load({buf_safe}_ptr + idx, mask=mask, other=0.0)")
        elif isinstance(source, ExpressionSource):
            for dependency in source.expression.dependencies:
                self._emit_variable_load(dependency, code_lines, emitted, is_2d)
            names = {
                dependency: self._get_safe_name(dependency)
                for dependency in source.expression.dependencies
            }
            expression = render_expression(
                source.expression, ExpressionDialect.TRITON, names,
            )
            indent = "        " if is_2d else "    "
            code_lines.append(f"{indent}{safe_var_name} = {expression}")

        emitted.add(var_name)

    def _generate_1d_vars_grouped(
        self: StatisticsRuntime, kernel_code_lines: List[str],
        dims_1d: List[str], indent: str, indent2: str,
        indent3: str, indent4: str,
    ) -> None:
        """
        Generate 1D variable processing code with conditions grouped for efficiency.
        All operations under the same condition are emitted in a single if block.
        Supports all ops including maxK/minK bubble insert.

        Arg operations (argmax, argmin, argmax3, etc.) are explicit compound
        operations and are emitted from their typed ``stores_index`` flag.
        """
        if not dims_1d:
            return

        kernel_code_lines.append(f"{indent}# 1D variables")

        # Phase 1: consume the backend-neutral source-load schedule.
        vars_need_val = {
            name for name in dims_1d
            if self._statistics_lowering.by_name[name].needs_unconditional_value
        }
        vars_conditional_only = set(dims_1d).difference(vars_need_val)

        # Helper to emit variable value load
        emitted_vars = set()

        def emit_val(v_name, to_lines):
            safe_v_name = self._get_safe_name(v_name)
            if safe_v_name in emitted_vars:
                return f"{safe_v_name}_val"

            source = self._statistics_ir.sources.get(v_name, TensorSource(v_name))
            if isinstance(source, TensorSource):
                # Real data (includes virtual source buffers)
                in_ptr_loc = f"{safe_v_name}_ptr + t * stride_input + idx"
                to_lines.append(f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
            elif isinstance(source, ScatterSource):
                buf_safe = self._get_safe_name(f"__scatter_buf_{v_name}")
                in_ptr_loc = f"{buf_safe}_ptr + t * stride_input + idx"
                to_lines.append(f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
            elif isinstance(source, ExpressionSource):
                names = {
                    dependency: emit_val(dependency, to_lines)
                    for dependency in source.expression.dependencies
                }
                expression = render_expression(
                    source.expression, ExpressionDialect.TRITON, names,
                )
                to_lines.append(f"{indent}{safe_v_name}_val = {expression}")

            emitted_vars.add(safe_v_name)
            return f"{safe_v_name}_val"

        # Phase 2: Collect all operations grouped by condition
        ops_unconditional = []
        ops_is_inner_first = []
        ops_not_is_inner_first = []
        ops_is_inner_last = []
        ops_is_inner_last_is_outer_first = []

        # Special storage for maxK/minK operations (need for loop)
        maxk_ops = []
        argmaxk_ops = []
        ops_is_inner_last_not_is_outer_first = []
        ops_is_inner_last_is_outer_last = []
        ops_is_inner_last_not_is_outer_last = []
        ops_is_middle = []

        # Track which inner aggregations are needed
        inner_aggregations_needed = (
            self._statistics_lowering.variables_by_inner(dims_1d)
        )

        for var in dims_1d:
            safe_var = self._get_safe_name(var)
            operations = self._statistics_lowering.operations(var)
            out_offset = "t * n_saved_points + offs"
            # Process each operation
            for operation in operations:
                op = operation.spelling
                out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"

                # ===== Compound operations (e.g., max_mean, min_mean) =====
                if operation.compound:
                    outer = operation.outer.value
                    inner = operation.inner.value
                    k_val = operation.k
                    is_arg_compound = operation.stores_index
                    outer_base = operation.outer.value

                    # Use variable-specific inner aggregation result
                    # For 'last' inner type, directly use the variable value (no intermediate state)
                    if inner == 'last':
                        val_var = f"{safe_var}_val"
                    else:
                        val_var = f"val_for_{safe_var}_{inner}"

                    if is_arg_compound:
                        # Compound argmax/argmin (e.g., argmax_mean, argmax3_mean)
                        arg_type = outer_base  # 'max' or 'min'
                        aux_ptr_base = f"{safe_var}_{arg_type}{k_val if k_val > 1 else ''}_aux_ptr"

                        if k_val == 1:
                            aux_ptr = f"{aux_ptr_base} + {out_offset}"
                            if arg_type == 'max':
                                ops_is_inner_last_is_outer_first.extend([
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                                    f"tl.store({aux_ptr}, {val_var}, mask=mask)",
                                ])
                                ops_is_inner_last_not_is_outer_first.extend([
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other={val_var})",
                                    f"{safe_var}_{op}_cond = {val_var} > {safe_var}_{op}_aux_old",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ])
                            else:  # min
                                ops_is_inner_last_is_outer_first.extend([
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                                    f"tl.store({aux_ptr}, {val_var}, mask=mask)",
                                ])
                                ops_is_inner_last_not_is_outer_first.extend([
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other={val_var})",
                                    f"{safe_var}_{op}_cond = {val_var} < {safe_var}_{op}_aux_old",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ])
                        else:
                            # ArgmaxK/ArgminK compound bubble insert
                            argmaxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': f'arg{arg_type}',
                                'has_val_output': False  # compound arg doesn't need val output
                            })
                    elif outer_base == 'max':
                        # Compound max without automatic arg (e.g., max_mean, max3_mean)
                        if k_val == 1:
                            ops_is_inner_last_is_outer_first.append(
                                f"tl.store({out_ptr}, {val_var}, mask=mask)")
                            ops_is_inner_last_not_is_outer_first.extend([
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                f"tl.store({out_ptr}, tl.maximum({safe_var}_{op}_old, {val_var}), mask=mask)",
                            ])
                        else:
                            # maxK bubble insert without arg tracking
                            maxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': 'max'
                            })

                    elif outer_base == 'min':
                        # Compound min without automatic arg (e.g., min_mean, min3_mean)
                        if k_val == 1:
                            ops_is_inner_last_is_outer_first.append(
                                f"tl.store({out_ptr}, {val_var}, mask=mask)")
                            ops_is_inner_last_not_is_outer_first.extend([
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                f"tl.store({out_ptr}, tl.minimum({safe_var}_{op}_old, {val_var}), mask=mask)",
                            ])
                        else:
                            # minK bubble insert without arg tracking
                            maxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': 'min'
                            })

                    elif outer == 'mean':
                        ops_is_inner_last_is_outer_first.append(
                            f"{safe_var}_{op}_accum = {val_var}")
                        ops_is_inner_last_not_is_outer_first.append(
                            f"{safe_var}_{op}_accum = tl.load({out_ptr}, mask=mask, other=0.0) + {val_var}")
                        ops_is_inner_last_is_outer_last.append(
                            f"tl.store({out_ptr}, {safe_var}_{op}_accum / num_macro_steps, mask=mask)")
                        ops_is_inner_last_not_is_outer_last.append(
                            f"tl.store({out_ptr}, {safe_var}_{op}_accum, mask=mask)")

                    elif outer == 'sum':
                        ops_is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)")
                        ops_is_inner_last_not_is_outer_first.extend([
                            f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other=0.0)",
                            f"tl.store({out_ptr}, {safe_var}_{op}_old + {val_var}, mask=mask)",
                        ])
                    elif outer == 'last':
                        # Compound last (e.g., last_mean) — store the last inner value
                        # Simply overwrite on every is_inner_last step
                        ops_is_inner_last.append(f"tl.store({out_ptr}, {val_var}, mask=mask)")
                    elif outer == 'first':
                        # Compound first (e.g., first_mean) — store only at is_outer_first
                        ops_is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)")
                    continue

                # ===== Simple operations (non-compound) =====
                if op == 'mean':
                    inner_ops = {
                        reduction.value
                        for reduction in self._statistics_lowering.inner_reductions(var)
                    }
                    if 'mean' in inner_ops:
                        # Reuse val_for_{safe_var}_mean from inner aggregation
                        ops_is_inner_last.append(f"tl.store({out_ptr}, val_for_{safe_var}_mean, mask=mask)")
                    else:
                        # Standalone mean - needs state (use variable-specific val)
                        ops_unconditional.extend([
                            f"# Standalone mean for {safe_var}",
                            f"{safe_var}_mean_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                            f"{safe_var}_mean_new = {safe_var}_mean_old + {safe_var}_val * weight",
                            f"{safe_var}_mean_out = tl.where(is_inner_last, {safe_var}_mean_new / total_weight, {safe_var}_mean_new)",
                        ])
                        ops_unconditional.append(f"tl.store({out_ptr}, {safe_var}_mean_out, mask=mask)")

                elif op == 'sum':
                    ops_unconditional.extend([
                        f"{safe_var}_sum_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                        f"tl.store({out_ptr}, {safe_var}_sum_old + {safe_var}_val * weight, mask=mask)",
                    ])

                # Standalone extrema never carry an index or top-k suffix;
                # parse_operation routes those semantics through compound ops.
                elif op == 'max':
                    ops_is_inner_first.extend([
                        f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                    ])
                    ops_not_is_inner_first.extend([
                        f"{safe_var}_max_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                        f"tl.store({out_ptr}, tl.maximum({safe_var}_max_old, {safe_var}_val), mask=mask)",
                    ])

                elif op == 'min':
                    ops_is_inner_first.extend([
                        f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                    ])
                    ops_not_is_inner_first.extend([
                        f"{safe_var}_min_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                        f"tl.store({out_ptr}, tl.minimum({safe_var}_min_old, {safe_var}_val), mask=mask)",
                    ])

                elif op == 'last':
                    if var in vars_conditional_only:
                        # Check if this var also has compound ops with 'last' inner type
                        # If so, the deferred load inside is_inner_last block will already load the val
                        has_compound_last = any(
                            other.inner is not None and other.inner.value == 'last'
                            for other in operations
                        )
                        if has_compound_last:
                            # Reuse the val loaded by deferred load (no duplicate load needed)
                            ops_is_inner_last.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        else:
                            # Load inline — use emit_val for scatter virtual support
                            _tmp = []
                            emit_val(var, _tmp)
                            ops_is_inner_last.extend(line.lstrip() for line in _tmp)
                            ops_is_inner_last.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                    else:
                        ops_is_inner_last.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")

                elif op == 'first':
                    if var in vars_conditional_only:
                        has_compound_first = any(
                            other.inner is not None and other.inner.value == 'first'
                            for other in operations
                        )
                        if has_compound_first:
                            # Val will be loaded elsewhere (from unconditional or conditional load)
                            ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        else:
                            # Load inline — use emit_val for scatter virtual support
                            _tmp = []
                            emit_val(var, _tmp)
                            ops_is_inner_first.extend(line.lstrip() for line in _tmp)
                            ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                    else:
                        ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")

                elif op == 'mid':
                    if var in vars_conditional_only:
                        # Load inline — use emit_val for scatter virtual support
                        _tmp = []
                        emit_val(var, _tmp)
                        ops_is_middle.extend(line.lstrip() for line in _tmp)
                        ops_is_middle.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                    else:
                        ops_is_middle.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")

        # Phase 3: Emit loads for vars that need unconditional val
        for var in vars_need_val:
            emit_val(var, kernel_code_lines)

        # For conditional-only vars used in compound ops with 'last' inner type,
        # ensure the variable val is emitted (will be loaded inside is_inner_last block later)
        # We need to track them but NOT emit unconditional loads here.
        # The load will be emitted inside the is_inner_last block in Phase 6.

        # Phase 4: Emit inner aggregation state updates (per-variable)
        # Each variable gets its own inner aggregation state (val_for_{safe_var}_{inner_type})
        # For 'last' inner type, no state is needed - the value is simply the current variable value
        # used directly inside the `if is_inner_last:` block.
        for reduction, inner_vars in inner_aggregations_needed.items():
            inner_type = reduction.value
            for var in inner_vars:
                safe_var = self._get_safe_name(var)
                out_offset = "t * n_saved_points + offs"
                val_for_var_inner = f"val_for_{safe_var}_{inner_type}"
                var_val = f"{safe_var}_val"

                if inner_type == 'last':
                    # 'last' is the simplest: val_for_X_last == X_val at is_inner_last.
                    # No state storage, no load/store needed.
                    pass
                elif inner_type == 'mean':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    weight_ptr = f"{safe_var}_{inner_type}_weight_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_weight_{inner_type}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_inner_{inner_type}_new = {safe_var}_inner_{inner_type}_old + {var_val} * weight",
                        f"{indent}{safe_var}_weight_{inner_type}_new = {safe_var}_weight_{inner_type}_old + weight",
                    ])
                    # Store based on condition - use tl.where for efficiency
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_weight_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new / {safe_var}_weight_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'sum':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_inner_{inner_type}_new = {safe_var}_inner_{inner_type}_old + {var_val} * weight",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'max':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first & (macro_step_index==0), {var_val}, tl.maximum({safe_var}_inner_{inner_type}_old, {var_val}))",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, -float('inf'), {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'min':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first & (macro_step_index==0), {var_val}, tl.minimum({safe_var}_inner_{inner_type}_old, {var_val}))",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, float('inf'), {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'first':
                    # 'first' inner: store the value at is_inner_first, read it back at is_inner_last
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, {var_val}, mask=mask & is_inner_first)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, tl.load({inner_ptr}, mask=mask, other=0.0), {val_for_var_inner})",
                    ])
                elif inner_type == 'mid':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, {var_val}, mask=mask & is_middle)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, tl.load({inner_ptr}, mask=mask, other=0.0), {val_for_var_inner})",
                    ])
                # Note: removed 'break' - now generate for each variable in inner_vars

        # Phase 5: Emit unconditional ops
        for line in ops_unconditional:
            kernel_code_lines.append(f"{indent}{line}")

        # Phase 6: Emit grouped conditional blocks
        if ops_is_inner_first:
            kernel_code_lines.append(f"{indent}if is_inner_first:")
            for line in ops_is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")

        if ops_not_is_inner_first:
            if ops_is_inner_first:
                kernel_code_lines.append(f"{indent}else:")
            else:
                kernel_code_lines.append(f"{indent}if not is_inner_first:")
            for line in ops_not_is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")

        if ops_is_middle:
            kernel_code_lines.append(f"{indent}if is_middle:")
            for line in ops_is_middle:
                kernel_code_lines.append(f"{indent2}{line}")

        # Nested conditions for is_inner_last with outer conditions
        has_argmaxk_ops = bool(argmaxk_ops)
        has_inner_last_ops = (ops_is_inner_last or ops_is_inner_last_is_outer_first or
                             ops_is_inner_last_not_is_outer_first or ops_is_inner_last_is_outer_last or
                             ops_is_inner_last_not_is_outer_last or maxk_ops or has_argmaxk_ops)

        if has_inner_last_ops:
            kernel_code_lines.append(f"{indent}if is_inner_last:")

            # Emit deferred loads for conditional-only vars used in compound ops
            # These vars are only needed inside is_inner_last, so we load them here
            for var in dims_1d:
                if (
                    var in vars_conditional_only
                    and var in inner_aggregations_needed.get(Reduction.LAST, ())
                ):
                    emit_val(var, kernel_code_lines)
                    # Patch the emitted line to use is_inner_last indentation
                    # The emit_val appends to kernel_code_lines with 'indent' (8 spaces)
                    # We need it at indent2 (12 spaces) since we're inside 'if is_inner_last:'
                    if kernel_code_lines and kernel_code_lines[-1].startswith(indent) and not kernel_code_lines[-1].startswith(indent2):
                        last_line = kernel_code_lines.pop()
                        kernel_code_lines.append(f"{indent2}{last_line.lstrip()}")

            # is_outer_first / not is_outer_first
            if ops_is_inner_last_is_outer_first or ops_is_inner_last_not_is_outer_first:
                kernel_code_lines.append(f"{indent2}if is_outer_first:")
                for line in ops_is_inner_last_is_outer_first:
                    kernel_code_lines.append(f"{indent3}{line}")
                if ops_is_inner_last_not_is_outer_first:
                    kernel_code_lines.append(f"{indent2}else:")
                    for line in ops_is_inner_last_not_is_outer_first:
                        kernel_code_lines.append(f"{indent3}{line}")

            # is_outer_last / not is_outer_last (for mean finalization)
            if ops_is_inner_last_is_outer_last or ops_is_inner_last_not_is_outer_last:
                kernel_code_lines.append(f"{indent2}if is_outer_last:")
                for line in ops_is_inner_last_is_outer_last:
                    kernel_code_lines.append(f"{indent3}{line}")
                if ops_is_inner_last_not_is_outer_last:
                    kernel_code_lines.append(f"{indent2}else:")
                    for line in ops_is_inner_last_not_is_outer_last:
                        kernel_code_lines.append(f"{indent3}{line}")

            # Simple is_inner_last ops
            for line in ops_is_inner_last:
                kernel_code_lines.append(f"{indent2}{line}")

            # ================================================================
            # Optimized MaxK/MinK + ArgmaxK/ArgminK bubble insert operations

            # Group by (var, k, out_offset) to share base offset across all operations
            from collections import defaultdict
            grouped_by_var_k = defaultdict(lambda: {'max': None, 'min': None, 'argmax': None, 'argmin': None})

            for maxk_op in maxk_ops:
                key = (maxk_op['var'], maxk_op['k'], maxk_op['out_offset'])
                grouped_by_var_k[key][maxk_op['type']] = maxk_op

            if argmaxk_ops:
                for argk_op in argmaxk_ops:
                    key = (argk_op['var'], argk_op['k'], argk_op['out_offset'])
                    op_type = 'argmax' if 'max' in argk_op['type'] else 'argmin'
                    grouped_by_var_k[key][op_type] = argk_op

            # Process grouped operations with shared offset
            for (safe_var, k_val, out_offset), ops_dict in grouped_by_var_k.items():
                has_max = ops_dict['max'] is not None
                has_min = ops_dict['min'] is not None
                has_argmax = ops_dict['argmax'] is not None
                has_argmin = ops_dict['argmin'] is not None

                # Get val_var from each operation (may differ: val for max/min, val_for_mean for argmax/argmin)
                max_val_var = ops_dict['max']['val_var'] if has_max else None
                min_val_var = ops_dict['min']['val_var'] if has_min else None
                argmax_val_var = ops_dict['argmax']['val_var'] if has_argmax else None
                argmin_val_var = ops_dict['argmin']['val_var'] if has_argmin else None

                # Compute shared base offset once
                out_offset_k = f"({out_offset}) * {k_val}"

                # Generate header comment
                op_names = []
                if has_max:
                    op_names.append(f"max{k_val}")
                if has_min:
                    op_names.append(f"min{k_val}")
                if has_argmax:
                    op_names.append(f"argmax{k_val}")
                if has_argmin:
                    op_names.append(f"argmin{k_val}")
                kernel_code_lines.append(f"{indent2}# Merged Bubble Insert [{'+'.join(op_names)}] for {safe_var} (shared offset, precise mask)")

                # Shared base offset computation
                kernel_code_lines.append(f"{indent2}{safe_var}_k{k_val}_base_offs = {out_offset_k}")

                # Initialize new values for bubble insert (using correct val_var for each op type)
                if has_max:
                    kernel_code_lines.append(f"{indent2}new_val_max_{safe_var} = {max_val_var}")
                if has_min:
                    kernel_code_lines.append(f"{indent2}new_val_min_{safe_var} = {min_val_var}")
                if has_argmax:
                    kernel_code_lines.append(f"{indent2}new_val_argmax_{safe_var} = {argmax_val_var}")
                if has_argmin:
                    kernel_code_lines.append(f"{indent2}new_val_argmin_{safe_var} = {argmin_val_var}")
                if has_argmax or has_argmin:
                    kernel_code_lines.append(f"{indent2}new_idx_{safe_var} = tl.full([BLOCK_SIZE], macro_step_index, dtype=tl.int32)")

                # is_outer_first branch: initialize all arrays
                kernel_code_lines.append(f"{indent2}if is_outer_first:")

                # First position stores the initial value
                if has_max:
                    max_ptr = f"{safe_var}_{ops_dict['max']['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs, new_val_max_{safe_var}, mask=mask)")
                if has_min:
                    min_ptr = f"{safe_var}_{ops_dict['min']['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs, new_val_min_{safe_var}, mask=mask)")
                if has_argmax:
                    argmax_op = ops_dict['argmax']
                    argmax_aux_ptr = f"{safe_var}_max{k_val}_aux_ptr"
                    argmax_idx_ptr = f"{safe_var}_{argmax_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs, new_idx_{safe_var}, mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                if has_argmin:
                    argmin_op = ops_dict['argmin']
                    argmin_aux_ptr = f"{safe_var}_min{k_val}_aux_ptr"
                    argmin_idx_ptr = f"{safe_var}_{argmin_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs, new_idx_{safe_var}, mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")

                # Initialize remaining positions with inf/-inf
                kernel_code_lines.append(f"{indent3}for k in tl.static_range(1, {k_val}):")
                if has_max:
                    kernel_code_lines.append(f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                if has_min:
                    kernel_code_lines.append(f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                if has_argmax:
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, 0, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                if has_argmin:
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, 0, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")

                # else branch: bubble insert
                kernel_code_lines.append(f"{indent2}else:")
                kernel_code_lines.append(f"{indent3}for k in tl.static_range({k_val}):")

                # Load old values and compute swap masks
                if has_max:
                    kernel_code_lines.extend([
                        f"{indent4}old_max_k = tl.load({max_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-float('inf'))",
                        f"{indent4}swap_max = new_val_max_{safe_var} > old_max_k",
                        f"{indent4}max_to_store = tl.where(swap_max, new_val_max_{safe_var}, old_max_k)",
                        f"{indent4}new_val_max_{safe_var} = tl.where(swap_max, old_max_k, new_val_max_{safe_var})",
                        f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, max_to_store, mask=mask & swap_max)",
                    ])
                if has_min:
                    kernel_code_lines.extend([
                        f"{indent4}old_min_k = tl.load({min_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('inf'))",
                        f"{indent4}swap_min = new_val_min_{safe_var} < old_min_k",
                        f"{indent4}min_to_store = tl.where(swap_min, new_val_min_{safe_var}, old_min_k)",
                        f"{indent4}new_val_min_{safe_var} = tl.where(swap_min, old_min_k, new_val_min_{safe_var})",
                        f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, min_to_store, mask=mask & swap_min)",
                    ])
                if has_argmax:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmax_aux_k = tl.load({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-float('inf'))",
                        f"{indent4}old_argmax_idx_k = tl.load({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=0)",
                        f"{indent4}swap_argmax = new_val_argmax_{safe_var} > old_argmax_aux_k",
                        f"{indent4}argmax_aux_store = tl.where(swap_argmax, new_val_argmax_{safe_var}, old_argmax_aux_k)",
                        f"{indent4}argmax_idx_store = tl.where(swap_argmax, new_idx_{safe_var}, old_argmax_idx_k)",
                        f"{indent4}new_val_argmax_{safe_var} = tl.where(swap_argmax, old_argmax_aux_k, new_val_argmax_{safe_var})",
                        f"{indent4}new_idx_{safe_var} = tl.where(swap_argmax, old_argmax_idx_k, new_idx_{safe_var})",
                        f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)",
                        f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_idx_store, mask=mask & swap_argmax)",
                    ])
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)")
                if has_argmin:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmin_aux_k = tl.load({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('inf'))",
                        f"{indent4}old_argmin_idx_k = tl.load({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=0)",
                        f"{indent4}swap_argmin = new_val_argmin_{safe_var} < old_argmin_aux_k",
                        f"{indent4}argmin_aux_store = tl.where(swap_argmin, new_val_argmin_{safe_var}, old_argmin_aux_k)",
                        f"{indent4}argmin_idx_store = tl.where(swap_argmin, new_idx_{safe_var}, old_argmin_idx_k)",
                        f"{indent4}new_val_argmin_{safe_var} = tl.where(swap_argmin, old_argmin_aux_k, new_val_argmin_{safe_var})",
                        f"{indent4}new_idx_{safe_var} = tl.where(swap_argmin, old_argmin_idx_k, new_idx_{safe_var})",
                        f"{indent4}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, argmin_aux_store, mask=mask & swap_argmin)",
                        f"{indent4}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, argmin_idx_store, mask=mask & swap_argmin)",
                    ])
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, argmin_aux_store, mask=mask & swap_argmin)")

        kernel_code_lines.append("")


    def _generate_kernel_for_group(
        self: StatisticsRuntime, kernel_code_lines: List[str],
        kernel_name: str, output_index: str, var_list: List[str],
    ) -> None:
        """Generate kernel code for a specific output_index group supporting ops."""
        dims_1d, dims_2d = self._statistics_lowering.split_indexed(var_list)

        # Header
        kernel_code_lines.extend([
            f"# Kernel for output_index: {output_index}",
            f"# Variables: {', '.join(var_list)}",
            f"# 1D: {', '.join(dims_1d) if dims_1d else 'None'}",
            f"# 2D: {', '.join(dims_2d) if dims_2d else 'None'}",
            "",
            "@triton.jit",
            f"def {kernel_name}(",
            f"    {output_index}_ptr,",
        ])

        input_ptrs = {
            name for var in var_list
            for name in self._statistics_ir.materialized_inputs(var)
        }

        # Pointers
        # Inputs
        sorted_inputs = sorted(list(input_ptrs))
        for var in sorted_inputs:
            safe_var = self._get_safe_name(var)
            # Avoid duplicate argument if output_index matches input var
            if safe_var == output_index:
                continue
            kernel_code_lines.append(f"    {safe_var}_ptr,")

        for var in var_list:
            safe_var = self._get_safe_name(var)
            # Track which extra state pointers have been added to avoid duplicates
            added_aux_ptrs = set()  # Track aux pointers already added (for explicit argmax/argmin)

            variable = self._statistics_lowering.by_name[var]
            for operation in variable.operations:
                op = operation.spelling
                kernel_code_lines.append(f"    {safe_var}_{op}_ptr,")

                # For EXPLICIT argmax/argmin operators, add aux pointer for tracking values
                # NO automatic argmax/argmin generation for max/min operations
                if operation.stores_index:
                    arg_type = operation.outer.value
                    arg_k_str = "" if operation.k == 1 else str(operation.k)
                    # aux pointer name: {safe_var}_{arg_type}{k}_aux_ptr (e.g., var_max_aux_ptr, var_max3_aux_ptr)
                    aux_name = f"{arg_type}{arg_k_str}_aux"  # e.g., 'max_aux', 'max3_aux'
                    if aux_name not in added_aux_ptrs:
                        kernel_code_lines.append(f"    {safe_var}_{aux_name}_ptr,")
                        added_aux_ptrs.add(aux_name)

            # Inner state pointers (only for ops that need cross-step state)
            added_inner = set()
            for operation in variable.operations:
                if operation.inner is not None:
                    inner = operation.inner.value
                    if inner not in added_inner:
                        # 'last' inner op directly uses current value, no state needed
                        if inner != 'last':
                            kernel_code_lines.append(f"    {safe_var}_{inner}_inner_state_ptr,")
                        if inner == 'mean':
                            kernel_code_lines.append(f"    {safe_var}_{inner}_weight_state_ptr,")
                        added_inner.add(inner)

        kernel_code_lines.extend([
            "    weight_ptr,",
            "    total_weight_ptr,",
            "    num_macro_steps_ptr,",
            "    sub_step_ptr,",
            "    num_sub_steps_ptr,",
            "    flags_ptr,",
            "    macro_step_index_ptr,",
            "    n_saved_points: tl.constexpr,",
        ])
        if dims_2d:
            kernel_code_lines.append("    n_levels: tl.constexpr,")
        kernel_code_lines.extend([
            "    BLOCK_SIZE: tl.constexpr,",
            "    num_trials: tl.constexpr,",
            "    stride_input: tl.constexpr,",
            "):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_saved_points",
            "",
            "    # Load scalar parameters from device tensors",
            "    weight = tl.load(weight_ptr)",
            "    total_weight = tl.load(total_weight_ptr)",
            "    num_macro_steps = tl.load(num_macro_steps_ptr)",
            "    sub_step = tl.load(sub_step_ptr).to(tl.int32)",
            "    num_sub_steps = tl.load(num_sub_steps_ptr).to(tl.int32)",
            "    flags = tl.load(flags_ptr).to(tl.int32)",
            "    macro_step_index = tl.load(macro_step_index_ptr).to(tl.int32)",
            "",
        ])

        # Only emit boolean computation lines for booleans actually used by ops
        needed_bools = self._statistics_lowering.required_flags
        if needed_bools:
            kernel_code_lines.append("    # Compute boolean flags from sub_step, num_sub_steps, flags")
            if 'is_inner_first' in needed_bools:
                kernel_code_lines.append("    is_inner_first = (flags & 1) != 0 and sub_step == 0")
            if 'is_inner_last' in needed_bools:
                kernel_code_lines.append("    is_inner_last = ((flags >> 1) & 1) != 0 and sub_step == num_sub_steps - 1")
            if 'is_middle' in needed_bools:
                kernel_code_lines.append("    is_middle = sub_step == num_sub_steps // 2")
            if 'is_outer_first' in needed_bools:
                kernel_code_lines.append("    is_outer_first = ((flags >> 2) & 1) != 0 and is_inner_last")
            if 'is_outer_last' in needed_bools:
                kernel_code_lines.append("    is_outer_last = ((flags >> 3) & 1) != 0 and is_inner_last")
            kernel_code_lines.append("")

        kernel_code_lines.extend([
            f"    idx = tl.load({output_index}_ptr + offs, mask=mask)",
            "",
        ])

        # Loop over trials - use tl.static_range for compile-time unrolling
        kernel_code_lines.append("    for t in tl.static_range(num_trials):")
        indent = "        "
        indent2 = indent + "    "
        indent3 = indent2 + "    "
        indent4 = indent3 + "    "
        indent5 = indent4 + "    "

        # 1D processing - use grouped generation for all vars
        if dims_1d:
            self._generate_1d_vars_grouped(kernel_code_lines, dims_1d,
                                           indent, indent2, indent3, indent4)

        # 2D processing
        if dims_2d:
            def is_last_only(name: str) -> bool:
                operations = self._statistics_lowering.operations(name)
                return (
                    len(operations) == 1
                    and not operations[0].compound
                    and operations[0].outer.value == "last"
                )

            non_last_only = [v for v in dims_2d if not is_last_only(v)]
            last_only_vars = [v for v in dims_2d if is_last_only(v)]

            if non_last_only:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (mean/min/max and mixed)",
                    f"{indent}for level in tl.static_range(n_levels):",
                ])
                emitted_vars_2d = set()
                def emit_val_2d(v_name):
                    safe_v_name = self._get_safe_name(v_name)
                    if safe_v_name in emitted_vars_2d:
                        return f"{safe_v_name}_val"

                    source = self._statistics_ir.sources.get(
                        v_name, TensorSource(v_name),
                    )
                    if isinstance(source, ExpressionSource):
                        names = {
                            dependency: emit_val_2d(dependency)
                            for dependency in source.expression.dependencies
                        }
                        expression = render_expression(
                            source.expression, ExpressionDialect.TRITON, names,
                        )
                        kernel_code_lines.append(
                            f"{indent2}{safe_v_name}_val = {expression}"
                        )
                    else:
                        key = (
                            f"__scatter_buf_{v_name}"
                            if isinstance(source, ScatterSource) else source.name
                        )
                        pointer = self._get_safe_name(key)
                        in_ptr_loc = f"{pointer}_ptr + (t * stride_input + idx) * n_levels + level"
                        kernel_code_lines.append(f"{indent2}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")

                    emitted_vars_2d.add(safe_v_name)
                    return f"{safe_v_name}_val"

                for var in non_last_only:
                    safe_var = self._get_safe_name(var)
                    out_offset = "(t * n_saved_points + offs) * n_levels + level"

                    val_name = emit_val_2d(var)
                    kernel_code_lines.append(f"{indent2}val = {val_name}")

                    # Inner states update
                    inner_ops = (
                        reduction.value
                        for reduction in self._statistics_lowering.inner_reductions(var)
                    )
                    for inner in inner_ops:
                        # Initialize val_for_inner to avoid UnboundLocalError/NameError in generated code
                        # This value is used if is_update_outer is True, where it gets overwritten.
                        kernel_code_lines.append(f"{indent2}val_for_{inner} = tl.zeros_like(val)")

                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + {out_offset}"
                        if inner == 'mean':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new / (weight_{inner}_new)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'sum':
                             kernel_code_lines.extend([
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'max':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and macro_step_index==0:",
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent3}inner_{inner}_new = tl.maximum(inner_{inner}_old, val)",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, -float('inf'), mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'min':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and macro_step_index==0:",
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent3}inner_{inner}_new = tl.minimum(inner_{inner}_old, val)",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, float('inf'), mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'first':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}val_stored = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}if weight_{inner}_old == 0.0:",
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_new = val_stored",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'last':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}val_for_{inner} = val",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'mid':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_middle:",
                                 f"{indent3}tl.store({inner_ptr}, val, mask=mask)",
                                 f"{indent2}if is_inner_last:",
                                 f"{indent3}val_for_{inner} = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])

                    for operation in self._statistics_lowering.operations(var):
                        op = operation.spelling
                        out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"
                        if operation.compound:
                            outer = operation.outer.value
                            inner = operation.inner.value
                            k_val = operation.k

                            val_var = f"val_for_{inner}"
                            kernel_code_lines.append(f"{indent2}if is_macro_step_end:")

                            if outer == 'max':
                                # argmax pointer (automatically created alongside max)
                                argmax_ptr = f"{safe_var}_arg{op}_ptr + {out_offset}"
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent4}tl.store({argmax_ptr}, macro_step_index, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} > old",
                                        f"{indent4}new = tl.maximum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                        f"{indent4}tl.store({argmax_ptr}, macro_step_index, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Bubble Insert Max K with ArgMax
                                    argmax_op = f"arg{op}"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Max K={k_val} with ArgMax (static_range optimized)",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], macro_step_index, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {safe_var}_{op}_ptr + k_offset",
                                        f"{indent3}idx_base_ptr = {safe_var}_{argmax_op}_ptr + k_offset",

                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent4}for k in tl.static_range(1, {k_val}):",
                                        f"{indent5}tl.store(base_ptr + k, -float('inf'), mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, 0, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in tl.static_range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=-float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = new_val > old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])

                            elif outer == 'min':
                                # argmin pointer (automatically created alongside min)
                                argmin_ptr = f"{safe_var}_arg{op}_ptr + {out_offset}"
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent4}tl.store({argmin_ptr}, macro_step_index, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} < old",
                                        f"{indent4}new = tl.minimum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                        f"{indent4}tl.store({argmin_ptr}, macro_step_index, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Min K with ArgMin
                                    argmin_op = f"arg{op}"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Min K={k_val} with ArgMin (static_range optimized)",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], macro_step_index, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {safe_var}_{op}_ptr + k_offset",
                                        f"{indent3}idx_base_ptr = {safe_var}_{argmin_op}_ptr + k_offset",

                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent4}for k in tl.static_range(1, {k_val}):",
                                        f"{indent5}tl.store(base_ptr + k, float('inf'), mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, 0, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in tl.static_range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = new_val < old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])
                            elif outer == 'sum':
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and macro_step_index==0:",
                                    f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                    f"{indent3}else:",
                                    f"{indent4}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent4}tl.store({out_ptr}, old + {val_var}, mask=mask)",
                                ])
                            elif outer == 'mean':
                                term = f"{val_var}"
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and macro_step_index==0:",
                                    f"{indent4}tl.store({out_ptr}, {term}, mask=mask)",
                                    f"{indent3}else:",
                                    f"{indent4}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent4}tl.store({out_ptr}, old + {term}, mask=mask)",
                                    f"{indent3}if is_outer_last:",
                                    f"{indent4}chk = tl.load({out_ptr}, mask=mask)",
                                    f"{indent4}tl.store({out_ptr}, chk / num_macro_steps, mask=mask)",
                                ])
                            continue

                        if op == 'mean':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new = old + val * weight",
                                f"{indent2}if is_inner_last:",
                                f"{indent3}new = new / total_weight",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'sum':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new = old + val * weight",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'max':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = tl.maximum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'min':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = tl.minimum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'argmax':
                            # 2D argmax logic
                            max_ptr = f"{safe_var}_max_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_max = tl.load({max_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val > curr_max",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask & cond_mask)",
                            ])
                        elif op == 'argmin':
                            min_ptr = f"{safe_var}_min_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_min = tl.load({min_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val < curr_min",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask & cond_mask)",
                            ])
                        elif op == 'last':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_last:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'first':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'mid':
                            kernel_code_lines.extend([
                                f"{indent2}if is_middle:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                kernel_code_lines.append("")

            if last_only_vars:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (last-only)",
                    f"{indent}if is_inner_last:",
                    f"{indent2}for level in tl.static_range(n_levels):",
                ])
                for var in last_only_vars:
                    safe_var = self._get_safe_name(var)
                    out_offset = "(t * n_saved_points + offs) * n_levels + level"

                    val_name = emit_val_2d(var)
                    kernel_code_lines.extend([
                        f"{indent3}val = {val_name}",
                        f"{indent3}tl.store({safe_var}_last_ptr + {out_offset}, val, mask=mask)",
                    ])
        kernel_code_lines.append("")


    def _generate_full_kernel_for_group(
        self: StatisticsRuntime, kernel_code_lines: List[str],
        output_index: str, var_list: List[str],
    ) -> None:
        """Generate a flat Triton kernel for variables saved at full tensor shape."""
        kernel_name = f"kernel_{output_index}"
        kernel_code_lines.extend([
            f"# Full-output kernel: {output_index}",
            f"# Variables: {', '.join(var_list)}",
            "",
            "@triton.jit",
            f"def {kernel_name}(",
        ])

        for var in var_list:
            safe_var = self._get_safe_name(var)
            kernel_code_lines.append(f"    {safe_var}_ptr,")
            operations = self._statistics_lowering.operations(var)
            for operation in operations:
                kernel_code_lines.append(f"    {safe_var}_{operation.spelling}_ptr,")
            added_inner = set()
            for operation in operations:
                if operation.inner is None:
                    continue
                inner = operation.inner.value
                if inner in added_inner or inner == "last":
                    continue
                kernel_code_lines.append(f"    {safe_var}_{inner}_inner_state_ptr,")
                if inner == "mean":
                    kernel_code_lines.append(f"    {safe_var}_{inner}_weight_state_ptr,")
                added_inner.add(inner)

        kernel_code_lines.extend([
            "    weight_ptr,",
            "    total_weight_ptr,",
            "    num_macro_steps_ptr,",
            "    sub_step_ptr,",
            "    num_sub_steps_ptr,",
            "    flags_ptr,",
            "    macro_step_index_ptr,",
            "    n_elements: tl.constexpr,",
            "    BLOCK_SIZE: tl.constexpr,",
            "):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_elements",
            "",
            "    weight = tl.load(weight_ptr)",
            "    total_weight = tl.load(total_weight_ptr)",
            "    num_macro_steps = tl.load(num_macro_steps_ptr)",
            "    sub_step = tl.load(sub_step_ptr).to(tl.int32)",
            "    num_sub_steps = tl.load(num_sub_steps_ptr).to(tl.int32)",
            "    flags = tl.load(flags_ptr).to(tl.int32)",
            "    macro_step_index = tl.load(macro_step_index_ptr).to(tl.int32)",
            "    is_inner_first = ((flags & 1) != 0) & (sub_step == 0)",
            "    is_inner_last = (((flags >> 1) & 1) != 0) & (sub_step == num_sub_steps - 1)",
            "    is_middle = sub_step == num_sub_steps // 2",
            "    is_outer_first = (((flags >> 2) & 1) != 0) & is_inner_last",
            "    is_outer_last = (((flags >> 3) & 1) != 0) & is_inner_last",
            "",
        ])

        indent = "    "
        indent2 = "        "
        for var in var_list:
            safe_var = self._get_safe_name(var)
            var_numel = int(self._tensor_registry[var].numel())
            kernel_code_lines.extend([
                f"{indent}# === full tensor variable: {var} ===",
                f"{indent}var_mask = offs < {var_numel}",
                f"{indent}{safe_var}_val = tl.load({safe_var}_ptr + offs, mask=var_mask, other=0.0)",
            ])

            for operation in self._statistics_lowering.operations(var):
                op = operation.spelling
                out_ptr = f"{safe_var}_{op}_ptr + offs"

                if operation.compound:
                    outer = operation.outer.value
                    inner = operation.inner.value
                    val_for = f"{safe_var}_{inner}_val"
                    if inner == "last":
                        kernel_code_lines.append(f"{indent}{val_for} = {safe_var}_val")
                    elif inner == "mean":
                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + offs"
                        weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + offs"
                        kernel_code_lines.extend([
                            f"{indent}inner_old = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                            f"{indent}weight_old = tl.load({weight_ptr}, mask=var_mask, other=0.0)",
                            f"{indent}inner_new = inner_old + {safe_var}_val * weight",
                            f"{indent}weight_new = weight_old + weight",
                            f"{indent}{val_for} = inner_new / weight_new",
                            f"{indent}if is_inner_last:",
                            f"{indent2}tl.store({inner_ptr}, 0.0, mask=var_mask)",
                            f"{indent2}tl.store({weight_ptr}, 0.0, mask=var_mask)",
                            f"{indent}else:",
                            f"{indent2}tl.store({inner_ptr}, inner_new, mask=var_mask)",
                            f"{indent2}tl.store({weight_ptr}, weight_new, mask=var_mask)",
                        ])
                    elif inner == "sum":
                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + offs"
                        kernel_code_lines.extend([
                            f"{indent}inner_old = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                            f"{indent}inner_new = inner_old + {safe_var}_val * weight",
                            f"{indent}{val_for} = inner_new",
                            f"{indent}if is_inner_last:",
                            f"{indent2}tl.store({inner_ptr}, 0.0, mask=var_mask)",
                            f"{indent}else:",
                            f"{indent2}tl.store({inner_ptr}, inner_new, mask=var_mask)",
                        ])
                    elif inner == "max":
                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + offs"
                        kernel_code_lines.extend([
                            f"{indent}inner_old = tl.load({inner_ptr}, mask=var_mask, other={safe_var}_val)",
                            f"{indent}inner_new = tl.where(is_inner_first, {safe_var}_val, tl.maximum(inner_old, {safe_var}_val))",
                            f"{indent}{val_for} = inner_new",
                            f"{indent}if is_inner_last:",
                            f"{indent2}tl.store({inner_ptr}, -float('inf'), mask=var_mask)",
                            f"{indent}else:",
                            f"{indent2}tl.store({inner_ptr}, inner_new, mask=var_mask)",
                        ])
                    elif inner == "min":
                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + offs"
                        kernel_code_lines.extend([
                            f"{indent}inner_old = tl.load({inner_ptr}, mask=var_mask, other={safe_var}_val)",
                            f"{indent}inner_new = tl.where(is_inner_first, {safe_var}_val, tl.minimum(inner_old, {safe_var}_val))",
                            f"{indent}{val_for} = inner_new",
                            f"{indent}if is_inner_last:",
                            f"{indent2}tl.store({inner_ptr}, float('inf'), mask=var_mask)",
                            f"{indent}else:",
                            f"{indent2}tl.store({inner_ptr}, inner_new, mask=var_mask)",
                        ])
                    elif inner in ("first", "mid"):
                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + offs"
                        cond = "is_inner_first" if inner == "first" else "is_middle"
                        kernel_code_lines.extend([
                            f"{indent}if {cond}:",
                            f"{indent2}tl.store({inner_ptr}, {safe_var}_val, mask=var_mask)",
                            f"{indent}{val_for} = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                        ])
                    else:
                        raise ValueError(f"Unsupported full-output inner op '{inner}'.")

                    kernel_code_lines.append(f"{indent}if is_inner_last:")
                    if outer == "max":
                        kernel_code_lines.extend([
                            f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other={val_for})",
                            f"{indent2}new = tl.where(is_outer_first, {val_for}, tl.maximum(old, {val_for}))",
                            f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                        ])
                    elif outer == "min":
                        kernel_code_lines.extend([
                            f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other={val_for})",
                            f"{indent2}new = tl.where(is_outer_first, {val_for}, tl.minimum(old, {val_for}))",
                            f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                        ])
                    elif outer == "sum":
                        kernel_code_lines.extend([
                            f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                            f"{indent2}new = tl.where(is_outer_first, {val_for}, old + {val_for})",
                            f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                        ])
                    elif outer == "mean":
                        kernel_code_lines.extend([
                            f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                            f"{indent2}accum = tl.where(is_outer_first, {val_for}, old + {val_for})",
                            f"{indent2}new = tl.where(is_outer_last, accum / num_macro_steps, accum)",
                            f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                        ])
                    elif outer == "last":
                        kernel_code_lines.append(f"{indent2}tl.store({out_ptr}, {val_for}, mask=var_mask)")
                    elif outer == "first":
                        kernel_code_lines.append(f"{indent2}tl.store({out_ptr}, {val_for}, mask=var_mask & is_outer_first)")
                    else:
                        raise ValueError(f"Unsupported full-output outer op '{outer}'.")
                    kernel_code_lines.append("")
                    continue

                if op == "mean":
                    kernel_code_lines.extend([
                        f"{indent}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                        f"{indent}accum = tl.where(is_inner_first, 0.0, old) + {safe_var}_val * weight",
                        f"{indent}new = tl.where(is_inner_last, accum / total_weight, accum)",
                        f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                    ])
                elif op == "sum":
                    kernel_code_lines.extend([
                        f"{indent}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                        f"{indent}new = tl.where(is_inner_first, 0.0, old) + {safe_var}_val * weight",
                        f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                    ])
                elif op == "max":
                    kernel_code_lines.extend([
                        f"{indent}old = tl.load({out_ptr}, mask=var_mask, other={safe_var}_val)",
                        f"{indent}new = tl.where(is_inner_first, {safe_var}_val, tl.maximum(old, {safe_var}_val))",
                        f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                    ])
                elif op == "min":
                    kernel_code_lines.extend([
                        f"{indent}old = tl.load({out_ptr}, mask=var_mask, other={safe_var}_val)",
                        f"{indent}new = tl.where(is_inner_first, {safe_var}_val, tl.minimum(old, {safe_var}_val))",
                        f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                    ])
                elif op == "last":
                    kernel_code_lines.append(f"{indent}tl.store({out_ptr}, {safe_var}_val, mask=var_mask & is_inner_last)")
                elif op == "first":
                    kernel_code_lines.append(f"{indent}tl.store({out_ptr}, {safe_var}_val, mask=var_mask & is_inner_first)")
                elif op == "mid":
                    kernel_code_lines.append(f"{indent}tl.store({out_ptr}, {safe_var}_val, mask=var_mask & is_middle)")
                else:
                    raise ValueError(f"Unsupported full-output op '{op}'.")
                kernel_code_lines.append("")
        kernel_code_lines.append("")



    def _generate_main_function(
        self: StatisticsRuntime, kernel_code_lines: List[str],
        grouped_by_output_index: Dict[str, List[str]],
    ) -> None:
        """Generate the main python function that calls kernels."""
        kernel_code_lines.extend([
            "# Main update function",
            "def internal_update_statistics(states, BLOCK_SIZE):",
        ])

        if self.num_trials > 1:
             kernel_code_lines.append(f"    num_trials = {self.num_trials}")
        else:
             kernel_code_lines.append("    num_trials = 1")

        scatters = self._statistics_ir.ordered_scatters()
        if scatters:
            kernel_code_lines.append(
                "    # Materialize all scatter virtuals in dependency order"
            )
        for variable in scatters:
            var = variable.name
            scatter = variable.source
            safe_var = self._get_safe_name(var)
            buf_key = f"__scatter_buf_{var}"
            is_mean = scatter.reduction.value == "mean"
            kernel_code_lines.append(
                f"    _N_{safe_var} = states['{buf_key}'].shape[-1]"
            )
            kernel_code_lines.append(
                f"    _M_{safe_var} = len(states['{scatter.index}'])"
            )
            zero_args = [f"states['{buf_key}']"]
            if is_mean:
                cnt_key = f"__scatter_cnt_{var}"
                zero_args.append(f"states['{cnt_key}']")
            zero_args.extend([f"_N_{safe_var}", "BLOCK_SIZE", "num_trials"])
            kernel_code_lines.append(
                f"    scatter_zero_{safe_var}["
                f"(triton.cdiv(_N_{safe_var}, BLOCK_SIZE),)]"
                f"({', '.join(zero_args)})"
            )
            add_args = [f"states['{buf_key}']"]
            if is_mean:
                add_args.append(f"states['{cnt_key}']")
            sorted_src = list(self._statistics_ir.scatter_inputs(var))
            for token in sorted_src:
                add_args.append(f"states['{token}']")
            add_args.extend([
                f"_M_{safe_var}", f"_N_{safe_var}", "BLOCK_SIZE", "num_trials",
            ])
            for token in sorted_src:
                tensor = self._tensor_registry.get(token)
                if tensor is None:
                    tensor = self._storage.get(token)
                if tensor is not None and self.num_trials > 1 and tensor.ndim >= 2:
                    add_args.append(str(tensor.shape[1]))
                else:
                    add_args.append("0")
            kernel_code_lines.append(
                f"    scatter_add_{safe_var}["
                f"(triton.cdiv(_M_{safe_var}, BLOCK_SIZE),)]"
                f"({', '.join(add_args)})"
            )
            if is_mean:
                div_args = [
                    f"states['{buf_key}']", f"states['{cnt_key}']",
                    f"_N_{safe_var}", "BLOCK_SIZE", "num_trials",
                ]
                kernel_code_lines.append(
                    f"    scatter_divide_{safe_var}["
                    f"(triton.cdiv(_N_{safe_var}, BLOCK_SIZE),)]"
                    f"({', '.join(div_args)})"
                )
        if scatters:
            kernel_code_lines.append("")

        for output_index, var_list in grouped_by_output_index.items():
            kernel_name = f"kernel_{output_index}"

            if output_index == "__full__":
                full_len = max(int(self._tensor_registry[var].numel()) for var in var_list)
                kernel_code_lines.extend([
                    "    # Launch full-output kernel",
                    f"    full_len = {full_len}",
                    "    grid___full__ = lambda meta: (triton.cdiv(full_len, meta['BLOCK_SIZE']),)",
                    f"    {kernel_name}[grid___full__](",
                ])
                for var in var_list:
                    safe_var = self._get_safe_name(var)
                    kernel_code_lines.append(f"        {safe_var}_ptr=states['{var}'],")
                    operations = self._statistics_lowering.operations(var)
                    for operation in operations:
                        op = operation.spelling
                        kernel_code_lines.append(f"        {safe_var}_{op}_ptr=states['{var}_{op}'],")
                    added_inner = set()
                    for operation in operations:
                        if operation.inner is None:
                            continue
                        inner = operation.inner.value
                        if inner in added_inner or inner == "last":
                            continue
                        kernel_code_lines.append(f"        {safe_var}_{inner}_inner_state_ptr=states['{var}_{inner}_inner_state'],")
                        if inner == "mean":
                            kernel_code_lines.append(f"        {safe_var}_{inner}_weight_state_ptr=states['{var}_{inner}_weight_state'],")
                        added_inner.add(inner)
                kernel_code_lines.extend([
                    "        weight_ptr=states['__weight'],",
                    "        total_weight_ptr=states['__total_weight'],",
                    "        num_macro_steps_ptr=states['__num_macro_steps'],",
                    "        sub_step_ptr=states['__sub_step'],",
                    "        num_sub_steps_ptr=states['__num_sub_steps'],",
                    "        flags_ptr=states['__flags'],",
                    "        macro_step_index_ptr=states['__macro_step_index'],",
                    "        n_elements=full_len,",
                    "        BLOCK_SIZE=BLOCK_SIZE,",
                    "    )",
                    "",
                ])
                continue

            # Get stride_input from metadata of first variable
            first_var = var_list[0]
            stride_input = 0
            for meta in self._metadata.values():
                if meta['original_variable'] == first_var:
                    stride_input = meta.get('stride_input', 0)
                    break

            kernel_code_lines.extend([
                f"    # Launch kernel for {output_index}",
                f"    output_index_len = len(states['{output_index}'])",
                f"    stride_input = {stride_input}",
            ])

            kernel_code_lines.extend([
                f"    grid_{output_index} = lambda meta: (triton.cdiv(output_index_len, meta['BLOCK_SIZE']),)",
                f"    {kernel_name}[grid_{output_index}](",
                f"        {output_index}_ptr=states['{output_index}'],",
            ])

            input_args = {
                name for var in var_list
                for name in self._statistics_ir.materialized_inputs(var)
            }

            # Add Input pointers
            sorted_inputs = sorted(list(input_args))
            for var in sorted_inputs:
                 safe_var = self._get_safe_name(var)
                 # Avoid duplicate argument if output_index matches input var
                 if safe_var == output_index:
                     continue
                 kernel_code_lines.append(f"        {safe_var}_ptr=states['{var}'],")

            # Add variable output pointers
            for var in var_list:
                safe_var = self._get_safe_name(var)
                added_aux_ptrs = set()  # Track aux pointers for explicit argmax/argmin
                operations = self._statistics_lowering.operations(var)
                for operation in operations:
                    op = operation.spelling
                    kernel_code_lines.append(f"        {safe_var}_{op}_ptr=states['{var}_{op}'],")

                    # For EXPLICIT argmax/argmin operations, add aux pointer
                    # NO automatic argmax/argmin generation for max/min operations
                    if operation.stores_index:
                        arg_type = operation.outer.value
                        arg_k_str = "" if operation.k == 1 else str(operation.k)
                        aux_name = f"{arg_type}{arg_k_str or ''}_aux"  # e.g., 'max_aux', 'max3_aux'
                        if aux_name not in added_aux_ptrs:
                            aux_storage_key = f"{var}_{arg_type}{arg_k_str if arg_k_str else ''}_aux"
                            kernel_code_lines.append(f"        {safe_var}_{aux_name}_ptr=states['{aux_storage_key}'],")
                            added_aux_ptrs.add(aux_name)

                # Inner state pointers (only for ops that need cross-step state)
                added_inner = set()
                for operation in operations:
                    if operation.inner is not None:
                        inner = operation.inner.value
                        if inner not in added_inner:
                             # 'last' inner op directly uses current value, no state needed
                             if inner != 'last':
                                 kernel_code_lines.append(f"        {safe_var}_{inner}_inner_state_ptr=states['{var}_{inner}_inner_state'],")
                             if inner == 'mean':
                                 kernel_code_lines.append(f"        {safe_var}_{inner}_weight_state_ptr=states['{var}_{inner}_weight_state'],")
                             added_inner.add(inner)

            kernel_code_lines.extend([
                "        weight_ptr=states['__weight'],",
                "        total_weight_ptr=states['__total_weight'],",
                "        num_macro_steps_ptr=states['__num_macro_steps'],",
                "        sub_step_ptr=states['__sub_step'],",
                "        num_sub_steps_ptr=states['__num_sub_steps'],",
                "        flags_ptr=states['__flags'],",
                "        macro_step_index_ptr=states['__macro_step_index'],",
                "        n_saved_points=output_index_len,",
            ])

            # Add second dimension if needed (use actual shape)
            _, dims_2d = self._statistics_lowering.split_indexed(var_list)

            if dims_2d:
                var_2d = dims_2d[0]
                actual_shape = (
                    self._statistics_lowering.by_name[var_2d].variable.actual_shape
                )
                n_levels = actual_shape[-1]
                kernel_code_lines.append(f"        n_levels={n_levels},")

            kernel_code_lines.extend([
                "        BLOCK_SIZE=BLOCK_SIZE,",
                "        num_trials=num_trials,",
                "        stride_input=stride_input,",
                "    )",
                "",
            ])

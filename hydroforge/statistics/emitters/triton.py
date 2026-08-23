# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from datetime import datetime
from math import prod
from typing import TYPE_CHECKING, Dict, List, Set

from hydroforge.statistics.ir import (
    ExpressionDialect, ExpressionSource, Reduction, ScatterSource, TensorSource,
    render_expression,
)
from hydroforge.statistics.emitters.common import StatisticsEmitter

if TYPE_CHECKING:
    from hydroforge.statistics.runtime import StatisticsRuntime


_FULL_OUTPUT_GROUP = "__full__"
_FULL_OUTPUT_KERNEL = "hydroforge_full_output_kernel"
_FULL_OUTPUT_GRID = "hydroforge_full_output_grid"


class TritonStatisticsEmitter(StatisticsEmitter):
    """Triton JIT kernel code generation for statistics aggregation."""

    def emit(self):
        self._generate_triton_aggregator_function()
        return self.result()

    def _triton_expression(
        self: StatisticsRuntime, name: str, expression, names: dict[str, str],
    ) -> str:
        dtype = self._statistics_layouts[name].dtype
        value_type = {
            "torch.float32": "float32",
            "torch.float64": "float64",
        }[str(dtype)]
        return render_expression(
            expression, ExpressionDialect.TRITON, names,
            value_type=value_type,
        )

    def _generate_triton_aggregator_function(self) -> None:
        groups = self._statistics_lowering.groups
        lines = self._generate_kernel_header()
        self._generate_scatter_kernels(lines)
        for output_index, variables in groups.items():
            if output_index == _FULL_OUTPUT_GROUP:
                self._generate_full_kernel_for_group(
                    lines, output_index, variables,
                )
            else:
                safe_output_index = self._get_safe_name(output_index)
                self._generate_kernel_for_group(
                    lines, f"kernel_{safe_output_index}", output_index,
                    variables,
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
            '- argmax/argmin store the macro-step index as int64;',
            '  conversion to datetime (if any) happens at the consumer via the',
            '  recorded macro-step time mapping, not at NC file write time',
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
            "@triton.jit",
            "def hydroforge_maximum(left, right):",
            "    # Ignore one-sided NaN without prescribing signed-zero bits.",
            "    return tl.where(left != left, right, tl.where(right != right, left, tl.maximum(left, right)))",
            "",
            "@triton.jit",
            "def hydroforge_minimum(left, right):",
            "    # Ignore one-sided NaN without prescribing signed-zero bits.",
            "    return tl.where(left != left, right, tl.where(right != right, left, tl.minimum(left, right)))",
            "",
            "@triton.jit",
            "def hydroforge_weighted_mean(old_value, old_weight, value, weight):",
            "    new_weight = old_weight + weight",
            "    return old_value * (old_weight / new_weight) + value * (weight / new_weight)",
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
            for variable in self._statistics_ir.ordered_scatters()
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
                    f"        tl.store({cnt_safe}_ptr + t * N + offs, 0, mask=mask)"
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
                    expression = self._triton_expression(
                        name, source.expression, names,
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
            value_expression = self._triton_expression(
                var_name, scatter.value, value_names,
            )
            kernel_code_lines.append(f"        _val = {value_expression}")
            kernel_code_lines.append(
                f"        tl.atomic_add({buf_safe}_ptr + t * N + idx, _val, mask=mask)"
            )
            if is_mean:
                kernel_code_lines.append(
                    f"        tl.atomic_add({cnt_safe}_ptr + t * N + idx, 1, mask=mask)"
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
                    f"        _cnt = tl.load({cnt_safe}_ptr + t * N + offs, mask=mask, other=1)",
                    f"        _val = tl.load({buf_safe}_ptr + t * N + offs, mask=mask, other=0.0)",
                    "        _mean = tl.where(_cnt > 0, _val / _cnt, float('nan'))",
                    f"        tl.store({buf_safe}_ptr + t * N + offs, _mean, mask=mask)",
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
            expression = self._triton_expression(
                var_name, source.expression, names,
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

        # Materialize dependencies used by unconditional operations first.
        def _collect_unconditional(name: str, acc: Set[str]) -> None:
            if name in acc:
                return
            acc.add(name)
            source = self._statistics_ir.sources.get(name, TensorSource(name))
            if isinstance(source, ExpressionSource):
                for dependency in source.expression.dependencies:
                    _collect_unconditional(dependency, acc)

        unconditional_names: Set[str] = set()
        for name in vars_need_val:
            _collect_unconditional(name, unconditional_names)

        vars_conditional_only = (
            set(dims_1d).difference(vars_need_val).difference(unconditional_names)
        )

        # Helper to emit variable value loads.  A value defined inside one
        # dynamic branch is not visible in a sibling branch in Triton SSA, so
        # memoization must be scoped rather than global.
        emitted_vars = set()
        emitted_by_scope: dict[str, set[str]] = {}
        unconditional_safe_names = {
            self._get_safe_name(name) for name in unconditional_names
        }

        def emit_val(v_name, to_lines, scope: str | None = None):
            safe_v_name = self._get_safe_name(v_name)
            scoped = (
                emitted_vars
                if scope is None else emitted_by_scope.setdefault(scope, set())
            )
            if (
                safe_v_name in emitted_vars
                or safe_v_name in scoped
                or (
                    scope is not None
                    and safe_v_name in unconditional_safe_names
                )
            ):
                return f"{safe_v_name}_val"

            source = self._statistics_ir.sources.get(v_name, TensorSource(v_name))
            if isinstance(source, TensorSource):
                # Real data (includes virtual source buffers)
                stride = self._source_stride(source.name)
                in_ptr_loc = f"{safe_v_name}_ptr + t * {stride} + idx"
                to_lines.append(f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
            elif isinstance(source, ScatterSource):
                buf_safe = self._get_safe_name(f"__scatter_buf_{v_name}")
                stride = self._source_stride(f"__scatter_buf_{v_name}")
                in_ptr_loc = f"{buf_safe}_ptr + t * {stride} + idx"
                to_lines.append(f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
            elif isinstance(source, ExpressionSource):
                names = {
                    dependency: emit_val(dependency, to_lines, scope)
                    for dependency in source.expression.dependencies
                }
                expression = self._triton_expression(
                    v_name, source.expression, names,
                )
                to_lines.append(f"{indent}{safe_v_name}_val = {expression}")

            scoped.add(safe_v_name)
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
                        aux_ptr_base = f"{safe_var}_{op}_aux_ptr"

                        if k_val == 1:
                            aux_ptr = f"{aux_ptr_base} + {out_offset}"
                            if self._statistics_layouts[
                                var
                            ].dtype.is_floating_point:
                                comparison = '>' if arg_type == 'max' else '<'
                                ops_is_inner_last_is_outer_first.extend([
                                    f"tl.store({out_ptr}, -1, mask=mask)",
                                    f"tl.store({aux_ptr}, float('nan'), mask=mask)",
                                ])
                                update = [
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other=float('nan'))",
                                    f"{safe_var}_{op}_valid = {val_var} == {val_var}",
                                    f"{safe_var}_{op}_cond = {safe_var}_{op}_valid & (({safe_var}_{op}_aux_old != {safe_var}_{op}_aux_old) | ({val_var} {comparison} {safe_var}_{op}_aux_old))",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ]
                                ops_is_inner_last_is_outer_first.extend(update)
                                ops_is_inner_last_not_is_outer_first.extend(update)
                            elif arg_type == 'max':
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
                            else:
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
                                f"tl.store({out_ptr}, hydroforge_maximum({safe_var}_{op}_old, {val_var}), mask=mask)",
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
                                f"tl.store({out_ptr}, hydroforge_minimum({safe_var}_{op}_old, {val_var}), mask=mask)",
                            ])
                        else:
                            # minK bubble insert without arg tracking
                            maxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': 'min'
                            })

                    elif outer == 'mean':
                        ops_is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)")
                        ops_is_inner_last_not_is_outer_first.extend([
                            f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other=0.0)",
                            f"{safe_var}_{op}_count = num_macro_steps.to({val_var}.dtype)",
                            f"tl.store({out_ptr}, hydroforge_weighted_mean({safe_var}_{op}_old, {safe_var}_{op}_count - 1.0, {val_var}, 1.0), mask=mask)",
                        ])

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
                        weight_ptr = (
                            f"{safe_var}_mean_sample_weight_state_ptr + "
                            f"{out_offset}"
                        )
                        ops_unconditional.extend([
                            f"# Standalone mean for {safe_var}",
                            f"{safe_var}_mean_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                            f"{safe_var}_mean_weight_old = tl.where(is_inner_first, 0.0, tl.load({weight_ptr}, mask=mask, other=0.0))",
                            f"{safe_var}_mean_weight_new = {safe_var}_mean_weight_old + weight",
                            f"{safe_var}_mean_out = hydroforge_weighted_mean({safe_var}_mean_old, {safe_var}_mean_weight_old, {safe_var}_val, weight)",
                            f"tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_mean_weight_new), mask=mask)",
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
                        f"tl.store({out_ptr}, hydroforge_maximum({safe_var}_max_old, {safe_var}_val), mask=mask)",
                    ])

                elif op == 'min':
                    ops_is_inner_first.extend([
                        f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                    ])
                    ops_not_is_inner_first.extend([
                        f"{safe_var}_min_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                        f"tl.store({out_ptr}, hydroforge_minimum({safe_var}_min_old, {safe_var}_val), mask=mask)",
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
                            emit_val(var, _tmp, "inner_last")
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
                            emit_val(var, _tmp, "inner_first")
                            ops_is_inner_first.extend(line.lstrip() for line in _tmp)
                            ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                    else:
                        ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")

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
                        f"{indent}{safe_var}_weight_{inner_type}_new = {safe_var}_weight_{inner_type}_old + weight",
                        f"{indent}{safe_var}_inner_{inner_type}_new = hydroforge_weighted_mean({safe_var}_inner_{inner_type}_old, {safe_var}_weight_{inner_type}_old, {var_val}, weight)",
                    ])
                    # Store based on condition - use tl.where for efficiency
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_weight_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
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
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first, {var_val}, hydroforge_maximum({safe_var}_inner_{inner_type}_old, {var_val}))",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, -float('inf'), {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'min':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first, {var_val}, hydroforge_minimum({safe_var}_inner_{inner_type}_old, {var_val}))",
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
                    deferred_lines = []
                    emit_val(var, deferred_lines, "inner_last")
                    # An expression can emit several dependency loads before
                    # its value.  Every emitted line belongs to this branch.
                    kernel_code_lines.extend(
                        f"{indent2}{line.lstrip()}" for line in deferred_lines
                    )

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

            # Group only operations that consume the exact same inner value.
            from collections import defaultdict
            grouped_by_var_k = defaultdict(lambda: {'max': None, 'min': None, 'argmax': None, 'argmin': None})

            for maxk_op in maxk_ops:
                key = (
                    maxk_op['var'], maxk_op['k'], maxk_op['out_offset'],
                    maxk_op['val_var'],
                )
                grouped_by_var_k[key][maxk_op['type']] = maxk_op

            if argmaxk_ops:
                for argk_op in argmaxk_ops:
                    key = (
                        argk_op['var'], argk_op['k'], argk_op['out_offset'],
                        argk_op['val_var'],
                    )
                    op_type = 'argmax' if 'max' in argk_op['type'] else 'argmin'
                    grouped_by_var_k[key][op_type] = argk_op

            # Process grouped operations with shared offset
            for (
                safe_var, k_val, out_offset, _value_expression,
            ), ops_dict in grouped_by_var_k.items():
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
                if has_argmax:
                    kernel_code_lines.append(
                        f"{indent2}new_idx_argmax_{safe_var} = "
                        "tl.full([BLOCK_SIZE], macro_step_index, dtype=tl.int64)"
                    )
                if has_argmin:
                    kernel_code_lines.append(
                        f"{indent2}new_idx_argmin_{safe_var} = "
                        "tl.full([BLOCK_SIZE], macro_step_index, dtype=tl.int64)"
                    )

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
                    argmax_aux_ptr = (
                        f"{safe_var}_{argmax_op['op']}_aux_ptr"
                    )
                    argmax_idx_ptr = f"{safe_var}_{argmax_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs, tl.where(new_val_argmax_{safe_var} == new_val_argmax_{safe_var}, new_idx_argmax_{safe_var}, -1), mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                if has_argmin:
                    argmin_op = ops_dict['argmin']
                    argmin_aux_ptr = (
                        f"{safe_var}_{argmin_op['op']}_aux_ptr"
                    )
                    argmin_idx_ptr = f"{safe_var}_{argmin_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs, tl.where(new_val_argmin_{safe_var} == new_val_argmin_{safe_var}, new_idx_argmin_{safe_var}, -1), mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")

                # Initialize remaining positions with inf/-inf
                kernel_code_lines.append(f"{indent3}for k in tl.static_range(1, {k_val}):")
                if has_max:
                    kernel_code_lines.append(f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, float('nan'), mask=mask)")
                if has_min:
                    kernel_code_lines.append(f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, float('nan'), mask=mask)")
                if has_argmax:
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, -1, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, float('nan'), mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, float('nan'), mask=mask)")
                if has_argmin:
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, -1, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, float('nan'), mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, float('nan'), mask=mask)")

                # else branch: bubble insert
                kernel_code_lines.append(f"{indent2}else:")
                kernel_code_lines.append(f"{indent3}for k in tl.static_range({k_val}):")

                # Load old values and compute swap masks
                if has_max:
                    kernel_code_lines.extend([
                        f"{indent4}old_max_k = tl.load({max_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('nan'))",
                        f"{indent4}swap_max = (new_val_max_{safe_var} == new_val_max_{safe_var}) & ((old_max_k != old_max_k) | (new_val_max_{safe_var} > old_max_k))",
                        f"{indent4}max_to_store = tl.where(swap_max, new_val_max_{safe_var}, old_max_k)",
                        f"{indent4}new_val_max_{safe_var} = tl.where(swap_max, old_max_k, new_val_max_{safe_var})",
                        f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, max_to_store, mask=mask & swap_max)",
                    ])
                if has_min:
                    kernel_code_lines.extend([
                        f"{indent4}old_min_k = tl.load({min_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('nan'))",
                        f"{indent4}swap_min = (new_val_min_{safe_var} == new_val_min_{safe_var}) & ((old_min_k != old_min_k) | (new_val_min_{safe_var} < old_min_k))",
                        f"{indent4}min_to_store = tl.where(swap_min, new_val_min_{safe_var}, old_min_k)",
                        f"{indent4}new_val_min_{safe_var} = tl.where(swap_min, old_min_k, new_val_min_{safe_var})",
                        f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, min_to_store, mask=mask & swap_min)",
                    ])
                if has_argmax:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmax_aux_k = tl.load({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('nan'))",
                        f"{indent4}old_argmax_idx_k = tl.load({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-1)",
                        f"{indent4}swap_argmax = (new_val_argmax_{safe_var} == new_val_argmax_{safe_var}) & ((old_argmax_aux_k != old_argmax_aux_k) | (new_val_argmax_{safe_var} > old_argmax_aux_k) | ((new_val_argmax_{safe_var} == old_argmax_aux_k) & (new_idx_argmax_{safe_var} < old_argmax_idx_k)))",
                        f"{indent4}argmax_aux_store = tl.where(swap_argmax, new_val_argmax_{safe_var}, old_argmax_aux_k)",
                        f"{indent4}argmax_idx_store = tl.where(swap_argmax, new_idx_argmax_{safe_var}, old_argmax_idx_k)",
                        f"{indent4}new_val_argmax_{safe_var} = tl.where(swap_argmax, old_argmax_aux_k, new_val_argmax_{safe_var})",
                        f"{indent4}new_idx_argmax_{safe_var} = tl.where(swap_argmax, old_argmax_idx_k, new_idx_argmax_{safe_var})",
                        f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)",
                        f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_idx_store, mask=mask & swap_argmax)",
                    ])
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)")
                if has_argmin:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmin_aux_k = tl.load({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('nan'))",
                        f"{indent4}old_argmin_idx_k = tl.load({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-1)",
                        f"{indent4}swap_argmin = (new_val_argmin_{safe_var} == new_val_argmin_{safe_var}) & ((old_argmin_aux_k != old_argmin_aux_k) | (new_val_argmin_{safe_var} < old_argmin_aux_k) | ((new_val_argmin_{safe_var} == old_argmin_aux_k) & (new_idx_argmin_{safe_var} < old_argmin_idx_k)))",
                        f"{indent4}argmin_aux_store = tl.where(swap_argmin, new_val_argmin_{safe_var}, old_argmin_aux_k)",
                        f"{indent4}argmin_idx_store = tl.where(swap_argmin, new_idx_argmin_{safe_var}, old_argmin_idx_k)",
                        f"{indent4}new_val_argmin_{safe_var} = tl.where(swap_argmin, old_argmin_aux_k, new_val_argmin_{safe_var})",
                        f"{indent4}new_idx_argmin_{safe_var} = tl.where(swap_argmin, old_argmin_idx_k, new_idx_argmin_{safe_var})",
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
        safe_output_index = self._get_safe_name(output_index)
        kernel_code_lines.extend([
            f"# Kernel for output_index: {output_index}",
            f"# Variables: {', '.join(var_list)}",
            f"# 1D: {', '.join(dims_1d) if dims_1d else 'None'}",
            f"# 2D: {', '.join(dims_2d) if dims_2d else 'None'}",
            "",
            "@triton.jit",
            f"def {kernel_name}(",
            f"    {safe_output_index}_ptr,",
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
            if var == output_index:
                continue
            kernel_code_lines.append(f"    {safe_var}_ptr,")

        for var in var_list:
            safe_var = self._get_safe_name(var)
            # Track which extra state pointers have been added to avoid duplicates
            variable = self._statistics_lowering.by_name[var]
            for operation in variable.operations:
                op = operation.spelling
                kernel_code_lines.append(f"    {safe_var}_{op}_ptr,")

                # For EXPLICIT argmax/argmin operators, add aux pointer for tracking values
                # NO automatic argmax/argmin generation for max/min operations
                if operation.stores_index:
                    kernel_code_lines.append(
                        f"    {safe_var}_{op}_aux_ptr,"
                    )
            if any(
                operation.inner is None
                and operation.outer.value == "mean"
                for operation in variable.operations
            ):
                kernel_code_lines.append(
                    f"    {safe_var}_mean_sample_weight_state_ptr,"
                )

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
            "    __hf_weight_ptr,",
            "    __hf_total_weight_ptr,",
            "    __hf_num_macro_steps_ptr,",
            "    __hf_sub_step_ptr,",
            "    __hf_num_sub_steps_ptr,",
            "    __hf_flags_ptr,",
            "    __hf_macro_step_index_ptr,",
            "    n_saved_points: tl.constexpr,",
        ])
        kernel_code_lines.extend([
            "    BLOCK_SIZE: tl.constexpr,",
            "    num_trials: tl.constexpr,",
            "):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_saved_points",
            "",
            "    # Load scalar parameters from device tensors",
            "    weight = tl.load(__hf_weight_ptr)",
            "    total_weight = tl.load(__hf_total_weight_ptr)",
            "    num_macro_steps = tl.load(__hf_num_macro_steps_ptr)",
            "    sub_step = tl.load(__hf_sub_step_ptr).to(tl.int32)",
            "    num_sub_steps = tl.load(__hf_num_sub_steps_ptr).to(tl.int32)",
            "    flags = tl.load(__hf_flags_ptr).to(tl.int32)",
            "    macro_step_index = tl.load(__hf_macro_step_index_ptr).to(tl.int64)",
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
            if 'is_outer_first' in needed_bools:
                kernel_code_lines.append("    is_outer_first = ((flags >> 2) & 1) != 0 and is_inner_last")
            if 'is_outer_last' in needed_bools:
                kernel_code_lines.append("    is_outer_last = ((flags >> 3) & 1) != 0 and is_inner_last")
            kernel_code_lines.append("")

        kernel_code_lines.extend([
            f"    idx = tl.load({safe_output_index}_ptr + offs, mask=mask)",
            "",
        ])

        # Loop over trials - use tl.static_range for compile-time unrolling
        kernel_code_lines.append("    for t in tl.static_range(num_trials):")
        indent = "        "
        indent2 = indent + "    "
        indent3 = indent2 + "    "
        indent4 = indent3 + "    "

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
                    and operations[0].outer.value == "last"
                )

            non_last_only = [v for v in dims_2d if not is_last_only(v)]
            last_only_vars = [v for v in dims_2d if is_last_only(v)]

            if non_last_only:
                for var in non_last_only:
                    safe_var = self._get_safe_name(var)
                    n_levels_var = self._statistics_layouts[var].actual_shape[-1]
                    kernel_code_lines.extend([
                        f"{indent}# 2D variable: {var}",
                        f"{indent}for level in tl.static_range({n_levels_var}):",
                    ])
                    emitted_vars_2d: set[str] = set()

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
                            expression = self._triton_expression(
                                v_name, source.expression, names,
                            )
                            kernel_code_lines.append(
                                f"{indent2}{safe_v_name}_val = {expression}"
                            )
                        else:
                            key = (
                                f"__scatter_buf_{v_name}"
                                if isinstance(source, ScatterSource)
                                else source.name
                            )
                            pointer = self._get_safe_name(key)
                            stride = self._source_stride(key)
                            in_ptr_loc = (
                                f"{pointer}_ptr + (t * {stride} + idx) * "
                                f"{n_levels_var} + level"
                            )
                            kernel_code_lines.append(
                                f"{indent2}{safe_v_name}_val = tl.load("
                                f"{in_ptr_loc}, mask=mask, other=0.0)"
                            )

                        emitted_vars_2d.add(safe_v_name)
                        return f"{safe_v_name}_val"

                    out_offset = (
                        f"(t * n_saved_points + offs) * {n_levels_var} + level"
                    )

                    val_name = emit_val_2d(var)
                    kernel_code_lines.append(f"{indent2}val = {val_name}")

                    for operation in self._statistics_lowering.operations(var):
                        op = operation.spelling
                        out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"
                        if operation.output is Reduction.MEAN:
                            weight_ptr = (
                                f"{safe_var}_mean_sample_weight_state_ptr + "
                                f"{out_offset}"
                            )
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent3}old_weight = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent3}old_weight = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new_weight = old_weight + weight",
                                f"{indent2}new = hydroforge_weighted_mean(old, old_weight, val, weight)",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                                f"{indent2}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, new_weight), mask=mask)",
                            ])
                        elif operation.output is Reduction.SUM:
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new = old + val * weight",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif operation.output is Reduction.MAX:
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = hydroforge_maximum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif operation.output is Reduction.MIN:
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = hydroforge_minimum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif operation.output is Reduction.LAST:
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_last:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif operation.output is Reduction.FIRST:
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                kernel_code_lines.append("")

            if last_only_vars:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (last-only)",
                    f"{indent}if is_inner_last:",
                ])
                for var in last_only_vars:
                    safe_var = self._get_safe_name(var)
                    n_levels_var = self._statistics_layouts[var].actual_shape[-1]
                    kernel_code_lines.append(
                        f"{indent2}for level in tl.static_range({n_levels_var}):"
                    )
                    emitted_last: set[str] = set()

                    def emit_last_value(v_name: str) -> str:
                        safe_name = self._get_safe_name(v_name)
                        if safe_name in emitted_last:
                            return f"{safe_name}_val"
                        source = self._statistics_ir.sources.get(
                            v_name, TensorSource(v_name),
                        )
                        if isinstance(source, ExpressionSource):
                            names = {
                                dependency: emit_last_value(dependency)
                                for dependency in source.expression.dependencies
                            }
                            expression = self._triton_expression(
                                v_name, source.expression, names,
                            )
                            kernel_code_lines.append(
                                f"{indent3}{safe_name}_val = {expression}"
                            )
                        else:
                            key = (
                                f"__scatter_buf_{v_name}"
                                if isinstance(source, ScatterSource)
                                else source.name
                            )
                            pointer = self._get_safe_name(key)
                            stride = self._source_stride(key)
                            offset = (
                                f"(t * {stride} + idx) * {n_levels_var} "
                                "+ level"
                            )
                            kernel_code_lines.append(
                                f"{indent3}{safe_name}_val = tl.load("
                                f"{pointer}_ptr + {offset}, mask=mask, "
                                "other=0.0)"
                            )
                        emitted_last.add(safe_name)
                        return f"{safe_name}_val"

                    out_offset = (
                        f"(t * n_saved_points + offs) * {n_levels_var} + level"
                    )
                    val_name = emit_last_value(var)
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
        kernel_code_lines.extend([
            f"# Full-output kernel: {output_index}",
            f"# Variables: {', '.join(var_list)}",
            "",
            "@triton.jit",
            f"def {_FULL_OUTPUT_KERNEL}(",
        ])

        sorted_inputs = sorted({
            name for var in var_list
            for name in self._statistics_ir.materialized_inputs(var)
        })
        for name in sorted_inputs:
            kernel_code_lines.append(
                f"    {self._get_safe_name(name)}_ptr,"
            )

        for var in var_list:
            safe_var = self._get_safe_name(var)
            operations = self._statistics_lowering.operations(var)
            for operation in operations:
                kernel_code_lines.append(f"    {safe_var}_{operation.spelling}_ptr,")
            if any(
                operation.inner is None
                and operation.outer.value == "mean"
                for operation in operations
            ):
                kernel_code_lines.append(
                    f"    {safe_var}_mean_sample_weight_state_ptr,"
                )
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
            "    __hf_weight_ptr,",
            "    __hf_total_weight_ptr,",
            "    __hf_num_macro_steps_ptr,",
            "    __hf_sub_step_ptr,",
            "    __hf_num_sub_steps_ptr,",
            "    __hf_flags_ptr,",
            "    __hf_macro_step_index_ptr,",
            "    n_elements: tl.constexpr,",
            "    BLOCK_SIZE: tl.constexpr,",
            "):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_elements",
            "",
            "    weight = tl.load(__hf_weight_ptr)",
            "    total_weight = tl.load(__hf_total_weight_ptr)",
            "    num_macro_steps = tl.load(__hf_num_macro_steps_ptr)",
            "    sub_step = tl.load(__hf_sub_step_ptr).to(tl.int32)",
            "    num_sub_steps = tl.load(__hf_num_sub_steps_ptr).to(tl.int32)",
            "    flags = tl.load(__hf_flags_ptr).to(tl.int32)",
            "    macro_step_index = tl.load(__hf_macro_step_index_ptr).to(tl.int64)",
            "    is_inner_first = ((flags & 1) != 0) & (sub_step == 0)",
            "    is_inner_last = (((flags >> 1) & 1) != 0) & (sub_step == num_sub_steps - 1)",
            "    is_outer_first = (((flags >> 2) & 1) != 0) & is_inner_last",
            "    is_outer_last = (((flags >> 3) & 1) != 0) & is_inner_last",
            "",
        ])

        indent = "    "
        indent2 = "        "
        for var in var_list:
            safe_var = self._get_safe_name(var)
            var_numel = prod(self._statistics_layouts[var].actual_shape)
            kernel_code_lines.extend([
                f"{indent}# === full tensor variable: {var} ===",
                f"{indent}var_mask = offs < {var_numel}",
            ])

            emitted: set[str] = set()

            def emit_full_value(name: str) -> str:
                safe_name = self._get_safe_name(name)
                if safe_name in emitted:
                    return f"{safe_name}_val"
                source = self._statistics_ir.sources.get(
                    name, TensorSource(name),
                )
                if isinstance(source, TensorSource):
                    kernel_code_lines.append(
                        f"{indent}{safe_name}_val = tl.load("
                        f"{safe_name}_ptr + offs, mask=var_mask, other=0.0)"
                    )
                elif isinstance(source, ScatterSource):
                    buffer_name = self._get_safe_name(
                        f"__scatter_buf_{name}"
                    )
                    kernel_code_lines.append(
                        f"{indent}{safe_name}_val = tl.load("
                        f"{buffer_name}_ptr + offs, mask=var_mask, other=0.0)"
                    )
                else:
                    names = {
                        dependency: emit_full_value(dependency)
                        for dependency in source.expression.dependencies
                    }
                    expression = self._triton_expression(
                        name, source.expression, names,
                    )
                    kernel_code_lines.append(
                        f"{indent}{safe_name}_val = {expression}"
                    )
                emitted.add(safe_name)
                return f"{safe_name}_val"

            emit_full_value(var)

            for reduction in self._statistics_lowering.inner_reductions(var):
                inner = reduction.value
                if inner == "last":
                    continue
                val_for = f"{safe_var}_{inner}_val"
                inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + offs"
                if inner == "mean":
                    weight_ptr = (
                        f"{safe_var}_{inner}_weight_state_ptr + offs"
                    )
                    kernel_code_lines.extend([
                        f"{indent}inner_{inner}_old = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                        f"{indent}weight_{inner}_old = tl.load({weight_ptr}, mask=var_mask, other=0.0)",
                        f"{indent}weight_{inner}_new = weight_{inner}_old + weight",
                        f"{indent}inner_{inner}_new = hydroforge_weighted_mean(inner_{inner}_old, weight_{inner}_old, {safe_var}_val, weight)",
                        f"{indent}{val_for} = inner_{inner}_new",
                        f"{indent}if is_inner_last:",
                        f"{indent2}tl.store({inner_ptr}, 0.0, mask=var_mask)",
                        f"{indent2}tl.store({weight_ptr}, 0.0, mask=var_mask)",
                        f"{indent}else:",
                        f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=var_mask)",
                        f"{indent2}tl.store({weight_ptr}, weight_{inner}_new, mask=var_mask)",
                    ])
                elif inner == "sum":
                    kernel_code_lines.extend([
                        f"{indent}inner_{inner}_old = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                        f"{indent}inner_{inner}_new = inner_{inner}_old + {safe_var}_val * weight",
                        f"{indent}{val_for} = inner_{inner}_new",
                        f"{indent}if is_inner_last:",
                        f"{indent2}tl.store({inner_ptr}, 0.0, mask=var_mask)",
                        f"{indent}else:",
                        f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=var_mask)",
                    ])
                elif inner in {"max", "min"}:
                    function = (
                        "hydroforge_maximum"
                        if inner == "max" else "hydroforge_minimum"
                    )
                    sentinel = "-float('inf')" if inner == "max" else "float('inf')"
                    kernel_code_lines.extend([
                        f"{indent}inner_{inner}_old = tl.load({inner_ptr}, mask=var_mask, other={safe_var}_val)",
                        f"{indent}inner_{inner}_new = tl.where(is_inner_first, {safe_var}_val, {function}(inner_{inner}_old, {safe_var}_val))",
                        f"{indent}{val_for} = inner_{inner}_new",
                        f"{indent}if is_inner_last:",
                        f"{indent2}tl.store({inner_ptr}, {sentinel}, mask=var_mask)",
                        f"{indent}else:",
                        f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=var_mask)",
                    ])
                elif inner == "first":
                    kernel_code_lines.extend([
                        f"{indent}if is_inner_first:",
                        f"{indent2}tl.store({inner_ptr}, {safe_var}_val, mask=var_mask)",
                        f"{indent}{val_for} = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                    ])

            for operation in self._statistics_lowering.operations(var):
                op = operation.spelling
                out_ptr = f"{safe_var}_{op}_ptr + offs"

                if operation.compound:
                    outer = operation.outer.value
                    inner = operation.inner.value
                    val_for = (
                        f"{safe_var}_val" if inner == "last"
                        else f"{safe_var}_{inner}_val"
                    )

                    kernel_code_lines.append(f"{indent}if is_inner_last:")
                    if outer == "max":
                        kernel_code_lines.extend([
                            f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other={val_for})",
                            f"{indent2}new = tl.where(is_outer_first, {val_for}, hydroforge_maximum(old, {val_for}))",
                            f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                        ])
                    elif outer == "min":
                        kernel_code_lines.extend([
                            f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other={val_for})",
                            f"{indent2}new = tl.where(is_outer_first, {val_for}, hydroforge_minimum(old, {val_for}))",
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
                            f"{indent2}count = num_macro_steps.to({val_for}.dtype)",
                            f"{indent2}new = tl.where(is_outer_first, {val_for}, hydroforge_weighted_mean(old, count - 1.0, {val_for}, 1.0))",
                            f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                        ])
                    elif outer == "last":
                        kernel_code_lines.append(f"{indent2}tl.store({out_ptr}, {val_for}, mask=var_mask)")
                    elif outer == "first":
                        kernel_code_lines.append(f"{indent2}tl.store({out_ptr}, {val_for}, mask=var_mask & is_outer_first)")
                    kernel_code_lines.append("")
                    continue

                if op == "mean":
                    weight_ptr = (
                        f"{safe_var}_mean_sample_weight_state_ptr + offs"
                    )
                    kernel_code_lines.extend([
                        f"{indent}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                        f"{indent}old_weight = tl.load({weight_ptr}, mask=var_mask, other=0.0)",
                        f"{indent}old = tl.where(is_inner_first, 0.0, old)",
                        f"{indent}old_weight = tl.where(is_inner_first, 0.0, old_weight)",
                        f"{indent}new_weight = old_weight + weight",
                        f"{indent}new = hydroforge_weighted_mean(old, old_weight, {safe_var}_val, weight)",
                        f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                        f"{indent}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, new_weight), mask=var_mask)",
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
                        f"{indent}new = tl.where(is_inner_first, {safe_var}_val, hydroforge_maximum(old, {safe_var}_val))",
                        f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                    ])
                elif op == "min":
                    kernel_code_lines.extend([
                        f"{indent}old = tl.load({out_ptr}, mask=var_mask, other={safe_var}_val)",
                        f"{indent}new = tl.where(is_inner_first, {safe_var}_val, hydroforge_minimum(old, {safe_var}_val))",
                        f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                    ])
                elif op == "last":
                    kernel_code_lines.append(f"{indent}tl.store({out_ptr}, {safe_var}_val, mask=var_mask & is_inner_last)")
                elif op == "first":
                    kernel_code_lines.append(f"{indent}tl.store({out_ptr}, {safe_var}_val, mask=var_mask & is_inner_first)")
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
            if self._storage[buf_key].shape[-1] > 0:
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
            if self._tensor_registry[scatter.index].numel() > 0:
                kernel_code_lines.append(
                    f"    scatter_add_{safe_var}["
                    f"(triton.cdiv(_M_{safe_var}, BLOCK_SIZE),)]"
                    f"({', '.join(add_args)})"
                )
            if is_mean and self._storage[buf_key].shape[-1] > 0:
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
            if output_index == _FULL_OUTPUT_GROUP:
                full_len = max(
                    prod(self._statistics_layouts[var].actual_shape)
                    for var in var_list
                )
                if full_len == 0:
                    kernel_code_lines.append(
                        "    # Skip empty full-output statistics group"
                    )
                    continue
                kernel_code_lines.extend([
                    "    # Launch full-output kernel",
                    f"    full_len = {full_len}",
                    f"    {_FULL_OUTPUT_GRID} = lambda meta: "
                    "(triton.cdiv(full_len, meta['BLOCK_SIZE']),)",
                    f"    {_FULL_OUTPUT_KERNEL}[{_FULL_OUTPUT_GRID}](",
                ])
                sorted_inputs = sorted({
                    name for var in var_list
                    for name in self._statistics_ir.materialized_inputs(var)
                })
                for name in sorted_inputs:
                    safe_name = self._get_safe_name(name)
                    kernel_code_lines.append(
                        f"        {safe_name}_ptr=states['{name}'],"
                    )
                for var in var_list:
                    safe_var = self._get_safe_name(var)
                    operations = self._statistics_lowering.operations(var)
                    for operation in operations:
                        op = operation.spelling
                        kernel_code_lines.append(f"        {safe_var}_{op}_ptr=states['{var}_{op}'],")
                    if any(
                        operation.inner is None
                        and operation.outer.value == "mean"
                        for operation in operations
                    ):
                        kernel_code_lines.append(
                            f"        {safe_var}_mean_sample_weight_state_ptr="
                            f"states['{var}_mean_sample_weight_state'],"
                        )
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
                    "        __hf_weight_ptr=states['__weight'],",
                    "        __hf_total_weight_ptr=states['__total_weight'],",
                    "        __hf_num_macro_steps_ptr=states['__num_macro_steps'],",
                    "        __hf_sub_step_ptr=states['__sub_step'],",
                    "        __hf_num_sub_steps_ptr=states['__num_sub_steps'],",
                    "        __hf_flags_ptr=states['__flags'],",
                    "        __hf_macro_step_index_ptr=states['__macro_step_index'],",
                    "        n_elements=full_len,",
                    "        BLOCK_SIZE=BLOCK_SIZE,",
                    "    )",
                    "",
                ])
                continue

            safe_output_index = self._get_safe_name(output_index)
            kernel_name = f"kernel_{safe_output_index}"
            if self._tensor_registry[output_index].numel() == 0:
                kernel_code_lines.append(
                    f"    # Skip empty statistics group {output_index}"
                )
                continue

            kernel_code_lines.extend([
                f"    # Launch kernel for {output_index}",
                f"    output_index_len = len(states['{output_index}'])",
            ])

            kernel_code_lines.extend([
                f"    grid_{safe_output_index} = lambda meta: (triton.cdiv(output_index_len, meta['BLOCK_SIZE']),)",
                f"    {kernel_name}[grid_{safe_output_index}](",
                f"        {safe_output_index}_ptr=states['{output_index}'],",
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
                 if var == output_index:
                     continue
                 kernel_code_lines.append(f"        {safe_var}_ptr=states['{var}'],")

            # Add variable output pointers
            for var in var_list:
                safe_var = self._get_safe_name(var)
                operations = self._statistics_lowering.operations(var)
                for operation in operations:
                    op = operation.spelling
                    kernel_code_lines.append(f"        {safe_var}_{op}_ptr=states['{var}_{op}'],")

                    # For EXPLICIT argmax/argmin operations, add aux pointer
                    # NO automatic argmax/argmin generation for max/min operations
                    if operation.stores_index:
                        aux_storage_key = f"{var}_{op}_aux"
                        kernel_code_lines.append(
                            f"        {safe_var}_{op}_aux_ptr="
                            f"states['{aux_storage_key}'],"
                        )

                if any(
                    operation.inner is None
                    and operation.outer.value == "mean"
                    for operation in operations
                ):
                    kernel_code_lines.append(
                        f"        {safe_var}_mean_sample_weight_state_ptr="
                        f"states['{var}_mean_sample_weight_state'],"
                    )

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
                "        __hf_weight_ptr=states['__weight'],",
                "        __hf_total_weight_ptr=states['__total_weight'],",
                "        __hf_num_macro_steps_ptr=states['__num_macro_steps'],",
                "        __hf_sub_step_ptr=states['__sub_step'],",
                "        __hf_num_sub_steps_ptr=states['__num_sub_steps'],",
                "        __hf_flags_ptr=states['__flags'],",
                "        __hf_macro_step_index_ptr=states['__macro_step_index'],",
                "        n_saved_points=output_index_len,",
            ])

            kernel_code_lines.extend([
                "        BLOCK_SIZE=BLOCK_SIZE,",
                "        num_trials=num_trials,",
                "    )",
                "",
            ])

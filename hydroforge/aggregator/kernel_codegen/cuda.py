# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator


class CudaCodegenMixin:
    """CUDA C++ kernel code generation for statistics aggregation."""

    # ========================================================================
    # CUDA C++ code generation
    # ========================================================================

    def _cuda_dtype_str(self: StatisticsAggregator, var_name: str) -> str:
        """Return the CUDA C++ scalar type string for a variable's tensor dtype."""
        import torch
        tensor = self._tensor_registry.get(var_name)
        if tensor is not None:
            dt = tensor.dtype
        else:
            # virtual or unknown → default float
            dt = torch.float32
        _map = {
            torch.float32: "float",
            torch.float64: "double",
            torch.int32: "int",
            torch.int64: "long long",
        }
        return _map.get(dt, "float")

    def _cuda_ptr_type(self: StatisticsAggregator, var_name: str) -> str:
        """Return the C++ pointer type for a variable (e.g. 'float*')."""
        return self._cuda_dtype_str(var_name) + "*"

    def _cuda_emit_val_load(self: StatisticsAggregator, var_name: str, 
                             lines: list, emitted: set, indent: str,
                             idx_expr: str = "idx") -> str:
        """Emit CUDA C++ code to load a variable value (handling virtuals).
        
        Returns the C++ variable name for the loaded value.
        """
        safe_var = self._get_safe_name(var_name)
        val_name = f"{safe_var}_val"
        if safe_var in emitted:
            return val_name

        info = self._field_registry.get(var_name)
        cat = getattr(info, 'json_schema_extra', {}).get('category', 'param')
        ctype = self._cuda_dtype_str(var_name)

        if cat == 'virtual':
            expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
            safe_expr = expr
            toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
            for t in toks:
                if t in self._field_registry or t in self._tensor_registry:
                    dep_val = self._cuda_emit_val_load(t, lines, emitted, indent, idx_expr)
                    safe_t = self._get_safe_name(t)
                    safe_expr = re.sub(r'\b' + t + r'\b', dep_val, safe_expr)
            # Transform pow to CUDA's powf/pow
            safe_expr = safe_expr.replace('**', ', ')
            if 'powf' not in safe_expr and ', ' in safe_expr:
                # rough transform: a, b → powf(a, b)
                safe_expr = f"powf({safe_expr})"
            lines.append(f'{indent}{ctype} {val_name} = ({ctype})({safe_expr});')
        else:
            lines.append(f'{indent}{ctype} {val_name} = p_{safe_var}[{idx_expr}];')

        emitted.add(safe_var)
        return val_name

    def _generate_cuda_kernel_for_group(self: StatisticsAggregator,
                                         cuda_lines: list, cpp_lines: list,
                                         save_idx: str, var_list: list,
                                         tensor_info: dict) -> None:
        """Generate CUDA __global__ kernel + C++ launcher for one save_idx group."""
        from collections import defaultdict
        num_trials = self.num_trials if self.num_trials > 1 else 1
        safe_save = self._get_safe_name(save_idx)
        kernel_name = f"aggr_kernel_{safe_save}"
        launcher_name = f"launch_aggr_{safe_save}"

        if num_trials > 1:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 3]
        else:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 1]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]

        # Collect all tensors needed
        input_vars = set()
        def _gather_inputs(name):
            info = self._field_registry.get(name)
            if getattr(info, 'json_schema_extra', {}).get('category') == 'virtual':
                expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
                for t in toks:
                    if t in self._field_registry or t in self._tensor_registry:
                        _gather_inputs(t)
            else:
                input_vars.add(name)
        for var in var_list:
            _gather_inputs(var)
        sorted_inputs = sorted(list(input_vars))

        # Collect output/state tensor names + types
        out_tensors = []  # (state_key, ctype, safe_param_name)
        for var in var_list:
            safe_var = self._get_safe_name(var)
            ctype = self._cuda_dtype_str(var)
            added_aux = set()
            for op in self._variable_ops[var]:
                out_key = f'{var}_{op}'
                op_parts = op.split('_')
                outer_op = op_parts[0]
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                if arg_match:
                    out_tensors.append((out_key, "int", f"p_{safe_var}_{op}"))
                    arg_type = arg_match.group(1)
                    arg_k_str = arg_match.group(2)
                    aux_name = f"{var}_{arg_type}{arg_k_str or ''}_aux"
                    safe_aux = f"p_{safe_var}_{arg_type}{arg_k_str or ''}_aux"
                    if aux_name not in added_aux:
                        out_tensors.append((aux_name, ctype, safe_aux))
                        added_aux.add(aux_name)
                else:
                    out_tensors.append((out_key, ctype, f"p_{safe_var}_{op}"))
                
                # Inner state for compound ops
                if '_' in op:
                    inner = op_parts[1] if len(op_parts) > 1 else ""
                    if inner and inner != 'last':
                        inner_key = f'{var}_{inner}_inner_state'
                        safe_inner = f"p_{safe_var}_{inner}_inner_state"
                        if not any(t[0] == inner_key for t in out_tensors):
                            out_tensors.append((inner_key, ctype, safe_inner))
                        if inner == 'mean':
                            w_key = f'{var}_{inner}_weight_state'
                            safe_w = f"p_{safe_var}_{inner}_weight_state"
                            if not any(t[0] == w_key for t in out_tensors):
                                out_tensors.append((w_key, ctype, safe_w))

        # Check if there are 2D variables
        has_2d = bool(dims_2d)
        n_levels_val = 0
        if has_2d:
            var_2d = dims_2d[0]
            actual_shape = tensor_info[var_2d]['actual_shape']
            n_levels_val = actual_shape[-1]

        # ---- Generate CUDA kernel ----
        I = "    "  # 4-space indent
        cuda_lines.append(f'__global__ void {kernel_name}(')
        # save_idx pointer
        cuda_lines.append(f'    const int* p_{safe_save},')
        # input var pointers
        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            if safe_inp == safe_save:
                continue
            ctype = self._cuda_dtype_str(inp)
            cuda_lines.append(f'    const {ctype}* p_{safe_inp},')
        # output/state pointers
        for _, ctype, pname in out_tensors:
            cuda_lines.append(f'    {ctype}* {pname},')
        # scalar params (device pointers — CUDA-Graph-safe)
        cuda_lines.append(f'    const float* p_weight_ptr, const float* p_total_weight_ptr, const float* p_num_macro_steps_ptr,')
        cuda_lines.append(f'    const int* p_sub_step_ptr, const int* p_num_sub_steps_ptr, const int* p_flags_ptr,')
        cuda_lines.append(f'    const int* p_macro_step_index_ptr,')
        cuda_lines.append(f'    int n_saved_points, int stride_input')
        if has_2d:
            cuda_lines[-1] += ', int n_levels'
        cuda_lines.append(') {')
        cuda_lines.append(f'{I}int tid = blockIdx.x * blockDim.x + threadIdx.x;')
        cuda_lines.append(f'{I}if (tid >= n_saved_points) return;')
        cuda_lines.append(f'{I}int idx = p_{safe_save}[tid];')
        cuda_lines.append(f'')
        # Dereference scalar device pointers
        cuda_lines.append(f'{I}float weight = *p_weight_ptr;')
        cuda_lines.append(f'{I}float total_weight = *p_total_weight_ptr;')
        cuda_lines.append(f'{I}float num_macro_steps = *p_num_macro_steps_ptr;')
        cuda_lines.append(f'{I}int sub_step = *p_sub_step_ptr;')
        cuda_lines.append(f'{I}int num_sub_steps = *p_num_sub_steps_ptr;')
        cuda_lines.append(f'{I}int flags = *p_flags_ptr;')
        cuda_lines.append(f'{I}int macro_step_index = *p_macro_step_index_ptr;')
        cuda_lines.append(f'')
        needed_bools = self._analyze_needed_booleans()
        if needed_bools:
            cuda_lines.append(f'{I}// Compute boolean flags from sub_step, num_sub_steps, flags')
            if 'is_inner_first' in needed_bools:
                cuda_lines.append(f'{I}bool is_inner_first = (flags & 1) && (sub_step == 0);')
            if 'is_inner_last' in needed_bools:
                cuda_lines.append(f'{I}bool is_inner_last = ((flags >> 1) & 1) && (sub_step == num_sub_steps - 1);')
            if 'is_middle' in needed_bools:
                cuda_lines.append(f'{I}bool is_middle = (sub_step == num_sub_steps / 2);')
            if 'is_outer_first' in needed_bools:
                cuda_lines.append(f'{I}bool is_outer_first = ((flags >> 2) & 1) && is_inner_last;')
            if 'is_outer_last' in needed_bools:
                cuda_lines.append(f'{I}bool is_outer_last = ((flags >> 3) & 1) && is_inner_last;')
        cuda_lines.append('')

        # For each trial
        if num_trials > 1:
            cuda_lines.append(f'{I}for (int t = 0; t < {num_trials}; t++) {{')
            I2 = I + "    "
        else:
            cuda_lines.append(f'{I}// Single trial')
            cuda_lines.append(f'{I}const int t = 0;')
            I2 = I

        # -- 1D variables --
        if dims_1d:
            cuda_lines.append(f'{I2}// === 1D variables ===')
            # Load values
            emitted = set()
            for var in dims_1d:
                self._cuda_emit_val_load(var, cuda_lines, emitted, I2, idx_expr="t * stride_input + idx" if num_trials > 1 else "idx")

            # Inner aggregation states for compound ops
            inner_agg_needed = defaultdict(set)
            for var in dims_1d:
                for op in self._variable_ops[var]:
                    if '_' in op:
                        inner = op.split('_')[1]
                        inner_agg_needed[inner].add(var)

            for inner_type, inner_vars in inner_agg_needed.items():
                for var in inner_vars:
                    safe_var = self._get_safe_name(var)
                    var_val = f"{safe_var}_val"
                    ctype = self._cuda_dtype_str(var)
                    val_for = f"val_for_{safe_var}_{inner_type}"
                    out_idx = f"t * n_saved_points + tid"

                    if inner_type == 'last':
                        pass  # val_for = var_val at is_inner_last, handled inline
                    elif inner_type == 'mean':
                        cuda_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}{{',
                            f'{I2}    {ctype} inner_old = p_{safe_var}_mean_inner_state[{out_idx}];',
                            f'{I2}    {ctype} w_old = p_{safe_var}_mean_weight_state[{out_idx}];',
                            f'{I2}    {ctype} inner_new = inner_old + {var_val} * ({ctype})weight;',
                            f'{I2}    {ctype} w_new = w_old + ({ctype})weight;',
                            f'{I2}    if (is_inner_last) {{',
                            f'{I2}        p_{safe_var}_mean_inner_state[{out_idx}] = ({ctype})0;',
                            f'{I2}        p_{safe_var}_mean_weight_state[{out_idx}] = ({ctype})0;',
                            f'{I2}        {val_for} = inner_new / w_new;',
                            f'{I2}    }} else {{',
                            f'{I2}        p_{safe_var}_mean_inner_state[{out_idx}] = inner_new;',
                            f'{I2}        p_{safe_var}_mean_weight_state[{out_idx}] = w_new;',
                            f'{I2}    }}',
                            f'{I2}}}',
                        ])
                    elif inner_type == 'sum':
                        cuda_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}{{',
                            f'{I2}    {ctype} inner_old = p_{safe_var}_sum_inner_state[{out_idx}];',
                            f'{I2}    {ctype} inner_new = inner_old + {var_val} * ({ctype})weight;',
                            f'{I2}    if (is_inner_last) {{',
                            f'{I2}        p_{safe_var}_sum_inner_state[{out_idx}] = ({ctype})0;',
                            f'{I2}        {val_for} = inner_new;',
                            f'{I2}    }} else {{',
                            f'{I2}        p_{safe_var}_sum_inner_state[{out_idx}] = inner_new;',
                            f'{I2}    }}',
                            f'{I2}}}',
                        ])
                    elif inner_type == 'max':
                        cuda_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}{{',
                            f'{I2}    {ctype} inner_old = p_{safe_var}_max_inner_state[{out_idx}];',
                            f'{I2}    {ctype} inner_new = (is_inner_first && macro_step_index == 0) ? {var_val} : fmaxf(inner_old, {var_val});',
                            f'{I2}    if (is_inner_last) {{',
                            f'{I2}        p_{safe_var}_max_inner_state[{out_idx}] = ({ctype})(-1e38);',
                            f'{I2}        {val_for} = inner_new;',
                            f'{I2}    }} else {{',
                            f'{I2}        p_{safe_var}_max_inner_state[{out_idx}] = inner_new;',
                            f'{I2}    }}',
                            f'{I2}}}',
                        ])
                    elif inner_type == 'min':
                        cuda_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}{{',
                            f'{I2}    {ctype} inner_old = p_{safe_var}_min_inner_state[{out_idx}];',
                            f'{I2}    {ctype} inner_new = (is_inner_first && macro_step_index == 0) ? {var_val} : fminf(inner_old, {var_val});',
                            f'{I2}    if (is_inner_last) {{',
                            f'{I2}        p_{safe_var}_min_inner_state[{out_idx}] = ({ctype})1e38;',
                            f'{I2}        {val_for} = inner_new;',
                            f'{I2}    }} else {{',
                            f'{I2}        p_{safe_var}_min_inner_state[{out_idx}] = inner_new;',
                            f'{I2}    }}',
                            f'{I2}}}',
                        ])
                    elif inner_type == 'first':
                        cuda_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}if (is_inner_first) p_{safe_var}_first_inner_state[{out_idx}] = {var_val};',
                            f'{I2}if (is_inner_last) {val_for} = p_{safe_var}_first_inner_state[{out_idx}];',
                        ])
                    elif inner_type == 'mid':
                        cuda_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}if (is_middle) p_{safe_var}_mid_inner_state[{out_idx}] = {var_val};',
                            f'{I2}if (is_inner_last) {val_for} = p_{safe_var}_mid_inner_state[{out_idx}];',
                        ])

            # Emit actual ops
            for var in dims_1d:
                safe_var = self._get_safe_name(var)
                var_val = f"{safe_var}_val"
                ctype = self._cuda_dtype_str(var)
                ops = self._variable_ops[var]
                out_idx = f"t * n_saved_points + tid"

                for op in ops:
                    out_key = f'{var}_{op}'
                    op_parts = op.split('_')

                    # ---- Compound ops ----
                    if len(op_parts) > 1:
                        outer = op_parts[0]
                        inner = op_parts[1]
                        is_arg = outer.startswith('arg')
                        match_k = re.match(r'(arg)?(max|min)(\d+)$', outer)
                        if match_k:
                            is_arg = match_k.group(1) is not None
                            outer_base = match_k.group(2)
                        else:
                            outer_base = outer.lstrip('arg')

                        if inner == 'last':
                            val_var = var_val
                        else:
                            val_var = f"val_for_{safe_var}_{inner}"

                        cuda_lines.append(f'{I2}// Compound {op} for {safe_var}')
                        if is_arg:
                            arg_type = outer_base
                            cmp_op = ">" if arg_type == "max" else "<"
                            arg_k_str = match_k.group(3) if match_k and match_k.group(3) != "1" else ""
                            aux_ptr = f"p_{safe_var}_{arg_type}{arg_k_str}_aux"
                            out_ptr = f"p_{safe_var}_{op}"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_last) {{',
                                f'{I2}    if (is_outer_first) {{',
                                f'{I2}        {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}        {aux_ptr}[{out_idx}] = {val_var};',
                                f'{I2}    }} else {{',
                                f'{I2}        {ctype} old_aux = {aux_ptr}[{out_idx}];',
                                f'{I2}        if ({val_var} {cmp_op} old_aux) {{',
                                f'{I2}            {aux_ptr}[{out_idx}] = {val_var};',
                                f'{I2}            {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}        }}',
                                f'{I2}    }}',
                                f'{I2}}}',
                            ])
                        elif outer_base in ('max', 'min'):
                            cmp_fn = "fmaxf" if outer_base == "max" else "fminf"
                            out_ptr = f"p_{safe_var}_{op}"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_last) {{',
                                f'{I2}    if (is_outer_first) {{',
                                f'{I2}        {out_ptr}[{out_idx}] = {val_var};',
                                f'{I2}    }} else {{',
                                f'{I2}        {out_ptr}[{out_idx}] = {cmp_fn}({out_ptr}[{out_idx}], {val_var});',
                                f'{I2}    }}',
                                f'{I2}}}',
                            ])
                        elif outer == 'mean':
                            out_ptr = f"p_{safe_var}_{op}"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_last) {{',
                                f'{I2}    if (is_outer_first) {{',
                                f'{I2}        {out_ptr}[{out_idx}] = {val_var};',
                                f'{I2}    }} else {{',
                                f'{I2}        {out_ptr}[{out_idx}] += {val_var};',
                                f'{I2}    }}',
                                f'{I2}    if (is_outer_last) {{',
                                f'{I2}        {out_ptr}[{out_idx}] /= num_macro_steps;',
                                f'{I2}    }}',
                                f'{I2}}}',
                            ])
                        elif outer == 'sum':
                            out_ptr = f"p_{safe_var}_{op}"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_last) {{',
                                f'{I2}    if (is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                                f'{I2}    else {{ {out_ptr}[{out_idx}] += {val_var}; }}',
                                f'{I2}}}',
                            ])
                        elif outer == 'last':
                            out_ptr = f"p_{safe_var}_{op}"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                            ])
                        elif outer == 'first':
                            out_ptr = f"p_{safe_var}_{op}"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_last && is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                            ])
                        continue

                    # ---- Simple ops ----
                    out_ptr = f"p_{safe_var}_{op}"
                    cuda_lines.append(f'{I2}// {op} for {safe_var}')

                    if op == 'mean':
                        cuda_lines.extend([
                            f'{I2}{{',
                            f'{I2}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[{out_idx}];',
                            f'{I2}    {ctype} new_val = old_val + {var_val} * ({ctype})weight;',
                            f'{I2}    {out_ptr}[{out_idx}] = is_inner_last ? new_val / ({ctype})total_weight : new_val;',
                            f'{I2}}}',
                        ])
                    elif op == 'sum':
                        cuda_lines.extend([
                            f'{I2}{{',
                            f'{I2}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[{out_idx}];',
                            f'{I2}    {out_ptr}[{out_idx}] = old_val + {var_val} * ({ctype})weight;',
                            f'{I2}}}',
                        ])
                    elif op == 'max':
                        has_argmax = 'argmax' in ops
                        if has_argmax:
                            aux_ptr = f"p_{safe_var}_max_aux"
                            argmax_ptr = f"p_{safe_var}_argmax"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_first) {{',
                                f'{I2}    {out_ptr}[{out_idx}] = {var_val};',
                                f'{I2}    {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}    {argmax_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}}} else {{',
                                f'{I2}    {ctype} old_v = {aux_ptr}[{out_idx}];',
                                f'{I2}    if ({var_val} > old_v) {{',
                                f'{I2}        {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}        {out_ptr}[{out_idx}] = {var_val};',
                                f'{I2}        {argmax_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}    }}',
                                f'{I2}}}',
                            ])
                        else:
                            cuda_lines.extend([
                                f'{I2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}',
                                f'{I2}else {{ {out_ptr}[{out_idx}] = fmaxf({out_ptr}[{out_idx}], {var_val}); }}',
                            ])
                    elif op == 'argmax':
                        if 'max' in ops:
                            pass  # handled by 'max' branch
                        else:
                            aux_ptr = f"p_{safe_var}_max_aux"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_first) {{',
                                f'{I2}    {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}    {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}}} else {{',
                                f'{I2}    {ctype} old_aux = {aux_ptr}[{out_idx}];',
                                f'{I2}    if ({var_val} > old_aux) {{',
                                f'{I2}        {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}        {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}    }}',
                                f'{I2}}}',
                            ])
                    elif op == 'min':
                        has_argmin = 'argmin' in ops
                        if has_argmin:
                            aux_ptr = f"p_{safe_var}_min_aux"
                            argmin_ptr = f"p_{safe_var}_argmin"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_first) {{',
                                f'{I2}    {out_ptr}[{out_idx}] = {var_val};',
                                f'{I2}    {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}    {argmin_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}}} else {{',
                                f'{I2}    {ctype} old_v = {aux_ptr}[{out_idx}];',
                                f'{I2}    if ({var_val} < old_v) {{',
                                f'{I2}        {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}        {out_ptr}[{out_idx}] = {var_val};',
                                f'{I2}        {argmin_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}    }}',
                                f'{I2}}}',
                            ])
                        else:
                            cuda_lines.extend([
                                f'{I2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}',
                                f'{I2}else {{ {out_ptr}[{out_idx}] = fminf({out_ptr}[{out_idx}], {var_val}); }}',
                            ])
                    elif op == 'argmin':
                        if 'min' in ops:
                            pass
                        else:
                            aux_ptr = f"p_{safe_var}_min_aux"
                            cuda_lines.extend([
                                f'{I2}if (is_inner_first) {{',
                                f'{I2}    {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}    {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}}} else {{',
                                f'{I2}    {ctype} old_aux = {aux_ptr}[{out_idx}];',
                                f'{I2}    if ({var_val} < old_aux) {{',
                                f'{I2}        {aux_ptr}[{out_idx}] = {var_val};',
                                f'{I2}        {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{I2}    }}',
                                f'{I2}}}',
                            ])
                    elif op == 'last':
                        cuda_lines.append(f'{I2}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {var_val}; }}')
                    elif op == 'first':
                        cuda_lines.append(f'{I2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}')
                    elif op == 'mid':
                        cuda_lines.append(f'{I2}if (is_middle) {{ {out_ptr}[{out_idx}] = {var_val}; }}')

        # -- 2D variables --
        if dims_2d:
            cuda_lines.append(f'{I2}// === 2D variables ===')
            for var in dims_2d:
                safe_var = self._get_safe_name(var)
                ctype = self._cuda_dtype_str(var)
                for op in self._variable_ops[var]:
                    out_key = f'{var}_{op}'
                    out_ptr = f"p_{safe_var}_{op}"
                    cuda_lines.append(f'{I2}for (int level = 0; level < n_levels; level++) {{')
                    idx_2d = f"(t * stride_input + idx) * n_levels + level" if num_trials > 1 else f"idx * n_levels + level"
                    out_2d = f"(t * n_saved_points + tid) * n_levels + level"
                    cuda_lines.append(f'{I2}    {ctype} val_2d = p_{safe_var}[{idx_2d}];')
                    if op == 'mean':
                        cuda_lines.extend([
                            f'{I2}    {ctype} old_2d = is_inner_first ? ({ctype})0 : {out_ptr}[{out_2d}];',
                            f'{I2}    {ctype} new_2d = old_2d + val_2d * ({ctype})weight;',
                            f'{I2}    {out_ptr}[{out_2d}] = is_inner_last ? new_2d / ({ctype})total_weight : new_2d;',
                        ])
                    elif op == 'max':
                        cuda_lines.extend([
                            f'{I2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}',
                            f'{I2}    else {{ {out_ptr}[{out_2d}] = fmaxf({out_ptr}[{out_2d}], val_2d); }}',
                        ])
                    elif op == 'min':
                        cuda_lines.extend([
                            f'{I2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}',
                            f'{I2}    else {{ {out_ptr}[{out_2d}] = fminf({out_ptr}[{out_2d}], val_2d); }}',
                        ])
                    elif op == 'last':
                        cuda_lines.append(f'{I2}    if (is_inner_last) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    elif op == 'first':
                        cuda_lines.append(f'{I2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    elif op == 'mid':
                        cuda_lines.append(f'{I2}    if (is_middle) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    cuda_lines.append(f'{I2}}}')

        if num_trials > 1:
            cuda_lines.append(f'{I}}}')  # end of for t loop
        cuda_lines.append('}')
        cuda_lines.append('')

        # ---- Generate C++ launcher (in cuda_sources, uses <<<>>> syntax) ----
        # Also build a forward declaration string for the cpp_sources binding file
        fwd_params = []
        cpp_lines.append(f'void {launcher_name}(')
        cpp_lines.append(f'    torch::Tensor t_{safe_save},')
        fwd_params.append('torch::Tensor')
        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            if safe_inp == safe_save:
                continue
            cpp_lines.append(f'    torch::Tensor t_{safe_inp},')
            fwd_params.append('torch::Tensor')
        for state_key, ctype, pname in out_tensors:
            safe_key = self._get_safe_name(state_key)
            cpp_lines.append(f'    torch::Tensor t_{safe_key},')
            fwd_params.append('torch::Tensor')
        cpp_lines.append(f'    torch::Tensor t_weight, torch::Tensor t_total_weight, torch::Tensor t_num_macro_steps,')
        fwd_params.extend(['torch::Tensor', 'torch::Tensor', 'torch::Tensor'])
        cpp_lines.append(f'    torch::Tensor t_sub_step, torch::Tensor t_num_sub_steps, torch::Tensor t_flags,')
        fwd_params.extend(['torch::Tensor', 'torch::Tensor', 'torch::Tensor'])
        cpp_lines.append(f'    torch::Tensor t_macro_step_index,')
        fwd_params.append('torch::Tensor')
        cpp_lines.append(f'    int n_saved_points, int stride_input')
        fwd_params.extend(['int', 'int'])
        if has_2d:
            cpp_lines[-1] += ', int n_levels'
            fwd_params.append('int')
        cpp_lines.append(') {')
        cpp_lines.append(f'    const int bs = 256;')
        cpp_lines.append(f'    const int grid = (n_saved_points + bs - 1) / bs;')
        cpp_lines.append(f'    const auto stream = at::cuda::getCurrentCUDAStream();')
        cpp_lines.append(f'    {kernel_name}<<<grid, bs, 0, stream>>>(')
        cpp_lines.append(f'        t_{safe_save}.data_ptr<int>(),')
        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            if safe_inp == safe_save:
                continue
            ctype = self._cuda_dtype_str(inp)
            cpp_lines.append(f'        t_{safe_inp}.data_ptr<{ctype}>(),')
        for state_key, ctype, pname in out_tensors:
            safe_key = self._get_safe_name(state_key)
            cpp_lines.append(f'        t_{safe_key}.data_ptr<{ctype}>(),')
        cpp_lines.append(f'        t_weight.data_ptr<float>(), t_total_weight.data_ptr<float>(), t_num_macro_steps.data_ptr<float>(),')
        cpp_lines.append(f'        t_sub_step.data_ptr<int>(), t_num_sub_steps.data_ptr<int>(), t_flags.data_ptr<int>(),')
        cpp_lines.append(f'        t_macro_step_index.data_ptr<int>(),')
        cpp_lines.append(f'        n_saved_points, stride_input')
        if has_2d:
            cpp_lines[-1] += ', n_levels'
        cpp_lines.append(f'    );')
        cpp_lines.append('}')
        cpp_lines.append('')

        # Store metadata for Python wrapper generation
        fwd_decl = f"void {launcher_name}({', '.join(fwd_params)});"
        return {
            'launcher_name': launcher_name,
            'save_idx': save_idx,
            'sorted_inputs': sorted_inputs,
            'out_tensors': out_tensors,
            'has_2d': has_2d,
            'n_levels_val': n_levels_val,
            'fwd_decl': fwd_decl,
        }

    def _generate_cuda_aggregator_function(self: StatisticsAggregator) -> None:
        """Generate and compile a CUDA C++ aggregation kernel (no Triton dependency).
        
        Generates raw CUDA __global__ kernels + C++ launchers + pybind11 bindings,
        compiles them via torch load_inline, and wraps in a Python function with the
        same interface as the Triton/PyTorch backends.
        """
        import hashlib

        from torch.utils.cpp_extension import load_inline

        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        tensor_info, grouped_by_save_idx = self._analyze_tensor_info()

        # CUDA kernels + launchers compiled by nvcc; pybind11 bindings by host g++
        cuda_lines = [
            '// Auto-generated CUDA aggregation kernels for hydroforge statistics',
            '#include <cuda_runtime.h>',
            '#include <cmath>',
            '#include <torch/extension.h>',
            '#include <c10/cuda/CUDAStream.h>',
            '',
        ]
        launcher_lines = []  # launchers appended after kernels (still in cuda_sources)
        fwd_decls = []  # forward declarations for cpp_sources (host g++)
        bind_lines = ['PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {']

        group_metas = []
        for save_idx, var_list in grouped_by_save_idx.items():
            meta = self._generate_cuda_kernel_for_group(
                cuda_lines, launcher_lines, save_idx, var_list, tensor_info
            )
            group_metas.append(meta)
            bind_lines.append(f'    m.def("{meta["launcher_name"]}", &{meta["launcher_name"]});')
            fwd_decls.append(meta['fwd_decl'])

        bind_lines.append('}')

        # cuda_sources: kernels + launchers
        cuda_lines.extend(launcher_lines)
        cuda_src = '\n'.join(cuda_lines)

        # cpp_sources: forward declarations + pybind11 bindings (compiled by host g++)
        cpp_lines = [
            '#include <torch/extension.h>',
            '',
        ]
        cpp_lines.extend(fwd_decls)
        cpp_lines.append('')
        cpp_lines.extend(bind_lines)
        cpp_src = '\n'.join(cpp_lines)

        # Compile
        unique = hashlib.md5((cuda_src + cpp_src).encode()).hexdigest()[:8]
        module_name = f"hydroforge_aggr_cuda_r{self.rank}_{unique}"

        compiled_mod = load_inline(
            name=module_name,
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=bool(
                __import__('os').environ.get("HYDROFORGE_CUDA_VERBOSE", "") == "1"
            ),
        )

        # Build the Python wrapper
        def _make_wrapper(mod, metas, grouped, ti, self_ref):
            """Create the internal_update_statistics function."""
            num_trials = self_ref.num_trials if self_ref.num_trials > 1 else 1

            # Pre-compute stride_input for each group
            strides = {}
            for meta in metas:
                si = meta['save_idx']
                var_list = grouped[si]
                first_var = var_list[0]
                stride = 0
                for out_name, m in self_ref._metadata.items():
                    if m['original_variable'] == first_var:
                        stride = m.get('stride_input', 0)
                        break
                strides[si] = stride

            def internal_update_statistics(states, BLOCK_SIZE):
                for meta in metas:
                    launcher = getattr(mod, meta['launcher_name'])
                    si = meta['save_idx']
                    n_saved = len(states[si])
                    stride = strides[si]
                    
                    args = [states[si]]
                    safe_save = self_ref._get_safe_name(si)
                    for inp in meta['sorted_inputs']:
                        safe_inp = self_ref._get_safe_name(inp)
                        if safe_inp == safe_save:
                            continue
                        args.append(states[inp])
                    for state_key, _, _ in meta['out_tensors']:
                        args.append(states[state_key])
                    args.extend([
                        states['__weight'], states['__total_weight'],
                        states['__num_macro_steps'],
                        states['__sub_step'], states['__num_sub_steps'],
                        states['__flags'],
                        states['__macro_step_index'],
                        n_saved, stride,
                    ])
                    if meta['has_2d']:
                        args.append(meta['n_levels_val'])
                    launcher(*args)

            return internal_update_statistics

        self._aggregator_function = _make_wrapper(
            compiled_mod, group_metas, grouped_by_save_idx, tensor_info, self
        )
        self._aggregator_generated = True

        # Save for debugging
        if self.save_kernels:
            combined = f"// === CUDA kernels + launchers ===\n{cuda_src}\n\n// === C++ bindings ===\n{cpp_src}"
            self._save_kernel_file(combined)

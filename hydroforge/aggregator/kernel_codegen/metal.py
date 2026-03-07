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


class MetalCodegenMixin:
    """Metal MSL kernel code generation for statistics aggregation."""

    # ========================================================================
    # Metal MSL code generation
    # ========================================================================

    def _metal_dtype_str(self: StatisticsAggregator, var_name: str) -> str:
        """Return the Metal Shading Language scalar type string for a variable."""
        import torch
        tensor = self._tensor_registry.get(var_name)
        if tensor is not None:
            dt = tensor.dtype
        else:
            dt = torch.float32
        _map = {
            torch.float32: "float",
            torch.int32: "int",
        }
        return _map.get(dt, "float")

    def _metal_emit_val_load(self: StatisticsAggregator, var_name: str,
                              lines: list, emitted: set, indent: str,
                              idx_expr: str = "idx") -> str:
        """Emit MSL code to load a variable value (handling virtuals).

        Returns the MSL variable name for the loaded value.
        """
        safe_var = self._get_safe_name(var_name)
        val_name = f"{safe_var}_val"
        if safe_var in emitted:
            return val_name

        info = self._field_registry.get(var_name)
        cat = getattr(info, 'json_schema_extra', {}).get('category', 'param')
        ctype = self._metal_dtype_str(var_name)

        if cat == 'virtual':
            expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
            safe_expr = expr
            toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
            for t in toks:
                if t in self._field_registry or t in self._tensor_registry:
                    dep_val = self._metal_emit_val_load(t, lines, emitted, indent, idx_expr)
                    safe_expr = re.sub(r'\b' + t + r'\b', dep_val, safe_expr)
            safe_expr = safe_expr.replace('**', ', ')
            if 'pow' not in safe_expr and ', ' in safe_expr:
                safe_expr = f"pow({safe_expr})"
            lines.append(f'{indent}{ctype} {val_name} = ({ctype})({safe_expr});')
        else:
            lines.append(f'{indent}{ctype} {val_name} = p_{safe_var}[{idx_expr}];')

        emitted.add(safe_var)
        return val_name

    def _generate_metal_kernel_for_group(self: StatisticsAggregator,
                                          msl_lines: list,
                                          save_idx: str, var_list: list,
                                          tensor_info: dict) -> dict:
        """Generate a Metal kernel function for one save_idx group.

        Returns metadata dict used by the Python wrapper.
        """
        from collections import defaultdict

        num_trials = self.num_trials if self.num_trials > 1 else 1
        safe_save = self._get_safe_name(save_idx)
        kernel_name = f"aggr_kernel_{safe_save}"

        if num_trials > 1:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 3]
        else:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 1]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]

        # Collect all input tensors needed (resolve virtuals)
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
            ctype = self._metal_dtype_str(var)
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

        has_2d = bool(dims_2d)
        n_levels_val = 0
        if has_2d:
            var_2d = dims_2d[0]
            actual_shape = tensor_info[var_2d]['actual_shape']
            n_levels_val = actual_shape[-1]

        # ---- Build MSL kernel signature with [[buffer(N)]] bindings ----
        buf_idx = 0
        arg_order = []  # track Python-side arg order: ('tensor', key) or ('scalar', name, msl_type)

        I = "    "
        msl_lines.append(f'kernel void {kernel_name}(')

        # save_idx buffer
        msl_lines.append(f'    device const int* p_{safe_save} [[buffer({buf_idx})]],')
        arg_order.append(('tensor', save_idx))
        buf_idx += 1

        # input var buffers
        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            if safe_inp == safe_save:
                continue
            ctype = self._metal_dtype_str(inp)
            msl_lines.append(f'    device const {ctype}* p_{safe_inp} [[buffer({buf_idx})]],')
            arg_order.append(('tensor', inp))
            buf_idx += 1

        # output/state buffers (read-write)
        for state_key, ctype, pname in out_tensors:
            msl_lines.append(f'    device {ctype}* {pname} [[buffer({buf_idx})]],')
            arg_order.append(('tensor', state_key))
            buf_idx += 1

        # varying scalar params → device buffer pointers (avoids host-device sync)
        varying_scalar_params = [
            ('weight', 'float', '__weight'), ('total_weight', 'float', '__total_weight'),
            ('num_macro_steps', 'float', '__num_macro_steps'),
            ('sub_step', 'int', '__sub_step'), ('num_sub_steps', 'int', '__num_sub_steps'),
            ('flags', 'int', '__flags'),
            ('macro_step_index', 'int', '__macro_step_index'),
        ]
        for sname, stype, state_key in varying_scalar_params:
            msl_lines.append(f'    device const {stype}* p_{sname}_ptr [[buffer({buf_idx})]],')
            arg_order.append(('tensor', state_key))
            buf_idx += 1

        # fixed scalar params (truly constant per-capture)
        fixed_scalar_params = [
            ('n_saved_points', 'int'), ('stride_input', 'int'),
        ]
        if has_2d:
            fixed_scalar_params.append(('n_levels', 'int'))

        for sname, stype in fixed_scalar_params:
            msl_lines.append(f'    constant {stype}& {sname} [[buffer({buf_idx})]],')
            arg_order.append(('scalar', sname, stype))
            buf_idx += 1

        # thread position — last param, no trailing comma
        msl_lines.append(f'    uint tid [[thread_position_in_grid]]')
        msl_lines.append(') {')

        # Bounds check
        msl_lines.append(f'{I}if ((int)tid >= n_saved_points) return;')
        msl_lines.append(f'{I}int idx = p_{safe_save}[tid];')
        msl_lines.append('')
        # Dereference varying scalar device pointers
        msl_lines.append(f'{I}float weight = *p_weight_ptr;')
        msl_lines.append(f'{I}float total_weight = *p_total_weight_ptr;')
        msl_lines.append(f'{I}float num_macro_steps = *p_num_macro_steps_ptr;')
        msl_lines.append(f'{I}int sub_step = *p_sub_step_ptr;')
        msl_lines.append(f'{I}int num_sub_steps = *p_num_sub_steps_ptr;')
        msl_lines.append(f'{I}int flags = *p_flags_ptr;')
        msl_lines.append(f'{I}int macro_step_index = *p_macro_step_index_ptr;')
        msl_lines.append('')
        needed_bools = self._analyze_needed_booleans()
        if needed_bools:
            msl_lines.append(f'{I}// Compute boolean flags from sub_step, num_sub_steps, flags')
            if 'is_inner_first' in needed_bools:
                msl_lines.append(f'{I}bool is_inner_first = (flags & 1) && (sub_step == 0);')
            if 'is_inner_last' in needed_bools:
                msl_lines.append(f'{I}bool is_inner_last = ((flags >> 1) & 1) && (sub_step == num_sub_steps - 1);')
            if 'is_middle' in needed_bools:
                msl_lines.append(f'{I}bool is_middle = (sub_step == num_sub_steps / 2);')
            if 'is_outer_first' in needed_bools:
                msl_lines.append(f'{I}bool is_outer_first = ((flags >> 2) & 1) && is_inner_last;')
            if 'is_outer_last' in needed_bools:
                msl_lines.append(f'{I}bool is_outer_last = ((flags >> 3) & 1) && is_inner_last;')
        msl_lines.append('')

        # Trial loop
        if num_trials > 1:
            msl_lines.append(f'{I}for (int t = 0; t < {num_trials}; t++) {{')
            I2 = I + "    "
        else:
            msl_lines.append(f'{I}const int t = 0;')
            I2 = I

        # -- 1D variables --
        if dims_1d:
            msl_lines.append(f'{I2}// === 1D variables ===')
            emitted = set()
            for var in dims_1d:
                self._metal_emit_val_load(var, msl_lines, emitted, I2,
                                          idx_expr="t * stride_input + idx" if num_trials > 1 else "idx")

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
                    ctype = self._metal_dtype_str(var)
                    val_for = f"val_for_{safe_var}_{inner_type}"
                    out_idx = f"t * n_saved_points + tid"

                    if inner_type == 'last':
                        pass
                    elif inner_type == 'mean':
                        msl_lines.extend([
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
                        msl_lines.extend([
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
                        msl_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}{{',
                            f'{I2}    {ctype} inner_old = p_{safe_var}_max_inner_state[{out_idx}];',
                            f'{I2}    {ctype} inner_new = (is_inner_first && macro_step_index == 0) ? {var_val} : fmax(inner_old, {var_val});',
                            f'{I2}    if (is_inner_last) {{',
                            f'{I2}        p_{safe_var}_max_inner_state[{out_idx}] = ({ctype})(-1e38);',
                            f'{I2}        {val_for} = inner_new;',
                            f'{I2}    }} else {{',
                            f'{I2}        p_{safe_var}_max_inner_state[{out_idx}] = inner_new;',
                            f'{I2}    }}',
                            f'{I2}}}',
                        ])
                    elif inner_type == 'min':
                        msl_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}{{',
                            f'{I2}    {ctype} inner_old = p_{safe_var}_min_inner_state[{out_idx}];',
                            f'{I2}    {ctype} inner_new = (is_inner_first && macro_step_index == 0) ? {var_val} : fmin(inner_old, {var_val});',
                            f'{I2}    if (is_inner_last) {{',
                            f'{I2}        p_{safe_var}_min_inner_state[{out_idx}] = ({ctype})1e38;',
                            f'{I2}        {val_for} = inner_new;',
                            f'{I2}    }} else {{',
                            f'{I2}        p_{safe_var}_min_inner_state[{out_idx}] = inner_new;',
                            f'{I2}    }}',
                            f'{I2}}}',
                        ])
                    elif inner_type == 'first':
                        msl_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}if (is_inner_first) p_{safe_var}_first_inner_state[{out_idx}] = {var_val};',
                            f'{I2}if (is_inner_last) {val_for} = p_{safe_var}_first_inner_state[{out_idx}];',
                        ])
                    elif inner_type == 'mid':
                        msl_lines.extend([
                            f'{I2}{ctype} {val_for} = ({ctype})0;',
                            f'{I2}if (is_middle) p_{safe_var}_mid_inner_state[{out_idx}] = {var_val};',
                            f'{I2}if (is_inner_last) {val_for} = p_{safe_var}_mid_inner_state[{out_idx}];',
                        ])

            # Emit actual ops
            for var in dims_1d:
                safe_var = self._get_safe_name(var)
                var_val = f"{safe_var}_val"
                ctype = self._metal_dtype_str(var)
                ops = self._variable_ops[var]
                out_idx = f"t * n_saved_points + tid"

                for op in ops:
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

                        msl_lines.append(f'{I2}// Compound {op} for {safe_var}')
                        if is_arg:
                            arg_type = outer_base
                            cmp_op = ">" if arg_type == "max" else "<"
                            arg_k_str = match_k.group(3) if match_k and match_k.group(3) != "1" else ""
                            aux_ptr = f"p_{safe_var}_{arg_type}{arg_k_str}_aux"
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
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
                            cmp_fn = "fmax" if outer_base == "max" else "fmin"
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
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
                            msl_lines.extend([
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
                            msl_lines.extend([
                                f'{I2}if (is_inner_last) {{',
                                f'{I2}    if (is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                                f'{I2}    else {{ {out_ptr}[{out_idx}] += {val_var}; }}',
                                f'{I2}}}',
                            ])
                        elif outer == 'last':
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{I2}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                            ])
                        elif outer == 'first':
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{I2}if (is_inner_last && is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                            ])
                        continue

                    # ---- Simple ops ----
                    out_ptr = f"p_{safe_var}_{op}"
                    msl_lines.append(f'{I2}// {op} for {safe_var}')

                    if op == 'mean':
                        msl_lines.extend([
                            f'{I2}{{',
                            f'{I2}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[{out_idx}];',
                            f'{I2}    {ctype} new_val = old_val + {var_val} * ({ctype})weight;',
                            f'{I2}    {out_ptr}[{out_idx}] = is_inner_last ? new_val / ({ctype})total_weight : new_val;',
                            f'{I2}}}',
                        ])
                    elif op == 'sum':
                        msl_lines.extend([
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
                            msl_lines.extend([
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
                            msl_lines.extend([
                                f'{I2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}',
                                f'{I2}else {{ {out_ptr}[{out_idx}] = fmax({out_ptr}[{out_idx}], {var_val}); }}',
                            ])
                    elif op == 'argmax':
                        if 'max' in ops:
                            pass
                        else:
                            aux_ptr = f"p_{safe_var}_max_aux"
                            msl_lines.extend([
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
                            msl_lines.extend([
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
                            msl_lines.extend([
                                f'{I2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}',
                                f'{I2}else {{ {out_ptr}[{out_idx}] = fmin({out_ptr}[{out_idx}], {var_val}); }}',
                            ])
                    elif op == 'argmin':
                        if 'min' in ops:
                            pass
                        else:
                            aux_ptr = f"p_{safe_var}_min_aux"
                            msl_lines.extend([
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
                        msl_lines.append(f'{I2}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {var_val}; }}')
                    elif op == 'first':
                        msl_lines.append(f'{I2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}')
                    elif op == 'mid':
                        msl_lines.append(f'{I2}if (is_middle) {{ {out_ptr}[{out_idx}] = {var_val}; }}')

        # -- 2D variables --
        if dims_2d:
            msl_lines.append(f'{I2}// === 2D variables ===')
            for var in dims_2d:
                safe_var = self._get_safe_name(var)
                ctype = self._metal_dtype_str(var)
                for op in self._variable_ops[var]:
                    out_ptr = f"p_{safe_var}_{op}"
                    msl_lines.append(f'{I2}for (int level = 0; level < n_levels; level++) {{')
                    idx_2d = f"(t * stride_input + idx) * n_levels + level" if num_trials > 1 else f"idx * n_levels + level"
                    out_2d = f"(t * n_saved_points + tid) * n_levels + level"
                    msl_lines.append(f'{I2}    {ctype} val_2d = p_{safe_var}[{idx_2d}];')
                    if op == 'mean':
                        msl_lines.extend([
                            f'{I2}    {ctype} old_2d = is_inner_first ? ({ctype})0 : {out_ptr}[{out_2d}];',
                            f'{I2}    {ctype} new_2d = old_2d + val_2d * ({ctype})weight;',
                            f'{I2}    {out_ptr}[{out_2d}] = is_inner_last ? new_2d / ({ctype})total_weight : new_2d;',
                        ])
                    elif op == 'max':
                        msl_lines.extend([
                            f'{I2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}',
                            f'{I2}    else {{ {out_ptr}[{out_2d}] = fmax({out_ptr}[{out_2d}], val_2d); }}',
                        ])
                    elif op == 'min':
                        msl_lines.extend([
                            f'{I2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}',
                            f'{I2}    else {{ {out_ptr}[{out_2d}] = fmin({out_ptr}[{out_2d}], val_2d); }}',
                        ])
                    elif op == 'last':
                        msl_lines.append(f'{I2}    if (is_inner_last) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    elif op == 'first':
                        msl_lines.append(f'{I2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    elif op == 'mid':
                        msl_lines.append(f'{I2}    if (is_middle) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    msl_lines.append(f'{I2}}}')

        if num_trials > 1:
            msl_lines.append(f'{I}}}')
        msl_lines.append('}')
        msl_lines.append('')

        return {
            'kernel_name': kernel_name,
            'save_idx': save_idx,
            'sorted_inputs': sorted_inputs,
            'out_tensors': out_tensors,
            'has_2d': has_2d,
            'n_levels_val': n_levels_val,
            'arg_order': arg_order,
        }

    def _generate_metal_aggregator_function(self: StatisticsAggregator) -> None:
        """Generate and compile a Metal MSL aggregation kernel.

        Generates raw MSL ``kernel`` functions, compiles them via
        ``torch.mps.compile_shader()``, and wraps in a Python function with
        the same interface as the Triton/PyTorch/CUDA backends.
        """

        import torch

        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        tensor_info, grouped_by_save_idx = self._analyze_tensor_info()

        msl_lines = [
            '// Auto-generated Metal aggregation kernels for hydroforge statistics',
            '#include <metal_stdlib>',
            'using namespace metal;',
            '',
        ]

        group_metas = []
        for save_idx, var_list in grouped_by_save_idx.items():
            meta = self._generate_metal_kernel_for_group(
                msl_lines, save_idx, var_list, tensor_info
            )
            group_metas.append(meta)

        msl_src = '\n'.join(msl_lines)

        # Compile
        compiled_lib = torch.mps.compile_shader(msl_src)

        # Build the Python wrapper
        def _make_wrapper(lib, metas, grouped, ti, self_ref):
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
                    kernel_fn = getattr(lib, meta['kernel_name'])
                    si = meta['save_idx']
                    n_saved = len(states[si])
                    stride = strides[si]

                    args = []

                    for kind, *rest in meta['arg_order']:
                        if kind == 'tensor':
                            key = rest[0]
                            args.append(states[key])
                        else:  # scalar (fixed constants only)
                            sname = rest[0]
                            if sname == 'n_saved_points':
                                args.append(n_saved)
                            elif sname == 'stride_input':
                                args.append(stride)
                            elif sname == 'n_levels':
                                args.append(meta['n_levels_val'])

                    kernel_fn(*args, threads=n_saved, group_size=256)

            return internal_update_statistics

        self._aggregator_function = _make_wrapper(
            compiled_lib, group_metas, grouped_by_save_idx, tensor_info, self
        )
        self._aggregator_generated = True

        # Save for debugging
        if self.save_kernels:
            self._save_kernel_file(msl_src)

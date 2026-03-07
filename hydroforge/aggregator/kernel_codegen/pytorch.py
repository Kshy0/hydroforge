# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator


class PyTorchCodegenMixin:
    """PyTorch (torch.compile) code generation for statistics aggregation."""

    # ========================================================================
    # PyTorch code generation (no Triton dependency)
    # ========================================================================

    # PyTorch code generation (no Triton dependency)
    # ========================================================================

    def _generate_pytorch_header(self: StatisticsAggregator) -> List[str]:
        """Generate header for PyTorch-based aggregation code."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        var_list = sorted(list(self._variables))
        return [
            '"""',
            f'Auto-generated PyTorch aggregation functions for hydroforge statistics.',
            f'Generated at: {timestamp}',
            f'Rank: {self.rank}',
            f'Variables: {", ".join(var_list)}',
            f'Device: {self.device}',
            '',
            'This module uses pure PyTorch operations (no Triton dependency).',
            '"""',
            '',
            'import torch',
            '',
        ]

    def _pytorch_emit_val_load(self: StatisticsAggregator, var_name: str,
                                lines: List[str], emitted: set,
                                indent: str, is_2d: bool = False) -> str:
        """Emit PyTorch code to load a variable value (handling virtuals recursively).
        
        Returns the expression name for the loaded value.
        """
        safe_var = self._get_safe_name(var_name)
        val_name = f"{safe_var}_val"
        if safe_var in emitted:
            return val_name

        info = self._field_registry.get(var_name)
        cat = getattr(info, 'json_schema_extra', {}).get('category', 'param')

        if cat == 'virtual':
            expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
            safe_expr = expr
            toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
            for t in toks:
                if t in self._field_registry or t in self._tensor_registry:
                    dep_val = self._pytorch_emit_val_load(t, lines, emitted, indent, is_2d)
                    safe_t = self._get_safe_name(t)
                    safe_expr = re.sub(r'\b' + t + r'\b', dep_val, safe_expr)
            safe_expr = self._transform_pow_expr(safe_expr)
            lines.append(f'{indent}{val_name} = {safe_expr}')
        else:
            if is_2d:
                lines.append(f'{indent}{val_name} = states["{var_name}"][(t * stride_input + idx) * n_levels + level]')
            else:
                lines.append(f'{indent}{val_name} = states["{var_name}"][t * stride_input + idx]')

        emitted.add(safe_var)
        return val_name

    def _generate_pytorch_group_function(self: StatisticsAggregator, lines: List[str],
                                          save_idx: str, var_list: List[str],
                                          tensor_info: Dict[str, Any]) -> None:
        """Generate a PyTorch function for one save_idx group."""
        num_trials = self.num_trials if self.num_trials > 1 else 1

        if num_trials > 1:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 3]
        else:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 1]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]

        func_name = f"_update_{save_idx}"
        lines.extend([
            f'def {func_name}(states, weight, total_weight, num_macro_steps,',
            f'               sub_step, num_sub_steps, flags,',
            f'               macro_step_index, num_trials, stride_input):',
        ])
        needed_bools = self._analyze_needed_booleans()
        if needed_bools:
            lines.append(f'    # Compute boolean flags from sub_step, num_sub_steps, flags')
            if 'is_inner_first' in needed_bools:
                lines.append(f'    is_inner_first = (flags & 1) != 0 and sub_step == 0')
            if 'is_inner_last' in needed_bools:
                lines.append(f'    is_inner_last = ((flags >> 1) & 1) != 0 and sub_step == num_sub_steps - 1')
            if 'is_middle' in needed_bools:
                lines.append(f'    is_middle = sub_step == num_sub_steps // 2')
            if 'is_outer_first' in needed_bools:
                lines.append(f'    is_outer_first = ((flags >> 2) & 1) != 0 and is_inner_last')
            if 'is_outer_last' in needed_bools:
                lines.append(f'    is_outer_last = ((flags >> 3) & 1) != 0 and is_inner_last')
        lines.extend([
            f'    idx = states["{save_idx}"]',
            f'    n = len(idx)',
            '',
        ])

        indent = '        '  # inside for t loop
        indent2 = indent + '    '

        lines.append(f'    for t in range(num_trials):')

        # ---------- 1D variables ----------
        if dims_1d:
            lines.append(f'{indent}# === 1D variables ===')
            emitted = set()

            # Pre-load all needed values
            for var in dims_1d:
                self._pytorch_emit_val_load(var, lines, emitted, indent, is_2d=False)

            # Inner aggregation states (for compound ops)
            from collections import defaultdict
            inner_agg_needed = defaultdict(set)
            for var in dims_1d:
                for op in self._variable_ops[var]:
                    if '_' in op:
                        inner = op.split('_')[1]
                        inner_agg_needed[inner].add(var)

            # Emit inner aggregation state updates
            for inner_type, inner_vars in inner_agg_needed.items():
                for var in inner_vars:
                    safe_var = self._get_safe_name(var)
                    var_val = f"{safe_var}_val"
                    val_for = f"val_for_{safe_var}_{inner_type}"
                    sl = 'slice(t * n, (t + 1) * n)'

                    if inner_type == 'last':
                        # val_for_X_last == X_val at is_inner_last, no state needed
                        pass
                    elif inner_type == 'mean':
                        inner_key = f'{var}_{inner_type}_inner_state'
                        weight_key = f'{var}_{inner_type}_weight_state'
                        lines.extend([
                            f'{indent}_isl = {sl}',
                            f'{indent}_inner_old = states["{inner_key}"][_isl].clone()',
                            f'{indent}_w_old = states["{weight_key}"][_isl].clone()',
                            f'{indent}_inner_new = _inner_old + {var_val} * weight',
                            f'{indent}_w_new = _w_old + weight',
                            f'{indent}if is_inner_last:',
                            f'{indent2}states["{inner_key}"][_isl] = 0.0',
                            f'{indent2}states["{weight_key}"][_isl] = 0.0',
                            f'{indent2}{val_for} = _inner_new / _w_new',
                            f'{indent}else:',
                            f'{indent2}states["{inner_key}"][_isl] = _inner_new',
                            f'{indent2}states["{weight_key}"][_isl] = _w_new',
                            f'{indent2}{val_for} = torch.zeros_like({var_val})',
                        ])
                    elif inner_type == 'sum':
                        inner_key = f'{var}_{inner_type}_inner_state'
                        lines.extend([
                            f'{indent}_isl = {sl}',
                            f'{indent}_inner_old = states["{inner_key}"][_isl].clone()',
                            f'{indent}_inner_new = _inner_old + {var_val} * weight',
                            f'{indent}if is_inner_last:',
                            f'{indent2}states["{inner_key}"][_isl] = 0.0',
                            f'{indent2}{val_for} = _inner_new',
                            f'{indent}else:',
                            f'{indent2}states["{inner_key}"][_isl] = _inner_new',
                            f'{indent2}{val_for} = torch.zeros_like({var_val})',
                        ])
                    elif inner_type == 'max':
                        inner_key = f'{var}_{inner_type}_inner_state'
                        lines.extend([
                            f'{indent}_isl = {sl}',
                            f'{indent}_inner_old = states["{inner_key}"][_isl].clone()',
                            f'{indent}if is_inner_first and macro_step_index == 0:',
                            f'{indent2}_inner_new = {var_val}',
                            f'{indent}else:',
                            f'{indent2}_inner_new = torch.maximum(_inner_old, {var_val})',
                            f'{indent}if is_inner_last:',
                            f'{indent2}states["{inner_key}"][_isl] = float("-inf")',
                            f'{indent2}{val_for} = _inner_new',
                            f'{indent}else:',
                            f'{indent2}states["{inner_key}"][_isl] = _inner_new',
                            f'{indent2}{val_for} = torch.zeros_like({var_val})',
                        ])
                    elif inner_type == 'min':
                        inner_key = f'{var}_{inner_type}_inner_state'
                        lines.extend([
                            f'{indent}_isl = {sl}',
                            f'{indent}_inner_old = states["{inner_key}"][_isl].clone()',
                            f'{indent}if is_inner_first and macro_step_index == 0:',
                            f'{indent2}_inner_new = {var_val}',
                            f'{indent}else:',
                            f'{indent2}_inner_new = torch.minimum(_inner_old, {var_val})',
                            f'{indent}if is_inner_last:',
                            f'{indent2}states["{inner_key}"][_isl] = float("inf")',
                            f'{indent2}{val_for} = _inner_new',
                            f'{indent}else:',
                            f'{indent2}states["{inner_key}"][_isl] = _inner_new',
                            f'{indent2}{val_for} = torch.zeros_like({var_val})',
                        ])
                    elif inner_type == 'first':
                        inner_key = f'{var}_{inner_type}_inner_state'
                        lines.extend([
                            f'{indent}_isl = {sl}',
                            f'{indent}if is_inner_first:',
                            f'{indent2}states["{inner_key}"][_isl] = {var_val}',
                            f'{indent}if is_inner_last:',
                            f'{indent2}{val_for} = states["{inner_key}"][_isl].clone()',
                            f'{indent}else:',
                            f'{indent2}{val_for} = torch.zeros_like({var_val})',
                        ])
                    elif inner_type == 'mid':
                        inner_key = f'{var}_{inner_type}_inner_state'
                        lines.extend([
                            f'{indent}_isl = {sl}',
                            f'{indent}if is_middle:',
                            f'{indent2}states["{inner_key}"][_isl] = {var_val}',
                            f'{indent}if is_inner_last:',
                            f'{indent2}{val_for} = states["{inner_key}"][_isl].clone()',
                            f'{indent}else:',
                            f'{indent2}{val_for} = torch.zeros_like({var_val})',
                        ])

            # Now emit the actual ops
            for var in dims_1d:
                safe_var = self._get_safe_name(var)
                var_val = f"{safe_var}_val"
                ops = self._variable_ops[var]

                for op in ops:
                    out_key = f'{var}_{op}'
                    sl_expr = 'slice(t * n, (t + 1) * n)'
                    op_parts = op.split('_')

                    # ---- Compound ops ----
                    if len(op_parts) > 1:
                        outer = op_parts[0]
                        inner = op_parts[1]
                        k_val = 1
                        is_arg = outer.startswith('arg')
                        match_k = re.match(r'(arg)?(max|min)(\d+)$', outer)
                        if match_k:
                            is_arg = match_k.group(1) is not None
                            outer_base = match_k.group(2)
                            k_val = int(match_k.group(3))
                        else:
                            outer_base = outer.lstrip('arg')

                        if inner == 'last':
                            val_var = var_val
                        else:
                            val_var = f"val_for_{safe_var}_{inner}"

                        lines.append(f'{indent}# Compound {op} for {safe_var}')
                        lines.append(f'{indent}_csl = {sl_expr}')

                        if is_arg:
                            # argmax_*/argmin_* compound
                            arg_type = outer_base
                            aux_key = f'{var}_{arg_type}{k_val if k_val > 1 else ""}_aux'
                            lines.extend([
                                f'{indent}if is_inner_last:',
                                f'{indent2}if is_outer_first:',
                                f'{indent2}    states["{out_key}"][_csl] = macro_step_index',
                                f'{indent2}    states["{aux_key}"][_csl] = {val_var}',
                                f'{indent2}else:',
                                f'{indent2}    _old_aux = states["{aux_key}"][_csl].clone()',
                                f'{indent2}    _cond = {val_var} {">" if arg_type == "max" else "<"} _old_aux',
                                f'{indent2}    states["{aux_key}"][_csl] = torch.where(_cond, {val_var}, _old_aux)',
                                f'{indent2}    _old_idx = states["{out_key}"][_csl].clone()',
                                f'{indent2}    _mi = torch.full_like(_old_idx, macro_step_index)',
                                f'{indent2}    states["{out_key}"][_csl] = torch.where(_cond, _mi, _old_idx)',
                            ])
                        elif outer_base in ('max', 'min') and k_val == 1:
                            cmp = 'torch.maximum' if outer_base == 'max' else 'torch.minimum'
                            lines.extend([
                                f'{indent}if is_inner_last:',
                                f'{indent2}if is_outer_first:',
                                f'{indent2}    states["{out_key}"][_csl] = {val_var}',
                                f'{indent2}else:',
                                f'{indent2}    _old = states["{out_key}"][_csl].clone()',
                                f'{indent2}    states["{out_key}"][_csl] = {cmp}(_old, {val_var})',
                            ])
                        elif outer == 'mean':
                            lines.extend([
                                f'{indent}if is_inner_last:',
                                f'{indent2}if is_outer_first:',
                                f'{indent2}    states["{out_key}"][_csl] = {val_var}',
                                f'{indent2}else:',
                                f'{indent2}    states["{out_key}"][_csl] += {val_var}',
                                f'{indent2}if is_outer_last:',
                                f'{indent2}    states["{out_key}"][_csl] /= num_macro_steps',
                            ])
                        elif outer == 'sum':
                            lines.extend([
                                f'{indent}if is_inner_last:',
                                f'{indent2}if is_outer_first:',
                                f'{indent2}    states["{out_key}"][_csl] = {val_var}',
                                f'{indent2}else:',
                                f'{indent2}    states["{out_key}"][_csl] += {val_var}',
                            ])
                        elif outer == 'last':
                            lines.extend([
                                f'{indent}if is_inner_last:',
                                f'{indent2}states["{out_key}"][_csl] = {val_var}',
                            ])
                        elif outer == 'first':
                            lines.extend([
                                f'{indent}if is_inner_last and is_outer_first:',
                                f'{indent2}states["{out_key}"][_csl] = {val_var}',
                            ])
                        continue

                    # ---- Simple ops ----
                    lines.append(f'{indent}# {op} for {safe_var}')
                    lines.append(f'{indent}_sl = {sl_expr}')

                    if op == 'mean':
                        lines.extend([
                            f'{indent}if is_inner_first:',
                            f'{indent2}_old = torch.zeros_like({var_val})',
                            f'{indent}else:',
                            f'{indent2}_old = states["{out_key}"][_sl].clone()',
                            f'{indent}_new = _old + {var_val} * weight',
                            f'{indent}if is_inner_last:',
                            f'{indent2}_new = _new / total_weight',
                            f'{indent}states["{out_key}"][_sl] = _new',
                        ])
                    elif op == 'sum':
                        lines.extend([
                            f'{indent}if is_inner_first:',
                            f'{indent2}_old = torch.zeros_like({var_val})',
                            f'{indent}else:',
                            f'{indent2}_old = states["{out_key}"][_sl].clone()',
                            f'{indent}states["{out_key}"][_sl] = _old + {var_val} * weight',
                        ])
                    elif op == 'max':
                        # Check if argmax also exists for merge
                        has_argmax = 'argmax' in self._variable_ops[var]
                        lines.extend([
                            f'{indent}if is_inner_first:',
                            f'{indent2}states["{out_key}"][_sl] = {var_val}',
                        ])
                        if has_argmax:
                            argmax_key = f'{var}_argmax'
                            aux_key = f'{var}_max_aux'
                            lines.extend([
                                f'{indent2}states["{aux_key}"][_sl] = {var_val}',
                                f'{indent2}states["{argmax_key}"][_sl] = macro_step_index',
                                f'{indent}else:',
                                f'{indent2}_old = states["{aux_key}"][_sl].clone()',
                                f'{indent2}_cond = {var_val} > _old',
                                f'{indent2}_new = torch.where(_cond, {var_val}, _old)',
                                f'{indent2}states["{aux_key}"][_sl] = _new',
                                f'{indent2}states["{out_key}"][_sl] = _new',
                                f'{indent2}_mi = torch.full_like(states["{argmax_key}"][_sl], macro_step_index)',
                                f'{indent2}states["{argmax_key}"][_sl] = torch.where(_cond, _mi, states["{argmax_key}"][_sl])',
                            ])
                        else:
                            lines.extend([
                                f'{indent}else:',
                                f'{indent2}_old = states["{out_key}"][_sl].clone()',
                                f'{indent2}states["{out_key}"][_sl] = torch.maximum(_old, {var_val})',
                            ])
                    elif op == 'argmax':
                        if 'max' in self._variable_ops[var]:
                            pass  # handled by max branch
                        else:
                            aux_key = f'{var}_max_aux'
                            lines.extend([
                                f'{indent}if is_inner_first:',
                                f'{indent2}states["{out_key}"][_sl] = macro_step_index',
                                f'{indent2}states["{aux_key}"][_sl] = {var_val}',
                                f'{indent}else:',
                                f'{indent2}_old_aux = states["{aux_key}"][_sl].clone()',
                                f'{indent2}_cond = {var_val} > _old_aux',
                                f'{indent2}states["{aux_key}"][_sl] = torch.where(_cond, {var_val}, _old_aux)',
                                f'{indent2}_mi = torch.full_like(states["{out_key}"][_sl], macro_step_index)',
                                f'{indent2}states["{out_key}"][_sl] = torch.where(_cond, _mi, states["{out_key}"][_sl])',
                            ])
                    elif op == 'min':
                        has_argmin = 'argmin' in self._variable_ops[var]
                        lines.extend([
                            f'{indent}if is_inner_first:',
                            f'{indent2}states["{out_key}"][_sl] = {var_val}',
                        ])
                        if has_argmin:
                            argmin_key = f'{var}_argmin'
                            aux_key = f'{var}_min_aux'
                            lines.extend([
                                f'{indent2}states["{aux_key}"][_sl] = {var_val}',
                                f'{indent2}states["{argmin_key}"][_sl] = macro_step_index',
                                f'{indent}else:',
                                f'{indent2}_old = states["{aux_key}"][_sl].clone()',
                                f'{indent2}_cond = {var_val} < _old',
                                f'{indent2}_new = torch.where(_cond, {var_val}, _old)',
                                f'{indent2}states["{aux_key}"][_sl] = _new',
                                f'{indent2}states["{out_key}"][_sl] = _new',
                                f'{indent2}_mi = torch.full_like(states["{argmin_key}"][_sl], macro_step_index)',
                                f'{indent2}states["{argmin_key}"][_sl] = torch.where(_cond, _mi, states["{argmin_key}"][_sl])',
                            ])
                        else:
                            lines.extend([
                                f'{indent}else:',
                                f'{indent2}_old = states["{out_key}"][_sl].clone()',
                                f'{indent2}states["{out_key}"][_sl] = torch.minimum(_old, {var_val})',
                            ])
                    elif op == 'argmin':
                        if 'min' in self._variable_ops[var]:
                            pass
                        else:
                            aux_key = f'{var}_min_aux'
                            lines.extend([
                                f'{indent}if is_inner_first:',
                                f'{indent2}states["{out_key}"][_sl] = macro_step_index',
                                f'{indent2}states["{aux_key}"][_sl] = {var_val}',
                                f'{indent}else:',
                                f'{indent2}_old_aux = states["{aux_key}"][_sl].clone()',
                                f'{indent2}_cond = {var_val} < _old_aux',
                                f'{indent2}states["{aux_key}"][_sl] = torch.where(_cond, {var_val}, _old_aux)',
                                f'{indent2}_mi = torch.full_like(states["{out_key}"][_sl], macro_step_index)',
                                f'{indent2}states["{out_key}"][_sl] = torch.where(_cond, _mi, states["{out_key}"][_sl])',
                            ])
                    elif op == 'last':
                        lines.extend([
                            f'{indent}if is_inner_last:',
                            f'{indent2}states["{out_key}"][_sl] = {var_val}',
                        ])
                    elif op == 'first':
                        lines.extend([
                            f'{indent}if is_inner_first:',
                            f'{indent2}states["{out_key}"][_sl] = {var_val}',
                        ])
                    elif op == 'mid':
                        lines.extend([
                            f'{indent}if is_middle:',
                            f'{indent2}states["{out_key}"][_sl] = {var_val}',
                        ])
                    else:
                        # maxK / minK / argmaxK / argminK
                        match_maxk = re.match(r'^(max|min)(\d+)$', op)
                        match_argk = re.match(r'^arg(max|min)(\d+)$', op)
                        if match_maxk:
                            op_type = match_maxk.group(1)
                            k_val = int(match_maxk.group(2))
                            cmp_fn = 'torch.maximum' if op_type == 'max' else 'torch.minimum'
                            init_val = 'float("-inf")' if op_type == 'max' else 'float("inf")'
                            cmp_op = '>' if op_type == 'max' else '<'
                            lines.extend([
                                f'{indent}# {op} bubble insert for {safe_var}',
                                f'{indent}_ksl = slice(t * n * {k_val}, (t + 1) * n * {k_val})',
                                f'{indent}if is_inner_first:',
                                f'{indent2}states["{out_key}"][_ksl] = {init_val}',
                                f'{indent2}states["{out_key}"][t * n * {k_val}:t * n * {k_val} + n] = {var_val}',
                                f'{indent}else:',
                                f'{indent2}_new_v = {var_val}.clone()',
                                f'{indent2}for k in range({k_val}):',
                                f'{indent2}    _k_off = t * n * {k_val} + k * n',
                                f'{indent2}    _old_k = states["{out_key}"][_k_off:_k_off + n].clone()',
                                f'{indent2}    _swap = _new_v {cmp_op} _old_k',
                                f'{indent2}    _to_store = torch.where(_swap, _new_v, _old_k)',
                                f'{indent2}    _new_v = torch.where(_swap, _old_k, _new_v)',
                                f'{indent2}    states["{out_key}"][_k_off:_k_off + n] = _to_store',
                            ])
                        elif match_argk:
                            pass  # handled by maxK/minK if merged, or standalone
                    lines.append('')

        # ---------- 2D variables ----------
        if dims_2d:
            lines.append(f'{indent}# === 2D variables ===')
            for var in dims_2d:
                safe_var = self._get_safe_name(var)
                actual_shape = tensor_info[var]['actual_shape']
                n_levels = actual_shape[-1]
                lines.append(f'{indent}n_levels = {n_levels}')

                info = self._field_registry.get(var)
                cat = getattr(info, 'json_schema_extra', {}).get('category', 'param')

                for op in self._variable_ops[var]:
                    out_key = f'{var}_{op}'
                    lines.append(f'{indent}# 2D {op} for {safe_var}')
                    lines.append(f'{indent}for level in range(n_levels):')

                    if cat == 'virtual':
                        # Virtual with 2D - load dependencies
                        expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                        lines.append(f'{indent2}# virtual: {expr}')
                        lines.append(f'{indent2}pass  # TODO: 2D virtual expression')
                    else:
                        lines.append(f'{indent2}_in_idx = (t * stride_input + idx) * n_levels + level')
                        lines.append(f'{indent2}_val = states["{var}"][_in_idx]')

                    lines.append(f'{indent2}_out_idx = (t * n + torch.arange(n, device=idx.device)) * n_levels + level')

                    if op == 'mean':
                        lines.extend([
                            f'{indent2}if is_inner_first:',
                            f'{indent2}    _old = torch.zeros_like(_val)',
                            f'{indent2}else:',
                            f'{indent2}    _old = states["{out_key}"][_out_idx]',
                            f'{indent2}_new = _old + _val * weight',
                            f'{indent2}if is_inner_last:',
                            f'{indent2}    _new = _new / total_weight',
                            f'{indent2}states["{out_key}"][_out_idx] = _new',
                        ])
                    elif op == 'max':
                        lines.extend([
                            f'{indent2}if is_inner_first:',
                            f'{indent2}    states["{out_key}"][_out_idx] = _val',
                            f'{indent2}else:',
                            f'{indent2}    _old = states["{out_key}"][_out_idx]',
                            f'{indent2}    states["{out_key}"][_out_idx] = torch.maximum(_old, _val)',
                        ])
                    elif op == 'min':
                        lines.extend([
                            f'{indent2}if is_inner_first:',
                            f'{indent2}    states["{out_key}"][_out_idx] = _val',
                            f'{indent2}else:',
                            f'{indent2}    _old = states["{out_key}"][_out_idx]',
                            f'{indent2}    states["{out_key}"][_out_idx] = torch.minimum(_old, _val)',
                        ])
                    elif op == 'last':
                        lines.extend([
                            f'{indent2}if is_inner_last:',
                            f'{indent2}    states["{out_key}"][_out_idx] = _val',
                        ])
                    elif op == 'first':
                        lines.extend([
                            f'{indent2}if is_inner_first:',
                            f'{indent2}    states["{out_key}"][_out_idx] = _val',
                        ])
                    elif op == 'mid':
                        lines.extend([
                            f'{indent2}if is_middle:',
                            f'{indent2}    states["{out_key}"][_out_idx] = _val',
                        ])
                    lines.append('')

        lines.append('')

    def _generate_pytorch_main_function(self: StatisticsAggregator, lines: List[str],
                                         grouped_by_save_idx: Dict[str, List[str]],
                                         tensor_info: Dict[str, Any]) -> None:
        """Generate the main entry-point function that calls per-group functions."""
        num_trials = self.num_trials if self.num_trials > 1 else 1
        lines.extend([
            '# Main update function',
            '@torch.compile',
            'def internal_update_statistics(states, BLOCK_SIZE):',
            '    weight = states["__weight"]',
            '    total_weight = states["__total_weight"]',
            '    num_macro_steps = states["__num_macro_steps"]',
            '    sub_step = states["__sub_step"]',
            '    num_sub_steps = states["__num_sub_steps"]',
            '    flags = states["__flags"]',
            '    macro_step_index = states["__macro_step_index"]',
            f'    num_trials = {num_trials}',
        ])

        for save_idx, var_list in grouped_by_save_idx.items():
            first_var = var_list[0]
            stride_input = 0
            for out_name, meta in self._metadata.items():
                if meta['original_variable'] == first_var:
                    stride_input = meta.get('stride_input', 0)
                    break
            lines.extend([
                f'    _update_{save_idx}(states, weight, total_weight, num_macro_steps,',
                f'                      sub_step, num_sub_steps, flags,',
                f'                      macro_step_index, num_trials, {stride_input})',
            ])
        lines.append('')

    def _generate_pytorch_aggregator_function(self: StatisticsAggregator) -> None:
        """Generate and compile a pure-PyTorch aggregation function (no Triton dependency)."""
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        tensor_info, grouped_by_save_idx = self._analyze_tensor_info()

        lines = self._generate_pytorch_header()

        for save_idx, var_list in grouped_by_save_idx.items():
            self._generate_pytorch_group_function(lines, save_idx, var_list, tensor_info)

        self._generate_pytorch_main_function(lines, grouped_by_save_idx, tensor_info)

        kernel_code = "\n".join(lines)
        self._write_and_import_kernels(kernel_code)

        if self.save_kernels:
            self._save_kernel_file(kernel_code)

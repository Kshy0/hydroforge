# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List

from hydroforge.statistics.ir import (
    ExpressionDialect, ExpressionSource, ScatterSource, TensorSource,
    render_expression,
)
from hydroforge.statistics.emitters.common import StatisticsEmitter

if TYPE_CHECKING:
    from hydroforge.statistics.runtime import StatisticsRuntime


class TorchStatisticsEmitter(StatisticsEmitter):
    """PyTorch (torch.compile) code generation for statistics aggregation."""

    def emit(self):
        self._generate_pytorch_aggregator_function()
        return self.result()

    # PyTorch code generation (no Triton dependency)
    # ========================================================================

    def _pytorch_state_expression(self: StatisticsRuntime, name: str) -> str:
        source = self._statistics_ir.sources.get(name, TensorSource(name))
        if isinstance(source, TensorSource):
            return f'states["{source.name}"]'
        if isinstance(source, ScatterSource):
            return f'states["__scatter_buf_{name}"]'
        names = {
            dependency: self._pytorch_state_expression(dependency)
            for dependency in source.expression.dependencies
        }
        return render_expression(
            source.expression, ExpressionDialect.TORCH, names,
        )

    def _generate_pytorch_header(self: StatisticsRuntime) -> List[str]:
        """Generate header for PyTorch-based aggregation code."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        var_list = sorted(list(self._variables))
        return [
            '"""',
            'Auto-generated PyTorch aggregation functions for hydroforge statistics.',
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

    def _pytorch_emit_val_load(self: StatisticsRuntime, var_name: str,
                                lines: List[str], emitted: set,
                                indent: str, is_2d: bool = False) -> str:
        """Emit PyTorch code to load a variable value (handling virtuals recursively).

        Returns the expression name for the loaded value.
        """
        safe_var = self._get_safe_name(var_name)
        val_name = f"{safe_var}_val"
        if safe_var in emitted:
            return val_name

        source = self._statistics_ir.sources.get(var_name, TensorSource(var_name))
        if isinstance(source, TensorSource):
            # Real data (includes virtual source buffers)
            if is_2d:
                lines.append(f'{indent}{val_name} = states["{var_name}"][(t * stride_input + idx) * n_levels + level]')
            else:
                lines.append(f'{indent}{val_name} = states["{var_name}"][t * stride_input + idx]')
        elif isinstance(source, ScatterSource):
            buf_key = f"__scatter_buf_{var_name}"
            if is_2d:
                lines.append(f'{indent}{val_name} = states["{buf_key}"][(t * stride_input + idx) * n_levels + level]')
            else:
                lines.append(f'{indent}{val_name} = states["{buf_key}"][t * stride_input + idx]')
        elif isinstance(source, ExpressionSource):
            names = {
                dependency: self._pytorch_emit_val_load(
                    dependency, lines, emitted, indent, is_2d,
                )
                for dependency in source.expression.dependencies
            }
            expression = render_expression(
                source.expression, ExpressionDialect.TORCH, names,
            )
            lines.append(f'{indent}{val_name} = {expression}')

        emitted.add(safe_var)
        return val_name

    def _generate_pytorch_full_function(
        self: StatisticsRuntime, lines: List[str], full_vars: List[str],
    ) -> None:
        """Generate a PyTorch function for variables saved at full tensor shape."""
        if not full_vars:
            return

        lines.extend([
            'def _update___full__(states, weight, total_weight, num_macro_steps,',
            '                     sub_step, num_sub_steps, flags, macro_step_index):',
            '    is_inner_first = (flags & 1) != 0 and sub_step == 0',
            '    is_inner_last = ((flags >> 1) & 1) != 0 and sub_step == num_sub_steps - 1',
            '    is_middle = sub_step == num_sub_steps // 2',
            '    is_outer_first = ((flags >> 2) & 1) != 0 and is_inner_last',
            '    is_outer_last = ((flags >> 3) & 1) != 0 and is_inner_last',
            '',
        ])

        for var in full_vars:
            safe_var = self._get_safe_name(var)
            lines.extend([
                f'    # === full tensor variable: {var} ===',
                f'    {safe_var}_val = states["{var}"]',
            ])

            for operation in self._statistics_lowering.operations(var):
                op = operation.spelling
                out_key = f'{var}_{op}'

                if operation.compound:
                    outer = operation.outer.value
                    inner = operation.inner.value
                    inner_val = f'{safe_var}_{inner}_val'
                    lines.append(f'    {inner_val} = torch.zeros_like({safe_var}_val)')

                    if inner == 'last':
                        lines.extend([
                            '    if is_inner_last:',
                            f'        {inner_val} = {safe_var}_val',
                        ])
                    elif inner == 'mean':
                        inner_key = f'{var}_{inner}_inner_state'
                        weight_key = f'{var}_{inner}_weight_state'
                        lines.extend([
                            f'    _inner_old = states["{inner_key}"].clone()',
                            f'    _w_old = states["{weight_key}"].clone()',
                            f'    _inner_new = _inner_old + {safe_var}_val * weight',
                            '    _w_new = _w_old + weight',
                            '    if is_inner_last:',
                            f'        {inner_val} = _inner_new / _w_new',
                            f'        states["{inner_key}"].zero_()',
                            f'        states["{weight_key}"].zero_()',
                            '    else:',
                            f'        states["{inner_key}"].copy_(_inner_new)',
                            f'        states["{weight_key}"].copy_(_w_new)',
                        ])
                    elif inner == 'sum':
                        inner_key = f'{var}_{inner}_inner_state'
                        lines.extend([
                            f'    _inner_old = states["{inner_key}"].clone()',
                            f'    _inner_new = _inner_old + {safe_var}_val * weight',
                            '    if is_inner_last:',
                            f'        {inner_val} = _inner_new',
                            f'        states["{inner_key}"].zero_()',
                            '    else:',
                            f'        states["{inner_key}"].copy_(_inner_new)',
                        ])
                    elif inner == 'max':
                        inner_key = f'{var}_{inner}_inner_state'
                        lines.extend([
                            '    if is_inner_first:',
                            f'        states["{inner_key}"].copy_({safe_var}_val)',
                            '    else:',
                            f'        states["{inner_key}"].copy_(torch.maximum(states["{inner_key}"], {safe_var}_val))',
                            '    if is_inner_last:',
                            f'        {inner_val} = states["{inner_key}"].clone()',
                            f'        states["{inner_key}"].fill_(float("-inf"))',
                        ])
                    elif inner == 'min':
                        inner_key = f'{var}_{inner}_inner_state'
                        lines.extend([
                            '    if is_inner_first:',
                            f'        states["{inner_key}"].copy_({safe_var}_val)',
                            '    else:',
                            f'        states["{inner_key}"].copy_(torch.minimum(states["{inner_key}"], {safe_var}_val))',
                            '    if is_inner_last:',
                            f'        {inner_val} = states["{inner_key}"].clone()',
                            f'        states["{inner_key}"].fill_(float("inf"))',
                        ])
                    elif inner == 'first':
                        inner_key = f'{var}_{inner}_inner_state'
                        lines.extend([
                            '    if is_inner_first:',
                            f'        states["{inner_key}"].copy_({safe_var}_val)',
                            '    if is_inner_last:',
                            f'        {inner_val} = states["{inner_key}"]',
                        ])
                    elif inner == 'mid':
                        inner_key = f'{var}_{inner}_inner_state'
                        lines.extend([
                            '    if is_middle:',
                            f'        states["{inner_key}"].copy_({safe_var}_val)',
                            '    if is_inner_last:',
                            f'        {inner_val} = states["{inner_key}"]',
                        ])
                    else:
                        raise ValueError(f"Unsupported full-output inner op '{inner}'.")

                    lines.append('    if is_inner_last:')
                    if outer == 'max':
                        lines.extend([
                            '        if is_outer_first:',
                            f'            states["{out_key}"].copy_({inner_val})',
                            '        else:',
                            f'            states["{out_key}"].copy_(torch.maximum(states["{out_key}"], {inner_val}))',
                        ])
                    elif outer == 'min':
                        lines.extend([
                            '        if is_outer_first:',
                            f'            states["{out_key}"].copy_({inner_val})',
                            '        else:',
                            f'            states["{out_key}"].copy_(torch.minimum(states["{out_key}"], {inner_val}))',
                        ])
                    elif outer == 'sum':
                        lines.extend([
                            '        if is_outer_first:',
                            f'            states["{out_key}"].copy_({inner_val})',
                            '        else:',
                            f'            states["{out_key}"].add_({inner_val})',
                        ])
                    elif outer == 'mean':
                        lines.extend([
                            '        if is_outer_first:',
                            f'            states["{out_key}"].copy_({inner_val})',
                            '        else:',
                            f'            states["{out_key}"].add_({inner_val})',
                            '        if is_outer_last:',
                            f'            states["{out_key}"].div_(num_macro_steps)',
                        ])
                    elif outer == 'last':
                        lines.append(f'        states["{out_key}"].copy_({inner_val})')
                    elif outer == 'first':
                        lines.extend([
                            '        if is_outer_first:',
                            f'            states["{out_key}"].copy_({inner_val})',
                        ])
                    else:
                        raise ValueError(f"Unsupported full-output outer op '{outer}'.")
                    lines.append('')
                    continue

                if op == 'mean':
                    lines.extend([
                        '    if is_inner_first:',
                        f'        states["{out_key}"].zero_()',
                        f'    states["{out_key}"].add_({safe_var}_val * weight)',
                        '    if is_inner_last:',
                        f'        states["{out_key}"].div_(total_weight)',
                    ])
                elif op == 'sum':
                    lines.extend([
                        '    if is_inner_first:',
                        f'        states["{out_key}"].zero_()',
                        f'    states["{out_key}"].add_({safe_var}_val * weight)',
                    ])
                elif op == 'max':
                    lines.extend([
                        '    if is_inner_first:',
                        f'        states["{out_key}"].copy_({safe_var}_val)',
                        '    else:',
                        f'        states["{out_key}"].copy_(torch.maximum(states["{out_key}"], {safe_var}_val))',
                    ])
                elif op == 'min':
                    lines.extend([
                        '    if is_inner_first:',
                        f'        states["{out_key}"].copy_({safe_var}_val)',
                        '    else:',
                        f'        states["{out_key}"].copy_(torch.minimum(states["{out_key}"], {safe_var}_val))',
                    ])
                elif op == 'last':
                    lines.extend([
                        '    if is_inner_last:',
                        f'        states["{out_key}"].copy_({safe_var}_val)',
                    ])
                elif op == 'first':
                    lines.extend([
                        '    if is_inner_first:',
                        f'        states["{out_key}"].copy_({safe_var}_val)',
                    ])
                elif op == 'mid':
                    lines.extend([
                        '    if is_middle:',
                        f'        states["{out_key}"].copy_({safe_var}_val)',
                    ])
                else:
                    raise ValueError(f"Unsupported full-output op '{op}'.")
                lines.append('')

    def _generate_pytorch_group_function(
        self: StatisticsRuntime, lines: List[str],
        output_index: str, var_list: List[str],
    ) -> None:
        """Generate a PyTorch function for one output_index group."""
        dims_1d, dims_2d = self._statistics_lowering.split_indexed(var_list)

        func_name = f"_update_{output_index}"
        lines.extend([
            f'def {func_name}(states, weight, total_weight, num_macro_steps,',
            '               sub_step, num_sub_steps, flags,',
            '               macro_step_index, num_trials, stride_input):',
            '    states = {key: value.reshape(-1) for key, value in states.items()}',
        ])
        needed_bools = self._statistics_lowering.required_flags
        if needed_bools:
            lines.append('    # Compute boolean flags from sub_step, num_sub_steps, flags')
            if 'is_inner_first' in needed_bools:
                lines.append('    is_inner_first = (flags & 1) != 0 and sub_step == 0')
            if 'is_inner_last' in needed_bools:
                lines.append('    is_inner_last = ((flags >> 1) & 1) != 0 and sub_step == num_sub_steps - 1')
            if 'is_middle' in needed_bools:
                lines.append('    is_middle = sub_step == num_sub_steps // 2')
            if 'is_outer_first' in needed_bools:
                lines.append('    is_outer_first = ((flags >> 2) & 1) != 0 and is_inner_last')
            if 'is_outer_last' in needed_bools:
                lines.append('    is_outer_last = ((flags >> 3) & 1) != 0 and is_inner_last')
        lines.extend([
            f'    idx = states["{output_index}"]',
            '    n = len(idx)',
            '',
        ])

        indent = '        '  # inside for t loop
        indent2 = indent + '    '

        lines.append('    for t in range(num_trials):')

        # ---------- 1D variables ----------
        if dims_1d:
            lines.append(f'{indent}# === 1D variables ===')
            emitted = set()

            # Pre-load all needed values
            for var in dims_1d:
                self._pytorch_emit_val_load(var, lines, emitted, indent, is_2d=False)

            # Inner aggregation states (for compound ops)
            # Emit inner aggregation state updates
            for reduction, inner_vars in (
                self._statistics_lowering.variables_by_inner(dims_1d).items()
            ):
                inner_type = reduction.value
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
                operations = self._statistics_lowering.operations(var)

                for operation in operations:
                    op = operation.spelling
                    out_key = f'{var}_{op}'
                    sl_expr = 'slice(t * n, (t + 1) * n)'

                    # ---- Compound ops ----
                    if operation.compound:
                        outer = operation.outer.value
                        inner = operation.inner.value
                        k_val = operation.k
                        is_arg = operation.stores_index
                        outer_base = operation.outer.value

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
                        lines.extend([
                            f'{indent}if is_inner_first:',
                            f'{indent2}states["{out_key}"][_sl] = {var_val}',
                            f'{indent}else:',
                            f'{indent2}_old = states["{out_key}"][_sl].clone()',
                            f'{indent2}states["{out_key}"][_sl] = torch.maximum(_old, {var_val})',
                        ])
                    elif op == 'min':
                        lines.extend([
                            f'{indent}if is_inner_first:',
                            f'{indent2}states["{out_key}"][_sl] = {var_val}',
                            f'{indent}else:',
                            f'{indent2}_old = states["{out_key}"][_sl].clone()',
                            f'{indent2}states["{out_key}"][_sl] = torch.minimum(_old, {var_val})',
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
                    lines.append('')

        # ---------- 2D variables ----------
        if dims_2d:
            lines.append(f'{indent}# === 2D variables ===')
            for var in dims_2d:
                safe_var = self._get_safe_name(var)
                actual_shape = (
                    self._statistics_lowering.by_name[var].variable.actual_shape
                )
                n_levels = actual_shape[-1]
                lines.append(f'{indent}n_levels = {n_levels}')

                for operation in self._statistics_lowering.operations(var):
                    op = operation.spelling
                    out_key = f'{var}_{op}'
                    lines.append(f'{indent}# 2D {op} for {safe_var}')
                    lines.append(f'{indent}for level in range(n_levels):')
                    emitted: set[str] = set()
                    var_val = self._pytorch_emit_val_load(
                        var, lines, emitted, indent2, is_2d=True,
                    )
                    lines.append(f'{indent2}_val = {var_val}')

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
                    elif op == 'sum':
                        lines.extend([
                            f'{indent2}if is_inner_first:',
                            f'{indent2}    _old = torch.zeros_like(_val)',
                            f'{indent2}else:',
                            f'{indent2}    _old = states["{out_key}"][_out_idx]',
                            f'{indent2}states["{out_key}"][_out_idx] = _old + _val * weight',
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

    def _generate_pytorch_main_function(
        self: StatisticsRuntime, lines: List[str],
        grouped_by_output_index: Dict[str, List[str]],
    ) -> None:
        """Generate the main entry-point function that calls per-group functions."""
        num_trials = self.num_trials if self.num_trials > 1 else 1
        full_vars = grouped_by_output_index.get("__full__", [])
        if not full_vars:
            lines.append('@torch.compile')
        lines.extend([
            '# Main update function',
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

        scatters = self._statistics_ir.ordered_scatters()
        if scatters:
            lines.append('    # Materialize all scatter virtuals in dependency order')
        for variable in scatters:
            var = variable.name
            scatter = variable.source
            buf_key = f"__scatter_buf_{var}"
            lines.append(f'    states["{buf_key}"].zero_()')
            names = {
                dependency: self._pytorch_state_expression(dependency)
                for dependency in scatter.value.dependencies
            }
            expression = render_expression(
                scatter.value, ExpressionDialect.TORCH, names,
            )
            lines.append(f'    _scatter_val = {expression}')
            lines.append(f'    _scatter_idx = states["{scatter.index}"].long()')
            if num_trials > 1:
                lines.append(
                    '    _scatter_idx_exp = '
                    '_scatter_idx.unsqueeze(0).expand_as(_scatter_val)'
                )
                lines.append(
                    f'    states["{buf_key}"].scatter_add_('
                    '1, _scatter_idx_exp, _scatter_val)'
                )
            else:
                lines.append(
                    f'    states["{buf_key}"].scatter_add_('
                    '0, _scatter_idx, _scatter_val)'
                )
            if scatter.reduction.value == 'mean':
                lines.append(
                    f'    _scatter_cnt = states["{buf_key}"].new_zeros('
                    f'states["{buf_key}"].shape)'
                )
                lines.append(
                    '    _scatter_ones = _scatter_val.new_ones(_scatter_val.shape)'
                )
                if num_trials > 1:
                    lines.append(
                        '    _scatter_cnt.scatter_add_('
                        '1, _scatter_idx_exp, _scatter_ones)'
                    )
                else:
                    lines.append(
                        '    _scatter_cnt.scatter_add_('
                        '0, _scatter_idx, _scatter_ones)'
                    )
                lines.append('    _scatter_cnt.clamp_(min=1.0)')
                lines.append(f'    states["{buf_key}"].div_(_scatter_cnt)')
        if scatters:
            lines.append('')

        if full_vars:
            lines.extend([
                '    _update___full__(states, weight, total_weight, num_macro_steps,',
                '                     sub_step, num_sub_steps, flags, macro_step_index)',
            ])

        for output_index, var_list in grouped_by_output_index.items():
            if output_index == "__full__":
                continue
            first_var = var_list[0]
            stride_input = 0
            for meta in self._metadata.values():
                if meta['original_variable'] == first_var:
                    stride_input = meta.get('stride_input', 0)
                    break

            lines.extend([
                f'    _update_{output_index}(states, weight, total_weight, num_macro_steps,',
                '                      sub_step, num_sub_steps, flags,',
                f'                      macro_step_index, num_trials, {stride_input})',
            ])
        lines.append('')

    def _generate_pytorch_aggregator_function(
        self: StatisticsRuntime,
    ) -> None:
        """Generate and compile a pure-PyTorch aggregation function (no Triton dependency)."""
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        grouped_by_output_index = self._statistics_lowering.groups

        lines = self._generate_pytorch_header()

        full_vars = grouped_by_output_index.get("__full__", [])
        self._generate_pytorch_full_function(lines, full_vars)

        for output_index, var_list in grouped_by_output_index.items():
            if output_index == "__full__":
                continue
            self._generate_pytorch_group_function(lines, output_index, var_list)

        self._generate_pytorch_main_function(lines, grouped_by_output_index)

        kernel_code = "\n".join(lines)
        self._compile_generated_kernels(kernel_code)

        if self.save_kernels:
            self._save_kernel_file(kernel_code)

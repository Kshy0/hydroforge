# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from typing import TYPE_CHECKING

from hydroforge.statistics.ir import (
    ExpressionDialect, ExpressionSource, ScatterSource, StatisticsIR,
    TensorSource, render_expression,
)
from hydroforge.statistics.emitters.common import StatisticsEmitter

if TYPE_CHECKING:
    from hydroforge.statistics.runtime import StatisticsRuntime


def _emit_argument_kernel_start(
    lines: list[str], kernel_name: str, fields: list[tuple[str, str, bool]],
) -> None:
    """Emit an argument-buffer struct and kernel entry point directly."""
    lines.append(f"struct {kernel_name}_args {{")
    for index, (type_decl, name, scalar) in enumerate(fields):
        native_type = "long" if scalar and type_decl == "int" else type_decl
        field_type = f"constant {native_type}*" if scalar else type_decl
        lines.append(f"    {field_type} arg_{index} [[id({index})]];")
    lines.append(
        f"    constant long* _grid_size [[id({len(fields)})]];"
    )
    lines.extend([
        "};", "", f"kernel void {kernel_name}(",
        f"    constant {kernel_name}_args& args [[buffer(0)]],",
        "    uint tid [[thread_position_in_grid]]", ") {",
    ])
    for index, (type_decl, name, scalar) in enumerate(fields):
        if scalar:
            native_type = "long" if type_decl == "int" else type_decl
            lines.append(
                f"    const {native_type} {name} = *args.arg_{index};"
            )
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

    def _metal_dtype_str(self: StatisticsRuntime, var_name: str) -> str:
        """Return the Metal Shading Language scalar type string for a variable."""
        import torch
        from hydroforge.kernels.backends.metal.types import tensor_type

        tensor = self._tensor_registry.get(var_name)
        if tensor is not None:
            dt = tensor.dtype
        else:
            stored = self._storage.get(var_name)
            dt = stored.dtype if stored is not None else torch.float32
        try:
            return tensor_type(dt)
        except TypeError as exc:
            raise TypeError(f"Metal statistics does not support dtype {dt}") from exc

    def _metal_emit_val_load(self: StatisticsRuntime, var_name: str,
                              lines: list, emitted: set, indent: str,
                              idx_expr: str = "idx") -> str:
        """Emit MSL code to load a variable value (handling virtuals).

        Returns the MSL variable name for the loaded value.
        """
        safe_var = self._get_safe_name(var_name)
        val_name = f"{safe_var}_val"
        if safe_var in emitted:
            return val_name

        ctype = self._metal_dtype_str(var_name)
        source = self._statistics_ir.sources.get(var_name, TensorSource(var_name))

        if isinstance(source, TensorSource):
            lines.append(f'{indent}{ctype} {val_name} = p_{safe_var}[{idx_expr}];')
        elif isinstance(source, ScatterSource):
            buffer_name = self._get_safe_name(f"__scatter_buf_{var_name}")
            lines.append(f'{indent}{ctype} {val_name} = p_{buffer_name}[{idx_expr}];')
        elif isinstance(source, ExpressionSource):
            names = {
                dependency: self._metal_emit_val_load(
                    dependency, lines, emitted, indent, idx_expr,
                )
                for dependency in source.expression.dependencies
            }
            expression = render_expression(
                source.expression, ExpressionDialect.METAL, names,
            )
            lines.append(f'{indent}{ctype} {val_name} = ({ctype})({expression});')

        emitted.add(safe_var)
        return val_name

    def _generate_metal_full_kernel_for_group(self: StatisticsRuntime,
                                              msl_lines: list,
                                              output_index: str,
                                              var_list: list) -> dict:
        """Generate a Metal kernel for variables saved at full tensor shape."""
        kernel_name = f"aggr_kernel_{self._get_safe_name(output_index)}"

        sorted_inputs = []
        for var in var_list:
            if var not in self._tensor_registry:
                raise ValueError(
                    f"Full-output Metal aggregation requires '{var}' to be a registered tensor"
                )
            sorted_inputs.append(var)
        sorted_inputs = sorted(dict.fromkeys(sorted_inputs))

        out_tensors = []
        seen_out = set()
        for var in var_list:
            safe_var = self._get_safe_name(var)
            ctype = self._metal_dtype_str(var)
            operations = self._statistics_lowering.operations(var)
            for operation in operations:
                op = operation.spelling
                if operation.stores_index or operation.k > 1:
                    raise ValueError(f"Full-output Metal aggregation does not support op '{op}'")

                out_key = f'{var}_{op}'
                if out_key not in seen_out:
                    out_tensors.append((out_key, ctype, f"p_{safe_var}_{op}"))
                    seen_out.add(out_key)

                if operation.inner is None:
                    continue
                inner = operation.inner.value
                if inner == 'last':
                    continue
                inner_key = f'{var}_{inner}_inner_state'
                if inner_key not in seen_out:
                    out_tensors.append((inner_key, ctype, f"p_{safe_var}_{inner}_inner_state"))
                    seen_out.add(inner_key)
                if inner == 'mean':
                    weight_key = f'{var}_{inner}_weight_state'
                    if weight_key not in seen_out:
                        out_tensors.append((weight_key, ctype, f"p_{safe_var}_{inner}_weight_state"))
                        seen_out.add(weight_key)

        full_total = max(int(self._tensor_registry[var].numel()) for var in var_list)

        arg_order = []
        abi_fields = []
        indent = "    "

        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            ctype = self._metal_dtype_str(inp)
            abi_fields.append((f'device const {ctype}*', f'p_{safe_inp}', False))
            arg_order.append(('tensor', inp, 'read'))

        for state_key, ctype, pname in out_tensors:
            abi_fields.append((f'device {ctype}*', pname, False))
            arg_order.append(('tensor', state_key, 'read_write'))

        varying_scalar_params = [
            ('weight', 'float', '__weight'), ('total_weight', 'float', '__total_weight'),
            ('num_macro_steps', 'float', '__num_macro_steps'),
            ('sub_step', 'int', '__sub_step'), ('num_sub_steps', 'int', '__num_sub_steps'),
            ('flags', 'int', '__flags'),
            ('macro_step_index', 'int', '__macro_step_index'),
        ]
        for sname, stype, state_key in varying_scalar_params:
            abi_fields.append((f'device const {stype}*', f'p_{sname}_ptr', False))
            arg_order.append(('tensor', state_key, 'read'))

        abi_fields.append(('int', 'n_elements', True))
        arg_order.append(('scalar', 'n_elements', 'int'))
        _emit_argument_kernel_start(msl_lines, kernel_name, abi_fields)
        msl_lines.append(f'{indent}if ((int)tid >= n_elements) return;')
        msl_lines.append(f'{indent}int out_idx = (int)tid;')
        msl_lines.append('')
        msl_lines.append(f'{indent}float weight = *p_weight_ptr;')
        msl_lines.append(f'{indent}float total_weight = *p_total_weight_ptr;')
        msl_lines.append(f'{indent}float num_macro_steps = *p_num_macro_steps_ptr;')
        msl_lines.append(f'{indent}int sub_step = *p_sub_step_ptr;')
        msl_lines.append(f'{indent}int num_sub_steps = *p_num_sub_steps_ptr;')
        msl_lines.append(f'{indent}int flags = *p_flags_ptr;')
        msl_lines.append(f'{indent}int macro_step_index = *p_macro_step_index_ptr;')
        msl_lines.append('')

        needed_bools = self._statistics_lowering.required_flags
        if needed_bools:
            if 'is_inner_first' in needed_bools:
                msl_lines.append(f'{indent}bool is_inner_first = ((flags & 1) != 0) && (sub_step == 0);')
            if 'is_inner_last' in needed_bools:
                msl_lines.append(f'{indent}bool is_inner_last = (((flags >> 1) & 1) != 0) && (sub_step == num_sub_steps - 1);')
            if 'is_middle' in needed_bools:
                msl_lines.append(f'{indent}bool is_middle = (sub_step == num_sub_steps / 2);')
            if 'is_outer_first' in needed_bools:
                msl_lines.append(f'{indent}bool is_outer_first = (((flags >> 2) & 1) != 0) && is_inner_last;')
            if 'is_outer_last' in needed_bools:
                msl_lines.append(f'{indent}bool is_outer_last = (((flags >> 3) & 1) != 0) && is_inner_last;')
            msl_lines.append('')

        for var in var_list:
            safe_var = self._get_safe_name(var)
            ctype = self._metal_dtype_str(var)
            operations = self._statistics_lowering.operations(var)
            var_numel = int(self._tensor_registry[var].numel())
            var_val = f"{safe_var}_val"
            indent2 = indent + "    "
            msl_lines.append(f'{indent}if ((int)tid < {var_numel}) {{')
            msl_lines.append(f'{indent2}{ctype} {var_val} = p_{safe_var}[out_idx];')

            inner_ops = (
                reduction.value
                for reduction in self._statistics_lowering.inner_reductions(var)
            )
            for inner_type in inner_ops:
                if inner_type == 'last':
                    continue
                val_for = f"val_for_{safe_var}_{inner_type}"
                msl_lines.append(f'{indent2}{ctype} {val_for} = ({ctype})0;')
                if inner_type == 'mean':
                    msl_lines.extend([
                        f'{indent2}{{',
                        f'{indent2}    {ctype} inner_old = p_{safe_var}_mean_inner_state[out_idx];',
                        f'{indent2}    {ctype} w_old = p_{safe_var}_mean_weight_state[out_idx];',
                        f'{indent2}    {ctype} inner_new = inner_old + {var_val} * ({ctype})weight;',
                        f'{indent2}    {ctype} w_new = w_old + ({ctype})weight;',
                        f'{indent2}    if (is_inner_last) {{',
                        f'{indent2}        p_{safe_var}_mean_inner_state[out_idx] = ({ctype})0;',
                        f'{indent2}        p_{safe_var}_mean_weight_state[out_idx] = ({ctype})0;',
                        f'{indent2}        {val_for} = inner_new / w_new;',
                        f'{indent2}    }} else {{',
                        f'{indent2}        p_{safe_var}_mean_inner_state[out_idx] = inner_new;',
                        f'{indent2}        p_{safe_var}_mean_weight_state[out_idx] = w_new;',
                        f'{indent2}    }}',
                        f'{indent2}}}',
                    ])
                elif inner_type == 'sum':
                    msl_lines.extend([
                        f'{indent2}{{',
                        f'{indent2}    {ctype} inner_new = p_{safe_var}_sum_inner_state[out_idx] + {var_val} * ({ctype})weight;',
                        f'{indent2}    if (is_inner_last) {{',
                        f'{indent2}        p_{safe_var}_sum_inner_state[out_idx] = ({ctype})0;',
                        f'{indent2}        {val_for} = inner_new;',
                        f'{indent2}    }} else {{',
                        f'{indent2}        p_{safe_var}_sum_inner_state[out_idx] = inner_new;',
                        f'{indent2}    }}',
                        f'{indent2}}}',
                    ])
                elif inner_type == 'max':
                    msl_lines.extend([
                        f'{indent2}{{',
                        f'{indent2}    {ctype} old_v = p_{safe_var}_max_inner_state[out_idx];',
                        f'{indent2}    {ctype} inner_new = is_inner_first ? {var_val} : fmax(old_v, {var_val});',
                        f'{indent2}    if (is_inner_last) {{',
                        f'{indent2}        p_{safe_var}_max_inner_state[out_idx] = ({ctype})(-1e38);',
                        f'{indent2}        {val_for} = inner_new;',
                        f'{indent2}    }} else {{',
                        f'{indent2}        p_{safe_var}_max_inner_state[out_idx] = inner_new;',
                        f'{indent2}    }}',
                        f'{indent2}}}',
                    ])
                elif inner_type == 'min':
                    msl_lines.extend([
                        f'{indent2}{{',
                        f'{indent2}    {ctype} old_v = p_{safe_var}_min_inner_state[out_idx];',
                        f'{indent2}    {ctype} inner_new = is_inner_first ? {var_val} : fmin(old_v, {var_val});',
                        f'{indent2}    if (is_inner_last) {{',
                        f'{indent2}        p_{safe_var}_min_inner_state[out_idx] = ({ctype})1e38;',
                        f'{indent2}        {val_for} = inner_new;',
                        f'{indent2}    }} else {{',
                        f'{indent2}        p_{safe_var}_min_inner_state[out_idx] = inner_new;',
                        f'{indent2}    }}',
                        f'{indent2}}}',
                    ])
                elif inner_type == 'first':
                    msl_lines.extend([
                        f'{indent2}if (is_inner_first) p_{safe_var}_first_inner_state[out_idx] = {var_val};',
                        f'{indent2}if (is_inner_last) {val_for} = p_{safe_var}_first_inner_state[out_idx];',
                    ])
                elif inner_type == 'mid':
                    msl_lines.extend([
                        f'{indent2}if (is_middle) p_{safe_var}_mid_inner_state[out_idx] = {var_val};',
                        f'{indent2}if (is_inner_last) {val_for} = p_{safe_var}_mid_inner_state[out_idx];',
                    ])
                else:
                    raise ValueError(f"Unsupported full-output inner op '{inner_type}'")

            for operation in operations:
                op = operation.spelling
                out_ptr = f"p_{safe_var}_{op}"
                if operation.compound:
                    outer = operation.outer.value
                    inner = operation.inner.value
                    val_var = var_val if inner == 'last' else f"val_for_{safe_var}_{inner}"
                    if outer == 'max':
                        msl_lines.extend([
                            f'{indent2}if (is_inner_last) {{',
                            f'{indent2}    if (is_outer_first) {{ {out_ptr}[out_idx] = {val_var}; }}',
                            f'{indent2}    else {{ {out_ptr}[out_idx] = fmax({out_ptr}[out_idx], {val_var}); }}',
                            f'{indent2}}}',
                        ])
                    elif outer == 'min':
                        msl_lines.extend([
                            f'{indent2}if (is_inner_last) {{',
                            f'{indent2}    if (is_outer_first) {{ {out_ptr}[out_idx] = {val_var}; }}',
                            f'{indent2}    else {{ {out_ptr}[out_idx] = fmin({out_ptr}[out_idx], {val_var}); }}',
                            f'{indent2}}}',
                        ])
                    elif outer == 'sum':
                        msl_lines.extend([
                            f'{indent2}if (is_inner_last) {{',
                            f'{indent2}    if (is_outer_first) {{ {out_ptr}[out_idx] = {val_var}; }}',
                            f'{indent2}    else {{ {out_ptr}[out_idx] += {val_var}; }}',
                            f'{indent2}}}',
                        ])
                    elif outer == 'mean':
                        msl_lines.extend([
                            f'{indent2}if (is_inner_last) {{',
                            f'{indent2}    if (is_outer_first) {{ {out_ptr}[out_idx] = {val_var}; }}',
                            f'{indent2}    else {{ {out_ptr}[out_idx] += {val_var}; }}',
                            f'{indent2}    if (is_outer_last) {{ {out_ptr}[out_idx] /= ({ctype})num_macro_steps; }}',
                            f'{indent2}}}',
                        ])
                    elif outer == 'last':
                        msl_lines.append(f'{indent2}if (is_inner_last) {{ {out_ptr}[out_idx] = {val_var}; }}')
                    elif outer == 'first':
                        msl_lines.append(f'{indent2}if (is_inner_last && is_outer_first) {{ {out_ptr}[out_idx] = {val_var}; }}')
                    else:
                        raise ValueError(f"Unsupported full-output outer op '{outer}'")
                    continue

                if op == 'mean':
                    msl_lines.extend([
                        f'{indent2}{{',
                        f'{indent2}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[out_idx];',
                        f'{indent2}    {ctype} new_val = old_val + {var_val} * ({ctype})weight;',
                        f'{indent2}    {out_ptr}[out_idx] = is_inner_last ? new_val / ({ctype})total_weight : new_val;',
                        f'{indent2}}}',
                    ])
                elif op == 'sum':
                    msl_lines.extend([
                        f'{indent2}{{',
                        f'{indent2}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[out_idx];',
                        f'{indent2}    {out_ptr}[out_idx] = old_val + {var_val} * ({ctype})weight;',
                        f'{indent2}}}',
                    ])
                elif op == 'max':
                    msl_lines.extend([
                        f'{indent2}if (is_inner_first) {{ {out_ptr}[out_idx] = {var_val}; }}',
                        f'{indent2}else {{ {out_ptr}[out_idx] = fmax({out_ptr}[out_idx], {var_val}); }}',
                    ])
                elif op == 'min':
                    msl_lines.extend([
                        f'{indent2}if (is_inner_first) {{ {out_ptr}[out_idx] = {var_val}; }}',
                        f'{indent2}else {{ {out_ptr}[out_idx] = fmin({out_ptr}[out_idx], {var_val}); }}',
                    ])
                elif op == 'last':
                    msl_lines.append(f'{indent2}if (is_inner_last) {{ {out_ptr}[out_idx] = {var_val}; }}')
                elif op == 'first':
                    msl_lines.append(f'{indent2}if (is_inner_first) {{ {out_ptr}[out_idx] = {var_val}; }}')
                elif op == 'mid':
                    msl_lines.append(f'{indent2}if (is_middle) {{ {out_ptr}[out_idx] = {var_val}; }}')
                else:
                    raise ValueError(f"Unsupported full-output op '{op}'")

            msl_lines.append(f'{indent}}}')
            msl_lines.append('')

        msl_lines.append('}')
        msl_lines.append('')

        return {
            'kernel_name': kernel_name,
            'output_index': output_index,
            'full_output': True,
            'n_elements_val': full_total,
            'arg_order': arg_order,
        }

    def _generate_metal_kernel_for_group(self: StatisticsRuntime,
                                          msl_lines: list,
                                          output_index: str,
                                          var_list: list) -> dict:
        """Generate a Metal kernel function for one output_index group.

        Returns metadata dict used by the Python wrapper.
        """
        if output_index == "__full__":
            return self._generate_metal_full_kernel_for_group(
                msl_lines, output_index, var_list
            )

        num_trials = self.num_trials if self.num_trials > 1 else 1
        safe_save = self._get_safe_name(output_index)
        kernel_name = f"aggr_kernel_{safe_save}"

        dims_1d, dims_2d = self._statistics_lowering.split_indexed(var_list)

        sorted_inputs = sorted({
            name for var in var_list
            for name in self._statistics_ir.materialized_inputs(var)
        })

        # Collect output/state tensor names + types
        out_tensors = []  # (state_key, ctype, safe_param_name)
        for var in var_list:
            safe_var = self._get_safe_name(var)
            ctype = self._metal_dtype_str(var)
            added_aux = set()
            operations = self._statistics_lowering.operations(var)
            for operation in operations:
                op = operation.spelling
                out_key = f'{var}_{op}'
                if operation.stores_index:
                    out_tensors.append((out_key, "int", f"p_{safe_var}_{op}"))
                    arg_type = operation.outer.value
                    arg_k_str = "" if operation.k == 1 else str(operation.k)
                    aux_name = f"{var}_{arg_type}{arg_k_str or ''}_aux"
                    safe_aux = f"p_{safe_var}_{arg_type}{arg_k_str or ''}_aux"
                    if aux_name not in added_aux:
                        out_tensors.append((aux_name, ctype, safe_aux))
                        added_aux.add(aux_name)
                else:
                    out_tensors.append((out_key, ctype, f"p_{safe_var}_{op}"))

                # Inner state for compound ops
                if operation.inner is not None:
                    inner = operation.inner.value
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
            actual_shape = (
                self._statistics_lowering.by_name[var_2d].variable.actual_shape
            )
            n_levels_val = actual_shape[-1]

        # ---- Build MSL kernel signature with [[buffer(N)]] bindings ----
        arg_order = []  # track Python-side arg order: ('tensor', key) or ('scalar', name, msl_type)
        abi_fields = []

        indent = "    "

        # output_index buffer
        index_ctype = self._metal_dtype_str(output_index)
        abi_fields.append((f'device const {index_ctype}*', f'p_{safe_save}', False))
        arg_order.append(('tensor', output_index, 'read'))

        # input var buffers
        for inp in sorted_inputs:
            safe_inp = self._get_safe_name(inp)
            if safe_inp == safe_save:
                continue
            ctype = self._metal_dtype_str(inp)
            abi_fields.append((f'device const {ctype}*', f'p_{safe_inp}', False))
            arg_order.append(('tensor', inp, 'read'))

        # output/state buffers (read-write)
        for state_key, ctype, pname in out_tensors:
            abi_fields.append((f'device {ctype}*', pname, False))
            arg_order.append(('tensor', state_key, 'read_write'))

        # varying scalar params → device buffer pointers (avoids host-device sync)
        varying_scalar_params = [
            ('weight', 'float', '__weight'), ('total_weight', 'float', '__total_weight'),
            ('num_macro_steps', 'float', '__num_macro_steps'),
            ('sub_step', 'int', '__sub_step'), ('num_sub_steps', 'int', '__num_sub_steps'),
            ('flags', 'int', '__flags'),
            ('macro_step_index', 'int', '__macro_step_index'),
        ]
        for sname, stype, state_key in varying_scalar_params:
            abi_fields.append((f'device const {stype}*', f'p_{sname}_ptr', False))
            arg_order.append(('tensor', state_key, 'read'))

        # fixed scalar params (truly constant per-capture)
        fixed_scalar_params = [
            ('n_saved_points', 'int'), ('stride_input', 'int'),
        ]
        if has_2d:
            fixed_scalar_params.append(('n_levels', 'int'))

        for sname, stype in fixed_scalar_params:
            abi_fields.append((stype, sname, True))
            arg_order.append(('scalar', sname, stype))

        _emit_argument_kernel_start(msl_lines, kernel_name, abi_fields)

        # Bounds check
        msl_lines.append(f'{indent}if ((int)tid >= n_saved_points) return;')
        msl_lines.append(f'{indent}int idx = p_{safe_save}[tid];')
        msl_lines.append('')
        # Dereference varying scalar device pointers
        msl_lines.append(f'{indent}float weight = *p_weight_ptr;')
        msl_lines.append(f'{indent}float total_weight = *p_total_weight_ptr;')
        msl_lines.append(f'{indent}float num_macro_steps = *p_num_macro_steps_ptr;')
        msl_lines.append(f'{indent}int sub_step = *p_sub_step_ptr;')
        msl_lines.append(f'{indent}int num_sub_steps = *p_num_sub_steps_ptr;')
        msl_lines.append(f'{indent}int flags = *p_flags_ptr;')
        msl_lines.append(f'{indent}int macro_step_index = *p_macro_step_index_ptr;')
        msl_lines.append('')
        needed_bools = self._statistics_lowering.required_flags
        if needed_bools:
            msl_lines.append(f'{indent}// Compute boolean flags from sub_step, num_sub_steps, flags')
            if 'is_inner_first' in needed_bools:
                msl_lines.append(f'{indent}bool is_inner_first = (flags & 1) && (sub_step == 0);')
            if 'is_inner_last' in needed_bools:
                msl_lines.append(f'{indent}bool is_inner_last = ((flags >> 1) & 1) && (sub_step == num_sub_steps - 1);')
            if 'is_middle' in needed_bools:
                msl_lines.append(f'{indent}bool is_middle = (sub_step == num_sub_steps / 2);')
            if 'is_outer_first' in needed_bools:
                msl_lines.append(f'{indent}bool is_outer_first = ((flags >> 2) & 1) && is_inner_last;')
            if 'is_outer_last' in needed_bools:
                msl_lines.append(f'{indent}bool is_outer_last = ((flags >> 3) & 1) && is_inner_last;')
        msl_lines.append('')

        # Trial loop
        if num_trials > 1:
            msl_lines.append(f'{indent}for (int t = 0; t < {num_trials}; t++) {{')
            indent2 = indent + "    "
        else:
            msl_lines.append(f'{indent}const int t = 0;')
            indent2 = indent

        # -- 1D variables --
        if dims_1d:
            msl_lines.append(f'{indent2}// === 1D variables ===')
            emitted = set()
            for var in dims_1d:
                self._metal_emit_val_load(var, msl_lines, emitted, indent2,
                                          idx_expr="t * stride_input + idx" if num_trials > 1 else "idx")

            # Inner aggregation states for compound ops
            for reduction, inner_vars in (
                self._statistics_lowering.variables_by_inner(dims_1d).items()
            ):
                inner_type = reduction.value
                for var in inner_vars:
                    safe_var = self._get_safe_name(var)
                    var_val = f"{safe_var}_val"
                    ctype = self._metal_dtype_str(var)
                    val_for = f"val_for_{safe_var}_{inner_type}"
                    out_idx = "t * n_saved_points + tid"

                    if inner_type == 'last':
                        pass
                    elif inner_type == 'mean':
                        msl_lines.extend([
                            f'{indent2}{ctype} {val_for} = ({ctype})0;',
                            f'{indent2}{{',
                            f'{indent2}    {ctype} inner_old = p_{safe_var}_mean_inner_state[{out_idx}];',
                            f'{indent2}    {ctype} w_old = p_{safe_var}_mean_weight_state[{out_idx}];',
                            f'{indent2}    {ctype} inner_new = inner_old + {var_val} * ({ctype})weight;',
                            f'{indent2}    {ctype} w_new = w_old + ({ctype})weight;',
                            f'{indent2}    if (is_inner_last) {{',
                            f'{indent2}        p_{safe_var}_mean_inner_state[{out_idx}] = ({ctype})0;',
                            f'{indent2}        p_{safe_var}_mean_weight_state[{out_idx}] = ({ctype})0;',
                            f'{indent2}        {val_for} = inner_new / w_new;',
                            f'{indent2}    }} else {{',
                            f'{indent2}        p_{safe_var}_mean_inner_state[{out_idx}] = inner_new;',
                            f'{indent2}        p_{safe_var}_mean_weight_state[{out_idx}] = w_new;',
                            f'{indent2}    }}',
                            f'{indent2}}}',
                        ])
                    elif inner_type == 'sum':
                        msl_lines.extend([
                            f'{indent2}{ctype} {val_for} = ({ctype})0;',
                            f'{indent2}{{',
                            f'{indent2}    {ctype} inner_old = p_{safe_var}_sum_inner_state[{out_idx}];',
                            f'{indent2}    {ctype} inner_new = inner_old + {var_val} * ({ctype})weight;',
                            f'{indent2}    if (is_inner_last) {{',
                            f'{indent2}        p_{safe_var}_sum_inner_state[{out_idx}] = ({ctype})0;',
                            f'{indent2}        {val_for} = inner_new;',
                            f'{indent2}    }} else {{',
                            f'{indent2}        p_{safe_var}_sum_inner_state[{out_idx}] = inner_new;',
                            f'{indent2}    }}',
                            f'{indent2}}}',
                        ])
                    elif inner_type == 'max':
                        msl_lines.extend([
                            f'{indent2}{ctype} {val_for} = ({ctype})0;',
                            f'{indent2}{{',
                            f'{indent2}    {ctype} inner_old = p_{safe_var}_max_inner_state[{out_idx}];',
                            f'{indent2}    {ctype} inner_new = (is_inner_first && macro_step_index == 0) ? {var_val} : fmax(inner_old, {var_val});',
                            f'{indent2}    if (is_inner_last) {{',
                            f'{indent2}        p_{safe_var}_max_inner_state[{out_idx}] = ({ctype})(-1e38);',
                            f'{indent2}        {val_for} = inner_new;',
                            f'{indent2}    }} else {{',
                            f'{indent2}        p_{safe_var}_max_inner_state[{out_idx}] = inner_new;',
                            f'{indent2}    }}',
                            f'{indent2}}}',
                        ])
                    elif inner_type == 'min':
                        msl_lines.extend([
                            f'{indent2}{ctype} {val_for} = ({ctype})0;',
                            f'{indent2}{{',
                            f'{indent2}    {ctype} inner_old = p_{safe_var}_min_inner_state[{out_idx}];',
                            f'{indent2}    {ctype} inner_new = (is_inner_first && macro_step_index == 0) ? {var_val} : fmin(inner_old, {var_val});',
                            f'{indent2}    if (is_inner_last) {{',
                            f'{indent2}        p_{safe_var}_min_inner_state[{out_idx}] = ({ctype})1e38;',
                            f'{indent2}        {val_for} = inner_new;',
                            f'{indent2}    }} else {{',
                            f'{indent2}        p_{safe_var}_min_inner_state[{out_idx}] = inner_new;',
                            f'{indent2}    }}',
                            f'{indent2}}}',
                        ])
                    elif inner_type == 'first':
                        msl_lines.extend([
                            f'{indent2}{ctype} {val_for} = ({ctype})0;',
                            f'{indent2}if (is_inner_first) p_{safe_var}_first_inner_state[{out_idx}] = {var_val};',
                            f'{indent2}if (is_inner_last) {val_for} = p_{safe_var}_first_inner_state[{out_idx}];',
                        ])
                    elif inner_type == 'mid':
                        msl_lines.extend([
                            f'{indent2}{ctype} {val_for} = ({ctype})0;',
                            f'{indent2}if (is_middle) p_{safe_var}_mid_inner_state[{out_idx}] = {var_val};',
                            f'{indent2}if (is_inner_last) {val_for} = p_{safe_var}_mid_inner_state[{out_idx}];',
                        ])

            # Emit actual ops
            for var in dims_1d:
                safe_var = self._get_safe_name(var)
                var_val = f"{safe_var}_val"
                ctype = self._metal_dtype_str(var)
                operations = self._statistics_lowering.operations(var)
                out_idx = "t * n_saved_points + tid"

                for operation in operations:
                    op = operation.spelling

                    # ---- Compound ops ----
                    if operation.compound:
                        outer = operation.outer.value
                        inner = operation.inner.value
                        is_arg = operation.stores_index
                        outer_base = operation.outer.value

                        if inner == 'last':
                            val_var = var_val
                        else:
                            val_var = f"val_for_{safe_var}_{inner}"

                        msl_lines.append(f'{indent2}// Compound {op} for {safe_var}')
                        if is_arg:
                            arg_type = outer_base
                            cmp_op = ">" if arg_type == "max" else "<"
                            arg_k_str = "" if operation.k == 1 else str(operation.k)
                            aux_ptr = f"p_{safe_var}_{arg_type}{arg_k_str}_aux"
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{indent2}if (is_inner_last) {{',
                                f'{indent2}    if (is_outer_first) {{',
                                f'{indent2}        {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{indent2}        {aux_ptr}[{out_idx}] = {val_var};',
                                f'{indent2}    }} else {{',
                                f'{indent2}        {ctype} old_aux = {aux_ptr}[{out_idx}];',
                                f'{indent2}        if ({val_var} {cmp_op} old_aux) {{',
                                f'{indent2}            {aux_ptr}[{out_idx}] = {val_var};',
                                f'{indent2}            {out_ptr}[{out_idx}] = macro_step_index;',
                                f'{indent2}        }}',
                                f'{indent2}    }}',
                                f'{indent2}}}',
                            ])
                        elif outer_base in ('max', 'min'):
                            cmp_fn = "fmax" if outer_base == "max" else "fmin"
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{indent2}if (is_inner_last) {{',
                                f'{indent2}    if (is_outer_first) {{',
                                f'{indent2}        {out_ptr}[{out_idx}] = {val_var};',
                                f'{indent2}    }} else {{',
                                f'{indent2}        {out_ptr}[{out_idx}] = {cmp_fn}({out_ptr}[{out_idx}], {val_var});',
                                f'{indent2}    }}',
                                f'{indent2}}}',
                            ])
                        elif outer == 'mean':
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{indent2}if (is_inner_last) {{',
                                f'{indent2}    if (is_outer_first) {{',
                                f'{indent2}        {out_ptr}[{out_idx}] = {val_var};',
                                f'{indent2}    }} else {{',
                                f'{indent2}        {out_ptr}[{out_idx}] += {val_var};',
                                f'{indent2}    }}',
                                f'{indent2}    if (is_outer_last) {{',
                                f'{indent2}        {out_ptr}[{out_idx}] /= num_macro_steps;',
                                f'{indent2}    }}',
                                f'{indent2}}}',
                            ])
                        elif outer == 'sum':
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{indent2}if (is_inner_last) {{',
                                f'{indent2}    if (is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                                f'{indent2}    else {{ {out_ptr}[{out_idx}] += {val_var}; }}',
                                f'{indent2}}}',
                            ])
                        elif outer == 'last':
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{indent2}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                            ])
                        elif outer == 'first':
                            out_ptr = f"p_{safe_var}_{op}"
                            msl_lines.extend([
                                f'{indent2}if (is_inner_last && is_outer_first) {{ {out_ptr}[{out_idx}] = {val_var}; }}',
                            ])
                        continue

                    # ---- Simple ops ----
                    out_ptr = f"p_{safe_var}_{op}"
                    msl_lines.append(f'{indent2}// {op} for {safe_var}')

                    if op == 'mean':
                        msl_lines.extend([
                            f'{indent2}{{',
                            f'{indent2}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[{out_idx}];',
                            f'{indent2}    {ctype} new_val = old_val + {var_val} * ({ctype})weight;',
                            f'{indent2}    {out_ptr}[{out_idx}] = is_inner_last ? new_val / ({ctype})total_weight : new_val;',
                            f'{indent2}}}',
                        ])
                    elif op == 'sum':
                        msl_lines.extend([
                            f'{indent2}{{',
                            f'{indent2}    {ctype} old_val = is_inner_first ? ({ctype})0 : {out_ptr}[{out_idx}];',
                            f'{indent2}    {out_ptr}[{out_idx}] = old_val + {var_val} * ({ctype})weight;',
                            f'{indent2}}}',
                        ])
                    elif op == 'max':
                        msl_lines.extend([
                            f'{indent2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}',
                            f'{indent2}else {{ {out_ptr}[{out_idx}] = fmax({out_ptr}[{out_idx}], {var_val}); }}',
                        ])
                    elif op == 'min':
                        msl_lines.extend([
                            f'{indent2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}',
                            f'{indent2}else {{ {out_ptr}[{out_idx}] = fmin({out_ptr}[{out_idx}], {var_val}); }}',
                        ])
                    elif op == 'last':
                        msl_lines.append(f'{indent2}if (is_inner_last) {{ {out_ptr}[{out_idx}] = {var_val}; }}')
                    elif op == 'first':
                        msl_lines.append(f'{indent2}if (is_inner_first) {{ {out_ptr}[{out_idx}] = {var_val}; }}')
                    elif op == 'mid':
                        msl_lines.append(f'{indent2}if (is_middle) {{ {out_ptr}[{out_idx}] = {var_val}; }}')

        # -- 2D variables --
        if dims_2d:
            msl_lines.append(f'{indent2}// === 2D variables ===')
            for var in dims_2d:
                safe_var = self._get_safe_name(var)
                ctype = self._metal_dtype_str(var)
                for operation in self._statistics_lowering.operations(var):
                    op = operation.spelling
                    out_ptr = f"p_{safe_var}_{op}"
                    msl_lines.append(f'{indent2}for (int level = 0; level < n_levels; level++) {{')
                    idx_2d = "(t * stride_input + idx) * n_levels + level" if num_trials > 1 else "idx * n_levels + level"
                    out_2d = "(t * n_saved_points + tid) * n_levels + level"
                    msl_lines.append(f'{indent2}    {ctype} val_2d = p_{safe_var}[{idx_2d}];')
                    if op == 'mean':
                        msl_lines.extend([
                            f'{indent2}    {ctype} old_2d = is_inner_first ? ({ctype})0 : {out_ptr}[{out_2d}];',
                            f'{indent2}    {ctype} new_2d = old_2d + val_2d * ({ctype})weight;',
                            f'{indent2}    {out_ptr}[{out_2d}] = is_inner_last ? new_2d / ({ctype})total_weight : new_2d;',
                        ])
                    elif op == 'max':
                        msl_lines.extend([
                            f'{indent2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}',
                            f'{indent2}    else {{ {out_ptr}[{out_2d}] = fmax({out_ptr}[{out_2d}], val_2d); }}',
                        ])
                    elif op == 'min':
                        msl_lines.extend([
                            f'{indent2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}',
                            f'{indent2}    else {{ {out_ptr}[{out_2d}] = fmin({out_ptr}[{out_2d}], val_2d); }}',
                        ])
                    elif op == 'last':
                        msl_lines.append(f'{indent2}    if (is_inner_last) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    elif op == 'first':
                        msl_lines.append(f'{indent2}    if (is_inner_first) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    elif op == 'mid':
                        msl_lines.append(f'{indent2}    if (is_middle) {{ {out_ptr}[{out_2d}] = val_2d; }}')
                    msl_lines.append(f'{indent2}}}')

        if num_trials > 1:
            msl_lines.append(f'{indent}}}')
        msl_lines.append('}')
        msl_lines.append('')

        return {
            'kernel_name': kernel_name,
            'output_index': output_index,
            'sorted_inputs': sorted_inputs,
            'out_tensors': out_tensors,
            'has_2d': has_2d,
            'n_levels_val': n_levels_val,
            'arg_order': arg_order,
        }

    def _generate_metal_scatter_kernels(
        self: StatisticsRuntime,
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
                f"__scatter_cnt_{name}"
                if source.reduction.value == "mean" else None
            )
            target_size = int(self._storage[buf_key].shape[-1])
            source_size = int(self._tensor_registry[source.index].numel())
            num_trials = int(self.num_trials)
            total_target = target_size * num_trials
            total_source = source_size * num_trials

            zero_name = f"aggr_scatter_zero_{safe}"
            zero_fields = [("device float*", "p_buf", False)]
            zero_order = [("tensor", buf_key, "write")]
            if cnt_key is not None:
                zero_fields.append(("device float*", "p_cnt", False))
                zero_order.append(("tensor", cnt_key, "write"))
            zero_fields.append(("int", "total", True))
            zero_order.append(("scalar", "total", total_target))
            _emit_argument_kernel_start(msl_lines, zero_name, zero_fields)
            msl_lines.extend([
                "    if ((int)tid >= total) return;",
                "    p_buf[tid] = 0.0f;",
            ])
            if cnt_key is not None:
                msl_lines.append("    p_cnt[tid] = 0.0f;")
            msl_lines.extend(["}", ""])
            metas.append({
                "kernel_name": zero_name, "arg_order": zero_order,
                "grid_size": total_target,
            })

            add_name = f"aggr_scatter_add_{safe}"
            add_fields = [("device atomic_float*", "p_buf", False)]
            add_order = [("tensor", buf_key, "atomic_add")]
            if cnt_key is not None:
                add_fields.append(("device atomic_float*", "p_cnt", False))
                add_order.append(("tensor", cnt_key, "atomic_add"))
            index_ctype = self._metal_dtype_str(source.index)
            index_safe = self._get_safe_name(source.index)
            add_fields.append((f"device const {index_ctype}*", f"p_{index_safe}", False))
            add_order.append(("tensor", source.index, "read"))

            leaf_inputs = tuple(
                key for key in ir.scatter_inputs(name) if key != source.index
            )
            strides: dict[str, int] = {}
            for key in leaf_inputs:
                ctype = self._metal_dtype_str(key)
                key_safe = self._get_safe_name(key)
                add_fields.append((f"device const {ctype}*", f"p_{key_safe}", False))
                add_order.append(("tensor", key, "read"))
                tensor = self._tensor_registry.get(key)
                if tensor is None:
                    tensor = self._storage[key]
                strides[key] = (
                    int(tensor.shape[1])
                    if num_trials > 1 and tensor.ndim >= 2 else 0
                )
            for scalar, value in (
                ("source_size", source_size), ("target_size", target_size),
                ("total", total_source),
            ):
                add_fields.append(("int", scalar, True))
                add_order.append(("scalar", scalar, value))
            _emit_argument_kernel_start(msl_lines, add_name, add_fields)
            msl_lines.extend([
                "    if ((int)tid >= total) return;",
                "    int t = (int)tid / source_size;",
                "    int src = (int)tid - t * source_size;",
                f"    int dst = (int)p_{index_safe}[src];",
            ])

            emitted: dict[str, str] = {}
            def emit_value(field: str) -> str:
                previous = emitted.get(field)
                if previous is not None:
                    return previous
                field_source = ir.sources.get(field, TensorSource(field))
                value_name = f"v_{self._get_safe_name(field)}"
                if isinstance(field_source, ExpressionSource):
                    names = {
                        dependency: emit_value(dependency)
                        for dependency in field_source.expression.dependencies
                    }
                    expression = render_expression(
                        field_source.expression, ExpressionDialect.METAL, names,
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
                source.value, ExpressionDialect.METAL, names,
            )
            msl_lines.append(f"    float value = (float)({expression});")
            msl_lines.append(
                "    atomic_fetch_add_explicit(p_buf + t * target_size + dst, "
                "value, memory_order_relaxed);"
            )
            if cnt_key is not None:
                msl_lines.append(
                    "    atomic_fetch_add_explicit(p_cnt + t * target_size + dst, "
                    "1.0f, memory_order_relaxed);"
                )
            msl_lines.extend(["}", ""])
            metas.append({
                "kernel_name": add_name, "arg_order": add_order,
                "grid_size": total_source,
            })

            if cnt_key is not None:
                divide_name = f"aggr_scatter_divide_{safe}"
                divide_fields = [
                    ("device float*", "p_buf", False),
                    ("device atomic_float*", "p_cnt", False),
                    ("int", "total", True),
                ]
                divide_order = [
                    ("tensor", buf_key, "read_write"),
                    ("tensor", cnt_key, "read"),
                    ("scalar", "total", total_target),
                ]
                _emit_argument_kernel_start(msl_lines, divide_name, divide_fields)
                msl_lines.extend([
                    "    if ((int)tid >= total) return;",
                    "    float count = atomic_load_explicit(p_cnt + tid, memory_order_relaxed);",
                    "    if (count > 0.0f) p_buf[tid] /= count;",
                    "}", "",
                ])
                metas.append({
                    "kernel_name": divide_name, "arg_order": divide_order,
                    "grid_size": total_target,
                })
        return metas

    def _generate_metal_aggregator_function(
        self: StatisticsRuntime,
    ) -> None:
        """Generate and compile a Metal MSL aggregation kernel.

        Generates raw MSL kernels and compiles them through HydroForge's
        native Metal pipeline bridge.
        """

        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        grouped_by_output_index = self._statistics_lowering.groups

        msl_lines = [
            '// Auto-generated Metal aggregation kernels for hydroforge statistics',
            '#include <metal_stdlib>',
            'using namespace metal;',
            '',
        ]

        scatter_metas = self._generate_metal_scatter_kernels(
            msl_lines, self._statistics_ir,
        )
        group_metas = []
        for output_index, var_list in grouped_by_output_index.items():
            meta = self._generate_metal_kernel_for_group(
                msl_lines, output_index, var_list
            )
            group_metas.append(meta)

        msl_src = '\n'.join(msl_lines)

        from hydroforge.contracts import KernelSpec
        from hydroforge.kernels.registry import make_metal_dispatcher
        dispatchers = {}
        for meta in (*scatter_metas, *group_metas):
            arg_names = tuple(
                f"arg_{index}" for index in range(len(meta['arg_order']))
            )
            buffer_access = {
                arg_names[index]: rest[1]
                for index, (kind, *rest) in enumerate(meta['arg_order'])
                if kind == 'tensor'
            }
            runtime_scalars = {
                arg_names[index]: (
                    'float32' if rest[-1] in {'float', 'double'} else 'index'
                )
                for index, (kind, *rest) in enumerate(meta['arg_order'])
                if kind == 'scalar'
            }
            runtime_scalars['_grid_size'] = 'index'
            parameters = (*arg_names, '_grid_size')
            spec = KernelSpec(
                name=meta['kernel_name'], parameters=parameters,
                size_key='_grid_size', buffers=buffer_access,
                runtime_scalars=runtime_scalars,
            )
            dispatchers[meta['kernel_name']] = make_metal_dispatcher(
                msl_src,
                meta['kernel_name'],
                spec=spec,
            )

        # Build the Python wrapper
        def _make_wrapper(compiled, scatters, metas, grouped, self_ref):
            # Pre-compute stride_input for each group
            strides = {}
            for meta in metas:
                si = meta['output_index']
                if meta.get('full_output'):
                    strides[si] = 0
                    continue
                var_list = grouped[si]
                first_var = var_list[0]
                stride = 0
                for m in self_ref._metadata.values():
                    if m['original_variable'] == first_var:
                        stride = m.get('stride_input', 0)
                        break
                strides[si] = stride

            def internal_update_statistics(states, BLOCK_SIZE):
                del BLOCK_SIZE
                for meta in scatters:
                    dispatcher = compiled[meta['kernel_name']]
                    args = [
                        states[rest[0]] if kind == 'tensor' else rest[1]
                        for kind, *rest in meta['arg_order']
                    ]
                    dispatcher(**{
                        **{f'arg_{i}': value for i, value in enumerate(args)},
                        '_grid_size': meta['grid_size'],
                    })
                for meta in metas:
                    dispatcher = compiled[meta['kernel_name']]
                    si = meta['output_index']
                    if meta.get('full_output'):
                        n_threads = meta['n_elements_val']
                        n_saved = n_threads
                        stride = 0
                    else:
                        n_saved = len(states[si])
                        n_threads = n_saved
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
                            elif sname == 'n_elements':
                                args.append(meta['n_elements_val'])

                    dispatcher(**{
                        **{f'arg_{i}': value for i, value in enumerate(args)},
                        '_grid_size': n_threads,
                    })

            return internal_update_statistics

        self._aggregator_function = _make_wrapper(
            dispatchers, scatter_metas, group_metas,
            grouped_by_output_index, self,
        )
        self._aggregator_generated = True

        # Save for debugging
        if self.save_kernels:
            self._save_kernel_file(msl_src)

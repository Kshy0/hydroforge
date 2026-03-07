# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

import re
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator

import ast


class CommonCodegenMixin:
    """Shared utilities for all kernel codegen backends."""

    def _save_kernel_file(self: StatisticsAggregator, kernel_code: str) -> None:
        """
        Save the generated kernel code to a permanent file for inspection.
        
        Args:
            kernel_code: Generated kernel code as string
        """
        # Use unique name generation
        unique_name = self._generate_unique_name()
        filename = f"kern_{unique_name}.py"
        
        self._saved_kernel_file = self.kernels_dir / filename
        
        with open(self._saved_kernel_file, 'w', encoding='utf-8') as f:
            f.write(kernel_code)

    


    def _transform_pow_expr(self: StatisticsAggregator, expr: str) -> str:
        """
        Transform power operations in an expression string to Triton-compatible tl.exp(log()).
        Power operator ** or ^ is transformed.
        """
        if '**' not in expr and '^' not in expr:
            return expr
            
        safe_expr = expr.replace('^', '**')
        
        def _visit_and_transform_pow(node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    new_list = []
                    for item in value:
                        if isinstance(item, ast.AST):
                            new_list.append(_visit_and_transform_pow(item))
                        else:
                            new_list.append(item)
                    setattr(node, field, new_list)
                elif isinstance(value, ast.AST):
                    setattr(node, field, _visit_and_transform_pow(value))
            
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                return ast.Call(
                    func=ast.Attribute(value=ast.Name(id='libdevice', ctx=ast.Load()), attr='pow', ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[]
                )
            return node

        try:
            expr_tree = ast.parse(safe_expr, mode='eval')
            expr_tree = _visit_and_transform_pow(expr_tree)
            return ast.unparse(expr_tree)
        except Exception as e:
            print(f"Warning: Failed to transform power expression '{safe_expr}': {e}")
            return safe_expr



    def _analyze_needed_booleans(self: StatisticsAggregator) -> Set[str]:
        """Determine which boolean flags are needed by the configured ops.
        
        Returns a set of needed boolean names from:
        {'is_inner_first', 'is_inner_last', 'is_middle', 'is_outer_first', 'is_outer_last'}
        """
        needed = set()
        for ops in self._variable_ops.values():
            for op in ops:
                if '_' in op:
                    # Compound ops need inner_last + outer conditions
                    needed.update(['is_inner_last', 'is_outer_first', 'is_outer_last'])
                else:
                    m = re.match(r'(arg)?(max|min|sum|mean|first|last|mid)(\d*)', op)
                    if not m:
                        continue
                    op_type = m.group(2)
                    if op_type == 'first':
                        needed.update(['is_inner_first', 'is_inner_last'])
                    elif op_type == 'last':
                        needed.add('is_inner_last')
                    elif op_type == 'mid':
                        needed.update(['is_middle', 'is_inner_last'])
                    elif op_type in ('max', 'min', 'sum'):
                        needed.update(['is_inner_first', 'is_inner_last'])
                    elif op_type == 'mean':
                        needed.update(['is_inner_first', 'is_inner_last', 'is_outer_last'])
        return needed

    def _analyze_tensor_info(self: StatisticsAggregator):
        """Analyze tensor information and group variables by save_idx.
        
        Returns:
            (tensor_info, grouped_by_save_idx) tuple.
        """
        tensor_info = {}
        grouped_by_save_idx = {}
        
        for var_name in self._variables:
            field_info = self._field_registry[var_name]
            tensor = self._tensor_registry.get(var_name)
            
            if tensor is None:
                 first_op = self._variable_ops[var_name][0]
                 out_name = f"{var_name}_{first_op}"
                 meta = self._metadata[out_name]
                 tensor_info[var_name] = {
                    'tensor': None,
                    'tensor_shape': meta['tensor_shape'],
                    'actual_shape': meta['actual_shape'],
                    'actual_ndim': meta['actual_ndim']
                }
            else:
                json_schema_extra = getattr(field_info, 'json_schema_extra', {})
                tensor_shape = json_schema_extra.get('tensor_shape', ())
                tensor_info[var_name] = {
                    'tensor': tensor,
                    'tensor_shape': tensor_shape,
                    'actual_shape': tensor.shape,
                    'actual_ndim': tensor.ndim
                }
                
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            save_idx = json_schema_extra.get('save_idx')
            if save_idx not in grouped_by_save_idx:
                grouped_by_save_idx[save_idx] = []
            grouped_by_save_idx[save_idx].append(var_name)
        
        return tensor_info, grouped_by_save_idx


    # ========================================================================

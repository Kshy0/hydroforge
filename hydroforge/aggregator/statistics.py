# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Set

import torch

from hydroforge.modeling.distributed import torch_to_numpy_dtype

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator

from hydroforge.aggregator.scatter_expr import parse_scatter_expr


class StatisticsMixin:
    """Mixin providing statistics initialization, kernel state preparation, and update logic."""

    def _prepare_kernel_states(self: StatisticsAggregator) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        required_tensors: Dict[str, torch.Tensor] = {}

        def get_dependencies(expr: str) -> Set[str]:
            # Handle scatter expressions
            scatter = parse_scatter_expr(expr)
            if scatter:
                deps = set()
                # Add index var
                if scatter.index_var in self._tensor_registry:
                    deps.add(scatter.index_var)
                # Recursively resolve value expression tokens
                for token in scatter.value_tokens:
                    if token in self._tensor_registry:
                        deps.add(token)
                    elif token in self._field_registry:
                        f_info = self._field_registry[token]
                        cat = getattr(f_info, 'json_schema_extra', {}).get('category')
                        if cat == 'virtual':
                            sub = getattr(f_info, 'json_schema_extra', {}).get('expr')
                            if sub:
                                deps.update(get_dependencies(sub))
                return deps

            from hydroforge.aggregator.scatter_expr import \
                extract_tokens as _et
            tokens = _et(expr)
            deps = set()
            for token in tokens:
                if token in self._tensor_registry:
                    deps.add(token)
                elif token in self._field_registry:
                    f_info = self._field_registry[token]
                    cat = getattr(f_info, 'json_schema_extra', {}).get('category')
                    if cat == 'virtual':
                        sub = getattr(f_info, 'json_schema_extra', {}).get('expr')
                        if sub:
                            deps.update(get_dependencies(sub))
            return deps

        # Add original variables and their output buffers
        for var_name, ops in self._variable_ops.items():
            field_info = self._field_registry.get(var_name)
            json_extra = getattr(field_info, 'json_schema_extra', {})
            category = json_extra.get('category', 'param')
            
            if category == 'virtual':
                expr = json_extra.get('expr')
                if expr:
                    deps = get_dependencies(expr)
                    for dep in deps:
                        if dep in self._tensor_registry:
                            required_tensors[dep] = self._tensor_registry[dep]
            elif var_name in self._tensor_registry:
                required_tensors[var_name] = self._tensor_registry[var_name]

            for op in ops:
                out_name = f"{var_name}_{op}"
                required_tensors[out_name] = self._storage[out_name]
                
                # For explicit argmax/argmin operations, add their auxiliary storage
                op_parts = op.split('_')
                outer_op = op_parts[0]
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                if arg_match:
                    arg_type = arg_match.group(1)
                    arg_k_str = arg_match.group(2)
                    aux_name = f"{var_name}_{arg_type}{arg_k_str or ''}_aux"
                    if aux_name in self._storage:
                        required_tensors[aux_name] = self._storage[aux_name]
                
                # Add inner states for compound ops
                if '_' in op:
                    parts = op.split('_')
                    inner = parts[1]
                    # 'last' inner op doesn't need cross-step state
                    if inner != 'last':
                        inner_name = f"{var_name}_{inner}_inner_state"
                        if inner_name in self._storage:
                            required_tensors[inner_name] = self._storage[inner_name]
                        if inner == 'mean':
                            w_name = f"{var_name}_{inner}_weight_state"
                            if w_name in self._storage:
                                required_tensors[w_name] = self._storage[w_name]

        # Collect required dimensions and save indices
        required_dims: Set[str] = set()
        required_save_indices: Set[str] = set()
        for var_name in self._variables:
            field_info = self._field_registry[var_name]
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            tensor_shape = json_schema_extra.get('tensor_shape', ())
            save_idx = json_schema_extra.get('save_idx')
            if save_idx:
                required_save_indices.add(save_idx)
            for dim_name in tensor_shape:
                if isinstance(dim_name, str):
                    required_dims.add(dim_name)

        # Add scatter buffers and their source/index tensors
        for var_name, scatter in getattr(self, '_scatter_virtuals', {}).items():
            buf_key = f"__scatter_buf_{var_name}"
            if buf_key in self._storage:
                required_tensors[buf_key] = self._storage[buf_key]
            if scatter.mode == 'mean':
                cnt_key = f"__scatter_cnt_{var_name}"
                if cnt_key in self._storage:
                    required_tensors[cnt_key] = self._storage[cnt_key]
            # Ensure all scatter source tensors and index are in required_tensors
            if scatter.index_var in self._tensor_registry:
                required_tensors[scatter.index_var] = self._tensor_registry[scatter.index_var]
            for tok in scatter.value_tokens:
                if tok in self._tensor_registry:
                    required_tensors[tok] = self._tensor_registry[tok]

        # Add save_idx tensors
        for save_idx in required_save_indices:
            if save_idx in self._tensor_registry:
                required_tensors[save_idx] = self._tensor_registry[save_idx]
            else:
                raise RuntimeError(f"Save index tensor '{save_idx}' not registered")

        # Add dimension tensors/scalars
        for dim_name in required_dims:
            if dim_name in self._tensor_registry:
                tensor = self._tensor_registry[dim_name]
                if isinstance(tensor, (int, float)):
                    required_tensors[dim_name] = torch.tensor(tensor, device=self.device)
                else:
                    required_tensors[dim_name] = tensor

        self._kernel_states = required_tensors

        # Scalar parameters as 1-element device tensors for CUDA Graph compatibility.
        # Kernel code loads these via tl.load (Triton) or reads from states dict,
        # so CUDA Graphs can replay without recapture when values change.
        self._kernel_states['__weight'] = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._kernel_states['__total_weight'] = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._kernel_states['__num_macro_steps'] = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._kernel_states['__sub_step'] = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._kernel_states['__num_sub_steps'] = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._kernel_states['__flags'] = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._kernel_states['__macro_step_index'] = torch.zeros(1, device=self.device, dtype=torch.int32)

    

    def initialize_statistics(self: StatisticsAggregator, variable_ops: Dict[str, List[str]]) -> None:
        """Initialize aggregation tensors and metadata for provided variables and ops."""
        # Reset generic state
        self._variables = set()
        # Normalize to lower-case list for each variable
        self._variable_ops = {}
        for var, ops in variable_ops.items():
            if ops is None:
                ops_list = ["mean"]
            elif isinstance(ops, str):
                ops_list = [ops]
            else:
                ops_list = list(ops)
            self._variable_ops[var] = [str(o).lower() for o in ops_list]
        self._storage.clear()
        self._output_keys = []
        self._metadata.clear()
        self._output_is_outer: Dict[str, bool] = {}
        
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states = None
        self._current_macro_step_count = 0.0
        self._outer_flags_ever_seen = False
        
        # Scatter virtual metadata: var_name -> ScatterExpr
        self._scatter_virtuals: Dict[str, Any] = {}

        # Clean up old temporary files
        self._cleanup_temp_files()

        # Validate and setup each variable
        for var_name, ops in self._variable_ops.items():
            import re

            # Sort ops to ensure consistent processing order
            ops.sort()
            
            tensor = None
            field_info = self._field_registry[var_name]
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            category = json_schema_extra.get('category', 'param')
            is_virtual = category == 'virtual'

            if var_name in self._tensor_registry:
                tensor = self._tensor_registry[var_name]
            elif is_virtual:
                tensor = None
            else:
                raise ValueError(f"Variable '{var_name}' not registered. Call register_tensor() first.")
            
            target_dtype = tensor.dtype if tensor is not None else torch.float32

            tensor_shape = json_schema_extra.get('tensor_shape', ())
            save_idx = json_schema_extra.get('save_idx')
            description = getattr(field_info, 'description', f"Variable {var_name}")
            save_coord = json_schema_extra.get('save_coord')
            dim_coords = json_schema_extra.get('dim_coords')

            if not save_idx:
                raise ValueError(f"Variable '{var_name}' must have save_idx in json_schema_extra")

            if save_idx in self._tensor_registry:
                ref_save_idx = self._tensor_registry[save_idx]
                if tensor is not None:
                     # Real tensor shape/dim logic
                     tensor_ndim = tensor.ndim
                     tensor_base_shape = tensor.shape
                else:
                     # Virtual tensor logic
                     # Infer ndim/shape from tensor_shape or dependencies
                     tensor_ndim = 1 + (1 if self.num_trials > 1 else 0) # Base guess
                     if len(tensor_shape) > 1: # Has extra dims
                          tensor_ndim += (len(tensor_shape) - 1)
                     
                     # Construct hypothetical shape for allocation size
                     tensor_base_shape = (len(ref_save_idx),) # minimum
                     if len(tensor_shape) > 1:
                           # Try to resolve dimensions from dependencies or registry
                           expr = json_schema_extra.get('expr')
                           from hydroforge.aggregator.scatter_expr import \
                               extract_tokens as _et
                           toks = _et(expr)
                           found_dep = False
                           for t in toks:
                                if t in self._tensor_registry:
                                     dep = self._tensor_registry[t]
                                     tensor_ndim = dep.ndim
                                     tensor_base_shape = dep.shape
                                     target_dtype = dep.dtype
                                     found_dep = True
                                     break
                           if not found_dep:
                                # Try to resolve dimensions from registry
                                try:
                                    extra_dims = []
                                    # tensor_shape[0] is the grid dimension, skip it
                                    # tensor_shape[1:] are the extra dimensions (e.g. levels)
                                    for dim_name in tensor_shape[1:]:
                                         if dim_name in self._tensor_registry:
                                              d_val = self._tensor_registry[dim_name]
                                              if d_val.numel() == 1:
                                                   extra_dims.append(int(d_val.item()))
                                              else:
                                                   raise ValueError
                                         else:
                                              raise ValueError
                                    
                                    # Construct a fake base shape that satisfies the slicing logic below
                                    # The logic below uses [2:] (if trials) or [1:] (if no trials)
                                    # to get the EXTRA dims.
                                    prefix_len = 2 if self.num_trials > 1 else 1
                                    tensor_base_shape = (1,) * prefix_len + tuple(extra_dims)
                                    target_dtype = torch.float32

                                except ValueError:
                                     raise ValueError(f"Virtual variable '{var_name}' has multi-dimensional shape {tensor_shape}. Dependencies not found in registry, and dimensions could not be resolved directly.")

                if self.num_trials > 1:
                    actual_shape = (self.num_trials, len(ref_save_idx)) + tensor_base_shape[2:] if tensor_ndim > 1 else (self.num_trials, len(ref_save_idx))
                else:
                    actual_shape = (len(ref_save_idx),) + tensor_base_shape[1:]
            else:
                raise ValueError(f"Save index '{save_idx}' not registered in tensor registry")
            
            actual_ndim = tensor_ndim
            max_ndim = 3 if self.num_trials > 1 else 2
            if actual_ndim > max_ndim:
                raise ValueError(f"Variable '{var_name}' has {actual_ndim} actual dimensions. Only up to {max_ndim}D variables are supported.")

            is_2d = (self.num_trials > 1 and actual_ndim == 3) or (self.num_trials == 1 and actual_ndim == 2)
            if is_2d and any(op.split('_')[0] in ['max', 'min'] or re.match(r'(max|min)\d+$', op.split('_')[0]) for op in ops):
                raise ValueError(f"max/min operations are not supported for 2D variable '{var_name}' (with levels).")

            # Track
            self._variables.add(var_name)

            # Detect scatter virtual and allocate materialized buffer
            if is_virtual:
                expr = json_schema_extra.get('expr')
                scatter = parse_scatter_expr(expr) if expr else None
                if scatter:
                    self._scatter_virtuals[var_name] = scatter
                    # Allocate a scatter buffer in the FULL target dimension
                    # (not num_saved, but the full grid/cell count).
                    # The kernel will index into it via save_idx just like any real tensor.
                    # Determine full target size from index tensor (max + 1) or from
                    # other real tensors sharing the same save_idx.
                    idx_tensor = self._tensor_registry.get(scatter.index_var)
                    if idx_tensor is not None:
                        full_target_size = int(idx_tensor.max().item()) + 1
                    else:
                        # Fallback: use save_idx length (approximate)
                        full_target_size = len(ref_save_idx)
                    
                    scatter_buf_key = f"__scatter_buf_{var_name}"
                    if self.num_trials > 1:
                        buf_shape = (self.num_trials, full_target_size)
                    else:
                        buf_shape = (full_target_size,)
                    self._storage[scatter_buf_key] = torch.zeros(
                        buf_shape, dtype=target_dtype, device=self.device
                    )
                    if scatter.mode == 'mean':
                        scatter_cnt_key = f"__scatter_cnt_{var_name}"
                        self._storage[scatter_cnt_key] = torch.zeros(
                            buf_shape, dtype=target_dtype, device=self.device
                        )

            for op in ops:
                out_name = f"{var_name}_{op}"
                
                # Parse op parts
                op_parts = op.split('_')
                outer_op = op_parts[0]
                
                # Check for K in max/min ops (e.g., max3, min3)
                k_val = 1
                match_k = re.match(r'(max|min)(\d+)$', outer_op)
                if match_k:
                    outer_base = match_k.group(1)
                    k_val = int(match_k.group(2))
                    outer_op = outer_base # normalize for allocation logic below (mostly)
                
                # Check for explicit argmax/argmin operators (e.g., argmax, argmax3)
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                arg_k_val = 1  # Default for arg ops
                if arg_match:
                    arg_k_str = arg_match.group(2)
                    arg_k_val = int(arg_k_str) if arg_k_str else 1
                
                # Reject standalone topK / argTopK ops — they must be compound
                # e.g. "max3" alone is invalid; use "max3_last", "max3_max", etc.
                is_standalone = len(op_parts) == 1
                if is_standalone and k_val > 1:
                    raise ValueError(
                        f"Standalone top-k op '{op}' is not allowed. "
                        f"Use a compound form like '{op}_last' or '{op}_max' instead."
                    )
                if is_standalone and arg_match and arg_k_val > 1:
                    raise ValueError(
                        f"Standalone argTopK op '{op}' is not allowed. "
                        f"Use a compound form like '{op}_mean' or '{op}_last' instead."
                    )
                
                # Allocate storage by op
                if k_val > 1:
                    alloc_shape = actual_shape + (k_val,)
                else:
                    alloc_shape = actual_shape

                if arg_match:
                    # Explicit argmax/argmin operator - store integer indices only
                    arg_type = arg_match.group(1)  # 'max' or 'min'
                    arg_k_str = arg_match.group(2)  # '' or '3' etc
                    # arg_k_val already computed above
                    
                    if arg_k_val > 1:
                        arg_alloc_shape = actual_shape + (arg_k_val,)
                    else:
                        arg_alloc_shape = actual_shape
                    
                    # Store integer indices (macro step index within the window)
                    init_tensor = torch.zeros(arg_alloc_shape, dtype=torch.int32, device=self.device)
                    # Also need to track the corresponding extreme values for comparison
                    aux_name = f"{var_name}_{arg_type}{arg_k_str or ''}_aux"
                    if arg_type == 'max':
                        self._storage[aux_name] = torch.full(arg_alloc_shape, -torch.inf, dtype=target_dtype, device=self.device)
                    else:
                        self._storage[aux_name] = torch.full(arg_alloc_shape, torch.inf, dtype=target_dtype, device=self.device)
                    # aux is not an output, just internal state
                elif outer_op == 'max':
                    # max or maxK - NO automatic argmax
                    init_tensor = torch.full(alloc_shape, -torch.inf, dtype=target_dtype, device=self.device)
                elif outer_op == 'min':
                    # min or minK - NO automatic argmin
                    init_tensor = torch.full(alloc_shape, torch.inf, dtype=target_dtype, device=self.device)
                elif outer_op == 'first':
                    # Similar to 'last', we just need storage. Zero initialization is fine as it will be overwritten on is_first.
                    init_tensor = torch.zeros(alloc_shape, dtype=target_dtype, device=self.device)
                else:
                    init_tensor = torch.zeros(alloc_shape, dtype=target_dtype, device=self.device)
                self._storage[out_name] = init_tensor
                self._output_keys.append(out_name)

                # For compound ops, allocate inner state
                if len(op_parts) > 1:
                    inner_op = op_parts[1]
                    # 'last' inner op doesn't need cross-step state - it directly uses current value
                    needs_inner_state = inner_op != 'last'
                    inner_state_name = f"{var_name}_{inner_op}_inner_state"
                    
                    if needs_inner_state and inner_state_name not in self._storage:
                         # Initialize inner state
                         if inner_op == 'mean':
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         elif inner_op == 'max':
                             init_inner = torch.full(actual_shape, -torch.inf, dtype=target_dtype, device=self.device)
                         elif inner_op == 'min':
                             init_inner = torch.full(actual_shape, torch.inf, dtype=target_dtype, device=self.device)
                         elif inner_op == 'sum':
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         elif inner_op in ('first', 'mid'):
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         else:
                             raise ValueError(f"Unsupported inner op '{inner_op}'.")
                         self._storage[inner_state_name] = init_inner
                         
                         # Allocate weight state only for inner ops that need it (mean)
                         if inner_op == 'mean':
                             weight_state_name = f"{var_name}_{inner_op}_weight_state"
                             if weight_state_name not in self._storage:
                                 self._storage[weight_state_name] = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)

                if save_coord and save_coord not in self._coord_cache:
                    coord_tensor = self._tensor_registry[save_coord]
                    self._coord_cache[save_coord] = coord_tensor.detach().cpu().numpy()
                
                # Downcast to save_precision if specified (e.g. float64 -> float32)
                save_dtype = target_dtype
                if self.save_precision is not None and target_dtype.is_floating_point:
                    save_dtype = self.save_precision
                out_dtype = torch_to_numpy_dtype(save_dtype)
                
                # Check if this is an argmax/argmin op and determine the k value
                is_arg_op = arg_match is not None
                # For arg ops, use arg_k_val; otherwise use k_val
                effective_k = arg_k_val if is_arg_op else k_val

                # Determine stride_input and scatter metadata
                scatter_info = None
                if is_virtual:
                    expr = json_schema_extra.get('expr')
                    scatter = parse_scatter_expr(expr) if expr else None
                    if scatter:
                        # For scatter virtuals, the kernel reads from the materialized
                        # scatter buffer which has target (cell) dimension, not source.
                        # stride_input must match the scatter buffer's dim-1 size.
                        buf_key = f"__scatter_buf_{var_name}"
                        scatter_buf = self._storage.get(buf_key)
                        if scatter_buf is not None and self.num_trials > 1:
                            stride_input = scatter_buf.shape[-1]
                        else:
                            stride_input = 0
                        scatter_info = {
                            'mode': scatter.mode,
                            'value_expr': scatter.value_expr,
                            'index_var': scatter.index_var,
                            'source_size': self._tensor_registry[scatter.index_var].shape[0] if scatter.index_var in self._tensor_registry else 0,
                        }
                    else:
                        stride_input = 0
                        # For plain virtuals, try to get stride from deps
                        if expr:
                            from hydroforge.aggregator.scatter_expr import \
                                extract_tokens as _et
                            toks = _et(expr)
                            for t in toks:
                                if t in self._tensor_registry:
                                    dep = self._tensor_registry[t]
                                    if self.num_trials > 1 and dep.ndim >= 2:
                                        stride_input = dep.shape[1]
                                    break
                else:
                    stride_input = tensor.shape[1] if tensor is not None and self.num_trials > 1 else 0

                meta = {
                    'original_variable': var_name,
                    'op': op,
                    'save_idx': save_idx,
                    'tensor_shape': tensor_shape,
                    'dtype': 'i4' if is_arg_op else out_dtype,  # int32 for arg ops
                    'actual_shape': actual_shape,
                    'actual_ndim': actual_ndim,
                    'save_coord': save_coord,
                    'nc_coord_name': dim_coords.split('.')[-1] if dim_coords else None,
                    'description': f"{description} ({op})",
                    'stride_input': stride_input,
                    'k': effective_k,
                    'is_time_index': is_arg_op,  # argmax/argmin store integer indices
                    'scatter': scatter_info,  # None for non-scatter, dict for scatter virtuals
                }
                self._metadata[out_name] = meta
                
                # Classify as outer if it is a compound op (e.g. max_mean)
                self._output_is_outer[out_name] = len(op_parts) > 1

        self._has_compound_ops = any(self._output_is_outer.values())

        # Generate kernels and prepare states for all requested variables/ops
        self._generate_aggregator_function()
        self._prepare_kernel_states()

    

    def update_statistics(self: StatisticsAggregator, sub_step: int, num_sub_steps: int, flags: int,
                          weight: float, total_weight: float = 0.0, 
                          BLOCK_SIZE: int = 128) -> None:
        if not self._aggregator_generated:
            raise RuntimeError("Statistics aggregation not initialized. Call initialize_streaming_aggregation() first.")
        
        # Compute boolean flags from sub_step, num_sub_steps, flags
        # flags bits: 0=stat_is_first, 1=stat_is_last, 2=stat_is_outer_first, 3=stat_is_outer_last
        is_inner_last = bool(flags & 2) and (sub_step == num_sub_steps - 1)
        is_outer_first = bool(flags & 4) and is_inner_last
        is_outer_last = bool(flags & 8) and is_inner_last
        
        # Reset macro_step_index at the start of each outer statistics period
        # This ensures argmax/argmin indices are always relative to the start of the period
        if is_outer_first:
            self._macro_step_index = 0
            self._current_macro_step_count = 0.0
            self._outer_flags_ever_seen = True
            
        if is_inner_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We only trigger dirty for non-outer (Standard) ops when inner loop ends
                 if not is_outer:
                     self._dirty_outputs.add(out_name)
        
        if is_outer_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We trigger dirty for outer ops when outer loop ends
                 if is_outer:
                     self._dirty_outputs.add(out_name)

        if is_inner_last:
            self._current_macro_step_count += 1.0

        num_macro_steps = self._current_macro_step_count

        # Fill scalar tensors so kernels read updated values from fixed addresses
        states = self._kernel_states
        states['__weight'].fill_(weight)
        states['__total_weight'].fill_(total_weight)
        states['__num_macro_steps'].fill_(num_macro_steps)
        states['__sub_step'].fill_(sub_step)
        states['__num_sub_steps'].fill_(num_sub_steps)
        states['__flags'].fill_(flags)
        states['__macro_step_index'].fill_(self._macro_step_index)

        self._aggregator_function(states, BLOCK_SIZE)

    

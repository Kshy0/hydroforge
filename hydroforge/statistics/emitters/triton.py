# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from math import prod
from typing import TypedDict

from hydroforge.statistics.emitters.common import StatisticsEmitter
from hydroforge.statistics.ir import (
    ExpressionDialect,
    ExpressionSource,
    Reduction,
    ScatterSource,
    TensorSource,
    render_expression,
)

_FULL_OUTPUT_GROUP = "__full__"
_FULL_OUTPUT_KERNEL = "hydroforge_full_output_kernel"
_FULL_OUTPUT_GRID = "hydroforge_full_output_grid"


class _TopKOperation(TypedDict):
    var: str
    op: str
    k: int
    val_var: str
    out_offset: str
    type: str


@dataclass(slots=True)
class _OperationGroups:
    """Rendered statements grouped by the lowering's update conditions."""

    unconditional: list[str] = field(default_factory=list)
    is_inner_first: list[str] = field(default_factory=list)
    not_is_inner_first: list[str] = field(default_factory=list)
    is_inner_last: list[str] = field(default_factory=list)
    is_inner_last_is_outer_first: list[str] = field(default_factory=list)
    maxk_ops: list[_TopKOperation] = field(default_factory=list)
    argmaxk_ops: list[_TopKOperation] = field(default_factory=list)
    is_inner_last_not_is_outer_first: list[str] = field(default_factory=list)


class TritonStatisticsEmitter(StatisticsEmitter):
    """Triton JIT kernel code generation for statistics aggregation."""

    def emit(self):
        self._generate_triton_aggregator_function()
        return self.result()

    def _triton_expression(
        self,
        name: str,
        expression,
        names: dict[str, str],
    ) -> str:
        dtype = self._statistics_layouts[name].dtype
        value_type = {
            "torch.float32": "float32",
            "torch.float64": "float64",
        }[str(dtype)]
        return render_expression(
            expression,
            ExpressionDialect.TRITON,
            names,
            value_type=value_type,
        )

    def _generate_triton_aggregator_function(self) -> None:
        groups = self._statistics_lowering.groups
        lines = self._generate_kernel_header()
        self._generate_scatter_kernels(lines)
        for output_index, variables in groups.items():
            if output_index == _FULL_OUTPUT_GROUP:
                self._generate_full_kernel_for_group(
                    lines,
                    output_index,
                    variables,
                )
            else:
                safe_output_index = self._get_safe_name(output_index)
                self._generate_kernel_for_group(
                    lines,
                    f"kernel_{safe_output_index}",
                    output_index,
                    variables,
                )
        self._generate_main_function(lines, groups)
        source = "\n".join(lines)
        self._compile_generated_kernels(source)
        if self.save_kernels:
            self._save_kernel_file(source)

    def _generate_kernel_header(self) -> list[str]:
        """Generate the header for the kernel file with documentation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        var_list = sorted(self._variables)

        header = [
            '"""',
            "Auto-generated Triton kernels for hydroforge statistics aggregation (mean/max/min/last)",
            f"Generated at: {timestamp}",
            f"Rank: {self.rank}",
            f"Variables: {', '.join(var_list)}",
            f"Device: {self.device}",
            "",
            "Kernel Logic:",
            "- Load output_index values to get original grid indices",
            "- Use idx to access original data: data[idx]",
            "- Store outputs using sequential indexing: out[offs]",
            "- explicit argmax/argmin ops store the macro-step index",
            "- argmax/argmin store the macro-step index as int64;",
            "  conversion to datetime (if any) happens at the consumer via the",
            "  recorded macro-step time mapping, not at NC file write time",
            "",
            "Optimizations Applied:",
            "- tl.static_range for compile-time loop unrolling (ensemble_size, bubble sort)",
            "- Base offset precomputation (shared across max/min/argmax/argmin for same var+K)",
            "- Merged maxK+minK bubble insert in single loop with shared offset",
            "- Precise mask for tl.store: mask & swap_mask to reduce write pressure",
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
            "# ============================================================================",
            f"# Generated Triton kernels for statistics aggregation - Rank {self.rank}",
            "# ============================================================================",
            "",
        ]
        return header

    def _generate_scatter_kernels(
        self,
        kernel_code_lines: list[str],
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
        kernel_code_lines.append("# Triton scatter pre-step kernels")
        kernel_code_lines.append(
            "# ======================================================================"
        )
        kernel_code_lines.append("")

        for var_name, scatter in scatter_virtuals.items():
            safe_var = self._get_safe_name(var_name)
            buf_safe = self._get_safe_name(f"__scatter_buf_{var_name}")
            is_mean = scatter.reduction.value == "mean"

            # ── 1. Zero kernel ──
            kernel_code_lines.append("@triton.jit")
            if is_mean:
                cnt_safe = self._get_safe_name(f"__scatter_cnt_{var_name}")
                kernel_code_lines.append(
                    f"def scatter_zero_{safe_var}("
                    f"{buf_safe}_ptr, {cnt_safe}_ptr, "
                    f"N, BLOCK_SIZE: tl.constexpr, ensemble_size: tl.constexpr):"
                )
            else:
                kernel_code_lines.append(
                    f"def scatter_zero_{safe_var}("
                    f"{buf_safe}_ptr, "
                    f"N, BLOCK_SIZE: tl.constexpr, ensemble_size: tl.constexpr):"
                )
            kernel_code_lines.extend(
                [
                    "    pid = tl.program_id(0)",
                    "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
                    "    mask = offs < N",
                    "    for t in tl.static_range(ensemble_size):",
                    f"        tl.store({buf_safe}_ptr + t * N + offs, 0.0, mask=mask)",
                ]
            )
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
            sig_parts.extend(
                [
                    "M",
                    "N",
                    "BLOCK_SIZE: tl.constexpr",
                    "ensemble_size: tl.constexpr",
                ]
            )
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
            kernel_code_lines.extend(
                [
                    "    pid = tl.program_id(0)",
                    "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
                    "    mask = offs < M",
                    f"    idx = tl.load({idx_safe}_ptr + offs, mask=mask, other=0).to(tl.int64)",
                    "    mask = mask & (idx >= 0) & (idx < N)",
                    "    for t in tl.static_range(ensemble_size):",
                ]
            )
            emitted_values: dict[str, str] = {}

            def emit_scatter_value(name: str) -> str:
                previous = emitted_values.get(name)
                if previous is not None:
                    return previous
                source = self._statistics_ir.sources.get(name) or TensorSource(name)
                safe_name = self._get_safe_name(name)
                value_name = f"{safe_name}_val"
                if isinstance(source, ExpressionSource):
                    names = {
                        dependency: emit_scatter_value(dependency)
                        for dependency in source.expression.dependencies
                    }
                    expression = self._triton_expression(
                        name,
                        source.expression,
                        names,
                    )
                    kernel_code_lines.append(f"        {value_name} = {expression}")
                else:
                    key = (
                        f"__scatter_buf_{name}"
                        if isinstance(source, ScatterSource)
                        else source.name
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
                var_name,
                scatter.value,
                value_names,
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
                    f"N, BLOCK_SIZE: tl.constexpr, ensemble_size: tl.constexpr):"
                )
                kernel_code_lines.extend(
                    [
                        "    pid = tl.program_id(0)",
                        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
                        "    mask = offs < N",
                        "    for t in tl.static_range(ensemble_size):",
                        f"        _cnt = tl.load({cnt_safe}_ptr + t * N + offs, mask=mask, other=1)",
                        f"        _val = tl.load({buf_safe}_ptr + t * N + offs, mask=mask, other=0.0)",
                        "        _mean = tl.where(_cnt > 0, _val / _cnt, float('nan'))",
                        f"        tl.store({buf_safe}_ptr + t * N + offs, _mean, mask=mask)",
                    ]
                )
                kernel_code_lines.append("")
        kernel_code_lines.append("")

    def _emit_value(
        self,
        name: str,
        lines: list[str],
        emitted: set[str],
        *,
        indent: str,
        offset: Callable[[str], str],
        mask: str = "mask",
    ) -> str:
        safe_name = self._get_safe_name(name)
        value = f"{safe_name}_val"
        if name in emitted:
            return value
        source = self._statistics_ir.sources.get(name) or TensorSource(name)
        if isinstance(source, ExpressionSource):
            names = {
                dependency: self._emit_value(
                    dependency, lines, emitted, indent=indent, offset=offset, mask=mask
                )
                for dependency in source.expression.dependencies
            }
            expression = self._triton_expression(name, source.expression, names)
            lines.append(f"{indent}{value} = {expression}")
        else:
            key = (
                f"__scatter_buf_{name}"
                if isinstance(source, ScatterSource)
                else source.name
            )
            pointer = self._get_safe_name(key)
            lines.append(
                f"{indent}{value} = tl.load({pointer}_ptr + {offset(key)}, mask={mask}, other=0.0)"
            )
        emitted.add(name)
        return value

    def _generate_1d_vars_grouped(
        self,
        kernel_code_lines: list[str],
        dims_1d: list[str],
        indent: str,
        indent2: str,
        indent3: str,
        indent4: str,
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
            name
            for name in dims_1d
            if self._statistics_lowering.by_name[name].needs_unconditional_value
        }

        # Materialize dependencies used by unconditional operations first.
        def _collect_unconditional(name: str, acc: set[str]) -> None:
            if name in acc:
                return
            acc.add(name)
            source = self._statistics_ir.sources.get(name) or TensorSource(name)
            if isinstance(source, ExpressionSource):
                for dependency in source.expression.dependencies:
                    _collect_unconditional(dependency, acc)

        unconditional_names: set[str] = set()
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
                if scope is None
                else emitted_by_scope.setdefault(scope, set())
            )
            if (
                safe_v_name in emitted_vars
                or safe_v_name in scoped
                or (scope is not None and safe_v_name in unconditional_safe_names)
            ):
                return f"{safe_v_name}_val"

            source = self._statistics_ir.sources.get(v_name) or TensorSource(v_name)
            if isinstance(source, TensorSource):
                # Real data (includes virtual source buffers)
                stride = self._source_stride(source.name)
                in_ptr_loc = f"{safe_v_name}_ptr + t * {stride} + idx"
                to_lines.append(
                    f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)"
                )
            elif isinstance(source, ScatterSource):
                buf_safe = self._get_safe_name(f"__scatter_buf_{v_name}")
                stride = self._source_stride(f"__scatter_buf_{v_name}")
                in_ptr_loc = f"{buf_safe}_ptr + t * {stride} + idx"
                to_lines.append(
                    f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)"
                )
            elif isinstance(source, ExpressionSource):
                names = {
                    dependency: emit_val(dependency, to_lines, scope)
                    for dependency in source.expression.dependencies
                }
                expression = self._triton_expression(
                    v_name,
                    source.expression,
                    names,
                )
                to_lines.append(f"{indent}{safe_v_name}_val = {expression}")

            scoped.add(safe_v_name)
            return f"{safe_v_name}_val"

        # Phase 2: Collect all operations grouped by condition
        groups = _OperationGroups()

        # Special storage for maxK/minK operations (need for loop)

        # Track which inner aggregations are needed
        inner_aggregations_needed = self._statistics_lowering.variables_by_inner(
            dims_1d
        )

        self._collect_scalar_updates(
            dims_1d=dims_1d,
            emit_val=emit_val,
            groups=groups,
            vars_conditional_only=vars_conditional_only,
        )

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
        self._emit_inner_reduction_updates(
            indent=indent,
            inner_aggregations_needed=inner_aggregations_needed,
            kernel_code_lines=kernel_code_lines,
        )
        # Phase 5: Emit unconditional ops
        for line in groups.unconditional:
            kernel_code_lines.append(f"{indent}{line}")

        # Phase 6: Emit grouped conditional blocks
        if groups.is_inner_first:
            kernel_code_lines.append(f"{indent}if is_inner_first:")
            for line in groups.is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")

        if groups.not_is_inner_first:
            if groups.is_inner_first:
                kernel_code_lines.append(f"{indent}else:")
            else:
                kernel_code_lines.append(f"{indent}if not is_inner_first:")
            for line in groups.not_is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")

        self._emit_window_results(
            dims_1d=dims_1d,
            emit_val=emit_val,
            groups=groups,
            indent=indent,
            indent2=indent2,
            indent3=indent3,
            indent4=indent4,
            inner_aggregations_needed=inner_aggregations_needed,
            kernel_code_lines=kernel_code_lines,
            vars_conditional_only=vars_conditional_only,
        )

        kernel_code_lines.append("")

    def _group_pointer_arguments(
        self, output_index: str, variables: list[str]
    ) -> dict[str, str]:
        """Map each generated pointer name to its state key, in ABI order."""
        params: dict[str, str] = {}
        if output_index != _FULL_OUTPUT_GROUP:
            params[f"{self._get_safe_name(output_index)}_ptr"] = output_index
        for name in sorted(
            {
                name
                for var in variables
                for name in self._statistics_ir.materialized_inputs(var)
            }
        ):
            params[f"{self._get_safe_name(name)}_ptr"] = name
        for var in variables:
            safe_var = self._get_safe_name(var)
            operations = self._statistics_lowering.operations(var)

            def add(suffix: str) -> None:
                params[f"{safe_var}_{suffix}_ptr"] = f"{var}_{suffix}"

            for operation in operations:
                add(operation.spelling)
                if operation.stores_index:
                    add(f"{operation.spelling}_aux")
            if any(
                op.inner is None and op.output is Reduction.MEAN for op in operations
            ):
                add("mean_sample_weight_state")
            for operation in operations:
                if (
                    operation.inner is None
                    or operation.value_reduction is Reduction.LAST
                ):
                    continue
                inner = operation.inner.value
                add(f"{inner}_inner_state")
                if operation.value_reduction is Reduction.MEAN:
                    add(f"{inner}_weight_state")
        for name in (
            "weight",
            "total_weight",
            "num_macro_steps",
            "sub_step",
            "num_sub_steps",
            "flags",
            "macro_step_index",
        ):
            params[f"__hf_{name}_ptr"] = f"__{name}"
        return params

    def _generate_kernel_for_group(
        self,
        kernel_code_lines: list[str],
        kernel_name: str,
        output_index: str,
        var_list: list[str],
    ) -> None:
        """Generate kernel code for a specific output_index group supporting ops."""
        dims_1d, dims_2d = self._statistics_lowering.split_indexed(var_list)

        # Header
        safe_output_index = self._get_safe_name(output_index)
        kernel_code_lines.extend(
            [
                f"# Kernel for output_index: {output_index}",
                f"# Variables: {', '.join(var_list)}",
                f"# 1D: {', '.join(dims_1d) if dims_1d else 'None'}",
                f"# 2D: {', '.join(dims_2d) if dims_2d else 'None'}",
                "",
                "@triton.jit",
                f"def {kernel_name}(",
            ]
        )

        kernel_code_lines.extend(
            f"    {name},"
            for name in self._group_pointer_arguments(output_index, var_list)
        )
        kernel_code_lines.extend(
            [
                "    n_saved_points: tl.constexpr,",
            ]
        )
        kernel_code_lines.extend(
            [
                "    BLOCK_SIZE: tl.constexpr,",
                "    ensemble_size: tl.constexpr,",
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
            ]
        )

        # Only emit boolean computation lines for booleans actually used by ops
        needed_bools = self._statistics_lowering.required_flags
        if needed_bools:
            kernel_code_lines.append(
                "    # Compute boolean flags from sub_step, num_sub_steps, flags"
            )
            if "is_inner_first" in needed_bools:
                kernel_code_lines.append(
                    "    is_inner_first = (flags & 1) != 0 and sub_step == 0"
                )
            if "is_inner_last" in needed_bools:
                kernel_code_lines.append(
                    "    is_inner_last = ((flags >> 1) & 1) != 0 and sub_step == num_sub_steps - 1"
                )
            if "is_outer_first" in needed_bools:
                kernel_code_lines.append(
                    "    is_outer_first = ((flags >> 2) & 1) != 0 and is_inner_last"
                )
            if "is_outer_last" in needed_bools:
                kernel_code_lines.append(
                    "    is_outer_last = ((flags >> 3) & 1) != 0 and is_inner_last"
                )
            kernel_code_lines.append("")

        kernel_code_lines.extend(
            [
                f"    idx = tl.load({safe_output_index}_ptr + offs, mask=mask)",
                "",
            ]
        )

        # Loop over members - use tl.static_range for compile-time unrolling
        kernel_code_lines.append("    for t in tl.static_range(ensemble_size):")
        indent = "        "

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
            body_indent = indent
            if not batched:
                kernel_code_lines.append(f"{indent}if t == 0:")
                body_indent += "    "
            self._generate_1d_vars_grouped(
                kernel_code_lines,
                vectors,
                body_indent,
                body_indent + "    ",
                body_indent + "        ",
                body_indent + "            ",
            )
            self._emit_indexed_vector_updates(
                dims_2d=levels,
                indent=body_indent,
                indent2=body_indent + "    ",
                indent3=body_indent + "        ",
                kernel_code_lines=kernel_code_lines,
            )
        kernel_code_lines.append("")

    def _generate_full_kernel_for_group(
        self,
        kernel_code_lines: list[str],
        output_index: str,
        var_list: list[str],
    ) -> None:
        """Generate a flat Triton kernel for variables saved at full tensor shape."""
        kernel_code_lines.extend(
            [
                f"# Full-output kernel: {output_index}",
                f"# Variables: {', '.join(var_list)}",
                "",
                "@triton.jit",
                f"def {_FULL_OUTPUT_KERNEL}(",
            ]
        )

        kernel_code_lines.extend(
            f"    {name},"
            for name in self._group_pointer_arguments(output_index, var_list)
        )
        kernel_code_lines.extend(
            [
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
            ]
        )

        indent = "    "
        indent2 = "        "
        self._emit_full_output_updates(
            indent=indent,
            indent2=indent2,
            kernel_code_lines=kernel_code_lines,
            var_list=var_list,
        )
        kernel_code_lines.append("")

    def _generate_main_function(
        self,
        kernel_code_lines: list[str],
        grouped_by_output_index: dict[str, list[str]],
    ) -> None:
        """Generate the main python function that calls kernels."""
        kernel_code_lines.extend(
            [
                "# Main update function",
                "def internal_update_statistics(states, BLOCK_SIZE):",
            ]
        )

        kernel_code_lines.append(f"    ensemble_size = {self.ensemble_size}")

        scatters = self._statistics_ir.ordered_scatters()
        if scatters:
            kernel_code_lines.append(
                "    # Materialize all scatter virtuals in dependency order"
            )
        self._emit_scatter_launchers(
            kernel_code_lines=kernel_code_lines, scatters=scatters
        )
        if scatters:
            kernel_code_lines.append("")

        self._emit_group_launchers(
            grouped_by_output_index=grouped_by_output_index,
            kernel_code_lines=kernel_code_lines,
        )

    def _collect_scalar_updates(
        self,
        *,
        dims_1d: list[str],
        emit_val: Callable[..., str],
        groups: _OperationGroups,
        vars_conditional_only: set[str],
    ) -> None:
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

                    # Use variable-specific inner aggregation result
                    # For 'last' inner type, directly use the variable value (no intermediate state)
                    if inner == "last":
                        val_var = f"{safe_var}_val"
                    else:
                        val_var = f"val_for_{safe_var}_{inner}"

                    if is_arg_compound:
                        # Compound argmax/argmin (e.g., argmax_mean, argmax3_mean)
                        aux_ptr_base = f"{safe_var}_{op}_aux_ptr"

                        if k_val == 1:
                            aux_ptr = f"{aux_ptr_base} + {out_offset}"
                            if self._statistics_layouts[var].dtype.is_floating_point:
                                comparison = ">" if outer == "max" else "<"
                                groups.is_inner_last_is_outer_first.extend(
                                    [
                                        f"tl.store({out_ptr}, -1, mask=mask)",
                                        f"tl.store({aux_ptr}, float('nan'), mask=mask)",
                                    ]
                                )
                                update = [
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other=float('nan'))",
                                    f"{safe_var}_{op}_valid = {val_var} == {val_var}",
                                    f"{safe_var}_{op}_cond = {safe_var}_{op}_valid & (({safe_var}_{op}_aux_old != {safe_var}_{op}_aux_old) | ({val_var} {comparison} {safe_var}_{op}_aux_old))",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ]
                                groups.is_inner_last_is_outer_first.extend(update)
                                groups.is_inner_last_not_is_outer_first.extend(update)
                            else:
                                comparison = ">" if outer == "max" else "<"
                                groups.is_inner_last_is_outer_first.extend(
                                    [
                                        f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                                        f"tl.store({aux_ptr}, {val_var}, mask=mask)",
                                    ]
                                )
                                groups.is_inner_last_not_is_outer_first.extend(
                                    [
                                        f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other={val_var})",
                                        f"{safe_var}_{op}_cond = {val_var} {comparison} {safe_var}_{op}_aux_old",
                                        f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                        f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                    ]
                                )

                        else:
                            # ArgmaxK/ArgminK compound bubble insert
                            groups.argmaxk_ops.append(
                                {
                                    "var": safe_var,
                                    "op": op,
                                    "k": k_val,
                                    "val_var": val_var,
                                    "out_offset": out_offset,
                                    "type": f"arg{outer}",
                                }
                            )
                    elif outer in {"max", "min"}:
                        fn = (
                            "hydroforge_maximum"
                            if outer == "max"
                            else "hydroforge_minimum"
                        )
                        # Compound max without automatic arg (e.g., max_mean, max3_mean)
                        if k_val == 1:
                            groups.is_inner_last_is_outer_first.append(
                                f"tl.store({out_ptr}, {val_var}, mask=mask)"
                            )
                            groups.is_inner_last_not_is_outer_first.extend(
                                [
                                    f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                    f"tl.store({out_ptr}, {fn}({safe_var}_{op}_old, {val_var}), mask=mask)",
                                ]
                            )
                        else:
                            # maxK bubble insert without arg tracking
                            groups.maxk_ops.append(
                                {
                                    "var": safe_var,
                                    "op": op,
                                    "k": k_val,
                                    "val_var": val_var,
                                    "out_offset": out_offset,
                                    "type": outer,
                                }
                            )

                    elif outer == "mean":
                        groups.is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)"
                        )
                        groups.is_inner_last_not_is_outer_first.extend(
                            [
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{safe_var}_{op}_count = num_macro_steps.to({val_var}.dtype)",
                                f"tl.store({out_ptr}, hydroforge_weighted_mean({safe_var}_{op}_old, {safe_var}_{op}_count - 1.0, {val_var}, 1.0), mask=mask)",
                            ]
                        )

                    elif outer == "sum":
                        groups.is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)"
                        )
                        groups.is_inner_last_not_is_outer_first.extend(
                            [
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"tl.store({out_ptr}, {safe_var}_{op}_old + {val_var}, mask=mask)",
                            ]
                        )
                    elif outer == "last":
                        # Compound last (e.g., last_mean) — store the last inner value
                        # Simply overwrite on every is_inner_last step
                        groups.is_inner_last.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)"
                        )
                    elif outer == "first":
                        # Compound first (e.g., first_mean) — store only at is_outer_first
                        groups.is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)"
                        )
                    continue

                # ===== Simple operations (non-compound) =====
                if op == "mean":
                    inner_ops = {
                        reduction.value
                        for reduction in self._statistics_lowering.inner_reductions(var)
                    }
                    if "mean" in inner_ops:
                        # Reuse val_for_{safe_var}_mean from inner aggregation
                        groups.is_inner_last.append(
                            f"tl.store({out_ptr}, val_for_{safe_var}_mean, mask=mask)"
                        )
                    else:
                        # Standalone mean - needs state (use variable-specific val)
                        weight_ptr = (
                            f"{safe_var}_mean_sample_weight_state_ptr + {out_offset}"
                        )
                        groups.unconditional.extend(
                            [
                                f"# Standalone mean for {safe_var}",
                                f"{safe_var}_mean_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                                f"{safe_var}_mean_weight_old = tl.where(is_inner_first, 0.0, tl.load({weight_ptr}, mask=mask, other=0.0))",
                                f"{safe_var}_mean_weight_new = {safe_var}_mean_weight_old + weight",
                                f"{safe_var}_mean_out = hydroforge_weighted_mean({safe_var}_mean_old, {safe_var}_mean_weight_old, {safe_var}_val, weight)",
                                f"tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_mean_weight_new), mask=mask)",
                            ]
                        )
                        groups.unconditional.append(
                            f"tl.store({out_ptr}, {safe_var}_mean_out, mask=mask)"
                        )

                elif op == "sum":
                    groups.unconditional.extend(
                        [
                            f"{safe_var}_sum_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                            f"tl.store({out_ptr}, {safe_var}_sum_old + {safe_var}_val * weight, mask=mask)",
                        ]
                    )

                # Standalone extrema never carry an index or top-k suffix;
                # parse_operation routes those semantics through compound ops.
                elif op in {"max", "min"}:
                    fn = "hydroforge_maximum" if op == "max" else "hydroforge_minimum"
                    groups.is_inner_first.extend(
                        [
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                        ]
                    )
                    groups.not_is_inner_first.extend(
                        [
                            f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                            f"tl.store({out_ptr}, {fn}({safe_var}_{op}_old, {safe_var}_val), mask=mask)",
                        ]
                    )

                elif op in {"first", "last"}:
                    updates = (
                        groups.is_inner_first if op == "first" else groups.is_inner_last
                    )
                    has_compound = any(
                        other.inner is not None and other.inner.value == op
                        for other in operations
                    )
                    if var in vars_conditional_only and not has_compound:
                        loads: list[str] = []
                        emit_val(var, loads, f"inner_{op}")
                        updates.extend(line.lstrip() for line in loads)
                    updates.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")

    def _emit_inner_reduction_updates(
        self,
        *,
        indent: str,
        inner_aggregations_needed: Mapping[Reduction, tuple[str, ...]],
        kernel_code_lines: list[str],
    ) -> None:
        for reduction, inner_vars in inner_aggregations_needed.items():
            inner_type = reduction.value
            for var in inner_vars:
                safe_var = self._get_safe_name(var)
                out_offset = "t * n_saved_points + offs"
                val_for_var_inner = f"val_for_{safe_var}_{inner_type}"
                var_val = f"{safe_var}_val"

                if inner_type == "last":
                    # 'last' is the simplest: val_for_X_last == X_val at is_inner_last.
                    # No state storage, no load/store needed.
                    pass
                elif inner_type == "mean":
                    inner_ptr = (
                        f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    )
                    weight_ptr = (
                        f"{safe_var}_{inner_type}_weight_state_ptr + {out_offset}"
                    )
                    kernel_code_lines.append(
                        f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})"
                    )
                    kernel_code_lines.extend(
                        [
                            f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                            f"{indent}{safe_var}_weight_{inner_type}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                            f"{indent}{safe_var}_weight_{inner_type}_new = {safe_var}_weight_{inner_type}_old + weight",
                            f"{indent}{safe_var}_inner_{inner_type}_new = hydroforge_weighted_mean({safe_var}_inner_{inner_type}_old, {safe_var}_weight_{inner_type}_old, {var_val}, weight)",
                        ]
                    )
                    # Store based on condition - use tl.where for efficiency
                    kernel_code_lines.extend(
                        [
                            f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                            f"{indent}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_weight_{inner_type}_new), mask=mask)",
                            f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                        ]
                    )
                elif inner_type == "sum":
                    inner_ptr = (
                        f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    )
                    kernel_code_lines.append(
                        f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})"
                    )
                    kernel_code_lines.extend(
                        [
                            f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                            f"{indent}{safe_var}_inner_{inner_type}_new = {safe_var}_inner_{inner_type}_old + {var_val} * weight",
                            f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                            f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                        ]
                    )
                elif inner_type in {"max", "min"}:
                    fn = (
                        "hydroforge_maximum"
                        if inner_type == "max"
                        else "hydroforge_minimum"
                    )
                    reset = "-float('inf')" if inner_type == "max" else "float('inf')"
                    inner_ptr = (
                        f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    )
                    kernel_code_lines.append(
                        f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})"
                    )
                    kernel_code_lines.extend(
                        [
                            f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                            f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first, {var_val}, {fn}({safe_var}_inner_{inner_type}_old, {var_val}))",
                            f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, {reset}, {safe_var}_inner_{inner_type}_new), mask=mask)",
                            f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                        ]
                    )
                elif inner_type == "first":
                    # 'first' inner: store the value at is_inner_first, read it back at is_inner_last
                    inner_ptr = (
                        f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    )
                    kernel_code_lines.append(
                        f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})"
                    )
                    kernel_code_lines.extend(
                        [
                            f"{indent}tl.store({inner_ptr}, {var_val}, mask=mask & is_inner_first)",
                            f"{indent}{val_for_var_inner} = tl.where(is_inner_last, tl.load({inner_ptr}, mask=mask, other=0.0), {val_for_var_inner})",
                        ]
                    )

    def _emit_window_results(
        self,
        *,
        dims_1d: list[str],
        emit_val: Callable[..., str],
        groups: _OperationGroups,
        indent: str,
        indent2: str,
        indent3: str,
        indent4: str,
        inner_aggregations_needed: Mapping[Reduction, tuple[str, ...]],
        kernel_code_lines: list[str],
        vars_conditional_only: set[str],
    ) -> None:
        if any(
            (
                groups.is_inner_last,
                groups.is_inner_last_is_outer_first,
                groups.is_inner_last_not_is_outer_first,
                groups.maxk_ops,
                groups.argmaxk_ops,
            )
        ):
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
            if (
                groups.is_inner_last_is_outer_first
                or groups.is_inner_last_not_is_outer_first
            ):
                kernel_code_lines.append(f"{indent2}if is_outer_first:")
                for line in groups.is_inner_last_is_outer_first:
                    kernel_code_lines.append(f"{indent3}{line}")
                if groups.is_inner_last_not_is_outer_first:
                    kernel_code_lines.append(f"{indent2}else:")
                    for line in groups.is_inner_last_not_is_outer_first:
                        kernel_code_lines.append(f"{indent3}{line}")

            # Simple is_inner_last ops
            for line in groups.is_inner_last:
                kernel_code_lines.append(f"{indent2}{line}")

            self._emit_topk_updates(
                groups, kernel_code_lines, indent2, indent3, indent4
            )

    def _emit_topk_updates(
        self,
        groups: _OperationGroups,
        lines: list[str],
        indent: str,
        branch_indent: str,
        loop_indent: str,
    ) -> None:
        # Only operations consuming the same inner value can share a loop.
        grouped: dict[tuple[str, int, str, str], dict[str, _TopKOperation]] = {}
        for op in (*groups.maxk_ops, *groups.argmaxk_ops):
            key = (op["var"], op["k"], op["out_offset"], op["val_var"])
            grouped.setdefault(key, {})[op["type"]] = op
        for (safe_var, k, offset, value), operations in grouped.items():
            kinds = [
                kind
                for kind in ("max", "min", "argmax", "argmin")
                if kind in operations
            ]
            base = f"{safe_var}_k{k}_base_offs"
            lines.append(f"{indent}{base} = ({offset}) * {k}")
            pointers = {}
            for kind in kinds:
                op = operations[kind]
                is_arg = kind.startswith("arg")
                out = f"{safe_var}_{op['op']}_ptr"
                values = f"{safe_var}_{op['op']}_aux_ptr" if is_arg else out
                pointers[kind] = (out, values)
                lines.append(f"{indent}new_val_{kind}_{safe_var} = {value}")
                if is_arg:
                    lines.append(
                        f"{indent}new_idx_{kind}_{safe_var} = tl.full([BLOCK_SIZE], macro_step_index, dtype=tl.int64)"
                    )
            lines.append(f"{indent}if is_outer_first:")
            for kind in kinds:
                out, values = pointers[kind]
                new_value = f"new_val_{kind}_{safe_var}"
                lines.append(
                    f"{branch_indent}tl.store({values} + {base}, {new_value}, mask=mask)"
                )
                if kind.startswith("arg"):
                    lines.append(
                        f"{branch_indent}tl.store({out} + {base}, tl.where({new_value} == {new_value}, new_idx_{kind}_{safe_var}, -1), mask=mask)"
                    )
            lines.append(f"{branch_indent}for k in tl.static_range(1, {k}):")
            for kind in kinds:
                out, values = pointers[kind]
                lines.append(
                    f"{loop_indent}tl.store({values} + {base} + k, float('nan'), mask=mask)"
                )
                if kind.startswith("arg"):
                    lines.append(
                        f"{loop_indent}tl.store({out} + {base} + k, -1, mask=mask)"
                    )
            lines.append(f"{indent}else:")
            lines.append(f"{branch_indent}for k in tl.static_range({k}):")
            for kind in kinds:
                out, values = pointers[kind]
                is_arg = kind.startswith("arg")
                comparison = ">" if kind.endswith("max") else "<"
                new_value = f"new_val_{kind}_{safe_var}"
                old_value = f"old_{kind}_k"
                swap = f"swap_{kind}"
                lines.append(
                    f"{loop_indent}{old_value} = tl.load({values} + {base} + k, mask=mask, other=float('nan'))"
                )
                better = f"({old_value} != {old_value}) | ({new_value} {comparison} {old_value})"
                if is_arg:
                    new_index = f"new_idx_{kind}_{safe_var}"
                    old_index = f"old_{kind}_idx_k"
                    lines.append(
                        f"{loop_indent}{old_index} = tl.load({out} + {base} + k, mask=mask, other=-1)"
                    )
                    better += f" | (({new_value} == {old_value}) & ({new_index} < {old_index}))"
                lines.extend(
                    [
                        f"{loop_indent}{swap} = ({new_value} == {new_value}) & ({better})",
                        f"{loop_indent}{kind}_to_store = tl.where({swap}, {new_value}, {old_value})",
                        f"{loop_indent}{new_value} = tl.where({swap}, {old_value}, {new_value})",
                        f"{loop_indent}tl.store({values} + {base} + k, {kind}_to_store, mask=mask & {swap})",
                    ]
                )
                if is_arg:
                    lines.extend(
                        [
                            f"{loop_indent}{kind}_idx_store = tl.where({swap}, {new_index}, {old_index})",
                            f"{loop_indent}{new_index} = tl.where({swap}, {old_index}, {new_index})",
                            f"{loop_indent}tl.store({out} + {base} + k, {kind}_idx_store, mask=mask & {swap})",
                        ]
                    )

    def _emit_indexed_vector_updates(
        self,
        *,
        dims_2d: list[str],
        indent: str,
        indent2: str,
        indent3: str,
        kernel_code_lines: list[str],
    ) -> None:
        if dims_2d:

            def is_last_only(name: str) -> bool:
                operations = self._statistics_lowering.operations(name)
                return len(operations) == 1 and operations[0].outer.value == "last"

            non_last_only = [v for v in dims_2d if not is_last_only(v)]
            last_only_vars = [v for v in dims_2d if is_last_only(v)]

            if non_last_only:
                for var in non_last_only:
                    safe_var = self._get_safe_name(var)
                    n_levels_var = self._statistics_layouts[var].actual_shape[-1]
                    kernel_code_lines.extend(
                        [
                            f"{indent}# 2D variable: {var}",
                            f"{indent}for level in tl.static_range({n_levels_var}):",
                        ]
                    )
                    out_offset = f"(t * n_saved_points + offs) * {n_levels_var} + level"

                    val_name = self._emit_value(
                        var,
                        kernel_code_lines,
                        set(),
                        indent=indent2,
                        offset=lambda key: (
                            f"(t * {self._source_stride(key, logical_rank=2)} + idx) * {n_levels_var} + level"
                        ),
                    )
                    kernel_code_lines.append(f"{indent2}val = {val_name}")

                    for operation in self._statistics_lowering.operations(var):
                        op = operation.spelling
                        out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"
                        if operation.output is Reduction.MEAN:
                            weight_ptr = (
                                f"{safe_var}_mean_sample_weight_state_ptr + "
                                f"{out_offset}"
                            )
                            kernel_code_lines.extend(
                                [
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
                                ]
                            )
                        elif operation.output is Reduction.SUM:
                            kernel_code_lines.extend(
                                [
                                    f"{indent2}if is_inner_first:",
                                    f"{indent3}old = tl.zeros_like(val)",
                                    f"{indent2}else:",
                                    f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent2}new = old + val * weight",
                                    f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                                ]
                            )
                        elif operation.output in {Reduction.MAX, Reduction.MIN}:
                            fn = (
                                "hydroforge_maximum"
                                if operation.output is Reduction.MAX
                                else "hydroforge_minimum"
                            )
                            kernel_code_lines.extend(
                                [
                                    f"{indent2}if is_inner_first:",
                                    f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                    f"{indent2}else:",
                                    f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                    f"{indent3}new = {fn}(old, val)",
                                    f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                                ]
                            )
                        elif operation.output is Reduction.LAST:
                            kernel_code_lines.extend(
                                [
                                    f"{indent2}if is_inner_last:",
                                    f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                ]
                            )
                        elif operation.output is Reduction.FIRST:
                            kernel_code_lines.extend(
                                [
                                    f"{indent2}if is_inner_first:",
                                    f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                ]
                            )
                kernel_code_lines.append("")

            if last_only_vars:
                kernel_code_lines.extend(
                    [
                        f"{indent}# 2D variables (last-only)",
                        f"{indent}if is_inner_last:",
                    ]
                )
                for var in last_only_vars:
                    safe_var = self._get_safe_name(var)
                    n_levels_var = self._statistics_layouts[var].actual_shape[-1]
                    kernel_code_lines.append(
                        f"{indent2}for level in tl.static_range({n_levels_var}):"
                    )
                    out_offset = f"(t * n_saved_points + offs) * {n_levels_var} + level"
                    val_name = self._emit_value(
                        var,
                        kernel_code_lines,
                        set(),
                        indent=indent3,
                        offset=lambda key: (
                            f"(t * {self._source_stride(key, logical_rank=2)} + idx) * {n_levels_var} + level"
                        ),
                    )
                    kernel_code_lines.extend(
                        [
                            f"{indent3}val = {val_name}",
                            f"{indent3}tl.store({safe_var}_last_ptr + {out_offset}, val, mask=mask)",
                        ]
                    )

    def _emit_full_output_updates(
        self,
        *,
        indent: str,
        indent2: str,
        kernel_code_lines: list[str],
        var_list: list[str],
    ) -> None:
        for var in var_list:
            safe_var = self._get_safe_name(var)
            var_numel = prod(self._statistics_layouts[var].actual_shape)
            kernel_code_lines.extend(
                [
                    f"{indent}# === full tensor variable: {var} ===",
                    f"{indent}var_mask = offs < {var_numel}",
                ]
            )

            self._emit_value(
                var,
                kernel_code_lines,
                set(),
                indent=indent,
                mask="var_mask",
                offset=lambda key: self._full_source_offset(key, var, "offs"),
            )

            for reduction in self._statistics_lowering.inner_reductions(var):
                inner = reduction.value
                if inner == "last":
                    continue
                val_for = f"{safe_var}_{inner}_val"
                inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + offs"
                if inner == "mean":
                    weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + offs"
                    kernel_code_lines.extend(
                        [
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
                        ]
                    )
                elif inner == "sum":
                    kernel_code_lines.extend(
                        [
                            f"{indent}inner_{inner}_old = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                            f"{indent}inner_{inner}_new = inner_{inner}_old + {safe_var}_val * weight",
                            f"{indent}{val_for} = inner_{inner}_new",
                            f"{indent}if is_inner_last:",
                            f"{indent2}tl.store({inner_ptr}, 0.0, mask=var_mask)",
                            f"{indent}else:",
                            f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=var_mask)",
                        ]
                    )
                elif inner in {"max", "min"}:
                    function = (
                        "hydroforge_maximum" if inner == "max" else "hydroforge_minimum"
                    )
                    sentinel = "-float('inf')" if inner == "max" else "float('inf')"
                    kernel_code_lines.extend(
                        [
                            f"{indent}inner_{inner}_old = tl.load({inner_ptr}, mask=var_mask, other={safe_var}_val)",
                            f"{indent}inner_{inner}_new = tl.where(is_inner_first, {safe_var}_val, {function}(inner_{inner}_old, {safe_var}_val))",
                            f"{indent}{val_for} = inner_{inner}_new",
                            f"{indent}if is_inner_last:",
                            f"{indent2}tl.store({inner_ptr}, {sentinel}, mask=var_mask)",
                            f"{indent}else:",
                            f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=var_mask)",
                        ]
                    )
                elif inner == "first":
                    kernel_code_lines.extend(
                        [
                            f"{indent}if is_inner_first:",
                            f"{indent2}tl.store({inner_ptr}, {safe_var}_val, mask=var_mask)",
                            f"{indent}{val_for} = tl.load({inner_ptr}, mask=var_mask, other=0.0)",
                        ]
                    )

            for operation in self._statistics_lowering.operations(var):
                op = operation.spelling
                out_ptr = f"{safe_var}_{op}_ptr + offs"

                if operation.compound:
                    outer = operation.outer.value
                    inner = operation.inner.value
                    val_for = (
                        f"{safe_var}_val"
                        if inner == "last"
                        else f"{safe_var}_{inner}_val"
                    )

                    kernel_code_lines.append(f"{indent}if is_inner_last:")
                    if outer in {"max", "min"}:
                        fn = (
                            "hydroforge_maximum"
                            if outer == "max"
                            else "hydroforge_minimum"
                        )
                        kernel_code_lines.extend(
                            [
                                f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other={val_for})",
                                f"{indent2}new = tl.where(is_outer_first, {val_for}, {fn}(old, {val_for}))",
                                f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                            ]
                        )
                    elif outer == "sum":
                        kernel_code_lines.extend(
                            [
                                f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                                f"{indent2}new = tl.where(is_outer_first, {val_for}, old + {val_for})",
                                f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                            ]
                        )
                    elif outer == "mean":
                        kernel_code_lines.extend(
                            [
                                f"{indent2}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                                f"{indent2}count = num_macro_steps.to({val_for}.dtype)",
                                f"{indent2}new = tl.where(is_outer_first, {val_for}, hydroforge_weighted_mean(old, count - 1.0, {val_for}, 1.0))",
                                f"{indent2}tl.store({out_ptr}, new, mask=var_mask)",
                            ]
                        )
                    elif outer == "last":
                        kernel_code_lines.append(
                            f"{indent2}tl.store({out_ptr}, {val_for}, mask=var_mask)"
                        )
                    elif outer == "first":
                        kernel_code_lines.append(
                            f"{indent2}tl.store({out_ptr}, {val_for}, mask=var_mask & is_outer_first)"
                        )
                    kernel_code_lines.append("")
                    continue

                if op == "mean":
                    weight_ptr = f"{safe_var}_mean_sample_weight_state_ptr + offs"
                    kernel_code_lines.extend(
                        [
                            f"{indent}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                            f"{indent}old_weight = tl.load({weight_ptr}, mask=var_mask, other=0.0)",
                            f"{indent}old = tl.where(is_inner_first, 0.0, old)",
                            f"{indent}old_weight = tl.where(is_inner_first, 0.0, old_weight)",
                            f"{indent}new_weight = old_weight + weight",
                            f"{indent}new = hydroforge_weighted_mean(old, old_weight, {safe_var}_val, weight)",
                            f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                            f"{indent}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, new_weight), mask=var_mask)",
                        ]
                    )
                elif op == "sum":
                    kernel_code_lines.extend(
                        [
                            f"{indent}old = tl.load({out_ptr}, mask=var_mask, other=0.0)",
                            f"{indent}new = tl.where(is_inner_first, 0.0, old) + {safe_var}_val * weight",
                            f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                        ]
                    )
                elif op in {"max", "min"}:
                    fn = "hydroforge_maximum" if op == "max" else "hydroforge_minimum"
                    kernel_code_lines.extend(
                        [
                            f"{indent}old = tl.load({out_ptr}, mask=var_mask, other={safe_var}_val)",
                            f"{indent}new = tl.where(is_inner_first, {safe_var}_val, {fn}(old, {safe_var}_val))",
                            f"{indent}tl.store({out_ptr}, new, mask=var_mask)",
                        ]
                    )
                elif op == "last":
                    kernel_code_lines.append(
                        f"{indent}tl.store({out_ptr}, {safe_var}_val, mask=var_mask & is_inner_last)"
                    )
                elif op == "first":
                    kernel_code_lines.append(
                        f"{indent}tl.store({out_ptr}, {safe_var}_val, mask=var_mask & is_inner_first)"
                    )
                kernel_code_lines.append("")

    def _emit_scatter_launchers(
        self, *, kernel_code_lines: list[str], scatters: tuple
    ) -> None:
        for variable in scatters:
            var = variable.name
            scatter = variable.source
            safe_var = self._get_safe_name(var)
            buf_key = f"__scatter_buf_{var}"
            is_mean = scatter.reduction.value == "mean"
            scatter_ensemble = str(
                self.ensemble_size if self._statistics_layouts[var].batched else 1
            )
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
            zero_args.extend([f"_N_{safe_var}", "BLOCK_SIZE", scatter_ensemble])
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
            add_args.extend(
                [
                    f"_M_{safe_var}",
                    f"_N_{safe_var}",
                    "BLOCK_SIZE",
                    scatter_ensemble,
                ]
            )
            for token in sorted_src:
                add_args.append(str(self._source_stride(token)))
            if self._tensor_registry[scatter.index].numel() > 0:
                kernel_code_lines.append(
                    f"    scatter_add_{safe_var}["
                    f"(triton.cdiv(_M_{safe_var}, BLOCK_SIZE),)]"
                    f"({', '.join(add_args)})"
                )
            if is_mean and self._storage[buf_key].shape[-1] > 0:
                div_args = [
                    f"states['{buf_key}']",
                    f"states['{cnt_key}']",
                    f"_N_{safe_var}",
                    "BLOCK_SIZE",
                    scatter_ensemble,
                ]
                kernel_code_lines.append(
                    f"    scatter_divide_{safe_var}["
                    f"(triton.cdiv(_N_{safe_var}, BLOCK_SIZE),)]"
                    f"({', '.join(div_args)})"
                )

    def _emit_group_launchers(
        self,
        *,
        grouped_by_output_index: dict[str, list[str]],
        kernel_code_lines: list[str],
    ) -> None:
        for output_index, var_list in grouped_by_output_index.items():
            full_output = output_index == _FULL_OUTPUT_GROUP
            if full_output:
                full_len = max(
                    prod(self._statistics_layouts[var].actual_shape) for var in var_list
                )
                if full_len == 0:
                    kernel_code_lines.append(
                        "    # Skip empty full-output statistics group"
                    )
                    continue
                extent = "full_len"
                kernel_name, grid_name = _FULL_OUTPUT_KERNEL, _FULL_OUTPUT_GRID
                kernel_code_lines.append(f"    full_len = {full_len}")
            else:
                if self._tensor_registry[output_index].numel() == 0:
                    kernel_code_lines.append(
                        f"    # Skip empty statistics group {output_index}"
                    )
                    continue
                safe_output_index = self._get_safe_name(output_index)
                extent = "output_index_len"
                kernel_name, grid_name = (
                    f"kernel_{safe_output_index}",
                    f"grid_{safe_output_index}",
                )
                kernel_code_lines.append(
                    f"    output_index_len = len(states['{output_index}'])"
                )
            kernel_code_lines.extend(
                [
                    f"    {grid_name} = lambda meta: (triton.cdiv({extent}, meta['BLOCK_SIZE']),)",
                    f"    {kernel_name}[{grid_name}](",
                ]
            )
            kernel_code_lines.extend(
                f"        {name}=states['{key}'],"
                for name, key in self._group_pointer_arguments(
                    output_index, var_list
                ).items()
            )
            extent_name = "n_elements" if full_output else "n_saved_points"
            kernel_code_lines.append(f"        {extent_name}={extent},")
            kernel_code_lines.append("        BLOCK_SIZE=BLOCK_SIZE,")
            if not full_output:
                kernel_code_lines.append("        ensemble_size=ensemble_size,")
            kernel_code_lines.extend(["    )", ""])

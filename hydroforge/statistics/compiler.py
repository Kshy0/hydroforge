# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Compile explicit statistics inputs into an uninstalled backend program."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from hydroforge.kernels.backends.ir_program import IrKernel, compile_ir_program
from hydroforge.kernels.codegen.ir import PhaseTest
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.statistics.ir import StatisticsIR
from hydroforge.statistics.kernel_plan import (
    StatisticsCompileContext,
    StatisticsKernel,
    StatisticsKernelPlan,
    plan_statistics,
)
from hydroforge.statistics.lowering import StatisticsLowering, lower_statistics
from hydroforge.statistics.metal_scatter import encoded_scatter
from hydroforge.statistics.phases import (
    CONTROL_PHASE,
    KERNEL_CONTROLS,
    SAMPLE_PHASES,
    phase_implies,
)
from hydroforge.statistics.storage import StoragePlan

if TYPE_CHECKING:
    pass


def _no_requests(
    states: Mapping[str, torch.Tensor], block_size: int
) -> tuple[CompileRequest, ...]:
    del states, block_size
    return ()


def _no_close() -> None:
    pass


@dataclass(frozen=True, slots=True)
class CompiledStatistics:
    """One backend's statistics program.

    ``function(states, block_size, phase)`` runs one sample; a negative
    ``phase`` reads the sample phase from device controls.
    ``settle(states, phase)`` folds a close without a sample whose ``phase``
    the runtime has also written to the device controls; it is ``None``
    without compound statistics.  ``requests(states, block_size)``
    describes the native compilation the first sample would otherwise do.
    """

    lowering: StatisticsLowering
    function: Callable[..., None]
    settle: Callable[..., None] | None
    module: object | None
    saved_kernel_file: Path | None
    generated_modules: tuple[tuple[str, str], ...] = ()
    requests: Callable[..., tuple[CompileRequest, ...]] = _no_requests
    close: Callable[[], None] = _no_close


def _implies(condition: PhaseTest, implied: PhaseTest) -> bool:
    return phase_implies(condition.bits, implied.bits)


def _to_ir_kernels(
    kernels: tuple[StatisticsKernel, ...] | list[StatisticsKernel],
) -> tuple[IrKernel, ...]:
    return tuple(
        IrKernel(
            function=kernel.function,
            scalars=kernel.scalars,
            extent=kernel.extent,
            phase_mask=kernel.phase_mask,
        )
        for kernel in kernels
    )


def _dialect_kernels(
    context: StatisticsCompileContext,
    plan: StatisticsKernelPlan,
    dialect: str,
) -> tuple[tuple[IrKernel, ...], tuple[IrKernel, ...], dict[str, torch.Tensor], bool]:
    settle = _to_ir_kernels(plan.settle_kernels("loop")) if plan.lowering.compound else ()
    if dialect == "cuda":
        return _to_ir_kernels(plan.kernels("threads")), settle, {}, False
    if dialect == "triton":
        kernels = (
            *plan.scatter_kernels(unrolled=True),
            *plan.group_kernels("tiles"),
        )
        return _to_ir_kernels(kernels), settle, {}, False
    if dialect == "torch":
        kernels = (
            *plan.scatter_kernels(unrolled=True),
            *plan.group_kernels("loop"),
        )
        return _to_ir_kernels(kernels), settle, {}, False
    if dialect == "msl":
        items: list[StatisticsKernel] = []
        extra: dict[str, torch.Tensor] = {}
        for scatter, declared in zip(
            plan.scatters, plan.lowering.ir.ordered_scatters(), strict=True
        ):
            if scatter is None:
                continue
            zero, add = scatter.zero, scatter.add
            if (
                context.storage[StoragePlan.scatter_buffer(declared.name)].dtype
                == torch.float64
            ):
                add, topology = encoded_scatter(
                    context, plan.lowering, declared.name, declared.source
                )
                extra.update(topology)
                zero = None
            items.extend(
                entry for entry in (zero, add, scatter.divide) if entry is not None
            )
        items.extend(plan.group_kernels("loop"))
        ir_kernels = _to_ir_kernels(items)
        emulated = any(
            layout.dtype == torch.float64 for layout in context.layouts.values()
        ) or any(
            param.type == torch.float64
            for kernel in (*ir_kernels, *settle)
            for param in kernel.function.params
        )
        return ir_kernels, settle, extra, emulated
    raise KeyError(dialect)


def compile_statistics_program(
    context: StatisticsCompileContext,
    ir: StatisticsIR,
    *,
    dialect: str,
) -> CompiledStatistics:
    """Emit the statistics program through the unified kernel-IR toolchain."""

    lowering = lower_statistics(ir)
    plan = plan_statistics(context, lowering)
    kernels, settle, extra, emulated = _dialect_kernels(context, plan, dialect)
    states = {**context.tensors, **context.storage}
    sizes = {
        **{name: tensor.numel() for name, tensor in context.storage.items()},
        **{name: tensor.numel() for name, tensor in context.tensors.items()},
        **dict.fromkeys(KERNEL_CONTROLS, 1),
    }
    compiled = compile_ir_program(
        dialect,
        device=context.device,
        kernels=kernels,
        settle_kernels=settle,
        states=states,
        sizes=sizes,
        scalar_source=context.tensors,
        extra_buffers=extra,
        emulated=emulated,
        implies=_implies,
        phase_control=CONTROL_PHASE,
        sample_phases=SAMPLE_PHASES,
        compile_torch_update=not any(group.output_index is None for group in plan.groups),
        physics=False,
        name_prefix="hydroforge_statistics",
        rank=context.rank,
        save_dir=context.kernels_dir if context.save_kernels else None,
    )
    return CompiledStatistics(
        lowering=lowering,
        function=compiled.function,
        settle=compiled.settle,
        module=compiled.module,
        saved_kernel_file=compiled.saved_kernel_file,
        generated_modules=compiled.generated_modules,
        requests=compiled.requests,
        close=compiled.close,
    )

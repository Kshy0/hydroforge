"""CUDA/HIP statistics: one runtime-compiled program, prepared launches.

The plan's kernels map one thread to each saved point, member and level.
NVRTC (hiprtc under ROCm) compiles the printed source without contracting
multiplies and adds; the launches of one bound state mapping and block size
are validated and packed once, then replayed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import groupby
from typing import Any

import torch

from hydroforge.kernels.codegen.c import CUDA
from hydroforge.kernels.toolchain import cuda as rtc
from hydroforge.statistics.kernel_plan import (
    StatisticsCompileContext,
    StatisticsKernel,
    StatisticsKernelPlan,
)
from hydroforge.statistics.launch import CompiledStatistics, save_source


def _bind(
    request: rtc.RtcRequest,
    kernels: Sequence[StatisticsKernel],
    states: Mapping[str, torch.Tensor],
    block_size: int,
) -> list[tuple[int | None, Any]]:
    """Validate the bound buffers once and pre-pack every launch."""

    owner = None
    buffers = {
        param.name: param
        for kernel in kernels
        for param in kernel.function.params
        if param.access is not None
    }
    for name, param in buffers.items():
        tensor = states[name]
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a CUDA/HIP tensor")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
        if tensor.dtype != param.type:
            raise RuntimeError(f"{name} has unexpected dtype")
        if owner is None:
            owner = tensor.device
        elif tensor.device != owner:
            raise RuntimeError(
                f"{name} must be on the same CUDA/HIP device as all statistics buffers"
            )
    steps = []
    for kernel in kernels:
        count = kernel.extent.value(states)
        if count == 0:
            continue
        arguments = tuple(
            rtc.pointer(states[param.name])
            if param.access is not None
            else rtc.scalar(kernel.scalars[param.name].value(states), param.type)
            for param in kernel.function.params
        )
        step = rtc.CudaLaunch(
            kernel.function.name, rtc.blocks(count, block_size), block_size, arguments
        )
        steps.append((kernel.phase_mask, step))
    # Scatter zero/add/divide and adjacent groups commonly share a phase.
    # One prepared sequence preserves their dependency order while checking
    # the current device, stream and allocation lifetime once for that run.
    return [
        (
            mask,
            rtc.prepare(request, tuple(step for _mask, step in group), owner.index),
        )
        for mask, group in groupby(steps, key=lambda item: item[0])
    ]


def compile_statistics(
    context: StatisticsCompileContext, plan: StatisticsKernelPlan
) -> CompiledStatistics:
    kernels = plan.kernels("threads")
    source = CUDA.program([kernel.function for kernel in kernels])
    program = rtc.RtcProgram(
        source, rtc.program_options((), physics=False), "hydroforge_statistics"
    )
    request = rtc.RtcRequest(program, tuple(kernel.function.name for kernel in kernels))
    device = torch.device(context.device).index
    if device is None:
        device = torch.cuda.current_device()
    bound: list[Any] = []

    def internal_update_statistics(states, BLOCK_SIZE, phase):
        # Rebinding follows a replaced state mapping or block size; the
        # mapping itself keeps every bound tensor alive.  A negative
        # ``phase`` launches every kernel to gate on device controls.
        if not bound or bound[0] is not states or bound[1] != BLOCK_SIZE:
            bound[:] = [states, BLOCK_SIZE, _bind(request, kernels, states, BLOCK_SIZE)]
        for mask, launch in bound[2]:
            if mask is None or phase < 0 or phase & mask:
                launch()

    return CompiledStatistics(
        lowering=plan.lowering,
        function=internal_update_statistics,
        settle=None,
        module=None,
        saved_kernel_file=(
            save_source(context, rtc.RTC_PRELUDE + source, ".cu")
            if context.save_kernels
            else None
        ),
        requests=lambda states, block_size: (rtc.precompile_request(request, device),),
    )

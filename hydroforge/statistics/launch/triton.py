"""Triton statistics: one generated module of block-program kernels.

The plan's kernels run a block of lanes per program: output groups map lanes
to saved points and unroll members (``tiles``), scatter adds map lanes to
source points.  Before printing, phase branches of different statistics
merge and loads move into the branches that use them
(:mod:`hydroforge.kernels.codegen.passes`).  The printed module is the same
text in every process, so Triton's disk cache serves it after the first
compilation; framework launch options keep multiplies and adds apart.  The
launches of one bound state mapping and block size are resolved once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any

import torch

from hydroforge.kernels.codegen.ir import KernelFunction, PhaseTest
from hydroforge.kernels.codegen.passes import hoist, sink
from hydroforge.kernels.codegen.triton import PRINTER
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain.python import (
    compile_generated_module,
    release_generated_module,
)
from hydroforge.kernels.toolchain.triton import launch_options, warmup_request
from hydroforge.platform.triton_driver import proven_triton_device, triton_call_device
from hydroforge.statistics.kernel_plan import (
    StatisticsCompileContext,
    StatisticsKernel,
    StatisticsKernelPlan,
)
from hydroforge.statistics.launch import CompiledStatistics, save_source, unique_name
from hydroforge.statistics.phases import phase_implies


def _implies(condition: PhaseTest, implied: PhaseTest) -> bool:
    return phase_implies(condition.bits, implied.bits)


def _has_writable_alias(
    function: KernelFunction, states: Mapping[str, torch.Tensor]
) -> bool:
    """Storage identity is conservative for disjoint views of one allocation."""
    seen: dict[tuple[torch.device, int], bool] = {}
    for param in function.params:
        tensor = states.get(param.name)
        if param.access is None or tensor is None or tensor.numel() == 0:
            continue
        key = (tensor.device, tensor.untyped_storage().data_ptr())
        writable = param.access != "read"
        if key in seen and (writable or seen[key]):
            return True
        seen[key] = writable
    return False


def _reordered(function: KernelFunction) -> KernelFunction:
    # Sinking first frees each branch from the values only it uses, so it
    # can rise and merge; sinking again moves values merged branches share.
    body = sink(hoist(sink(function.body), _implies))
    return replace(function, body=body)


def _bind(
    kernels: Sequence[tuple[StatisticsKernel, Any]],
    states: Mapping[str, torch.Tensor],
    block_size: int,
) -> list[tuple[int | None, Any, Any, tuple, dict]]:
    """Resolve every nonempty launch of one state mapping and block size."""

    launches = []
    for kernel, function in kernels:
        extent = kernel.extent.value(states)
        if extent == 0:
            continue
        arguments = tuple(
            states[param.name]
            if param.access is not None
            else kernel.scalars[param.name].value(states)
            for param in kernel.function.params
        )
        grid = ((extent + block_size - 1) // block_size,)
        with proven_triton_device(triton_call_device(arguments)).active():
            options = {
                **launch_options(function, physics=False),
                "BLOCK_SIZE": block_size,
            }
        launches.append((kernel.phase_mask, function, grid, arguments, options))
    return launches


def program(
    plan: StatisticsKernelPlan,
    *,
    states: Mapping[str, torch.Tensor] | None = None,
) -> tuple[tuple[StatisticsKernel, ...], str]:
    """The kernels in launch order and the module source defining them."""

    kernels = tuple(
        kernel
        if states is not None and _has_writable_alias(kernel.function, states)
        else replace(kernel, function=_reordered(kernel.function))
        for kernel in (
            *plan.scatter_kernels(unrolled=True),
            *plan.group_kernels("tiles"),
        )
    )
    return kernels, PRINTER.program([kernel.function for kernel in kernels])


def compile_statistics(
    context: StatisticsCompileContext, plan: StatisticsKernelPlan
) -> CompiledStatistics:
    kernels, source = program(plan, states={**context.tensors, **context.storage})
    name = f"hydroforge_statistics_r{context.rank}_{unique_name(context.rank)}"
    module = compile_generated_module(source, name=name)
    try:
        jit = tuple(
            (kernel, getattr(module, kernel.function.name)) for kernel in kernels
        )
        saved = save_source(context, source, ".py") if context.save_kernels else None
    except BaseException:
        release_generated_module(name, module.__file__)
        raise
    bound: list[Any] = []

    def internal_update_statistics(states, BLOCK_SIZE, phase):
        # Rebinding follows a replaced state mapping or block size.  A
        # negative ``phase`` launches every kernel to gate on device controls.
        if not bound or bound[0] is not states or bound[1] != BLOCK_SIZE:
            launches = _bind(jit, states, BLOCK_SIZE)
            bound[:] = [
                states,
                BLOCK_SIZE,
                [
                    (
                        mask,
                        function[grid],
                        arguments,
                        options,
                    )
                    for mask, function, grid, arguments, options in launches
                ],
            ]
        for mask, launch, arguments, options in bound[2]:
            if mask is None or phase < 0 or phase & mask:
                launch(*arguments, **options)

    def requests(
        states: Mapping[str, torch.Tensor], block_size: int
    ) -> tuple[CompileRequest, ...]:
        def warmup(function, grid, arguments, options):
            with proven_triton_device(triton_call_device(arguments)).active():
                return function.warmup(*arguments, grid=grid, **options)

        return tuple(
            warmup_request(
                partial(
                    warmup,
                    function,
                    grid,
                    arguments,
                    options,
                )
            )
            for _mask, function, grid, arguments, options in _bind(
                jit, states, block_size
            )
        )

    return CompiledStatistics(
        lowering=plan.lowering,
        function=internal_update_statistics,
        settle=None,
        module=module,
        saved_kernel_file=saved,
        generated_modules=((name, module.__file__),),
        requests=requests,
    )

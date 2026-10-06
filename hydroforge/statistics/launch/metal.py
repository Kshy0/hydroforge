# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Metal statistics: one MSL library, argument bindings reused per states.

The plan's kernels map one thread to each saved point and loop over members
and levels.  Every kernel of the program shares the printed source, so the
library compiles once; the launches of one bound state mapping keep their
argument bindings until another mapping replaces them.
"""

from __future__ import annotations

from typing import Any

import torch

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.kernels.codegen.c import MSL, msl_arguments
from hydroforge.kernels.codegen.msl import emulated_msl
from hydroforge.kernels.metal import MetalArgument, MetalProgram
from hydroforge.platform.backend import METAL
from hydroforge.statistics.kernel_plan import (
    StatisticsCompileContext,
    StatisticsKernelPlan,
)
from hydroforge.statistics.launch import (
    CompiledStatistics,
    save_source,
    source_path,
)
from hydroforge.statistics.launch.metal_scatter import encoded_scatter
from hydroforge.statistics.storage import StoragePlan

# The launch value naming a program's thread count; no argument field.
_THREADS = "__threads"


def compile_statistics(
    context: StatisticsCompileContext, plan: StatisticsKernelPlan
) -> CompiledStatistics:
    kernels = []
    extra = {}
    for scatter, declared in zip(
        plan.scatters, plan.lowering.ir.ordered_scatters(), strict=True
    ):
        zero, add = scatter.zero, scatter.add
        if (
            context.storage[StoragePlan.scatter_buffer(declared.name)].dtype
            == torch.float64
        ):
            # The destination-owned add writes every target and count itself.
            add, topology = encoded_scatter(
                context, plan.lowering, declared.name, declared.source
            )
            extra.update(topology)
            zero = None
        kernels.extend(item for item in (zero, add, scatter.divide) if item is not None)
    kernels.extend(plan.group_kernels("loop"))
    sample_count = len(kernels)
    kernels.extend(plan.settle_kernels("loop"))
    emulated = any(
        layout.dtype == torch.float64 for layout in context.layouts.values()
    ) or any(
        param.type == torch.float64
        for kernel in kernels
        for param in kernel.function.params
    )
    printer = emulated_msl() if emulated else MSL
    source = printer.program([kernel.function for kernel in kernels])
    programs = []
    for kernel in kernels:
        fields = msl_arguments(kernel.function, emulated=emulated)
        program = MetalProgram(
            source,
            kernel.function.name,
            tuple(MetalArgument(*field) for field in fields),
            extent=(_THREADS,),
            encoding="float32x2" if emulated else "native",
        )
        names = tuple(zip(kernel.function.params, (field[0] for field in fields)))
        programs.append((kernel, program, names))
        source = program.source

    def bind(states, selected):
        launches = []
        for kernel, program, names in selected:
            values = {
                field: extra[param.name]
                if param.name in extra
                else states[param.name]
                if param.access is not None
                else kernel.scalars[param.name].value(states)
                for param, field in names
            }
            values[_THREADS] = kernel.extent.value(states)
            launches.append(
                (kernel.phase_mask, program.specialize(values, METAL.block.fixed))
            )
        return launches

    bound: list[Any] = []
    settled: list[Any] = []
    closed = False

    def internal_update_statistics(states, block_size, phase):
        if closed:
            raise RuntimeError("Metal statistics program is closed")
        # The launches, and so the argument bindings, of one bound state
        # mapping serve every sample; Metal launches a fixed width.
        del block_size
        if not bound or bound[0] is not states:
            previous = bound[1] if bound else ()
            bound[:] = [states, bind(states, programs[:sample_count])]
            with cleanup_on_exit(
                "Metal statistics bindings",
                (
                    launch.close
                    for _mask, launch in previous
                    if hasattr(launch, "close")
                ),
            ):
                pass
        for mask, launch in bound[1]:
            if mask is None or phase < 0 or phase & mask:
                launch()

    def settle(states, phase):
        if closed:
            raise RuntimeError("Metal statistics program is closed")
        # The settle kernels read ``phase`` from the device controls, which
        # the runtime wrote before this call.
        del phase
        if not settled or settled[0] is not states:
            previous = settled[1] if settled else ()
            settled[:] = [states, bind(states, programs[sample_count:])]
            with cleanup_on_exit(
                "Metal settle bindings",
                (
                    launch.close
                    for _mask, launch in previous
                    if hasattr(launch, "close")
                ),
            ):
                pass
        for _mask, launch in settled[1]:
            launch()

    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        launches = tuple(
            launch
            for held in (bound, settled)
            if held
            for _mask, launch in held[1]
            if hasattr(launch, "close")
        )
        bound.clear()
        settled.clear()
        extra.clear()
        with cleanup_on_exit(
            "Metal statistics program", (launch.close for launch in launches)
        ):
            pass

    saved = None
    if context.save_kernels:
        # A program saves its effective unit (with hp helpers) and sidecar.
        if programs:
            saved = programs[0][1].save_source(source_path(context, ".metal"))
        else:
            saved = save_source(context, source, ".metal")
    return CompiledStatistics(
        lowering=plan.lowering,
        function=internal_update_statistics,
        settle=settle if sample_count < len(kernels) else None,
        module=None,
        saved_kernel_file=saved,
        close=close,
    )

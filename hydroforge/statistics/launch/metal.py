"""Metal statistics: one MSL library, argument bindings reused per states.

The plan's kernels map one thread to each saved point and loop over members
and levels.  Every kernel of the program shares the printed source, so the
library compiles once; the launches of one bound state mapping keep their
argument bindings until another mapping replaces them.
"""

from __future__ import annotations

from typing import Any

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.kernels.codegen.c import MSL, msl_arguments
from hydroforge.kernels.metal import MetalArgument, MetalProgram
from hydroforge.platform.backend import METAL
from hydroforge.statistics.kernel_plan import (
    StatisticsCompileContext,
    StatisticsKernelPlan,
)
from hydroforge.statistics.launch import CompiledStatistics, save_source

# The launch value naming a program's thread count; no argument field.
_THREADS = "__threads"


def compile_statistics(
    context: StatisticsCompileContext, plan: StatisticsKernelPlan
) -> CompiledStatistics:
    kernels = plan.kernels("loop")
    source = MSL.program([kernel.function for kernel in kernels])
    programs = []
    for kernel in kernels:
        fields = msl_arguments(kernel.function)
        program = MetalProgram(
            source,
            kernel.function.name,
            tuple(MetalArgument(*field) for field in fields),
            extent=(_THREADS,),
        )
        names = tuple(zip(kernel.function.params, (field[0] for field in fields)))
        programs.append((kernel, program, names))

    def bind(states):
        launches = []
        for kernel, program, names in programs:
            values = {
                field: states[param.name]
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

    def internal_update_statistics(states, BLOCK_SIZE, phase):
        # The launches, and so the argument bindings, of one bound state
        # mapping serve every sample; Metal launches a fixed width.
        del BLOCK_SIZE
        if not bound or bound[0] is not states:
            previous = bound[1] if bound else ()
            bound[:] = [states, bind(states)]
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

    return CompiledStatistics(
        lowering=plan.lowering,
        function=internal_update_statistics,
        settle=None,
        module=None,
        saved_kernel_file=(
            save_source(context, source, ".metal") if context.save_kernels else None
        ),
    )

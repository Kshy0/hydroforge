"""Compile explicit statistics inputs into an uninstalled backend program."""

from dataclasses import replace
from functools import partial

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.kernels.toolchain.python import release_generated_module
from hydroforge.statistics.ir import StatisticsIR
from hydroforge.statistics.kernel_plan import StatisticsCompileContext, plan_statistics
from hydroforge.statistics.launch import CompiledStatistics, cuda, metal, torch, triton
from hydroforge.statistics.lowering import lower_statistics

_LAUNCHERS = {
    "cuda": cuda.compile_statistics,
    "msl": metal.compile_statistics,
    "torch": torch.compile_statistics,
    "triton": triton.compile_statistics,
}


def compile_statistics_program(
    context: StatisticsCompileContext,
    ir: StatisticsIR,
    *,
    dialect: str,
) -> CompiledStatistics:
    """Emit the statistics program in the backend's framework dialect."""

    lowering = lower_statistics(ir)
    plan = plan_statistics(context, lowering)
    compiled = _LAUNCHERS[dialect](context, plan)
    if compiled.settle is not None or not lowering.compound:
        return compiled
    # Other backends fold a close without a sample through the PyTorch
    # settle kernels over the same storage.
    try:
        settle, generated = torch.compile_settle(context, plan)
    except BaseException:
        with cleanup_on_exit(
            "statistics settle compilation",
            (
                compiled.close,
                *(
                    partial(release_generated_module, name, filename)
                    for name, filename in compiled.generated_modules
                ),
            ),
        ):
            raise
    return replace(
        compiled,
        settle=settle,
        generated_modules=(*compiled.generated_modules, generated),
    )

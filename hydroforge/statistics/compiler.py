"""Compile explicit statistics inputs into an uninstalled backend program."""

from hydroforge.contracts.errors import cleanup_on_exit
from hydroforge.statistics.emitters.common import (
    CompiledStatistics,
    StatisticsCompileContext,
)
from hydroforge.statistics.emitters.cuda import CudaStatisticsEmitter
from hydroforge.statistics.emitters.metal import MetalStatisticsEmitter
from hydroforge.statistics.emitters.torch import TorchStatisticsEmitter
from hydroforge.statistics.emitters.triton import TritonStatisticsEmitter
from hydroforge.statistics.ir import StatisticsIR
from hydroforge.statistics.lowering import lower_statistics

_EMITTERS = {
    "torch": TorchStatisticsEmitter,
    "triton": TritonStatisticsEmitter,
    "cuda": CudaStatisticsEmitter,
    "metal": MetalStatisticsEmitter,
}


def compile_statistics_program(
    context: StatisticsCompileContext,
    ir: StatisticsIR,
    *,
    backend: str,
) -> CompiledStatistics:
    emitter = _EMITTERS[backend](context, lower_statistics(ir))
    try:
        return emitter.emit()
    except BaseException:
        with cleanup_on_exit(
            "statistics compilation", (emitter.release_generated_modules,)
        ):
            raise

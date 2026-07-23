"""Compile a statistics specification into one backend program."""

from __future__ import annotations

from typing import Any

from hydroforge.statistics.emitters.common import CompiledStatistics
from hydroforge.statistics.emitters.cuda import CudaStatisticsEmitter
from hydroforge.statistics.emitters.metal import MetalStatisticsEmitter
from hydroforge.statistics.emitters.torch import TorchStatisticsEmitter
from hydroforge.statistics.emitters.triton import TritonStatisticsEmitter
from hydroforge.statistics.ir import build_statistics_ir
from hydroforge.statistics.lowering import lower_statistics


_EMITTERS = {
    "torch": TorchStatisticsEmitter,
    "triton": TritonStatisticsEmitter,
    "cuda": CudaStatisticsEmitter,
    "metal": MetalStatisticsEmitter,
}


class StatisticsCompiler:
    """Initialization-only compiler with explicit backend emitter ownership."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def compile(self) -> CompiledStatistics:
        owner = self.owner
        backend = self._backend()
        ir = build_statistics_ir(owner)
        lowering = lower_statistics(ir, num_trials=owner.num_trials)
        owner._statistics_ir = ir
        owner._statistics_lowering = lowering
        result = _EMITTERS[backend](owner, ir, lowering).emit()
        owner._aggregator_function = result.function
        owner._kernel_module = result.module
        owner._aggregator_generated = result.generated
        owner._saved_kernel_file = result.saved_kernel_file
        return result

    def _backend(self) -> str:
        owner = self.owner
        device_type = owner.device.type
        if device_type == "cpu":
            return "torch"
        if device_type == "mps":
            return "metal"
        if device_type != "cuda":
            return "torch"
        if owner.backend not in {"cuda", "triton", "torch"}:
            raise ValueError(
                f"Backend {owner.backend!r} cannot run CUDA tensors."
            )
        return owner.backend

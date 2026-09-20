"""Compile a statistics specification into one backend program."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from hydroforge.statistics.emitters.common import (
    CompiledStatistics,
    StatisticsCompileContext,
)
from hydroforge.statistics.emitters.cuda import CudaStatisticsEmitter
from hydroforge.statistics.emitters.metal import MetalStatisticsEmitter
from hydroforge.statistics.emitters.torch import TorchStatisticsEmitter
from hydroforge.statistics.emitters.triton import TritonStatisticsEmitter
from hydroforge.statistics.ir import build_statistics_ir
from hydroforge.statistics.lowering import lower_statistics

if TYPE_CHECKING:
    from hydroforge.statistics.runtime import StatisticsRuntime


_EMITTERS = {
    "torch": TorchStatisticsEmitter,
    "triton": TritonStatisticsEmitter,
    "cuda": CudaStatisticsEmitter,
    "metal": MetalStatisticsEmitter,
}


class StatisticsCompiler:
    """Initialization-only compiler with explicit backend emitter ownership."""

    def __init__(self, owner: StatisticsRuntime) -> None:
        self.owner = owner

    def compile(self) -> CompiledStatistics:
        owner = self.owner
        backend = self._backend()
        ir = build_statistics_ir(owner)
        lowering = lower_statistics(ir)
        owner._statistics_ir = ir
        owner._statistics_lowering = lowering
        context = StatisticsCompileContext(
            device=owner.device,
            rank=owner.rank,
            ensemble_size=owner.ensemble_size,
            save_kernels=owner.save_kernels,
            kernels_dir=owner.kernels_dir,
            variables=frozenset(owner._variables),
            metadata=MappingProxyType(
                {
                    name: MappingProxyType(dict(values))
                    for name, values in owner._metadata.items()
                }
            ),
            layouts=MappingProxyType(dict(owner._statistics_layouts)),
            storage=MappingProxyType(dict(owner._storage)),
            tensors=MappingProxyType(dict(owner._tensor_registry)),
            symbol_names=MappingProxyType(dict(owner._safe_name_cache)),
            control_dtype=owner._statistics_control_dtype(),
        )
        emitter = _EMITTERS[backend](context, lowering)
        try:
            result = emitter.emit()
        except BaseException:
            emitter.release_generated_modules()
            raise
        owner._generated_modules.extend(result.generated_modules)
        owner._aggregator_function = result.function
        owner._kernel_module = result.module
        owner._saved_kernel_file = result.saved_kernel_file
        return result

    def _backend(self) -> str:
        owner = self.owner
        device_type = owner.device.type
        if device_type == "cpu":
            return "torch"
        if device_type == "mps":
            return "metal"
        return owner.backend

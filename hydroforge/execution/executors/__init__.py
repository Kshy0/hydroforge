"""Loop executors: the one place execution selects a backend path.

:class:`EagerExecutor` launches recorded programs directly under host
control, :class:`CudaGraphExecutor` replays captured CUDA graphs (a fixed
loop's iteration graph once per iteration; adaptive and predicate loops as
device-side conditional graphs), and :class:`MetalIcbExecutor` replays
online-lowered Metal command buffers.
"""

from __future__ import annotations

from typing import Any

from hydroforge.execution.executors.base import LoopExecutor
from hydroforge.execution.executors.cuda_graph import CudaGraphExecutor
from hydroforge.execution.executors.eager import EagerExecutor
from hydroforge.execution.executors.metal_icb import MetalIcbExecutor
from hydroforge.kernels.toolchain.cuda import conditional_graphs


def select_executor(plan: Any) -> LoopExecutor:
    """Return the executor of a model plan's backend and capture policy.

    A CUDA graph executor also needs conditional WHILE graphs on the device.
    """

    device = plan.device
    options = {"world_size": plan.world_size}
    if plan.capture:
        capture = plan.backend.capture
        expected = {"cuda_graph": "cuda", "metal_icb": "mps"}.get(capture)
        if expected is not None and device.type != expected:
            # Capture is an auto-mode preference, not a backend device contract.
            # In particular, Triton also supports XPU without CUDA graphs.
            return EagerExecutor(device, **options)
        if capture == "cuda_graph" and conditional_graphs(device):
            return CudaGraphExecutor(device, **options)
        if capture == "metal_icb":
            return MetalIcbExecutor(device, **options)
    return EagerExecutor(device, **options)


__all__ = [
    "CudaGraphExecutor",
    "EagerExecutor",
    "LoopExecutor",
    "MetalIcbExecutor",
    "select_executor",
]

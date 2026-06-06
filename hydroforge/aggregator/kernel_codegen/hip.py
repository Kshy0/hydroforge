# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator


class HipCodegenMixin:
    """HIP C++ extension code generation for statistics aggregation."""

    def _generate_hip_aggregator_function(self: StatisticsAggregator) -> None:
        self._generate_cuda_like_aggregator_function(
            backend="hip",
            loader_factory=self._hip_extension_loader,
            cflags=("-O3", "-ffast-math"),
            source_transform=self._hip_source_transform,
        )

    def _hip_extension_loader(self):
        from hydroforge.runtime.hip_kernel import load_inline_hip_module

        return load_inline_hip_module

    def _hip_source_transform(self, cpp_sources: str, cuda_sources: str) -> tuple[str, str]:
        replacements = (
            ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
            ("#include <c10/cuda/CUDAStream.h>", "#include <c10/hip/HIPStream.h>"),
            ("cudaStream_t", "hipStream_t"),
            ("c10::cuda::getCurrentCUDAStream()", "c10::hip::getCurrentHIPStream()"),
        )
        for old, new in replacements:
            cuda_sources = cuda_sources.replace(old, new)
        return cpp_sources, cuda_sources

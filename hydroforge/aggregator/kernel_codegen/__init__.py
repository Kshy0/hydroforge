# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator

from hydroforge.aggregator.kernel_codegen.common import CommonCodegenMixin
from hydroforge.aggregator.kernel_codegen.cuda import CudaCodegenMixin
from hydroforge.aggregator.kernel_codegen.metal import MetalCodegenMixin
from hydroforge.aggregator.kernel_codegen.pytorch import PyTorchCodegenMixin
from hydroforge.aggregator.kernel_codegen.static import StaticCodegenMixin
from hydroforge.aggregator.kernel_codegen.triton import TritonCodegenMixin


class KernelCodegenMixin(
    MetalCodegenMixin,
    PyTorchCodegenMixin,
    CudaCodegenMixin,
    StaticCodegenMixin,
    TritonCodegenMixin,
    CommonCodegenMixin,
):
    """Mixin providing multi-backend kernel code generation and compilation.

    Backends: CUDA C++ extensions (which PyTorch hipifies under ROCm), Triton,
    PyTorch (torch.compile), and Metal MSL.
    """

    def _generate_aggregator_function(self: StatisticsAggregator) -> None:
        """
        Generate and compile the aggregation kernel function.

        Dispatches to CUDA C++, Metal, PyTorch, or Triton code generation
        based on HYDROFORGE_BACKEND.
        """
        from hydroforge.runtime.backend import BackendRegistry

        registry = BackendRegistry(
            {
                "torch": lambda: self._generate_pytorch_aggregator_function,
                "metal": lambda: self._generate_metal_aggregator_function,
                "cuda": lambda: self._generate_cuda_aggregator_function,
                "triton": lambda: self._generate_triton_aggregator_function,
            },
            name="statistics code generator",
        )
        from hydroforge.runtime.backend import KERNEL_BACKEND

        device_type = self.device.type
        if device_type == "cpu":
            backend = "torch"
        elif device_type == "mps":
            backend = "metal"
        elif device_type == "cuda":
            if KERNEL_BACKEND not in {"cuda", "triton", "torch"}:
                raise ValueError(
                    f"Backend {KERNEL_BACKEND!r} cannot run CUDA tensors."
                )
            backend = KERNEL_BACKEND
        else:
            backend = "torch"

        generator = registry.resolve(backend)
        generator()

    def _generate_triton_aggregator_function(self: StatisticsAggregator) -> None:
        """Generate and compile the Triton aggregation kernel function."""
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        tensor_info, grouped_by_output_index = self._analyze_tensor_info()

        # Generate kernel code
        kernel_code_lines = self._generate_kernel_header()

        # Generate scatter pre-step kernels
        self._generate_scatter_kernels(kernel_code_lines, grouped_by_output_index)

        # Generate kernels for each output_index/full-output group
        for output_index, var_list in grouped_by_output_index.items():
            if output_index == "__full__":
                self._generate_full_kernel_for_group(kernel_code_lines, output_index, var_list, tensor_info)
                continue
            kernel_name = f"kernel_{output_index}"
            self._generate_kernel_for_group(kernel_code_lines, kernel_name, output_index, var_list, tensor_info)

        # Generate main function
        self._generate_main_function(kernel_code_lines, grouped_by_output_index, tensor_info)

        # Write kernel code to temporary file and import
        kernel_code = "\n".join(kernel_code_lines)
        self._write_and_import_kernels(kernel_code)

        # Save kernel file for external inspection if enabled
        if self.save_kernels:
            self._save_kernel_file(kernel_code)

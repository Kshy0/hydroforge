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
from hydroforge.aggregator.kernel_codegen.hip import HipCodegenMixin
from hydroforge.aggregator.kernel_codegen.metal import MetalCodegenMixin
from hydroforge.aggregator.kernel_codegen.pytorch import PyTorchCodegenMixin
from hydroforge.aggregator.kernel_codegen.triton import TritonCodegenMixin


class KernelCodegenMixin(
    MetalCodegenMixin,
    HipCodegenMixin,
    CudaCodegenMixin,
    PyTorchCodegenMixin,
    TritonCodegenMixin,
    CommonCodegenMixin,
):
    """Mixin providing multi-backend kernel code generation and compilation.

    Backends: Triton, PyTorch (torch.compile), CUDA C++, Metal MSL.
    """

    def _generate_aggregator_function(self: StatisticsAggregator) -> None:
        """
        Generate and compile the aggregation kernel function.
        
        Dispatches to CUDA C++, Metal, PyTorch, or Triton code generation
        based on HYDROFORGE_BACKEND.

        For the ``cuda`` backend, CUDA C++ codegen is always used.  The
        compiled shared library is cached via ``build_directory`` so that
        subsequent runs (or ``compile_only`` pre-compilation) avoid the
        ~27 s cold-start nvcc penalty.
        """
        from hydroforge.runtime.backend import KERNEL_BACKEND

        if KERNEL_BACKEND == "torch":
            self._generate_pytorch_aggregator_function()
            return

        if KERNEL_BACKEND == "cuda":
            self._generate_cuda_aggregator_function()
            return

        if KERNEL_BACKEND == "hip":
            self._generate_hip_aggregator_function()
            return

        if KERNEL_BACKEND == "metal":
            self._generate_metal_aggregator_function()
            return

        # ── Triton path (default for 'triton' backend, also used by 'cuda') ──
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        tensor_info, grouped_by_save_idx = self._analyze_tensor_info()
        
        # Generate kernel code
        kernel_code_lines = self._generate_kernel_header()
        
        # Generate kernels for each save_idx group
        for save_idx, var_list in grouped_by_save_idx.items():
            kernel_name = f"kernel_{save_idx}"
            self._generate_kernel_for_group(kernel_code_lines, kernel_name, save_idx, var_list, tensor_info)
        
        # Generate main function
        self._generate_main_function(kernel_code_lines, grouped_by_save_idx, tensor_info)
        
        # Write kernel code to temporary file and import
        kernel_code = "\n".join(kernel_code_lines)
        self._write_and_import_kernels(kernel_code)
        
        # Save kernel file for external inspection if enabled
        if self.save_kernels:
            self._save_kernel_file(kernel_code)

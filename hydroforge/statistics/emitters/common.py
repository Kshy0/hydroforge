# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from math import prod
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch

if TYPE_CHECKING:
    from hydroforge.statistics.layout import StatisticsVariableLayout
    from hydroforge.statistics.lowering import StatisticsLowering

from hydroforge.compiler.generated import (
    compile_generated_module,
    release_generated_module,
)
from hydroforge.contracts.naming import sanitize_symbol
from hydroforge.serialization.files import atomic_write_text


@dataclass(frozen=True, slots=True)
class StatisticsCompileContext:
    """Explicit read-only bindings needed by backend code generation.

    Tensor handles refer to runtime-owned storage; emitters neither allocate
    model state nor retain the model or its lifecycle service.
    """

    device: torch.device
    rank: int
    ensemble_size: int
    save_kernels: bool
    kernels_dir: Path | None
    variables: frozenset[str]
    metadata: Mapping[str, Mapping[str, Any]]
    layouts: Mapping[str, StatisticsVariableLayout]
    storage: Mapping[str, torch.Tensor]
    tensors: Mapping[str, torch.Tensor]
    symbol_names: Mapping[str, str]
    control_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class CompiledStatistics:
    """Backend program produced by one statistics emitter."""

    function: Callable[..., None]
    module: object | None
    saved_kernel_file: Path | None
    generated_modules: tuple[tuple[str, str], ...] = ()


class StatisticsEmitter:
    """Explicit initialization-only context shared by backend emitters."""

    def __init__(
        self, context: StatisticsCompileContext, lowering: StatisticsLowering
    ) -> None:
        self.device = context.device
        self.rank = context.rank
        self.ensemble_size = context.ensemble_size
        self.save_kernels = context.save_kernels
        self.kernels_dir = context.kernels_dir
        self._variables = context.variables
        self._metadata = context.metadata
        self._statistics_layouts = context.layouts
        self._storage = context.storage
        self._tensor_registry = context.tensors
        self._safe_name_cache = dict(context.symbol_names)
        self._generated_modules: list[tuple[str, str]] = []
        self._statistics_ir = lowering.ir
        self._statistics_lowering = lowering
        self._control_dtype = context.control_dtype
        self._kernel_module = None
        self._saved_kernel_file = None

    def result(self) -> CompiledStatistics:
        return CompiledStatistics(
            function=self._aggregator_function,
            module=self._kernel_module,
            saved_kernel_file=self._saved_kernel_file,
            generated_modules=tuple(self._generated_modules),
        )

    def release_generated_modules(self) -> None:
        """Discard untransferred modules after an unsuccessful compilation."""

        for name, filename in self._generated_modules:
            release_generated_module(name, filename)
        self._generated_modules.clear()

    def _get_safe_name(self, name: str) -> str:
        if name not in self._safe_name_cache:
            self._safe_name_cache[name] = sanitize_symbol(name)
        return self._safe_name_cache[name]

    def _stride_input(self, name: str) -> int:
        """Return one output variable's member stride from its compiled layout."""
        return int(self._statistics_layouts[name].stride_input)

    def _source_stride(self, name: str, *, logical_rank: int = 1) -> int:
        """Return the logical-axis stride for one materialized input buffer."""
        tensor = (
            self._tensor_registry[name]
            if name in self._tensor_registry
            else self._storage[name]
        )
        if self.ensemble_size > 1 and tensor.ndim == logical_rank + 1:
            return int(tensor.shape[1])
        return 0

    def _full_source_offset(self, name: str, output: str, index: str) -> str:
        """Address shared leaves within the full output's per-member spatial plane."""
        layout = self._statistics_layouts[output]
        if not layout.batched or self._source_stride(
            name, logical_rank=layout.actual_ndim - 1
        ):
            return index
        plane = max(1, prod(layout.actual_shape[1:]))
        return f"({index} % {plane})"

    def _generate_unique_name(self) -> str:
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{timestamp}_r{self.rank}_{uuid4().hex}"

    def _save_kernel_file(self, kernel_code: str) -> None:
        """
        Save the generated kernel code to a permanent file for inspection.

        Args:
            kernel_code: Generated kernel code as string
        """
        # Use unique name generation
        unique_name = self._generate_unique_name()
        filename = f"kern_{unique_name}.py"

        self._saved_kernel_file = self.kernels_dir / filename

        atomic_write_text(self._saved_kernel_file, kernel_code)

    def _compile_generated_module(
        self,
        kernel_code: str,
        *,
        prefix: str,
    ) -> ModuleType:
        """Compile generated source in memory with an inspectable lifetime."""
        unique = self._generate_unique_name()
        module_name = f"hydroforge_{prefix}_r{self.rank}_{unique}"
        with warnings.catch_warnings():
            # Ignore the unrelated torch.jit warning raised by torch.compile.
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.jit\.script_method` is not supported.*",
                category=DeprecationWarning,
                module=r"torch\.jit\._script",
            )
            module = compile_generated_module(kernel_code, name=module_name)
        self._generated_modules.append((module_name, module.__file__))
        return module

    def _compile_generated_kernels(self, kernel_code: str) -> None:
        module = self._compile_generated_module(
            kernel_code,
            prefix="statistics",
        )
        self._kernel_module = module
        self._aggregator_function = getattr(
            module,
            "internal_update_statistics",
        )

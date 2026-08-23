# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

from dataclasses import dataclass
import linecache
import hashlib
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from hydroforge.contracts.naming import sanitize_symbol
from hydroforge.serialization.files import atomic_write_text


@dataclass(frozen=True, slots=True)
class CompiledStatistics:
    """Backend program produced by one statistics emitter."""

    function: Any
    module: ModuleType | Any | None
    saved_kernel_file: Path | None


class StatisticsEmitter:
    """Explicit initialization-only context shared by backend emitters."""

    def __init__(self, owner: Any, lowering: Any) -> None:
        self.device = owner.device
        self.rank = owner.rank
        self.num_trials = owner.num_trials
        self.save_kernels = owner.save_kernels
        self.kernels_dir = owner.kernels_dir
        self._variables = owner._variables
        self._metadata = owner._metadata
        self._statistics_layouts = owner._statistics_layouts
        self._storage = owner._storage
        self._tensor_registry = owner._tensor_registry
        self._safe_name_cache = owner._safe_name_cache
        self._generated_modules = owner._generated_modules
        self._statistics_ir = lowering.ir
        self._statistics_lowering = lowering
        self._control_dtype = owner._statistics_control_dtype()
        self._kernel_module = None
        self._saved_kernel_file = None

    def result(self) -> CompiledStatistics:
        return CompiledStatistics(
            function=self._aggregator_function,
            module=self._kernel_module,
            saved_kernel_file=self._saved_kernel_file,
        )

    def _get_safe_name(self, name: str) -> str:
        if name not in self._safe_name_cache:
            self._safe_name_cache[name] = sanitize_symbol(name)
        return self._safe_name_cache[name]

    def _stride_input(self, name: str) -> int:
        """Return one output variable's trial stride from its compiled layout."""
        return int(self._statistics_layouts[name].stride_input)

    def _source_stride(self, name: str) -> int:
        """Return the logical-axis stride for one materialized input buffer."""
        tensor = (
            self._tensor_registry[name]
            if name in self._tensor_registry
            else self._storage[name]
        )
        if self.num_trials > 1 and tensor.ndim >= 2:
            return int(tensor.shape[1])
        return 0

    def _generate_unique_name(self) -> str:
        timestamp = datetime.now().strftime("%H%M%S")
        seed = f"{self.rank}_{timestamp}_{random.randint(1000, 9999)}"
        return f"{timestamp}_r{self.rank}_{hashlib.md5(seed.encode()).hexdigest()[:6]}"

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
        filename = f"<{module_name}>"
        lines = kernel_code.splitlines(keepends=True)
        if kernel_code and not kernel_code.endswith("\n"):
            lines[-1] += "\n"
        linecache.cache[filename] = (
            len(kernel_code), None, lines, filename,
        )
        module = ModuleType(module_name)
        module.__file__ = filename
        module.__package__ = ""
        sys.modules[module_name] = module
        try:
            with warnings.catch_warnings():
                # Ignore the unrelated torch.jit warning raised by torch.compile.
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch\.jit\.script_method` is not supported.*",
                    category=DeprecationWarning,
                    module=r"torch\.jit\._script",
                )
                exec(compile(kernel_code, filename, "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(module_name, None)
            linecache.cache.pop(filename, None)
            raise
        self._generated_modules.append((module_name, filename))
        return module

    def _compile_generated_kernels(self, kernel_code: str) -> None:
        module = self._compile_generated_module(
            kernel_code, prefix="statistics",
        )
        self._kernel_module = module
        self._aggregator_function = getattr(
            module, "internal_update_statistics",
        )

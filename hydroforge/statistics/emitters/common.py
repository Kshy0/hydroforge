# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

import linecache
import hashlib
import random
import sys
import warnings
from dataclasses import dataclass
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
    generated: bool
    saved_kernel_file: Path | None


class StatisticsEmitter:
    """Explicit initialization-only context shared by backend emitters."""

    def __init__(self, owner: Any, ir: Any, lowering: Any) -> None:
        self.device = owner.device
        self.rank = owner.rank
        self.num_trials = owner.num_trials
        self.save_kernels = owner.save_kernels
        self.kernels_dir = owner.kernels_dir
        self._variables = owner._variables
        self._metadata = owner._metadata
        self._storage = owner._storage
        self._tensor_registry = owner._tensor_registry
        self._safe_name_cache = owner._safe_name_cache
        self._generated_modules = owner._generated_modules
        self._statistics_ir = ir
        self._statistics_lowering = lowering
        self._kernel_module = None
        self._aggregator_function = None
        self._aggregator_generated = False
        self._saved_kernel_file = None

    def result(self) -> CompiledStatistics:
        return CompiledStatistics(
            function=self._aggregator_function,
            module=self._kernel_module,
            generated=self._aggregator_generated,
            saved_kernel_file=self._saved_kernel_file,
        )

    def _get_safe_name(self, name: str) -> str:
        if name not in self._safe_name_cache:
            self._safe_name_cache[name] = sanitize_symbol(name)
        return self._safe_name_cache[name]

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
        if self.kernels_dir is None:
            raise RuntimeError("save_kernels requires a kernels directory")
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
                # PyTorch 2.x imports its legacy MKLDNN ScriptModule while
                # initializing torch.compile on Python 3.14. HydroForge does
                # not use that API; suppress only this upstream import warning.
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
        self._aggregator_generated = True

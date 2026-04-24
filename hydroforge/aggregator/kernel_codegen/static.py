# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#


from __future__ import annotations

import importlib.util
import sys
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydroforge.aggregator.aggregator import StatisticsAggregator


class StaticCodegenMixin:
    """Codegen for the one-shot static-variable gather kernel.

    ``static_vars`` entries carry a raw tensor plus an optional
    ``save_idx`` index tensor.  This mixin emits a tiny ``torch.compile``
    wrapper to a generated .py file and imports it — matching the
    write-file-and-import convention used by the other backend mixins —
    so the gather stays consistent with the rest of hydroforge's
    codegen-driven kernels and is inspectable when ``save_kernels=True``.
    """

    def _generate_static_gather_function(self: StatisticsAggregator) -> None:
        """Generate, write, and import the static-var gather kernel.

        Binds the compiled callable to ``self._static_gather_function``.
        The kernel is invoked once per ``(tensor, save_idx)`` pair at
        aggregator init; callers are responsible for detaching and
        moving the result to CPU / numpy.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kernel_code = "\n".join([
            '"""',
            'Auto-generated static-variable gather kernel for hydroforge.',
            f'Generated at: {timestamp}',
            f'Rank: {self.rank}',
            f'Device: {self.device}',
            '',
            'One-shot gather used at aggregator init to materialise',
            'static_vars (raw tensor + optional save_idx index tensor)',
            'into saved-point order for the NetCDF writer.',
            '"""',
            '',
            'from typing import Optional',
            '',
            'import torch',
            '',
            '',
            '@torch.compile(dynamic=True)',
            'def gather_static_var(tensor: torch.Tensor,',
            '                      save_idx: Optional[torch.Tensor]'
            ') -> torch.Tensor:',
            '    if save_idx is None:',
            '        return tensor',
            '    return tensor[save_idx]',
            '',
        ])

        unique_name = self._generate_unique_name()
        with tempfile.NamedTemporaryFile(
                mode='w', suffix=f'_static_{unique_name}.py',
                delete=False) as f:
            f.write(kernel_code)
            self._static_kernel_file = f.name

        module_name = f"aggr_static_r{self.rank}_{unique_name}"
        spec = importlib.util.spec_from_file_location(
            module_name, self._static_kernel_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        self._static_kernel_module = module
        self._static_gather_function = module.gather_static_var

        if self.save_kernels:
            saved = self.kernels_dir / f"kern_static_{unique_name}.py"
            saved.write_text(kernel_code, encoding='utf-8')

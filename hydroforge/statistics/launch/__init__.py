# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Backend programs of compiled statistics and what every backend returns.

The submodules are named after their dialects (``cuda``, ``metal``,
``torch``, ``triton``); this package never imports the libraries of those
names, so ``launch.torch`` and ``launch.triton`` are always the submodules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from hydroforge.io.files import atomic_write_text
from hydroforge.kernels.toolchain import CompileRequest

if TYPE_CHECKING:
    import torch

    from hydroforge.statistics.kernel_plan import StatisticsCompileContext
    from hydroforge.statistics.lowering import StatisticsLowering


def _no_requests(
    states: Mapping[str, torch.Tensor], block_size: int
) -> tuple[CompileRequest, ...]:
    del states, block_size
    return ()


def _no_close() -> None:
    pass


@dataclass(frozen=True, slots=True)
class CompiledStatistics:
    """One backend's statistics program.

    ``function(states, block_size, phase)`` runs one sample; a negative
    ``phase`` reads the sample phase from device controls.
    ``settle(states, phase)`` folds a close without a sample whose ``phase``
    the runtime has also written to the device controls; it is ``None``
    without compound statistics.  ``requests(states, block_size)``
    describes the native compilation the first sample would otherwise do.
    """

    lowering: StatisticsLowering
    function: Callable[..., None]
    settle: Callable[..., None] | None
    module: object | None
    saved_kernel_file: Path | None
    generated_modules: tuple[tuple[str, str], ...] = ()
    requests: Callable[..., tuple[CompileRequest, ...]] = _no_requests
    close: Callable[[], None] = _no_close


def unique_name(rank: int) -> str:
    """A process-unique name for generated code of one rank."""

    return f"{datetime.now().strftime('%H%M%S')}_r{rank}_{uuid4().hex}"


def source_path(context: StatisticsCompileContext, suffix: str) -> Path:
    """A fresh path for generated source under the kernels directory."""

    return context.kernels_dir / f"kern_{unique_name(context.rank)}{suffix}"


def save_source(context: StatisticsCompileContext, text: str, suffix: str) -> Path:
    """Write generated source for inspection under the kernels directory."""

    path = source_path(context, suffix)
    atomic_write_text(path, text)
    return path

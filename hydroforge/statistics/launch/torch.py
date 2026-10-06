# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""PyTorch statistics: one generated module of vectorized kernels.

The plan's kernels run on flat views of the bound state mapping, made once
per mapping; each output group loops over members and levels in Python and
vectorizes its saved points (``loop``), and scatter adds vectorize their
source points.  Sizes of the compiled topology are literals of the source,
and binding checks that the state mapping still has them.  Programs without
full-layout outputs run under ``torch.compile``, one entry per reachable
sample phase (:data:`~hydroforge.statistics.phases.SAMPLE_PHASES` reduced to
the tested bits), each compiled on first use: one frame per phase keeps every
compilation within Dynamo's per-frame recompile limit.  Full-layout programs,
measured on this reference backend, only recovered their compile cost on
long runs of large CPU grids.

The CUDA and Triton programs close a window without a sample through this
module's settle kernels (:func:`compile_settle`), which run eagerly on the
same storage; Metal prints its own.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from hydroforge.kernels.codegen.ir import PhaseTest
from hydroforge.kernels.codegen.passes import subexpressions
from hydroforge.kernels.codegen.torch import PRELUDE, TorchPrinter
from hydroforge.kernels.toolchain.python import (
    compile_generated_module,
    release_generated_module,
)
from hydroforge.statistics.kernel_plan import (
    StatisticsCompileContext,
    StatisticsKernel,
    StatisticsKernelPlan,
)
from hydroforge.statistics.launch import CompiledStatistics, save_source, unique_name
from hydroforge.statistics.phases import CONTROL_PHASE, KERNEL_CONTROLS, SAMPLE_PHASES


def _phase_bits(kernels: Iterable[StatisticsKernel]) -> int:
    """The phase bits the kernels and their launch gates test."""

    bits = 0
    pending: list = []
    for kernel in kernels:
        bits |= kernel.phase_mask or 0
        pending.extend(kernel.function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, PhaseTest):
            bits |= node.bits
        pending.extend(subexpressions(node))
    return bits


def _program_mask(kernels: Iterable[StatisticsKernel]) -> int | None:
    """The phase bits any kernel launches at; ``None`` for every phase."""

    mask = 0
    for kernel in kernels:
        if kernel.phase_mask is None:
            return None
        mask |= kernel.phase_mask
    return mask


def _entry(
    name: str, kernels: Iterable[StatisticsKernel], phase: int | None = None
) -> list[str]:
    """``name(states, phase)``, or ``name_{phase}(states)`` for one phase."""

    if phase is None:
        lines = [f"def {name}(states, phase):"]
    else:
        lines = [f"def {name}_{phase}(states):"]
    for kernel in kernels:
        mask = kernel.phase_mask
        if phase is None:
            call = f"{kernel.function.name}(states, phase)"
            if mask is None:
                lines.append(f"    {call}")
            else:
                lines.extend((f"    if (phase & {mask}) != 0:", f"        {call}"))
        elif mask is None or phase & mask:
            lines.append(f"    {kernel.function.name}(states, {phase})")
    return lines if len(lines) > 1 else [*lines, "    pass"]


def _sizes(context: StatisticsCompileContext) -> dict[str, int]:
    """Element counts of every buffer a statistics kernel binds."""

    return {
        **{name: tensor.numel() for name, tensor in context.storage.items()},
        **{name: tensor.numel() for name, tensor in context.tensors.items()},
        **dict.fromkeys(KERNEL_CONTROLS, 1),
    }


def _source(
    context: StatisticsCompileContext,
    entries: Mapping[str, tuple[StatisticsKernel, ...]],
    phases: Iterable[int] = (),
) -> str:
    """The module: prelude, device, every kernel and one entry per name,
    plus an ``update_{phase}`` entry per phase of ``phases``."""

    printer = TorchPrinter("device")
    sizes = _sizes(context)
    kernels = {
        kernel.function.name: kernel for group in entries.values() for kernel in group
    }
    blocks = [
        PRELUDE.rstrip("\n"),
        f"device = torch.device({str(context.device)!r})",
        *(
            printer.kernel(
                kernel.function,
                sizes=sizes,
                scalars={
                    name: count.value(context.tensors)
                    for name, count in kernel.scalars.items()
                },
            )
            for kernel in kernels.values()
        ),
        *("\n".join(_entry(name, group)) for name, group in entries.items()),
        *("\n".join(_entry("update", entries["update"], phase)) for phase in phases),
    ]
    return "\n\n\n".join(blocks) + "\n"


def _module(context: StatisticsCompileContext, source: str, prefix: str):
    name = f"hydroforge_{prefix}_{unique_name(context.rank)}"
    with warnings.catch_warnings():
        # Ignore the unrelated torch.jit warning raised by torch.compile.
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script_method` is not supported.*",
            category=DeprecationWarning,
            module=r"torch\.jit\._script",
        )
        module = compile_generated_module(source, name=name)
    return module, (name, module.__file__)


def _binder(
    context: StatisticsCompileContext, kernels: Iterable[StatisticsKernel]
) -> Callable[[Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Flat views of a state mapping, checked against the compiled sizes."""

    params = {
        param.name: param
        for kernel in kernels
        for param in kernel.function.params
        if param.access is not None
    }
    sizes = _sizes(context)

    def bind(states: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        flat = {}
        for name, param in params.items():
            tensor = states[name]
            if tensor.dtype != param.type:
                raise RuntimeError(f"{name} has unexpected dtype {tensor.dtype}")
            if name in sizes and tensor.numel() != sizes[name]:
                raise RuntimeError(
                    f"{name} has {tensor.numel()} elements; the statistics "
                    f"program was compiled for {sizes[name]}"
                )
            flat[name] = tensor.view(-1)
        return flat

    return bind


def compile_statistics(
    context: StatisticsCompileContext, plan: StatisticsKernelPlan
) -> CompiledStatistics:
    update = (*plan.scatter_kernels(unrolled=True), *plan.group_kernels("loop"))
    settle = plan.settle_kernels("loop")
    bits = _phase_bits(update)
    mask = _program_mask(update)
    compiled = not any(group.output_index is None for group in plan.groups)
    phases = sorted({phase & bits for phase in SAMPLE_PHASES}) if compiled else []
    bind = _binder(context, (*update, *settle))
    source = _source(context, {"update": update, "settle": settle}, phases)
    module, generated = _module(context, source, "statistics")
    try:
        saved = save_source(context, source, ".py") if context.save_kernels else None
    except BaseException:
        release_generated_module(*generated)
        raise
    bound: list[Any] = []
    runs: dict[int, Callable[[dict[str, torch.Tensor]], None]] = {}

    def flat(states):
        if not bound or bound[0] is not states:
            bound[:] = [states, bind(states)]
        return bound[1]

    def run(states, phase):
        if phase not in phases:
            module.update(states, phase)
            return
        entry = runs.get(phase)
        if entry is None:
            entry = runs[phase] = torch.compile(
                getattr(module, f"update_{phase}"), dynamic=False
            )
        entry(states)

    def internal_update_statistics(states, block_size, phase):
        # A negative phase is read once from the device control, outside
        # torch.compile, so exact values create no guards.
        del block_size
        if phase < 0:
            phase = int(states[CONTROL_PHASE].item())
        if mask is not None and not phase & mask:
            return
        run(flat(states), phase & bits)

    def internal_settle_statistics(states, phase):
        module.settle(flat(states), phase)

    return CompiledStatistics(
        lowering=plan.lowering,
        function=internal_update_statistics,
        settle=internal_settle_statistics if settle else None,
        module=module,
        saved_kernel_file=saved,
        generated_modules=(generated,),
    )


def compile_settle(
    context: StatisticsCompileContext, plan: StatisticsKernelPlan
) -> tuple[Callable[..., None], tuple[str, str]]:
    """The settle of another backend's program over the same storage, and
    its generated module."""

    kernels = plan.settle_kernels("loop")
    bind = _binder(context, kernels)
    module, generated = _module(
        context, _source(context, {"settle": kernels}), "statistics_settle"
    )
    bound: list[Any] = []

    def internal_settle_statistics(states, phase):
        if not bound or bound[0] is not states:
            bound[:] = [states, bind(states)]
        module.settle(bound[1], phase)

    return internal_settle_statistics, generated

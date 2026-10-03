"""PyTorch statistics: one generated module of vectorized kernels.

The plan's kernels run on flat views of the bound state mapping, made once
per mapping; each output group loops over members and levels in Python and
vectorizes its saved points (``loop``), and scatter adds vectorize their
source points.  Sizes of the compiled topology are literals of the source,
and binding checks that the state mapping still has them.  Programs without
full-layout outputs run under ``torch.compile`` with the sample phase
specialized: the phase takes few values, and full-layout programs, measured
on this reference backend, only recovered their compile cost on long runs
of large CPU grids.

Every backend closes a window without a sample through this module's settle
kernels (:func:`compile_settle`), which run eagerly on the same storage.
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
from hydroforge.statistics.phases import CONTROL_PHASE, KERNEL_CONTROLS, SampleFlags

_SETTLE = int(SampleFlags.INNER_LAST)
_SETTLE_OUTER_FIRST = int(SampleFlags.INNER_LAST | SampleFlags.OUTER_FIRST)


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


def _entry(name: str, kernels: Iterable[StatisticsKernel]) -> list[str]:
    lines = [f"def {name}(states, phase):"]
    for kernel in kernels:
        call = f"{kernel.function.name}(states, phase)"
        if kernel.phase_mask is None:
            lines.append(f"    {call}")
        else:
            lines.extend(
                (f"    if (phase & {kernel.phase_mask}) != 0:", f"        {call}")
            )
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
) -> str:
    """The module: prelude, device, every kernel and one entry per name."""

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
    ]
    return "\n\n\n".join(blocks) + "\n"


def _module(context: StatisticsCompileContext, source: str, prefix: str):
    name = f"hydroforge_{prefix}_r{context.rank}_{unique_name(context.rank)}"
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
    bind = _binder(context, (*update, *settle))
    source = _source(context, {"update": update, "settle": settle})
    module, generated = _module(context, source, "statistics")
    try:
        run = module.update
        if not any(group.output_index is None for group in plan.groups):
            run = torch.compile(run, dynamic=False)
        saved = save_source(context, source, ".py") if context.save_kernels else None
    except BaseException:
        release_generated_module(*generated)
        raise
    bound: list[Any] = []

    def flat(states):
        if not bound or bound[0] is not states:
            bound[:] = [states, bind(states)]
        return bound[1]

    def internal_update_statistics(states, BLOCK_SIZE, phase):
        # A negative phase is read once from the device control, outside
        # torch.compile, so exact values create no guards.
        del BLOCK_SIZE
        if phase < 0:
            phase = int(states[CONTROL_PHASE].item())
        run(flat(states), phase & bits)

    def internal_settle_statistics(states, is_outer_first):
        module.settle(flat(states), _SETTLE_OUTER_FIRST if is_outer_first else _SETTLE)

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

    def internal_settle_statistics(states, is_outer_first):
        if not bound or bound[0] is not states:
            bound[:] = [states, bind(states)]
        module.settle(bound[1], _SETTLE_OUTER_FIRST if is_outer_first else _SETTLE)

    return internal_settle_statistics, generated

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified four-backend compiler and launcher for generated kernel-IR programs.

Generated kernel-IR programs (streaming statistics, device step-clock fields)
share one compilation, IR optimization, caching and resource-ownership layer
across ``cuda`` (NVRTC/hiprtc), ``triton``, ``msl`` (Metal) and ``torch``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.io.files import atomic_write_text
from hydroforge.kernels.codegen.c import CUDA, MSL, msl_arguments
from hydroforge.kernels.codegen.ir import KernelFunction, PhaseTest
from hydroforge.kernels.codegen.msl import emulated_msl
from hydroforge.kernels.codegen.passes import Implies, hoist, sink, subexpressions
from hydroforge.kernels.codegen.torch import PRELUDE as TORCH_PRELUDE
from hydroforge.kernels.codegen.torch import TorchPrinter
from hydroforge.kernels.codegen.triton import PRINTER as TRITON
from hydroforge.kernels.metal import MetalArgument, MetalProgram
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain import cuda as rtc
from hydroforge.kernels.toolchain.python import (
    compile_generated_module,
    release_generated_module,
)
from hydroforge.kernels.toolchain.triton import (
    launch_options,
    warming_up,
    warmup_request,
)
from hydroforge.kernels.triton import launch_triton_kernel
from hydroforge.platform.backend import METAL
from hydroforge.platform.triton_driver import proven_triton_device, triton_call_device

_METAL_THREADS = "__threads"
_DIGEST_SUFFIXES = {"cuda": ".cu", "triton": ".py", "msl": ".metal", "torch": ".py"}


def _no_requests(
    states: Mapping[str, torch.Tensor], block_size: int
) -> tuple[CompileRequest, ...]:
    del states, block_size
    return ()


def _no_close() -> None:
    pass


def _eval_bound(spec: Any, states: Mapping[str, torch.Tensor]) -> Any:
    if hasattr(spec, "value"):
        return spec.value(states)
    if callable(spec):
        return spec(states)
    return spec


def unique_ir_name(rank: int = 0) -> str:
    """Return a process-unique module/file suffix for generated IR code."""

    return f"{datetime.now().strftime('%H%M%S')}_r{rank}_{uuid4().hex}"


def save_ir_source(directory: Path, rank: int, text: str, suffix: str) -> Path:
    """Persist generated IR source atomically under ``directory``."""

    path = directory / f"kern_{unique_ir_name(rank)}{suffix}"
    atomic_write_text(path, text)
    return path


@dataclass(frozen=True, slots=True)
class IrKernel:
    """One kernel-IR entry with symbolic scalar bindings and optional phase gate."""

    function: KernelFunction
    scalars: Mapping[str, Any] = field(default_factory=dict)
    extent: Any = None
    phase_mask: int | None = None


@dataclass(frozen=True, slots=True)
class CompiledIrProgram:
    """Compiled multi-kernel IR program for one target dialect."""

    function: Callable[..., None]
    settle: Callable[..., None] | None
    module: object | None
    source: str
    saved_kernel_file: Path | None
    generated_modules: tuple[tuple[str, str], ...] = ()
    requests: Callable[..., tuple[CompileRequest, ...]] = _no_requests
    close: Callable[[], None] = _no_close


def _has_writable_alias(
    function: KernelFunction, states: Mapping[str, torch.Tensor]
) -> bool:
    """Return whether any writable buffer shares storage with another parameter."""

    seen: dict[tuple[torch.device, int], bool] = {}
    for param in function.params:
        tensor = states.get(param.name)
        if param.access is None or tensor is None or tensor.numel() == 0:
            continue
        key = (tensor.device, tensor.untyped_storage()._cdata)
        writable = param.access != "read"
        if key in seen and (writable or seen[key]):
            return True
        seen[key] = writable
    return False


def optimize_ir_kernels(
    kernels: Sequence[IrKernel],
    *,
    implies: Implies | None,
    states: Mapping[str, torch.Tensor] | None = None,
) -> tuple[IrKernel, ...]:
    """Apply ``sink -> hoist -> sink`` phase-branch merging where alias-free."""

    if implies is None:
        return tuple(kernels)
    optimized = []
    for kernel in kernels:
        if states is not None and _has_writable_alias(kernel.function, states):
            optimized.append(kernel)
        else:
            body = sink(hoist(sink(kernel.function.body), implies))
            optimized.append(
                replace(kernel, function=replace(kernel.function, body=body))
            )
    return tuple(optimized)


def _phase_bits(kernels: Iterable[IrKernel]) -> int:
    bits = 0
    pending: list[Any] = []
    for kernel in kernels:
        bits |= kernel.phase_mask or 0
        pending.extend(kernel.function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, PhaseTest):
            bits |= node.bits
        pending.extend(subexpressions(node))
    return bits


def _program_mask(kernels: Iterable[IrKernel]) -> int | None:
    mask = 0
    for kernel in kernels:
        if kernel.phase_mask is None:
            return None
        mask |= kernel.phase_mask
    return mask


def _compile_cuda_program(
    kernels: tuple[IrKernel, ...],
    *,
    device: torch.device,
    physics: bool,
    name_prefix: str,
    rank: int,
    save_dir: Path | None,
) -> CompiledIrProgram:
    source = CUDA.program([kernel.function for kernel in kernels])
    program = rtc.RtcProgram(source, rtc.program_options((), physics=physics), name_prefix)
    request = rtc.RtcRequest(program, tuple(kernel.function.name for kernel in kernels))
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()

    def bind(
        states: Mapping[str, torch.Tensor], block_size: int
    ) -> list[tuple[int | None, Any]]:
        owner = None
        buffers = {
            param.name: param
            for kernel in kernels
            for param in kernel.function.params
            if param.access is not None
        }
        for name, param in buffers.items():
            tensor = states[name]
            if not tensor.is_cuda:
                raise RuntimeError(f"{name} must be a CUDA/HIP tensor")
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
            if tensor.dtype != param.type:
                raise RuntimeError(f"{name} has unexpected dtype")
            if owner is None:
                owner = tensor.device
            elif tensor.device != owner:
                raise RuntimeError(
                    f"{name} must be on the same CUDA/HIP device as all bound buffers"
                )
        steps = []
        for kernel in kernels:
            count = 1 if kernel.extent is None else int(_eval_bound(kernel.extent, states))
            if count == 0:
                continue
            width = 1 if kernel.extent is None else block_size
            arguments = tuple(
                rtc.pointer(states[param.name])
                if param.access is not None
                else rtc.scalar(_eval_bound(kernel.scalars[param.name], states), param.type)
                for param in kernel.function.params
            )
            step = rtc.CudaLaunch(
                kernel.function.name, rtc.blocks(count, width), width, arguments
            )
            steps.append((kernel.phase_mask, step))
        target_index = device_index if owner is None else owner.index
        return [
            (
                mask,
                rtc.prepare(request, tuple(step for _m, step in group), target_index),
            )
            for mask, group in groupby(steps, key=lambda item: item[0])
        ]

    bound: list[Any] = []

    def run(states: Mapping[str, torch.Tensor], block_size: int, phase: int) -> None:
        if not bound or bound[0] is not states or bound[1] != block_size:
            bound[:] = [states, block_size, bind(states, block_size)]
        for mask, launch in bound[2]:
            if mask is None or phase < 0 or phase & mask:
                launch()

    saved = (
        save_ir_source(save_dir, rank, rtc.RTC_PRELUDE + source, ".cu")
        if save_dir is not None
        else None
    )
    return CompiledIrProgram(
        function=run,
        settle=None,
        module=None,
        source=source,
        saved_kernel_file=saved,
        requests=lambda states, block_size: (
            rtc.precompile_request(request, device_index),
        ),
    )


def _compile_triton_program(
    kernels: tuple[IrKernel, ...],
    *,
    physics: bool,
    name_prefix: str,
    rank: int,
    save_dir: Path | None,
) -> CompiledIrProgram:
    source = TRITON.program([kernel.function for kernel in kernels])
    module_name = f"{name_prefix}_{unique_ir_name(rank)}"
    module = compile_generated_module(source, name=module_name)
    try:
        jit = tuple(
            (kernel, getattr(module, kernel.function.name)) for kernel in kernels
        )
        saved = (
            save_ir_source(save_dir, rank, source, ".py")
            if save_dir is not None
            else None
        )
    except BaseException:
        release_generated_module(module_name, module.__file__)
        raise

    def bind(
        states: Mapping[str, torch.Tensor], block_size: int
    ) -> list[tuple[int | None, Any, Any, tuple[Any, ...], dict[str, Any]]]:
        launches = []
        for kernel, function in jit:
            extent = (
                1 if kernel.extent is None else int(_eval_bound(kernel.extent, states))
            )
            if extent == 0:
                continue
            width = 1 if kernel.extent is None else block_size
            arguments = tuple(
                states[param.name]
                if param.access is not None
                else _eval_bound(kernel.scalars[param.name], states)
                for param in kernel.function.params
            )
            grid = ((extent + width - 1) // width,)
            with proven_triton_device(triton_call_device(arguments)).active():
                options = {
                    **launch_options(function, physics=physics),
                    "BLOCK_SIZE": width,
                }
            launches.append((kernel.phase_mask, function, grid, arguments, options))
        return launches

    bound: list[Any] = []

    def run(states: Mapping[str, torch.Tensor], block_size: int, phase: int) -> None:
        if not bound or bound[0] is not states or bound[1] != block_size:
            launches = bind(states, block_size)
            bound[:] = [
                states,
                block_size,
                [
                    (mask, function[grid], arguments, options)
                    for mask, function, grid, arguments, options in launches
                ],
            ]
        for mask, launch, arguments, options in bound[2]:
            if mask is None or phase < 0 or phase & mask:
                launch(*arguments, **options)

    def requests(
        states: Mapping[str, torch.Tensor], block_size: int
    ) -> tuple[CompileRequest, ...]:
        def warmup(function, grid, arguments, options):
            with proven_triton_device(triton_call_device(arguments)).active():
                return function.warmup(*arguments, grid=grid, **options)

        return tuple(
            warmup_request(partial(warmup, function, grid, arguments, options))
            for _mask, function, grid, arguments, options in bind(states, block_size)
        )

    return CompiledIrProgram(
        function=run,
        settle=None,
        module=module,
        source=source,
        saved_kernel_file=saved,
        generated_modules=((module_name, module.__file__),),
        requests=requests,
    )


def _compile_metal_program(
    kernels: tuple[IrKernel, ...],
    settle_kernels: tuple[IrKernel, ...],
    *,
    emulated: bool,
    extra_buffers: Mapping[str, torch.Tensor],
    physics: bool,
    rank: int,
    save_dir: Path | None,
) -> CompiledIrProgram:
    all_kernels = (*kernels, *settle_kernels)
    sample_count = len(kernels)
    printer = emulated_msl() if emulated else MSL
    source = printer.program([kernel.function for kernel in all_kernels])
    origin = "physics" if physics else "framework"
    programs = []
    extra = dict(extra_buffers)
    for kernel in all_kernels:
        fields = msl_arguments(kernel.function, emulated=emulated)
        program = MetalProgram(
            source,
            kernel.function.name,
            tuple(MetalArgument(*item) for item in fields),
            extent=() if kernel.extent is None else (_METAL_THREADS,),
            encoding="float32x2" if emulated else "native",
            origin=origin,
        )
        names = tuple(zip(kernel.function.params, (item[0] for item in fields)))
        programs.append((kernel, program, names))
        source = program.source

    def bind(states: Mapping[str, torch.Tensor], selected: Sequence[Any]):
        launches = []
        for kernel, program, names in selected:
            values = {
                field_name: extra[param.name]
                if param.name in extra
                else states[param.name]
                if param.access is not None
                else _eval_bound(kernel.scalars[param.name], states)
                for param, field_name in names
            }
            if kernel.extent is not None:
                values[_METAL_THREADS] = int(_eval_bound(kernel.extent, states))
            width = 1 if kernel.extent is None else METAL.block.fixed
            launches.append((kernel.phase_mask, program.specialize(values, width)))
        return launches

    bound: list[Any] = []
    settled: list[Any] = []
    closed = False

    def run(states: Mapping[str, torch.Tensor], block_size: int, phase: int) -> None:
        del block_size
        if closed:
            raise RuntimeError("Metal IR program is closed")
        if not bound or bound[0] is not states:
            previous = bound[1] if bound else ()
            bound[:] = [states, bind(states, programs[:sample_count])]
            with cleanup_on_exit(
                "Metal IR bindings",
                (
                    launch.close
                    for _mask, launch in previous
                    if hasattr(launch, "close")
                ),
            ):
                pass
        for mask, launch in bound[1]:
            if mask is None or phase < 0 or phase & mask:
                launch()

    def settle(states: Mapping[str, torch.Tensor], phase: int) -> None:
        del phase
        if closed:
            raise RuntimeError("Metal IR program is closed")
        if not settled or settled[0] is not states:
            previous = settled[1] if settled else ()
            settled[:] = [states, bind(states, programs[sample_count:])]
            with cleanup_on_exit(
                "Metal settle bindings",
                (
                    launch.close
                    for _mask, launch in previous
                    if hasattr(launch, "close")
                ),
            ):
                pass
        for _mask, launch in settled[1]:
            launch()

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        launches = tuple(
            launch
            for held in (bound, settled)
            if held
            for _mask, launch in held[1]
            if hasattr(launch, "close")
        )
        bound.clear()
        settled.clear()
        extra.clear()
        with cleanup_on_exit("Metal IR program", (launch.close for launch in launches)):
            pass

    saved = None
    if save_dir is not None:
        path = save_dir / f"kern_{unique_ir_name(rank)}.metal"
        if programs:
            saved = programs[0][1].save_source(path)
        else:
            atomic_write_text(path, source)
            saved = path

    return CompiledIrProgram(
        function=run,
        settle=settle if settle_kernels else None,
        module=None,
        source=source,
        saved_kernel_file=saved,
        close=close,
    )


def _torch_entry_lines(
    name: str, kernels: Iterable[IrKernel], phase: int | None = None
) -> list[str]:
    lines = [
        f"def {name}(states, phase):"
        if phase is None
        else f"def {name}_{phase}(states):"
    ]
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


def _load_torch_module(source: str, prefix: str, rank: int):
    name = f"{prefix}_{unique_ir_name(rank)}"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script_method` is not supported.*",
            category=DeprecationWarning,
            module=r"torch\.jit\._script",
        )
        module = compile_generated_module(source, name=name)
    return module, (name, module.__file__)


def _torch_source(
    device: torch.device,
    sizes: Mapping[str, int],
    scalar_source: Mapping[str, torch.Tensor],
    entries: Mapping[str, tuple[IrKernel, ...]],
    phases: Sequence[int] = (),
) -> str:
    printer = TorchPrinter("device")
    unique_kernels = {
        kernel.function.name: kernel for group in entries.values() for kernel in group
    }
    blocks = [
        TORCH_PRELUDE.rstrip("\n"),
        f"device = torch.device({str(device)!r})",
        *(
            printer.kernel(
                kernel.function,
                sizes=sizes,
                scalars={
                    name: _eval_bound(spec, scalar_source)
                    for name, spec in kernel.scalars.items()
                },
            )
            for kernel in unique_kernels.values()
        ),
        *("\n".join(_torch_entry_lines(name, group)) for name, group in entries.items()),
        *(
            "\n".join(_torch_entry_lines("update", entries["update"], phase))
            for phase in phases
        ),
    ]
    return "\n\n\n".join(blocks) + "\n"


def _torch_binder(
    sizes: Mapping[str, int], kernels: Iterable[IrKernel]
) -> Callable[[Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]:
    params = {
        param.name: param
        for kernel in kernels
        for param in kernel.function.params
        if param.access is not None
    }

    def bind(states: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        flat = {}
        for name, param in params.items():
            tensor = states[name]
            if tensor.dtype != param.type:
                raise RuntimeError(f"{name} has unexpected dtype {tensor.dtype}")
            if name in sizes and tensor.numel() != sizes[name]:
                raise RuntimeError(
                    f"{name} has {tensor.numel()} elements; the IR "
                    f"program was compiled for {sizes[name]}"
                )
            flat[name] = tensor.view(-1)
        return flat

    return bind


def _tracked(states: Mapping[str, torch.Tensor]) -> bool:
    return torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in states.values()
    )


def _compile_torch_program(
    kernels: tuple[IrKernel, ...],
    settle_kernels: tuple[IrKernel, ...],
    *,
    device: torch.device,
    sizes: Mapping[str, int],
    scalar_source: Mapping[str, torch.Tensor],
    phase_control: str | None,
    sample_phases: Iterable[int],
    compile_update: bool,
    name_prefix: str,
    rank: int,
    save_dir: Path | None,
) -> CompiledIrProgram:
    bits = _phase_bits(kernels)
    mask = _program_mask(kernels)
    phases = sorted({phase & bits for phase in sample_phases}) if compile_update else []
    bind = _torch_binder(sizes, (*kernels, *settle_kernels))
    entries: dict[str, tuple[IrKernel, ...]] = {"update": kernels}
    if settle_kernels:
        entries["settle"] = settle_kernels
    source = _torch_source(device, sizes, scalar_source, entries, phases)
    module, generated = _load_torch_module(source, name_prefix, rank)
    try:
        saved = (
            save_ir_source(save_dir, rank, source, ".py")
            if save_dir is not None
            else None
        )
    except BaseException:
        release_generated_module(*generated)
        raise

    bound: list[Any] = []
    runs: dict[int, Callable[[dict[str, torch.Tensor]], None]] = {}

    def flat(states: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not bound or bound[0] is not states:
            bound[:] = [states, bind(states)]
        return bound[1]

    def dispatch(states: dict[str, torch.Tensor], phase: int) -> None:
        if _tracked(states):
            with torch.autograd.graph.saved_tensors_hooks(torch.clone, lambda x: x):
                module.update(states, phase)
            return
        if phase not in phases:
            module.update(states, phase)
            return
        entry = runs.get(phase)
        if entry is None:
            entry = runs[phase] = torch.compile(
                getattr(module, f"update_{phase}"), dynamic=False
            )
        entry(states)

    def run(states: Mapping[str, torch.Tensor], block_size: int, phase: int) -> None:
        del block_size
        if phase < 0 and phase_control is not None:
            phase = int(states[phase_control].item())
        if mask is not None and not phase & mask:
            return
        dispatch(flat(states), phase & bits)

    def settle(states: Mapping[str, torch.Tensor], phase: int) -> None:
        flat_states = flat(states)
        if _tracked(flat_states):
            with torch.autograd.graph.saved_tensors_hooks(torch.clone, lambda x: x):
                module.settle(flat_states, phase)
            return
        module.settle(flat_states, phase)

    return CompiledIrProgram(
        function=run,
        settle=settle if settle_kernels else None,
        module=module,
        source=source,
        saved_kernel_file=saved,
        generated_modules=(generated,),
    )


def compile_torch_settle(
    settle_kernels: Sequence[IrKernel],
    *,
    device: torch.device,
    sizes: Mapping[str, int],
    scalar_source: Mapping[str, torch.Tensor],
    name_prefix: str = "hydroforge_statistics_settle",
    rank: int = 0,
) -> tuple[Callable[..., None], tuple[str, str]]:
    """Compile a standalone PyTorch settle entry over shared storage."""

    kernels = tuple(settle_kernels)
    bind = _torch_binder(sizes, kernels)
    source = _torch_source(device, sizes, scalar_source, {"settle": kernels})
    module, generated = _load_torch_module(source, name_prefix, rank)
    bound: list[Any] = []

    def settle(states: Mapping[str, torch.Tensor], phase: int) -> None:
        if not bound or bound[0] is not states:
            bound[:] = [states, bind(states)]
        module.settle(bound[1], phase)

    return settle, generated


def compile_ir_program(
    dialect: str,
    *,
    device: torch.device,
    kernels: Sequence[IrKernel],
    settle_kernels: Sequence[IrKernel] = (),
    states: Mapping[str, torch.Tensor] | None = None,
    sizes: Mapping[str, int] | None = None,
    scalar_source: Mapping[str, torch.Tensor] | None = None,
    extra_buffers: Mapping[str, torch.Tensor] = (),
    emulated: bool = False,
    implies: Implies | None = None,
    phase_control: str | None = None,
    sample_phases: Iterable[int] = (),
    compile_torch_update: bool = False,
    physics: bool = False,
    name_prefix: str = "hydroforge_ir",
    rank: int = 0,
    save_dir: Path | None = None,
) -> CompiledIrProgram:
    """Compile a sequence of IR kernels in ``dialect`` with optional settle fallback."""

    optimized = optimize_ir_kernels(kernels, implies=implies, states=states)
    optimized_settle = tuple(settle_kernels)
    extra_map = dict(extra_buffers)
    if dialect == "cuda":
        compiled = _compile_cuda_program(
            optimized,
            device=device,
            physics=physics,
            name_prefix=name_prefix,
            rank=rank,
            save_dir=save_dir,
        )
    elif dialect == "triton":
        compiled = _compile_triton_program(
            optimized,
            physics=physics,
            name_prefix=name_prefix,
            rank=rank,
            save_dir=save_dir,
        )
    elif dialect == "msl":
        compiled = _compile_metal_program(
            optimized,
            optimized_settle,
            emulated=emulated,
            extra_buffers=extra_map,
            physics=physics,
            rank=rank,
            save_dir=save_dir,
        )
    elif dialect == "torch":
        compiled = _compile_torch_program(
            optimized,
            optimized_settle,
            device=device,
            sizes={} if sizes is None else sizes,
            scalar_source={} if scalar_source is None else scalar_source,
            phase_control=phase_control,
            sample_phases=sample_phases,
            compile_update=compile_torch_update,
            name_prefix=name_prefix,
            rank=rank,
            save_dir=save_dir,
        )
    else:
        raise ValueError(f"unsupported kernel IR dialect {dialect!r}")

    if compiled.settle is not None or not optimized_settle:
        return compiled

    try:
        settle, generated = compile_torch_settle(
            optimized_settle,
            device=device,
            sizes={} if sizes is None else sizes,
            scalar_source={} if scalar_source is None else scalar_source,
            name_prefix=f"{name_prefix}_settle",
            rank=rank,
        )
    except BaseException:
        with cleanup_on_exit(
            "IR settle compilation",
            (
                compiled.close,
                *(
                    partial(release_generated_module, mod_name, filename)
                    for mod_name, filename in compiled.generated_modules
                ),
            ),
        ):
            raise
    return replace(
        compiled,
        settle=settle,
        generated_modules=(*compiled.generated_modules, generated),
    )


def compile_ir_scalar_variants(
    dialect: str,
    *,
    device: torch.device,
    function: KernelFunction,
    tensors: tuple[torch.Tensor, ...],
    variants: Mapping[Any, Mapping[str, int | float | bool]],
    physics: bool = False,
    name_prefix: str = "hydroforge_scalar_ir",
) -> tuple[
    Callable[[Any], None],
    tuple[CompileRequest, ...],
    Callable[[], None],
    str,
]:
    """Compile a single-lane scalar IR kernel specialized over ``variants``."""

    buffer_params = [param for param in function.params if param.access is not None]
    scalar_params = [param for param in function.params if param.access is None]
    states = dict(
        zip((param.name for param in buffer_params), tensors, strict=True)
    )
    if dialect == "torch":
        sizes = {name: tensor.numel() for name, tensor in states.items()}
        printer = TorchPrinter("device")
        names = {
            key: f"{function.name}_{index}"
            for index, key in enumerate(variants)
        }
        source = (
            "\n\n\n".join(
                (
                    TORCH_PRELUDE.rstrip("\n"),
                    f"device = torch.device({str(device)!r})",
                    *(
                        printer.kernel(
                            replace(function, name=names[key]),
                            sizes=sizes,
                            scalars={
                                k: int(v) if isinstance(v, bool) else v
                                for k, v in spec.items()
                            },
                        )
                        for key, spec in variants.items()
                    ),
                )
            )
            + "\n"
        )
        module, generated = _load_torch_module(source, name_prefix, 0)
        entries = {key: getattr(module, names[key]) for key in variants}

        def close_torch() -> None:
            release_generated_module(*generated)

        return (
            lambda key: entries[key](states, 0),
            (),
            close_torch,
            source,
            module,
            generated[1],
        )

    if dialect == "triton":
        source = TRITON.program((function,))
        module_name = f"{name_prefix}_{uuid4().hex}"
        module = compile_generated_module(source, name=module_name)
        filename = module.__file__
        launch = launch_triton_kernel(
            getattr(module, function.name), (1,), physics=physics
        )

        def run_triton(key: Any) -> None:
            spec = variants[key]
            scalars = tuple(spec[param.name] for param in scalar_params)
            launch(*tensors, *scalars, BLOCK_SIZE=1, num_warps=1)

        def warmup() -> None:
            with warming_up():
                for key in variants:
                    run_triton(key)

        def close_triton() -> None:
            release_generated_module(module_name, filename)

        return (
            run_triton,
            (warmup_request(warmup, cost=len(source)),),
            close_triton,
            source,
            module,
            filename,
        )

    if dialect == "cuda":
        source = CUDA.program((function,))
        program = rtc.RtcProgram(
            source, rtc.program_options((), physics=physics), name_prefix
        )
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        request = rtc.RtcRequest(program, (function.name,))
        launches: dict[Any, Any] = {}

        def run_cuda(key: Any) -> None:
            prepared = launches.get(key)
            if prepared is None:
                spec = variants[key]
                args = tuple(
                    rtc.pointer(states[param.name])
                    if param.access is not None
                    else rtc.scalar(spec[param.name], param.type)
                    for param in function.params
                )
                step = rtc.CudaLaunch(function.name, 1, 1, args)
                prepared = launches[key] = rtc.prepare(request, (step,), device_index)
            prepared()

        return (
            run_cuda,
            (rtc.precompile_request(request, device_index),),
            _no_close,
            source,
            None,
            None,
        )

    if dialect == "msl":
        source = MSL.program((function,))
        fields = msl_arguments(function)
        program = MetalProgram(
            source,
            function.name,
            tuple(MetalArgument(*item) for item in fields),
            extent=(),
            origin="physics" if physics else "framework",
        )
        launches = {}
        try:
            for key, spec in variants.items():
                values = {
                    field_name: states[param.name]
                    if param.access is not None
                    else spec[param.name]
                    for param, (field_name, *_rest) in zip(
                        function.params, fields, strict=True
                    )
                }
                launches[key] = program.specialize(values, 1)
        except BaseException:
            with cleanup_on_exit(
                "Metal scalar variant setup",
                (
                    launch.close
                    for launch in launches.values()
                    if hasattr(launch, "close")
                ),
            ):
                raise

        def close_metal() -> None:
            items = tuple(launches.values())
            launches.clear()
            with cleanup_on_exit(
                "Metal scalar variants",
                (launch.close for launch in items if hasattr(launch, "close")),
            ):
                pass

        return (lambda key: launches[key]()), (), close_metal, source, None, None

    raise ValueError(f"unsupported kernel IR dialect {dialect!r}")


__all__ = [
    "CompiledIrProgram",
    "IrKernel",
    "compile_ir_program",
    "compile_ir_scalar_variants",
    "compile_torch_settle",
    "optimize_ir_kernels",
    "save_ir_source",
    "unique_ir_name",
]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Metal control kernels of compiled loops.

The kernels are the rules of :mod:`hydroforge.kernels.backends.loop_control`
printed in MSL, one lane each (an empty launch extent), and are recorded
around the loop body in one indirect command buffer.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import torch

from hydroforge.kernels.backends import loop_control
from hydroforge.kernels.backends.loop_control import (
    FIXED_END,
    NEXT,
    SamplePhase,
    cell,
    statistics_control,
)
from hydroforge.kernels.codegen.c import MSL, identifier, msl_arguments
from hydroforge.kernels.codegen.ir import (
    Binary,
    Cast,
    Compare,
    Const,
    KernelFunction,
    Param,
    Select,
    Store,
    Var,
)
from hydroforge.kernels.codegen.msl import emulated_msl
from hydroforge.kernels.metal import MetalArgument, MetalCommand, MetalProgram

_REAL, _INT = torch.float32, torch.int32


@cache
def _program(function: KernelFunction) -> MetalProgram:
    emulated = any(param.type == torch.float64 for param in function.params)
    printer = emulated_msl() if emulated else MSL
    return MetalProgram(
        printer.program((function,)),
        function.name,
        tuple(
            MetalArgument(*field)
            for field in msl_arguments(function, emulated=emulated)
        ),
        extent=(),
        encoding="float32x2" if emulated else "native",
    )


def _command(function: KernelFunction, **values: Any) -> MetalCommand:
    """A launch of ``function`` with its parameters bound by name."""

    return MetalCommand(
        _program(function),
        {identifier(param): values[param.name] for param in function.params},
    )


def fixed_control_command(
    *,
    count: torch.Tensor,
    counter: torch.Tensor,
    continue_flag: torch.Tensor,
) -> MetalCommand:
    return _command(
        FIXED_END, count=count, counter=counter, continue_flag=continue_flag
    )


@cache
def _statistics_control(
    sample_phase: SamplePhase, source: torch.dtype, destination: torch.dtype
) -> KernelFunction:
    return statistics_control(sample_phase, source, destination, fixed=False)


def statistics_control_command(
    *,
    sample_phase: SamplePhase,
    weight_source: torch.Tensor,
    continue_flag: torch.Tensor,
    counter: torch.Tensor,
    flags: torch.Tensor,
    weight: torch.Tensor,
    phase: torch.Tensor,
) -> MetalCommand:
    """``sample_phase(flags, first, last)`` is the statistics phase rule."""

    return _command(
        _statistics_control(sample_phase, weight_source.dtype, weight.dtype),
        weight_source=weight_source,
        continue_flag=continue_flag,
        counter=counter,
        flags=flags,
        weight=weight,
        phase=phase,
    )


_CANDIDATE = Param("candidate", _REAL, "write")
_ADAPTIVE_BEGIN = KernelFunction(
    "hf_adaptive_begin",
    (_CANDIDATE, Param("maximum", _REAL)),
    (Store(_CANDIDATE.name, Const(0), Var("maximum", _REAL)),),
)


def _adaptive_accept() -> KernelFunction:
    params = (
        Param("candidate", _REAL, "read"),
        Param("duration", _REAL, "read"),
        Param("elapsed", _REAL, "read"),
        Param("dt", _REAL, "write"),
        Param("error", _INT, "write"),
    )
    candidate, duration, elapsed, dt, error = map(cell, params)
    return KernelFunction(
        "hf_adaptive_accept",
        params,
        loop_control.adaptive_accept(
            candidate=candidate, duration=duration, elapsed=elapsed, dt=dt, error=error
        ),
    )


# ``status[0]`` gains this bit when a scatter of the iteration wrote an
# out-of-range index (see :func:`adaptive_control_commands`).
SCATTER_ERROR_STATUS = 2


def _adaptive_end(*, scatter_error: bool = False) -> KernelFunction:
    params = (
        Param("duration", _REAL, "read"),
        Param("dt", _REAL, "read"),
        Param("elapsed", _REAL, "read_write"),
        Param("counter", _INT, "read_write"),
        Param("continue_flag", _INT, "write"),
        Param("error", _INT, "read_write"),
        Param("status", _REAL, "write"),
        Param("maximum_steps", torch.int64),
        *((Param("scatter_error", _INT, "read"),) if scatter_error else ()),
    )
    duration, dt, elapsed, counter, continue_flag, error = map(cell, params[:6])
    failed: Any = Cast(error, _REAL)
    if scatter_error:
        failed = Binary(
            "+",
            failed,
            Select(
                Compare("!=", cell(params[-1]), Const(0, _INT)),
                Const(float(SCATTER_ERROR_STATUS), _REAL),
                Const(0.0, _REAL),
            ),
        )
    return KernelFunction(
        "hf_adaptive_end_checked" if scatter_error else "hf_adaptive_end",
        params,
        (
            *loop_control.adaptive_end(
                duration=duration,
                dt=dt,
                elapsed=elapsed,
                counter=counter,
                continue_flag=continue_flag,
                error=error,
                maximum_steps=Var("maximum_steps", torch.int64),
            ),
            # ``(error, continue, dt)`` for one host read.
            Store("status", Const(0), failed),
            Store("status", Const(1), Cast(NEXT, _REAL)),
            Store("status", Const(2), dt),
        ),
    )


_ADAPTIVE_ACCEPT = _adaptive_accept()
_ADAPTIVE_END = _adaptive_end()
_ADAPTIVE_END_CHECKED = _adaptive_end(scatter_error=True)


def adaptive_control_commands(
    *,
    candidate: torch.Tensor,
    maximum: float,
    duration: torch.Tensor,
    elapsed: torch.Tensor,
    dt: torch.Tensor,
    counter: torch.Tensor,
    continue_flag: torch.Tensor,
    error_flag: torch.Tensor,
    status: torch.Tensor,
    maximum_steps: int,
    scatter_error: torch.Tensor | None = None,
) -> tuple[MetalCommand, MetalCommand, MetalCommand]:
    """``status`` receives ``(error, continue, dt)`` for one host read.

    With the iteration's ``(1,)`` int32 ``scatter_error`` bounds flag,
    ``status[0]`` also adds :data:`SCATTER_ERROR_STATUS` when that flag is
    set, so the host needs no separate read of it per iteration.
    """

    begin = _command(_ADAPTIVE_BEGIN, candidate=candidate, maximum=maximum)
    accept = _command(
        _ADAPTIVE_ACCEPT,
        candidate=candidate,
        duration=duration,
        elapsed=elapsed,
        dt=dt,
        error=error_flag,
    )
    end = _command(
        _ADAPTIVE_END if scatter_error is None else _ADAPTIVE_END_CHECKED,
        duration=duration,
        dt=dt,
        elapsed=elapsed,
        counter=counter,
        continue_flag=continue_flag,
        error=error_flag,
        status=status,
        maximum_steps=maximum_steps,
        **({} if scatter_error is None else {"scatter_error": scatter_error}),
    )
    return begin, accept, end

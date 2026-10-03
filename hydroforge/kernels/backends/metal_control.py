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
    Cast,
    Const,
    KernelFunction,
    Param,
    Store,
    Var,
)
from hydroforge.kernels.metal import MetalArgument, MetalCommand, MetalProgram

_REAL, _INT = torch.float32, torch.int32


@cache
def _program(function: KernelFunction) -> MetalProgram:
    return MetalProgram(
        MSL.program((function,)),
        function.name,
        tuple(MetalArgument(*field) for field in msl_arguments(function)),
        extent=(),
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
def _statistics_control(sample_phase: SamplePhase) -> KernelFunction:
    return statistics_control(sample_phase, _REAL, _REAL, fixed=False)


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
        _statistics_control(sample_phase),
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


def _adaptive_end() -> KernelFunction:
    params = (
        Param("duration", _REAL, "read"),
        Param("dt", _REAL, "read"),
        Param("elapsed", _REAL, "read_write"),
        Param("counter", _INT, "read_write"),
        Param("continue_flag", _INT, "write"),
        Param("error", _INT, "read_write"),
        Param("status", _REAL, "write"),
        Param("maximum_steps", torch.int64),
    )
    duration, dt, elapsed, counter, continue_flag, error = map(cell, params[:6])
    return KernelFunction(
        "hf_adaptive_end",
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
            Store("status", Const(0), Cast(error, _REAL)),
            Store("status", Const(1), Cast(NEXT, _REAL)),
            Store("status", Const(2), dt),
        ),
    )


_ADAPTIVE_ACCEPT = _adaptive_accept()
_ADAPTIVE_END = _adaptive_end()


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
) -> tuple[MetalCommand, MetalCommand, MetalCommand]:
    """``status`` receives ``(error, continue, dt)`` for one host read."""

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
        _ADAPTIVE_END,
        duration=duration,
        dt=dt,
        elapsed=elapsed,
        counter=counter,
        continue_flag=continue_flag,
        error=error_flag,
        status=status,
        maximum_steps=maximum_steps,
    )
    return begin, accept, end

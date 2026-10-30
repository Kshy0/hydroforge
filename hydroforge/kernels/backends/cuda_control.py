# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The runtime-compiled control kernels of CUDA graph loops.

A conditional-graph WHILE node (:class:`hydroforge.kernels.toolchain.cuda.
ConditionalWhileGraph`) folds a loop into one graph launch: the body and its
continuation predicate run on the device, so the host issues one launch per
adaptive or predicate loop.  A fixed loop replays bounded batches of iterations
from the host.  The control kernels are the rules of
:mod:`hydroforge.kernels.backends.loop_control` printed in CUDA, so an
adaptive iteration's controls are two launches; the continuation predicate
calls the CUDA-only ``set_conditional`` intrinsic.
Each control program compiles once with every kernel it holds, the
statistics program with one entry per weight source and destination type.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import torch

from hydroforge.kernels.backends import loop_control
from hydroforge.kernels.backends.loop_control import (
    FIXED_CLOCK,
    FIXED_END,
    SamplePhase,
    cell,
    statistics_control,
)
from hydroforge.kernels.codegen.c import CUDA
from hydroforge.kernels.codegen.ir import (
    Call,
    Compare,
    Const,
    Evaluate,
    If,
    KernelFunction,
    Param,
    Select,
    Var,
)
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain import cuda as rtc

_REALS = (torch.float32, torch.float64)

_CONTINUE = Param("continue_flag", torch.int32, "read")
# Continuation predicate of adaptive and predicate loops: the model's
# ``(1,)`` int continue flag, set only while capturing.
_SET_CONDITIONAL = KernelFunction(
    "hf_set_conditional",
    (Param("handle", torch.uint64), _CONTINUE, Param("set_condition", torch.bool)),
    (
        If(
            Var("set_condition", torch.bool),
            (
                Evaluate(
                    Call(
                        "set_conditional",
                        (
                            Var("handle", torch.uint64),
                            Select(
                                Compare("!=", cell(_CONTINUE), Const(0)),
                                Const(1),
                                Const(0),
                            ),
                        ),
                        None,
                    )
                ),
            ),
        ),
    ),
)


def _adaptive_accept(real: torch.dtype) -> KernelFunction:
    params = (
        Param("candidate", real, "read"),
        Param("duration", torch.int64, "read"),
        Param("elapsed", torch.int64, "read"),
        Param("step", torch.int64, "write"),
        Param("dt", real, "write"),
        Param("error", torch.int32, "write"),
    )
    candidate, duration, elapsed, step, dt, error = map(cell, params)
    return KernelFunction(
        f"hf_adaptive_accept_{str(real).removeprefix('torch.')}",
        params,
        loop_control.adaptive_accept(
            candidate=candidate,
            duration=duration,
            elapsed=elapsed,
            step=step,
            dt=dt,
            error=error,
        ),
    )


def _adaptive_end() -> KernelFunction:
    params = (
        Param("duration", torch.int64, "read"),
        Param("step", torch.int64, "read"),
        Param("elapsed", torch.int64, "read_write"),
        Param("counter", torch.int32, "read_write"),
        Param("continue_flag", torch.int32, "write"),
        Param("error", torch.int32, "read_write"),
        Param("maximum_steps", torch.int64),
    )
    duration, step, elapsed, counter, continue_flag, error = map(cell, params[:6])
    return KernelFunction(
        "hf_adaptive_end",
        params,
        loop_control.adaptive_end(
            duration=duration,
            step=step,
            elapsed=elapsed,
            counter=counter,
            continue_flag=continue_flag,
            error=error,
            maximum_steps=Var("maximum_steps", torch.int64),
        ),
    )


# Accept an adaptive proposal (per width type) and advance the loop.
_ADAPTIVE_ACCEPT = {real: _adaptive_accept(real) for real in _REALS}
_ADAPTIVE_END = _adaptive_end()


def _request(
    name: str,
    functions: tuple[KernelFunction, ...],
    *,
    header: str = "",
    options: tuple[str, ...] = (),
) -> rtc.RtcRequest:
    program = rtc.RtcProgram(
        header + CUDA.program(functions),
        rtc.program_options(options, physics=False),
        name,
    )
    return rtc.RtcRequest(program, tuple(function.name for function in functions))


@functools.cache
def _control_request() -> rtc.RtcRequest:
    return _request(
        "hydroforge_loop_control",
        (_SET_CONDITIONAL, FIXED_END, FIXED_CLOCK, *_ADAPTIVE_ACCEPT.values(), _ADAPTIVE_END),
        # ``cudaGraphSetConditional`` is declared by the device runtime.
        header="#include <cuda_device_runtime_api.h>\n\n",
        options=rtc.toolkit_include_options(),
    )


@functools.cache
def _statistics_functions(
    sample_phase: SamplePhase,
) -> Mapping[tuple[bool, torch.dtype, torch.dtype], KernelFunction]:
    return {
        (fixed, source, destination): statistics_control(
            sample_phase, source, destination, fixed=fixed
        )
        for fixed in (True, False)
        for source in _REALS
        for destination in _REALS
    }


@functools.cache
def _statistics_request(sample_phase: SamplePhase) -> rtc.RtcRequest:
    return _request(
        "hydroforge_statistics_loop_control",
        tuple(_statistics_functions(sample_phase).values()),
    )


def control_requests(
    device: int, sample_phase: SamplePhase | None
) -> tuple[CompileRequest, ...]:
    """The control programs a graph loop may launch; statistics folding
    needs ``sample_phase``."""

    requests = [_control_request()]
    if sample_phase is not None:
        requests.append(_statistics_request(sample_phase))
    return tuple(rtc.precompile_request(request, device) for request in requests)


def _launch(
    request: rtc.RtcRequest,
    function: KernelFunction,
    values: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Launch the one lane of ``function`` with parameters bound by name, on
    the current stream of ``device`` (the capture stream while capturing).

    Control launches run while warming up and capturing graphs, so each
    binds afresh: a process-wide cache of bindings would keep the control
    tensors of closed models alive.  The compiled program stays cached.
    """

    args = tuple(
        rtc.pointer(values[param.name])
        if param.access is not None
        else rtc.scalar(values[param.name], param.type)
        for param in function.params
    )
    index = torch.cuda.current_device() if device.index is None else device.index
    rtc.prepare(request, (rtc.CudaLaunch(function.name, 1, 1, args),), index)()


def fixed_clock(
    *, counter: torch.Tensor, count: torch.Tensor, clock: torch.Tensor
) -> None:
    """Locate a fixed substep on its outer interval's exact tick partition."""

    _launch(
        _control_request(),
        FIXED_CLOCK,
        {"counter": counter, "count": count, "clock": clock},
        counter.device,
    )


def fixed_end(
    *, counter: torch.Tensor, continue_flag: torch.Tensor, count: torch.Tensor
) -> None:
    """Advance a fixed loop's counter and continue flag."""

    _launch(
        _control_request(),
        FIXED_END,
        {"counter": counter, "continue_flag": continue_flag, "count": count},
        counter.device,
    )


def adaptive_accept(
    *,
    candidate: torch.Tensor,
    duration: torch.Tensor,
    elapsed: torch.Tensor,
    step: torch.Tensor,
    dt: torch.Tensor,
    error: torch.Tensor,
) -> None:
    """Accept an adaptive proposal: the ``step`` ticks and the width ``dt``
    (float32/float64) of the iteration, and its ``error``."""

    function = _ADAPTIVE_ACCEPT.get(dt.dtype)
    if function is None:
        raise TypeError(f"adaptive widths must be float32/float64, got {dt.dtype}")
    _launch(
        _control_request(),
        function,
        {
            "candidate": candidate,
            "duration": duration,
            "elapsed": elapsed,
            "step": step,
            "dt": dt,
            "error": error,
        },
        dt.device,
    )


def adaptive_end(
    *,
    duration: torch.Tensor,
    step: torch.Tensor,
    elapsed: torch.Tensor,
    counter: torch.Tensor,
    continue_flag: torch.Tensor,
    error: torch.Tensor,
    maximum_steps: int,
) -> None:
    """Advance an adaptive loop by its accepted step."""

    _launch(
        _control_request(),
        _ADAPTIVE_END,
        {
            "duration": duration,
            "step": step,
            "elapsed": elapsed,
            "counter": counter,
            "continue_flag": continue_flag,
            "error": error,
            "maximum_steps": maximum_steps,
        },
        counter.device,
    )


def set_conditional(
    *, continue_flag: torch.Tensor, handle: int, set_cond: bool
) -> None:
    """Set a WHILE condition from the ``(1,)`` int32 ``continue_flag``.

    ``set_cond`` is ``False`` during warmup, outside graph capture, where
    ``cudaGraphSetConditional`` is invalid.
    """

    _launch(
        _control_request(),
        _SET_CONDITIONAL,
        {"handle": handle, "continue_flag": continue_flag, "set_condition": set_cond},
        continue_flag.device,
    )


class StatisticsControls:
    """Control kernels that place folded statistics samples in their loop.

    ``sample_phase(flags, first, last)`` is the statistics phase rule.
    """

    def __init__(self, sample_phase: SamplePhase) -> None:
        self._request = _statistics_request(sample_phase)
        self._functions = _statistics_functions(sample_phase)

    def _launch(self, fixed: bool, values: Mapping[str, torch.Tensor]) -> None:
        source, destination = values["weight_source"].dtype, values["weight"].dtype
        function = self._functions.get((fixed, source, destination))
        if function is None:
            raise TypeError(
                "statistics weights must be float32/float64, got "
                f"{source} and {destination}"
            )
        _launch(self._request, function, values, values["counter"].device)

    def fixed_end(
        self,
        *,
        counter: torch.Tensor,
        continue_flag: torch.Tensor,
        count: torch.Tensor,
        weight_src: torch.Tensor,
        flags: torch.Tensor,
        weight: torch.Tensor,
        phase: torch.Tensor,
    ) -> None:
        """Advance a fixed loop and write the controls of its sample."""

        self._launch(
            True,
            {
                "counter": counter,
                "continue_flag": continue_flag,
                "count": count,
                "weight_source": weight_src,
                "flags": flags,
                "weight": weight,
                "phase": phase,
            },
        )

    def adaptive(
        self,
        *,
        weight_src: torch.Tensor,
        continue_flag: torch.Tensor,
        counter: torch.Tensor,
        flags: torch.Tensor,
        weight: torch.Tensor,
        phase: torch.Tensor,
    ) -> None:
        """Write the controls of an advanced adaptive iteration's sample."""

        self._launch(
            False,
            {
                "weight_source": weight_src,
                "continue_flag": continue_flag,
                "counter": counter,
                "flags": flags,
                "weight": weight,
                "phase": phase,
            },
        )

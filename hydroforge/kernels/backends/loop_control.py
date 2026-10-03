"""Device loop control rules shared by the CUDA and Metal control kernels.

Compiled loops keep their control scalars on the device, so small one-lane
control kernels advance the iteration counter, place each statistics sample
in its loop and accept adaptive widths.  Each rule is written here once as
kernel IR over single-element control buffers (:func:`cell`); the CUDA and
Metal control kernels print it with the C printer, and the eager executor
implements the same rules with tensor operations.  The ecosystem tests
compare the implementations.

Fixed advance
    ``counter`` counts completed iterations.  After an iteration it advances
    to ``next``, and the loop continues while ``next < count``.
Sample position
    With a known count, the sample of the iteration that advanced the counter
    to ``next`` is the first when ``next == 1`` and the last when
    ``next == count``.  An adaptive loop does not know its count ahead; after
    its advance a sample is the first when ``counter == 1`` and the last when
    the continue flag is zero.  A sample's controls are the weight in the
    aggregation's type and the phase its step bits take at that position.
Adaptive iteration
    ``begin`` proposes the maximum width and the model's proposal may lower
    it.  ``accept`` clips the proposal to the remaining interval with NaN
    propagation (the ``torch.minimum`` contract).  An invalid (NaN or
    non-positive) width sets the error flag and is replaced by the remainder,
    so the loop body stays finite while the error ends the loop.  ``end``
    advances the elapsed time and the counter, flags an interval not complete
    after ``maximum_steps`` iterations, and continues while time remains and
    no error is set.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from hydroforge.kernels.codegen.ir import (
    Binary,
    Call,
    Cast,
    Compare,
    Const,
    Expr,
    If,
    KernelFunction,
    Let,
    Load,
    Logical,
    Param,
    Select,
    Stmt,
    Store,
    Var,
    cast,
)

# The statistics phase rule, injected by its owner: ``(flags, first, last)``.
SamplePhase = Callable[[Expr, Expr, Expr], Expr]

_INT = torch.int32
NEXT = Var("next", _INT)
"""The advanced counter (fixed) or continue value (adaptive) a rule declares."""


def cell(param: Param) -> Load:
    """The single element of the control buffer ``param``."""

    return Load(param.name, Const(0), param.type)


def _store(target: Load, value: Expr) -> Store:
    return Store(target.buffer, target.index, value)


def _fixed_advance(counter: Load, continue_flag: Load, count: Expr) -> tuple[Stmt, ...]:
    """Advance a fixed loop; declares :data:`NEXT` for the sample position."""

    return (
        Let(NEXT, Binary("+", counter, Const(1, _INT))),
        _store(counter, NEXT),
        _store(continue_flag, Cast(Compare("<", NEXT, count), _INT)),
    )


_FIXED_FIRST = Compare("==", NEXT, Const(1, _INT))


def _fixed_last(count: Expr) -> Expr:
    return Compare("==", NEXT, count)


def _adaptive_first(counter: Expr) -> Expr:
    return Compare("==", counter, Const(1, _INT))


def _adaptive_last(continue_flag: Expr) -> Expr:
    return Compare("==", continue_flag, Const(0, _INT))


def _sample_controls(
    sample_phase: SamplePhase,
    *,
    first: Expr,
    last: Expr,
    weight_source: Load,
    flags: Load,
    weight: Load,
    phase: Load,
) -> tuple[Stmt, ...]:
    """Write the weight and phase of the sample at ``first``/``last``."""

    first_var, last_var = Var("first", torch.bool), Var("last", torch.bool)
    return (
        Let(first_var, first),
        Let(last_var, last),
        _store(weight, cast(weight_source, weight.type)),
        _store(phase, sample_phase(flags, first_var, last_var)),
    )


def adaptive_accept(
    *, candidate: Expr, duration: Load, elapsed: Load, dt: Load, error: Load
) -> tuple[Stmt, ...]:
    """Clip the proposal ``candidate`` into ``dt`` and set ``error``."""

    real = dt.type
    proposal = Var("candidate", real)
    remaining = Var("remaining", real)
    accepted = Var("accepted", real)
    invalid = Var("invalid", torch.bool)
    return (
        Let(proposal, candidate),
        Let(remaining, Binary("-", duration, elapsed)),
        Let(accepted, Select(Compare("<", remaining, proposal), remaining, proposal)),
        Let(
            invalid,
            Logical(
                "or",
                (
                    Call("isnan", (proposal,), torch.bool),
                    Call("isnan", (remaining,), torch.bool),
                    Compare("<=", accepted, Const(0.0, real)),
                ),
            ),
        ),
        _store(error, Cast(invalid, error.type)),
        _store(dt, Select(invalid, remaining, accepted)),
    )


def adaptive_end(
    *,
    duration: Load,
    dt: Load,
    elapsed: Load,
    counter: Load,
    continue_flag: Load,
    error: Load,
    maximum_steps: Expr,
) -> tuple[Stmt, ...]:
    """Advance an adaptive loop; declares :data:`NEXT` (the continue value)."""

    time = Var("time", elapsed.type)
    count = Var("count", counter.type)
    return (
        Let(time, Binary("+", elapsed, dt)),
        _store(elapsed, time),
        Let(count, Binary("+", counter, Const(1, _INT))),
        _store(counter, count),
        If(
            Logical(
                "and",
                (Compare(">=", count, maximum_steps), Compare("<", time, duration)),
            ),
            (_store(error, Const(1, error.type)),),
        ),
        Let(
            NEXT,
            Cast(
                Logical(
                    "and",
                    (
                        Compare("<", time, duration),
                        Compare("==", error, Const(0, error.type)),
                    ),
                ),
                _INT,
            ),
        ),
        _store(continue_flag, NEXT),
    )


# Control buffers of the kernels both device loops launch.
_COUNTER = Param("counter", _INT, "read_write")
_CONTINUE = Param("continue_flag", _INT, "write")
_COUNT = Param("count", _INT, "read")
_FLAGS = Param("flags", _INT, "read")
_PHASE = Param("phase", _INT, "write")


# Advance a fixed loop's counter and continue flag.
FIXED_END = KernelFunction(
    "hf_fixed_end",
    (_COUNTER, _CONTINUE, _COUNT),
    _fixed_advance(cell(_COUNTER), cell(_CONTINUE), cell(_COUNT)),
)


def statistics_control(
    sample_phase: SamplePhase,
    source: torch.dtype,
    destination: torch.dtype,
    *,
    fixed: bool,
) -> KernelFunction:
    """Write the controls of an iteration's sample from a ``source`` weight.

    ``fixed`` advances a fixed loop first and places the sample by its
    count; otherwise the iteration has advanced and the adaptive rule places
    it, which after a fixed advance (counter ``next``, continue flag
    ``next < count``) gives the same position.
    """

    weight_source = Param("weight_source", source, "read")
    weight = Param("weight", destination, "write")
    controls = (_FLAGS, weight, _PHASE)
    if fixed:
        params = (_COUNTER, _CONTINUE, _COUNT, weight_source, *controls)
        advance = _fixed_advance(cell(_COUNTER), cell(_CONTINUE), cell(_COUNT))
        first, last = _FIXED_FIRST, _fixed_last(cell(_COUNT))
    else:
        continue_flag = Param(_CONTINUE.name, _INT, "read")
        counter = Param(_COUNTER.name, _INT, "read")
        params = (weight_source, continue_flag, counter, *controls)
        advance = ()
        first, last = (
            _adaptive_first(cell(counter)),
            _adaptive_last(cell(continue_flag)),
        )
    kind = "fixed" if fixed else "adaptive"
    return KernelFunction(
        f"hf_{kind}_statistics_{_suffix(source)}_{_suffix(destination)}",
        params,
        (
            *advance,
            *_sample_controls(
                sample_phase,
                first=first,
                last=last,
                weight_source=cell(weight_source),
                flags=cell(_FLAGS),
                weight=cell(weight),
                phase=cell(_PHASE),
            ),
        ),
    )


def _suffix(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")

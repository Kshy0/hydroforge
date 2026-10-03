"""Sample phases shared by host control, device loops and generated kernels.

A statistics *sample* is one aggregation launch after one physical substep of
a step that collects output.  Its phase comes from the window bits the step
carries and the substep position through exactly one rule,
:func:`sample_phase`:

``INNER_FIRST``
    First sample of an inner window: accumulators restart.  The first step
    of the window that collects output carries it; only that step's first
    substep keeps it.
``INNER_LAST``
    The fold closing an inner window: the last substep of the window's
    closing step.
``OUTER_FIRST``
    This fold restarts the outer accumulators.  Compound statistics fold
    complete inner results, so an outer window restarts at its first fold
    (its first inner window with samples closing), not at its first step.
``OUTER_LAST``
    This fold also closes the outer window.  Windows are validated so that an
    outer boundary is an inner boundary, which makes it the last substep of
    the outer window's last time point.
``STEP_LAST``
    Last substep of every step that collects output.  ``last`` reductions
    record here, so a window closed by a step without output still reports
    its last sampled value.
``SETTLE``
    A close without a sample, issued by the host when the closing step of a
    window with samples collects no output.

``INNER_FIRST`` to ``OUTER_LAST`` are the *step bits*; a sample keeps the
first-substep bits only at its first substep and the closing bits only at
its last substep.  Device loop controls derive the same phase from the same
rule through :func:`sample_phase_expr`, and generated kernels test its bits
(kernel IR ``PhaseTest``); :func:`phase_implies` states which tests one
sample phase makes imply others.
"""

from __future__ import annotations

from enum import IntFlag

import torch

from hydroforge.kernels.codegen.ir import Binary, Const, Expr, Select


class SampleFlags(IntFlag):
    """Step window bits and the phase bits of one statistics sample."""

    INNER_FIRST = 1
    INNER_LAST = 2
    OUTER_FIRST = 4
    OUTER_LAST = 8
    STEP_LAST = 16
    SETTLE = 32


STEP_BITS = int(
    SampleFlags.INNER_FIRST
    | SampleFlags.INNER_LAST
    | SampleFlags.OUTER_FIRST
    | SampleFlags.OUTER_LAST
)
_FIRST_SUBSTEP = int(SampleFlags.INNER_FIRST)
_LAST_SUBSTEP = int(
    SampleFlags.INNER_LAST | SampleFlags.OUTER_FIRST | SampleFlags.OUTER_LAST
)
_STEP_LAST = int(SampleFlags.STEP_LAST)


def sample_phase(flags: int, *, first: bool, last: bool) -> int:
    """Return the phase of one sample of a step carrying ``flags``."""

    if last:
        return flags & (_FIRST_SUBSTEP | _LAST_SUBSTEP if first else _LAST_SUBSTEP) | (
            _STEP_LAST
        )
    return flags & _FIRST_SUBSTEP if first else 0


def sample_phase_expr(flags: Expr, first: Expr, last: Expr) -> Expr:
    """:func:`sample_phase` as kernel IR: int32 ``flags``, bool positions."""

    def bits(position: Expr, value: int) -> Expr:
        return Select(position, Const(value, torch.int32), Const(0, torch.int32))

    substep = Binary("|", bits(first, _FIRST_SUBSTEP), bits(last, _LAST_SUBSTEP))
    return Binary("|", Binary("&", flags, substep), bits(last, _STEP_LAST))


_SAMPLE_PHASES = frozenset(
    sample_phase(flags, first=first, last=last)
    for flags in range(STEP_BITS + 1)
    for first in (False, True)
    for last in (False, True)
)


def phase_implies(bits: int, implied: int) -> bool:
    """Whether every sample phase with any of ``bits`` has any of ``implied``.

    ``INNER_LAST`` implies ``STEP_LAST``, for example: a window closes at the
    last substep of a step.
    """

    return all(phase & implied for phase in _SAMPLE_PHASES if phase & bits)


# Control scalars written once per sample and read by generated kernels.
# Widest entries come first so one packed buffer keeps every view aligned;
# ``None`` stands for the statistics control dtype.
CONTROL_MACRO_STEPS = "__num_macro_steps"
CONTROL_MACRO_INDEX = "__macro_step_index"
CONTROL_WEIGHT = "__weight"
CONTROL_FLAGS = "__flags"
CONTROL_PHASE = "__phase"
CONTROL_LAYOUT: tuple[tuple[str, torch.dtype | None], ...] = (
    (CONTROL_MACRO_STEPS, torch.int64),
    (CONTROL_MACRO_INDEX, torch.int64),
    (CONTROL_WEIGHT, None),
    (CONTROL_FLAGS, torch.int32),
    (CONTROL_PHASE, torch.int32),
)
# Control names share the statistics kernel state namespace.
CONTROL_STATE = frozenset(name for name, _dtype in CONTROL_LAYOUT)
# Scalars an aggregation kernel reads; ``__flags`` feeds only loop controls.
KERNEL_CONTROLS = (
    CONTROL_WEIGHT,
    CONTROL_MACRO_STEPS,
    CONTROL_PHASE,
    CONTROL_MACRO_INDEX,
)

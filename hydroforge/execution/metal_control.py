"""Online-generated Metal control kernels fused around substep physics."""

from __future__ import annotations

from functools import cache

import torch

from hydroforge.kernels.backends.metal.online import (
    MetalBuffer, MetalCommand, MetalScalar, make_online_metal_dispatcher,
)


@cache
def _fixed_begin_dispatcher():
    return make_online_metal_dispatcher(
        "hf_fixed_substep_begin",
        buffers=(
            MetalBuffer("count_ptr", torch.int32, "read"),
            MetalBuffer("counter_ptr", torch.int32, "read"),
            MetalBuffer("midpoint_ptr", torch.float32, "write"),
        ),
        scalars=(MetalScalar("n", "index"),), size_key="n",
        body="""    if (i == 0) {
        args.midpoint_ptr[0] =
            ((float)args.counter_ptr[0] + 0.5f) / (float)args.count_ptr[0];
    }""",
    )


@cache
def _fixed_end_dispatcher():
    return make_online_metal_dispatcher(
        "hf_fixed_substep_end",
        buffers=(
            MetalBuffer("count_ptr", torch.int32, "read"),
            MetalBuffer("counter_ptr", torch.int32, "read_write"),
            MetalBuffer("continue_ptr", torch.int32, "write"),
        ),
        scalars=(MetalScalar("n", "index"),), size_key="n",
        body="""    if (i == 0) {
        int next = args.counter_ptr[0] + 1;
        args.counter_ptr[0] = next;
        args.continue_ptr[0] = next < args.count_ptr[0];
    }""",
    )


def fixed_control_commands(
    *,
    count: torch.Tensor,
    counter: torch.Tensor,
    midpoint: torch.Tensor,
    continue_flag: torch.Tensor,
) -> tuple[MetalCommand, MetalCommand]:
    begin = MetalCommand(
        _fixed_begin_dispatcher(),
        {
            "count_ptr": count, "counter_ptr": counter,
            "midpoint_ptr": midpoint, "n": 1,
        },
    )
    end = MetalCommand(
        _fixed_end_dispatcher(),
        {
            "count_ptr": count, "counter_ptr": counter,
            "continue_ptr": continue_flag, "n": 1,
        },
    )
    return begin, end


@cache
def _statistics_control_dispatcher():
    return make_online_metal_dispatcher(
        "hf_statistics_substep_control",
        buffers=(
            MetalBuffer("weight_source_ptr", torch.float32, "read"),
            MetalBuffer("continue_ptr", torch.int32, "read"),
            MetalBuffer("counter_ptr", torch.int32, "read"),
            MetalBuffer("weight_ptr", torch.float32, "write"),
            MetalBuffer("sub_step_ptr", torch.int32, "write"),
            MetalBuffer("num_sub_steps_ptr", torch.int32, "write"),
        ),
        scalars=(MetalScalar("n", "index"),), size_key="n",
        body="""    if (i == 0) {
        bool first = args.counter_ptr[0] == 1;
        bool last = args.continue_ptr[0] == 0;
        int sub_step;
        int num_sub_steps;
        if (first && last) { sub_step = 0; num_sub_steps = 1; }
        else if (first) { sub_step = 0; num_sub_steps = 2; }
        else if (last) { sub_step = 1; num_sub_steps = 2; }
        else { sub_step = 1; num_sub_steps = 3; }
        args.weight_ptr[0] = args.weight_source_ptr[0];
        args.sub_step_ptr[0] = sub_step;
        args.num_sub_steps_ptr[0] = num_sub_steps;
    }""",
    )


def statistics_control_command(
    *,
    weight_source: torch.Tensor,
    continue_flag: torch.Tensor,
    counter: torch.Tensor,
    weight: torch.Tensor,
    sub_step: torch.Tensor,
    num_sub_steps: torch.Tensor,
) -> MetalCommand:
    return MetalCommand(
        _statistics_control_dispatcher(),
        {
            "weight_source_ptr": weight_source,
            "continue_ptr": continue_flag,
            "counter_ptr": counter,
            "weight_ptr": weight,
            "sub_step_ptr": sub_step,
            "num_sub_steps_ptr": num_sub_steps,
            "n": 1,
        },
    )


@cache
def _adaptive_begin_dispatcher():
    return make_online_metal_dispatcher(
        "hf_adaptive_substep_begin",
        buffers=(MetalBuffer("candidate_ptr", torch.float32, "write"),),
        scalars=(
            MetalScalar("maximum", "float32"), MetalScalar("n", "index"),
        ),
        size_key="n",
        body="""    if (i == 0) {
        args.candidate_ptr[0] = *args.maximum;
    }""",
    )


@cache
def _adaptive_accept_dispatcher():
    return make_online_metal_dispatcher(
        "hf_adaptive_substep_accept",
        buffers=(
            MetalBuffer("candidate_ptr", torch.float32, "read"),
            MetalBuffer("duration_ptr", torch.float32, "read"),
            MetalBuffer("elapsed_ptr", torch.float32, "read"),
            MetalBuffer("dt_ptr", torch.float32, "write"),
            MetalBuffer("midpoint_ptr", torch.float32, "write"),
            MetalBuffer("error_ptr", torch.int32, "write"),
        ),
        scalars=(MetalScalar("n", "index"),), size_key="n",
        body="""    if (i == 0) {
        float remaining = args.duration_ptr[0] - args.elapsed_ptr[0];
        float dt = min(args.candidate_ptr[0], remaining);
        bool invalid = !isfinite(dt) || dt <= 0.0f;
        args.error_ptr[0] = invalid;
        if (invalid) dt = remaining;
        args.dt_ptr[0] = dt;
        args.midpoint_ptr[0] =
            (args.elapsed_ptr[0] + 0.5f * dt) / args.duration_ptr[0];
    }""",
    )


@cache
def _adaptive_end_dispatcher():
    return make_online_metal_dispatcher(
        "hf_adaptive_substep_end",
        buffers=(
            MetalBuffer("duration_ptr", torch.float32, "read"),
            MetalBuffer("dt_ptr", torch.float32, "read"),
            MetalBuffer("elapsed_ptr", torch.float32, "read_write"),
            MetalBuffer("counter_ptr", torch.int32, "read_write"),
            MetalBuffer("continue_ptr", torch.int32, "write"),
            MetalBuffer("error_ptr", torch.int32, "read_write"),
        ),
        scalars=(
            MetalScalar("maximum_steps", "index"),
            MetalScalar("n", "index"),
        ), size_key="n",
        body="""    if (i == 0) {
        float elapsed = args.elapsed_ptr[0] + args.dt_ptr[0];
        args.elapsed_ptr[0] = elapsed;
        args.counter_ptr[0] += 1;
        bool exhausted = args.counter_ptr[0] >= *args.maximum_steps &&
            elapsed < args.duration_ptr[0];
        if (exhausted) args.error_ptr[0] = 1;
        args.continue_ptr[0] =
            elapsed < args.duration_ptr[0] && args.error_ptr[0] == 0;
    }""",
    )


def adaptive_control_commands(
    *,
    candidate: torch.Tensor,
    maximum: float,
    duration: torch.Tensor,
    elapsed: torch.Tensor,
    dt: torch.Tensor,
    midpoint: torch.Tensor,
    counter: torch.Tensor,
    continue_flag: torch.Tensor,
    error_flag: torch.Tensor,
    maximum_steps: int,
) -> tuple[MetalCommand, MetalCommand, MetalCommand]:
    begin = MetalCommand(
        _adaptive_begin_dispatcher(),
        {"candidate_ptr": candidate, "maximum": maximum, "n": 1},
    )
    accept = MetalCommand(
        _adaptive_accept_dispatcher(),
        {
            "candidate_ptr": candidate, "duration_ptr": duration,
            "elapsed_ptr": elapsed, "dt_ptr": dt,
            "midpoint_ptr": midpoint, "error_ptr": error_flag, "n": 1,
        },
    )
    end = MetalCommand(
        _adaptive_end_dispatcher(),
        {
            "duration_ptr": duration, "dt_ptr": dt,
            "elapsed_ptr": elapsed, "counter_ptr": counter,
            "continue_ptr": continue_flag, "error_ptr": error_flag, "n": 1,
            "maximum_steps": maximum_steps,
        },
    )
    return begin, accept, end

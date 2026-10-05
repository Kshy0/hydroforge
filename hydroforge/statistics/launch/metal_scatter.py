"""Destination-owned encoded scatter: stable CSR topology built once.

Two float32 limbs cannot be independently atomically added. One lane owns
each destination and reduces its incoming values in original source order.
The cold CPU topology build is O(N log N), replay work is O(N + targets),
and no per-sample CPU arithmetic or full source-by-target scan is needed.
"""

import torch

from hydroforge.core.expr import Reduction
from hydroforge.kernels.codegen.ir import (
    Assign,
    Binary,
    Compare,
    Const,
    Guard,
    KernelFunction,
    Let,
    Load,
    Names,
    Param,
    Store,
    ThreadIndex,
    Var,
    While,
    cast,
)
from hydroforge.statistics.kernel_plan import (
    _INDEX,
    _RESERVED,
    _SCATTER_SOURCE,
    _SCATTER_TARGET,
    LINEAR,
    MEMBER,
    TARGET_SIZE,
    TOTAL,
    Count,
    StatisticsKernel,
    _gated,
    _member_offset,
    _Values,
)
from hydroforge.statistics.phases import CONTROL_PHASE
from hydroforge.statistics.storage import COUNT_DTYPE, StoragePlan


def encoded_scatter(context, lowering, name, source):
    buffer = StoragePlan.scatter_buffer(name)
    count = (
        StoragePlan.scatter_count(name) if source.reduction is Reduction.MEAN else None
    )
    width = context.storage[buffer].shape[-1]
    members = context.ensemble_size if context.layouts[name].batched else 1
    index = context.tensors[source.index].detach().cpu().flatten().to(torch.int64)
    valid = (index >= 0) & (index < width)
    positions = torch.nonzero(valid).flatten()
    targets = index[valid]
    order = positions[torch.argsort(targets, stable=True)]
    counts = torch.bincount(targets, minlength=width)
    if (
        count is not None
        and counts.numel()
        and counts.max().item() > torch.iinfo(COUNT_DTYPE).max
    ):
        raise OverflowError("Metal scatter mean source count exceeds int32 range")
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    symbol = context.symbol(name)
    offset_name, order_name = f"__hf_csr_offsets_{symbol}", f"__hf_csr_order_{symbol}"
    if {offset_name, order_name}.intersection(
        set(context.tensors) | set(context.storage)
    ):
        raise ValueError(
            "Metal scatter generated topology name collides with a model buffer"
        )
    extra = {
        offset_name: offsets.to(context.device),
        order_name: order.to(context.device),
    }
    mask = lowering.scatter_phase_mask(name)
    gate_params, gate = _gated(mask, Param(CONTROL_PHASE, torch.int32, "read"))
    names = Names(_RESERVED)
    values = _Values(
        context,
        lowering,
        names,
        "scatter",
        torch.float64,
        lambda key: _member_offset(context.member_stride(key), _SCATTER_SOURCE),
    )
    value = names.var("contribution", torch.float64)
    values.statements.append(
        Let(value, cast(values.expression(source.value), torch.float64))
    )
    cursor, end, first = (
        Var(f"csr_{name}", torch.int64) for name in ("cursor", "end", "first")
    )
    leaves = [key for key in lowering.ir.scatter_inputs(name) if key != source.index]
    params = (
        Param(buffer, torch.float64, "read_write"),
        *((Param(count, COUNT_DTYPE, "write"),) if count is not None else ()),
    )
    params += (
        Param(offset_name, torch.int64, "read"),
        Param(order_name, torch.int64, "read"),
        *(Param(key, context.buffer(key).dtype, "read") for key in leaves),
        *gate_params,
        Param(TARGET_SIZE.name, _INDEX),
        Param(TOTAL.name, _INDEX),
    )
    body = (
        *gate,
        Let(LINEAR, ThreadIndex()),
        Guard(Compare("<", LINEAR, TOTAL)),
        Let(MEMBER, Binary("/", LINEAR, TARGET_SIZE)),
        Let(_SCATTER_TARGET, Binary("%", LINEAR, TARGET_SIZE)),
        Let(first, Load(offset_name, _SCATTER_TARGET, _INDEX)),
        Let(cursor, first),
        Let(
            end,
            Load(offset_name, Binary("+", _SCATTER_TARGET, Const(1, _INDEX)), _INDEX),
        ),
        While(
            Compare("<", cursor, end),
            (
                Let(_SCATTER_SOURCE, Load(order_name, cursor, _INDEX)),
                *values.statements,
                Store(
                    buffer,
                    LINEAR,
                    Binary("+", Load(buffer, LINEAR, torch.float64), value),
                ),
                Assign(cursor, Binary("+", cursor, Const(1, _INDEX))),
            ),
            per_lane=True,
        ),
        *(
            (Store(count, LINEAR, cast(Binary("-", end, first), COUNT_DTYPE)),)
            if count is not None
            else ()
        ),
    )
    return StatisticsKernel(
        KernelFunction(f"hf_scatter_add_{symbol}", params, body),
        {"target_size": Count(width), "total": Count(width * members)},
        Count(width * members),
        mask,
    ), extra

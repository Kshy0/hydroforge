// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

// Exact table-order CSR weighted gather-accumulate for hydroforge.execution.remap.
//
// Each thread computes one (channel, linked_target) output by summing its CSR
// slice [csr_ptr[t], csr_ptr[t + 1]) in exact original link-table order,
// preserving bit-for-bit equivalence with sequential table-order accumulation
// across arbitrary rank counts and arbitrary target fan-in (e.g. global river
// basins draining tens of thousands of land cells into one coastal estuary).

template <class Ts, class Tw, class Tt>
__global__ void remap_csr_gather(
    const Ts* staging,
    const Tw* weights,
    const int64_t* csr_rows,
    const int64_t* csr_ptr,
    const int64_t* csr_slots,
    const int64_t* csr_entries,
    Tt* target,
    int64_t channels,
    int64_t num_targets,
    int64_t outer,
    int64_t n_tgt,
    int64_t inner,
    int64_t n_stage,
    int32_t use_entries
) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= channels * num_targets) return;
    const int64_t ch = idx / num_targets;
    const int64_t t = idx % num_targets;
    const int64_t o = ch / inner;
    const int64_t in_i = ch % inner;
    const int64_t start = csr_ptr[t];
    const int64_t end = csr_ptr[t + 1];
    const Ts* st = staging + ch * n_stage;
    Tt acc = Tt(0);
    for (int64_t p = start; p < end; ++p) {
        const Tw w = use_entries ? weights[csr_entries[p]] : weights[p];
        acc = acc + static_cast<Tt>(static_cast<Ts>(w) * st[csr_slots[p]]);
    }
    target[(o * n_tgt + csr_rows[t]) * inner + in_i] = acc;
}

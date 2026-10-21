// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

// Halo exchange of the CUDA backend (hydroforge.parallel.halo): senders
// store straight into the receiving peer's buffer and signal its arrival.
// grid.x is the chunk of a message, grid.y the message: sends, then receives.

#include "peer.cuh"

template <typename T0, typename T1, typename T2, typename TC, int NF,
          bool ACCUMULATE, bool HAS_CLEAR>
__global__ void halo_exchange(
    T0* f0, T1* f1, T2* f2, TC* clear, const int32_t* index, double* recv,
    int64_t* signals, int64_t* expected, const int64_t* recv_peers,
    const int64_t* signal_peers, int32_t out_peer, int32_t out_start,
    int32_t out_dst, int32_t out_idx, int32_t in_peer, int32_t in_start,
    int32_t in_off, int32_t in_idx, int32_t slot_offset, int32_t signal_offset,
    int32_t n_out, int32_t my_pe, int32_t max_chunks, int32_t chunk) {
    const int32_t c = blockIdx.x;
    const int32_t m = blockIdx.y;
    if (m < n_out) {
        const int32_t peer = index[out_peer + m];
        const int32_t base = index[out_start + m];
        const int32_t start = base + c * chunk;
        const int32_t end = min(index[out_start + m + 1], start + chunk);
        if (start >= end) return;
        double* out = reinterpret_cast<double*>(recv_peers[peer]) + slot_offset
                      + int64_t(index[out_dst + m] - base) * NF;
        for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
            const int32_t row = index[out_idx + i];
            double* slot = out + int64_t(i) * NF;
            slot[0] = double(f0[row]);
            if (ACCUMULATE) f0[row] = T0(0);
            if (NF > 1) {
                slot[1] = double(f1[row]);
                if (ACCUMULATE) f1[row] = T1(0);
            }
            if (NF > 2) {
                slot[2] = double(f2[row]);
                if (ACCUMULATE) f2[row] = T2(0);
            }
            if (HAS_CLEAR) clear[row] = TC(0);
        }
        hf_peer_arrive(reinterpret_cast<int64_t*>(signal_peers[peer]) + signal_offset
                       + my_pe * max_chunks + c);
        return;
    }
    const int32_t j = m - n_out;
    const int32_t peer = index[in_peer + j];
    const int32_t base = index[in_start + j];
    const int32_t start = base + c * chunk;
    const int32_t end = min(index[in_start + j + 1], start + chunk);
    if (start >= end) return;
    int64_t* count = expected + signal_offset + peer * max_chunks + c;
    const int64_t arrivals = *count + 1;
    hf_peer_wait(signals + signal_offset + peer * max_chunks + c, arrivals);
    if (threadIdx.x == 0) *count = arrivals;
    const double* in = recv + slot_offset + int64_t(index[in_off + j] - base) * NF;
    for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        const int32_t row = index[in_idx + i];
        // Written by a peer during this kernel: bypass the caches.
        const double* slot = in + int64_t(i) * NF;
        if (ACCUMULATE) {
            atomicAdd(f0 + row, T0(__ldcv(slot)));
            if (NF > 1) atomicAdd(f1 + row, T1(__ldcv(slot + 1)));
            if (NF > 2) atomicAdd(f2 + row, T2(__ldcv(slot + 2)));
        } else {
            f0[row] = T0(__ldcv(slot));
            if (NF > 1) f1[row] = T1(__ldcv(slot + 1));
            if (NF > 2) f2[row] = T2(__ldcv(slot + 2));
        }
    }
}

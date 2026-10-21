// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

// Minimum, maximum or sum of a few values over ranks mapped peer-to-peer
// (hydroforge.parallel.reduce). One block stores this rank's values into
// slot ``rank`` of every rank, signals each of them, waits for every rank's
// arrival and reduces the slots in rank order, so every rank holds the same
// result. Successive calls alternate two slot sets: a rank can run at most
// one call ahead of a peer still reading the previous one.

#include "peer.cuh"

__device__ __forceinline__ int64_t hf_bits(double value) { return __double_as_longlong(value); }
__device__ __forceinline__ int64_t hf_bits(float value) { return __float_as_int(value); }
__device__ __forceinline__ int64_t hf_bits(int32_t value) { return value; }
__device__ __forceinline__ int64_t hf_bits(int64_t value) { return value; }

template <typename T> __device__ __forceinline__ T hf_from_bits(int64_t bits) { return T(bits); }
template <> __device__ __forceinline__ double hf_from_bits<double>(int64_t bits) {
    return __longlong_as_double(bits);
}
template <> __device__ __forceinline__ float hf_from_bits<float>(int64_t bits) {
    return __int_as_float(int32_t(bits));
}

// OP: 0 minimum, 1 maximum, 2 sum.
template <typename T, int OP>
__global__ void peer_all_reduce(
    T* values, int64_t* slots, int64_t* signals, int64_t* calls,
    const int64_t* slot_peers, const int64_t* signal_peers, int32_t numel,
    int32_t rank, int32_t world, int32_t capacity) {
    const int64_t call = *calls;
    const int64_t set = (call & 1) * world;
    for (int32_t t = threadIdx.x; t < world * numel; t += blockDim.x) {
        const int32_t peer = t / numel;
        const int32_t i = t - peer * numel;
        int64_t* slot = reinterpret_cast<int64_t*>(slot_peers[peer]);
        slot[(set + rank) * capacity + i] = hf_bits(values[i]);
    }
    __syncthreads();
    if (threadIdx.x < world) {
        __threadfence_system();
        atomicAdd_system(
            reinterpret_cast<unsigned long long*>(signal_peers[threadIdx.x]), 1ull);
    }
    hf_peer_wait(signals, (call + 1) * world);
    for (int32_t i = threadIdx.x; i < numel; i += blockDim.x) {
        T result = hf_from_bits<T>(__ldcv(slots + set * capacity + i));
        for (int32_t peer = 1; peer < world; ++peer) {
            const T value = hf_from_bits<T>(__ldcv(slots + (set + peer) * capacity + i));
            result = OP == 2   ? result + value
                     : OP == 1 ? (value > result ? value : result)
                               : (value < result ? value : result);
        }
        values[i] = result;
    }
    if (threadIdx.x == 0) *calls = call + 1;
}

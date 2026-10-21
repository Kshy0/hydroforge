// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

// Arrival signals between ranks whose buffers are mapped peer-to-peer
// (hydroforge.parallel.symmetric). A signal word counts arrivals; its owner
// waits until the count reaches its own expectation, so the protocol holds
// under any number of identical executions on every rank.

// One acquiring load at system scope: peers write the word over NVLink/PCIe.
__device__ __forceinline__ int64_t hf_peer_acquire(const int64_t* address) {
    int64_t value;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];"
                 : "=l"(value) : "l"(address) : "memory");
    return value;
}

// Block-wide arrival: every thread's earlier stores to the peer are visible
// before the peer observes the incremented signal.
__device__ __forceinline__ void hf_peer_arrive(int64_t* peer_signal) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        atomicAdd_system(reinterpret_cast<unsigned long long*>(peer_signal), 1ull);
    }
}

// Block-wide wait until ``signal`` counts ``arrivals``.
__device__ __forceinline__ void hf_peer_wait(const int64_t* signal, int64_t arrivals) {
    if (threadIdx.x == 0) {
        while (hf_peer_acquire(signal) < arrivals) {
        }
    }
    __syncthreads();
}

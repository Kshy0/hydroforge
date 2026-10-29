// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

// Halo access from model kernels (hydroforge.kernels.halo).
//
// A KernelSpec declaring ``halo`` or ``publishes`` gets one parameter after
// its others, ``halo``, which CUDA kernels take by value:
//
//   __global__ void flux(..., const HfHalo halo)
//
// The framework prepares it once per call site (when the program is
// recorded). The descriptors of the first HF_HALO_INLINE reads and
// publications live in the struct, in the kernel's parameter space, so their
// interior rows compare against parameters, load no tables and wait for
// nothing; a kernel declaring more fields finds the rest in a table. Calls
// with no halo work (another link or backend, one rank, eager calls) get a
// struct whose interior covers every row.
//
// Every launch first calls ``hf_halo_begin(halo)`` (generated entries do;
// a hand-written kernel calls it before anything else, or its call is
// rejected): it returns the slots of messages earlier launches finished
// with to their senders.
//
// Reading (``halo``): read a declared field's element at any row through
// ``hf_halo_load(halo, k, field, index, o, row, i)`` (``k`` the field's
// position in ``halo``, ``index`` its offset in ``field``; ``o``, ``row``,
// ``i`` its outer index, row along the decomposed axis and inner index).
// Rows inside the field's interior band come from the field. A ghost row
// whose owner published it since comes from this rank's receive ring: the
// warp's threads waiting for one message elect one to wait until it
// arrived (acquire at system scope, so its bytes are visible), read it
// bypassing the L1 cache and store it into the field's ghost row, so later
// readers find it there. The kernel must read every ghost row within its
// declared distance of the rows it computes, for all of them, and compute
// every owned row.
//
// Message numbers: each directed pair numbers its messages; a rank counts
// those of each peer it finished with (``received``, its message number
// base) and pays their credits (the peer's slot is free again). Both
// happen in launch prologues and receive kernels the schedule picks, never
// in a launch reading from that peer, so the number of the message a read
// wants is ``received`` plus a count fixed per call site (``skip``).
//
// Publishing (``publishes``): while writing a published field, store each
// written element through ``hf_publish(halo, stored, f, o, row, i, value)``
// (``f`` the field's position in ``publishes``, ``stored`` a thread-local
// int counting the thread's ring stores, starting at 0), and call
// ``hf_publish_done(halo, stored)`` after the thread's last one (no barrier:
// threads may return whenever they like otherwise). Rows some peer holds as
// ghosts go straight into that peer's receive ring, once the peer returned
// the slot (its credit for the message the slot held before); the warp
// completing the launch's stores counts the message sent and counts its
// arrival at the peer. The kernel must store every element of every such
// row exactly once in that launch. A launch may read rings and publish:
// the credit, not the order of its threads, keeps unread messages intact.
// In a loop body whose schedule proved every slot free when stored (its
// own waits order each reuse after the read), publishing waits for no
// credit and prologues return none (``checks`` 0).
//
// Row tables (outside a field's interior band [lo, hi), row r at index
// ``r < lo ? r : r - (hi - lo)``) hold ring elements (HfRing): the byte
// offset of the row's element (o 0, i 0) in slot 0 from the window holding
// the ring, the peer (-1: none, read the field) and the elements per outer
// index. Message n of a pair sits ``(n & slot_mask) * capacity`` bytes
// further (a mask, not a 64-bit remainder: that would be a call).

#pragma once

// The CUDA backend defines these per call from its argument: a field this
// call reads from no ring (HF_HALO_READS(k) false for field k), a call
// publishing nothing, returning no slots or storing into slots proven free
// compiles that path away.
#ifndef HF_HALO_READS
#define HF_HALO_READS(k) true
#endif
#ifndef HF_HALO_PUBLISHES
#define HF_HALO_PUBLISHES 1
#endif
#ifndef HF_HALO_COMMITS
#define HF_HALO_COMMITS 1
#endif
#ifndef HF_HALO_CREDITS
#define HF_HALO_CREDITS 1
#endif

// Reads and publications whose descriptors sit in the struct itself.
#define HF_HALO_INLINE 8

struct HfHaloRead {
    int64_t lo, hi;          // rows read from the field without further checks
    const int64_t* ghosts;   // HfRing per row outside [lo, hi)
    const int64_t* skip;     // per peer: the message's number past ``received``
};

struct HfHaloPublish {
    int64_t lo, hi;          // rows published nowhere
    const int64_t* rowptr;   // per row outside [lo, hi): its HfRing entries
    const int64_t* sends;    // HfRing per (row, peer holding it)
};

struct HfHalo {
    const int64_t* peers;    // every rank's window, as this process maps it
    int64_t* state;          // sent[world], received[world], done[world], ...
    const int64_t* window;   // this rank's window: arrivals, credits, its ring
    int64_t capacity;        // slot bytes
    int64_t slot_mask;       // slots per directed pair - 1 (a power of two)
    int64_t world, rank;
    int64_t expected;        // ring stores of a publishing launch
    const int64_t* commit;   // the prologue: n, then n (peer, credits, counted)
    const int64_t* targets;  // the peers a publishing launch sends to
    int64_t messages;        // how many
    int64_t checks;          // publishing waits for slot credits (HF_HALO_CREDITS)
    HfHaloRead read[HF_HALO_INLINE];        // slots below HF_HALO_INLINE
    HfHaloPublish publish[HF_HALO_INLINE];
    const HfHaloRead* more_reads;           // slots from HF_HALO_INLINE on, or null
    const HfHaloPublish* more_publishes;
};

namespace hf_halo_access {
// Descriptor of declaration slot k (a constant at call sites, so the choice
// folds away); a slot past the inline ones needs its table.
__device__ __forceinline__ const HfHaloRead& read_slot(const HfHalo& h, int k) {
    return k < HF_HALO_INLINE ? h.read[k] : h.more_reads[k - HF_HALO_INLINE];
}
__device__ __forceinline__ const HfHaloPublish& publish_slot(const HfHalo& h, int f) {
    return f < HF_HALO_INLINE ? h.publish[f] : h.more_publishes[f - HF_HALO_INLINE];
}

__device__ __forceinline__ long long acquire(const int64_t* word) {
    long long value;
    asm volatile("ld.acquire.sys.global.s64 %0, [%1];" : "=l"(value) : "l"(word) : "memory");
    return value;
}

// ``amount`` counts at ``word`` of a peer's window, released at system scope.
__device__ __forceinline__ void signal(const int64_t* word, int64_t amount) {
    asm volatile("red.release.sys.global.add.u64 [%0], %1;"
                 ::"l"(const_cast<int64_t*>(word)), "l"(amount) : "memory");
}

// ``amount`` credits at ``word`` of a sender's window. A credit only says the
// slot's reads are over, and they returned their values before it is issued
// (in earlier launches), so no fence orders anything before it.
__device__ __forceinline__ void credit(const int64_t* word, int64_t amount) {
    asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;"
                 ::"l"(const_cast<int64_t*>(word)), "l"(amount) : "memory");
}

// Tables are read-only for the launch: through the read-only path.
__device__ __forceinline__ int64_t word(const int64_t* p) {
    return (int64_t)__ldg(reinterpret_cast<const long long*>(p));
}

__device__ __forceinline__ float uncached(const float* p) { return __ldcv(p); }
__device__ __forceinline__ double uncached(const double* p) { return __ldcv(p); }
__device__ __forceinline__ int uncached(const int* p) { return __ldcv(p); }
__device__ __forceinline__ long long uncached(const long long* p) { return __ldcv(p); }
__device__ __forceinline__ long uncached(const long* p) {
    return (long)__ldcv(reinterpret_cast<const long long*>(p));
}

__device__ __forceinline__ unsigned lane_id() {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// The prologue: messages earlier launches finished with are counted
// received and their slots credited to their senders (launches before this
// one in the stream are complete, so are their reads of those slots).
__device__ __forceinline__ void commit(const HfHalo& h) {
    const int64_t* list = h.commit;
    const int64_t n = word(list);
    for (int64_t e = 0; e < n; ++e) {
        const int64_t peer = word(list + 1 + 3 * e);
        const int64_t credits = word(list + 2 + 3 * e);
        const int64_t counted = word(list + 3 + 3 * e);
        if (counted) h.state[h.world + peer] += counted;
        if (credits)
            credit(reinterpret_cast<const int64_t*>(word(h.peers + peer)) + h.world + h.rank,
                   credits);
    }
}

// A ghost row from the ring (rows outside the interior band only).
template <class T>
__device__ __forceinline__ T ring_read(const HfHalo& h, const HfHaloRead& r, T* field,
                                       int64_t index, int64_t o, int64_t row, int64_t i) {
    const int64_t* ring = r.ghosts + 2 * (row < r.lo ? row : row - (r.hi - r.lo));
    const int64_t packed = word(ring + 1);
    const int peer = int(packed & 0xffffffff);
    if (peer < 0) return field[index];
    // ``received`` changes between launches only (never in one reading
    // from this peer): a plain load.
    const int64_t n = h.state[h.world + peer] + word(r.skip + peer);
    // Wait for this message's arrival; acquire makes its bytes visible. Lanes
    // waiting for the same message elect one to acquire; the warp barrier
    // orders the others' reads after it.
    const unsigned same = __match_any_sync(
        __activemask(), (unsigned long long)n * 4096ull + (unsigned long long)peer);
    if (lane_id() == unsigned(__ffs(same) - 1))
        while (acquire(h.window + peer) < n + 1) __nanosleep(64);
    __syncwarp(same);
    const T* element = reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(h.window) + word(ring) + (n & h.slot_mask) * h.capacity);
    const T value = uncached(element + o * (packed >> 32) + i);
    field[index] = value;  // the ghost row's current value, for later readers
    return value;
}

// Stores ``value`` into the ring of every peer holding ``row``; the stores.
template <class T>
__device__ __forceinline__ int ring_store(const HfHalo& h, const HfHaloPublish& p, int64_t o,
                                          int64_t row, int64_t i, T value) {
    const int64_t x = row < p.lo ? row : row - (p.hi - p.lo);
    const int64_t first = word(p.rowptr + x), last = word(p.rowptr + x + 1);
    for (int64_t e = first; e < last; ++e) {
        const int64_t packed = word(p.sends + 2 * e + 1);
        const int peer = int(packed & 0xffffffff);
        // Message n of the pair: this rank's sends counted so far.
        const int64_t n = h.state[peer];
#if HF_HALO_CREDITS
        // Its slot is free once the peer credited message n - depth (credits
        // count in this rank's window); lanes storing to one peer elect one
        // to wait.
        const unsigned same = __match_any_sync(__activemask(), unsigned(peer));
        if (lane_id() == unsigned(__ffs(same) - 1))
            while (acquire(h.window + h.world + peer) < n - h.slot_mask) __nanosleep(64);
        __syncwarp(same);
#endif
        T* element = reinterpret_cast<T*>(reinterpret_cast<char*>(word(h.peers + peer)) +
                                          word(p.sends + 2 * e) + (n & h.slot_mask) * h.capacity);
        element[o * (packed >> 32) + i] = value;
    }
    return int(last - first);
}

// The stores of the calling lanes, counted per warp: the warp completing
// the launch's stores counts each message sent and counts its arrival in
// the peer's window, released at system scope after every counted store.
__device__ __forceinline__ void finish_publish(const HfHalo& h, int stored) {
    const unsigned mask = __activemask();
    unsigned long long total = 0;
#if __CUDA_ARCH__ >= 800
    total = __reduce_add_sync(mask, unsigned(stored));
#else
    // Usually every lane stored the same number of elements (one per
    // published field): a vote instead of a shuffle per lane.
    if (__all_sync(mask, stored == 1)) {
        total = __popc(mask);
    } else if (__all_sync(mask, stored == 2)) {
        total = 2ull * __popc(mask);
    } else {
        for (unsigned rest = mask; rest; rest &= rest - 1)
            total += unsigned(__shfl_sync(mask, stored, __ffs(rest) - 1));
    }
#endif
    __syncwarp(mask);  // orders the lanes' stores before the leader's fence
    if (lane_id() != unsigned(__ffs(mask) - 1)) return;
    // GPU scope suffices here: the completing warp fences at system scope
    // before its release, and causality runs through the counter.
    __threadfence();
    // The done counter of this rank's own slot (no message to self).
    unsigned long long* done =
        reinterpret_cast<unsigned long long*>(h.state + 2 * h.world + h.rank);
    if (atomicAdd(done, total) + total != (unsigned long long)h.expected) return;
    *done = 0;
    __threadfence_system();
    for (int64_t m = 0; m < h.messages; ++m) {
        const int64_t peer = word(h.targets + m);
        h.state[peer] += 1;
        signal(reinterpret_cast<const int64_t*>(word(h.peers + peer)) + h.rank, 1);
    }
}
}  // namespace hf_halo_access

// First in every launch (see the top of this file).
__device__ __forceinline__ void hf_halo_begin(const HfHalo& h) {
#if HF_HALO_COMMITS
    if ((threadIdx.x | threadIdx.y | threadIdx.z | blockIdx.x | blockIdx.y | blockIdx.z) == 0 &&
        h.commit != nullptr)
        hf_halo_access::commit(h);
#endif
}

template <class T>
__device__ __forceinline__ T hf_halo_load(const HfHalo& h, int k, T* field, int64_t index,
                                          int64_t o, int64_t row, int64_t i) {
    // A field no ring holds in this call: a plain load (k is a constant at
    // the call site, so the test folds away).
    if (!HF_HALO_READS(k) || (k >= HF_HALO_INLINE && h.more_reads == nullptr))
        return field[index];
    const HfHaloRead& r = hf_halo_access::read_slot(h, k);
    // Interior rows: two compares against kernel parameters.
    if (row >= r.lo && row < r.hi) return field[index];
    return hf_halo_access::ring_read(h, r, field, index, o, row, i);
}

// The read descriptor of declaration slot k, by value: interior everywhere
// when no ring holds the field in this call.
__device__ __forceinline__ HfHaloRead hf_halo_read(const HfHalo& h, int k) {
    if (k >= HF_HALO_INLINE && h.more_reads == nullptr)
        return HfHaloRead{0, int64_t(1) << 62, nullptr, nullptr};
    return hf_halo_access::read_slot(h, k);
}

template <class T>
__device__ __forceinline__ void hf_publish(const HfHalo& h, int& stored, int f, int64_t o,
                                           int64_t row, int64_t i, T value) {
#if HF_HALO_PUBLISHES
    if (f >= HF_HALO_INLINE && h.more_publishes == nullptr) return;
    const HfHaloPublish& p = hf_halo_access::publish_slot(h, f);
    // Rows published nowhere: two compares against kernel parameters.
    if (row >= p.lo && row < p.hi) return;
    stored += hf_halo_access::ring_store(h, p, o, row, i, value);
#endif
}

// Whether row ``row`` of published field ``f`` goes to some ring: test it
// first where the value to publish costs a load (``if (hf_publishes_row(
// halo, f, row)) hf_publish(halo, stored, f, o, row, i, field[index])``), so
// rows published nowhere load nothing.
__device__ __forceinline__ bool hf_publishes_row(const HfHalo& h, int f, int64_t row) {
#if HF_HALO_PUBLISHES
    if (f >= HF_HALO_INLINE && h.more_publishes == nullptr) return false;
    const HfHaloPublish& p = hf_halo_access::publish_slot(h, f);
    return row < p.lo || row >= p.hi;
#else
    return false;
#endif
}

// After a thread's last hf_publish (see the top of this file).
__device__ __forceinline__ void hf_publish_done(const HfHalo& h, int stored) {
#if HF_HALO_PUBLISHES
    if (stored != 0) hf_halo_access::finish_publish(h, stored);
#endif
}

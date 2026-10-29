// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

// Halo exchange of the CUDA backend (hydroforge.parallel.halo, launched by
// hydroforge.kernels.halo).
//
// A message to one peer carries the rows of its declared fields, field after
// field, each element in its own bits (no conversion). A field is viewed as
// (outer, extent, inner) around its decomposed axis; a message holds, for
// each outer index, the listed rows of that axis with all their inner
// elements. A mirror lists its rows nearest the receiver's owned rows first,
// and a field of depth d carries the leading rows within d (its record
// counts them). grid.x is a chunk of a message, grid.y the message.
//
// Peers NCCL (or CUDA IPC) maps take stores straight into their receive
// ring: every directed pair owns DEPTH slots, message n of the pair using
// slot n % DEPTH. The sender stores, then counts an arrival in the
// receiver's window; the receiver waits for it and reads; once the
// receiver finished with it, it counts a credit in the sender's window;
// the sender of message n first waits for the credit of message n - DEPTH.
// So a message never overwrites one its receiver has not finished reading,
// however far the sender runs ahead. Other peers go through NCCL
// send/receive between the two kernels, from and into local staging slots.
//
// Kernels publish into the same rings and read from them
// (halo_access.cuh), so a receiver may finish with a message long after it
// arrived: each exchange gets a site table from the recorded schedule
// (hydroforge.execution.halo_schedule). Per peer, ``skip`` places the
// message it receives past ``received`` (messages before it still pending),
// ``settle`` counts how many messages its receive finishes (``received``
// advances: its own, those before it read since, or none while one before
// it is still pending) and ``post`` credits as many. Its first launch also
// runs the prologue of halo_access.cuh (counts and credits of messages
// earlier launches finished with), before any wait of its own.
//
// In a loop body whose schedule proved every slot free when stored, sends
// wait for no credit (``checks`` 0) and nothing returns any: halo_drain
// first waits until every message sent before the loop is credited, and
// halo_flush afterwards sets each peer's credits to what this rank counted.
//
// An accumulate message may list one owned row twice (a periodic slab over
// two ranks): REPEATS marks it, and its receive runs once per pass, pass p
// adding the entries the row table ranks p-th for their row, so no two
// threads of a launch add into one element.

#include "peer.cuh"

#ifndef HF_HALO_FIELDS
#define HF_HALO_FIELDS 8
#endif

namespace hf_halo {

constexpr int SLOTS = HF_HALO_FIELDS + 1;  // exchanged fields and the clear field
constexpr int WRITTEN_WORDS = (SLOTS + 63) / 64;
constexpr int KINDS = 3;
constexpr int COUNTS = 4 + KINDS;  // a record's row count of each kind
constexpr int WITHIN = 4 + 2 * KINDS;  // then its rows within each depth
constexpr int LINK_PEER = 0;
constexpr int REPEATS = 32;  // flag of the link word: some row is listed twice
constexpr int LINK = 15;  // the link code bits

}  // namespace hf_halo

// By value, sized by the field count this kernel was compiled for.
struct HaloFields {
    char* data[hf_halo::SLOTS];
    int64_t outer[hf_halo::SLOTS];
    int64_t extent[hf_halo::SLOTS];
    int64_t inner[hf_halo::SLOTS];
    int32_t size[hf_halo::SLOTS];  // bytes per element
    int32_t type[hf_halo::SLOTS];  // accumulate: 1 f32, 2 f64, 3 i32, 4 i64
    int32_t kind[hf_halo::SLOTS];
    int32_t depth[hf_halo::SLOTS];  // ghost rows within this depth (0: all)
    int32_t count;                 // exchanged fields
    int32_t clear;                 // the last slot is a field to zero
    uint64_t written[hf_halo::WRITTEN_WORDS];  // receives: bit k writes field k
};

// Message record: peer, link (and REPEATS), ring slot (the sender's slot in
// the receiver's ring), staging slot, the row-table offset and row count of
// every kind, then per kind its rows within each depth 1 .. levels.
struct HaloRing {
    char* window;          // arrivals[world], credits[world], ring
    const int64_t* peers;  // every rank's window as this process maps it
    int64_t* state;        // sent[world], received[world], sends done, receives done
    char* send_stage;
    const char* recv_stage;
    const int32_t* rows;   // every entry's row, then its pass
    const int32_t* messages;
    int64_t capacity;      // bytes of one message slot
    int64_t ring;          // byte offset of the ring in a window
    int64_t entries;       // entries of the row table (where passes start)
    // The site table: skip[world], settle[world], post[world], then the
    // prologue: n, n (peer, credits, counted).
    const int64_t* site;
    int32_t world;
    int32_t rank;
    int32_t depth;
    int32_t chunk;         // elements one block moves
    int32_t checks;        // sends wait for their slot's credit
    int32_t record;        // words of a message record
    int32_t levels;        // depths a record counts rows within
};

namespace hf_halo {

struct Layout {
    int64_t count[SLOTS];   // elements of each field in the message
    int64_t offset[SLOTS];  // byte offset of each field's segment
    int32_t first[SLOTS];   // the field's rows in the row table
    int32_t rows[SLOTS];
    int64_t packed;         // elements of the exchanged fields
    int64_t blocks;         // blocks the message needs
};

__device__ __forceinline__ int64_t aligned(int64_t bytes) { return (bytes + 15) & ~int64_t(15); }

// The message's layout, by the whole block: one thread per field reads its
// rows, thread 0 then lays the segments out.
__device__ void layout(const HaloFields& f, const HaloRing& r, const int32_t* rec, Layout& out) {
    for (int k = threadIdx.x; k < SLOTS; k += blockDim.x) {
        if (k < f.count || (k == SLOTS - 1 && f.clear)) {
            const int depth = f.depth[k] < r.levels ? f.depth[k] : 0;
            out.first[k] = __ldg(rec + 4 + f.kind[k]);
            out.rows[k] = __ldg(depth ? rec + WITHIN + f.kind[k] * r.levels + depth - 1
                                      : rec + COUNTS + f.kind[k]);
            out.count[k] = int64_t(out.rows[k]) * f.outer[k] * f.inner[k];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int64_t bytes = 0;
        out.packed = 0;
        for (int j = 0; j < f.count; ++j) {
            out.offset[j] = bytes;
            bytes += aligned(out.count[j] * f.size[j]);
            out.packed += out.count[j];
        }
        const int64_t cleared = f.clear ? out.count[SLOTS - 1] : 0;
        const int64_t longest = out.packed > cleared ? out.packed : cleared;
        out.blocks = (longest + r.chunk - 1) / r.chunk;
    }
    __syncthreads();
}

// Field element of element e (field-local) of field k in the message: the
// message holds, for each outer index, the field's rows with their inner
// elements. Message elements fit 32 bits; field offsets need not. ``entry``
// receives the element's entry in the row table.
__device__ __forceinline__ int64_t element(const HaloFields& f, const Layout& s,
                                           const int32_t* rows, int k, int64_t e,
                                           int32_t* entry = nullptr) {
    const uint32_t inner = uint32_t(f.inner[k]);
    const uint32_t plane = uint32_t(s.rows[k]) * inner;
    const uint32_t local = uint32_t(e);
    const uint32_t o = local / plane;
    const uint32_t rest = local - o * plane;
    const uint32_t i = rest / inner;
    if (entry) *entry = s.first[k] + int32_t(i);
    const int64_t row = __ldg(rows + s.first[k] + i);
    return (int64_t(o) * f.extent[k] + row) * inner + (rest - i * inner);
}

// Element bits of 1, 2, 4 or 8 bytes. A peer wrote the ring during this
// kernel: ``bypass`` loads skip the caches.
__device__ __forceinline__ unsigned long long load_bits(const char* from, int size, bool bypass) {
    switch (size) {
        case 8: {
            const long long* p = reinterpret_cast<const long long*>(from);
            return (unsigned long long)(bypass ? __ldcv(p) : *p);
        }
        case 4: {
            const unsigned int* p = reinterpret_cast<const unsigned int*>(from);
            return bypass ? __ldcv(p) : *p;
        }
        case 2: {
            const unsigned short* p = reinterpret_cast<const unsigned short*>(from);
            return bypass ? __ldcv(p) : *p;
        }
        default: {
            const unsigned char* p = reinterpret_cast<const unsigned char*>(from);
            return bypass ? __ldcv(p) : *p;
        }
    }
}

__device__ __forceinline__ void store_bits(char* to, unsigned long long bits, int size) {
    switch (size) {
        case 8: *reinterpret_cast<unsigned long long*>(to) = bits; break;
        case 4: *reinterpret_cast<unsigned int*>(to) = (unsigned int)bits; break;
        case 2: *reinterpret_cast<unsigned short*>(to) = (unsigned short)bits; break;
        default: *reinterpret_cast<unsigned char*>(to) = (unsigned char)bits;
    }
}

__device__ __forceinline__ void add_bits(char* to, unsigned long long bits, int type) {
    switch (type) {
        case 1: *reinterpret_cast<float*>(to) += __int_as_float(int(bits)); break;
        case 2: *reinterpret_cast<double*>(to) += __longlong_as_double((long long)bits); break;
        case 3: *reinterpret_cast<int*>(to) += int(bits); break;
        default: *reinterpret_cast<long long*>(to) += (long long)bits; break;
    }
}

// Moves elements [begin, end) of a message between the fields and the wire,
// UNROLL elements per thread with their loads in flight together. PASSES
// moves only the entries ``passes`` ranks ``pass`` for their row.
template <bool SEND, bool ACCUMULATE, bool PASSES = false>
__device__ __forceinline__ void move(const HaloFields& f, const Layout& s, const int32_t* rows,
                                     char* wire, int64_t begin, int64_t end, bool bypass,
                                     const int32_t* passes = nullptr, int32_t pass = 0) {
    constexpr int UNROLL = 4;
    for (int64_t e0 = begin + threadIdx.x; e0 < end; e0 += int64_t(blockDim.x) * UNROLL) {
        char* field[UNROLL];
        char* message[UNROLL];
        int size[UNROLL], type[UNROLL];
        bool live[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int64_t e = e0 + int64_t(u) * blockDim.x;
            live[u] = e < end;
            if (!live[u]) continue;
            int k = 0;
            int64_t local = e;
            while (local >= s.count[k]) local -= s.count[k++];
            if (!SEND && !(f.written[k >> 6] >> (k & 63) & 1)) {
                live[u] = false;
                continue;
            }
            size[u] = f.size[k];
            type[u] = f.type[k];
            int32_t entry;
            field[u] = f.data[k] + element(f, s, rows, k, local, PASSES ? &entry : nullptr) * size[u];
            if (PASSES && __ldg(passes + entry) != pass) {
                live[u] = false;
                continue;
            }
            message[u] = wire + s.offset[k] + local * size[u];
        }
        unsigned long long bits[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            if (live[u]) bits[u] = load_bits(SEND ? field[u] : message[u], size[u], !SEND && bypass);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if (!live[u]) continue;
            if (SEND) {
                store_bits(message[u], bits[u], size[u]);
                if (ACCUMULATE) store_bits(field[u], 0, size[u]);
            } else if (ACCUMULATE) {
                add_bits(field[u], bits[u], type[u]);
            } else {
                store_bits(field[u], bits[u], size[u]);
            }
        }
    }
}

// The slot message n of the pair uses in the receiver's ring.
__device__ __forceinline__ int64_t ring_slot(const HaloRing& r, const int32_t* rec, int64_t n) {
    return r.ring + (int64_t(rec[2]) * r.depth + n % r.depth) * r.capacity;
}

// Called by thread 0 once its block is done with a message (after a block
// barrier): whether it is the last of the message's ``blocks``. Each block of
// a multi-block message fences its work before counting it.
__device__ __forceinline__ bool last_block(int64_t* done, int64_t blocks) {
    if (blocks == 1) return true;
    __threadfence_system();
    const unsigned long long previous =
        atomicAdd(reinterpret_cast<unsigned long long*>(done), 1ull);
    if (int64_t(previous) != blocks - 1) return false;
    *done = 0;
    return true;
}

// ``amount`` counts in a peer's window, released at system scope: the
// stores of every block that counted before it are visible first.
__device__ __forceinline__ void count(const int64_t* window, int64_t index,
                                      int64_t amount = 1) {
    unsigned long long* word =
        reinterpret_cast<unsigned long long*>(const_cast<int64_t*>(window) + index);
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" ::"l"(word), "l"(amount) : "memory");
}

// ``amount`` credits in a sender's window. A credit only says the slot's
// reads are over; they returned their values before it is issued (earlier
// launches, or this launch's blocks before their barrier and done count),
// so no fence orders anything before it.
__device__ __forceinline__ void credit(const int64_t* window, int64_t index, int64_t amount) {
    unsigned long long* word =
        reinterpret_cast<unsigned long long*>(const_cast<int64_t*>(window) + index);
    asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(word), "l"(amount) : "memory");
}

// The prologue of an exchange's first launch (block 0, thread 0, before
// any wait): counts and credits of messages earlier launches finished with.
__device__ __forceinline__ void prologue(const HaloRing& r) {
    if ((threadIdx.x | blockIdx.x | blockIdx.y) != 0) return;
    const int64_t* list = r.site + 3 * r.world;
    for (int64_t e = 0, n = list[0]; e < n; ++e) {
        const int64_t peer = list[1 + 3 * e], credits = list[2 + 3 * e];
        const int64_t counted = list[3 + 3 * e];
        if (counted) r.state[r.world + peer] += counted;
        if (credits)
            credit(reinterpret_cast<const int64_t*>(r.peers[peer]), r.world + r.rank, credits);
    }
}

template <bool ACCUMULATE>
__device__ void send(const HaloFields& f, const HaloRing& r, int32_t m) {
    const int32_t* rec = r.messages + m * r.record;
    __shared__ Layout s;
    layout(f, r, rec, s);
    if (int64_t(blockIdx.x) >= s.blocks) return;
    const int32_t peer = rec[0];
    const bool direct = (rec[1] & LINK) == LINK_PEER;
    int64_t* sent = r.state + peer;
    const int64_t n = *sent;
    char* out = nullptr;
    if (s.packed > 0) {
        if (direct) {
            // Message n - DEPTH left this slot once its credit arrived.
            if (r.checks)
                hf_peer_wait(reinterpret_cast<const int64_t*>(r.window) + r.world + peer,
                             n - r.depth + 1);
            out = reinterpret_cast<char*>(r.peers[peer]) + ring_slot(r, rec, n);
        } else {
            out = r.send_stage + int64_t(rec[3]) * r.capacity;
        }
    }
    const int64_t begin = int64_t(blockIdx.x) * r.chunk;
    const int64_t stop = begin + r.chunk < s.packed ? begin + r.chunk : s.packed;
    move<true, ACCUMULATE>(f, s, r.rows, out, begin, stop, false);
    if (ACCUMULATE && f.clear) {
        const int k = SLOTS - 1;
        const int64_t end = begin + r.chunk < s.count[k] ? begin + r.chunk : s.count[k];
        for (int64_t e = begin + threadIdx.x; e < end; e += blockDim.x)
            store_bits(f.data[k] + element(f, s, r.rows, k, e) * f.size[k], 0, f.size[k]);
    }
    if (s.packed == 0) return;
    __syncthreads();
    if (threadIdx.x == 0 && last_block(r.state + 2 * r.world + peer, s.blocks)) {
        *sent = n + 1;
        // The arrival in the receiver's window.
        if (direct) count(reinterpret_cast<const int64_t*>(r.peers[peer]), r.rank);
    }
}

// Pass ``pass`` of the message's receive; the ``last`` pass counts it read.
template <bool ACCUMULATE>
__device__ void receive(const HaloFields& f, const HaloRing& r, int32_t m, int32_t pass,
                        bool last) {
    const int32_t* rec = r.messages + m * r.record;
    __shared__ Layout s;
    layout(f, r, rec, s);
    // Receivers move the exchanged fields only (no clear).
    if (int64_t(blockIdx.x) >= (s.packed + r.chunk - 1) / r.chunk) return;
    const int32_t peer = rec[0];
    const bool direct = (rec[1] & LINK) == LINK_PEER;
    const int32_t* passes = ACCUMULATE && (rec[1] & REPEATS) ? r.rows + r.entries : nullptr;
    int64_t* received = r.state + r.world + peer;
    // Messages of this peer still pending precede this one.
    const int64_t n = *received + r.site[peer];
    char* in;
    if (direct) {
        hf_peer_wait(reinterpret_cast<const int64_t*>(r.window) + peer, n + 1);
        in = r.window + ring_slot(r, rec, n);
    } else {
        in = const_cast<char*>(r.recv_stage) + int64_t(rec[3]) * r.capacity;
    }
    const int64_t begin = int64_t(blockIdx.x) * r.chunk;
    const int64_t stop = begin + r.chunk < s.packed ? begin + r.chunk : s.packed;
    if (ACCUMULATE && passes)
        move<false, ACCUMULATE, true>(f, s, r.rows, in, begin, stop, direct, passes, pass);
    else
        move<false, ACCUMULATE>(f, s, r.rows, in, begin, stop, direct);
    __syncthreads();
    if (threadIdx.x == 0 &&
        last_block(r.state + 3 * r.world + peer, (s.packed + r.chunk - 1) / r.chunk) && last) {
        *received += r.site[r.world + peer];
        // The credits in the sender's window: the slots are free again.
        const int64_t credits = r.site[2 * r.world + peer];
        if (direct && credits)
            credit(reinterpret_cast<const int64_t*>(r.peers[peer]), r.world + r.rank, credits);
    }
}

}  // namespace hf_halo

// Messages first .. first + gridDim.y of the record table; ``begin``: the
// exchange's first launch (it runs the prologue).
template <bool ACCUMULATE>
__global__ void halo_send(HaloFields f, HaloRing r, int32_t first, int32_t begin) {
    if (begin) hf_halo::prologue(r);
    hf_halo::send<ACCUMULATE>(f, r, first + blockIdx.y);
}

// A message with repeated rows (accumulate) launches once per pass.
template <bool ACCUMULATE>
__global__ void halo_receive(HaloFields f, HaloRing r, int32_t first, int32_t pass,
                             int32_t last, int32_t begin) {
    if (begin) hf_halo::prologue(r);
    hf_halo::receive<ACCUMULATE>(f, r, first + blockIdx.y, pass, last != 0);
}

// Only the prologue: messages finished with, counted and credited.
__global__ void halo_book(HaloRing r) { hf_halo::prologue(r); }

// Before a loop storing without credits: every message this rank sent so
// far is credited (its receiver finished with it), so the loop reuses no
// slot still unread.
__global__ void halo_drain(HaloRing r) {
    for (int32_t peer = threadIdx.x; peer < r.world; peer += blockDim.x) {
        const int64_t sent = r.state[peer];
        while (hf_peer_acquire(reinterpret_cast<const int64_t*>(r.window) + r.world + peer) <
               sent) {
        }
    }
}

// After it: the prologue's counts, then each peer's credits set to every
// message of it this rank counted (finished with). This rank alone writes
// them, its earlier launches' credits are complete, and the slots' reads
// returned before.
__global__ void halo_flush(HaloRing r) {
    if ((threadIdx.x | blockIdx.x | blockIdx.y) != 0) return;
    hf_halo::prologue(r);
    for (int32_t peer = 0; peer < r.world; ++peer) {
        const int64_t received = r.state[r.world + peer];
        if (peer == r.rank || received == 0) continue;
        int64_t* word = reinterpret_cast<int64_t*>(r.peers[peer]) + r.world + r.rank;
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(word), "l"(received) : "memory");
    }
}

// A mirror whose messages all go to mapped peers: sends (lower blocks,
// dispatched first) then receives, in one launch.
__global__ void halo_mirror(HaloFields f, HaloRing r, int32_t sends, int32_t count,
                            int32_t receives) {
    hf_halo::prologue(r);
    if (int32_t(blockIdx.y) < count)
        hf_halo::send<false>(f, r, sends + blockIdx.y);
    else
        hf_halo::receive<false>(f, r, receives + blockIdx.y - count, 0, true);
}

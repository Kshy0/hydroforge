# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
CUDA Graph automation for hydroforge models.

Internal CUDA Graph and conditional-WHILE support used by explicit compiled
substep scopes. Downstream models declare only physical operator order; they
do not select, capture, or replay backend graphs themselves.

Mutable state discovery is automatic. Registered-kernel access metadata and
ATen alias schemas provide exact write sets; capture warmup is rolled back
before the live launch.

"""

from __future__ import annotations

import functools

import torch

# Categories whose tensors are mutated during a physics step
_MUTABLE_CATEGORIES = frozenset({"init_state", "state", "shared_state"})


# ====================================================================== #
# Device-side conditional-graph (WHILE node) support
# ====================================================================== #
# A CUDA conditional-graph WHILE node folds a variable-length sub-step loop into
# one graph launch: the body and its continuation predicate run on-device, so
# the host issues one launch per interval with zero per-iteration sync.
# Requires a CUDA toolkit/driver with conditional graph nodes (CUDA >= 12.4).

_COND_CPP = """
int64_t cwg_create();
void cwg_begin_capture(int64_t h, int64_t stream);
void cwg_end_capture(int64_t h, int64_t stream);
void cwg_set_conditional(int64_t h, at::Tensor cont, int64_t set_cond, int64_t stream);
void cwg_fixed_begin(at::Tensor count, at::Tensor counter,
                     at::Tensor midpoint, int64_t stream);
void cwg_fixed_end(at::Tensor count, at::Tensor counter,
                   at::Tensor cont, int64_t stream);
void cwg_fixed_stats_end(at::Tensor count, at::Tensor counter,
                         at::Tensor cont, at::Tensor weight_src,
                         at::Tensor weight, at::Tensor sub_step,
                         at::Tensor num_sub_steps, int64_t stream);
void cwg_stats_control(at::Tensor weight_src, at::Tensor cont, at::Tensor counter,
                       at::Tensor weight, at::Tensor sub_step, at::Tensor num_sub_steps,
                       int64_t stream);
void cwg_instantiate(int64_t h);
void cwg_launch(int64_t h, int64_t stream);
void cwg_destroy(int64_t h);
"""

_COND_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CWG_CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  TORCH_CHECK(false, "CUDA ", #x, " -> ", cudaGetErrorString(e)); }}while(0)

struct CondWhileGraph {
    cudaGraph_t graph = nullptr;
    cudaGraph_t body = nullptr;
    cudaGraphConditionalHandle handle = 0;
    cudaGraphExec_t exec = nullptr;
};

// Outer graph holding one WHILE conditional node.  Handle default value 1 makes
// the body run at least once per launch (the first sub-step always executes).
int64_t cwg_create() {
    auto* g = new CondWhileGraph();
    CWG_CK(cudaGraphCreate(&g->graph, 0));
    CWG_CK(cudaGraphConditionalHandleCreate(&g->handle, g->graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams cp = {};
    cp.type = cudaGraphNodeTypeConditional;
    cp.conditional.handle = g->handle;
    cp.conditional.type = cudaGraphCondTypeWhile;
    cp.conditional.size = 1;
    cudaGraphNode_t cnode;
#if CUDART_VERSION >= 13000
    CWG_CK(cudaGraphAddNode(
        &cnode, g->graph, nullptr, nullptr, 0, &cp));
#else
    CWG_CK(cudaGraphAddNode_v2(
        &cnode, g->graph, nullptr, nullptr, 0, &cp));
#endif
    g->body = cp.conditional.phGraph_out[0];
    return reinterpret_cast<int64_t>(g);
}

void cwg_begin_capture(int64_t h, int64_t stream) {
    auto* g = reinterpret_cast<CondWhileGraph*>(h);
    CWG_CK(cudaStreamBeginCaptureToGraph((cudaStream_t)stream, g->body, nullptr,
                                         nullptr, 0, cudaStreamCaptureModeThreadLocal));
}

void cwg_end_capture(int64_t h, int64_t stream) {
    (void)h;
    cudaGraph_t out;
    CWG_CK(cudaStreamEndCapture((cudaStream_t)stream, &out));
}

// Generic continuation predicate: read the model's (1,) int "continue?" flag and
// feed it to cudaGraphSetConditional.  ``set_cond`` is false during warmup, where
// the call is illegal outside conditional execution, so the kernel still loads.
__global__ void k_set_conditional(cudaGraphConditionalHandle handle,
                                  const int* __restrict__ cont, int set_cond) {
    if (set_cond) cudaGraphSetConditional(handle, (*cont) ? 1u : 0u);
}

void cwg_set_conditional(int64_t h, at::Tensor cont, int64_t set_cond, int64_t stream) {
    auto* g = reinterpret_cast<CondWhileGraph*>(h);
    k_set_conditional<<<1, 1, 0, (cudaStream_t)stream>>>(
        g->handle, cont.data_ptr<int>(), (int)set_cond);
}

template <typename T>
__global__ void k_fixed_begin(const int* __restrict__ count,
        const int* __restrict__ counter, T* __restrict__ midpoint) {
    *midpoint = (static_cast<T>(*counter) + static_cast<T>(0.5))
        / static_cast<T>(*count);
}

void cwg_fixed_begin(at::Tensor count, at::Tensor counter,
                     at::Tensor midpoint, int64_t stream) {
    AT_DISPATCH_FLOATING_TYPES(midpoint.scalar_type(), "cwg_fixed_begin", [&] {
        k_fixed_begin<scalar_t><<<1, 1, 0, (cudaStream_t)stream>>>(
            count.data_ptr<int>(), counter.data_ptr<int>(),
            midpoint.data_ptr<scalar_t>());
    });
}

__global__ void k_fixed_end(const int* __restrict__ count,
        int* __restrict__ counter, int* __restrict__ cont) {
    int next = *counter + 1;
    *counter = next;
    *cont = next < *count;
}

void cwg_fixed_end(at::Tensor count, at::Tensor counter,
                   at::Tensor cont, int64_t stream) {
    k_fixed_end<<<1, 1, 0, (cudaStream_t)stream>>>(
        count.data_ptr<int>(), counter.data_ptr<int>(), cont.data_ptr<int>());
}

template <typename T>
__global__ void k_fixed_stats_end(const int* __restrict__ count,
        int* __restrict__ counter, int* __restrict__ cont,
        const T* __restrict__ weight_src, float* __restrict__ weight,
        int* __restrict__ sub_step, int* __restrict__ num_sub_steps) {
    int next = *counter + 1;
    bool first = next == 1;
    bool last = next == *count;
    *counter = next;
    *cont = !last;
    int ss, n;
    if (first && last) { ss = 0; n = 1; }
    else if (first)    { ss = 0; n = 2; }
    else if (last)     { ss = 1; n = 2; }
    else               { ss = 1; n = 3; }
    *weight = static_cast<float>(*weight_src);
    *sub_step = ss;
    *num_sub_steps = n;
}

void cwg_fixed_stats_end(at::Tensor count, at::Tensor counter,
                         at::Tensor cont, at::Tensor weight_src,
                         at::Tensor weight, at::Tensor sub_step,
                         at::Tensor num_sub_steps, int64_t stream) {
    AT_DISPATCH_FLOATING_TYPES(weight_src.scalar_type(), "cwg_fixed_stats_end", [&] {
        k_fixed_stats_end<scalar_t><<<1, 1, 0, (cudaStream_t)stream>>>(
            count.data_ptr<int>(), counter.data_ptr<int>(), cont.data_ptr<int>(),
            weight_src.data_ptr<scalar_t>(), weight.data_ptr<float>(),
            sub_step.data_ptr<int>(), num_sub_steps.data_ptr<int>());
    });
}

// Statistics-control bridge for the folded aggregator path.  From the 1-based
// sub-step counter, the continue_flag (0 on the final sub-step) and the
// per-sub-step weight (e.g. dt), writes the aggregator's __weight / __sub_step /
// __num_sub_steps so its is_inner_first (sub_step == 0) and is_inner_last
// (sub_step == num_sub_steps - 1) fire without knowing the total count ahead:
//   first & last -> (0, 1);  first -> (0, 2);  last -> (1, 2);  middle -> (1, 3).
// Exact for inner ops {last, mean, sum, max, min, first}; not mid / arg*.
template <typename T>
__global__ void k_stats_control(const T* __restrict__ weight_src,
        const int* __restrict__ cont, const int* __restrict__ counter,
        float* __restrict__ weight, int* __restrict__ sub_step,
        int* __restrict__ num_sub_steps) {
    bool first = (*counter == 1);
    bool last = (*cont == 0);
    int ss, n;
    if (first && last) { ss = 0; n = 1; }
    else if (first)    { ss = 0; n = 2; }
    else if (last)     { ss = 1; n = 2; }
    else               { ss = 1; n = 3; }
    *weight = (float)(*weight_src);
    *sub_step = ss;
    *num_sub_steps = n;
}

void cwg_stats_control(at::Tensor weight_src, at::Tensor cont, at::Tensor counter,
                       at::Tensor weight, at::Tensor sub_step, at::Tensor num_sub_steps,
                       int64_t stream) {
    AT_DISPATCH_FLOATING_TYPES(weight_src.scalar_type(), "cwg_stats_control", [&] {
        k_stats_control<scalar_t><<<1, 1, 0, (cudaStream_t)stream>>>(
            weight_src.data_ptr<scalar_t>(), cont.data_ptr<int>(), counter.data_ptr<int>(),
            weight.data_ptr<float>(), sub_step.data_ptr<int>(), num_sub_steps.data_ptr<int>());
    });
}

void cwg_instantiate(int64_t h) {
    auto* g = reinterpret_cast<CondWhileGraph*>(h);
    CWG_CK(cudaGraphInstantiate(&g->exec, g->graph, 0));
}

void cwg_launch(int64_t h, int64_t stream) {
    auto* g = reinterpret_cast<CondWhileGraph*>(h);
    CWG_CK(cudaGraphLaunch(g->exec, (cudaStream_t)stream));
}

void cwg_destroy(int64_t h) {
    auto* g = reinterpret_cast<CondWhileGraph*>(h);
    if (g->exec) cudaGraphExecDestroy(g->exec);
    if (g->graph) cudaGraphDestroy(g->graph);
    delete g;
}
"""


@functools.lru_cache(maxsize=1)
def _cond_ext():
    from hydroforge.kernels.backends.cuda.build import load_inline_cu_module
    return load_inline_cu_module(
        name="hydroforge_conditional_while_graph",
        cpp_sources=_COND_CPP,
        cuda_sources=_COND_CUDA,
        functions=["cwg_create", "cwg_begin_capture", "cwg_end_capture",
                   "cwg_set_conditional", "cwg_fixed_begin", "cwg_fixed_end",
                   "cwg_fixed_stats_end", "cwg_stats_control",
                   "cwg_instantiate", "cwg_launch", "cwg_destroy"],
        extra_cuda_cflags=("-O3",),
    )


def fixed_control_begin(
    count: torch.Tensor, counter: torch.Tensor, midpoint: torch.Tensor,
    stream_ptr: int,
) -> None:
    _cond_ext().cwg_fixed_begin(count, counter, midpoint, stream_ptr)


def fixed_control_end(
    count: torch.Tensor, counter: torch.Tensor, continue_flag: torch.Tensor,
    stream_ptr: int,
) -> None:
    _cond_ext().cwg_fixed_end(count, counter, continue_flag, stream_ptr)


def statistics_control(
    *, weight_src: torch.Tensor, continue_flag: torch.Tensor,
    counter: torch.Tensor, weight: torch.Tensor, sub_step: torch.Tensor,
    num_sub_steps: torch.Tensor, stream_ptr: int,
) -> None:
    _cond_ext().cwg_stats_control(
        weight_src, continue_flag, counter, weight, sub_step,
        num_sub_steps, stream_ptr,
    )


def fixed_statistics_end(
    *, count: torch.Tensor, counter: torch.Tensor,
    continue_flag: torch.Tensor, weight_src: torch.Tensor,
    weight: torch.Tensor, sub_step: torch.Tensor,
    num_sub_steps: torch.Tensor, stream_ptr: int,
) -> None:
    _cond_ext().cwg_fixed_stats_end(
        count, counter, continue_flag, weight_src, weight, sub_step,
        num_sub_steps, stream_ptr,
    )


class ConditionalWhileGraph:
    """Owns one CUDA conditional-graph ``WHILE`` node and its instantiation.

    The loop body is captured into the node's body graph by :meth:`begin_capture`
    / :meth:`end_capture`; :meth:`set_conditional` appends the generic predicate
    kernel that reads the model's ``continue_flag``.  :meth:`launch` then runs the
    whole loop from a single host launch.
    """

    def __init__(self) -> None:
        self._ext = _cond_ext()
        self._h = self._ext.cwg_create()
        self._instantiated = False

    def begin_capture(self, stream_ptr: int) -> None:
        self._ext.cwg_begin_capture(self._h, stream_ptr)

    def end_capture(self, stream_ptr: int) -> None:
        self._ext.cwg_end_capture(self._h, stream_ptr)

    def set_conditional(self, continue_flag: torch.Tensor, set_cond: bool, stream_ptr: int) -> None:
        """Append the predicate kernel reading ``continue_flag`` (``(1,)`` int32).

        ``set_cond`` must be ``False`` during warmup (outside graph capture, where
        ``cudaGraphSetConditional`` is invalid) and ``True`` when capturing the body.
        """
        self._ext.cwg_set_conditional(self._h, continue_flag, 1 if set_cond else 0, stream_ptr)

    def stats_control(self, *, weight_src: torch.Tensor, continue_flag: torch.Tensor,
                      counter: torch.Tensor, weight: torch.Tensor, sub_step: torch.Tensor,
                      num_sub_steps: torch.Tensor, stream_ptr: int) -> None:
        """Write the aggregator control scalars from the loop state (folded path)."""
        statistics_control(
            weight_src=weight_src, continue_flag=continue_flag,
            counter=counter, weight=weight, sub_step=sub_step,
            num_sub_steps=num_sub_steps, stream_ptr=stream_ptr,
        )

    def instantiate(self) -> None:
        self._ext.cwg_instantiate(self._h)
        self._instantiated = True

    def launch(self, stream_ptr: int) -> None:
        if not self._instantiated:
            raise RuntimeError("ConditionalWhileGraph.instantiate() must run before launch()")
        self._ext.cwg_launch(self._h, stream_ptr)

    def destroy(self) -> None:
        if getattr(self, "_h", None) is not None:
            handle = self._h
            self._h = None
            self._ext.cwg_destroy(handle)

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

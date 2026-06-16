# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
CUDA Graph automation for hydroforge models.

Provides :class:`CUDAGraphMixin` that can be mixed into any
:class:`AbstractModel` subclass to add automatic CUDA Graph
capture and replay.  Downstream models only need to:

1. Inherit from ``CUDAGraphMixin`` (before ``AbstractModel``).
2. Implement ``cuda_graph_target(self, **kwargs)`` — the function
   whose kernel launches will be captured.
3. Call ``self.cuda_graph_replay(cache_key, **kwargs)`` in their
   time-stepping loop instead of calling ``cuda_graph_target`` directly.

Mutable state discovery is **automatic** — every tensor field with
``category`` in ``{init_state, state, shared_state}`` across all opened
modules is saved before warmup and restored after capture.

Example
-------
::

    class MyModel(CUDAGraphMixin, AbstractModel):
        module_list = {...}

        def cuda_graph_target(self, *, runoff, time_sub_step, **kw):
            self.do_physics(runoff, time_sub_step)

        def step(self, runoff, dt, n_sub):
            self.enable_cuda_graph()          # once
            for i in range(n_sub):
                self.cuda_graph_replay(
                    cache_key=n_sub,
                    runoff=runoff,
                    time_sub_step=dt / n_sub,
                )
"""

from __future__ import annotations

import functools
from abc import abstractmethod
from typing import Any, Callable, Dict, Tuple

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
    CWG_CK(cudaGraphAddNode(&cnode, g->graph, nullptr, nullptr, 0, &cp));
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
    from hydroforge.runtime.cuda_kernel import load_inline_cu_module
    return load_inline_cu_module(
        name="hydroforge_conditional_while_graph",
        cpp_sources=_COND_CPP,
        cuda_sources=_COND_CUDA,
        functions=["cwg_create", "cwg_begin_capture", "cwg_end_capture",
                   "cwg_set_conditional", "cwg_stats_control", "cwg_instantiate",
                   "cwg_launch", "cwg_destroy"],
        extra_cuda_cflags=("-O3",),
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
        self._ext.cwg_stats_control(
            weight_src, continue_flag, counter, weight, sub_step, num_sub_steps, stream_ptr)

    def instantiate(self) -> None:
        self._ext.cwg_instantiate(self._h)
        self._instantiated = True

    def launch(self, stream_ptr: int) -> None:
        if not self._instantiated:
            raise RuntimeError("ConditionalWhileGraph.instantiate() must run before launch()")
        self._ext.cwg_launch(self._h, stream_ptr)

    def destroy(self) -> None:
        if getattr(self, "_h", None) is not None:
            self._ext.cwg_destroy(self._h)
            self._h = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


class CUDAGraphMixin:
    """Mixin that adds automatic CUDA Graph capture/replay to an AbstractModel.

    The mixin auto-discovers mutable state tensors from the module field
    metadata (``category ∈ {init_state, state, shared_state}``), so
    downstream models never need to enumerate them manually.
    """

    # ------------------------------------------------------------------ #
    # Private attributes (injected into the Pydantic model via PrivateAttr)
    # ------------------------------------------------------------------ #
    # These are set in enable_cuda_graph() and cleared in disable_cuda_graph().
    # We don't use PrivateAttr here because this is a plain mixin;
    # the attributes are stored on self.__dict__ directly.

    # ------------------------------------------------------------------ #
    # Abstract: the user supplies the function to capture
    # ------------------------------------------------------------------ #
    @abstractmethod
    def cuda_graph_target(self, **kwargs: Any) -> None:
        """The kernel-launch sequence to capture into a CUDA Graph.

        Subclasses must override this.  Typical implementation::

            def cuda_graph_target(self, *, runoff, time_sub_step, **kw):
                self.do_one_sub_step(time_sub_step, runoff, sub_step=0, output_enabled=False)
        """
        ...

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def enable_cuda_graph(self, *, warmup_iters: int = 3) -> None:
        """Enable CUDA Graph capture for subsequent ``cuda_graph_replay`` calls.

        Args:
            warmup_iters: Number of warmup iterations before capture (default 3).
                          Warmup compiles Triton kernels and populates CUDA caches.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph requires a CUDA device")
        self.__dict__["_cg_enabled"] = True
        self.__dict__["_cg_cache"] = {}          # cache_key -> (graph, input_buffers)
        self.__dict__["_cg_pool"] = torch.cuda.graph_pool_handle()
        self.__dict__["_cg_warmup_iters"] = warmup_iters
        self.__dict__["_cg_input_buffers"] = {}  # kwarg_name -> persistent tensor

    def disable_cuda_graph(self) -> None:
        """Disable CUDA Graph and release all cached graphs."""
        self.__dict__["_cg_enabled"] = False
        cache = self.__dict__.get("_cg_cache")
        if cache:
            cache.clear()
        self.__dict__["_cg_pool"] = None
        self.__dict__["_cg_input_buffers"] = {}

    @property
    def cuda_graph_enabled(self) -> bool:
        return self.__dict__.get("_cg_enabled", False)

    # ------------------------------------------------------------------ #
    # Auto-discovery of mutable state
    # ------------------------------------------------------------------ #
    def _cg_discover_mutable_tensors(self) -> Dict[Tuple[str, str], torch.Tensor]:
        """Discover all mutable tensor fields across opened modules.

        Returns a dict mapping ``(module_name, field_name)`` to the live tensor.
        """
        result: Dict[Tuple[str, str], torch.Tensor] = {}

        for module_name in self.opened_modules:  # type: ignore[attr-defined]
            module = self.get_module(module_name)  # type: ignore[attr-defined]
            if module is None:
                continue

            # Regular TensorField fields
            for field_name, field_info in module.__class__.get_model_fields().items():
                extra = getattr(field_info, "json_schema_extra", None) or {}
                cat = extra.get("category", "")
                if cat in _MUTABLE_CATEGORIES:
                    t = getattr(module, field_name, None)
                    if isinstance(t, torch.Tensor):
                        result[(module_name, field_name)] = t

            # computed_tensor_field fields
            for field_name, field_info in module.__class__.get_model_computed_fields().items():
                extra = getattr(field_info, "json_schema_extra", None) or {}
                cat = extra.get("category", "")
                if cat in _MUTABLE_CATEGORIES:
                    t = getattr(module, field_name, None)
                    if isinstance(t, torch.Tensor):
                        result[(module_name, field_name)] = t

        return result

    def _cg_save_state(self) -> Dict[Tuple[str, str], torch.Tensor]:
        """Clone all auto-discovered mutable tensors."""
        return {key: t.clone() for key, t in self._cg_discover_mutable_tensors().items()}

    def _cg_restore_state(self, state: Dict[Tuple[str, str], torch.Tensor]) -> None:
        """Restore model state from cloned tensors."""
        for (module_name, field_name), value in state.items():
            module = self.get_module(module_name)  # type: ignore[attr-defined]
            if module is not None:
                live = getattr(module, field_name, None)
                if live is not None:
                    live.copy_(value)

    # ------------------------------------------------------------------ #
    # Graph capture
    # ------------------------------------------------------------------ #
    def _cg_capture(self, cache_key: Any, **kwargs: Any) -> Tuple["torch.cuda.CUDAGraph", Tuple[str, ...]]:
        """Warmup + capture ``cuda_graph_target`` into a CUDA Graph."""
        warmup_iters = self.__dict__.get("_cg_warmup_iters", 3)
        pool = self.__dict__["_cg_pool"]
        input_buffers = self.__dict__["_cg_input_buffers"]

        # Identify tensor kwargs → create persistent input buffers
        tensor_keys = []
        capture_kwargs = {}
        for name, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                tensor_keys.append(name)
                buf = input_buffers.get(name)
                if buf is None or buf.shape != val.shape or buf.dtype != val.dtype:
                    buf = torch.empty_like(val)
                    input_buffers[name] = buf
                buf.copy_(val)
                capture_kwargs[name] = buf
            else:
                capture_kwargs[name] = val

        tensor_keys_tuple = tuple(tensor_keys)

        # Save mutable state
        saved = self._cg_save_state()

        # Warmup iterations
        for _ in range(warmup_iters):
            self.cuda_graph_target(**capture_kwargs)

        self._cg_restore_state(saved)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            self.cuda_graph_target(**capture_kwargs)

        # Restore after capture
        self._cg_restore_state(saved)

        return (g, tensor_keys_tuple)

    def cuda_graph_replay(self, cache_key: Any, **kwargs: Any) -> None:
        """Execute ``cuda_graph_target`` via CUDA Graph replay.

        On first call for a given *cache_key*, the graph is warmup'd and
        captured.  Subsequent calls with the same key replay the graph
        after copying any changed tensor inputs into persistent buffers.

        Args:
            cache_key: Hashable key for graph caching (e.g. ``num_sub_steps``).
            **kwargs:  Arguments forwarded to ``cuda_graph_target``.
                       Tensor kwargs are auto-buffered; scalars are fixed at
                       capture time (change of scalar → new cache_key needed).
        """
        if not self.__dict__.get("_cg_enabled", False):
            # Fallback: direct call without graph
            self.cuda_graph_target(**kwargs)
            return

        cache = self.__dict__["_cg_cache"]
        entry = cache.get(cache_key)

        if entry is None:
            entry = self._cg_capture(cache_key, **kwargs)
            cache[cache_key] = entry

        graph, tensor_keys = entry

        # Copy fresh tensor inputs into persistent buffers
        input_buffers = self.__dict__["_cg_input_buffers"]
        for name in tensor_keys:
            buf = input_buffers.get(name)
            val = kwargs.get(name)
            if buf is not None and val is not None:
                buf.copy_(val)

        graph.replay()

    # ------------------------------------------------------------------ #
    # Device-side conditional-graph (WHILE node) capture/launch
    # ------------------------------------------------------------------ #
    # These reuse the mutable-state discovery above so a single capture leaves
    # model state pristine after warmup.  A model supplies the per-iteration
    # body (one sub-step ending by writing ``continue_flag``) and a reset hook.
    def enable_conditional_graph(self, *, warmup_iters: int = 3) -> None:
        """Enable device-side conditional-``WHILE`` capture (independent of the
        host-loop CUDA graph cache; can be enabled alongside it)."""
        if not torch.cuda.is_available():
            raise RuntimeError("Conditional graph requires a CUDA device")
        self.__dict__["_condg_warmup_iters"] = warmup_iters

    def build_conditional_graph(
        self,
        *,
        body_fn: Callable[[ConditionalWhileGraph, bool, int], None],
        reset_fn: Callable[[], None],
        continue_flag: torch.Tensor,
        extra_snapshot: "list[torch.Tensor] | None" = None,
    ) -> ConditionalWhileGraph:
        """Warm up and capture ``body_fn`` into a ``WHILE`` graph.

        ``body_fn(graph, set_cond, stream_ptr)`` runs one loop iteration and, as
        its final device action, writes ``continue_flag`` (``1`` keep / ``0``
        stop).  ``graph`` is the graph being built, so the body can append the
        folded statistics kernel before it is stored on the model.  ``reset_fn``
        re-arms per-launch loop state before each warmup pass and the capture.
        ``extra_snapshot`` lists device tensors -- beyond auto-discovered module
        state -- that the body mutates and must be restored after warmup (e.g.
        statistics accumulators).  Returns the instantiated graph.
        """
        device = self.device  # type: ignore[attr-defined]
        warmup = self.__dict__.get("_condg_warmup_iters", 3)
        graph = ConditionalWhileGraph()
        extra = list(extra_snapshot or ())

        cur = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device)
        side.wait_stream(cur)
        saved = self._cg_save_state()
        saved_extra = [t.clone() for t in extra]
        with torch.cuda.stream(side):
            sp = side.cuda_stream
            for _ in range(warmup):
                reset_fn()
                body_fn(graph, False, sp)
                graph.set_conditional(continue_flag, False, sp)
            self._cg_restore_state(saved)
            for live, base in zip(extra, saved_extra):
                live.copy_(base)

            reset_fn()
            pool = torch.cuda.graph_pool_handle()
            torch._C._cuda_beginAllocateToPool(device.index, pool)
            try:
                graph.begin_capture(sp)
                body_fn(graph, True, sp)
                graph.set_conditional(continue_flag, True, sp)
                graph.end_capture(sp)
            finally:
                torch._C._cuda_endAllocateToPool(device.index, pool)
            graph.instantiate()
        cur.wait_stream(side)
        self._cg_restore_state(saved)
        for live, base in zip(extra, saved_extra):
            live.copy_(base)
        return graph

    # ------------------------------------------------------------------ #
    # Folded statistics aggregation (optional helpers)
    # ------------------------------------------------------------------ #
    # The statistics aggregator kernel reads all varying inputs from device
    # scalars, so it can be captured straight into the WHILE body for true
    # per-sub-step (time-weighted) accumulation.  These helpers make that fold
    # reusable by any model whose device loop exposes a sub-step ``counter`` and
    # a ``continue_flag``.

    # Inner reductions the folded path reproduces exactly.  The folded
    # ``stats_control`` kernel encodes only enough of ``(sub_step,
    # num_sub_steps)`` to drive ``is_inner_first`` / ``is_inner_last`` -- it
    # cannot express ``mid`` (true midpoint) or ``arg*`` (needs the real
    # sub-step / macro index), so those force the host loop instead.
    _CONDG_FOLD_INNER_OPS = frozenset({"last", "mean", "sum", "max", "min", "first"})

    @classmethod
    def conditional_stats_device_compatible(cls, aggregator: Any) -> bool:
        """Whether every configured op can run on the device march.

        ``True`` when each op is a folded-compatible inner reduction (optionally
        wrapped in a compound outer op).  ``mid`` and ``arg*`` ops return
        ``False`` so the caller falls back to the host sub-step loop.
        """
        if aggregator is None or not getattr(aggregator, "_aggregator_generated", False):
            return True  # no stats -> nothing to fold, device path is fine
        for ops in aggregator._variable_ops.values():
            for op in ops:
                inner = op.split("_")[-1]
                if inner not in cls._CONDG_FOLD_INNER_OPS:
                    return False
        return True

    @classmethod
    def conditional_stats_should_fold(cls, aggregator: Any) -> bool:
        """Whether the per-sub-step accumulation must run inside the body.

        Time-varying reductions (``mean`` / ``sum`` / ``max`` / ``min`` /
        ``first``, and any compound op built on them) need per-sub-step state, so
        they are folded.  A pure ``last`` (optionally compounded) is exact from a
        single end-of-interval finalize, so it skips the fold (far fewer kernel
        launches).
        """
        if aggregator is None or not getattr(aggregator, "_aggregator_generated", False):
            return False
        for ops in aggregator._variable_ops.values():
            for op in ops:
                inner = op.split("_")[-1]
                if inner in cls._CONDG_FOLD_INNER_OPS and inner != "last":
                    return True
        return False

    @staticmethod
    def conditional_stats_accumulators(aggregator: Any) -> "list[torch.Tensor]":
        """Aggregator accumulator tensors to pass as ``extra_snapshot`` so the
        warmup passes do not corrupt them."""
        states = aggregator._kernel_states
        return [v for k, v in states.items()
                if isinstance(v, torch.Tensor) and not k.startswith("__")]

    def conditional_stats_body(
        self,
        aggregator: Any,
        *,
        graph: ConditionalWhileGraph,
        weight_src: torch.Tensor,
        counter: torch.Tensor,
        continue_flag: torch.Tensor,
        BLOCK_SIZE: int,
        stream_ptr: int,
    ) -> None:
        """Run one folded aggregator update inside the captured body.

        ``weight_src`` is the per-sub-step weight (e.g. the device ``dt`` tensor);
        ``counter`` (1-based) and ``continue_flag`` come from the model's device
        loop.  Call this after the physics sub-step, before the predicate kernel.
        """
        states = aggregator._kernel_states
        graph.stats_control(
            weight_src=weight_src, continue_flag=continue_flag, counter=counter,
            weight=states["__weight"], sub_step=states["__sub_step"],
            num_sub_steps=states["__num_sub_steps"], stream_ptr=stream_ptr)
        aggregator._aggregator_function(states, BLOCK_SIZE)

    @staticmethod
    def conditional_stats_prelaunch(aggregator: Any, flags: int, total_weight: float) -> None:
        """Per-interval host bookkeeping for the folded path.

        The per-sub-step accumulation runs on-device inside the body; the host
        still updates the macro-step / dirty-output bookkeeping once per interval
        and sets the host-constant control scalars (mirrors the standard
        ``update_statistics`` flag handling)."""
        states = aggregator._kernel_states
        is_inner_last = bool(flags & 2)
        is_outer_first = bool(flags & 4) and is_inner_last
        is_outer_last = bool(flags & 8) and is_inner_last
        if is_outer_first:
            aggregator._macro_step_index = 0
            aggregator._current_macro_step_count = 0.0
            aggregator._outer_flags_ever_seen = True
        if is_inner_last or is_outer_last:
            for out_name, out_is_outer in aggregator._output_is_outer.items():
                if (not out_is_outer and is_inner_last) or (out_is_outer and is_outer_last):
                    aggregator._dirty_outputs.add(out_name)
        if is_inner_last:
            aggregator._current_macro_step_count += 1.0
        states["__total_weight"].fill_(total_weight)
        states["__flags"].fill_(flags)
        states["__num_macro_steps"].fill_(aggregator._current_macro_step_count)
        states["__macro_step_index"].fill_(aggregator._macro_step_index)

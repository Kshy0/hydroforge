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

Mutable state is compiled automatically from registered-kernel access metadata
and ATen alias schemas. Those exact write sets let capture warmup roll back all
declared state before the live launch.

"""

from __future__ import annotations

import functools
import importlib.util
from typing import Any

import torch

from hydroforge.kernels.backends.cuda import rtc

# Categories whose tensors are mutated during a physics step
_MUTABLE_CATEGORIES = frozenset({"init_state", "state", "shared_state"})


def _cuda_version_tuple(version: object) -> tuple[int, int] | None:
    """Parse the CUDA toolkit version exposed by the active Torch build."""

    if not isinstance(version, str):
        return None
    parts = version.split(".", 2)
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def supports_conditional_cuda_graph(
    device: torch.device | str | None = None,
) -> bool:
    """Return whether HydroForge's CUDA conditional-WHILE support is available.

    PyTorch exposes AMD/ROCm devices through the ``cuda`` device type, but
    HIP does not provide conditional graph nodes. The CUDA API is available
    from CUDA 12.4 onward; the host side goes through ``cuda.bindings``.
    """

    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    cuda_version = _cuda_version_tuple(getattr(torch.version, "cuda", None))
    if cuda_version is None or cuda_version < (12, 4):
        return False
    try:
        return importlib.util.find_spec("cuda.bindings") is not None
    except ModuleNotFoundError:
        return False


# ====================================================================== #
# Device-side loop control
# ====================================================================== #
# A CUDA conditional-graph WHILE node folds a variable-length sub-step loop into
# one graph launch: the body and its continuation predicate run on-device, so
# the host issues one launch per interval with zero per-iteration sync.  The
# counter/statistics kernels below are plain device code and also serve fixed
# CUDA/HIP graphs; only the predicate kernel needs conditional-node support.

_CONTROL_SOURCE = r"""
__global__ void k_fixed_end(const int* __restrict__ count,
        int* __restrict__ counter, int* __restrict__ cont) {
    int next = *counter + 1;
    *counter = next;
    *cont = next < *count;
}

template <typename SourceT, typename DestinationT>
__device__ void write_statistics_control(bool first, bool last,
        const SourceT* weight_src, DestinationT* weight,
        int* sub_step, int* num_sub_steps) {
    int ss, n;
    if (first && last) { ss = 0; n = 1; }
    else if (first)    { ss = 0; n = 2; }
    else if (last)     { ss = 1; n = 2; }
    else               { ss = 1; n = 3; }
    *weight = static_cast<DestinationT>(*weight_src);
    *sub_step = ss;
    *num_sub_steps = n;
}

template <typename SourceT, typename DestinationT>
__global__ void k_fixed_stats_end(const int* __restrict__ count,
        int* __restrict__ counter, int* __restrict__ cont,
        const SourceT* __restrict__ weight_src,
        DestinationT* __restrict__ weight,
        int* __restrict__ sub_step, int* __restrict__ num_sub_steps) {
    int next = *counter + 1;
    bool first = next == 1;
    bool last = next == *count;
    *counter = next;
    *cont = !last;
    write_statistics_control(first, last, weight_src, weight, sub_step, num_sub_steps);
}

// Statistics-control bridge for the folded aggregator path.  From the 1-based
// sub-step counter, the continue_flag (0 on the final sub-step) and the
// per-sub-step weight (e.g. dt), writes the aggregator's __weight / __sub_step /
// __num_sub_steps so its is_inner_first (sub_step == 0) and is_inner_last
// (sub_step == num_sub_steps - 1) fire without knowing the total count ahead:
//   first & last -> (0, 1);  first -> (0, 2);  last -> (1, 2);
//   interior -> (1, 3).
// Exact for supported inner ops {last, mean, sum, max, min, first}; not arg*.
template <typename SourceT, typename DestinationT>
__global__ void k_stats_control(const SourceT* __restrict__ weight_src,
        const int* __restrict__ cont, const int* __restrict__ counter,
        DestinationT* __restrict__ weight, int* __restrict__ sub_step,
        int* __restrict__ num_sub_steps) {
    write_statistics_control(*counter == 1, *cont == 0,
                             weight_src, weight, sub_step, num_sub_steps);
}
"""

# Generic continuation predicate: read the model's (1,) int "continue?" flag and
# feed it to cudaGraphSetConditional.  ``set_cond`` is false during warmup, where
# the call is illegal outside conditional execution, so the kernel still loads.
_CONDITIONAL_SOURCE = r"""
#include <cuda_device_runtime_api.h>

__global__ void k_set_conditional(cudaGraphConditionalHandle handle,
                                  const int* __restrict__ cont, int set_cond) {
    if (set_cond) cudaGraphSetConditional(handle, (*cont) ? 1u : 0u);
}
"""


@functools.cache
def _control_program() -> rtc.RtcProgram:
    return rtc.RtcProgram(_CONTROL_SOURCE, (), "hydroforge_loop_control")


@functools.cache
def _conditional_program() -> rtc.RtcProgram:
    return rtc.RtcProgram(
        _CONDITIONAL_SOURCE,
        rtc.toolkit_include_options(),
        "hydroforge_conditional_while_graph",
    )


@functools.lru_cache(maxsize=256)
def _launcher(
    program: rtc.RtcProgram,
    kernel: str,
    args: tuple[rtc.KernelArgument, ...],
    device: int,
):
    step = rtc.CudaLaunch(kernel, 1, 1, args)
    return rtc.prepare(rtc.RtcRequest(program, (kernel,)), (step,), device)


def _launch(
    program: rtc.RtcProgram,
    kernel: str,
    args: tuple[rtc.KernelArgument, ...],
    device: torch.device,
    stream_ptr: int,
) -> None:
    index = torch.cuda.current_device() if device.index is None else device.index
    _launcher(program, kernel, args, index)(stream_ptr)


def _int32(tensor: torch.Tensor) -> rtc.KernelArgument:
    if tensor.dtype != torch.int32:
        raise TypeError(f"loop control tensors must be int32, got {tensor.dtype}")
    return rtc.pointer(tensor)


def _real(tensor: torch.Tensor) -> str:
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"statistics weights must be float32/float64, got {tensor.dtype}"
        )
    return rtc.ctype(tensor)


def fixed_control_end(
    count: torch.Tensor,
    counter: torch.Tensor,
    continue_flag: torch.Tensor,
    stream_ptr: int,
) -> None:
    _launch(
        _control_program(),
        "k_fixed_end",
        (_int32(count), _int32(counter), _int32(continue_flag)),
        counter.device,
        stream_ptr,
    )


def statistics_control(
    *,
    weight_src: torch.Tensor,
    continue_flag: torch.Tensor,
    counter: torch.Tensor,
    weight: torch.Tensor,
    sub_step: torch.Tensor,
    num_sub_steps: torch.Tensor,
    stream_ptr: int,
) -> None:
    _launch(
        _control_program(),
        f"k_stats_control<{_real(weight_src)}, {_real(weight)}>",
        (
            rtc.pointer(weight_src),
            _int32(continue_flag),
            _int32(counter),
            rtc.pointer(weight),
            _int32(sub_step),
            _int32(num_sub_steps),
        ),
        counter.device,
        stream_ptr,
    )


def fixed_statistics_end(
    *,
    count: torch.Tensor,
    counter: torch.Tensor,
    continue_flag: torch.Tensor,
    weight_src: torch.Tensor,
    weight: torch.Tensor,
    sub_step: torch.Tensor,
    num_sub_steps: torch.Tensor,
    stream_ptr: int,
) -> None:
    _launch(
        _control_program(),
        f"k_fixed_stats_end<{_real(weight_src)}, {_real(weight)}>",
        (
            _int32(count),
            _int32(counter),
            _int32(continue_flag),
            rtc.pointer(weight_src),
            rtc.pointer(weight),
            _int32(sub_step),
            _int32(num_sub_steps),
        ),
        counter.device,
        stream_ptr,
    )


# ====================================================================== #
# Host-side conditional graph (driver API)
# ====================================================================== #


def _driver() -> Any:
    from cuda.bindings import driver

    return driver


def _check(result: Any, action: str) -> Any:
    """Unpack a ``cuda.bindings`` result tuple, raising on a driver error."""

    driver = _driver()
    error, *values = result if isinstance(result, tuple) else (result,)
    if error != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(error)
        raise RuntimeError(
            f"CUDA {action} failed: {name.decode() if name else int(error)}"
        )
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class ConditionalWhileGraph:
    """Owns one CUDA conditional-graph ``WHILE`` node and its instantiation.

    The loop body is captured into the node's body graph by :meth:`begin_capture`
    / :meth:`end_capture`; :meth:`set_conditional` appends the generic predicate
    kernel that reads the model's ``continue_flag``.  :meth:`launch` then runs the
    whole loop from a single host launch.
    """

    def __init__(self) -> None:
        driver = _driver()
        self._device = torch.cuda.current_device()
        self._h = None
        self._exec = None
        # The primary context is the one PyTorch launches into; retaining it
        # keeps the conditional handle's context alive for this graph's life.
        self._cu_device = _check(driver.cuDeviceGet(self._device), "device lookup")
        self._context = _check(
            driver.cuDevicePrimaryCtxRetain(self._cu_device), "primary context retain"
        )
        try:
            self._h = _check(driver.cuGraphCreate(0), "graph creation")
            # Default value 1 makes the body run at least once per launch
            # (the first sub-step always executes).
            self._handle = _check(
                driver.cuGraphConditionalHandleCreate(
                    self._h,
                    self._context,
                    1,
                    driver.CU_GRAPH_COND_ASSIGN_DEFAULT,
                ),
                "conditional handle creation",
            )
            params = driver.CUgraphNodeParams()
            params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
            params.conditional.handle = self._handle
            params.conditional.type = (
                driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
            )
            params.conditional.size = 1
            params.conditional.ctx = self._context
            add_node = getattr(driver, "cuGraphAddNode_v2", driver.cuGraphAddNode)
            _check(
                add_node(self._h, None, None, 0, params),
                "conditional node creation",
            )
            self._body = params.conditional.phGraph_out[0]
        except BaseException:
            self.destroy()
            raise

    def begin_capture(self, stream_ptr: int) -> None:
        driver = _driver()
        with torch.cuda.device(self._device):
            _check(
                driver.cuStreamBeginCapture(
                    stream_ptr,
                    driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                ),
                "stream capture begin",
            )

    def end_capture(self, stream_ptr: int) -> None:
        driver = _driver()
        with torch.cuda.device(self._device):
            captured = _check(
                driver.cuStreamEndCapture(stream_ptr), "stream capture end"
            )
            try:
                _check(
                    driver.cuGraphAddChildGraphNode(self._body, None, 0, captured),
                    "loop body insertion",
                )
            finally:
                _check(driver.cuGraphDestroy(captured), "captured graph destruction")

    def set_conditional(
        self, continue_flag: torch.Tensor, set_cond: bool, stream_ptr: int
    ) -> None:
        """Append the predicate kernel reading ``continue_flag`` (``(1,)`` int32).

        ``set_cond`` must be ``False`` during warmup (outside graph capture, where
        ``cudaGraphSetConditional`` is invalid) and ``True`` when capturing the body.
        """
        _launch(
            _conditional_program(),
            "k_set_conditional",
            (
                rtc.uint64(int(self._handle)),
                _int32(continue_flag),
                rtc.int32(1 if set_cond else 0),
            ),
            torch.device("cuda", self._device),
            stream_ptr,
        )

    def stats_control(
        self,
        *,
        weight_src: torch.Tensor,
        continue_flag: torch.Tensor,
        counter: torch.Tensor,
        weight: torch.Tensor,
        sub_step: torch.Tensor,
        num_sub_steps: torch.Tensor,
        stream_ptr: int,
    ) -> None:
        """Write the aggregator control scalars from the loop state (folded path)."""
        with torch.cuda.device(self._device):
            statistics_control(
                weight_src=weight_src,
                continue_flag=continue_flag,
                counter=counter,
                weight=weight,
                sub_step=sub_step,
                num_sub_steps=num_sub_steps,
                stream_ptr=stream_ptr,
            )

    def instantiate(self) -> None:
        driver = _driver()
        with torch.cuda.device(self._device):
            self._exec = _check(
                driver.cuGraphInstantiate(self._h, 0), "graph instantiation"
            )

    def launch(self, stream_ptr: int) -> None:
        _check(_driver().cuGraphLaunch(self._exec, stream_ptr), "graph launch")

    def destroy(self) -> None:
        context = getattr(self, "_context", None)
        if context is None:
            return
        driver = _driver()
        executable, graph = self._exec, self._h
        self._exec = self._h = self._context = None
        try:
            if executable is not None:
                _check(driver.cuGraphExecDestroy(executable), "graph exec destruction")
            if graph is not None:
                _check(driver.cuGraphDestroy(graph), "graph destruction")
        finally:
            _check(
                driver.cuDevicePrimaryCtxRelease(self._cu_device),
                "primary context release",
            )

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

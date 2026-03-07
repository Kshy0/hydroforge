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

from abc import abstractmethod
from typing import Any, Dict, Tuple

import torch

# Categories whose tensors are mutated during a physics step
_MUTABLE_CATEGORIES = frozenset({"init_state", "state", "shared_state"})


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

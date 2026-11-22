# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Weighted exchange of a field between two spatial domains.

A link table lists ``(target ID, source ID, weight)`` in a fixed global order
and is held complete on every rank. An ID names one *point* of a port's ID
dimensions (:attr:`~hydroforge.contracts.coupling.Port.dims`):

- a slab field: the row-major index over the dimensions' global extents,
  the slab axis counted in global rows (by default every dimension of the
  field: ``(lat, lon)`` cells);
- a partition field: the coordinate ID of its key dimension;
- located entities (a ``grid_index`` dimension): their place in the
  complete table.

The field's other dimensions, and an ensemble member axis, are *carried*:
each point moves with all of them, and the source and target carry the same
extents in the same order (an ``(ch, lat, lon)`` ocean field and a
``(ch, cell)`` land field both carry ``ch``).

Each rank owns some source and target points. Its exchange stages the
source points it owns that any link needs, then, as ghost rows, the ones its
links need from other ranks; those arrive point to point over the halo links
of a plan of their own (:mod:`hydroforge.parallel.halo`, ``key`` kind), bit
for bit. The rank then adds the links into the target points it owns in
table order, so the result does not depend on the number of ranks. Device
memory and messages scale with the points a rank owns and receives, not
with the table; the host holds the table and builds the plan once.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

import hydroforge.parallel
from hydroforge.compiler.partition import bare
from hydroforge.contracts.slab import INTERIOR
from hydroforge.core.errors import SubstepCompileError
from hydroforge.execution.context import managed_step_active
from hydroforge.execution.session import ModelRuntime
from hydroforge.kernels.calls import recording_sink
from hydroforge.kernels.cuda import CudaKernel, CudaSource
from hydroforge.kernels.registry import BackendRegistry
from hydroforge.kernels.spec import KernelSpec, constant
from hydroforge.parallel.halo import MAX_FIELDS, FieldLayout, occurrences

_LAYOUT = FieldLayout("key", 0)
_GATHER_BYTES = 1 << 21  # per rank and round of a plan's host-table gather


@cache
def _csr_registry() -> BackendRegistry:
    source = CudaSource(
        path=Path(hydroforge.parallel.__file__).resolve().parent / "remap.cu"
    )
    spec = KernelSpec(
        name="remap_csr_gather",
        size=("channels", "num_targets"),
        parameters={
            "staging": "read",
            "weights": "read",
            "csr_rows": "read",
            "csr_ptr": "read",
            "csr_slots": "read",
            "csr_entries": "read",
            "target": "read_write",
            "channels": "index",
            "num_targets": "index",
            "outer": "index",
            "n_tgt": "index",
            "inner": "index",
            "n_stage": "index",
            "use_entries": constant("int32"),
        },
    )
    return BackendRegistry(
        spec,
        {"cuda": CudaKernel(source, kernel="remap_csr_gather<{staging},{weights},{target}>")},
    )


@dataclass(slots=True)
class _Side:
    """One port's field: ``view`` holds it as (outer, points, inner) around
    its ID dimensions; ``rows`` are the stored points this rank owns and
    ``ids`` their global IDs; ``carried`` the extents of every other axis."""

    tensor: torch.Tensor
    view: torch.Tensor
    rows: np.ndarray
    ids: np.ndarray
    carried: tuple[int, ...]


def _tokens(shape: tuple[Any, ...]) -> list[str]:
    return [bare(item) if isinstance(item, str) else str(item) for item in shape]


def _side(runtime: Any, qualified: str, dims: tuple[str, ...] | None) -> _Side:
    plan = runtime.plan
    entry = plan.fields.entries.get(qualified)
    if entry is None or not entry.active:
        raise ValueError(f"exchange field {qualified!r} is not an active field")
    module, name = runtime.modules[entry.module], entry.name
    tensor = getattr(module, name)
    tokens = _tokens(tuple(entry.tensor.shape))
    lead = int(module.is_batched(name))
    if tensor.ndim != len(tokens) + lead or not tensor.is_contiguous():
        raise ValueError(
            f"exchange field {qualified!r} must be a contiguous tensor of its "
            f"declared shape {tuple(tokens)}"
        )
    info = plan.fields.slab.get(qualified)
    coordinate = bare(entry.tensor.dim_coords) if entry.tensor.dim_coords else ""
    located = [token for token in tokens if token in plan.fields.located]
    owner = None
    if info is not None:
        if info.kind == INTERIOR:
            raise ValueError(f"exchange field {qualified!r} is on an interior axis")
        key, default = tokens[info.position], tuple(tokens)
    elif coordinate:
        owner = next(
            item
            for item in plan.fields.entries.values()
            if item.active and item.name == coordinate
        )
        key = _tokens(tuple(owner.tensor.shape))[0]
        default = (key,)
    elif len(located) == 1:
        key = located[0]
        default = (key,)
    else:
        raise ValueError(f"exchange field {qualified!r} belongs to no spatial domain")
    dims = default if dims is None else tuple(dims)
    if info is None and dims != (key,):
        raise ValueError(
            f"exchange field {qualified!r}: the ID dimension of a partition or "
            f"located field is its key dimension {key!r}, not {dims}"
        )
    if not dims or len(set(dims)) != len(dims) or not set(dims) <= set(tokens):
        raise ValueError(
            f"exchange field {qualified!r}: ID dimensions {dims} must be distinct "
            f"dimensions of its shape {tuple(tokens)}"
        )
    first = tokens.index(dims[0])
    if [tokens.index(d) for d in dims] != list(range(first, first + len(dims))):
        raise ValueError(
            f"exchange field {qualified!r}: ID dimensions {dims} must be adjacent "
            f"and in the order of its shape {tuple(tokens)}"
        )
    if key not in dims:
        raise ValueError(
            f"exchange field {qualified!r}: ID dimensions {dims} must include its "
            f"decomposed dimension {key!r}"
        )
    shape = tuple(int(n) for n in tensor.shape)
    start, stop = lead + first, lead + first + len(dims)
    stored = shape[start:stop]
    outer = int(np.prod(shape[:start], dtype=np.int64))
    inner = int(np.prod(shape[stop:], dtype=np.int64))
    view = tensor.view(outer, int(np.prod(stored, dtype=np.int64)), inner)
    carried = shape[:start] + shape[stop:]
    if info is not None:
        slab = runtime.slabs[info.domain]
        axis = dims.index(key)
        owned = slab.owned_slice(info.kind)
        global_rows = np.asarray(slab.global_rows(info.kind), dtype=np.int64)
        local = [np.arange(n, dtype=np.int64) for n in stored]
        local[axis] = owned.start + np.arange(global_rows.size, dtype=np.int64)
        wide = list(local)
        wide[axis] = global_rows
        extents = list(stored)
        extents[axis] = slab.total_of(info.kind)
        rows = np.ravel_multi_index(np.meshgrid(*local, indexing="ij"), stored)
        ids = np.ravel_multi_index(np.meshgrid(*wide, indexing="ij"), extents)
        return _Side(
            tensor, view, rows.reshape(-1).astype(np.int64),
            ids.reshape(-1).astype(np.int64), carried,
        )
    if owner is not None:
        group = plan.fields.variable_groups.get(name)
        if group is None:
            raise ValueError(f"exchange field {qualified!r} belongs to no spatial domain")
        count = runtime.partition.rank_indices(group).size
        ids = getattr(runtime.modules[owner.module], coordinate)[:count]
        ids = ids.cpu().numpy().astype(np.int64)
    else:
        entities = runtime.located.get(key)
        count = stored[0]
        ids = (
            np.arange(count, dtype=np.int64)
            if entities is None
            else np.asarray(entities.positions, dtype=np.int64)
        )
        if ids.size != count:
            raise ValueError(
                f"exchange field {qualified!r} holds {count} entities of {key!r}; "
                f"this rank owns {ids.size}"
            )
    return _Side(tensor, view, np.arange(count, dtype=np.int64), ids, carried)


def _positions(table: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Where each of ``values`` sits in the sorted ``table`` (-1: absent)."""

    at = np.searchsorted(table, values)
    hit = at < table.size
    hit[hit] = table[at[hit]] == values[hit]
    return np.where(hit, at, -1)


def _each_once(ids: np.ndarray, owned: list[np.ndarray], label: str) -> None:
    """Raise alike on every rank unless exactly one rank owns each table ID."""

    counts = np.zeros(ids.size, np.int64)
    for positions in owned:
        np.add.at(counts, positions, 1)
    wrong = np.flatnonzero(counts != 1)
    if wrong.size:
        raise ValueError(
            f"exchange {label} IDs {ids[wrong[:5]].tolist()} are not owned by "
            "exactly one rank"
        )


def _gathered(runtime: Any, value: Any) -> list[Any]:
    """Every rank's ``value``, gathered in pieces: over NCCL an object
    collective holds all ranks' payloads in device memory at once."""

    import torch.distributed as dist

    world = runtime.plan.spatial_world_size
    if world == 1:
        return [value]
    group = runtime.spatial_group()
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sizes: list[Any] = [None] * world
    dist.all_gather_object(sizes, len(payload), group=group)
    parts = [bytearray(size) for size in sizes]
    for start in range(0, max(sizes), _GATHER_BYTES):
        found: list[Any] = [None] * world
        dist.all_gather_object(
            found, payload[start : start + _GATHER_BYTES], group=group
        )
        for part, piece in zip(parts, found, strict=True):
            part[start : start + len(piece)] = piece
    return [pickle.loads(part) for part in parts]


@dataclass(slots=True)
class _Exchange:
    source: torch.Tensor
    target: torch.Tensor
    target_ids: torch.Tensor
    source_ids: torch.Tensor
    weights: torch.Tensor
    sides: tuple[_Side, _Side]
    send_rows: torch.Tensor  # source points this rank owns and a link needs
    staging: torch.Tensor  # (carried, owned then received source points)
    linked: torch.Tensor  # target points this rank owns that links fill
    # Per pass (non-CUDA fallback): target points (each once), staging rows, table entries.
    passes: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    pass_entries: tuple[np.ndarray, ...]
    halo: Any = None
    launch: Any = None
    # CUDA CSR fast path: one kernel launch regardless of target fan-in.
    csr_binder: Any = None
    csr_pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    csr_entries_host: np.ndarray | None = None
    csr_launch: Any = None
    # Host weights: device buffer(s) a captured graph reads.
    weighted: tuple[torch.Tensor, ...] | None = None
    graph: Any = None
    owner: Any = None  # the executor that holds ``graph``
    stats: dict[str, int] = field(default_factory=dict)

    def _bind_csr(self, weight_tensor: torch.Tensor, use_entries: int) -> Any:
        assert self.csr_pack is not None and self.csr_binder is not None
        csr_rows, csr_ptr, csr_slots, csr_entries = self.csr_pack
        view = self.sides[1].view
        outer, n_tgt, inner = int(view.shape[0]), int(view.shape[1]), int(view.shape[2])
        args = dict(
            staging=self.staging,
            weights=weight_tensor,
            csr_rows=csr_rows,
            csr_ptr=csr_ptr,
            csr_slots=csr_slots,
            csr_entries=csr_entries,
            target=self.target,
            channels=outer * inner,
            num_targets=int(csr_rows.numel()),
            outer=outer,
            n_tgt=n_tgt,
            inner=inner,
            n_stage=int(self.staging.shape[1]),
            use_entries=use_entries,
        )
        return self.csr_binder.bind(_csr_registry(), args).compile()

    def reweigh(self, weights: torch.Tensor) -> None:
        """Take a new weight table of the same links (a local re-slice)."""

        if weights.shape != self.target_ids.shape:
            raise ValueError(
                f"exchange weights of shape {tuple(weights.shape)} do not match "
                f"the link table of shape {tuple(self.target_ids.shape)}"
            )
        device, dtype = self.source.device, self.source.dtype
        if self.csr_binder is not None:
            if self.csr_pack is None:
                self.weights = weights
                return
            if weights.device == device:
                if (
                    self.weighted is not None
                    or weights is not self.weights
                    or self.csr_launch is None
                ):
                    self.drop_graph()
                    self.weights, self.weighted = weights, None
                    self.csr_launch = self._bind_csr(weights, use_entries=1)
                return
            host = weights.detach().cpu().numpy()
            assert self.csr_entries_host is not None
            values = torch.as_tensor(host[self.csr_entries_host])
            if self.weighted is None:
                self.drop_graph()
                buf = values.to(device=device, dtype=dtype)
                self.weighted = (buf,)
                self.csr_launch = self._bind_csr(buf, use_entries=0)
            else:
                self.weighted[0].copy_(values)
            self.weights = weights
            return

        if weights.device == device:
            if self.weighted is not None or weights is not self.weights:
                self.drop_graph()
            self.weights, self.weighted = weights, None
            return
        host = weights.detach().cpu().numpy()
        slices = [torch.as_tensor(host[idx]) for idx in self.pass_entries]
        if self.weighted is None:
            self.drop_graph()
            self.weighted = tuple(
                values.to(device=device, dtype=dtype) for values in slices
            )
        else:
            for buffer, values in zip(self.weighted, slices, strict=True):
                buffer.copy_(values)
        self.weights = weights

    def drop_graph(self) -> None:
        """Release the captured graph, unless its executor already has."""

        graph, owner = self.graph, self.owner
        self.graph, self.owner = None, None
        if graph is not None and owner.owns(graph):
            owner.release(graph)

    def body(self) -> None:
        source, target = self.sides
        staging = self.staging
        count = self.send_rows.numel()
        if count:
            picked = source.view.index_select(1, self.send_rows)
            staging[:, :count].copy_(
                picked.permute(0, 2, 1).reshape(staging.shape[0], count)
            )
        if self.launch is not None:
            self.launch()
        if self.csr_binder is not None:
            if self.csr_launch is not None:
                self.csr_launch()
            return
        view = target.view
        view.index_fill_(1, self.linked, 0)
        outer, inner = view.shape[0], view.shape[2]
        dtype = self.source.dtype
        for turn, (rows, slots, entries) in enumerate(self.passes):
            weights = (
                self.weighted[turn]
                if self.weighted is not None
                else self.weights.index_select(0, entries).to(dtype)
            )
            values = (weights * staging.index_select(1, slots)).to(target.tensor.dtype)
            view.index_add_(1, rows, values.view(outer, inner, -1).permute(0, 2, 1))

    def run(self, executor: Any) -> None:
        capture = getattr(executor, "capture", None)
        if (
            capture is not None
            and self.source.device.type == "cuda"
            and (self.halo is None or self.halo.capturable)
        ):
            if self.graph is not None and (
                self.owner is not executor or not executor.owns(self.graph)
            ):
                self.drop_graph()  # the executor invalidated its captures
            if self.graph is None:
                # Every rank makes the same calls, so every rank captures
                # (its warm-up exchanges included) at the same call.
                self.graph = capture(
                    self.body, mutated_state=(self.target, self.staging)
                )
                self.owner = executor
            self.graph.replay()
            return
        self.body()

    def close(self) -> None:
        halo, self.halo, self.launch = self.halo, None, None
        try:
            self.drop_graph()
        finally:
            if halo is not None:
                halo.close()


def _build(
    runtime: Any,
    source_name: str,
    target_name: str,
    dims: tuple[tuple[str, ...] | None, tuple[str, ...] | None],
    target_ids: torch.Tensor,
    source_ids: torch.Tensor,
    weights: torch.Tensor,
) -> _Exchange:
    from hydroforge.kernels.halo import attach
    from hydroforge.parallel.halo import HaloPlan, HaloRuntime, RowLayout

    source = _side(runtime, source_name, dims[0])
    target = _side(runtime, target_name, dims[1])
    if source.carried != target.carried:
        raise ValueError(
            f"exchange {source_name!r} -> {target_name!r}: the source carries "
            f"axes of extents {source.carried}, the target {target.carried}"
        )
    if not target_ids.shape == source_ids.shape == weights.shape or target_ids.ndim != 1:
        raise ValueError("exchange link tables must be one-dimensional and equal in length")
    device = source.tensor.device
    to_ids = target_ids.detach().cpu().numpy().astype(np.int64)
    from_ids = source_ids.detach().cpu().numpy().astype(np.int64)
    needed, slot = np.unique(from_ids, return_inverse=True)
    slot = slot.reshape(-1)
    linked_ids = np.unique(to_ids)
    # The source points this rank owns that a link needs, in needed order.
    have = _positions(needed, source.ids)
    holding = np.flatnonzero(have >= 0)
    holding = holding[np.argsort(have[holding], kind="stable")]
    owned_needed = have[holding]
    send = source.rows[holding]
    mine = _positions(linked_ids, target.ids)
    # This rank's links, in table order: those into target points it owns.
    order = np.argsort(target.ids, kind="stable")
    place = np.searchsorted(target.ids[order], to_ids)
    take = place < target.ids.size
    take[take] = target.ids[order][place[take]] == to_ids[take]
    entries = np.flatnonzero(take)
    rows = target.rows[order][place[take]]
    ghosts = np.setdiff1d(np.unique(slot[entries]), owned_needed)
    everyone = _gathered(runtime, (owned_needed, mine[mine >= 0], ghosts))
    _each_once(needed, [item[0] for item in everyone], "source")
    _each_once(linked_ids, [item[1] for item in everyone], "target")
    # Staging rows: owned needed points, then received ones.
    stored = np.concatenate((owned_needed, ghosts)).astype(np.int64)
    staging_row = np.full(needed.size, -1, np.int64)
    staging_row[stored] = np.arange(stored.size, dtype=np.int64)
    slots = staging_row[slot[entries]]
    as_device = lambda value: torch.as_tensor(value, dtype=torch.int64, device=device)
    channels = int(np.prod(source.carried, dtype=np.int64))
    staging = torch.zeros(
        (channels, stored.size), dtype=source.tensor.dtype, device=device
    )
    halo = launch = None
    if runtime.plan.spatial_world_size > 1 and any(item[2].size for item in everyone):
        backend = runtime.plan.backend.name
        if backend not in ("cuda", "torch"):
            raise ValueError(
                f"exchange_ across ranks is not supported on backend {backend!r} "
                "(use HYDROFORGE_BACKEND=cuda)"
            )
        layouts = [
            RowLayout(
                np.concatenate((item[0], item[2])).astype(np.int64),
                np.arange(item[0].size + item[2].size) < item[0].size,
            )
            for item in everyone
        ]
        # One field per message: a slot of MAX_FIELDS rows of this width
        # holds every row of it.
        width = -(-channels * source.tensor.element_size() // MAX_FIELDS)
        halo = attach(
            HaloRuntime(
                HaloPlan(runtime.plan.spatial_rank, {"key": layouts}),
                device,
                backend,
                runtime.spatial_group(),
                {"key": max(width, 1)},
            )
        )
        try:
            arguments = halo.exchange_arguments([staging], [_LAYOUT], "mirror")
            if arguments is not None:
                launch = runtime.execution.kernel_binding.bind(
                    halo.registry, arguments
                ).compile()
        except BaseException:
            halo.close()
            raise

    use_csr = device.type == "cuda" and runtime.plan.backend.name == "cuda"
    csr_binder = runtime.execution.kernel_binding if use_csr else None
    csr_pack = None
    csr_entries_host = None
    passes: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] = ()
    pass_entries: tuple[np.ndarray, ...] = ()
    if use_csr:
        if rows.size:
            csr_order = np.argsort(rows, kind="stable")
            sorted_rows = rows[csr_order]
            csr_rows_np, counts = np.unique(sorted_rows, return_counts=True)
            csr_ptr_np = np.empty(counts.size + 1, dtype=np.int64)
            csr_ptr_np[0] = 0
            np.cumsum(counts, out=csr_ptr_np[1:])
            csr_entries_host = entries[csr_order]
            csr_pack = (
                as_device(csr_rows_np),
                as_device(csr_ptr_np),
                as_device(slots[csr_order]),
                as_device(csr_entries_host),
            )
        else:
            csr_entries_host = np.zeros(0, np.int64)
        linked_d = csr_pack[0] if csr_pack is not None else as_device(np.zeros(0, np.int64))
    else:
        turns = occurrences(rows) if rows.size else np.zeros(0, np.int64)
        if turns.size:
            t_order = np.argsort(turns, kind="stable")
            counts = np.bincount(turns)
            splits = np.cumsum(counts)[:-1]
            groups = np.split(t_order, splits)
            passes = tuple(
                (
                    as_device(rows[g]),
                    as_device(slots[g]),
                    as_device(entries[g]),
                )
                for g in groups
            )
            pass_entries = tuple(entries[g] for g in groups)
        linked_d = as_device(np.unique(rows))

    exchange = _Exchange(
        source.tensor,
        target.tensor,
        target_ids,
        source_ids,
        weights,
        (source, target),
        as_device(send),
        staging,
        linked_d,
        passes,
        pass_entries,
        halo,
        launch,
        csr_binder=csr_binder,
        csr_pack=csr_pack,
        csr_entries_host=csr_entries_host,
        stats=dict(
            links=int(entries.size),
            owned=int(owned_needed.size),
            received=int(ghosts.size),
            staging_bytes=int(staging.numel() * staging.element_size()),
            window_bytes=0 if halo is None else int(halo.window_bytes),
        ),
    )
    exchange.reweigh(weights)
    return exchange


@torch.inference_mode()
def exchange_(
    model: Any,
    port: str,
    *,
    target_ids: torch.Tensor,
    source_ids: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Fill import ``port`` (``"module.port"``) from its export through a link
    table: each linked target point becomes the weighted sum of its sources.

    Call it in a managed step between scopes; every rank makes the same
    calls with the same tables. Unlinked target points keep their values.
    The exchange is not differentiable: it raises when the source, target
    or weights require gradients.

    One plan is cached per port. A different ID table or field tensor
    rebuilds it (collectively); a different weight tensor of the same links
    only re-slices the weights. Weights on the model device are read at
    every call, so they may change in place; ID tables are read when the
    plan is built. Under a CUDA-graph executor the exchange is captured at
    its first call and replayed after: a new host weight tensor is copied
    into the captured buffers, while a new device tensor recaptures, so
    change device weights in place when they change often.
    """

    if not managed_step_active() or recording_sink() is not None:
        raise SubstepCompileError(
            "exchange_ runs inside a managed step, between its scopes"
        )
    runtime = ModelRuntime.of(model)
    module_name, _, port_name = port.partition(".")
    module = runtime.modules.get(module_name)
    declared = None if module is None else type(module).imports.get(port_name)
    if declared is None:
        raise ValueError(f"{port!r} is not a declared import port")
    provider, _, export = declared.source.partition(".")
    offered = type(runtime.modules[provider]).exports[export]
    source = getattr(runtime.modules[provider], offered.field)
    target = getattr(module, declared.field)
    if any(tensor.requires_grad for tensor in (source, target, weights)):
        raise SubstepCompileError(
            f"exchange_ {port!r} is not differentiable; its source, target and "
            "weights must not require gradients"
        )
    exchange = runtime.remaps.get(port)
    if (
        exchange is None
        or exchange.target_ids is not target_ids
        or exchange.source_ids is not source_ids
        or exchange.source is not source
        or exchange.target is not target
    ):
        if exchange is not None:
            runtime.remaps.pop(port).close()
        exchange = _build(
            runtime,
            f"{provider}.{offered.field}",
            f"{module_name}.{declared.field}",
            (offered.dims, declared.dims),
            target_ids,
            source_ids,
            weights,
        )
        runtime.remaps[port] = exchange
    elif exchange.weights is not weights:
        exchange.reweigh(weights)
    exchange.run(runtime.execution.executor)


__all__ = ["exchange_"]

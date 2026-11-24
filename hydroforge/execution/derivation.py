# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""What each ``computed_cache`` feeds, and its re-derivation after updates.

The graph is recorded only for models declaring a cache. Its nodes are the
caches and the resident values that may read one: computed tensors, computed
scalars (dimensions) and kernel fields of every module. Each node maps to
the declared fields and nodes its formula reads directly.

A node is reached by a change when it reads a changed field or a reached
node. A reached cache is stale; so is a node reading a stale node, and a
reached node a stale node reads (a cache reading a field and a computed
tensor of it never mixes their old and new values). Reached values that no
stale node reads keep the framework's update semantics: they are not
re-derived.

- A structural or content update evaluates every stale computed tensor
  against the staged tensors before anything mutates; the fresh tensors join
  the update as replacements. Stale caches take their staged values; stale
  dimensions and kernel fields are evaluated again on their next read.
  Computed tensors viewing one storage (rows of one stacked cache value)
  cannot be replaced separately: such a formula must return its own copy.
- A parameter change drops stale caches; only derived parameters, which the
  change refreshes, may read them. Other stale values, bound or not, must
  keep the value they had before the change.
- Either update is rejected before mutation when it would leave a value fixed
  by a discarded input stale, or re-derive computed state.
- Formulas run without autograd history.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import FunctionType
from typing import Any, Literal

import torch

from hydroforge.declare.computed_cache import ComputedCache
from hydroforge.declare.kernel_field import _KernelField

NodeKind = Literal["cache", "tensor", "scalar", "kernel"]


@dataclass(frozen=True, slots=True)
class _Node:
    qualified: str
    module: Any
    name: str
    kind: NodeKind
    func: Callable[[Any], Any]


def _formula(module: Any, name: str) -> Callable[[Any], Any]:
    descriptor = inspect.getattr_static(type(module), name)
    if isinstance(descriptor, (ComputedCache, cached_property)):
        return descriptor.func
    if isinstance(descriptor, property):
        return descriptor.fget
    raise TypeError(f"{module.module_name}.{name} has no formula")


@dataclass(slots=True)
class DerivationGraph:
    """Recorded reads of every cache and of the values that may read one."""

    names: dict[int, dict[str, str]]
    owners: dict[str, tuple[Any, str]]
    candidates: dict[str, _Node]
    discarded: frozenset[str]
    nodes: dict[str, _Node] = field(default_factory=dict)
    reads: dict[str, frozenset[str]] = field(default_factory=dict)

    def record(self, *, unbound: bool = False) -> None:
        """Record every resident node not recorded yet.

        With ``unbound``, kernel fields are recorded even when not resident
        (an invalidation drops kernel fields until the next binding evaluates
        them again).
        """

        with torch.no_grad():
            self._record(unbound)

    def _record(self, unbound: bool) -> None:
        from hydroforge.execution.parameters import _ReadRecorder

        pending = [
            node
            for qualified, node in self.candidates.items()
            if qualified not in self.nodes
            and (_resident(node) or (unbound and node.kind == "kernel"))
        ]
        while pending:
            node = pending.pop()
            if node.qualified in self.nodes:
                continue
            reads: set[str] = set()
            try:
                node.func(_ReadRecorder(node.module, self.names, reads))
            except Exception as error:
                raise ValueError(
                    f"cannot record what {node.qualified} reads: {error}"
                ) from error
            self.nodes[node.qualified] = node
            self.reads[node.qualified] = frozenset(reads)
            # Reading a cache made it resident: record it as well.
            pending.extend(
                self.candidates[name]
                for name in reads
                if name in self.candidates and name not in self.nodes
            )

    @property
    def fixed(self) -> frozenset[str]:
        """Nodes reading a discarded input, directly or through a node."""

        fixed: set[str] = set()
        changed = True
        while changed:
            changed = False
            for qualified, reads in self.reads.items():
                if qualified not in fixed and not reads.isdisjoint(
                    self.discarded | fixed
                ):
                    fixed.add(qualified)
                    changed = True
        return frozenset(fixed)

    def stale(self, changed: Iterable[str], *, sources: Iterable[str] = ()) -> set[str]:
        """Nodes a change of ``changed`` fields leaves stale.

        ``sources`` are nodes whose new value the caller supplies.
        """

        changed = set(changed)
        reached: set[str] = set()
        grew = True
        while grew:
            grew = False
            for qualified, reads in self.reads.items():
                if qualified not in reached and not reads.isdisjoint(changed | reached):
                    reached.add(qualified)
                    grew = True
        reached.difference_update(sources)
        stale = {q for q in reached if self.nodes[q].kind == "cache"}
        grew = True
        while grew:
            grew = False
            # What a stale node reads, besides what reads it, must be fresh.
            inputs = set().union(*(self.reads[q] for q in stale))
            for qualified in reached - stale:
                if qualified in inputs or not self.reads[qualified].isdisjoint(stale):
                    stale.add(qualified)
                    grew = True
        return stale

    def reject_fixed(self, stale: set[str], changed: Iterable[str]) -> None:
        fixed = sorted(stale & self.fixed)
        if fixed:
            raise ValueError(
                f"{fixed[0]} was derived from a discarded input and is fixed; "
                f"changing {', '.join(sorted(set(changed)))} would leave it stale"
            )


def _resident(node: _Node) -> bool:
    value = node.module.__dict__.get(node.name, _ABSENT)
    if node.kind == "tensor":
        return isinstance(value, torch.Tensor)
    return value is not _ABSENT


_ABSENT = object()


def _graph(runtime: Any) -> DerivationGraph | None:
    modules = [runtime.modules[name] for name in runtime.plan.modules]
    if not any(module.spec().computed_caches for module in modules):
        return None
    names: dict[int, dict[str, str]] = {}
    owners: dict[str, tuple[Any, str]] = {}
    candidates: dict[str, _Node] = {}
    discarded = set()
    for module_name in runtime.plan.modules:
        module = runtime.modules[module_name]
        spec = module.spec()
        module_names = names.setdefault(id(module), {})

        def declare(name: str, kind: NodeKind | None) -> None:
            qualified = f"{module_name}.{name}"
            module_names[name] = qualified
            owners[qualified] = (module, name)
            if kind is not None:
                candidates[qualified] = _Node(
                    qualified, module, name, kind, _formula(module, name)
                )

        for item in spec.fields.values():
            if item.tensor is not None:
                declare(item.name, "tensor" if item.computed else None)
                if getattr(
                    item.tensor, "mode", "device"
                ) == "discard" and module._is_tensor_field_active(item.name):
                    discarded.add(f"{module_name}.{item.name}")
            elif item.computed:
                declare(item.name, "scalar")
        for name in module._binding.plan.reference_indices:
            declare(name, None)
        for name in spec.kernel_fields:
            if isinstance(inspect.getattr_static(type(module), name), _KernelField):
                declare(name, "kernel")
        for name in spec.computed_caches:
            declare(name, "cache")
    return DerivationGraph(names, owners, candidates, frozenset(discarded))


def record_before_discard(runtime: Any) -> None:
    """Record the graph while discarded inputs are still resident."""

    graph = _graph(runtime)
    if graph is not None and graph.discarded:
        runtime.derivations = graph
        graph.record()


def _stored(module: Any, name: str) -> Any:
    """A declared tensor's stored value, without evaluating a lazy one."""

    value = module.__dict__.get(name)
    if value is None and name in module._binding.plan.reference_indices:
        cache_name = inspect.getattr_static(type(module), name).cache_name
        value = module.__dict__.get(cache_name)
    return value


def derivations(runtime: Any) -> DerivationGraph | None:
    """The model's derivation graph, extended by newly resident nodes."""

    graph = getattr(runtime, "derivations", None)
    if graph is None:
        graph = _graph(runtime)
        if graph is None:
            return None
        runtime.derivations = graph
    graph.record(unbound=True)
    return graph


# ---------------------------------------------------------------- staging


class _Staged:
    """Stand-in ``self`` reading staged replacements and re-derived values."""

    __slots__ = ("_staged_module", "_staging")

    def __init__(self, module: Any, staging: _Staging) -> None:
        self._staged_module = module
        self._staging = staging

    @property
    def __class__(self) -> type:
        return type(self._staged_module)

    def __getattr__(self, name: str) -> Any:
        from hydroforge.execution.parameters import _FRAMEWORK_ATTRIBUTES

        module = self._staged_module
        staging = self._staging
        qualified = staging.graph.names.get(id(module), {}).get(name)
        if qualified is not None:
            return staging.value(qualified)
        if name in module.spec().references:
            sibling = getattr(module, name)
            return None if sibling is None else _Staged(sibling, staging)
        attribute = inspect.getattr_static(type(module), name, None)
        if (
            isinstance(attribute, (FunctionType, property))
            and name not in _FRAMEWORK_ATTRIBUTES
        ):
            return attribute.__get__(self, type(module))
        return getattr(module, name)


@dataclass(slots=True)
class _Staging:
    graph: DerivationGraph
    replacements: Mapping[int, torch.Tensor]
    stale: set[str]
    fresh: dict[str, Any] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)

    def value(self, qualified: str) -> Any:
        if qualified in self.fresh:
            return self.fresh[qualified]
        if qualified in self.stale:
            if qualified in self.active:
                raise ValueError(f"{qualified} reads itself")
            node = self.graph.nodes[qualified]
            self.active.add(qualified)
            try:
                value = node.func(_Staged(node.module, self))
            finally:
                self.active.discard(qualified)
            if node.kind == "tensor":
                metadata = node.module.spec().tensor_fields[node.name].tensor
                if (
                    isinstance(value, torch.Tensor)
                    and metadata.contiguous
                    and value.layout is torch.strided
                ):
                    value = value.contiguous()
            self.fresh[qualified] = value
            return value
        module, name = self.graph.owners[qualified]
        value = getattr(module, name)
        if isinstance(value, torch.Tensor):
            return self.replacements.get(id(value), value)
        return value


@dataclass(frozen=True, slots=True)
class Restaging:
    """The re-derivation a staged update needs.

    ``pairs`` re-derive stale computed tensors; ``finish`` installs the
    staged cache values, drops the stale caches nothing evaluated, and
    forgets the dimensions and kernel fields whose value changes once the
    update has mutated the model.
    """

    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    caches: tuple[tuple[Any, str, Any], ...]
    dropped: tuple[tuple[Any, str], ...]
    forgotten: tuple[tuple[Any, str], ...]

    @property
    def resizes(self) -> bool:
        return any(current.shape != fresh.shape for current, fresh in self.pairs)

    @property
    def rebinds(self) -> bool:
        """Kernel fields or dimensions change: kernel bindings must rebuild."""

        return bool(self.forgotten)

    def finish(self) -> None:
        for module, name, value in self.caches:
            module.__dict__[name] = value
        for module, name in (*self.dropped, *self.forgotten):
            module.__dict__.pop(name, None)


def restage(
    runtime: Any,
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    derived: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> Restaging | None:
    """Re-derive what reads a cache the staged ``pairs`` (and the reference
    indices ``derived`` from them) leave stale, before mutation."""

    graph = derivations(runtime)
    if graph is None:
        return None
    replacements = {id(current): fresh for current, fresh in (*pairs, *derived)}
    changed: set[str] = set()
    staged_nodes: set[str] = set()
    for qualified, (module, name) in graph.owners.items():
        value = _stored(module, name)
        if isinstance(value, torch.Tensor) and id(value) in replacements:
            changed.add(qualified)
            if qualified in graph.nodes:
                staged_nodes.add(qualified)
    stale = graph.stale(changed, sources=staged_nodes)
    if not stale:
        return None
    graph.reject_fixed(stale, changed)
    for qualified in sorted(stale):
        node = graph.nodes[qualified]
        if node.kind != "tensor":
            continue
        category = node.module.spec().tensor_fields[node.name].tensor.category
        if category not in {"topology", "derived_param"}:
            raise ValueError(
                f"the update would re-derive computed {category} "
                f"{qualified} from a cache it reads; stage it explicitly"
            )
    with torch.no_grad():
        return _restaged(graph, pairs, derived, replacements, stale)


def _restaged(
    graph: DerivationGraph,
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    derived: Iterable[tuple[torch.Tensor, torch.Tensor]],
    replacements: Mapping[int, torch.Tensor],
    stale: set[str],
) -> Restaging:
    staging = _Staging(graph, replacements, stale)
    fresh_pairs = []
    # Fresh tensors must own their storage, as staged replacements do.
    storages = {
        (tensor.device, tensor.untyped_storage()._cdata)
        for current, replacement in (*pairs, *derived)
        for tensor in (current, replacement)
    }
    # Replaced targets must own their storage: name what shares it.
    targets = {
        (current.device, current.untyped_storage()._cdata): "a staged tensor"
        for current, _ in pairs
    }
    for qualified in sorted(stale):
        node = graph.nodes[qualified]
        current = node.module.__dict__.get(node.name)
        if node.kind != "tensor" or not isinstance(current, torch.Tensor):
            continue
        target = (current.device, current.untyped_storage()._cdata)
        if target in targets:
            raise ValueError(
                f"{targets[target]} and {qualified} share storage and cannot be "
                "replaced separately; a computed tensor read from a "
                "computed_cache must return its own copy (e.g. .clone()) to be "
                "re-derived by an update"
            )
        targets[target] = qualified
        fresh = staging.value(qualified)
        if not isinstance(fresh, torch.Tensor):
            raise ValueError(f"{qualified} must evaluate to a tensor")
        storage = (fresh.device, fresh.untyped_storage()._cdata)
        if storage in storages:
            fresh = fresh.clone(memory_format=torch.contiguous_format)
            storage = (fresh.device, fresh.untyped_storage()._cdata)
        storages.add(storage)
        fresh_pairs.append((current, fresh))
    caches = tuple(
        (graph.nodes[q].module, graph.nodes[q].name, staging.fresh[q])
        for q in sorted(stale)
        if graph.nodes[q].kind == "cache" and q in staging.fresh
    )
    dropped = []
    forgotten = []
    for qualified in sorted(stale):
        node = graph.nodes[qualified]
        if node.kind in {"scalar", "kernel"}:
            # A dimension or kernel value the update keeps needs no rebind.
            current = node.module.__dict__.get(node.name, _ABSENT)
            if current is _ABSENT or not same_value(current, staging.value(qualified)):
                forgotten.append((node.module, node.name))
        elif node.kind == "cache" and qualified not in staging.fresh:
            dropped.append((node.module, node.name))
    return Restaging(tuple(fresh_pairs), caches, tuple(dropped), tuple(forgotten))


def same_value(left: Any, right: Any) -> bool:
    """Whether a re-evaluated value equals the one in use."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and bool(torch.equal(left.cpu(), right.cpu()))
        )
    try:
        return bool(left == right)
    except Exception:
        return False


__all__ = [
    "DerivationGraph",
    "Restaging",
    "derivations",
    "record_before_discard",
    "restage",
]

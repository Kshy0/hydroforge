# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Halo scheduling of one recorded program.

Kernels never communicate: they read ghost rows from their fields like any
other row. The schedule records the exchanges (:mod:`hydroforge.parallel.halo`)
that make those rows current, from what kernels declare and the write sets
the recording sees. Every field the halo exchanges is, at each point of a
recording, either *current* within a depth (its ghost rows within that many
rows of the owned ones hold their owners' values: mirrored here, and nothing
recorded since wrote it) or *dirty* (anything else, the state at the start
of a recording):

- a kernel declaring ``halo={"h": r}`` gets a mirror of ``h`` recorded
  before it unless ``h`` is current within the rows ``r`` reaches (a mirror
  carries the ghost rows within its reader's distance; fields one call
  needs are mirrored together);
- a kernel declaring ``contributes=("s",)`` adds to rows of ``s`` other
  ranks own (its ghost rows): before anything but a contributor reads or
  writes ``s``, the contributions are summed into their owners (an
  accumulate, zeroing the ghost rows), and a ``halo`` reader gets a mirror
  of the sums after that. Before a contribution the ghost rows of ``s``
  must be zero: when they hold anything else (copies a mirror wrote) they
  are zeroed, by the next sum recorded once a reader used the copies, or
  by a zeroing of their own. Programs end with every contribution summed.

A loop body (fixed or adaptive physics) runs after itself. A field a
declared reader finds before anything in the loop wrote it is assumed
current at the body's entry: the loop mirrors it once before its first
iteration (:class:`LoopHalo`), and the body mirrors it at its end when it
left it dirty, so iterations exchange it once, not before every reader.
Ghost rows of what the body contributes to are likewise assumed zero at its
entry (the loop zeroes them before it runs) and left so.

Every decision depends on declarations, recorded write sets and the plan
(the same on every rank), never on data, so ranks record the same
exchanges in the same order.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace
from typing import Any

import torch


Storage = tuple[torch.device, int]


def _storage(tensor: torch.Tensor) -> Storage:
    # The storage itself, not its address: empty storages all sit at 0.
    return tensor.device, tensor.untyped_storage()._cdata


@dataclass
class _Entry:
    tensor: torch.Tensor
    index: int  # operators recorded when the state was set
    entry: bool = False  # assumed at loop entry: nothing in the body wrote it
    depth: int = 0  # ghost rows current within it; 0 all


def _at_depth(layout: Any, depth: int) -> Any:
    """``layout`` carrying the ghost rows within ``depth`` (0: every one)."""

    return replace(layout, depth=depth) if depth else layout


@dataclass(eq=False)
class _Ghosts:
    """The ghost rows of a field some kernel contributes to
    (``KernelSpec.contributes``): ``zero``, ``partial`` (contributions not
    yet summed into their owners) or ``copied`` (anything else: owners'
    values a mirror wrote, or unknown); ``read``: read since."""

    tensor: torch.Tensor
    state: str
    read: bool = False


class HaloSchedule:
    """The halo states of one recorder (see the module documentation)."""

    def __init__(self, recorder: Any) -> None:
        # The recorder owns this schedule; it must release recorded tensors
        # without waiting for cyclic garbage collection.
        self.recorder = weakref.proxy(recorder)
        self.entries: dict[int, _Entry] = {}
        self.busy = False
        # A loop body: fields first met by a declared reader are assumed
        # current at entry (the module documentation); ``clobbered`` holds
        # the storages the loop's other programs write.
        self.carried = getattr(recorder, "scope_kind", None) in (
            "fixed",
            "adaptive physics",
        )
        self.clobbered: frozenset[Storage] = frozenset()
        self.assumed: dict[int, torch.Tensor] = {}
        # The depth every reader of an assumed field needs at entry.
        self.assumed_depth: dict[int, int] = {}
        # Ghost rows of fields exchanged or contributed to here (by storage);
        # the storages kernels contribute to; in a loop body, those whose
        # ghost rows are assumed zero at entry.
        self.ghosts: dict[Storage, _Ghosts] = {}
        self.contributed: set[Storage] = set()
        self.zeroed: dict[Storage, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    @property
    def runtime(self) -> Any:
        return getattr(self.recorder.execution, "runtime", None)

    @property
    def halo(self) -> Any:
        return getattr(self.runtime, "halo", None)

    def _entry(self, tensor: torch.Tensor) -> _Entry | None:
        """The live entry of ``tensor``: none once a later operator wrote
        its storage (then it is dirty)."""

        entry = self.entries.get(id(tensor))
        if entry is None or entry.tensor is not tensor:
            return None
        storage = _storage(tensor)
        for operator in self.recorder.operators[entry.index :]:
            for written in getattr(operator, "writes", ()):
                if isinstance(written, torch.Tensor) and _storage(written) == storage:
                    del self.entries[id(tensor)]
                    return None
        return entry

    def _set(self, tensor: torch.Tensor, entry: bool = False, depth: int = 0) -> None:
        self.entries[id(tensor)] = _Entry(
            tensor, len(self.recorder.operators), entry, depth
        )

    def _assume(self, tensor: torch.Tensor, depth: int) -> _Entry | None:
        """In a loop body, a field nothing in the loop wrote yet is current
        at entry within ``depth``."""

        storage = _storage(tensor)
        if (
            not self.carried
            or storage in self.clobbered
            or storage in written_storages(self.recorder.operators)
        ):
            return None
        self.assumed[id(tensor)] = tensor
        self.assumed_depth[id(tensor)] = depth
        self._set(tensor, entry=True, depth=depth)
        return self.entries[id(tensor)]

    @property
    def entry(self) -> LoopHalo:
        """What the loop does before its first iteration (nothing outside a
        halo)."""

        if self.halo is None:
            return LoopHalo()
        return LoopHalo(
            tuple(self.assumed.values()),
            tuple(self.assumed_depth.get(key, 0) for key in self.assumed),
            zero=tuple(self.zeroed.values()),
        )

    def _record(self, function, *args, **kwargs) -> None:
        """Record framework exchanges without scheduling them again."""

        self.busy, busy = True, self.busy
        try:
            function(*args, **kwargs)
        finally:
            self.busy = busy

    # ------------------------------------------------------------------ #
    def mirror(
        self, fields: tuple[torch.Tensor, ...], depths: tuple[int, ...] = ()
    ) -> None:
        """Record a mirror of ``fields`` (any state) within ``depths``
        (default every ghost row) and make them current."""

        if not fields:
            return
        from hydroforge.parallel.halo import MAX_FIELDS

        depths = depths or (0,) * len(fields)
        runtime = self.runtime
        layouts = [
            _at_depth(runtime.halo_layout(t), d)
            for t, d in zip(fields, depths, strict=True)
        ]
        for start in range(0, len(fields), MAX_FIELDS):
            stop = start + MAX_FIELDS
            self._record(
                self.halo.exchange, fields[start:stop], layouts[start:stop], "mirror"
            )
        for tensor, depth in zip(fields, depths, strict=True):
            self._set(tensor, depth=depth)
            self._ghost(tensor, "copied")

    def before_operator(self, accessed: Any) -> None:
        """An operator other than a kernel is about to read or write
        ``accessed``: contributions to them are summed first, and it may
        write any row of them."""

        if self.busy or not self.ghosts or self.halo is None:
            return
        tensors = [t for t in accessed if isinstance(t, torch.Tensor)]
        self._reduce([t for t in tensors if self._state(t) == "partial"])
        for tensor in tensors:
            if _storage(tensor) in self.contributed:
                self._ghost(tensor, "copied")

    def settle(self) -> None:
        """A nested program follows: it finds every contribution summed."""

        if self.halo is not None:
            self._reduce(
                [g.tensor for g in self.ghosts.values() if g.state == "partial"]
            )

    def resume(self, nested: HaloSchedule) -> None:
        """A nested program (recorded after :meth:`settle`, run at least
        once) returned: ghost rows it exchanged or contributed to are as it
        left them; those of contributed fields it otherwise accessed are
        unknown (its operators may write any row)."""

        if self.halo is None:
            return
        self.contributed |= nested.contributed
        for storage, ghosts in nested.ghosts.items():
            self._ghost(ghosts.tensor, ghosts.state)
            self.ghosts[storage].read = ghosts.read
        for operator in nested.recorder.operators:
            for tensor in (
                *getattr(operator, "reads", ()),
                *getattr(operator, "writes", ()),
            ):
                if not isinstance(tensor, torch.Tensor):
                    continue
                storage = _storage(tensor)
                if storage in self.contributed and storage not in nested.ghosts:
                    self._ghost(tensor, "copied")

    # ------------------------------------------------------------------ #
    # Contributions (KernelSpec.contributes)
    def _ghost(self, tensor: torch.Tensor, state: str) -> None:
        storage = _storage(tensor)
        previous = self.ghosts.get(storage)
        # Readers may pass views. Retain the field with its complete halo
        # layout, rather than replacing it with a slice of the storage.
        self.ghosts[storage] = _Ghosts(
            tensor if previous is None else previous.tensor, state
        )

    def _state(self, tensor: torch.Tensor) -> str:
        """``zero``, ``partial``, ``copied``, or ``entry`` (nothing here
        touched its ghost rows yet). States are per storage: a view mirrored
        or assumed current at entry gives its field's ghost rows copies."""

        storage = _storage(tensor)
        found = self.ghosts.get(storage)
        if found is not None:
            return found.state
        if any(
            _storage(t) == storage
            for t in (
                *self.assumed.values(),
                *(e.tensor for e in self.entries.values()),
            )
        ):
            return "copied"
        return "entry"

    def _reduce(
        self, fields: list[torch.Tensor], clears: list[torch.Tensor] = ()
    ) -> None:
        """Sum the contributions to ``fields`` into their owners (zeroing
        their ghost rows), and zero the ghost rows of ``clears``: each sum
        zeroes one more field's, a field read since its ghost rows took
        copies when none is asked for (they are spent: the next
        contribution would zero them)."""

        from hydroforge.parallel.halo import MAX_FIELDS

        # An operator may name a storage more than once (or through a view).
        # Exchanging aliases together races reads against ghost zeroing.
        storages = dict.fromkeys(_storage(t) for t in fields)
        fields = [self.ghosts[storage].tensor for storage in storages]
        clears = list(
            {_storage(t): t for t in clears if _storage(t) not in storages}.values()
        )
        runtime = self.runtime
        for start in range(0, len(fields), MAX_FIELDS):
            chunk = tuple(fields[start : start + MAX_FIELDS])
            taken = {_storage(t) for t in chunk}
            clear = None
            for index, candidate in enumerate(clears):
                if _storage(candidate) not in taken:
                    clear = clears.pop(index)
                    break
            if clear is None:
                clear = next(
                    (
                        g.tensor
                        for storage, g in self.ghosts.items()
                        if g.state == "copied"
                        and g.read
                        and storage in self.contributed
                        and storage not in taken
                    ),
                    None,
                )
            self._record(
                self.halo.exchange,
                chunk,
                [runtime.halo_layout(t) for t in chunk],
                "accumulate",
                clear,
                None if clear is None else runtime.halo_layout(clear),
            )
            for tensor in (*chunk, *(() if clear is None else (clear,))):
                self.entries.pop(id(tensor), None)
                self._ghost(tensor, "zero")
        for tensor in clears:
            # An accumulate of no fields: it only zeroes the ghost rows.
            self._record(
                self.halo.exchange,
                (),
                [],
                "accumulate",
                tensor,
                runtime.halo_layout(tensor),
            )
            self.entries.pop(id(tensor), None)
            self._ghost(tensor, "zero")

    def prepare_sums(self, registry: Any, spec: Any, arguments: dict) -> Any:
        """Before a kernel call: sum the contributions to what it accesses
        (but contributes to), and zero the ghost rows of what it
        contributes to unless they hold contributions or nothing (in a loop
        body, nothing at entry: the loop zeroes them before it runs). The
        returned callable notes the call once it is recorded."""

        from hydroforge.execution.binding import _contributions
        from hydroforge.execution.operators import buffer_access_semantics

        if self.busy or self.halo is None or not (spec.contributes or self.ghosts):
            return None
        bound, _ = self.recorder.binder.arguments(registry, arguments)
        contributes = {
            _storage(tensor): tensor for tensor in _contributions(spec, bound)
        }
        accessed, read = [], []
        for name, access in spec.buffers.items():
            tensor = bound.get(name)
            if not isinstance(tensor, torch.Tensor) or _storage(tensor) in contributes:
                continue
            accessed.append(tensor)
            if buffer_access_semantics(access).reads:
                read.append(tensor)
        clears = []
        for storage, tensor in contributes.items():
            self.contributed.add(storage)
            state = self._state(tensor)
            if state == "entry" and self.carried and storage not in self.clobbered:
                self.zeroed[storage] = tensor
                self._ghost(tensor, "zero")
            elif state in ("entry", "copied"):
                clears.append(tensor)
        self._reduce([t for t in accessed if self._state(t) == "partial"], clears)

        def recorded() -> None:
            for tensor in read:
                found = self.ghosts.get(_storage(tensor))
                if found is not None and found.state == "copied":
                    found.read = True
            for tensor in contributes.values():
                self._ghost(tensor, "partial")

        return recorded

    # ------------------------------------------------------------------ #
    def prepare_reads(self, registry: Any, spec: Any, arguments: dict) -> None:
        """Before a call of a kernel declaring ``halo``: record a mirror of
        each declared field not current within the rows the kernel reads."""

        from hydroforge.parallel.halo import covers, deeper, read_depth

        runtime, halo = self.runtime, self.halo
        if halo is None:
            return
        bound, _ = self.recorder.binder.arguments(registry, arguments)
        stale: dict[int, tuple[torch.Tensor, int]] = {}
        for name, distance in spec.halo.items():
            tensor = bound[name]
            if tensor is None:
                continue
            if isinstance(distance, str):
                distance = bound[distance]
                if distance < 1:
                    raise ValueError(f"{spec.name}: halo distance must be positive")
            limit = runtime.halo_depth(runtime.halo_layout(tensor))
            if limit and distance > limit:
                raise ValueError(
                    f"{spec.name} reads {name} {distance} rows away; the "
                    f"slab holds {limit} ghost rows"
                )
            need = read_depth(distance, limit)
            entry = self._entry(tensor) or self._assume(tensor, need)
            if entry is not None and not covers(entry.depth, need):
                if entry.entry:
                    # Read farther than assumed: the loop provides it deeper
                    # at entry (its mirrors before the first iteration and
                    # at the body's end).
                    entry.depth = deeper(entry.depth, need)
                    key = id(tensor)
                    self.assumed_depth[key] = deeper(self.assumed_depth[key], need)
                else:
                    entry = None
            if entry is None:
                key = id(tensor)
                stale[key] = (
                    tensor,
                    deeper(stale[key][1], need) if key in stale else need,
                )
        self.mirror(
            tuple(t for t, _ in stale.values()), tuple(d for _, d in stale.values())
        )

    # ------------------------------------------------------------------ #
    def finish(self) -> None:
        """The recording ends: contributions are summed, ghost rows a loop
        body assumed zero at entry are zero again, and fields it assumed
        current at entry are current again."""

        from hydroforge.parallel.halo import covers

        if self.halo is None:
            return
        self._reduce(
            [g.tensor for g in self.ghosts.values() if g.state == "partial"],
            [
                t
                for storage, t in self.zeroed.items()
                if self.ghosts[storage].state == "copied"
            ],
        )
        dirty = {}
        for key, tensor in self.assumed.items():
            need = self.assumed_depth[key]
            live = self._entry(tensor)
            if live is None or not (live.entry or covers(live.depth, need)):
                dirty[key] = (tensor, need)
        self.mirror(
            tuple(t for t, _ in dirty.values()), tuple(d for _, d in dirty.values())
        )


def written_storages(operators: Any) -> frozenset[Storage]:
    """The storages ``operators`` write."""

    return frozenset(
        _storage(written)
        for operator in operators
        for written in getattr(operator, "writes", ())
        if isinstance(written, torch.Tensor)
    )


class LoopHalo:
    """What a loop does before each run (outside any recording): it zeroes
    the ghost rows of the fields its body contributes to (``zero``) and
    mirrors the fields its body assumes current, within ``depths`` (default
    every ghost row). Launches are bound and compiled once: an eager
    exchange would bind its call again on every run, host work each step
    would then wait for."""

    def __init__(
        self,
        fields: tuple[torch.Tensor, ...] = (),
        depths: tuple[int, ...] = (),
        *,
        zero: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self.fields = fields
        self.depths = depths or (0,) * len(fields)
        self.zero = zero
        self.launches: tuple[Any, ...] | None = None

    def enter(self, execution: Any) -> None:
        if not (self.fields or self.zero):
            return
        if self.launches is None:
            self.launches = tuple(
                execution.kernel_binding.bind(
                    execution.runtime.halo.registry, arguments
                ).compile()
                for arguments in self._arguments(execution)
            )
        for launch in self.launches:
            launch()

    def _arguments(self, execution: Any) -> list[dict]:
        from hydroforge.parallel.halo import MAX_FIELDS

        runtime = execution.runtime
        halo = runtime.halo
        found = [
            halo.exchange_arguments(
                (), [], "accumulate", tensor, runtime.halo_layout(tensor)
            )
            for tensor in self.zero
        ]
        for start in range(0, len(self.fields), MAX_FIELDS):
            chunk = self.fields[start : start + MAX_FIELDS]
            found.append(
                halo.exchange_arguments(
                    chunk,
                    [
                        _at_depth(runtime.halo_layout(t), d)
                        for t, d in zip(chunk, self.depths[start : start + MAX_FIELDS])
                    ],
                    "mirror",
                )
            )
        return [arguments for arguments in found if arguments is not None]


__all__ = ["HaloSchedule", "LoopHalo", "written_storages"]

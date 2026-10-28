# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Halo scheduling of one recorded program.

Every field the halo exchanges is, at each point of a recording, in one of
three states: *current* (its ghost rows hold their owners' values: mirrored
here, and nothing recorded since wrote it), *ring* (a kernel declaring
``publishes`` wrote it and stored its rows into the peers' rings: this
rank's ghost rows of it wait in this rank's receive ring), or *dirty*
(anything else, the state at the start of a recording). The schedule keeps
them and records exchanges where they are needed, so model code only
declares what kernels read and write:

- a kernel declaring ``halo={"h": r}`` finds ``h`` current, or reads its
  ghost rows from the ring (CUDA kernels on peer links, writing them back
  as it reads), or gets a mirror of ``h`` recorded before it when it is
  dirty; if it also writes ``h``, ``h`` is dirty after it (or in its ring,
  when it publishes ``h``);
- any other operator reading or writing a ring field gets the ring's
  receipt recorded before it (a kernel declaring ``halo`` or ``publishes``
  reads ghost rows of its ``halo`` fields only: what it writes merely
  supersedes the ring);
- ``halo_exchange_`` mirrors only dirty fields (current ones are pruned)
  and receives ring ones;
- a kernel declaring ``contributes=("s",)`` adds to rows of ``s`` other
  ranks own (its ghost rows): before anything but a contributor reads or
  writes ``s``, the contributions are summed into their owners (an
  accumulate, zeroing the ghost rows), and a ``halo`` reader gets a mirror
  of the sums after that. Before a contribution the ghost rows of ``s``
  must be zero: when they hold anything else (copies a mirror wrote) they
  are zeroed, by the next sum recorded once a reader used the copies, or
  by a zeroing of their own. Programs end with every contribution summed;
  a loop body assumes the ghost rows of what it contributes to zero at its
  entry (the loop zeroes them before it runs) and leaves them so.

Messages and slots (``halo_access.cuh``, ``halo.cu``): a rank numbers the
messages of each peer, counts those it finished with (``received``) and
returns their slots with a credit; a sender reuses a slot only after its
credit. A launch reading a ring wants message ``received + skip``, so the
schedule fixes when each message is counted and credited: in the prologue
of a later launch the program makes anyway (a declared kernel reading no
ring, any exchange), so returning a slot costs no launch of its own. To
decide it, the recording logs its halo launches (*sites*) and, when it
ends, replays them over every rank's messages: which message each read
wants, which launch counts and credits what, and whether a sender would
wait for a slot nothing returns (then a receipt is recorded early enough).
Every decision depends on declarations, recorded write sets and the plan
(the same on every rank), never on data, so ranks record alike.

A loop body (a fixed body without a final tail, an adaptive physics body
whose proposal exchanges nothing) runs after itself: the messages it
leaves uncounted at its end are the ones it finds at its next start. A
ring a kernel publishes at the end of the body is read straight from the
ring at the top of the next iteration, and the loop sends those messages
once before its first iteration and receives them after its last
(:class:`LoopHalo`). A field a declared reader finds before anything in
the loop wrote it, and that no such message carries, is assumed current
at entry: the body mirrors it at its end if it leaves it dirty, and the
loop mirrors it once before its first iteration.

Such a body often needs no credits at all: when every sender, before
storing message ``n`` of a pair, waited (in an earlier launch) for a
message its receiver sent only after finishing with message ``n - DEPTH``
of that pair, the slot is free by then whatever the timing
(:func:`slots_free` proves it over the replay, for every pair). The body
then neither waits for credits nor returns them; the loop first waits
until every message sent before it is credited, and afterwards sets the
credits it skipped (``HYDROFORGE_HALO_CREDITS=1`` keeps them, for
comparisons).
"""

from __future__ import annotations

import bisect
import os
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils._python_dispatch import _disable_current_modes

from hydroforge.kernels.spec import HALO_PARAMETER, HaloArgument

# A read of a field nothing in this loop iteration wrote yet: from the ring
# when a message the previous iteration left pending carries it.
_ENTRY = "entry"
# HYDROFORGE_HALO_CREDITS=1 credits every slot, even in loop bodies whose
# slots are proven free (for comparisons).
_ALWAYS_CREDIT = os.environ.get("HYDROFORGE_HALO_CREDITS", "0") == "1"


def _storage(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


@dataclass
class _Entry:
    tensor: torch.Tensor
    ring: Any  # the _Site whose publication holds the ghost rows, or None
    index: int  # operators recorded when the state was set
    entry: bool = False  # assumed at loop entry: nothing in the body wrote it


@dataclass(eq=False)
class _Site:
    """One halo launch of a recording: a declared kernel call or an
    exchange. Its argument is bound when it is recorded and filled in when
    the recording ends."""

    kind: str  # "kernel" or "exchange"
    argument: HaloArgument
    # kernel: (tensor, layout, source (_Site, _ENTRY or None), consumes)
    reads: list[tuple[torch.Tensor, Any, Any, bool]] = field(default_factory=list)
    publish: tuple[tuple[torch.Tensor, Any], ...] = ()
    # storages it writes (they supersede rings holding them)
    writes: frozenset[int] = frozenset()
    # exchange: its fields, mode and phase; a receipt's (site, "current" or
    # "entry") publication
    fields: tuple[torch.Tensor, ...] = ()
    layouts: tuple[Any, ...] = ()
    mode: str = "mirror"
    phase: str = "both"
    receipt: tuple[Any, str] | None = None
    operator: Any = None  # None: it launches nothing on this rank
    index: int = 0  # operators recorded before it
    name: str = ""
    empty: bool = False  # a kernel call launching nothing on this rank
    carrier: _Site | None = None  # the book running an empty call's prologue
    carried: bool = False  # this is a carrier

    @property
    def group(self) -> Any:
        """The fields and layouts of the messages it sends or receives."""

        from hydroforge.kernels.halo import Published

        if self.kind == "kernel":
            return Published(
                tuple(t for t, _ in self.publish), tuple(layout for _, layout in self.publish)
            )
        return Published(self.fields, self.layouts)


@dataclass(eq=False)
class _Ghosts:
    """The ghost rows of a field some kernel contributes to
    (``KernelSpec.contributes``): ``zero``, ``partial`` (contributions not
    yet summed into their owners) or ``copied`` (anything else: owners'
    values a mirror wrote, or unknown); ``read``: read since."""

    tensor: torch.Tensor
    state: str
    read: bool = False


@dataclass(eq=False)
class _Event:
    """A point of the recording the replay may need receipts before."""

    kind: str  # "access" (a plain operator) or "settle" (a nested loop)
    index: int  # operators recorded before it
    storages: frozenset[int] = frozenset()


@dataclass(eq=False)
class _Item:
    """One message instance (from every sender of its site)."""

    site: _Site
    pending: set[int]  # storages of published fields not finished with
    pairs: tuple[tuple[int, int], ...]  # (sender, receiver)
    entry: bool = False  # left by the previous iteration
    done: bool = False


@dataclass(frozen=True)
class CarriedState:
    """What a loop body leaves uncounted at its end (and finds at its
    start): its messages in order, each with the storages still pending
    (none: finished with, not yet counted), and how many lead credited."""

    items: tuple[tuple[_Site, frozenset[int]], ...] = ()
    credited: int = 0


class _Insert(Exception):
    """The replay needs launches recorded before ``before`` (a site or an
    event): ``receipts`` of items, then a book when ``book``."""

    def __init__(self, before: Any, receipts: list[_Item], book: bool = False) -> None:
        super().__init__()
        self.before = before
        self.receipts = receipts
        self.book = book


@dataclass
class _Replay:
    """The outcome of one replay: tables per site and the state left."""

    kernel: dict[int, tuple[list, list]]  # id(site) -> (reads, commit)
    exchange: dict[int, tuple]  # id(site) -> site_words arguments
    state: CarriedState
    fused: int = 0
    receipts: int = 0
    # Per launch, in order: the messages it waits for (item, the pairs it
    # waits on), the items it publishes, those it finishes with; then the
    # items and how many lead counted at the end (for slots_free).
    trace: list[tuple[list, list[int], list[int]]] = field(default_factory=list)
    items: list[_Item] = field(default_factory=list)
    counted: int = 0


class HaloSchedule:
    """The halo states of one recorder (see the module documentation)."""

    def __init__(self, recorder: Any) -> None:
        # The recorder owns this schedule; it must release recorded tensors
        # without waiting for cyclic garbage collection.
        self.recorder = weakref.proxy(recorder)
        self.entries: dict[int, _Entry] = {}
        self.busy = False
        # Fields the call being prepared may read from rings and writes back.
        self.written_back: tuple[torch.Tensor, ...] = ()
        # A loop body: fields first met by a declared reader are assumed
        # current at entry (the module documentation); ``clobbered`` holds
        # the storages the loop's other programs write, ``outside`` those
        # they access and whether they exchange halos.
        self.carried = getattr(recorder, "scope_kind", None) in (
            "fixed",
            "adaptive physics",
        )
        self.clobbered: frozenset[int] = frozenset()
        self.outside: tuple[frozenset[int], bool] = (frozenset(), False)
        self.assumed: dict[int, torch.Tensor] = {}
        self.log: list[_Site | _Event] = []
        self.state = CarriedState()
        self.free = False  # a loop body storing into slots without credits
        # Ghost rows of fields exchanged or contributed to here (by storage);
        # the storages kernels contribute to; in a loop body, those whose
        # ghost rows are assumed zero at entry.
        self.ghosts: dict[int, _Ghosts] = {}
        self.contributed: set[int] = set()
        self.zeroed: dict[int, torch.Tensor] = {}
        self._fusable: bool | None = None

    # ------------------------------------------------------------------ #
    @property
    def runtime(self) -> Any:
        return getattr(self.recorder.execution, "runtime", None)

    @property
    def halo(self) -> Any:
        return getattr(self.runtime, "halo", None)

    @property
    def fusable(self) -> bool:
        """Kernels publish and read rings (peer links everywhere, CUDA)."""

        if self._fusable is None:
            from hydroforge.kernels.halo import publishable

            halo = self.halo
            self._fusable = (
                halo is not None
                and self.recorder.execution.backend.name == "cuda"
                and not publishable(halo)
            )
        return self._fusable

    def _entry(self, tensor: torch.Tensor) -> _Entry | None:
        """The live entry of ``tensor``: none once a later operator wrote
        its storage (then it is dirty)."""

        entry = self.entries.get(id(tensor))
        if entry is None or entry.tensor is not tensor:
            return None
        storage = _storage(tensor)
        operators = self.recorder.operators
        for operator in operators[entry.index :]:
            for written in getattr(operator, "writes", ()):
                if isinstance(written, torch.Tensor) and _storage(written) == storage:
                    del self.entries[id(tensor)]
                    return None
        return entry

    def _assume(self, tensor: torch.Tensor) -> _Entry | None:
        """In a loop body, a field nothing in the loop wrote yet is current
        at entry (or waits in a ring the previous iteration left)."""

        storage = _storage(tensor)
        if (
            not self.carried
            or storage in self.clobbered
            or storage in written_storages(self.recorder.operators)
        ):
            return None
        self.assumed[id(tensor)] = tensor
        self._set(tensor, entry=True)
        return self.entries[id(tensor)]

    @property
    def entry(self) -> LoopHalo:
        """What the loop does before its first iteration and after its last
        (nothing outside a halo)."""

        if self.halo is None:
            return LoopHalo()
        pending = {s for _, p in self.state.items for s in p}
        fields = tuple(
            t for t in self.assumed.values() if _storage(t) not in pending
        )
        return LoopHalo(
            fields,
            self.state,
            free=self.free,
            anchor=self._any_fields()[:8],
            zero=tuple(self.zeroed.values()),
        )

    def _set(self, tensor: torch.Tensor, ring: Any = None, entry: bool = False) -> None:
        self.entries[id(tensor)] = _Entry(
            tensor, ring, len(self.recorder.operators), entry
        )

    def _record(self, function, *args, **kwargs) -> None:
        """Record framework exchanges without scheduling them again."""

        self.busy, busy = True, self.busy
        try:
            function(*args, **kwargs)
        finally:
            self.busy = busy

    def _exchange(
        self,
        fields: tuple[torch.Tensor, ...],
        mode: str = "mirror",
        *,
        clear: torch.Tensor | None = None,
        phase: str = "both",
        receipt: tuple[Any, str] | None = None,
        record: Any = None,
    ) -> _Site | None:
        """Record one exchange (``record``: the model's own call, given the
        site argument) as a site (``None`` without ring work)."""

        halo, runtime = self.halo, self.runtime
        argument = HaloArgument(halo.plain_site.words) if self.fusable else None
        layouts = tuple(runtime.halo_layout(t) for t in fields)
        operators = self.recorder.operators
        before = len(operators)
        if record is not None:
            self._record(record, argument)
        else:
            clear_layout = None if clear is None else runtime.halo_layout(clear)
            self._record(
                halo.exchange,
                fields,
                list(layouts),
                mode,
                clear,
                clear_layout,
                site=argument,
            )
        if not self.fusable:
            return None
        # Logged on every rank, launching or not, so every rank replays the
        # same sites.
        site = _Site(
            "exchange",
            argument,
            fields=tuple(fields),
            layouts=layouts,
            mode=mode,
            phase=phase,
            receipt=receipt,
            operator=operators[-1] if len(operators) > before else None,
            index=before,
            writes=frozenset(
                _storage(t) for t in (*fields, *(() if clear is None else (clear,)))
            ),
        )
        self.log.append(site)
        return site

    # ------------------------------------------------------------------ #
    def mirror(self, fields: tuple[torch.Tensor, ...]) -> None:
        """Record a mirror of ``fields`` (any state) and make them current."""

        if not fields:
            return
        for start in range(0, len(fields), 8):
            self._exchange(tuple(fields[start : start + 8]))
        for tensor in fields:
            self._set(tensor)
            self._ghost(tensor, "copied")

    def receipt(self, published: _Site) -> None:
        """Record the receipt of a publication's ring (no messages)."""

        waiting = [
            tensor
            for tensor, _ in published.publish
            if (entry := self._entry(tensor)) is not None and entry.ring is published
        ]
        self._exchange(
            tuple(t for t, _ in published.publish),
            phase="receive",
            receipt=(published, "current"),
        )
        for tensor in waiting:
            self._set(tensor)
        for tensor, _ in published.publish:
            self._ghost(tensor, "copied")

    def resolve_ring(self, tensors: Any = None) -> None:
        """Receive every ring field (of ``tensors``, or all of them)."""

        storages = None if tensors is None else {_storage(t) for t in tensors}
        pending: dict[int, _Site] = {}
        for entry in list(self.entries.values()):
            live = self._entry(entry.tensor)
            if live is None or live.ring is None:
                continue
            if storages is not None and _storage(live.tensor) not in storages:
                continue
            pending[id(live.ring)] = live.ring
        for published in pending.values():
            self.receipt(published)

    def _access(self, tensors: list[torch.Tensor]) -> None:
        """A plain operator accesses ``tensors``: fields assumed at loop
        entry may wait in a ring (the replay receives it first)."""

        storages = frozenset(
            _storage(t)
            for t in tensors
            if (entry := self._entry(t)) is not None and entry.entry
        )
        if storages and self.fusable:
            self.log.append(_Event("access", len(self.recorder.operators), storages))

    # ------------------------------------------------------------------ #
    def before_operator(self, accessed: Any, *, kernel: bool = False) -> None:
        """An operator about to be recorded reads or writes ``accessed``:
        ring fields among them are received first, and contributions to
        them summed (a kernel's are, by :meth:`prepare_sums`); an operator
        other than a kernel may write any row of them."""

        if self.busy:
            return
        tensors = [t for t in accessed if isinstance(t, torch.Tensor)]
        if not kernel and self.ghosts and self.halo is not None:
            self._reduce([t for t in tensors if self._state(t) == "partial"])
            for tensor in tensors:
                if _storage(tensor) in self.contributed:
                    self._ghost(tensor, "copied")
        if tensors and self.entries:
            self._access(tensors)
            self.resolve_ring(tensors)

    def settle(self) -> None:
        """A nested loop follows: it finds every message counted."""

        if self.halo is None:
            return
        self._reduce([g.tensor for g in self.ghosts.values() if g.state == "partial"])
        self.resolve_ring()
        if self.fusable:
            self.log.append(_Event("settle", len(self.recorder.operators)))

    def request_mirror(self, fields: tuple[torch.Tensor, ...]) -> None:
        """``halo_exchange_`` of ``fields`` in mirror mode."""

        halo = self.halo
        halo.mirrors_requested += len(fields)
        dirty, ring = [], []
        for tensor in fields:
            entry = self._entry(tensor)
            if entry is None:
                dirty.append(tensor)
            elif entry.ring is not None:
                ring.append(tensor)
        halo.mirrors_pruned += len(fields) - len(dirty) - len(ring)
        self._access(list(fields))
        if ring:
            self.resolve_ring(ring)
        self.mirror(tuple(dirty))

    def other_exchange(
        self,
        fields: tuple[torch.Tensor, ...],
        record,
        *,
        mode: str = "accumulate",
        clear: torch.Tensor | None = None,
    ) -> None:
        """An accumulate exchange of ``fields`` (zeroing ``clear``'s
        ghosts); ``record(site)`` records it. It rewrites them."""

        touched = (*fields, *(() if clear is None else (clear,)))
        self._access(list(touched))
        self.resolve_ring(touched)
        self._exchange(tuple(fields), mode, clear=clear, record=record)
        for tensor in touched:
            self.entries.pop(id(tensor), None)
            self._ghost(tensor, "zero")

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
        touched its ghost rows yet)."""

        found = self.ghosts.get(_storage(tensor))
        if found is not None:
            return found.state
        if id(tensor) in self.assumed or id(tensor) in self.entries:
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

        # An operator may name a storage more than once (or through a view).
        # Exchanging aliases together races reads against ghost zeroing.
        storages = dict.fromkeys(_storage(t) for t in fields)
        fields = [self.ghosts[storage].tensor for storage in storages]
        clears = list(
            {
                _storage(t): t
                for t in clears
                if _storage(t) not in storages
            }.values()
        )
        for start in range(0, len(fields), 8):
            chunk = tuple(fields[start : start + 8])
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
            self.other_exchange(chunk, None, mode="accumulate", clear=clear)
        for tensor in clears:
            self._clear(tensor)

    def _clear(self, tensor: torch.Tensor) -> None:
        """Zero the ghost rows of ``tensor`` (an accumulate of no fields)."""

        runtime = self.runtime
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
        values = dict(arguments)
        if spec.runtime_scalars.get(HALO_PARAMETER) == "halo":
            values.setdefault(HALO_PARAMETER, self.runtime.halo_quiet)
        bound, _ = self.recorder.binder.arguments(registry, values)
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
    def prepare_call(
        self, registry: Any, spec: Any, arguments: dict
    ) -> tuple[dict, Any]:
        """Arguments of a call of a kernel declaring ``halo``/``publishes``
        (with its ``halo`` argument), after recording what it needs first;
        and what to note once it is recorded."""

        from hydroforge.execution.operators import buffer_access_semantics
        from hydroforge.kernels.halo import consumes

        self.written_back = ()
        runtime, halo = self.runtime, self.halo
        quiet = runtime.halo_quiet
        placeholders = {} if HALO_PARAMETER in arguments else {HALO_PARAMETER: quiet}
        bound, _ = self.recorder.binder.arguments(
            registry, {**arguments, **placeholders}
        )
        if halo is None:
            return {**arguments, **placeholders}, None
        fusable = self.fusable
        states = []
        for name, distance in spec.halo.items():
            tensor = bound[name]
            layout = runtime.halo_layout(tensor)
            limit = runtime.halo_depth(layout)
            if limit and distance > limit:
                raise ValueError(
                    f"{spec.name} reads {name} {distance} rows away; the "
                    f"slab holds {limit} ghost rows"
                )
            entry = self._entry(tensor) or self._assume(tensor)
            states.append((tensor, layout, distance, limit, entry))
        self.mirror(tuple(t for t, _, _, _, entry in states if entry is None))
        if not fusable:
            return {**arguments, **placeholders}, None
        reads = []
        for tensor, layout, distance, limit, entry in states:
            source = None
            if entry is not None and entry.ring is not None:
                source = entry.ring
            elif entry is not None and entry.entry:
                source = _ENTRY
            reads.append((tensor, layout, source, consumes(layout, distance, limit)))
        site = _Site(
            "kernel",
            HaloArgument(quiet.words),
            reads=reads,
            publish=tuple(
                (bound[name], runtime.halo_layout(bound[name]))
                for name in spec.publishes
            ),
            writes=frozenset(
                _storage(bound[name])
                for name, access in spec.buffers.items()
                if buffer_access_semantics(access).writes
                and isinstance(bound.get(name), torch.Tensor)
            ),
            index=len(self.recorder.operators),
            name=spec.name,
        )
        self.log.append(site)
        fused = [(t, s, c) for t, _, s, c in reads if s is not None]

        def recorded() -> None:
            # The call is the last operator; it may write back the ghost
            # rows it reads from rings (it records them written). Reading
            # them all makes a ring field current; an assumed one stays
            # assumed (the replay sees where its ring is). A field it also
            # writes is dirty after it, or waits in its own ring when it
            # publishes it: its writes supersede what it read.
            operator = self.recorder.operators[-1]
            site.operator = operator
            from hydroforge.kernels.cuda import takes_halo

            if not takes_halo(operator.call.implementation):
                raise ValueError(
                    f"{spec.name} declares halo or publishes: on the cuda "
                    "backend it needs a CUDA kernel taking the halo argument"
                )
            site.empty = operator.call.empty
            if site.empty:
                if site.publish:
                    raise ValueError(
                        f"{spec.name} publishes but launches nothing on this "
                        "rank; kernels declaring publishes launch on every rank"
                    )
                # Its prologue runs in a book of its own.
                fields = tuple(t for t, *_ in reads) or tuple(
                    t for t, _ in site.publish
                )
                if fields:
                    site.carrier = self._exchange(fields[:8], phase="book")
                    site.carrier.carried = True
            for tensor, _, _ in fused:
                self._ghost(tensor, "copied")
            for tensor, source, consumed in fused:
                if _storage(tensor) in site.writes:
                    self.entries.pop(id(tensor), None)
                elif source is _ENTRY:
                    self._set(tensor, entry=True)
                else:
                    self._set(tensor, None if consumed else source)
            for tensor, _ in site.publish:
                self._set(tensor, site)

        self.written_back = tuple(t for t, _, _ in fused)
        return {**arguments, HALO_PARAMETER: site.argument}, recorded

    # ------------------------------------------------------------------ #
    def finish(self) -> None:
        """The recording ends: decide every count, credit and receipt (the
        module documentation), and fill in the recorded arguments."""

        halo = self.halo
        if halo is None:
            return
        # Contributions are summed by the end (later programs read owners),
        # and ghost rows a loop body assumed zero at entry are zero again.
        self._reduce(
            [g.tensor for g in self.ghosts.values() if g.state == "partial"],
            [
                t
                for storage, t in self.zeroed.items()
                if self.ghosts[storage].state == "copied"
            ],
        )
        # Fields assumed current at entry and left dirty: mirrored at the
        # end (those a ring carries are not dirty).
        self.mirror(tuple(t for t in self.assumed.values() if self._entry(t) is None))
        if not self.fusable:
            return
        accessed, exchanges = self.outside
        carry = (
            self.carried
            and not exchanges
            and not any(isinstance(e, _Event) and e.kind == "settle" for e in self.log)
        )
        # The loop's halo launches are this body's alone (its other
        # programs and nested loops exchange nothing): it may go uncredited.
        alone = carry
        if carry:
            # Rings the loop's other programs read: received at the end.
            self.resolve_ring(
                [
                    e.tensor
                    for e in list(self.entries.values())
                    if _storage(e.tensor) in accessed
                ]
            )
        else:
            self.resolve_ring()
        state, seen = CarriedState(), set()
        for _ in range(64):
            try:
                result = self._replay(state if carry else CarriedState())
            except _Insert as insert:
                self._insert(insert)
                continue
            left = result.state
            if not carry:
                if not left.items:
                    break
                # Received and counted at the end.
                for site, pending in left.items:
                    if pending:
                        self._exchange(
                            site.group.fields, phase="receive", receipt=(site, "current")
                        )
                self._book()
                continue
            if left == state:
                break
            if self._carriable(left) and left not in seen:
                seen.add(state)
                state = left
                continue
            carry = False
            self._drop_entry_receipts()
        else:
            raise RuntimeError("the halo schedule did not settle")
        self.state = result.state if carry else CarriedState()
        self.free = (
            alone
            and not _ALWAYS_CREDIT
            and slots_free(self.halo.plan, result, state if carry else CarriedState())
        )
        self._fill(result)

    def _carriable(self, state: CarriedState) -> bool:
        """Whether a loop can send ``state``'s messages before its first
        iteration: at most :data:`DEPTH` per pair (no credit comes back in
        between)."""

        from hydroforge.parallel.halo import DEPTH

        sites = [site for site, _ in state.items]
        if len(set(map(id, sites))) != len(sites):
            return False  # a message per iteration is left uncounted
        counts: dict[tuple[int, int], int] = {}
        for site in sites:
            for pair in message_pairs(self.halo.plan, site):
                counts[pair] = counts.get(pair, 0) + 1
        return all(n <= DEPTH for n in counts.values())

    def _drop_entry_receipts(self) -> None:
        """Without a carried state, receipts of entry messages receive
        nothing: they only count and credit."""

        for site in self.log:
            if isinstance(site, _Site) and site.receipt and site.receipt[1] == "entry":
                site.receipt = None
                site.phase = "book"

    # ------------------------------------------------------------------ #
    def _replay(self, state: CarriedState) -> _Replay:
        """Number every message, place counts and credits; raises
        :class:`_Insert` when receipts must be recorded first."""

        from hydroforge.parallel.halo import DEPTH

        plan = self.halo.plan
        me = plan.rank
        items: list[_Item] = []
        by_pair: dict[tuple[int, int], list[int]] = {}
        index: dict[tuple[int, tuple[int, int]], int] = {}  # (item, pair) -> on pair

        def add(item: _Item) -> int:
            position = len(items)
            items.append(item)
            for pair in item.pairs:
                listed = by_pair.setdefault(pair, [])
                index[(position, pair)] = len(listed)
                listed.append(position)
            return position

        def on(pair: tuple[int, int], upto: int) -> int:
            """Messages of ``pair`` among the first ``upto`` items."""

            return bisect.bisect_left(by_pair.get(pair, ()), upto)

        entry_items: dict[int, _Item] = {}
        for site, pending in state.items:
            item = _Item(site, set(pending), message_pairs(plan, site), entry=True)
            item.done = not item.pending
            entry_items[id(site)] = item
            add(item)
        counted = 0
        credited = min(state.credited, len(items))
        latest: dict[int, _Item] = {}
        result = _Replay({}, {}, state)
        peers = list(plan.neighbours[me])

        def prefix() -> int:
            done = credited
            while done < len(items) and items[done].done:
                done += 1
            return done

        def mine(start: int, stop: int) -> dict[int, int]:
            """Per peer, this rank's messages among items [start, stop)."""

            found = {}
            for peer in peers:
                n = on((peer, me), stop) - on((peer, me), start)
                if n:
                    found[peer] = n
            return found

        def offsets(item_position: int, base: int) -> dict[int, int]:
            item = items[item_position]
            return {
                s: index[(item_position, (s, r))] - on((s, r), base)
                for s, r in item.pairs
                if r == me
            }

        def deadline(site: Any, position: int) -> None:
            """Senders store message ``position`` only into a slot credited:
            the message DEPTH earlier on each pair must be credited by now."""

            for pair in items[position].pairs:
                m = index[(position, pair)]
                if m < DEPTH:
                    continue
                needed = by_pair[pair][m - DEPTH]
                if needed < credited:
                    continue
                waiting = [
                    item
                    for item in items[credited : needed + 1]
                    if not item.done and item.site.kind == "kernel"
                ]
                if not waiting:
                    raise RuntimeError(
                        "halo schedule: a message slot is never credited"
                    )
                raise _Insert(site, waiting)

        def drop(storages: frozenset[int], keep: set[int]) -> None:
            for position in range(credited, len(items)):
                item = items[position]
                if position in keep or item.done or not item.pending:
                    continue
                item.pending -= storages
                item.done = not item.pending

        for event in self.log:
            if isinstance(event, _Event):
                if event.kind == "access":
                    waiting = [
                        item
                        for item in items[credited:]
                        if item.entry and not item.done and item.pending & event.storages
                    ]
                    if waiting:
                        raise _Insert(event, waiting)
                elif counted < len(items):
                    raise _Insert(
                        event, [i for i in items[credited:] if not i.done], book=True
                    )
                continue
            site = event
            if site.carried:
                continue
            unfinished = [p for p in range(len(items)) if not items[p].done]
            if site.kind == "kernel":
                # What it reads from rings: the publication's message, or
                # one the previous iteration left pending.
                reads, reading = [], set()
                for tensor, layout, source, consumed in site.reads:
                    item_position = None
                    storage = _storage(tensor)
                    if isinstance(source, _Site):
                        item = latest.get(id(source))
                        if item is not None and storage in item.pending:
                            item_position = items.index(item)
                    elif source == _ENTRY:
                        for position in range(len(items)):
                            item = items[position]
                            if item.entry and not item.done and storage in item.pending:
                                item_position = position
                    reads.append((tensor, layout, item_position, consumed))
                    if item_position is not None:
                        reading.add(item_position)
                if reading and site.empty:
                    raise ValueError(
                        f"{site.name} reads rings but launches nothing on this rank"
                    )
                drop(site.writes, reading)
                start = prefix()
                commit = mine(credited, start)
                credits, credited = commit, start
                counts = {}
                if not reading:
                    counts, counted = mine(counted, start), start
                entries = [
                    (peer, credits.get(peer, 0), counts.get(peer, 0))
                    for peer in peers
                    if credits.get(peer) or counts.get(peer)
                ]
                tables = []
                for tensor, layout, item_position, _ in reads:
                    if item_position is None:
                        tables.append((tensor, layout, None, {}))
                    else:
                        item = items[item_position]
                        tables.append(
                            (tensor, layout, item.site.group, offsets(item_position, counted))
                        )
                        result.fused += 1
                result.kernel[id(site)] = (tables, entries)
                published = []
                if site.publish:
                    position = add(
                        _Item(
                            site,
                            {_storage(t) for t, _ in site.publish},
                            message_pairs(plan, site),
                        )
                    )
                    latest[id(site)] = items[position]
                    deadline(site, position)
                    published.append(position)
                for tensor, layout, item_position, consumed in reads:
                    if item_position is not None and consumed:
                        items[item_position].pending.discard(_storage(tensor))
                for position in reading:
                    item = items[position]
                    item.pending -= site.writes
                    item.done = not item.pending
                # A reader of every ghost row (a full-depth slab read) waits
                # for the message of each pair holding rows of its kind; a
                # partial one need not touch every pair's rows.
                waits = [
                    (p, rows_of(plan, items[p].pairs, layout))
                    for _, layout, p, consumed in reads
                    if p is not None and consumed
                ]
                result.trace.append(
                    (waits, published, [p for p in unfinished if items[p].done])
                )
                continue
            # An exchange: a book, a receipt of a pending message, or new
            # messages received at its end.
            target = None
            phase = site.phase
            if site.receipt is not None:
                published, generation = site.receipt
                item = (
                    entry_items.get(id(published))
                    if generation == "entry"
                    else latest.get(id(published))
                )
                if item is not None and not item.done:
                    target = items.index(item)
                else:
                    phase = "book"
            elif phase == "both":
                drop(site.writes, set())
            before_counted, before_credited = counted, credited
            start = credited = prefix()
            receiving: dict[int, int] = {}
            written = (1 << 8) - 1
            # Its receive waits for every message of the item it receives.
            waits, published = [], []
            if target is not None:
                item = items[target]
                fields = [_storage(t) for t in item.site.group.fields]
                written = sum(1 << k for k, s in enumerate(fields) if s in item.pending)
                receiving = offsets(target, counted)
                item.pending.clear()
                item.done = True
                result.receipts += 1
                waits.append((target, item.pairs))
            elif phase == "both":
                position = add(_Item(site, set(), message_pairs(plan, site)))
                deadline(site, position)
                receiving = offsets(position, counted)
                items[position].done = True
                waits.append((position, items[position].pairs))
                published.append(position)
                unfinished.append(position)
            result.trace.append(
                (waits, published, [p for p in unfinished if items[p].done])
            )
            end = prefix()
            # Peers it receives from: credits of what came before in the
            # prologue, the rest counted and credited as its receive ends.
            # Others: everything in the prologue (none of theirs waits for
            # this launch).
            prologue, settle, post = [], {}, {}
            for peer in peers:
                pair = (peer, me)
                if peer in receiving:
                    credit = on(pair, start) - on(pair, before_credited)
                    settle[peer] = on(pair, end) - on(pair, before_counted)
                    post[peer] = on(pair, end) - on(pair, start)
                    count = 0
                else:
                    credit = on(pair, end) - on(pair, before_credited)
                    count = on(pair, end) - on(pair, before_counted)
                if credit or count:
                    prologue.append((peer, credit, count))
            counted = credited = end
            result.exchange[id(site)] = (
                phase, written, receiving, settle, post, prologue
            )
        result.state = CarriedState(
            tuple((item.site, frozenset(item.pending)) for item in items[counted:]),
            credited - counted,
        )
        result.items, result.counted = items, counted
        return result

    # ------------------------------------------------------------------ #
    def _insert(self, insert: _Insert) -> None:
        """Record the receipts (and book) the replay asked for, before the
        site or event it named."""

        operators = self.recorder.operators
        anchor = insert.before
        at = anchor.index
        log_at = self.log.index(anchor)
        logged, before = len(self.log), len(operators)
        for item in insert.receipts:
            self._exchange(
                item.site.group.fields,
                phase="receive",
                receipt=(item.site, "entry" if item.entry else "current"),
            )
        if insert.book:
            fields = (
                insert.receipts[0].site.group.fields
                if insert.receipts
                else self._any_fields()
            )
            self._exchange(tuple(fields[:8]), phase="book")
        added = self.log[logged:]
        del self.log[logged:]
        moved = operators[before:]
        del operators[before:]
        operators[at:at] = moved
        for entry in self.log[log_at:]:
            entry.index += len(moved)
        position = at
        for site in added:
            site.index = position
            if site.operator is not None:
                position += 1
        self.log[log_at:log_at] = added
        if not added:
            raise RuntimeError("halo schedule: nothing to receive with")

    def _book(self) -> None:
        """Record a book (counts and credits only) at the end."""

        self._exchange(tuple(self._any_fields()[:8]), phase="book")

    def _any_fields(self) -> tuple[torch.Tensor, ...]:
        for site in self.log:
            if isinstance(site, _Site):
                fields = site.group.fields or tuple(t for t, *_ in site.reads)
                if fields:
                    return fields
        return ()

    def _fill(self, result: _Replay) -> None:
        """Write the replay's tables into the recorded arguments."""

        from hydroforge.kernels.halo import access_argument
        from hydroforge.parallel.halo import site_words

        halo = self.halo
        world = halo.plan.world
        free = self.free

        def uncredited(prologue: list) -> list:
            """A free body's prologue: counts only."""

            return [(p, 0, n) for p, _, n in prologue if n] if free else prologue

        for site in self.log:
            if not isinstance(site, _Site) or site.carried:
                continue
            if site.kind == "kernel":
                reads, commit = result.kernel[id(site)]
                commit = uncredited(commit)
                with _disable_current_modes():
                    argument = access_argument(
                        halo, halo.device, list(site.publish), reads, commit,
                        checks=not free,
                    )
                if site.publish:
                    halo.publications += 1
                if site.carrier is not None:
                    site.carrier.argument.words = site_words(
                        world, "book", 0, prologue=commit
                    )
                    continue
                site.argument.words, site.argument.keep = argument.words, argument.keep
                continue
            phase, written, receiving, settle, post, prologue = result.exchange[id(site)]
            site.argument.words = site_words(
                world,
                phase,
                written,
                receiving,
                settle,
                {} if free else post,
                uncredited(prologue),
                checks=not free,
            )
        halo.fused_reads += result.fused
        halo.receipts += result.receipts
        halo.credit_free += int(free)


def message_pairs(plan: Any, site: _Site) -> tuple[tuple[int, int], ...]:
    """Every (sender, receiver) pair of ranks carrying ``site``'s messages."""

    found = site.__dict__.get("pairs")
    if found is None:
        from hydroforge.parallel.halo import KINDS

        if site.kind == "kernel" or site.receipt is not None:
            layouts, mode = site.group.layouts, "mirror"
        else:
            layouts, mode = site.layouts, site.mode
        kinds = sorted({KINDS.index(layout.kind) for layout in layouts})
        if not kinds:
            return ()
        owners, holders = np.nonzero(plan.counts[kinds].sum(0))
        if mode == "mirror":
            found = tuple(zip(owners.tolist(), holders.tolist()))
        else:
            found = tuple(zip(holders.tolist(), owners.tolist()))
        site.__dict__["pairs"] = found
    return found


def rows_of(
    plan: Any, pairs: tuple[tuple[int, int], ...], layout: Any
) -> tuple[tuple[int, int], ...]:
    """The ``pairs`` whose messages hold rows of ``layout``'s kind."""

    from hydroforge.parallel.halo import KINDS

    counts = plan.counts[KINDS.index(layout.kind)]
    return tuple(pair for pair in pairs if counts[pair])


def slots_free(plan: Any, result: _Replay, state: CarriedState) -> bool:
    """Whether a loop body, replayed from ``state`` into ``result``, stores
    every message into a slot its receiver finished with, credits or not.

    Message ``n`` of a pair ``s -> r`` reuses the slot of message
    ``n - DEPTH``. It is free when, in a launch before the one storing ``n``,
    ``s`` waited for a message ``k`` of ``r -> s`` that ``r`` stored in a
    launch after the one finishing with ``n - DEPTH`` (launches of a rank
    run in order; the wait acquires what ``r`` released after storing
    ``k``). A launch's own waits prove nothing for its own stores: its
    blocks run in any order. Waits are those of receives (every message of
    the item) and of readers of every ghost row of a slab field; partial
    and partition key readers may skip a pair.

    The body is replayed for ``DEPTH + 3`` iterations (the entry messages
    of each the previous one's last); each iteration repeats the same
    launches, so the proof found for the last one, which looks back at most
    ``DEPTH + 1`` iterations, holds for every later one. Messages ``n <
    DEPTH`` of a pair reuse slots of messages sent before the loop: the loop
    waits for their credits at entry (:class:`LoopHalo`), then sends the
    entry messages (credited as ever)."""

    from hydroforge.parallel.halo import DEPTH

    items, counted, trace = result.items, result.counted, result.trace
    entries = len(state.items)
    fresh = len(items) - entries  # messages each iteration adds
    if not trace or fresh <= 0 or len(items) - counted != entries or counted < entries:
        return False

    def number(iteration: int, position: int) -> int:
        """The message of item ``position`` of an iteration, counted from
        the first entry message (an entry item is an item of the previous
        iteration)."""

        while position < entries and iteration > 0:
            iteration, position = iteration - 1, counted + position
        if position < entries:
            return position
        return entries + iteration * fresh + position - entries

    # Times: launches in order; the entry messages are sent at -1, and those
    # finished with at entry were finished before (-2).
    sent: dict[int, int] = {}
    finished: dict[int, int] = {}
    pairs: dict[int, tuple[tuple[int, int], ...]] = {}
    waited: dict[tuple[int, int], list[tuple[int, int]]] = {}  # pair -> (time, message)
    for position, (_, pending) in enumerate(state.items):
        sent[position] = -1
        pairs[position] = items[position].pairs
        if not pending:
            finished[position] = -2
    for iteration in range(DEPTH + 3):
        for j, (waits, published, done) in enumerate(trace):
            time = iteration * len(trace) + j
            for position in published:
                message = number(iteration, position)
                sent[message] = time
                pairs[message] = items[position].pairs
            for position, on in waits:
                for pair in on:
                    waited.setdefault(pair, []).append((time, number(iteration, position)))
            for position in done:
                finished.setdefault(number(iteration, position), time)
    messages: dict[tuple[int, int], list[int]] = {}
    for message in sorted(sent):
        for pair in pairs[message]:
            messages.setdefault(pair, []).append(message)
    for (s, r), listed in messages.items():
        # When s waited for r's messages, and the latest sending time of
        # any message waited for so far.
        times, latest, best = [], [], -3
        for time, message in sorted(waited.get((r, s), ())):
            best = max(best, sent.get(message, -3))
            times.append(time)
            latest.append(best)
        for n in range(DEPTH, len(listed)):
            start = sent[listed[n]]
            if start < 0:
                continue  # an entry message: sent before the loop, credited
            reused = finished.get(listed[n - DEPTH])
            before = bisect.bisect_left(times, start)
            if reused is None or before == 0 or latest[before - 1] <= reused:
                return False
    return True


def written_storages(operators: Any) -> frozenset[int]:
    """The storages ``operators`` write."""

    return frozenset(
        _storage(written)
        for operator in operators
        for written in getattr(operator, "writes", ())
        if isinstance(written, torch.Tensor)
    )


def neighbourhood(operators: Any, halo: Any) -> tuple[frozenset[int], bool]:
    """The storages ``operators`` access, and whether they exchange halos
    (an exchange or a declared kernel)."""

    accessed = frozenset(
        _storage(tensor)
        for operator in operators
        for tensor in (*getattr(operator, "reads", ()), *getattr(operator, "writes", ()))
        if isinstance(tensor, torch.Tensor)
    )
    exchanges = False
    for operator in operators:
        registry = getattr(operator, "registry", None)
        if registry is None:
            continue
        spec = registry.spec
        if (halo is not None and registry is halo.registry) or spec.halo or spec.publishes:
            exchanges = True
    return accessed, exchanges


class LoopHalo:
    """What a loop does around its runs (outside any recording): before the
    first iteration it mirrors the fields its body assumes current and sends
    the messages its body finds pending (:class:`CarriedState`); after the
    last it receives those still pending and counts and credits them all.
    A body storing without credits (``free``, :func:`slots_free`) first
    waits until every message sent before it is credited, and afterwards
    sets the credits its iterations skipped (``anchor``: fields to launch
    with when it carries nothing). The ghost rows of the fields a body
    contributes to (``zero``) are zeroed before its first iteration.
    Launches are bound and compiled once: an eager exchange would bind its
    call again on every run, host work each step would then wait for."""

    def __init__(
        self,
        fields: tuple[torch.Tensor, ...] = (),
        state: CarriedState = CarriedState(),
        *,
        free: bool = False,
        anchor: tuple[torch.Tensor, ...] = (),
        zero: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self.fields = fields
        self.state = state
        self.free = free
        self.anchor = anchor
        self.zero = zero
        self.launches: dict[str, tuple[Any, ...]] = {}

    @property
    def carries(self) -> bool:
        return bool(self.state.items)

    def enter(self, execution: Any) -> None:
        self._run(execution, "enter")

    def leave(self, execution: Any) -> None:
        self._run(execution, "leave")

    def refresh(self, execution: Any) -> None:
        """After a capture warmed the body up: its rings hold what the
        warm-ups published, not what the restored fields hold. Count and
        credit them unread, then send them again."""

        if self.carries:
            self._run(execution, "drop")
            self._run(execution, "enter")

    def _run(self, execution: Any, which: str) -> None:
        if not (self.fields or self.state.items or self.free or self.zero):
            return
        launches = self.launches.get(which)
        if launches is None:
            launches = self.launches[which] = tuple(
                execution.kernel_binding.bind(execution.runtime.halo.registry, a).compile()
                for a in self._arguments(execution, which)
            )
        for launch in launches:
            launch()

    def _arguments(self, execution: Any, which: str) -> list[dict]:
        from hydroforge.parallel.halo import site_words

        runtime = execution.runtime
        halo = runtime.halo
        plan = halo.plan
        me, world = plan.rank, plan.world
        found: list[dict] = []

        def exchange(site, phase, written=0, skip=None, prologue=()) -> None:
            fields = site.group.fields if site is not None else self.anchor
            mode = "mirror" if site is None or site.kind == "kernel" else site.mode
            arguments = halo.exchange_arguments(
                fields,
                [runtime.halo_layout(t) for t in fields],
                mode,
                site=HaloArgument(
                    site_words(world, phase, written, skip, prologue=prologue)
                ),
            )
            if arguments is not None:
                found.append(arguments)

        if which == "enter":
            for tensor in self.zero:
                arguments = halo.exchange_arguments(
                    (), [], "accumulate", tensor, runtime.halo_layout(tensor)
                )
                if arguments is not None:
                    found.append(arguments)
            for start in range(0, len(self.fields), 8):
                chunk = self.fields[start : start + 8]
                arguments = halo.exchange_arguments(
                    chunk, [runtime.halo_layout(t) for t in chunk], "mirror"
                )
                if arguments is not None:
                    found.append(arguments)
            if self.free:
                # The body reuses slots of messages sent before it (these
                # mirrors' included) without their credits.
                exchange(None, "drain")
        items = self.state.items
        if not items:
            if self.free and which == "leave":
                exchange(None, "flush")
            return found
        pairs = [message_pairs(plan, site) for site, _ in items]
        peers = sorted({s for listed in pairs for s, r in listed if r == me})

        def mine(stop: int, start: int = 0) -> dict[int, int]:
            return {
                p: sum((p, me) in listed for listed in pairs[start:stop]) for p in peers
            }

        credited = self.state.credited
        if which == "enter":
            for site, _ in items:
                exchange(site, "send")
            if credited and not self.free:
                credit = mine(credited)
                exchange(
                    items[0][0], "book", prologue=[(p, n, 0) for p, n in credit.items() if n]
                )
            return found
        if which == "leave":
            for position, (site, pending) in enumerate(items):
                if not pending:
                    continue
                fields = [_storage(t) for t in site.group.fields]
                written = sum(1 << k for k, s in enumerate(fields) if s in pending)
                skip = {p: n for p, n in mine(position).items() if (p, me) in pairs[position]}
                exchange(site, "receive", written, skip)
        counts = mine(len(items))
        if self.free:
            # Count them, then set every credit the body skipped.
            exchange(
                items[0][0], "flush", prologue=[(p, 0, counts[p]) for p in peers if counts[p]]
            )
            return found
        credits = mine(len(items), credited)
        exchange(
            items[0][0],
            "book",
            prologue=[(p, credits[p], counts[p]) for p in peers if counts[p] or credits[p]],
        )
        return found


__all__ = [
    "CarriedState",
    "HaloSchedule",
    "LoopHalo",
    "message_pairs",
    "neighbourhood",
    "rows_of",
    "slots_free",
    "written_storages",
]

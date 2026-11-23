# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Every cross-rank agreement of one model runtime.

``LocalChannel`` serves a single-rank runtime: each agreement is decided
locally and no distributed code runs.  ``ProcessGroupChannel`` exchanges one
fixed wire row per agreement,
``(sequence, kind, a, b, c, begin_digest, digest, flag)``, under a single
sequence shared by public transactions and managed-step events, so any
divergence in the order of agreements is detected.

* A public transaction publishes the digest of ``(sequence, phase,
  signature)``; ``flag`` requests the full record exchange that a failure,
  a payload or any difference needs.
* A warm managed step stages its invocation preflight and ``BEGIN`` metadata
  without a network round-trip and carries both on its first wire event
  (``STEP_FINALIZED`` for steps without host collectives, so each warm step
  costs one handshake).  A failed rank publishes kind ``-1``.

Model bodies outside managed steps (``initialize_model_state`` and
between-steps methods) use the same events for their collectives and close
with one more (:meth:`ProcessGroupChannel.close_body`), so a rank's failure
reaches every peer at the collective or the close it waits in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast

import torch
import torch.distributed as dist

from hydroforge.core.devices import devices_match
from hydroforge.core.errors import (
    ResourceCleanupError,
    distributed_failure_error,
    failure_description,
)
from hydroforge.core.identity import digest63
from hydroforge.parallel.launch import communication_backend

Failures = tuple[dict[str, str] | None, ...]

# Collective event kinds carry this bit for ensemble-scope collectives.
ENSEMBLE_COLLECTIVE_FLAG = 1 << 60
_DIGEST_MASK = (1 << 63) - 1
_NO_SIGNATURE = (0, 0, 0)
_FAILED = -1
_DETAILS = 1
_PREFLIGHT = 2


def _full_records_ready(rows: Any) -> bool:
    return all(row[1] == StepEvent.TRANSACTION or row[7] & _PREFLIGHT for row in rows)


class PeerFailureError(RuntimeError):
    """A peer rank published a failure where this rank waited for it; the
    peer's own error describes it."""


class StepEvent(IntEnum):
    """Managed-step sequence points; collectives use kinds from 10 upward."""

    TRANSACTION = 0
    BEGIN = 1
    STEP_FINALIZED = 4
    BODY_COMPLETE = 5
    PREDICATE_SYNC = 6
    OUTPUT_COMMITTED = 7
    PREDICATE_IDLE = 8
    EXCHANGE_SYNC = 9


@dataclass(frozen=True, slots=True)
class StagedPreflight:
    """An invocation check published inside the next managed-step event."""

    error: BaseException | None
    phase: str
    scope: str
    signature: Any = None
    digest: int | None = None

    def resolved_signature(self) -> tuple[Any, ...] | None:
        value = self.signature
        return value() if callable(value) else value


class LocalChannel:
    """Single-rank channel: every agreement is local."""

    distributed = False
    rejection: BaseException | None = None

    def preflight(
        self,
        error: BaseException | None,
        *,
        phase: str,
        scope: str,
        signature: tuple[Any, ...] | None = None,
    ) -> None:
        del phase, scope, signature
        if error is not None:
            raise error

    def gather(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
    ) -> Failures:
        del phase, signature
        return (None if error is None else failure_description(error),)

    def exchange(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
        payload: Any = None,
    ) -> tuple[Failures, tuple[Any, ...]]:
        return self.gather(error, phase=phase, signature=signature), (payload,)

    def open_step(self, preflight: StagedPreflight | None = None) -> None:
        del preflight

    def stage_begin(self, signature: tuple[int, int, int]) -> None:
        del signature

    def event(self, kind: int, signature: tuple[int, int, int] = _NO_SIGNATURE) -> None:
        del kind, signature

    def predicate_continue(
        self, continuing: bool, *, coupled: bool = False
    ) -> tuple[bool, bool]:
        del coupled
        return bool(continuing), bool(continuing)

    def predicate_wait(self) -> bool:
        return False

    def exchange_sync(self, level: int, tag: int) -> tuple[int, int]:
        del tag
        return level, level

    def abort(self) -> None:
        pass

    def open_body(self) -> None:
        pass

    def close_body(self, error: BaseException | None) -> BaseException | None:
        return error


def require_process_group(what: str) -> None:
    """Reject distributed work before ``torch.distributed`` is initialized."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{what} requires an initialized torch.distributed process group"
        )


_CONTROL_PLANE: tuple[Any, tuple[Any, bool]] | None = None


def distributed_control_plane() -> tuple[Any, bool]:
    """Return the rank-handshake group and whether it carries CPU tensors.

    Handshakes stay off accelerator streams: a Gloo default group is reused
    and an NCCL/XCCL default group gets one dedicated Gloo group. Creation is
    lazy at the first multi-rank exchange, which every rank reaches in the
    same order as ``dist.new_group`` requires.
    """

    global _CONTROL_PLANE
    world = dist.group.WORLD
    cached = _CONTROL_PLANE
    if cached is not None and cached[0] is world:
        return cached[1]
    default_backend = dist.get_backend()
    try:
        # A per-device default group ("cpu:gloo,cuda:nccl") carries CPU
        # tensors through its CPU entry.
        host_backend = communication_backend(default_backend, device_type="cpu")
    except ValueError:
        if ":" not in str(default_backend):
            raise
        host_backend = None  # a per-device list without a CPU entry
    if host_backend == "gloo":
        plane: tuple[Any, bool] = (None, True)
    elif dist.is_gloo_available():
        plane = (dist.new_group(backend="gloo"), True)
    else:
        plane = (None, False)
    _CONTROL_PLANE = (world, plane)
    return plane


class ProcessGroupChannel:
    """Rank agreements of one multi-rank runtime over torch.distributed."""

    distributed = True

    def __init__(self, world_size: int, device: torch.device, mesh: Any) -> None:
        self.world_size = world_size
        self.device = device
        self.mesh = mesh
        self.sequence = 0
        self.rejection: BaseException | None = None
        self._group: Any = None
        self._source: torch.Tensor | None = None
        self._staging = torch.empty(8, dtype=torch.int64)
        self._wire = self._staging.numpy()
        self._gathered: torch.Tensor | None = None
        self._outputs: list[torch.Tensor] = []
        self._preflight: StagedPreflight | None = None
        self._begin_digest: int = 0
        self._terminal = False
        self._own_spatial_ranks: tuple[int, ...] | None = None

    def _spatial_ranks(self) -> tuple[int, ...]:
        """World ranks of this rank's spatial group."""

        if self._own_spatial_ranks is None:
            rank = dist.get_rank()
            groups = (
                (tuple(range(self.world_size)),)
                if self.mesh is None
                else self.mesh.rank_groups("spatial")
            )
            self._own_spatial_ranks = next(g for g in groups if rank in g)
        return self._own_spatial_ranks

    def _rows(
        self,
        kind: int,
        signature: tuple[int, int, int],
        begin_digest: int,
        digest: int,
        flag: int,
    ) -> list[list[int]]:
        """Exchange one wire row under the next sequence number."""

        group, host_plane = distributed_control_plane()
        device = torch.device("cpu") if host_plane else self.device
        if (
            self._gathered is None
            or self._group is not group
            or not devices_match(self._gathered.device, device)
        ):
            self._gathered = torch.empty(
                (self.world_size, 8), dtype=torch.int64, device=device
            )
            self._outputs = list(self._gathered.unbind(0))
            self._source = (
                self._staging
                if host_plane
                else torch.empty_like(self._staging, device=device)
            )
            self._group = group
        self._wire[:] = (self.sequence, kind, *signature, begin_digest, digest, flag)
        if self._source is not self._staging:
            self._source.copy_(self._staging)
        dist.all_gather(self._outputs, self._source, group=group)
        self.sequence += 1
        return self._gathered.cpu().tolist()

    # ------------------------------------------------------------------ public
    def preflight(
        self,
        error: BaseException | None,
        *,
        phase: str,
        scope: str,
        signature: tuple[Any, ...] | None = None,
    ) -> None:
        """Coordinate an entry check before its guarded side effects begin."""

        failures = self.gather(error, phase=phase, signature=signature)
        if any(failure is not None for failure in failures):
            if error is not None:
                raise error
            raise distributed_failure_error(scope, failures)

    def gather(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
    ) -> Failures:
        """Publish one phase-tagged public transaction result to every rank."""

        failures, _payloads = self.exchange(error, phase=phase, signature=signature)
        return failures

    def exchange(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
        payload: Any = None,
    ) -> tuple[Failures, tuple[Any, ...]]:
        """Exchange one tagged transaction record and optional phase payload."""

        require_process_group("a multi-rank model transaction")
        if type(phase) is not str or not phase:
            raise RuntimeError(
                "distributed public transaction phase must be a non-empty string"
            )
        sequence = self.sequence
        full = error is not None or payload is not None
        rows = self._rows(
            StepEvent.TRANSACTION,
            _NO_SIGNATURE,
            0,
            digest63((sequence, phase, signature)),
            int(full),
        )
        if not rows[0][7] and all(row == rows[0] for row in rows):
            empty = (None,) * self.world_size
            return empty, empty
        if not _full_records_ready(rows):
            self._terminal = True
            failed = tuple(rank for rank, row in enumerate(rows) if row[1] == _FAILED)
            if failed:
                raise PeerFailureError(
                    f"model call failed on peer rank(s) before {phase!r}: {failed}"
                )
            raise RuntimeError(
                f"distributed transaction and managed-step events differ across ranks: {rows}"
            )
        return self._complete(
            sequence, error, phase=phase, signature=signature, payload=payload
        )

    def _complete(
        self,
        sequence: int,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None,
        payload: Any = None,
    ) -> tuple[Failures, tuple[Any, ...]]:
        """Exchange full records of one transaction whose rows disagreed."""

        local = (
            sequence,
            phase,
            signature,
            None if error is None else failure_description(error),
            payload,
        )
        observed: list[Any] = [None] * self.world_size
        dist.all_gather_object(observed, local, group=self._group)
        if any(
            not isinstance(value, tuple)
            or len(value) != 5
            or type(value[0]) is not int
            or type(value[1]) is not str
            for value in observed
        ):
            raise RuntimeError(
                "distributed public transaction protocol received a malformed "
                f"record: {observed!r}"
            )
        identities = tuple((value[0], value[1]) for value in observed)
        if len(set(identities)) != 1:
            raise RuntimeError(
                f"distributed public phase mismatch across ranks: {identities!r}"
            )
        failures = tuple(value[3] for value in observed)
        if not any(failure is not None for failure in failures):
            signatures = tuple(value[2] for value in observed)
            if any(value != signatures[0] for value in signatures[1:]):
                raise RuntimeError(
                    "distributed public transaction inputs differ across ranks "
                    f"during {phase!r}: {signatures!r}"
                )
        return (
            cast(Failures, failures),
            tuple(value[4] for value in observed),
        )

    # ------------------------------------------------------------ managed step
    def open_step(self, preflight: StagedPreflight | None = None) -> None:
        """Start the event sequence of one managed-step invocation."""

        require_process_group("a multi-rank managed step")
        self._terminal = False
        self._preflight = preflight
        self._begin_digest = 0
        self.rejection = None

    def stage_begin(self, signature: tuple[int, int, int]) -> None:
        """Stage ``BEGIN`` metadata to ride on the step's first wire event."""

        a, b, c = signature
        self._begin_digest = (
            1
            + (
                (a * 0x9E3779B97F4A7C15)
                ^ ((b + 2) * 0xBF58476D1CE4E5B9)
                ^ ((c + 3) * 0x94D049BB133111EB)
            )
        ) & _DIGEST_MASK

    def event(self, kind: int, signature: tuple[int, int, int] = _NO_SIGNATURE) -> None:
        """Match the next rank event or propagate a peer's local failure."""

        self._event(kind, signature)

    def _staged_rows(
        self, kind: int, signature: tuple[int, int, int]
    ) -> tuple[int, list[list[int]]]:
        """Exchange one row carrying what the step staged for its first event."""

        preflight, self._preflight = self._preflight, None
        begin_digest, self._begin_digest = self._begin_digest, 0
        sequence = self.sequence
        digest = flag = 0
        if preflight is not None:
            if preflight.digest is not None:
                digest = (
                    (sequence * 0x9E3779B97F4A7C15) ^ preflight.digest
                ) & _DIGEST_MASK
            else:
                digest = digest63(
                    (sequence, preflight.phase, preflight.resolved_signature())
                )
            flag = _PREFLIGHT | int(preflight.error is not None)
        rows = self._rows(kind, signature, begin_digest, digest, flag)
        ready = _full_records_ready(rows)
        if not ready and any(
            row[1] == StepEvent.TRANSACTION or row[7] & _PREFLIGHT for row in rows
        ):
            self._terminal = True
            raise RuntimeError(
                f"distributed transaction and managed-step events differ across ranks: {rows}"
            )
        if preflight is not None and (
            len({(row[6], row[7]) for row in rows}) != 1 or rows[0][7] & _DETAILS
        ):
            self._resolve_preflight(sequence, preflight)
        return sequence, rows

    def _raise_failed(self, observed: Any, sequence: int) -> None:
        failed_ranks = tuple(
            rank for rank, value in enumerate(observed) if value[1] == _FAILED
        )
        if failed_ranks:
            self._terminal = True
            raise PeerFailureError(
                "model call failed on peer rank(s) before distributed "
                f"event {sequence}: {failed_ranks}"
            )

    def _diverged(self, observed: Any) -> RuntimeError:
        self._terminal = True
        return RuntimeError(
            "managed-step distributed event or collective ABI differs "
            f"across ranks: {observed}"
        )

    def _collectives_match(self, observed: Any) -> bool:
        """Whether the running ranks issue one collective; a finished spatial
        group of a predicate loop idles through a spatial-scope one."""

        running = tuple(v for v in observed if v[1] != StepEvent.PREDICATE_IDLE)
        if not running or len({(v[0], v[1], v[5]) for v in running}) != 1:
            return False
        kind = running[0][1]
        if self.mesh is None:
            return len(running) == len(observed) and len(set(running)) == 1
        scope = "ensemble" if kind & ENSEMBLE_COLLECTIVE_FLAG else "spatial"
        if scope == "ensemble" and len(running) != len(observed):
            return False  # a coupled body never leaves a partition idle
        idle = StepEvent.PREDICATE_IDLE
        if any(
            len({observed[rank][1] == idle for rank in ranks}) != 1
            for ranks in self.mesh.rank_groups("spatial")
        ):
            return False
        return all(
            len({observed[r][2:5] for r in ranks if observed[r][1] != idle}) <= 1
            for ranks in self.mesh.rank_groups(scope)
        )

    def _predicate_round(
        self, rows: list[list[int]], sequence: int
    ) -> tuple[bool, bool] | None:
        """``(this spatial group continues, any group does)`` when every
        running rank reached this loop's decision, else ``None``."""

        observed = tuple(tuple(row[:6]) for row in rows)
        self._raise_failed(observed, sequence)
        idle = StepEvent.PREDICATE_IDLE
        running = tuple(v for v in observed if v[1] != idle)
        if not running:
            raise self._diverged(observed)
        if running[0][1] != StepEvent.PREDICATE_SYNC:
            # A running group's body collective: only an idle rank gets here.
            if rows[dist.get_rank()][1] != idle or running[0][1] < 10:
                raise self._diverged(observed)
            if not self._collectives_match(observed):
                raise self._diverged(observed)
            return None
        if len({(v[0], v[1], v[3], v[4], v[5]) for v in running}) != 1 or any(
            len({observed[rank][1] == idle for rank in ranks}) != 1
            for ranks in (() if self.mesh is None else self.mesh.rank_groups("spatial"))
        ):
            raise self._diverged(observed)
        group = any(observed[rank][2] != 0 for rank in self._spatial_ranks())
        return group, any(value[2] != 0 for value in running)

    def predicate_continue(
        self, continuing: bool, *, coupled: bool = False
    ) -> tuple[bool, bool]:
        """Decide one iteration of a rank-synchronized predicate loop.

        Returns whether this rank's spatial group iterates again and whether
        any group does. A group that stops calls :meth:`predicate_wait`
        until no group continues, so it still sees a peer's failure.

        ``coupled``: the body reduces across ensemble ranks, so a member that
        stopped would leave that reduction without a defined contribution.
        Spatial groups that disagree raise on every rank before another
        iteration writes any state.
        """

        if self._terminal:
            return False, False
        sequence, rows = self._staged_rows(
            StepEvent.PREDICATE_SYNC,
            (int(bool(continuing)), 0, 0),
        )
        decided = self._predicate_round(rows, sequence)
        if decided is None:
            raise self._diverged(tuple(tuple(row[:6]) for row in rows))
        if coupled and self.mesh is not None:
            decisions = {
                any(rows[rank][2] != 0 for rank in ranks)
                for ranks in self.mesh.rank_groups("spatial")
            }
            if len(decisions) != 1:
                self._terminal = True
                raise RuntimeError(
                    "a predicate loop whose body reduces across the ensemble "
                    "needs one exit decision: ensemble partitions disagree on "
                    "continuing. Reduce the continue flag across the ensemble "
                    "(all_reduce_(iteration.continue_flag, reduction='max', "
                    "scope='ensemble')) or move the reduction out of the loop"
                )
        return decided

    def predicate_wait(self) -> bool:
        """Idle one round of a loop this spatial group finished; return
        whether any group iterates again."""

        while not self._terminal:
            sequence = self.sequence
            rows = self._rows(StepEvent.PREDICATE_IDLE, _NO_SIGNATURE, 0, 0, 0)
            decided = self._predicate_round(rows, sequence)
            if decided is not None:
                return decided[1]
        return False

    def exchange_sync(self, level: int, tag: int) -> tuple[int, int]:
        """Agree what one ``exchange_`` call does before it communicates.

        ``level``: 0 replay, 1 recapture, 2 rebuild. Returns the highest
        level of this rank's spatial group and of every rank.
        """

        if self._terminal:
            raise RuntimeError("exchange_ after a failed distributed event")
        sequence, rows = self._staged_rows(
            StepEvent.EXCHANGE_SYNC, (int(level), tag, 0)
        )
        observed = tuple(tuple(row[:6]) for row in rows)
        self._raise_failed(observed, sequence)
        if len({(v[0], v[1], v[3], v[4], v[5]) for v in observed}) != 1:
            raise self._diverged(observed)
        return (
            max(observed[rank][2] for rank in self._spatial_ranks()),
            max(value[2] for value in observed),
        )

    def abort(self) -> None:
        """Publish a caught local failure at the next synchronization event."""

        self._event(_FAILED, _NO_SIGNATURE)

    def open_body(self) -> None:
        """Start a model body (initialization or a between-steps method)
        whose collectives are events of this channel."""

        self.open_step()

    def close_body(self, error: BaseException | None) -> BaseException | None:
        """End a model body on every rank at once.

        A rank that failed publishes its failure; one that finished publishes
        the end of its body. Either reaches a peer still waiting in a
        collective or at its own end, so every rank leaves together. Returns
        this rank's ``error``, or the peer failure (or divergence) the closing
        event revealed; ``None`` exactly when every rank finished the body.
        """

        if error is not None:
            try:
                self.abort()
            except BaseException as coordination:
                return ResourceCleanupError(
                    "model body failure propagation", (error, coordination)
                )
            return error
        if self._terminal:
            # A collective of this body already failed across ranks (and
            # the body went on): there is no aligned event left to publish.
            return RuntimeError("a collective of this model body failed on some rank")
        try:
            self._event(StepEvent.BODY_COMPLETE, _NO_SIGNATURE)
        except BaseException as peer:
            return peer
        return None

    def _event(self, kind: int, signature: tuple[int, int, int]) -> None:
        if self._terminal:
            return
        preflight, self._preflight = self._preflight, None
        begin_digest, self._begin_digest = self._begin_digest, 0
        sequence = self.sequence
        digest = flag = 0
        if preflight is not None:
            if preflight.digest is not None:
                digest = (
                    (sequence * 0x9E3779B97F4A7C15) ^ preflight.digest
                ) & _DIGEST_MASK
            else:
                digest = digest63(
                    (sequence, preflight.phase, preflight.resolved_signature())
                )
            flag = _PREFLIGHT | int(preflight.error is not None)
        rows = self._rows(kind, signature, begin_digest, digest, flag)
        ready = _full_records_ready(rows)
        if not ready and any(
            row[1] == StepEvent.TRANSACTION or row[7] & _PREFLIGHT for row in rows
        ):
            self._terminal = True
            if kind == _FAILED:
                return
            raise RuntimeError(
                f"distributed transaction and managed-step events differ across ranks: {rows}"
            )
        if preflight is not None and (
            len({(row[6], row[7]) for row in rows}) != 1 or rows[0][7] & _DETAILS
        ):
            self._resolve_preflight(sequence, preflight)
        observed = tuple(tuple(row[:6]) for row in rows)
        failed_ranks = tuple(
            rank for rank, value in enumerate(observed) if value[1] == _FAILED
        )
        matching = len(set(observed)) == 1
        if self.mesh is not None and kind >= 10 and not failed_ranks:
            matching = self._collectives_match(observed)
        if failed_ranks or not matching:
            self._terminal = True
            if kind == _FAILED:
                return
            if failed_ranks:
                raise PeerFailureError(
                    "model call failed on peer rank(s) before distributed "
                    f"event {sequence}: {failed_ranks}"
                )
            raise RuntimeError(
                "managed-step distributed event or collective ABI differs "
                f"across ranks: {observed}"
            )

    def _resolve_preflight(self, sequence: int, preflight: StagedPreflight) -> None:
        """Exchange full preflight records after a digest or failure differs.

        Every rank reaches this from the same gathered rows. A rejection is
        raised before any rank commits the step, so it is not a poisoning
        step failure.
        """

        self._terminal = True
        try:
            failures, _payloads = self._complete(
                sequence,
                preflight.error,
                phase=preflight.phase,
                signature=preflight.resolved_signature(),
            )
            if any(failure is not None for failure in failures):
                if preflight.error is not None:
                    raise preflight.error
                raise distributed_failure_error(preflight.scope, failures)
        except BaseException as rejection:
            self.rejection = rejection
            raise
        self._terminal = False

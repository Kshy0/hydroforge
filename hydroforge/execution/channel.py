"""Every cross-rank agreement of one model runtime.

``LocalChannel`` serves a single-rank runtime: each agreement is decided
locally and no distributed code runs.  ``ProcessGroupChannel`` exchanges one
fixed wire row per agreement, ``(sequence, kind, a, b, c, digest, flag)``,
under a single sequence shared by public transactions and managed-step
events, so any divergence in the order of agreements is detected.

* A public transaction publishes the digest of ``(sequence, phase,
  signature)``; ``flag`` requests the full record exchange that a failure,
  a payload or any difference needs.
* A managed-step event publishes its kind and a three-int signature; the
  first event of a warm step carries the invocation preflight in ``digest``
  and ``flag``.  A failed rank publishes kind ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast

import torch
import torch.distributed as dist

from hydroforge.core.devices import devices_match
from hydroforge.core.errors import distributed_failure_error, failure_description
from hydroforge.core.identity import digest63
from hydroforge.parallel.launch import communication_backend

Failures = tuple[dict[str, str] | None, ...]

# Collective event kinds carry this bit for ensemble-scope collectives.
ENSEMBLE_COLLECTIVE_FLAG = 1 << 60
_NO_SIGNATURE = (0, 0, 0)
_FAILED = -1
_DETAILS = 1
_PREFLIGHT = 2


def _full_records_ready(rows: Any) -> bool:
    return all(row[1] == StepEvent.TRANSACTION or row[6] & _PREFLIGHT for row in rows)


class StepEvent(IntEnum):
    """Managed-step sequence points; collectives use kinds from 10 upward."""

    TRANSACTION = 0
    BEGIN = 1
    SUBSTEP = 2
    USER_STEP_COMPLETE = 3
    STEP_FINALIZED = 4


@dataclass(frozen=True, slots=True)
class StagedPreflight:
    """An invocation check published inside the next managed-step event."""

    error: BaseException | None
    phase: str
    scope: str
    signature: tuple[Any, ...] | None


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

    def event(self, kind: int, signature: tuple[int, int, int] = _NO_SIGNATURE) -> None:
        del kind, signature

    def abort(self) -> None:
        pass


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
    if communication_backend(dist.get_backend()) == "gloo":
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
        self._staging = torch.empty(7, dtype=torch.int64)
        self._wire = self._staging.numpy()
        self._gathered: torch.Tensor | None = None
        self._outputs: list[torch.Tensor] = []
        self._preflight: StagedPreflight | None = None
        self._terminal = False

    @staticmethod
    def _require_process_group(what: str) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                f"{what} require an initialized torch.distributed process group"
            )

    def _rows(
        self, kind: int, signature: tuple[int, int, int], digest: int, flag: int
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
                (self.world_size, 7), dtype=torch.int64, device=device
            )
            self._outputs = list(self._gathered.unbind(0))
            self._source = (
                self._staging
                if host_plane
                else torch.empty_like(self._staging, device=device)
            )
            self._group = group
        self._wire[:] = (self.sequence, kind, *signature, digest, flag)
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

        self._require_process_group("multi-rank model transactions")
        if type(phase) is not str or not phase:
            raise RuntimeError(
                "distributed public transaction phase must be a non-empty string"
            )
        sequence = self.sequence
        full = error is not None or payload is not None
        rows = self._rows(
            StepEvent.TRANSACTION,
            _NO_SIGNATURE,
            digest63((sequence, phase, signature)),
            int(full),
        )
        if not rows[0][6] and all(row == rows[0] for row in rows):
            empty = (None,) * self.world_size
            return empty, empty
        if not _full_records_ready(rows):
            self._terminal = True
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

        self._require_process_group("multi-rank managed steps")
        self._terminal = False
        self._preflight = preflight
        self.rejection = None

    def event(self, kind: int, signature: tuple[int, int, int] = _NO_SIGNATURE) -> None:
        """Match the next rank event or propagate a peer's local failure."""

        self._event(kind, signature)

    def abort(self) -> None:
        """Publish a caught local failure at the next synchronization event."""

        self._event(_FAILED, _NO_SIGNATURE)

    def _event(self, kind: int, signature: tuple[int, int, int]) -> None:
        if self._terminal:
            return
        preflight, self._preflight = self._preflight, None
        sequence = self.sequence
        digest = flag = 0
        if preflight is not None:
            digest = digest63((sequence, preflight.phase, preflight.signature))
            flag = _PREFLIGHT | int(preflight.error is not None)
        rows = self._rows(kind, signature, digest, flag)
        ready = _full_records_ready(rows)
        if not ready and any(
            row[1] == StepEvent.TRANSACTION or row[6] & _PREFLIGHT for row in rows
        ):
            self._terminal = True
            if kind == _FAILED:
                return
            raise RuntimeError(
                f"distributed transaction and managed-step events differ across ranks: {rows}"
            )
        if preflight is not None and (
            len({tuple(row[5:]) for row in rows}) != 1 or rows[0][6] & _DETAILS
        ):
            self._resolve_preflight(sequence, preflight)
        observed = tuple(tuple(row[:5]) for row in rows)
        failed_ranks = tuple(
            rank for rank, value in enumerate(observed) if value[1] == _FAILED
        )
        matching = len(set(observed)) == 1
        mesh = self.mesh
        if mesh is not None and kind >= 10 and not failed_ranks:
            scope = "ensemble" if kind & ENSEMBLE_COLLECTIVE_FLAG else "spatial"
            matching = len({value[:2] for value in observed}) == 1 and all(
                len({observed[rank][2:] for rank in ranks}) == 1
                for ranks in mesh.rank_groups(scope)
            )
        if failed_ranks or not matching:
            self._terminal = True
            if kind == _FAILED:
                return
            if failed_ranks:
                raise RuntimeError(
                    "managed step failed on peer rank(s) before distributed "
                    f"event {sequence}: {failed_ranks}"
                )
            raise RuntimeError(
                "managed-step distributed event or collective ABI differs "
                f"across ranks: {observed}"
            )

    def _resolve_preflight(self, sequence: int, preflight: StagedPreflight) -> None:
        """Exchange full preflight records after a digest or failure differs.

        Every rank reaches this from the same gathered rows. A rejection is
        raised before any rank enters model code, so it is not a poisoning
        step failure.
        """

        self._terminal = True
        try:
            failures, _payloads = self._complete(
                sequence,
                preflight.error,
                phase=preflight.phase,
                signature=preflight.signature,
            )
            if any(failure is not None for failure in failures):
                if preflight.error is not None:
                    raise preflight.error
                raise distributed_failure_error(preflight.scope, failures)
        except BaseException as rejection:
            self.rejection = rejection
            raise
        self._terminal = False

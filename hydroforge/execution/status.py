# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Device error statuses that kernels report and the runtime checks.

A module declares an ``ErrorField``: an int32 status ``[failures, code,
*info]`` that kernels bind like any field. A failing thread counts itself;
the first failure also stores its code and info values (where it happened),
so the status always describes one failure consistently. Kernels never stop
the run, and a status only grows, so a status copied after some work tells
whether that work, or any before it, failed. The physical state is not
rolled back.

No managed step waits for the device to learn its statuses:

* each statistics window carries a :class:`StatusTicket`, the statuses
  copied behind the work that produced it (asynchronously on CUDA), and its
  sink verifies the ticket before the window becomes visible: before
  ``model.results`` returns it, or before its rows go to a NetCDF file. The
  sinks already wait for the window's own copy there, so the check costs no
  extra wait. A failed window and every later one are dropped; earlier
  windows are kept and written.
* initialization, a checkpoint save (no failed state is saved) and
  ``close`` check the statuses directly, waiting for the work enqueued so
  far.

So a failure surfaces where the output it spoiled would have appeared: the
next window close, a ``model.results`` read, a save or ``close`` (on several
ranks the rank channel fails the others in that call; ``close`` exchanges
every rank's failure). A managed step that returns has therefore not
confirmed its own kernels: it and the steps and between-steps calls after
it commit (the clock advances), and only their output is withheld. A model
without statistics output learns of a failure at its next save or
``close``. A model without an ``ErrorField`` never checks.

A failure raises a :class:`~hydroforge.core.errors.DeviceStatusError` naming
the first failure and its info values, and poisons the model. Compiled
substeps and CUDA graphs are unaffected: kernels only write device memory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._python_dispatch import _disable_current_modes

from hydroforge.core.devices import devices_match
from hydroforge.core.errors import DeviceStatusError

if TYPE_CHECKING:
    from hydroforge.execution.session import ModelRuntime


@dataclass(frozen=True, slots=True)
class _Status:
    """One declared status, its codes and the names of its info values."""

    field: str
    description: str
    errors: tuple[tuple[int, str], ...]
    info: tuple[str, ...]
    tensor: torch.Tensor

    def describe(self, words: list[int]) -> str:
        failures, code, *info = words
        meaning = dict(self.errors).get(code, f"undeclared error code {code}")
        values = "".join(
            f", {name}={value}" for name, value in zip(self.info, info, strict=True)
        )
        later = f" (first of {failures} failures)" if failures > 1 else ""
        return (
            f"{self.field} ({self.description}): {meaning} [code {code}{values}]{later}"
        )


class StatusTicket:
    """The statuses as they stood after the work enqueued before the ticket.

    On CUDA they are copied into pinned host memory behind that work; on
    other devices they were read already. :meth:`verify` waits for the copy
    (a sink calls it once the output that ticket travels with has landed)
    and names the first failure, if any.
    """

    __slots__ = ("status", "host", "event", "words")

    def __init__(
        self,
        status: DeviceStatus,
        *,
        host: torch.Tensor | None = None,
        event: Any = None,
        words: list[int] | None = None,
    ) -> None:
        self.status = status
        self.host = host
        self.event = event
        self.words = words

    def verify(self) -> DeviceStatusError | None:
        if self.words is None:
            self.event.synchronize()
            self.words = self.host.tolist()
            self.status.free.append(self.host)
            self.host = None
        return self.status.failure(self.words)


class DeviceStatus:
    """The declared error statuses of one model runtime.

    ``on_failure`` receives the first failure found (to poison the model),
    by a check or by a sink verifying a ticket.
    """

    def __init__(
        self,
        statuses: tuple[_Status, ...] = (),
        rank: int = 0,
        on_failure: Callable[[DeviceStatusError], None] | None = None,
    ) -> None:
        self.statuses = statuses
        self.rank = rank
        self.on_failure = on_failure
        self.reported = False
        # The first failure found, by a check or a sink's ticket.
        self.error: DeviceStatusError | None = None
        self.offsets = []
        offset = 0
        for status in statuses:
            self.offsets.append(offset)
            offset += status.tensor.numel()
        self.words = offset
        device = statuses[0].tensor.device if statuses else None
        self.cuda = device is not None and device.type == "cuda"
        # Pinned buffers of verified tickets, reused by later ones.
        self.free: list[torch.Tensor] = []

    @classmethod
    def declared(cls, runtime: ModelRuntime) -> DeviceStatus:
        """Watch every active ``ErrorField`` of the runtime's modules."""

        statuses = []
        for entry in runtime.plan.fields.entries.values():
            if not entry.active or not entry.tensor.errors:
                continue
            tensor = getattr(runtime.modules[entry.module], entry.name)
            words = 2 + len(entry.tensor.error_info)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.int32
                or tuple(tensor.shape) != (words,)
                or not tensor.is_contiguous()
                or not devices_match(tensor.device, runtime.plan.device)
            ):
                raise ValueError(
                    f"error status {entry.qualified} must be {words} contiguous "
                    "int32 words on the model device"
                )
            statuses.append(
                _Status(
                    entry.qualified,
                    entry.description,
                    entry.tensor.errors,
                    entry.tensor.error_info,
                    tensor,
                )
            )
        return cls(
            tuple(statuses),
            runtime.plan.rank,
            partial(runtime.execution.poison, phase="device error status"),
        )

    def ticket(self) -> StatusTicket | None:
        """The statuses behind the work enqueued so far, without waiting."""

        if not self.statuses:
            return None
        if not self.cuda:
            return StatusTicket(self, words=self._read())
        with _disable_current_modes(), torch.no_grad():
            host = (
                self.free.pop()
                if self.free
                else torch.empty(self.words, dtype=torch.int32, pin_memory=True)
            )
            for offset, status in zip(self.offsets, self.statuses, strict=True):
                words = status.tensor.numel()
                host[offset : offset + words].copy_(status.tensor, non_blocking=True)
            device = self.statuses[0].tensor.device
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        return StatusTicket(self, host=host, event=event)

    def check(self) -> DeviceStatusError | None:
        """The failure kernels reported in all work enqueued so far, waiting
        for that work; ``None`` once a failure was reported."""

        if self.reported:
            return None
        ticket = self.ticket()
        return None if ticket is None else ticket.verify()

    def _read(self) -> list[int]:
        with _disable_current_modes():
            return [
                int(word) for status in self.statuses for word in status.tensor.tolist()
            ]

    def failure(self, words: list[int]) -> DeviceStatusError | None:
        """The first status that failed in ``words``, in declaration order:
        one failure. The first one found is reported to ``on_failure``."""

        for offset, status in zip(self.offsets, self.statuses, strict=True):
            own = words[offset : offset + status.tensor.numel()]
            if own[0]:
                error = DeviceStatusError(
                    f"kernels reported a fatal condition on rank {self.rank}: "
                    + status.describe(own)
                )
                if not self.reported:
                    self.reported = True
                    self.error = error
                    if self.on_failure is not None:
                        self.on_failure(error)
                return error
        return None


__all__ = ["DeviceStatus", "StatusTicket"]

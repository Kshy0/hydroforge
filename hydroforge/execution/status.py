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
the run, and the host never waits for a step just to read the statuses:

* after each managed step they are copied to pinned host memory behind the
  step's work, one copy in flight at a time (a status only grows, so a
  skipped copy loses nothing);
* the next managed step and every between-steps call read the last copy
  once the device finished it, without waiting;
* initialization, a checkpoint save and ``close`` wait for them, so no
  failed state is saved and no failure goes unreported.

A failure fails the rank's next managed step (or the save) with a
:class:`~hydroforge.core.errors.DeviceStatusError` naming the first failure
and its info values, and poisons the model; on several ranks the rank
channel fails the others at the same step. Compiled substeps and CUDA
graphs are unaffected: kernels only write device memory, and the copies
follow the work on the caller's stream. Devices other than CUDA read the
statuses directly at those points.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class DeviceStatus:
    """The declared error statuses of one model runtime."""

    def __init__(self, statuses: tuple[_Status, ...] = (), rank: int = 0) -> None:
        self.statuses = statuses
        self.rank = rank
        self.reported = False
        self.offsets = []
        offset = 0
        for status in statuses:
            self.offsets.append(offset)
            offset += status.tensor.numel()
        device = statuses[0].tensor.device if statuses else None
        self.cuda = device is not None and device.type == "cuda"
        self.host: torch.Tensor | None = None
        self.event: Any = None
        self.copied = False
        if self.cuda:
            with _disable_current_modes():
                self.host = torch.zeros(offset, dtype=torch.int32, pin_memory=True)
            self.event = torch.cuda.Event()

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
        return cls(tuple(statuses), runtime.plan.rank)

    def _copy(self) -> None:
        with _disable_current_modes(), torch.no_grad():
            for offset, status in zip(self.offsets, self.statuses, strict=True):
                words = status.tensor.numel()
                self.host[offset : offset + words].copy_(
                    status.tensor, non_blocking=True
                )
            device = self.statuses[0].tensor.device
            self.event.record(torch.cuda.current_stream(device))
        self.copied = True

    def watch(self) -> None:
        """Copy the statuses behind the work enqueued so far, unless the
        last copy is still in flight."""

        if self.cuda and not self.reported and (not self.copied or self.event.query()):
            self._copy()

    def poll(self) -> DeviceStatusError | None:
        """The failure the device already reported, without waiting."""

        if not self.statuses or self.reported:
            return None
        if not self.cuda:
            return self._failure(self._read())
        if not self.copied or not self.event.query():
            return None
        return self._failure(self.host.tolist())

    def wait(self) -> DeviceStatusError | None:
        """The failure the device reported for all work enqueued so far."""

        if not self.statuses or self.reported:
            return None
        if not self.cuda:
            return self._failure(self._read())
        self._copy()
        self.event.synchronize()
        return self._failure(self.host.tolist())

    def _read(self) -> list[int]:
        with _disable_current_modes():
            return [
                int(word) for status in self.statuses for word in status.tensor.tolist()
            ]

    def _failure(self, words: list[int]) -> DeviceStatusError | None:
        # The first status that failed, in declaration order: one failure.
        for offset, status in zip(self.offsets, self.statuses, strict=True):
            own = words[offset : offset + status.tensor.numel()]
            if own[0]:
                self.reported = True
                return DeviceStatusError(
                    f"kernels reported a fatal condition on rank {self.rank}: "
                    + status.describe(own)
                )
        return None


__all__ = ["DeviceStatus"]

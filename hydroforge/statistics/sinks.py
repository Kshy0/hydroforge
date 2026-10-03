"""Destinations of finalized statistics samples.

A sink receives the storage tensors of every output that closed a window,
``append(dt, values)``, and must not retain them: the runtime keeps
accumulating into the same storage.  :class:`MemorySink` keeps owned copies
for ``model.results``; the NetCDF sink is
:class:`hydroforge.io.rank_output.writer.RankOutputWriter`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from hydroforge.io.netcdf.encoding import checked_narrowing, raise_narrowing_failures


class StatisticsSink(Protocol):
    """What the statistics runtime calls on its output destination."""

    def append(self, dt: Any, values: Mapping[str, torch.Tensor]) -> None:
        """Take one finalized sample of the outputs in ``values``."""

    def flush(self, dt: Any) -> None:
        """Make every accepted sample durable (e.g. before a checkpoint)."""

    def poll(self, dt: Any) -> None:
        """Raise a completed background failure without waiting."""

    def reset(self) -> None:
        """Prepare for a restarted output timeline."""

    def set_run_id(self, run_id: str) -> None:
        """Adopt the output identity shared by every rank of one run."""

    def close(self) -> None:
        """Finish pending work and release every owned resource."""


class MemorySink:
    """Retain every finalized sample as an owned tensor on ``device``.

    ``outputs`` maps each output to its storage shape and retained dtype.
    """

    def __init__(
        self,
        outputs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
        *,
        device: torch.device,
    ) -> None:
        self.device = device
        self._layouts = dict(outputs)
        self._results: dict[str, list[torch.Tensor]] = {name: [] for name in outputs}

    def append(self, dt: Any, values: Mapping[str, torch.Tensor]) -> None:
        # One host check covers every narrowed output of the sample.
        flags: list[tuple[torch.Tensor, str, str]] = []
        converted = {
            name: value.detach()
            if self._layouts[name][1].itemsize > value.dtype.itemsize
            else checked_narrowing(
                value, self._layouts[name][1], name=name, flags=flags
            )
            for name, value in values.items()
        }
        raise_narrowing_failures(flags)
        copies = {}
        for name, value in converted.items():
            dtype = self._layouts[name][1]
            if value.dtype == dtype:
                copies[name] = value.to(
                    device=self.device, copy=value.dtype == values[name].dtype
                )
            else:
                # Exact widening follows the move: the sampling device may
                # lack the wider type (MPS has no float64).
                copies[name] = value.to(device=self.device).to(dtype)
        for name, value in copies.items():
            self._results[name].append(value)

    def _snapshot(
        self, name: str, selection: slice, *, as_stacked: bool
    ) -> torch.Tensor | list[torch.Tensor]:
        values = self._results[name][selection]
        if not as_stacked:
            return [
                value.clone(memory_format=torch.preserve_format) for value in values
            ]
        if values:
            return torch.stack(values, dim=0)
        shape, dtype = self._layouts[name]
        return torch.empty((0, *shape), dtype=dtype, device=self.device)

    def get(
        self,
        name: str,
        *,
        as_stacked: bool = True,
        start: int | None = None,
        stop: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return isolated copies of an optional interval of one output."""

        return self._snapshot(name, slice(start, stop), as_stacked=as_stacked)

    def all(
        self,
        *,
        as_stacked: bool = True,
        start: int | None = None,
        stop: int | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Return isolated copies of an optional interval of every output."""

        return {
            name: self._snapshot(name, slice(start, stop), as_stacked=as_stacked)
            for name in self._results
        }

    def drain(
        self, max_steps: int | None = None, *, as_stacked: bool = True
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Copy then release the oldest samples, keeping the output timeline."""

        result = self.all(as_stacked=as_stacked, stop=max_steps)
        for values in self._results.values():
            del values[:max_steps]
        return result

    def pop(self, name: str) -> torch.Tensor | None:
        """Remove and return the newest sample of one output."""

        values = self._results[name]
        return values.pop() if values else None

    def reset(self) -> None:
        for values in self._results.values():
            values.clear()

    def flush(self, dt: Any) -> None:
        del dt

    def poll(self, dt: Any) -> None:
        del dt

    def set_run_id(self, run_id: str) -> None:
        del run_id

    def close(self) -> None:
        pass

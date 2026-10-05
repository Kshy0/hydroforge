"""Destinations of finalized statistics samples.

A sink receives the storage tensors of every output that closed a window,
``append(dt, values)``, and must not retain them: the runtime keeps
accumulating into the same storage.  :class:`MemorySink` keeps owned copies
for ``model.results``; the NetCDF sink is
:class:`hydroforge.io.rank_output.writer.RankOutputWriter`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Protocol

import torch

from hydroforge.io.netcdf.encoding import (
    checked_narrowing,
    decoded_output_tensor,
    raise_narrowing_failures,
)
from hydroforge.kernels.emulated import EmulatedTensor


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
    CPU results are ordinary tensors. On MPS, declared ``encoded_outputs``
    retained as float64 remain explicitly encoded tensors, including stacked
    and empty results; other float64 results are rejected at construction.
    """

    def __init__(
        self,
        outputs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
        *,
        device: torch.device,
        encoded_outputs: Collection[str] = (),
    ) -> None:
        self.device = torch.device(device)
        self._layouts = dict(outputs)
        self._encoded_outputs = frozenset(encoded_outputs)
        if self.device.type == "mps":
            for name, (_shape, dtype) in self._layouts.items():
                if dtype == torch.float64 and name not in self._encoded_outputs:
                    raise ValueError(
                        f"statistics output {name!r}: float64 results on MPS "
                        "require float32x2-encoded storage; use result_device='cpu' "
                        "or save_precision='float32'"
                    )
        self._results: dict[str, list[torch.Tensor]] = {name: [] for name in outputs}

    def append(self, dt: Any, values: Mapping[str, torch.Tensor]) -> None:
        # One host check covers every narrowed output of the sample.
        flags: list[tuple[torch.Tensor, str, str]] = []
        converted = {}
        for name, value in values.items():
            dtype = self._layouts[name][1]
            if self.device.type == "mps" and dtype == torch.float64:
                if not isinstance(value, EmulatedTensor):
                    raise TypeError(
                        f"statistics output {name!r} requires encoded float64 storage"
                    )
                converted[name] = value.detach()
                continue
            source = decoded_output_tensor(value)
            converted[name] = (
                source
                if dtype.itemsize > source.dtype.itemsize
                else checked_narrowing(source, dtype, name=name, flags=flags)
            )
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
            if isinstance(values[0], EmulatedTensor):
                # Stacking is a bit-preserving layout operation, not encoded
                # arithmetic. Stack owned carriers without decoding to FP32.
                return EmulatedTensor(torch.stack([value.carrier for value in values]))
            return torch.stack(values, dim=0)
        shape, dtype = self._layouts[name]
        if self.device.type == "mps" and dtype == torch.float64:
            return EmulatedTensor(
                torch.empty((0, *shape), dtype=torch.int64, device=self.device)
            )
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

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

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

from hydroforge.io.netcdf.encoding import narrowing_flag, raise_narrowing_failures
from hydroforge.kernels.emulated import EmulatedTensor


def decoded_output_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detach ordinary storage, or snapshot encoded values as CPU float64.

    An encoded tensor's int64 carrier is never an integer output. Decode at
    the export boundary before narrowing, NumPy conversion or copying into
    ordinary host storage. ``copy=True`` is required even for a CPU-backed
    encoded tensor: a no-op ``cpu()`` would leave the wrapper undecoded.
    """

    source = tensor.detach()
    if isinstance(source, EmulatedTensor):
        return source.to(device="cpu", copy=True)
    return source


def checked_narrowing(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    name: str,
    flags: list[tuple[torch.Tensor, str, str]] | None = None,
) -> torch.Tensor:
    """Convert logical values, rejecting finite values out of range.

    Tiny values may round to subnormals or zero.  With ``flags`` the
    device-side overflow flag is queued for a deferred host check; otherwise
    it is checked before returning. Encoded Metal values are snapshotted and
    decoded on the CPU first; ordinary tensors stay on their device.
    """

    source = decoded_output_tensor(tensor)
    if source.dtype == target_dtype:
        return source
    converted = source.to(dtype=target_dtype)
    entry = narrowing_flag(source, target_dtype, name=name)
    if entry is not None:
        if flags is None:
            raise_narrowing_failures((entry,))
        else:
            flags.append(entry)
    return converted


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

    Samples leaving a CUDA device for CPU results are copied asynchronously
    into one pinned staging buffer per dtype and land in owned pageable
    storage before the next sample or any read, so a sample never waits for
    the device queue.
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
        self._staging: dict[torch.dtype, torch.Tensor] = {}
        # (copy event, ((name, staged view), ...)) of the last staged sample.
        self._staged: tuple[Any, tuple[tuple[str, torch.Tensor], ...]] | None = None

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
        self._land()
        copies = {}
        # Values leaving their device move in one transfer per dtype: a
        # blocking copy per output would wait on the device once per output.
        # CUDA holds every result dtype, so CUDA samples for CPU results are
        # widened on the device and staged asynchronously, grouped by result
        # dtype so that each staging buffer serves one transfer.
        staging = self.device.type == "cpu"
        moving: dict[torch.dtype, list[str]] = {}
        for name, value in converted.items():
            if isinstance(value, EmulatedTensor):
                copies[name] = value.to(copy=True)
            elif value.device == self.device:
                copies[name] = value.to(
                    dtype=self._layouts[name][1],
                    copy=value.dtype == values[name].dtype,
                )
            else:
                staged_here = staging and value.device.type == "cuda"
                key = self._layouts[name][1] if staged_here else value.dtype
                moving.setdefault(key, []).append(name)
        staged: list[tuple[str, torch.Tensor]] = []
        stream = None
        for dtype, names in moving.items():
            parts = [converted[name] for name in names]
            device = parts[0].device
            if staging and device.type == "cuda":
                parts = [part.to(dtype=dtype) for part in parts]
            flat = (
                parts[0].reshape(-1)
                if len(parts) == 1
                else torch.cat([part.reshape(-1) for part in parts])
            )
            sizes = [part.numel() for part in parts]
            if staging and device.type == "cuda":
                stream = torch.cuda.current_stream(device)
                for name, part, value in zip(
                    names, parts, self._stage(flat).split(sizes), strict=True
                ):
                    staged.append((name, value.view(part.shape)))
                continue
            moved = flat.to(device=self.device).split(sizes)
            for name, part, value in zip(names, parts, moved, strict=True):
                copies[name] = value.view(part.shape)
        if staged:
            event = torch.cuda.Event()
            event.record(stream)
            self._staged = (event, tuple(staged))
        for name, value in copies.items():
            # Exact widening follows the move: the sampling device may lack
            # the wider type (MPS has no float64).
            dtype = self._layouts[name][1]
            self._results[name].append(
                value if value.dtype == dtype else value.to(dtype)
            )

    def _stage(self, flat: torch.Tensor) -> torch.Tensor:
        """Enqueue ``flat``'s copy into the pinned buffer of its dtype."""

        buffer = self._staging.get(flat.dtype)
        if buffer is None or buffer.numel() < flat.numel():
            buffer = torch.empty(flat.numel(), dtype=flat.dtype, pin_memory=True)
            self._staging[flat.dtype] = buffer
        target = buffer[: flat.numel()]
        target.copy_(flat, non_blocking=True)
        return target

    def _land(self) -> None:
        """Move the staged sample into owned results once its copy completed."""

        staged, self._staged = self._staged, None
        if staged is None:
            return
        event, values = staged
        event.synchronize()
        for name, value in values:
            self._results[name].append(value.clone())

    def _snapshot(
        self, name: str, selection: slice, *, as_stacked: bool
    ) -> torch.Tensor | list[torch.Tensor]:
        self._land()
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

        self._land()
        values = self._results[name]
        return values.pop() if values else None

    def reset(self) -> None:
        self._land()
        for values in self._results.values():
            values.clear()

    def flush(self, dt: Any) -> None:
        del dt
        self._land()

    def poll(self, dt: Any) -> None:
        del dt

    def set_run_id(self, run_id: str) -> None:
        del run_id

    def close(self) -> None:
        self._land()
        self._staging.clear()

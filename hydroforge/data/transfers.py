"""Bounded transfer staging with explicit stream and buffer ownership."""

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import closing, nullcontext
from typing import Annotated, Any

import torch
from pydantic import BeforeValidator, Field, validate_call

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.contracts.validation import HydroForgeModel


def _exact_integer(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("pipeline sizes and counts must use exact integers")
    return value


_NonnegativeCount = Annotated[int, BeforeValidator(_exact_integer), Field(ge=0)]
_PositiveCount = Annotated[_NonnegativeCount, Field(gt=0)]


@validate_call(config=HydroForgeModel.model_config)
def plan_pipeline_buffers(
    total_bytes: _PositiveCount,
    *,
    input_batch_bytes: _NonnegativeCount,
    loader_workers: _NonnegativeCount = 0,
    prefetch_factor: _PositiveCount = 2,
    input_cache_bytes: _NonnegativeCount = 0,
    device_prefetch_bytes: _NonnegativeCount = 64 * 1024 * 1024,
) -> dict[str, int]:
    """Budget known pipeline buffers, not model memory or total process RSS.

    Reserve raw/pinned input queues, worker scratch and lookahead; allow three
    output payload copies for parent, transport and worker staging. Callers
    must separately account for library caches and model allocations.
    """

    input_slots = 2 * max(1, loader_workers * prefetch_factor) + loader_workers + 3
    input_reserved = input_cache_bytes + input_slots * input_batch_bytes
    output_bytes = (total_bytes - input_reserved - device_prefetch_bytes) // 3
    if output_bytes <= 0:
        raise ValueError(
            "pipeline budget cannot cover input queues and device lookahead"
        )
    return {
        "total_bytes": total_bytes,
        "input_reserved_bytes": input_reserved,
        "device_prefetch_bytes": device_prefetch_bytes,
        "max_pending_output_bytes": output_bytes,
    }


class _PendingCPUTransfer:
    """One enqueued device-to-host staging whose outputs are valid after wait."""

    def __init__(self, outputs, events, sources) -> None:
        self._outputs = outputs
        self._events = events
        self._sources = sources

    @property
    def asynchronous(self) -> bool:
        return bool(self._events)

    def wait(self) -> dict[str, torch.Tensor]:
        """Block until every copy landed, then release the device sources."""

        failures: list[BaseException] = []
        for event in self._events:
            try:
                event.synchronize()
            except BaseException as error:
                failures.append(error)
        self._events = ()
        self._sources = ()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            error = ResourceCleanupError("CPU transfer completion", failures)
            raise error from failures[0]
        return self._outputs


class _TensorCPUStager:
    """Return borrowed CPU snapshots in pinned buffers bounded by ``max_bytes``.

    Pinned buffers are cached by slot, name, device, dtype, and shape; a miss
    evicts least recently used buffers only as needed.  ``stage`` blocks until
    the copies land.  ``stage_async`` alternates between two buffer slots, so
    one transfer's outputs stay valid while the next one is in flight.
    """

    def __init__(self, max_bytes: int = 8 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._buffers: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._slot = 0
        self._copy_streams: dict[torch.device, Any] = {}

    @property
    def allocated_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in self._buffers.values())

    def clear(self) -> None:
        self._buffers.clear()

    def _enqueue(self, tensors, *, slot, copy_stream: bool):
        cached = self._buffers
        cached_bytes = sum(_tensor_bytes(value) for value in cached.values())
        retained = OrderedDict()
        outputs = {}
        streams = {}
        sources = []
        remaining = self.max_bytes
        failures: list[BaseException] = []
        try:
            for name, value in tensors.items():
                tensor = value.detach()
                size = tensor.numel() * tensor.element_size()
                if tensor.device.type != "cuda" or size > remaining:
                    outputs[name] = tensor.cpu()
                    continue
                key = (name, tensor.device, tensor.dtype, tuple(tensor.shape))
                if slot is not None:
                    key = (slot, *key)
                buffer = cached.pop(key, None)
                if buffer is None:
                    # Buffers retained by this call occupy max_bytes - remaining.
                    while cached and (
                        cached_bytes + self.max_bytes - remaining + size
                        > self.max_bytes
                    ):
                        cached_bytes -= _tensor_bytes(cached.popitem(last=False)[1])
                    buffer = torch.empty(
                        tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
                    )
                else:
                    cached_bytes -= size
                retained[key] = buffer
                remaining -= size
                stream = streams.get(tensor.device)
                if stream is None:
                    stream = torch.cuda.current_stream(tensor.device)
                    if copy_stream:
                        # Copy beside compute: the side stream starts after the
                        # producer's queued work and never delays later kernels.
                        side = self._copy_streams.get(tensor.device)
                        if side is None:
                            side = torch.cuda.Stream(device=tensor.device)
                            self._copy_streams[tensor.device] = side
                        side.wait_stream(stream)
                        stream = side
                    streams[tensor.device] = stream
                if copy_stream:
                    with torch.cuda.stream(stream):
                        buffer.copy_(tensor, non_blocking=True)
                    tensor.record_stream(stream)
                    sources.append(tensor)
                else:
                    buffer.copy_(tensor, non_blocking=True)
                outputs[name] = buffer
        except BaseException as error:
            failures.append(error)
        cached.update(retained)
        self._buffers = cached
        return outputs, streams, sources, failures

    @staticmethod
    def _record(streams, failures):
        events = []
        for stream in streams.values():
            try:
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
            except BaseException as error:
                failures.append(error)
        return events

    @staticmethod
    def _raise(failures):
        if len(failures) == 1:
            raise failures[0]
        if failures:
            error = ResourceCleanupError("CPU transfer staging", failures)
            raise error from failures[0]

    def stage(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs, streams, _sources, failures = self._enqueue(
            tensors, slot=None, copy_stream=False
        )
        for stream in streams.values():
            try:
                event = torch.cuda.Event()
                event.record(stream)
                event.synchronize()
            except BaseException as error:
                failures.append(error)
        self._raise(failures)
        return outputs

    def stage_async(self, tensors: Mapping[str, torch.Tensor]) -> _PendingCPUTransfer:
        """Enqueue copies of caller-owned snapshots without waiting.

        The caller must not mutate the device tensors, and must finish using
        the previous transfer's outputs before the next-but-one call.
        """

        slot, self._slot = self._slot, 1 - self._slot
        outputs, streams, sources, failures = self._enqueue(
            tensors, slot=slot, copy_stream=True
        )
        events = self._record(streams, failures)
        if failures:
            for event in events:
                try:
                    event.synchronize()
                except BaseException as error:
                    failures.append(error)
            self._raise(failures)
        return _PendingCPUTransfer(outputs, tuple(events), tuple(sources))


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _map_tensors(value, operation):
    if isinstance(value, torch.Tensor):
        return operation(value)
    if isinstance(value, Mapping):
        return {name: _map_tensors(item, operation) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [_map_tensors(item, operation) for item in value]
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*items)
        return type(value)(items)
    return value


@validate_call(config=HydroForgeModel.model_config)
def prefetch_to_device(
    batches, device, *, max_prefetch_bytes: _NonnegativeCount = 64 * 1024 * 1024
):
    """Yield batches with one lookahead transfer; close the iterator on early exit.

    Pinned DataLoader tensors enable overlap. The byte limit controls the
    lookahead batch, not model state or the caller's retained outputs.
    Oversized batches use the synchronous path without lookahead.
    """

    device = torch.device(device)
    iterator = iter(batches)
    lifetime = (
        closing(iterator)
        if callable(getattr(iterator, "close", None))
        else nullcontext()
    )
    with lifetime:
        if device.type != "cuda" or max_prefetch_bytes == 0:
            for batch in iterator:
                yield _map_tensors(batch, lambda tensor: tensor.to(device))
            return
        stream = torch.cuda.Stream(device=device)
        hosts = []
        sentinel = object()

        def load(batch):
            while len(hosts) >= 2:
                hosts.pop(0)[0].synchronize()
            hosts[:] = [(event, owner) for event, owner in hosts if not event.query()]
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                value = _map_tensors(
                    batch, lambda tensor: tensor.to(device, non_blocking=True)
                )
                event = torch.cuda.Event()
                event.record(stream)
            hosts.append((event, batch))
            return value, event

        def size_of(batch):
            sizes = []
            _map_tensors(
                batch,
                lambda tensor: sizes.append(tensor.numel() * tensor.element_size()),
            )
            return sum(sizes)

        pending = None
        try:
            batch = next(iterator, sentinel)
            while batch is not sentinel:
                if pending is None:
                    pending = load(batch)
                current, ready = pending
                consumer = torch.cuda.current_stream(device)
                consumer.wait_event(ready)
                _map_tensors(current, lambda tensor: tensor.record_stream(consumer))
                if size_of(batch) <= max_prefetch_bytes:
                    batch = next(iterator, sentinel)
                    pending = (
                        load(batch)
                        if batch is not sentinel
                        and size_of(batch) <= max_prefetch_bytes
                        else None
                    )
                    yield current
                else:
                    ready.synchronize()
                    pending = None
                    yield current
                    batch = next(iterator, sentinel)
        finally:
            try:
                stream.synchronize()
            finally:
                hosts.clear()

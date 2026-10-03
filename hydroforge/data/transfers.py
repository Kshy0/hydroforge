"""Device prefetch of input batches with explicit stream ownership."""

from collections.abc import Mapping
from contextlib import closing, nullcontext

import torch
from pydantic import NonNegativeInt, validate_call

from hydroforge.core.validation import HydroForgeModel


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
    batches, device, *, max_prefetch_bytes: NonNegativeInt = 64 * 1024 * 1024
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

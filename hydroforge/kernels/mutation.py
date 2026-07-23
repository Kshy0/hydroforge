"""Backend-independent mutation tracing used during capture warmup."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable

import torch

from hydroforge.contracts import buffer_access_semantics


_ACTIVE_TRACE: ContextVar[MutationTrace | None] = ContextVar(
    "hydroforge_mutation_trace", default=None,
)


class MutationTrace:
    """Discover tensor writes without importing a capture backend."""

    def __init__(
        self,
        candidates: Iterable[torch.Tensor],
        snapshots: dict[int, torch.Tensor] | None = None,
    ) -> None:
        self.candidates = tuple(dict.fromkeys(candidates))
        self.versions = {id(tensor): tensor._version for tensor in self.candidates}
        self.snapshots = dict(snapshots or {})
        self.snapshots.update({
            id(tensor): tensor.detach().to(device="cpu", copy=True)
            for tensor in self.candidates if id(tensor) not in self.snapshots
        })
        self.tensors = {id(tensor): tensor for tensor in self.candidates}
        self.native_writes: set[int] = set()

    def record(self, metadata: Any, arguments: dict[str, Any]) -> None:
        for name, access in metadata.buffers.items():
            tensor = arguments.get(name)
            if (
                not buffer_access_semantics(access).writes
                or not isinstance(tensor, torch.Tensor)
            ):
                continue
            identity = id(tensor)
            if identity not in self.snapshots:
                self.snapshots[identity] = tensor.detach().to(device="cpu", copy=True)
                self.tensors[identity] = tensor
                self.versions[identity] = tensor._version
            self.native_writes.add(identity)

    def mutated(self) -> tuple[torch.Tensor, ...]:
        identities = self.native_writes | {
            identity for identity, tensor in self.tensors.items()
            if tensor._version != self.versions[identity]
        }
        return tuple(self.tensors[identity] for identity in identities)

    def restore_all(self) -> None:
        for identity, tensor in self.tensors.items():
            tensor.copy_(self.snapshots[identity])

    def snapshots_for(self, tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        return [self.snapshots[id(tensor)] for tensor in tensors]


@contextmanager
def trace_mutations(
    candidates: Iterable[torch.Tensor],
    snapshots: dict[int, torch.Tensor] | None = None,
):
    trace = MutationTrace(candidates, snapshots)
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE_TRACE.reset(token)


def record_kernel_writes(metadata: Any, arguments: dict[str, Any]) -> None:
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        trace.record(metadata, arguments)

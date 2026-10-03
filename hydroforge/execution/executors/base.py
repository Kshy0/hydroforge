"""The loop-executor protocol and the capture ownership every executor shares."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, ClassVar

import torch

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.kernels.toolchain import CompileRequest


class LoopExecutor:
    """Launch one model's compiled loops on its backend and own their captures.

    ``fixed``, ``adaptive``, ``predicate`` and ``outer`` turn a bound loop or
    outer operator program into a runner with ``run(...)`` and ``close()``;
    ``repeated`` does the same for a body launched many times, and ``sample``
    launches one host-issued statistics sample.  Graphs and command buffers a
    runner captures are registered here and released by the runner, by
    :meth:`invalidate` or by :meth:`close`.
    """

    name: ClassVar[str]

    def __init__(self, device: torch.device, *, world_size: int = 1) -> None:
        if not isinstance(device, torch.device):
            raise TypeError("executor device must be a torch.device")
        if type(world_size) is not int or world_size < 1:
            raise ValueError("executor world_size must be a positive exact int")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.world_size = world_size
        self._resources: list[tuple[Any, Callable[[], None]]] = []

    @staticmethod
    def _finalizer(resource: Any) -> Callable[[], None]:
        for method_name in ("close", "destroy", "reset"):
            finalizer = getattr(resource, method_name, None)
            if callable(finalizer):
                return finalizer
        raise TypeError(
            "backend capture resources must define close(), destroy(), or reset()"
        )

    def compile_requests(self, statistics: bool) -> tuple[CompileRequest, ...]:
        """Programs this executor launches besides the recorded ones."""

        del statistics
        return ()

    def register(self, resource: Any) -> Any:
        """Register a closeable backend resource under model ownership."""

        self._resources.append((resource, self._finalizer(resource)))
        return resource

    def release(self, resource: Any) -> None:
        """Release one owned resource and remove every retained reference."""

        index = next(
            index
            for index, (owned, _finalizer) in enumerate(self._resources)
            if owned is resource
        )
        _owned, finalizer = self._resources.pop(index)
        finalizer()

    def release_all(self, resources: Iterable[Any], *, scope: str) -> None:
        """Release distinct owned resources, retaining every failure."""

        distinct = {
            id(resource): resource for resource in resources if resource is not None
        }
        with cleanup_on_exit(
            scope, (partial(self.release, resource) for resource in distinct.values())
        ):
            pass

    def invalidate(self) -> None:
        """Release captures whose fixed bindings may have become stale."""

        resources, self._resources = self._resources, []
        with cleanup_on_exit(
            "backend capture resources",
            (finalizer for _resource, finalizer in reversed(resources)),
        ):
            pass

    def close(self) -> None:
        self.invalidate()


class DirectRunner:
    """Launch a body directly; nothing is captured."""

    __slots__ = ("body",)

    def __init__(self, body: Callable[[], None]) -> None:
        self.body = body

    def run(self) -> None:
        self.body()

    def close(self) -> None:
        pass

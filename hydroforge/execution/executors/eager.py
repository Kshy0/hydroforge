# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Direct launches of recorded programs under host-controlled loops."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from hydroforge.execution.executors.base import DirectRunner, LoopExecutor


class EagerFixed:
    """Launch a fixed loop's iterations and samples from the host."""

    __slots__ = ("loop",)

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def run(self, count: int, duration: float, step: Any) -> int:
        loop = self.loop
        width = duration / count
        controlled = loop.controlled
        loop.prepare(count, width, controls=controlled, weight=loop.weighted)
        body, final = loop.body, loop.final
        for index in range(count):
            last = index == count - 1
            body.launch()
            if last and final is not None:
                final.launch()
            if controlled:
                loop.counter.add_(1)
            step.sample(first=index == 0, last=last, weight=width)
        return count

    def close(self) -> None:
        pass


def host_adaptive(
    loop: Any,
    step: Any,
    iterate: Callable[[], None],
    sources: tuple[Any, ...],
    *,
    check_failed: Callable[[int], None] | None = None,
) -> int:
    """Run an adaptive loop whose iterations the host launches one by one.

    Each iteration reads ``(error, continue, dt)`` once from ``loop.status``,
    packed from ``sources`` unless the iteration writes it itself.
    ``check_failed`` replaces ``loop.check_completion`` for an iteration that
    packs further error bits into the status word.
    """

    loop.reset()
    count = 0
    continuing = True
    while continuing:
        iterate()
        failed, flag, weight = loop.fetch(sources, loop.status, loop.status_host)
        (check_failed or loop.check_completion)(int(failed))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"adaptive substep proposal produced an invalid accepted width {weight}"
            )
        count += 1
        continuing = flag != 0
        step.sample(first=count == 1, last=not continuing, weight=weight)
    return count


class EagerAdaptive:
    """Launch an adaptive loop's torch iterations from the host."""

    __slots__ = ("loop",)

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def run(self, duration: float, step: Any) -> int:
        loop = self.loop
        return host_adaptive(loop, step, loop.iterate, loop.status_sources)

    def close(self) -> None:
        pass


class EagerPredicate:
    """Launch a predicate loop's torch iterations until its flag clears."""

    __slots__ = ("loop",)

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def run(self) -> None:
        loop = self.loop
        loop.reset()
        while True:
            loop.iterate()
            if int(loop.continue_flag.item()) == 0:
                break

    def close(self) -> None:
        pass


class EagerExecutor(LoopExecutor):
    """Launch every recorded program directly; loops are host-controlled."""

    name = "eager"

    def fixed(self, loop: Any) -> EagerFixed:
        return EagerFixed(loop)

    def adaptive(self, loop: Any) -> EagerAdaptive:
        return EagerAdaptive(loop)

    def predicate(self, loop: Any) -> EagerPredicate:
        return EagerPredicate(loop)

    def outer(self, program: Any) -> DirectRunner:
        return DirectRunner(program.launch)

    def repeated(
        self, body: Callable[[], None], *, mutated_state: tuple[Any, ...]
    ) -> DirectRunner:
        del mutated_state
        return DirectRunner(body)

    def sample(self, launch: Any, phase: int) -> None:
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                launch(phase)
        else:
            launch(phase)

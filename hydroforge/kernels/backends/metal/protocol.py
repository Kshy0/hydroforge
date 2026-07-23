"""Nominal command contract for one online-compiled Metal program."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetalCommandNode(ABC):
    """An explicitly recordable node accepted by Metal ICB construction."""

    reads: tuple[Any, ...]
    writes: tuple[Any, ...]

    @abstractmethod
    def record(self) -> None:
        """Record this node into the active Metal command sequence."""

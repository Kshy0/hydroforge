"""Typed model capability contracts; free-form requirement dicts are forbidden."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


_PRECISIONS = frozenset({"float32", "float64"})


@dataclass(frozen=True, slots=True)
class BackendRequirement:
    """Model-wide restrictions not already defined by the backend runtime."""

    precision: frozenset[str] | None = None
    mixed_precision: bool = True
    trials: bool = True
    min_block_size: int | None = None
    max_block_size: int | None = None
    block_size: int | None = None

    def __post_init__(self) -> None:
        if self.precision is not None:
            if type(self.precision) is not frozenset or not self.precision:
                raise TypeError(
                    "backend precision must be a non-empty exact frozenset"
                )
            unknown = self.precision.difference(_PRECISIONS)
            if unknown:
                raise ValueError(
                    f"backend precision contains unknown values: {sorted(unknown)}"
                )
        for name in ("mixed_precision", "trials"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"backend requirement {name} must be bool")
        for name in ("min_block_size", "max_block_size", "block_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(
                    f"backend requirement {name} must be a positive exact int"
                )
        if (
            self.min_block_size is not None
            and self.max_block_size is not None
            and self.min_block_size > self.max_block_size
        ):
            raise ValueError("backend block-size range is empty")
        if self.block_size is not None and (
            self.min_block_size is not None
            and self.block_size < self.min_block_size
            or self.max_block_size is not None
            and self.block_size > self.max_block_size
        ):
            raise ValueError("fixed backend block size is outside its range")

    def validate_block_size(self, value: int, *, backend: str) -> None:
        """Validate one resolved model or per-kernel launch width."""

        if type(value) is not int or not 1 <= value <= 1024:
            raise ValueError(
                f"backend {backend!r} BLOCK_SIZE must be an exact int in "
                f"[1, 1024], got {value!r}"
            )
        if self.min_block_size is not None and value < self.min_block_size:
            raise ValueError(
                f"backend {backend!r} requires BLOCK_SIZE >= "
                f"{self.min_block_size}, got {value}"
            )
        if self.max_block_size is not None and value > self.max_block_size:
            raise ValueError(
                f"backend {backend!r} requires BLOCK_SIZE <= "
                f"{self.max_block_size}, got {value}"
            )
        if self.block_size is not None and value != self.block_size:
            raise ValueError(
                f"backend {backend!r} requires BLOCK_SIZE={self.block_size}, "
                f"got {value}"
            )

    def validate_precision(
        self, precision: str, mixed_precision: bool, *, backend: str,
    ) -> None:
        """Validate model precision against one runtime or model restriction."""

        if self.precision is not None and precision not in self.precision:
            raise ValueError(
                f"backend {backend!r} requires precision in {self.precision}, "
                f"got {precision!r}"
            )
        if not self.mixed_precision and mixed_precision:
            raise ValueError(
                f"backend {backend!r} does not support mixed precision"
            )


@dataclass(frozen=True, slots=True)
class ModuleRequirement:
    """Restrictions introduced only when one optional module is open."""

    trials: bool = True

    def __post_init__(self) -> None:
        if type(self.trials) is not bool:
            raise TypeError("module requirement trials must be bool")


DEFAULT_BACKEND_REQUIREMENT = BackendRequirement()
DEFAULT_MODULE_REQUIREMENT = ModuleRequirement()

# Intrinsic runtime limits belong to HydroForge, not to every downstream
# model. Model ``backend_requirements`` may only add stricter constraints.
RUNTIME_BACKEND_REQUIREMENTS = MappingProxyType({
    "metal": BackendRequirement(
        precision=frozenset({"float32"}), mixed_precision=False,
    ),
})

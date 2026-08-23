"""Typed model capability contracts; free-form requirement dicts are forbidden."""

from __future__ import annotations

from types import MappingProxyType
from typing import Self

from pydantic import model_validator

from hydroforge.contracts.validation import HydroForgeModel


_PRECISIONS = frozenset({"float32", "float64"})
DEFAULT_BLOCK_SIZE = 256
MODEL_OWNED_MODULE_FIELDS = (
    "opened_modules",
    "rank",
    "device",
    "precision",
    "mixed_precision",
    "num_trials",
)


def validate_runtime_block_size(value: int, *, backend: str) -> None:
    """Validate launch-width constraints intrinsic to one backend runtime."""

    if type(value) is not int or not 1 <= value <= 1024:
        raise ValueError(
            f"backend {backend!r} BLOCK_SIZE must be an exact int in "
            f"[1, 1024], got {value!r}"
        )
    if backend == "triton" and value & (value - 1):
        raise ValueError(
            "backend 'triton' BLOCK_SIZE must be a power of two, "
            f"got {value}"
        )


def _effective_block_size(
    configured: int | None,
    *,
    backend: str,
    default: int = DEFAULT_BLOCK_SIZE,
) -> int:
    """Resolve the launch width actually used by one backend."""

    if backend == "metal":
        return DEFAULT_BLOCK_SIZE
    return default if configured is None else configured


class BackendRequirement(HydroForgeModel):
    """Model-wide restrictions not already defined by the backend runtime."""

    precision: frozenset[str] | None = None
    mixed_precision: bool = True
    trials: bool = True
    min_block_size: int | None = None
    max_block_size: int | None = None
    block_size: int | None = None

    @model_validator(mode="after")
    def _validate_requirement(self) -> Self:
        if self.precision is not None:
            if type(self.precision) is not frozenset or not self.precision:
                raise ValueError(
                    "backend precision must be a non-empty exact frozenset"
                )
            unknown = self.precision.difference(_PRECISIONS)
            if unknown:
                raise ValueError(
                    f"backend precision contains unknown values: {sorted(unknown)}"
                )
        for name in ("mixed_precision", "trials"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"backend requirement {name} must be bool")
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
        return self

    def _validate_block_size(self, value: int, *, backend: str) -> None:
        """Validate one resolved model or per-kernel launch width."""

        validate_runtime_block_size(value, backend=backend)
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

    def _validate_precision(
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


class ModuleRequirement(HydroForgeModel):
    """Restrictions introduced only when one optional module is open."""

    trials: bool = True

    @model_validator(mode="after")
    def _validate_requirement(self) -> Self:
        if type(self.trials) is not bool:
            raise ValueError("module requirement trials must be bool")
        return self


DEFAULT_BACKEND_REQUIREMENT = BackendRequirement()
DEFAULT_MODULE_REQUIREMENT = ModuleRequirement()

# Intrinsic runtime limits belong to HydroForge, not to every downstream
# model. Launch-width rules are enforced by ``validate_runtime_block_size``;
# these declarative requirements capture the remaining backend capabilities.
# Model ``backend_requirements`` may only add stricter constraints.
RUNTIME_BACKEND_REQUIREMENTS = MappingProxyType({
    "metal": BackendRequirement(
        precision=frozenset({"float32"}), mixed_precision=False,
    ),
})

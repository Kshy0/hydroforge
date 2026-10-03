"""Typed model capability contracts; free-form requirement dicts are forbidden."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from hydroforge.core.validation import HydroForgeModel

MODEL_OWNED_MODULE_FIELDS = (
    "opened_modules",
    "rank",
    "device",
    "precision",
    "mixed_precision",
    "metal_emulation",
    "ensemble_size",
)


class BackendRequirement(HydroForgeModel):
    """Model-wide restrictions not already defined by the backend runtime.

    Block-size limits bind every launch width the model resolves, including
    framework statistics kernels, which use the backend's default width unless
    a model-wide block size is configured.  ``default_block_size`` is that
    model-wide width when ``model.block_size`` is omitted, and
    ``capture=False`` keeps ``execution_mode="auto"`` eager on this
    backend. ``default_mixed_precision`` overrides the backend default when
    the caller leaves ``mixed_precision`` unset. The runtime limits of each
    backend remain facts of :class:`hydroforge.platform.Backend`.
    """

    precision: frozenset[Literal["float32", "float64"]] | None = Field(
        default=None, min_length=1
    )
    mixed_precision: bool = True
    default_mixed_precision: bool | None = None
    ensemble: bool = True
    min_block_size: int | None = Field(default=None, ge=1)
    max_block_size: int | None = Field(default=None, ge=1)
    block_size: int | None = Field(default=None, ge=1)
    default_block_size: int | None = Field(default=None, ge=1)
    capture: bool = True

    @model_validator(mode="after")
    def _validate_requirement(self) -> Self:
        if (
            self.min_block_size is not None
            and self.max_block_size is not None
            and self.min_block_size > self.max_block_size
        ):
            raise ValueError("backend block-size range is empty")
        for label, value in (
            ("fixed", self.block_size),
            ("default", self.default_block_size),
        ):
            if value is not None and (
                self.min_block_size is not None
                and value < self.min_block_size
                or self.max_block_size is not None
                and value > self.max_block_size
            ):
                raise ValueError(f"{label} backend block size is outside its range")
        return self

    def _validate_precision(
        self,
        precision: str,
        mixed_precision: bool,
        *,
        backend: str,
    ) -> None:
        """Validate model precision against one runtime or model restriction."""

        if self.precision is not None and precision not in self.precision:
            raise ValueError(
                f"backend {backend!r} requires precision in {self.precision}, "
                f"got {precision!r}"
            )
        if not self.mixed_precision and mixed_precision:
            raise ValueError(f"backend {backend!r} does not support mixed precision")


class ModuleRequirement(HydroForgeModel):
    """Restrictions introduced only when one optional module is open.

    ``clock`` marks a module that records absolute times and therefore needs
    a ``simulation_schedule`` or an ``initial_time``.
    """

    ensemble: bool = True
    clock: bool = False


DEFAULT_BACKEND_REQUIREMENT = BackendRequirement()
DEFAULT_MODULE_REQUIREMENT = ModuleRequirement()

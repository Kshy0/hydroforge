# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Typed model capability contracts; free-form requirement dicts are forbidden."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from hydroforge.core.validation import HydroForgeModel

MODEL_OWNED_MODULE_FIELDS = (
    "opened_modules",
    "rank",
    "device",
    "precision",
    "storage_precision",
    "matmul_tf32",
    "mixed_precision",
    "metal_emulation",
    "ensemble_size",
    "options",
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

    precision: (
        frozenset[Literal["float64", "float32", "tf32", "bfloat16", "float16"]] | None
    ) = Field(default=None, min_length=1)
    """Precision tiers the model's kernels serve on this backend, compute
    and storage alike; ``None`` serves float64, float32 and tf32. 16-bit
    tiers must be named: kernels then read and write 16-bit elements."""
    mixed_precision: bool = True
    default_mixed_precision: bool | None = None
    ensemble: bool = True
    min_block_size: int | None = Field(default=None, ge=1)
    max_block_size: int | None = Field(default=None, ge=1)
    block_size: int | None = Field(default=None, ge=1)
    default_block_size: int | None = Field(default=None, ge=1)
    capture: bool = True

    @field_validator("precision", mode="before")
    @classmethod
    def _freeze_precision(cls, value: Any) -> Any:
        """Accept a set literal for the strict frozenset declaration."""

        return frozenset(value) if type(value) is set else value

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
        if (
            self.block_size is not None
            and self.default_block_size is not None
            and self.block_size != self.default_block_size
        ):
            raise ValueError(
                "default backend block size must equal the fixed block size"
            )
        return self

    def validate_precision(
        self,
        precisions: Any,
        mixed_precision: bool | None = None,
        *,
        backend: str,
    ) -> None:
        """Validate the model's tiers (:class:`~hydroforge.contracts.precision.
        Precisions`, or a compute tier with ``mixed_precision``) against this
        model restriction."""

        if isinstance(precisions, str):
            # A bare compute tier: the tier and the promotion request.
            if self.precision is not None and precisions not in self.precision:
                raise ValueError(
                    f"backend {backend!r} requires precision in "
                    f"{sorted(self.precision)}, got {precisions!r}"
                )
            if not self.mixed_precision and mixed_precision:
                raise ValueError(
                    f"backend {backend!r} does not support mixed precision"
                )
            return

        served = (
            frozenset(("float64", "float32", "tf32"))
            if self.precision is None
            else self.precision
        )
        for role, tier in (
            ("precision", precisions.compute),
            ("storage_precision", precisions.storage),
        ):
            if tier not in served:
                raise ValueError(
                    f"backend {backend!r} requires precision in "
                    f"{sorted(served)} for this model, got {role}={tier!r}"
                )
        if not self.mixed_precision and precisions.mixed:
            raise ValueError(f"backend {backend!r} does not support mixed precision")


class ModuleRequirement(HydroForgeModel):
    """Restrictions introduced only when one optional module is open.

    ``clock`` marks a module that records absolute times and therefore needs
    a ``simulation_schedule`` or an ``initial_time``. ``halo`` states that the
    model's managed steps exchange this module's halo ghost rows (see
    :mod:`hydroforge.parallel.halo`): partition groups may split references to
    the partition key only while every open module declares it.
    """

    ensemble: bool = True
    clock: bool = False
    halo: bool = False


DEFAULT_BACKEND_REQUIREMENT = BackendRequirement()
DEFAULT_MODULE_REQUIREMENT = ModuleRequirement()

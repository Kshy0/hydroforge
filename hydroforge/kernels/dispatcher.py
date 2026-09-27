"""Shared dispatcher contracts for already validated kernel arguments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hydroforge.contracts.kernels import (
    BackendLoweringSpec,
    BufferDTypeABI,
    KernelMetadata,
    KernelSpec,
)


def _reject_unproven_uint32_runtime_scalars(
    spec: KernelSpec,
    backend: str,
) -> None:
    """Reject a backend that cannot prove fixed-width unsigned semantics."""

    names = sorted(
        name for name, kind in spec.runtime_scalars.items() if kind == "uint32"
    )
    if names:
        raise TypeError(
            f"{spec.name}: {backend} backend does not support canonical "
            f"uint32 runtime scalar(s) {names}; it cannot prove a fixed-width "
            "unsigned native representation"
        )


def _empty_launch() -> None:
    return None


class _SpecializedDispatcher:
    """Non-callable backend declaration with one trusted specializer."""

    def __init__(
        self,
        metadata: KernelMetadata,
        lowering: BackendLoweringSpec,
        specializer: Callable,
    ) -> None:
        self.__hydroforge_kernel__ = metadata
        self.__hydroforge_lowering__ = lowering
        self._specializer = specializer

    def specialize(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        return self._specializer(arguments, buffer_dtypes=buffer_dtypes)

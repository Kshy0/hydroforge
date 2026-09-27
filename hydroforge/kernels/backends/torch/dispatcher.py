"""Canonical-ABI dispatch for PyTorch kernels."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.kernels import (
    BackendLoweringSpec,
    BufferDTypeABI,
    KernelSpec,
    validate_launch_extent,
)
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.context import resolve_factory_spec
from hydroforge.kernels.dispatcher import (
    _empty_launch,
    _reject_unproven_uint32_runtime_scalars,
)


def _torch_compile(fn: Callable) -> Callable:
    """Apply torch.compile with inference-optimized settings.

    All physics kernels mutate inputs via ``.copy_()`` / indexed assignment,
    so ``reduce-overhead`` mode (which relies on internal CUDA graphs) can
    never actually use its main optimisation and only produces warnings.
    We use ``fullgraph=True`` so that compilation errors surface at the
    first call rather than lazily on a rare code-path hours later.
    """
    import torch

    return torch.compile(fn, fullgraph=True)


class TorchDispatcher:
    """Strict canonical-ABI dispatcher for a native PyTorch implementation."""

    def __init__(
        self,
        kernel: Callable,
        spec: KernelSpec,
        *,
        compile: bool = True,
    ) -> None:
        _reject_unproven_uint32_runtime_scalars(spec, "Torch")
        signature = inspect.signature(kernel)
        parameters = tuple(signature.parameters)
        if parameters != spec.parameters:
            raise TypeError(
                f"{spec.name}: torch signature {parameters!r} must exactly match "
                f"KernelSpec {spec.parameters!r}"
            )
        if any(
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            for parameter in signature.parameters.values()
        ):
            raise TypeError(
                f"{spec.name}: torch kernels must accept canonical arguments "
                "by keyword and may not use positional-only, *args, or **kwargs"
            )
        self._kernel = _torch_compile(kernel) if compile else kernel
        self.spec = spec
        self._parameters = frozenset(spec.parameters)
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    def specialize(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable:
        """Return a zero-argument launch for an already validated call."""
        del buffer_dtypes
        extent = validate_launch_extent(
            self.spec.name,
            self.spec.size_key,
            arguments,
        )
        if extent == 0:
            return _empty_launch
        static = {
            name: value for name, value in arguments.items() if name in self._parameters
        }

        def launch():
            return self._kernel(**static)

        return launch


class _TorchDispatcherDeclaration(HydroForgeModel):
    kernel: Callable
    spec: KernelSpec | None = None
    compile: bool = True

    _dispatcher: TorchDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            spec = resolve_factory_spec(self.spec, factory="make_torch_dispatcher")
            self._dispatcher = TorchDispatcher(
                self.kernel,
                spec,
                compile=self.compile,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self


def make_torch_dispatcher(
    kernel: Callable,
    spec: KernelSpec | None = None,
    *,
    compile: bool = True,
) -> TorchDispatcher:
    """Build a formal Torch backend from the active canonical Spec."""

    return _TorchDispatcherDeclaration(
        kernel=kernel,
        spec=spec,
        compile=compile,
    )._dispatcher

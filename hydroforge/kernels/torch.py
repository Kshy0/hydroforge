# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""PyTorch kernel declarations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch

from hydroforge.kernels.registry import (
    KernelCall,
    KernelDeclaration,
    KernelImplementation,
    Launch,
)
from hydroforge.kernels.spec import KernelSpec
from hydroforge.platform.backend import Backend


class _TorchImplementation(KernelImplementation):
    def __init__(
        self, spec: KernelSpec, backend: Backend, declaration: TorchKernel
    ) -> None:
        super().__init__(spec, backend)
        signature = inspect.signature(declaration.function)
        if tuple(signature.parameters) != spec.parameter_names:
            raise TypeError(
                f"{spec.name}: torch signature {tuple(signature.parameters)!r} "
                f"must exactly match KernelSpec {spec.parameter_names!r}"
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
        # ``fullgraph`` surfaces compilation errors at the first call rather
        # than lazily on a rare code path; kernels mutate their inputs, so
        # ``reduce-overhead`` CUDA graphs would never apply.
        self.function = (
            torch.compile(declaration.function, fullgraph=True)
            if declaration.compile
            else declaration.function
        )

    def _compile(self, call: KernelCall) -> Launch:
        function = self.function
        arguments = {name: call.arguments[name] for name in self.spec.parameters}

        def launch() -> Any:
            return function(**arguments)

        return launch


class TorchKernel(KernelDeclaration):
    """A PyTorch function taking exactly the spec's parameters by keyword.

    It runs under any execution backend, including inside CUDA graph capture;
    ``compile`` wraps it in ``torch.compile(fullgraph=True)``.
    """

    function: Callable[..., Any]
    compile: bool = True

    def __init__(self, function: Callable[..., Any], /, **fields: Any) -> None:
        super().__init__(function=function, **fields)

    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        return _TorchImplementation(spec, backend, self)


__all__ = ["TorchKernel"]

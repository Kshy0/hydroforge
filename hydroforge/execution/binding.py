# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Kernel argument values of one constructed model.

:class:`KernelBinder` completes the calls of registered kernels from the
compiled :class:`~hydroforge.compiler.kernel_binding.KernelBindingPlan` and is
the call sink of a managed step outside recordings, where it launches at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from hydroforge.compiler.fields import BindingSource
from hydroforge.compiler.kernel_binding import (
    Batched,
    FieldValue,
    Fixed,
    Gated,
    KernelBinding,
    KernelBindingPlan,
    ModuleFlagValue,
    Source,
    StepFieldValue,
    Unresolved,
)
from hydroforge.contracts.fields import precision_dtype
from hydroforge.core.errors import error_message
from hydroforge.kernels.registry import BackendRegistry, KernelCall
from hydroforge.kernels.spec import KernelSpec, KernelWorkspace

if TYPE_CHECKING:
    from hydroforge.execution.session import ModelRuntime


class UnboundKernelArgument(KeyError):
    """A canonical ABI parameter has no owner in the model namespace."""


@dataclass(frozen=True, slots=True)
class BindingResolution:
    """One resolved parameter value, its binding rule and its owner."""

    value: Any
    source: str
    owner: str | None = None


class KernelBinder:
    """Complete kernel calls from one model and launch them outside recordings."""

    recording = False

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self.plan = KernelBindingPlan(runtime.plan)
        # Registries are nominal operators keyed by identity; each value
        # retains its registry, so an ``id`` cannot be reused while cached.
        # Calls fully bound by the model reuse their values and launch until
        # :meth:`invalidate`.
        self._complete_cache: dict[int, tuple[Any, Mapping[str, Any], Mapping]] = {}
        self._launch_cache: dict[int, tuple[Any, Any]] = {}
        self._workspaces: dict[str, tuple[KernelWorkspace, torch.Tensor | None]] = {}
        # Retain the source objects as well as identities to prevent ID reuse.
        self._workspace_sources: dict[int, torch.Tensor] = {}

    def invalidate(self) -> None:
        """Drop bindings whose scalar specializations may have changed."""

        self._complete_cache.clear()
        self._launch_cache.clear()
        self._workspaces.clear()
        self._workspace_sources.clear()

    def content_requires_rebind(self, tensors: Iterable[torch.Tensor]) -> bool:
        """Whether a content transaction changes a cached CSR dependency."""
        return any(id(tensor) in self._workspace_sources for tensor in tensors)

    def call(self, registry: BackendRegistry, supplied: dict[str, Any]) -> None:
        """Launch one call of a managed step outside any recording."""

        if not supplied:
            cached = self._launch_cache.get(id(registry))
            if cached is not None:
                cached[1]()
                return
        launch = self.bind(registry, supplied).compile()
        if not supplied:
            self._launch_cache[id(registry)] = (registry, launch)
        launch()

    def bind(
        self, registry: BackendRegistry, supplied: Mapping[str, Any]
    ) -> KernelCall:
        """Complete and validate one call for the model backend, uncompiled."""

        arguments, buffer_dtypes = self.arguments(registry, supplied)
        plan = self.runtime.plan
        try:
            implementation = registry.implementation(
                plan.backend, precision=plan.precision
            )
            return implementation.call(arguments, buffer_dtypes=buffer_dtypes)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(error_message(error)) from error

    def arguments(
        self, registry: BackendRegistry, supplied: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, torch.dtype]]:
        """The model-bound arguments and buffer dtypes of one call."""

        if (
            not supplied
            and (cached := self._complete_cache.get(id(registry))) is not None
        ):
            return cached[1], cached[2]
        spec = registry.spec_for(self.runtime.plan.backend)
        try:
            binding = self.plan.bind(spec)
            for name in supplied:
                source = binding.sources[name]
                if isinstance(source, StepFieldValue):
                    raise TypeError(
                        f"{spec.name}.{name} is already resolved from step_field; omit the redundant call-site value"
                    )
            for name, source in binding.sources.items():
                if isinstance(source, StepFieldValue):
                    continue
                if name in supplied:
                    continue
                if isinstance(source, FieldValue) and not source.owners:
                    continue
                self.resolve(binding, name)
            if spec.step_fields:
                self.runtime.execution.step_fields.bind_many(spec.step_fields.values())
            if supplied:
                arguments = self._complete_supplied(spec, binding, supplied)
                buffer_dtypes = self._buffer_dtypes(spec, binding, arguments)
            else:
                cached = self._complete_cache.get(id(registry))
                if cached is None:
                    values = {
                        name: self.resolve(binding, name).value
                        for name in spec.parameters
                    }
                    values["BLOCK_SIZE"] = binding.block_size
                    cached = (
                        registry,
                        MappingProxyType(values),
                        self._buffer_dtypes(spec, binding, values),
                    )
                    self._complete_cache[id(registry)] = cached
                arguments, buffer_dtypes = cached[1], cached[2]
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(error_message(error)) from error
        return arguments, buffer_dtypes

    def _complete_supplied(
        self, spec: KernelSpec, binding: KernelBinding, supplied: Mapping[str, Any]
    ) -> dict[str, Any]:
        for name in supplied:
            try:
                resolution = self.resolve(binding, name)
            except UnboundKernelArgument:
                if name.startswith(("HAS_", "batched_")):
                    raise
                continue
            raise TypeError(
                f"{spec.name}.{name} is already resolved from {resolution.source} "
                f"{resolution.owner!r}; omit the redundant call-site value"
            )
        arguments = dict(supplied)
        for name in spec.parameters:
            if name not in arguments:
                arguments[name] = self.resolve(binding, name).value
        arguments["BLOCK_SIZE"] = binding.block_size
        return arguments

    def _buffer_dtypes(
        self, spec: KernelSpec, binding: KernelBinding, arguments: Mapping[str, Any]
    ) -> Mapping[str, torch.dtype]:
        result: dict[str, torch.dtype] = {}
        for name in spec.buffers:
            declared = binding.buffer_dtypes[name]
            if isinstance(declared, Unresolved):
                raise declared.error(declared.message)
            value = arguments[name]
            if name in spec.step_fields:
                result[name] = declared
            elif isinstance(value, torch.Tensor):
                if declared is not None and value.dtype != declared:
                    raise TypeError(
                        f"{spec.name}.{name} has dtype {value.dtype}, but its "
                        f"model field declares {declared}"
                    )
                result[name] = value.dtype if declared is None else declared
            elif value is not None or name not in spec.optional:
                raise TypeError(f"{spec.name}.{name} has no concrete tensor dtype")
            elif declared is None:
                raise TypeError(
                    f"{spec.name}.{name} is disabled and its dtype cannot be "
                    "resolved from a declared model/module field"
                )
            else:
                result[name] = declared
        return MappingProxyType(result)

    def resolve(self, binding: KernelBinding, name: str) -> BindingResolution:
        """Read the value of one parameter the call site omits."""

        return self._resolve(binding, name, binding.sources[name])

    def _resolve(
        self, binding: KernelBinding, name: str, source: Source
    ) -> BindingResolution:
        if isinstance(source, KernelWorkspace):
            return BindingResolution(
                self._workspace(binding, source), "workspace", source.key
            )
        if isinstance(source, Fixed):
            return BindingResolution(source.value, source.source, source.owner)
        if isinstance(source, StepFieldValue):
            field = source.field
            return BindingResolution(
                self.runtime.execution.step_fields.bind(field),
                "step_field",
                field.source,
            )
        if isinstance(source, Gated):
            enabled = self._resolve(
                binding, source.feature, binding.sources[source.feature]
            ).value
            if not enabled:
                return BindingResolution(source.disabled, "optional", source.feature)
            return self._resolve(binding, name, source.source)
        if isinstance(source, ModuleFlagValue):
            value = getattr(self.runtime.modules[source.module], source.field)
            if type(value) is not bool:
                raise TypeError(
                    f"kernel feature {source.parameter!r} source "
                    f"{source.module}.{source.field} must be an exact bool, "
                    f"got {type(value).__name__}"
                )
            return BindingResolution(value, "compile_time", source.parameter)
        if isinstance(source, Unresolved):
            raise source.error(source.message)
        owners = self._owners(source.field, source.owners)
        if isinstance(source, FieldValue) and source.optional and not owners:
            return BindingResolution(None, "optional", None)
        if len(owners) != 1:
            if owners:
                raise ValueError(
                    f"kernel argument {name!r} is ambiguous across "
                    f"{[owner for owner, _object in owners]}; kernel ABI names "
                    "must match unique fields"
                )
            raise UnboundKernelArgument(
                f"kernel argument {name!r} has no model/module field; rename "
                "the ABI/field to match or supply it explicitly at the "
                "recording call site"
            )
        owner, holder = owners[0]
        label = f"{owner}.{source.field}"
        if isinstance(source, Batched):
            return BindingResolution(holder.is_batched(source.field), "batched", label)
        value = getattr(holder, source.field)
        if isinstance(value, Enum):
            value = value.value
        return BindingResolution(
            value, "optional" if source.optional else "field", label
        )

    def _workspace(
        self, binding: KernelBinding, declaration: KernelWorkspace
    ) -> torch.Tensor | None:
        cached = self._workspaces.get(declaration.key)
        if cached is not None:
            if cached[0] != declaration:
                raise ValueError(
                    f"conflicting workspace declarations for {declaration.key!r}"
                )
            return cached[1]
        plan = self.runtime.plan
        if (
            declaration.module is not None
            and declaration.module not in plan.spec.modules
        ):
            raise ValueError(
                f"workspace {declaration.key!r} references unknown module {declaration.module!r}"
            )
        if (declaration.emulated_only and plan.metal_emulation == "native") or (
            declaration.module is not None and declaration.module not in plan.modules
        ):
            self._workspaces[declaration.key] = (declaration, None)
            return None

        def value(field: str) -> Any:
            if field == "ensemble_size":
                return self.plan.ensemble_size
            return self._resolve(
                binding, field, FieldValue(field, plan.fields.binding.get(field, ()))
            ).value

        def dimension(field: str | int) -> int:
            result = value(field) if isinstance(field, str) else field
            if type(result) is not int or result < 0:
                raise ValueError(
                    f"workspace dimension {field!r} must be a non-negative exact int"
                )
            return result

        shape = tuple(dimension(field) for field in declaration.shape)
        dtype = precision_dtype(declaration.dtype, plan.dtype)
        if prod(shape) > (2**63 - 1) // dtype.itemsize:
            raise OverflowError(
                f"workspace {declaration.key!r} exceeds int64 byte size"
            )
        if declaration.initialize == "zeros":
            tensor = torch.zeros(shape, device=plan.device, dtype=dtype)
        else:
            source = value(declaration.source)
            if (
                not isinstance(source, torch.Tensor)
                or source.ndim != 1
                or source.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("CSR source must be a one-dimensional integer tensor")
            targets = dimension(declaration.target_count)
            indices = source.to(device="cpu", dtype=torch.int64)
            if indices.numel() > 2**31 - 1 or targets > 2**31 - 1:
                raise OverflowError("CSR workspace exceeds int32 index range")
            if indices.numel() and (
                int(indices.min()) < 0 or int(indices.max()) >= targets
            ):
                raise ValueError("CSR source indices are outside the target range")
            if declaration.initialize == "csr_offsets":
                counts = torch.bincount(indices, minlength=targets)
                result = counts.cumsum(0) - counts
            else:
                result = torch.argsort(indices, stable=True)
            if tuple(result.shape) != shape:
                raise ValueError(
                    f"workspace {declaration.key!r} shape disagrees with its topology"
                )
            tensor = result.to(device=plan.device, dtype=dtype)
            self._workspace_sources[id(source)] = source
        self._workspaces[declaration.key] = (declaration, tensor)
        return tensor

    def _owners(
        self, field: str, candidates: tuple[BindingSource, ...]
    ) -> list[tuple[str, Any]]:
        """Owners holding ``field``; buffer virtuals bind once materialized."""

        runtime = self.runtime
        owners = []
        for source in candidates:
            if source.owner == "model":
                holder = runtime.owner
            elif source.owner.startswith("model."):
                holder = getattr(runtime.owner, source.owner.removeprefix("model."))
            else:
                holder = runtime.modules[source.owner]
            if not source.lazy or field in holder.__dict__:
                owners.append((source.owner, holder))
        return owners

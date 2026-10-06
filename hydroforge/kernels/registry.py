# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Kernel registries, implementation declarations and validated calls.

A :class:`BackendRegistry` pairs one :class:`KernelSpec` with a declaration
per execution backend.  The registry resolves the spec's precision, checks its
scalar kinds against the backend once, and builds the declaration into a
:class:`KernelImplementation`.  An implementation validates one complete call
into a :class:`KernelCall`, which compiles to a zero-argument launch.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, ClassVar

import torch
from pydantic import AfterValidator, Field, InstanceOf, PrivateAttr, model_validator

from hydroforge.core.devices import devices_match
from hydroforge.core.validation import FrozenMapping, HydroForgeModel
from hydroforge.kernels.calls import current_sink
from hydroforge.kernels.spec import KernelSpec
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.platform.backend import Backend, BackendName, Toolchain, backend_named

Launch = Callable[[], Any]


def empty_launch() -> None:
    return None


def _named_callable(value: Any) -> Any:
    if value is not None and (
        not callable(value) or not getattr(value, "__name__", "").isidentifier()
    ):
        raise ValueError("launch plans and checks must be named callables")
    return value


NamedCallable = Annotated[Any, AfterValidator(_named_callable)]


def named_parameters(spec: KernelSpec, function: Callable) -> tuple[str, ...]:
    """Canonical names ``function`` receives; ``**values`` receives all of them.

    ``BLOCK_SIZE`` counts as a canonical name.  Parameters must be plain
    keywords without defaults and must name parameters of ``spec``.
    """

    parameters = inspect.signature(function).parameters.values()
    unsupported = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        not in (
            parameter.KEYWORD_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.VAR_KEYWORD,
        )
        or parameter.default is not parameter.empty
    ]
    if unsupported:
        raise ValueError(
            f"{spec.name}: {function.__name__} parameters must be plain "
            f"keyword parameters without defaults: {unsupported}"
        )
    names = tuple(
        parameter.name
        for parameter in parameters
        if parameter.kind is not parameter.VAR_KEYWORD
    )
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
        names += tuple(
            name for name in (*spec.parameter_names, "BLOCK_SIZE") if name not in names
        )
    unknown = set(names).difference(spec.parameters, {"BLOCK_SIZE"})
    if unknown:
        raise ValueError(
            f"{spec.name}: {function.__name__} takes names outside its "
            f"KernelSpec: {sorted(unknown)}"
        )
    return names


class KernelImplementation(ABC):
    """One declaration built for a resolved spec on one backend.

    Subclasses validate their native requirements in ``_validate``, describe
    their compilation in ``_requests`` and build the launch in ``_compile``;
    calls with an empty launch extent never reach ``_compile``.
    """

    def __init__(self, spec: KernelSpec, backend: Backend) -> None:
        self.spec = spec
        self.backend = backend

    def call(
        self,
        arguments: Mapping[str, Any],
        *,
        buffer_dtypes: Mapping[str, torch.dtype | None] | None = None,
    ) -> KernelCall:
        """Validate one complete call; ``BLOCK_SIZE`` defaults to the backend's."""

        spec = self.spec
        backend = self.backend
        arguments = dict(arguments)
        if "BLOCK_SIZE" not in arguments:
            arguments["BLOCK_SIZE"] = backend.block.resolve(
                None, kernel=spec.block_sizes.get(backend.name), backend=backend.name
            )
        supplied = set(arguments).difference({"BLOCK_SIZE"})
        if supplied != set(spec.parameters):
            raise ValueError(
                f"{spec.name} call ABI mismatch: "
                f"missing={sorted(set(spec.parameters) - supplied)}, "
                f"extra={sorted(supplied - set(spec.parameters))}"
            )
        backend.block.validate(arguments["BLOCK_SIZE"], backend=backend.name)
        spec.validate_host_values(arguments)
        tensors: list[tuple[str, torch.Tensor]] = []
        for name in spec.buffers:
            value = arguments[name]
            if value is None and name in spec.optional:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{spec.name}.{name} must be a tensor, got {type(value).__name__}"
                )
            if value.layout is not torch.strided or not value.is_contiguous():
                raise ValueError(
                    f"{spec.name}.{name} must be a contiguous strided tensor"
                )
            tensors.append((name, value))
        wrong = [
            f"{name}={tensor.device}"
            for name, tensor in tensors
            if not backend.accepts(tensor.device)
        ]
        if wrong:
            required = " or ".join(sorted(backend.devices))
            raise ValueError(
                f"{spec.name}: {backend.name} buffers must be on {required}; "
                f"got {', '.join(wrong)}"
            )
        if tensors:
            reference = tensors[0][1].device
            mismatched = [
                f"{name}={tensor.device}"
                for name, tensor in tensors[1:]
                if not devices_match(tensor.device, reference)
            ]
            if mismatched:
                raise ValueError(
                    f"{spec.name}: buffers must share one device; expected "
                    f"{reference}, got {', '.join(mismatched)}"
                )
        if buffer_dtypes is None:
            buffer_dtypes = {
                name: getattr(arguments[name], "dtype", None) for name in spec.buffers
            }
        elif set(buffer_dtypes) != set(spec.buffers):
            raise ValueError(
                f"{spec.name}: buffer dtype ABI mismatch: "
                f"missing={sorted(set(spec.buffers) - set(buffer_dtypes))}, "
                f"extra={sorted(set(buffer_dtypes) - set(spec.buffers))}"
            )
        for name, dtype in buffer_dtypes.items():
            value = arguments[name]
            if dtype is None:
                if value is None and name in spec.optional:
                    continue
                raise ValueError(f"{spec.name}.{name} buffer dtype must be torch.dtype")
            if value is not None and value.dtype != dtype:
                raise ValueError(
                    f"{spec.name}.{name} call declares {dtype}, but the tensor "
                    f"has dtype {value.dtype}"
                )
        call = KernelCall(
            self, MappingProxyType(arguments), MappingProxyType(dict(buffer_dtypes))
        )
        self._validate(call)
        return call

    def specialize(
        self,
        arguments: Mapping[str, Any],
        *,
        buffer_dtypes: Mapping[str, torch.dtype | None] | None = None,
    ) -> Launch:
        """Validate and compile one call into a zero-argument launch."""

        return self.call(arguments, buffer_dtypes=buffer_dtypes).compile()

    def __call__(self, **arguments: Any) -> None:
        """Validate, compile and execute one call."""

        self.specialize(arguments)()

    def _validate(self, call: KernelCall) -> None:
        """Reject a call the native implementation cannot serve."""

    def _requests(self, call: KernelCall) -> tuple[CompileRequest, ...]:
        return ()

    @abstractmethod
    def _compile(self, call: KernelCall) -> Launch: ...


@dataclass(frozen=True, slots=True)
class KernelCall:
    """One complete, validated and not yet compiled kernel call.

    Recording keeps calls uncompiled so that
    :func:`hydroforge.kernels.toolchain.compile_calls` can compile a whole
    program's kernels in one batch per toolchain.
    """

    implementation: KernelImplementation
    arguments: Mapping[str, Any]
    buffer_dtypes: Mapping[str, torch.dtype | None]

    @property
    def empty(self) -> bool:
        return self.implementation.spec.launch_extent(self.arguments) == 0

    def requests(self) -> tuple[CompileRequest, ...]:
        """The compilations this call needs before :meth:`compile`."""

        return () if self.empty else self.implementation._requests(self)

    def compile(self) -> Launch:
        """Compile (or reuse) the launch of this call."""

        launch = empty_launch if self.empty else self.implementation._compile(self)
        for wrap in _INTERCEPTS.get():
            launch = wrap(self, launch)
        return launch


_INTERCEPTS: ContextVar[tuple[Callable[[KernelCall, Launch], Launch], ...]] = (
    ContextVar("hydroforge_kernel_intercepts", default=())
)


@contextmanager
def intercepting(wrap: Callable[[KernelCall, Launch], Launch]) -> Iterator[None]:
    """Wrap every launch compiled inside the block."""

    token = _INTERCEPTS.set((*_INTERCEPTS.get(), wrap))
    try:
        yield
    finally:
        _INTERCEPTS.reset(token)


class KernelDeclaration(HydroForgeModel, ABC):
    """How one backend implements a kernel, independent of its spec.

    ``toolchain`` is the backend toolchain a native declaration requires;
    ``None`` runs under any backend.
    """

    toolchain: ClassVar[Toolchain | None] = None

    def build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        """Check this declaration against ``spec`` and ``backend`` once.

        Scalar kinds the backend cannot represent are rejected here; an
        unresolved ``precision`` kind is left to the implementation.
        """

        if self.toolchain is not None and backend.toolchain != self.toolchain:
            raise ValueError(
                f"{spec.name}: {type(self).__name__} requires a "
                f"{self.toolchain!r} backend, got {backend.name!r}"
            )
        backend.validate_scalars(
            spec.name,
            {
                name: kind
                for name, kind in (
                    *spec.compile_time.items(),
                    *spec.runtime_scalars.items(),
                )
                if kind != "precision"
            },
        )
        return self._build(spec, backend)

    @abstractmethod
    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation: ...


class BackendRegistry(HydroForgeModel):
    """One logical kernel and its implementation per execution backend.

    A value is a declaration or a zero-argument factory returning one; a
    factory defers imports a backend may lack.  Calling the registry inside a
    managed step records or launches the kernel; inside a model's
    ``initialize_model_state`` and ``@between_steps`` bodies it binds and
    launches eagerly. ``backend_specs`` explicitly
    declares backend workspace bindings; model calls keep the shared interface.
    """

    spec: KernelSpec
    implementations: FrozenMapping[
        BackendName, InstanceOf[KernelDeclaration] | Callable[[], KernelDeclaration]
    ]
    backend_specs: FrozenMapping[BackendName, KernelSpec] = Field(default_factory=dict)
    _built: dict[tuple[str, str | None], KernelImplementation] = PrivateAttr(
        default_factory=dict
    )

    def __init__(
        self,
        spec: KernelSpec,
        implementations: Mapping[str, KernelDeclaration | Callable[[], Any]],
        /,
        *,
        backend_specs: Mapping[str, KernelSpec] | None = None,
    ) -> None:
        super().__init__(
            spec=spec,
            implementations=implementations,
            backend_specs={} if backend_specs is None else backend_specs,
        )

    @model_validator(mode="after")
    def _validate_backend_specs(self) -> BackendRegistry:
        for backend, spec in self.backend_specs.items():
            if backend not in self.implementations:
                raise ValueError(
                    f"{self.name}: spec for unregistered backend {backend!r}"
                )
            if spec.name != self.spec.name or spec.size != self.spec.size:
                raise ValueError(
                    "backend specs must preserve the kernel name and launch size"
                )
        return self

    def spec_for(self, backend: Backend | str) -> KernelSpec:
        """Explicit backend ABI, or the shared spec when no override is declared."""

        name = backend.name if isinstance(backend, Backend) else backend
        return self.backend_specs.get(name, self.spec)

    @property
    def name(self) -> str:
        return self.spec.name

    def __call__(self, **arguments: Any) -> None:
        sink = current_sink()
        if arguments:
            if "BLOCK_SIZE" in arguments:
                raise ValueError(
                    f"{self.name}.BLOCK_SIZE is compiler-owned; configure the "
                    "model block size instead"
                )
            unknown = set(arguments).difference(self.spec.parameters)
            if unknown:
                raise ValueError(
                    f"{self.name} received arguments outside its KernelSpec: "
                    f"{sorted(unknown)}"
                )
        if sink is None:
            raise ValueError(
                f"{self.name} may be called only inside a managed model step, "
                "initialize_model_state or a @between_steps method"
            )
        sink.call(self, arguments)

    def implementation(
        self, backend: Backend | str, *, precision: str | None = None
    ) -> KernelImplementation:
        """Build the implementation for one backend, once per precision."""

        if not isinstance(backend, Backend):
            backend = backend_named(backend)
        selected_spec = self.spec_for(backend)
        key = (backend.name, precision if selected_spec.uses_precision else None)
        built = self._built.get(key)
        if built is None:
            declaration = self.implementations.get(backend.name)
            if declaration is None:
                raise ValueError(
                    f"Backend {backend.name!r} is not registered for {self.name}; "
                    f"available={tuple(self.implementations)}"
                )
            spec = selected_spec.resolved(key[1])
            if not isinstance(declaration, KernelDeclaration):
                declaration = declaration()
                if not isinstance(declaration, KernelDeclaration):
                    raise TypeError(
                        f"{self.name}: the {backend.name} factory must return a "
                        f"kernel declaration, got {type(declaration).__name__}"
                    )
            built = self._built[key] = declaration.build(spec, backend)
        return built


__all__ = [
    "BackendRegistry",
    "KernelCall",
    "KernelDeclaration",
    "KernelImplementation",
]

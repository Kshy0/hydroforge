"""Triton kernel declarations and their launch adapters."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, ClassVar, Literal, Self

import torch
from pydantic import model_validator

from hydroforge.core.naming import Identifier
from hydroforge.kernels.codegen.types import SCALARS
from hydroforge.kernels.registry import (
    KernelCall,
    KernelDeclaration,
    KernelImplementation,
    Launch,
    NamedCallable,
    empty_launch,
    named_parameters,
)
from hydroforge.kernels.spec import KernelSpec
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain.triton import (
    adopt,
    launch_options,
    launch_variant,
    precision_variant,
    triton_precision_context,
    warming,
    warmup_request,
)
from hydroforge.platform.backend import TRITON, Backend
from hydroforge.platform.triton_driver import proven_triton_device

_FLOAT_KINDS = frozenset(("float32", "float64"))


def launch_triton_kernel(kernel: Any, grid: Any, *, physics: bool = True) -> Callable:
    """Return a precision-aware launch proxy for an inner Triton kernel.

    This is intended for compound-program helpers.  The returned callable
    binds the actual launch arguments first, identifies declared floating
    scalar parameters, and dispatches a cached fp32/fp64 JIT variant.  Outside
    a :class:`TritonProgram` the scalar ABI remains unchanged.  Launches use
    the HydroForge math defaults of physics or (``physics=False``) framework
    kernels, which explicit launch options override, and compile under the
    HydroForge compile policy.
    """

    def launch(*args: Any, **kwargs: Any):
        selected = launch_variant(kernel, args, kwargs)
        options = {**launch_options(selected, physics=physics), **kwargs}
        if warming():
            return selected.warmup(*args, grid=grid, **options)
        return selected[grid](*args, **options)

    return launch


def _precision(spec: KernelSpec) -> tuple[str | None, frozenset[str]]:
    """The resolved scalar precision of ``spec`` and the names it types."""

    if spec.uses_precision:
        raise TypeError(
            f"{spec.name}: a Triton precision-dependent KernelSpec must be "
            "resolved to float32 or float64 before it is built"
        )
    kinds = {**spec.compile_time, **spec.runtime_scalars}
    names = frozenset(name for name, kind in kinds.items() if kind in _FLOAT_KINDS)
    precisions = {kinds[name] for name in names}
    return (next(iter(precisions)) if len(precisions) == 1 else None), names


def _native_parameters(kernel: Any, spec: KernelSpec, label: str) -> frozenset[str]:
    parameters = tuple(
        name for name in getattr(kernel, "arg_names", ()) if name != "BLOCK_SIZE"
    )
    if len(parameters) != len(set(parameters)):
        raise TypeError(f"{spec.name}: {label} Triton kernel has duplicate parameters")
    return frozenset(parameters)


def _validate_kernel(
    kernel: Any,
    spec: KernelSpec,
    label: str,
    *,
    complete: bool,
    check_annotations: bool = True,
) -> None:
    """Check one Triton kernel's parameters and scalar annotations.

    A shared variant may omit scalars used only to select or index a batched
    layout; buffers are never optional, and the consumed subset must be a
    valid projection of ``spec``.
    """

    observed = _native_parameters(kernel, spec, label)
    extra = observed.difference(spec.parameters)
    if extra:
        raise TypeError(
            f"{spec.name}: {label} Triton kernel has parameters outside "
            f"KernelSpec: {sorted(extra)}"
        )
    missing = set(spec.parameters).difference(observed)
    if missing.intersection(spec.buffers):
        raise TypeError(
            f"{spec.name}: {label} Triton kernel omits canonical buffers: "
            f"{sorted(missing.intersection(spec.buffers))}"
        )
    if complete and missing:
        raise TypeError(
            f"{spec.name}: {label} Triton kernel must consume the complete "
            f"canonical ABI: missing={sorted(missing)}"
        )
    if missing:
        spec.project(omit=tuple(name for name in spec.parameters if name in missing))
    # Precision-dependent values are lowered to typed runtime scalars by
    # ``precision_variant``; a stale ``tl.constexpr`` would bypass that.
    if not check_annotations:
        return
    names = _precision(spec)[1].intersection(observed)
    if not names:
        return
    kinds = {**spec.compile_time, **spec.runtime_scalars}
    parameters = {
        parameter.name: parameter for parameter in getattr(kernel, "params", ())
    }
    expected = {name: SCALARS[kinds[name]].triton for name in names}
    invalid = sorted(
        name
        for name in names
        if name not in parameters
        or getattr(parameters[name], "is_constexpr", False)
        or parameters[name].annotation_type != expected[name]
    )
    if invalid:
        detail = "tl.float64" if "fp64" in expected.values() else "tl.float32"
        raise TypeError(
            f"{spec.name}: {label} Triton precision runtime scalar ABI requires "
            f"explicit {detail} annotations; expected={expected}, invalid={invalid}"
        )


def _check_arguments(spec: KernelSpec, check: Callable | None) -> tuple[str, ...]:
    return () if check is None else named_parameters(spec, check)


class _TritonKernelImplementation(KernelImplementation):
    def __init__(
        self,
        spec: KernelSpec,
        backend: Backend,
        declaration: TritonKernel,
    ) -> None:
        super().__init__(spec, backend)
        precision, names = _precision(spec)
        kernel, batched = declaration.kernel, declaration.batched
        if batched is None:
            _validate_kernel(
                kernel, spec, "single", complete=True, check_annotations=False
            )
        else:
            _validate_kernel(
                kernel, spec, "shared", complete=False, check_annotations=False
            )
            _validate_kernel(
                batched, spec, "batched", complete=True, check_annotations=False
            )
        if names:
            kinds = {**spec.compile_time, **spec.runtime_scalars}
            scalar_types = {name: kinds[name] for name in names}
            kernel = precision_variant(
                kernel,
                precision,
                scalar_types={
                    name: kind
                    for name, kind in scalar_types.items()
                    if name in kernel.arg_names
                },
            )
            if batched is not None:
                batched = precision_variant(
                    batched, precision, scalar_types=scalar_types
                )
        _validate_kernel(
            kernel,
            spec,
            "single" if batched is None else "shared",
            complete=batched is None,
        )
        if batched is not None:
            _validate_kernel(batched, spec, "batched", complete=True)
            adopt(batched)
        adopt(kernel)
        self.kernel = kernel
        self.batched = batched
        self.batch_axis = declaration.batch_axis
        self.batch_layout = declaration.batch_layout
        self.check = declaration.check
        self.check_arguments = _check_arguments(spec, declaration.check)

    def _validate(self, call: KernelCall) -> None:
        if self.check is not None:
            arguments = call.arguments
            self.check(**{name: arguments[name] for name in self.check_arguments})

    def _plan(self, arguments: Mapping[str, Any]) -> tuple[Any, Any, dict] | None:
        """Select the variant, grid and complete launch arguments of one call."""

        block = arguments["BLOCK_SIZE"]
        members = arguments[self.batch_axis] if self.batch_axis is not None else 1
        if type(members) is not int or members < 0:
            raise ValueError("Triton batch extent must be a non-negative exact int")
        if members == 0:
            return None
        batched = self.batched is not None and members > 1
        selected = self.batched if batched else self.kernel
        extent = 1
        for key in self.spec.size_keys:
            extent *= arguments[key]
        # A loop grid covers the cells once, but its member offsets
        # ``member * n + offs`` span the same flat range as a flat grid.
        flat = extent * members if batched else extent
        if batched and self.batch_layout == "flat":
            extent = flat
        if extent == 0:
            return None
        TRITON.validate_extent(self.spec.name, flat, block)
        accepted = frozenset(selected.arg_names).difference({"BLOCK_SIZE"})
        static = {
            "BLOCK_SIZE": block,
            **{name: value for name, value in arguments.items() if name in accepted},
            **launch_options(selected, physics=True),
        }
        return selected, ((extent + block - 1) // block,), static

    def _requests(self, call: KernelCall) -> tuple[CompileRequest, ...]:
        planned = self._plan(call.arguments)
        if planned is None:
            return ()
        selected, grid, static = planned
        return (warmup_request(partial(selected.warmup, grid=grid, **static)),)

    def _compile(self, call: KernelCall) -> Launch:
        planned = self._plan(call.arguments)
        if planned is None:
            return empty_launch
        selected, grid, static = planned
        launcher = selected[grid]
        device = next(
            (
                value.device
                for value in static.values()
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        if device is None:
            from triton.runtime import driver

            device = driver.active.get_active_torch_device()
        return partial(proven_triton_device(device).run, launcher, static)


class TritonKernel(KernelDeclaration):
    """One ``@triton.jit`` kernel over the spec's launch extent.

    ``batched`` runs instead when ``batch_axis`` holds more than one member;
    its ``batch_layout`` is ``"flat"`` (one thread per cell and member) or
    ``"loop"`` (one thread per cell looping over members).  ``kernel`` may then
    omit scalars used only by the batched layout.  ``check`` names canonical
    values and raises on a call the kernels cannot serve.
    """

    toolchain: ClassVar[str] = "triton"

    kernel: Any
    batched: Any = None
    batch_axis: Identifier | None = None
    batch_layout: Literal["flat", "loop"] = "flat"
    check: NamedCallable = None

    def __init__(self, kernel: Any, /, **fields: Any) -> None:
        super().__init__(kernel=kernel, **fields)

    @model_validator(mode="after")
    def _validate_batching(self) -> Self:
        if (self.batched is None) != (self.batch_axis is None):
            raise ValueError("batched and batch_axis must be declared together")
        return self

    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        if self.batch_axis is not None and (
            spec.runtime_scalars.get(self.batch_axis) != "index"
        ):
            raise ValueError(
                f"{spec.name}: batch_axis {self.batch_axis!r} must be an index "
                "runtime scalar"
            )
        return _TritonKernelImplementation(spec, backend, self)


class _TritonSequenceImplementation(KernelImplementation):
    def __init__(
        self, spec: KernelSpec, backend: Backend, declaration: TritonSequence
    ) -> None:
        super().__init__(spec, backend)
        _precision(spec)
        components = []
        for kernel, size in declaration.steps:
            native = _native_parameters(kernel, spec, "sequence")
            component = spec.project(
                omit=tuple(name for name in spec.parameters if name not in native),
                size=size,
            )
            components.append(
                _TritonKernelImplementation(component, backend, TritonKernel(kernel))
            )
        consumed = frozenset().union(
            *(component.spec.parameters for component in components)
        )
        if consumed != frozenset(spec.parameters):
            raise ValueError(
                f"{spec.name}: Triton sequence ABI mismatch: "
                f"missing={sorted(set(spec.parameters) - consumed)}, "
                f"extra={sorted(consumed - set(spec.parameters))}"
            )
        self.components = tuple(components)
        self.check = declaration.check
        self.check_arguments = _check_arguments(spec, declaration.check)

    def _validate(self, call: KernelCall) -> None:
        if self.check is not None:
            arguments = call.arguments
            self.check(**{name: arguments[name] for name in self.check_arguments})

    def _calls(self, call: KernelCall) -> tuple[KernelCall, ...]:
        return tuple(
            KernelCall(
                component,
                {
                    name: value
                    for name, value in call.arguments.items()
                    if name in component.spec.parameters or name == "BLOCK_SIZE"
                },
                {
                    name: dtype
                    for name, dtype in call.buffer_dtypes.items()
                    if name in component.spec.parameters
                },
            )
            for component in self.components
        )

    def _requests(self, call: KernelCall) -> tuple[CompileRequest, ...]:
        return tuple(
            request
            for component in self._calls(call)
            for request in component.implementation._requests(component)
        )

    def _compile(self, call: KernelCall) -> Launch:
        launches = tuple(
            component.implementation._compile(component)
            for component in self._calls(call)
        )

        def run() -> None:
            for launch in launches:
                launch()

        return run


class TritonSequence(KernelDeclaration):
    """Ordered Triton kernels, each over its own launch extent.

    Each step pairs a kernel with the size keys of its launch; a kernel takes
    an exact-name subset of the spec and the steps together take all of it.
    """

    toolchain: ClassVar[str] = "triton"

    steps: tuple[tuple[Any, Identifier | tuple[Identifier, ...]], ...]
    check: NamedCallable = None

    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        return _TritonSequenceImplementation(spec, backend, self)


class _TritonProgramImplementation(KernelImplementation):
    def __init__(
        self, spec: KernelSpec, backend: Backend, declaration: TritonProgram
    ) -> None:
        super().__init__(spec, backend)
        self.precision, self.names = _precision(spec)
        kinds = {**spec.compile_time, **spec.runtime_scalars}
        self.scalar_types = {name: kinds[name] for name in self.names}
        self.prepare = declaration.prepare

    def _compile(self, call: KernelCall) -> Launch:
        arguments = dict(call.arguments)
        if not self.names:
            return self.prepare(arguments, call.buffer_dtypes)
        precision, names = self.precision, self.names
        with triton_precision_context(precision, names, scalar_types=self.scalar_types):
            prepared = self.prepare(arguments, call.buffer_dtypes)

        def launch() -> None:
            with triton_precision_context(
                precision, names, scalar_types=self.scalar_types
            ):
                prepared()

        return launch


class TritonProgram(KernelDeclaration):
    """A host program of dependent Triton launches behind one spec.

    ``prepare(arguments, buffer_dtypes)`` runs once per call specialization
    and returns the launch; its inner kernels launch through
    :func:`launch_triton_kernel`, which applies the resolved scalar precision.
    """

    toolchain: ClassVar[str] = "triton"

    prepare: Callable[..., Callable[[], Any]]

    def __init__(self, prepare: Callable[..., Callable[[], Any]], /) -> None:
        super().__init__(prepare=prepare)

    @model_validator(mode="after")
    def _validate_prepare(self) -> Self:
        if tuple(inspect.signature(self.prepare).parameters) != (
            "arguments",
            "buffer_dtypes",
        ):
            raise ValueError(
                "Triton program prepare signature must be exactly "
                "(arguments, buffer_dtypes)"
            )
        return self

    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        return _TritonProgramImplementation(spec, backend, self)


__all__ = ["TritonKernel", "TritonProgram", "TritonSequence", "launch_triton_kernel"]

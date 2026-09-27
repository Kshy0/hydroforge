"""Declared CUDA launches derived from a KernelSpec.

A route lists :class:`CudaKernel` launches and :class:`CudaFill` operations
that run in stream order, over scratch buffers declared as
:class:`CudaWorkspace`. A kernel step names either a hand-written
``__global__`` kernel or a ``__device__`` function for which HydroForge
generates the ``__global__`` entry: the entry takes the step's arguments,
computes the item index, drops out-of-range threads and forwards the
arguments to the device function.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from string import Formatter
from typing import Any, Literal, Self

import torch
from pydantic import Field, FiniteFloat, PositiveInt, model_validator

from hydroforge.contracts.kernels import BufferDTypeABI, KernelSpec
from hydroforge.contracts.naming import Identifier
from hydroforge.contracts.validation import FrozenMapping, HydroForgeModel
from hydroforge.kernels.backends.cuda import rtc

_CONDITION = re.compile(r"(not\s+)?([A-Za-z_]\w*)")
_LANES = frozenset({1, 2, 4, 8, 16, 32})


def _check_condition(value: str | None) -> str | None:
    if value is not None and not _CONDITION.fullmatch(value):
        raise ValueError(f"step condition must be NAME or 'not NAME', got {value!r}")
    return value


class CudaWorkspace(HydroForgeModel):
    """Zero-initialized scratch buffer HydroForge allocates per specialization.

    ``shape`` multiplies canonical integer values and integers; steps refer to
    the workspace by its name like a canonical buffer.
    """

    dtype: Literal["bool", "int8", "uint8", "int32", "int64", "float32", "float64"]
    shape: tuple[Identifier | PositiveInt, ...] = Field(min_length=1)


class CudaFill(HydroForgeModel):
    """Fill a buffer or workspace with ``value`` in stream order."""

    buffer: Identifier
    value: bool | int | FiniteFloat = 0
    when: str | None = None

    @model_validator(mode="after")
    def _validate_condition(self) -> Self:
        _check_condition(self.when)
        return self


class CudaKernel(HydroForgeModel):
    """One kernel launch of a route.

    ``kernel`` names a hand-written ``__global__`` kernel; ``device`` names a
    ``__device__`` function whose ``__global__`` entry HydroForge generates.
    Both are C++ name expressions: ``{buffer}`` becomes the tensor's element
    type and ``{compile_time}`` its literal.

    ``arguments`` defaults to the canonical values in ``KernelSpec.parameters``
    order (buffers as pointers, scalars in the device type of their kind);
    an explicit tuple may name canonical values or workspaces, with ``None`` for
    a null pointer. ``pack="struct"`` passes them as one by-value struct; a
    generated entry then declares that struct with the argument names as
    fields and passes it as ``args``.

    The grid covers ``size`` (default ``size_key``) with ``lanes`` threads per
    item. Blocks have ``BLOCK_SIZE`` threads; with ``lanes > 1`` or
    ``max_block`` they are rounded up to whole warps and capped at
    ``max_block``. ``parallel_axes`` add grid.y (and grid.z) extents from
    canonical values or integers. With ``batch_axis`` above one member, the
    ``batched`` expression runs instead over grid.y = members.
    ``shared_memory`` reserves per-thread dynamic shared elements typed like
    the named buffers. ``when`` runs the step only if a compile-time bool
    (``"NAME"`` or ``"not NAME"``) holds.

    A generated entry computes ``hydroforge_index`` (the thread's item, or the
    C++ ``index`` expression over the entry's parameters), returns when it is
    negative or past the extent, and calls the device function with the
    arguments, then ``hydroforge_index`` if ``pass_index``, then the member
    (``blockIdx.y``) for the ``batched`` expression. ``guard=False`` keeps every
    thread for device functions that synchronize the block; they check the
    index themselves.
    """

    kernel: str | None = Field(default=None, min_length=1)
    device: str | None = Field(default=None, min_length=1)
    arguments: tuple[Identifier | None, ...] | None = None
    size: Identifier | tuple[Identifier, ...] | None = None
    lanes: PositiveInt = 1
    max_block: PositiveInt | None = Field(default=None, le=1024)
    parallel_axes: tuple[Identifier | PositiveInt, ...] = Field(
        default=(), max_length=2
    )
    pack: Literal["arguments", "struct"] = "arguments"
    shared_memory: FrozenMapping[Identifier, PositiveInt] = Field(default_factory=dict)
    index: str | None = Field(default=None, min_length=1)
    pass_index: bool = False
    guard: bool = True
    batched: str | None = Field(default=None, min_length=1)
    batch_axis: Identifier | None = None
    when: str | None = None

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if (self.kernel is None) == (self.device is None):
            raise ValueError(
                "CUDA kernel step requires exactly one of kernel or device"
            )
        if self.kernel is not None and (
            self.index is not None or self.pass_index or not self.guard
        ):
            raise ValueError(
                "index, pass_index and guard describe a generated device entry"
            )
        if (self.batched is None) != (self.batch_axis is None):
            raise ValueError("batched and batch_axis must be declared together")
        if self.batched is not None and self.parallel_axes:
            raise ValueError("a batched step takes grid.y from batch_axis")
        if self.lanes not in _LANES:
            raise ValueError(f"lanes must be one of {sorted(_LANES)}")
        _check_condition(self.when)
        return self

    @property
    def name(self) -> str:
        """Unqualified name of the kernel or device function."""
        expression = self.kernel or self.device
        return expression.split("<", 1)[0].rsplit("::", 1)[-1].strip()


CudaStep = CudaKernel | CudaFill


# ---------------------------------------------------------------------- #
# Construction-time compilation
# ---------------------------------------------------------------------- #


def _placeholders(expression: str | None) -> frozenset[str]:
    if expression is None:
        return frozenset()
    names = set()
    for _text, field, format_spec, conversion in Formatter().parse(expression):
        if field is None:
            continue
        if not field.isidentifier() or format_spec or conversion:
            raise ValueError(
                f"CUDA kernel expression placeholders must be bare canonical "
                f"names, got {{{field}}} in {expression!r}"
            )
        names.add(field)
    return frozenset(names)


def _condition(value: str | None) -> tuple[bool, str] | None:
    if value is None:
        return None
    negate, name = _CONDITION.fullmatch(value).groups()
    return negate is None, name


@dataclass(frozen=True, slots=True)
class _KernelStep:
    declaration: CudaKernel
    arguments: tuple[str | None, ...]
    size: tuple[str, ...]
    condition: tuple[bool, str] | None


@dataclass(frozen=True, slots=True)
class _FillStep:
    buffer: str
    value: bool | int | float
    condition: tuple[bool, str] | None


def compile_steps(
    spec: KernelSpec,
    steps: tuple[CudaStep, ...],
    workspace: Mapping[str, CudaWorkspace],
    skipped: set[str],
) -> tuple[_KernelStep | _FillStep, ...]:
    """Validate declared steps against the spec and fix their argument lists.

    ``skipped`` holds canonical values no default argument list passes:
    ignored values, projection-fixed values and grouped mask members.
    """

    canonical = set(spec.parameters)
    buffers = set(spec.buffers).union(workspace)
    counts = {
        name
        for name, kind in spec.runtime_scalars.items()
        if kind in ("index", "int32", "uint32")
    }

    def require(label: str, names: set[str], allowed: set[str]) -> None:
        unknown = names.difference(allowed)
        if unknown:
            raise ValueError(f"{spec.name}: CUDA {label}: {sorted(unknown)}")

    overlap = canonical.intersection(workspace)
    if overlap:
        raise ValueError(f"{spec.name}: workspaces shadow canonical values {overlap}")
    for workspace_name, declaration in workspace.items():
        require(
            f"workspace {workspace_name} shape names non-integer values",
            {axis for axis in declaration.shape if isinstance(axis, str)},
            counts,
        )

    def condition(step: CudaStep) -> tuple[bool, str] | None:
        parsed = _condition(step.when)
        if parsed is not None and spec.compile_time.get(parsed[1]) != "bool":
            raise ValueError(
                f"{spec.name}: CUDA step condition {parsed[1]!r} must be a "
                "compile-time bool"
            )
        return parsed

    compiled: list[_KernelStep | _FillStep] = []
    for step in steps:
        if isinstance(step, CudaFill):
            require("fill targets unknown buffers", {step.buffer}, buffers)
            compiled.append(_FillStep(step.buffer, step.value, condition(step)))
            continue
        placeholders = set()
        for expression in (step.kernel, step.device, step.batched):
            placeholders |= _placeholders(expression)
        require(
            "kernel placeholders must name buffers or compile-time values",
            placeholders,
            buffers.union(spec.compile_time),
        )
        require("shared_memory must name buffers", set(step.shared_memory), buffers)
        size = step.size if step.size is not None else spec.size_key
        size = (size,) if isinstance(size, str) else tuple(size)
        axes = {axis for axis in step.parallel_axes if isinstance(axis, str)}
        if step.batch_axis is not None:
            axes.add(step.batch_axis)
        require("grid extents must be integer values", set(size) | axes, counts)
        if step.arguments is None:
            arguments = tuple(
                name
                for name in spec.parameters
                if name not in skipped
                and not (name in spec.compile_time and name in placeholders)
            )
        else:
            arguments = step.arguments
            require(
                "arguments must name canonical values or workspaces",
                {name for name in arguments if name is not None},
                canonical.union(workspace),
            )
        compiled.append(_KernelStep(step, arguments, size, condition(step)))
    return tuple(compiled)


# ---------------------------------------------------------------------- #
# Specialization
# ---------------------------------------------------------------------- #

_SCALAR_ARGUMENTS: Mapping[str, Callable[[Any], rtc.KernelArgument]] = {
    "bool": rtc.boolean,
    "int32": rtc.int32,
    "uint32": rtc.uint32,
    "index": rtc.int64,
    "float32": rtc.float32,
    "float64": rtc.float64,
}
_SCALAR_TYPES = {
    "bool": "bool",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "index": "int64_t",
    "float32": "float",
    "float64": "double",
}


def constant_literal(kind: str, value: Any) -> str:
    """C++ literal of a compile-time value of a resolved scalar kind."""
    if kind == "bool":
        return "true" if value else "false"
    if kind == "int32":
        return str(value)
    if kind == "uint32":
        return f"{value}u"
    if kind == "float32":
        return f"{repr(value)}f"
    if kind == "float64":
        return repr(value)
    raise TypeError(f"unsupported CUDA compile-time kind {kind!r}")


def scalar_kind(spec: KernelSpec, name: str) -> str:
    kind = spec.runtime_scalars.get(name, spec.compile_time.get(name))
    if kind == "precision":
        raise TypeError(f"{spec.name}: {name!r} has an unresolved precision kind")
    return kind


def allocate_workspace(
    workspace: Mapping[str, CudaWorkspace], values: Mapping[str, Any], device: int
) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(
            math.prod(
                values[axis] if isinstance(axis, str) else axis
                for axis in declaration.shape
            ),
            dtype=getattr(torch, declaration.dtype),
            device=torch.device("cuda", device),
        )
        for name, declaration in workspace.items()
    }


@dataclass(frozen=True, slots=True)
class _Rendering:
    spec: KernelSpec
    values: Mapping[str, Any]
    buffers: frozenset[str]
    buffer_dtypes: BufferDTypeABI | None

    def enabled(self, condition: tuple[bool, str] | None) -> bool:
        return condition is None or bool(self.values[condition[1]]) == condition[0]

    def substitute(self, expression: str) -> str:
        substitutions = {}
        for name in _placeholders(expression):
            value = self.values[name]
            if name in self.buffers:
                if value is None:
                    raise ValueError(
                        f"{self.spec.name}: kernel type placeholder {name!r} is absent"
                    )
                substitutions[name] = rtc.ctype(value)
            else:
                substitutions[name] = constant_literal(
                    scalar_kind(self.spec, name), value
                )
        return expression.format_map(substitutions)

    def argument(self, name: str | None) -> rtc.KernelArgument:
        if name is None or name in self.buffers:
            return rtc.pointer(None if name is None else self.values[name])
        return _SCALAR_ARGUMENTS[scalar_kind(self.spec, name)](self.values[name])

    def declaration(self, name: str | None, position: int) -> str:
        if name is None:
            return f"decltype(nullptr) hydroforge_null_{position}"
        if name not in self.buffers:
            return f"{_SCALAR_TYPES[scalar_kind(self.spec, name)]} {name}"
        value = self.values[name]
        dtype = (
            value.dtype if value is not None else (self.buffer_dtypes or {}).get(name)
        )
        # An absent buffer without a declared dtype converts to any pointer.
        return f"{rtc.ctype(dtype)}* {name}" if dtype else f"decltype(nullptr) {name}"

    def launch(
        self, step: _KernelStep, entry: str
    ) -> tuple[rtc.CudaLaunch | None, str]:
        declared = step.declaration
        block_size = self.values["BLOCK_SIZE"]
        extent = math.prod(self.values[name] for name in step.size)
        members = self.values[declared.batch_axis] if declared.batch_axis else 1
        if members == 0:
            return None, ""
        batched = members > 1
        axes = (
            (members,)
            if batched
            else tuple(
                self.values[axis] if isinstance(axis, str) else axis
                for axis in declared.parallel_axes
            )
        )
        if extent * math.prod(axes) == 0:
            return None, ""
        if declared.lanes == 1 and declared.max_block is None:
            threads = block_size
        else:
            threads = min((block_size + 31) // 32 * 32, declared.max_block or 1024)
        grid = (rtc.blocks(extent, threads // declared.lanes), *axes)
        expression = self.substitute(
            declared.batched if batched else declared.kernel or declared.device
        )
        arguments = [self.argument(name) for name in step.arguments]
        if declared.pack == "struct":
            arguments = [rtc.struct(*arguments)]
        shared = threads * sum(
            self.values[name].element_size() * count
            for name, count in declared.shared_memory.items()
        )
        if declared.device is None:
            return (
                rtc.CudaLaunch(expression, grid, threads, tuple(arguments), shared),
                "",
            )
        source = self.entry(step, entry, expression, batched)
        return (
            rtc.CudaLaunch(
                entry, grid, threads, (*arguments, rtc.int64(extent)), shared
            ),
            source,
        )

    def entry(self, step: _KernelStep, entry: str, call: str, batched: bool) -> str:
        declared = step.declaration
        declarations = [
            self.declaration(name, position)
            for position, name in enumerate(step.arguments)
        ]
        names = [
            f"hydroforge_null_{position}" if name is None else name
            for position, name in enumerate(step.arguments)
        ]
        if declared.pack == "struct":
            fields = "".join(f"    {declaration};\n" for declaration in declarations)
            prefix = f"struct {entry}_arguments {{\n{fields}}};\n"
            parameters = [f"{entry}_arguments args"]
            forwarded = ["args"]
        else:
            prefix = ""
            parameters = declarations
            forwarded = names
        thread = "(int64_t(blockIdx.x) * blockDim.x + threadIdx.x)"
        index = declared.index or (
            thread if declared.lanes == 1 else f"{thread} / {declared.lanes}"
        )
        if declared.pass_index:
            forwarded = [*forwarded, "hydroforge_index"]
        if batched:
            forwarded = [*forwarded, "int64_t(blockIdx.y)"]
        signature = ", ".join([*parameters, "int64_t hydroforge_extent"])
        return (
            f"\n{prefix}__global__ void {entry}({signature}) {{\n"
            f"    const int64_t hydroforge_index = {index};\n"
            + (
                "    if (hydroforge_index < 0 || hydroforge_index >= hydroforge_extent) "
                "return;\n"
                if declared.guard
                else ""
            )
            + f"    {call}({', '.join(forwarded)});\n"
            "}\n"
        )


def render_steps(
    spec: KernelSpec,
    steps: tuple[_KernelStep | _FillStep, ...],
    values: Mapping[str, Any],
    workspace: Mapping[str, torch.Tensor],
    buffer_dtypes: BufferDTypeABI | None,
) -> tuple[tuple[rtc.LaunchStep, ...], str]:
    """Launches and fills of one specialization, plus generated entry source."""

    rendering = _Rendering(
        spec,
        {**values, **workspace},
        frozenset(spec.buffers).union(workspace),
        buffer_dtypes,
    )
    launches: list[rtc.LaunchStep] = []
    sources: list[str] = []
    for position, step in enumerate(steps):
        if not rendering.enabled(step.condition):
            continue
        if isinstance(step, _FillStep):
            target = rendering.values[step.buffer]
            if target is not None:
                launches.append(partial(target.fill_, step.value))
            continue
        launch, source = rendering.launch(step, f"hydroforge_entry_{position}")
        if launch is not None:
            launches.append(launch)
            sources.append(source)
    return tuple(launches), "".join(sources)


__all__ = [
    "CudaFill",
    "CudaKernel",
    "CudaStep",
    "CudaWorkspace",
    "compile_steps",
    "render_steps",
]

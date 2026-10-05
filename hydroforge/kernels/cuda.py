"""Runtime-compiled CUDA/HIP kernel declarations.

A :class:`CudaSource` is one device-only source and its compiler options; any
number of :class:`CudaKernel` declarations may share it.  A kernel declares
its launches as ``kernel=`` (one hand-written ``__global__``), ``steps=``
(:class:`CudaCall` launches and :class:`CudaFill` operations in stream order)
or ``launch=`` (a plan building its own launches).

A call step names either a hand-written ``__global__`` kernel or a
``__device__`` function for which HydroForge generates the ``__global__``
entry: the entry takes the step's arguments, computes the item index, drops
out-of-range threads and forwards the arguments to the device function.

``constants`` states how the source consumes compile-time values:

- ``"arguments"``: ``{NAME}`` placeholders in name expressions become
  literals and the remaining values are ordinary kernel arguments, so one
  compiled program serves every value;
- ``"source"``: each distinct tuple of values compiles its own program,
  prefixed with ``static constexpr`` definitions of every value and
  ``hydroforge::KernelParameters`` naming them; values are never arguments.

``masks`` packs bool compile-time values into ``#define HYDROFORGE_<MASK>``
bit masks ahead of the source; their members are never arguments.

This module's public names are the helpers of hand-written launch plans.
"""

from __future__ import annotations

import ctypes
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from string import Formatter
from typing import Annotated, Any, ClassVar, Literal, Self

import torch
from pydantic import (
    Field,
    FiniteFloat,
    PositiveInt,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.core.naming import Identifier
from hydroforge.core.validation import FrozenMapping, HydroForgeModel
from hydroforge.kernels.codegen.types import SCALARS
from hydroforge.kernels.codegen.types import scalar as native_scalar
from hydroforge.kernels.registry import (
    KernelCall,
    KernelDeclaration,
    KernelImplementation,
    Launch,
    NamedCallable,
    named_parameters,
)
from hydroforge.kernels.spec import (
    KernelSpec,
    expected_host_scalar,
    host_scalar_is_valid,
    launch_extent,
)
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain import cuda as rtc
from hydroforge.kernels.toolchain.cuda import (
    CudaLaunch,
    blocks,
    boolean,
    ctype,
    float32,
    float64,
    int32,
    int64,
    pointer,
    scalar,
    struct,
    uint32,
    uint64,
)
from hydroforge.platform.backend import Backend

_NonemptyText = Annotated[str, Field(min_length=1)]
_SourcePath = Annotated[Path, Field(strict=False)]
_CONDITION = re.compile(r"(not\s+)?([A-Za-z_]\w*)")
_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
_LANES = frozenset({1, 2, 4, 8, 16, 32})


class CudaSource(HydroForgeModel):
    """One device-only source compiled at runtime, shareable by kernels.

    The source is ``path`` (expanded with its ``prefixes`` and quoted
    includes) or literal ``text``.  A quoted include resolves against
    ``inline_includes`` by basename, then relative to the including file
    inside ``include_root``.  ``options`` use the runtime compiler's spelling
    (``--ftz=false``, ``-DNAME=value``); ``HYDROFORGE_FAST_MATH`` decides fast
    math, so sources do not set it. Expanded source and toolkit headers can
    use the persistent binary cache. Other external headers (including custom
    ``-I`` paths or macro includes) compile once per process and must remain
    unchanged in that process; use ``include_root`` or ``inline_includes`` to
    make application dependencies part of the persistent cache identity.
    """

    path: _SourcePath | None = None
    text: _NonemptyText | None = None
    options: tuple[_NonemptyText, ...] = ()
    prefixes: tuple[_SourcePath, ...] = ()
    inline_includes: tuple[_SourcePath, ...] = ()
    include_root: _SourcePath | None = None

    _source: str | None = PrivateAttr(default=None)

    @field_validator("options", "prefixes")
    @classmethod
    def _unique_entries(cls, values: tuple, info: ValidationInfo) -> tuple:
        if len(values) != len(set(values)):
            raise ValueError(f"CUDA source {info.field_name} must be unique")
        return values

    @field_validator("inline_includes")
    @classmethod
    def _unique_include_names(cls, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        counts = Counter(path.name for path in paths)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"CUDA inline include basenames must be unique: {duplicates}"
            )
        return paths

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.path is None) == (self.text is None):
            raise ValueError("CUDA source requires exactly one of path or text")
        if self.text is not None and (
            self.prefixes or self.inline_includes or self.include_root is not None
        ):
            raise ValueError("prefixes and includes expand a source path")
        return self

    def read(self) -> str:
        """The complete device source, read and expanded once."""

        if self._source is None:
            self._source = self.text if self.text is not None else self._expand()
        return self._source

    def _expand(self) -> str:
        includes = {path.name: path for path in self.inline_includes}
        expanded: dict[Path, str] = {}
        expanding: set[Path] = set()
        encountered: set[Path] = set()
        backedges = 0
        root = None if self.include_root is None else self.include_root.resolve()
        if root is not None and not root.is_dir():
            raise ValueError(f"CUDA include_root is not a directory: {root}")

        def expand(text: str, origin: Path) -> str:
            def replace(match: re.Match[str]) -> str:
                nonlocal backedges
                name = match.group(1)
                # The include spelling selects a declared file, never a new
                # filesystem path. Root-relative resolution stays below.
                path = includes.get(Path(name).name)
                if path is None and root is not None:
                    path = (origin.parent / name).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError(
                            f"CUDA include {name!r} escapes include_root {root}"
                        )
                if path is None:
                    return match.group(0)
                path = path.resolve()
                encountered.add(path)
                if path in expanding:
                    backedges += 1
                    return ""
                payload = expanded.get(path)
                if payload is None:
                    previous = backedges
                    expanding.add(path)
                    payload = expand(path.read_text(), path)
                    expanding.remove(path)
                    # A cyclic fragment depends on its active ancestors. Only
                    # context-independent payloads can be reused by another root.
                    if previous == backedges:
                        expanded[path] = payload
                guard = "HYDROFORGE_INCLUDE_" + sha256(str(path).encode()).hexdigest()
                return f"#ifndef {guard}\n#define {guard}\n{payload}\n#endif\n"

            return _INCLUDE.sub(replace, text)

        source = ""
        for path in (*self.prefixes, self.path):
            if source and not source.endswith("\n"):
                source += "\n"
            source += expand(path.read_text(), path)
        unused = sorted(
            str(path)
            for path in self.inline_includes
            if path.resolve() not in encountered
        )
        if unused:
            raise ValueError(f"CUDA inline includes are not referenced: {unused}")
        unresolved = sorted(set(_INCLUDE.findall(source)))
        if unresolved:
            raise ValueError(
                "CUDA quoted includes must be declared through "
                f"inline_includes: {unresolved}"
            )
        return source


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


class CudaCall(HydroForgeModel):
    """One kernel launch of a :class:`CudaKernel`.

    ``kernel`` names a hand-written ``__global__`` kernel; ``device`` names a
    ``__device__`` function whose ``__global__`` entry HydroForge generates.
    Both are C++ name expressions: ``{buffer}`` becomes the tensor's element
    type and ``{compile_time}`` its literal.

    ``arguments`` defaults to the canonical values in parameter order
    (buffers as pointers, scalars in the device type of their kind); an
    explicit tuple may name canonical values or workspaces, with ``None`` for
    a null pointer. ``pack="struct"`` passes them as one by-value struct; a
    generated entry then declares that struct with the argument names as
    fields and passes it as ``args``.

    The grid covers ``size`` (default the spec's size) with ``lanes`` threads
    per item. Blocks have ``BLOCK_SIZE`` threads; with ``lanes > 1`` or
    ``max_block`` they are rounded up to whole warps and capped at
    ``max_block``, itself a whole number of warps, so an item's lanes never
    straddle a block. ``parallel_axes`` add grid.y (and grid.z) extents from
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
            raise ValueError("CUDA call step requires exactly one of kernel or device")
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
        if self.max_block is not None and self.max_block % 32:
            raise ValueError(
                f"max_block must be a multiple of the 32-thread warp, got "
                f"{self.max_block}"
            )
        _check_condition(self.when)
        return self


CudaStep = CudaCall | CudaFill


# ---------------------------------------------------------------------- #
# Build-time step compilation
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


@dataclass(frozen=True, slots=True)
class _CallStep:
    declaration: CudaCall
    arguments: tuple[str | None, ...]
    size: tuple[str, ...]
    condition: tuple[bool, str] | None


@dataclass(frozen=True, slots=True)
class _FillStep:
    buffer: str
    value: bool | int | float
    condition: tuple[bool, str] | None


def _compile_steps(
    spec: KernelSpec,
    steps: tuple[CudaStep, ...],
    workspace: Mapping[str, CudaWorkspace],
    skipped: set[str],
) -> tuple[_CallStep | _FillStep, ...]:
    """Validate declared steps against the spec and fix their argument lists.

    ``skipped`` holds canonical values no default argument list passes:
    ignored, fixed, source-defined and masked values.
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
        if step.when is None:
            return None
        negate, name = _CONDITION.fullmatch(step.when).groups()
        if spec.compile_time.get(name) != "bool":
            raise ValueError(
                f"{spec.name}: CUDA step condition {name!r} must be a compile-time bool"
            )
        return negate is None, name

    compiled: list[_CallStep | _FillStep] = []
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
        size = step.size if step.size is not None else spec.size
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
        compiled.append(_CallStep(step, arguments, size, condition(step)))
    return tuple(compiled)


# ---------------------------------------------------------------------- #
# Specialization
# ---------------------------------------------------------------------- #


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


def compile_time_source(spec: KernelSpec, values: Mapping[str, Any]) -> str:
    """Define every compile-time value and ``hydroforge::KernelParameters``."""

    declarations = "\n".join(
        f"static constexpr {native_scalar(kind, 'cuda')} {name} = "
        f"{constant_literal(kind, values[name])};"
        for name, kind in spec.compile_time.items()
    )
    members = "\n".join(
        f"    static constexpr auto {name} = ::{name};" for name in spec.compile_time
    )
    return declarations + (
        f"\nnamespace hydroforge {{\nstruct KernelParameters {{\n{members}\n}};\n}}\n"
    )


def _scalar_kind(spec: KernelSpec, name: str) -> str:
    return spec.runtime_scalars.get(name) or spec.compile_time[name]


def _allocate_workspace(
    workspace: Mapping[str, CudaWorkspace],
    values: Mapping[str, Any],
    device: int,
    *,
    materialize: bool,
) -> dict[str, torch.Tensor]:
    shapes = {}
    for name, declaration in workspace.items():
        dimensions = tuple(
            values[axis] if isinstance(axis, str) else axis
            for axis in declaration.shape
        )
        if any(type(size) is not int or size < 0 for size in dimensions):
            raise ValueError(
                f"workspace {name!r} dimensions must be non-negative exact ints"
            )
        count = math.prod(dimensions)
        if count * getattr(torch, declaration.dtype).itemsize >= 2**63:
            raise OverflowError(f"workspace {name!r} byte size exceeds int64 range")
        shapes[name] = count
    return {
        name: torch.zeros(
            shapes[name],
            dtype=getattr(torch, declaration.dtype),
            device=torch.device("cuda", device)
            if materialize
            else torch.device("meta"),
        )
        for name, declaration in workspace.items()
    }


@dataclass(frozen=True, slots=True)
class _Rendering:
    spec: KernelSpec
    values: Mapping[str, Any]
    buffers: frozenset[str]
    buffer_dtypes: Mapping[str, torch.dtype | None] | None

    def enabled(self, condition: tuple[bool, str] | None) -> bool:
        return condition is None or bool(self.values[condition[1]]) == condition[0]

    def substitute(self, expression: str) -> str:
        substitutions = {}
        for name in _placeholders(expression):
            value = self.values[name]
            if name in self.buffers:
                if value is None:
                    value = (self.buffer_dtypes or {}).get(name)
                    if value is None:
                        raise ValueError(
                            f"{self.spec.name}: kernel type placeholder {name!r} is absent"
                        )
                substitutions[name] = ctype(value)
            else:
                substitutions[name] = constant_literal(
                    _scalar_kind(self.spec, name), value
                )
        return expression.format_map(substitutions)

    def argument(self, name: str | None) -> rtc.KernelArgument:
        if name is None or name in self.buffers:
            value = None if name is None else self.values[name]
            if value is not None and value.is_meta:
                return rtc.KernelArgument(ctypes.c_void_p, None, value.dtype)
            return pointer(value)
        return scalar(self.values[name], SCALARS[_scalar_kind(self.spec, name)].dtype)

    def declaration(self, name: str | None, position: int) -> str:
        if name is None:
            return f"decltype(nullptr) hydroforge_null_{position}"
        if name not in self.buffers:
            return f"{native_scalar(_scalar_kind(self.spec, name), 'cuda')} {name}"
        value = self.values[name]
        dtype = (
            value.dtype if value is not None else (self.buffer_dtypes or {}).get(name)
        )
        # An absent buffer without a declared dtype converts to any pointer.
        return f"{ctype(dtype)}* {name}" if dtype else f"decltype(nullptr) {name}"

    def launch(self, step: _CallStep, entry: str) -> tuple[CudaLaunch | None, str]:
        declared = step.declaration
        block_size = self.values["BLOCK_SIZE"]
        extent = launch_extent(self.spec.name, step.size, self.values)
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
        grid = (blocks(extent, threads // declared.lanes), *axes)
        expression = self.substitute(
            declared.batched if batched else declared.kernel or declared.device
        )
        arguments = [self.argument(name) for name in step.arguments]
        if declared.pack == "struct":
            arguments = [struct(*arguments)]
        shared = threads * sum(
            self.values[name].element_size() * count
            for name, count in declared.shared_memory.items()
        )
        if declared.device is None:
            return CudaLaunch(expression, grid, threads, tuple(arguments), shared), ""
        source = self.entry(step, entry, expression, batched)
        return (
            CudaLaunch(entry, grid, threads, (*arguments, int64(extent)), shared),
            source,
        )

    def entry(self, step: _CallStep, entry: str, call: str, batched: bool) -> str:
        declared = step.declaration
        declarations = [
            self.declaration(name, position)
            for position, name in enumerate(step.arguments)
        ]
        names = [
            f"hydroforge_null_{position}" if name is None else name
            for position, name in enumerate(step.arguments)
        ]
        reserved = {"hydroforge_extent", "hydroforge_index"}
        if len(set(names)) != len(names) or any(
            name in reserved
            or name.startswith("hydroforge_null_")
            and original is not None
            for name, original in zip(names, step.arguments, strict=True)
        ):
            raise ValueError(
                "CUDA generated entry parameter names collide or use reserved names"
            )
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


def _render_steps(
    spec: KernelSpec,
    steps: tuple[_CallStep | _FillStep, ...],
    values: Mapping[str, Any],
    workspace: Mapping[str, torch.Tensor],
    buffer_dtypes: Mapping[str, torch.dtype | None] | None,
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


def _launch_device(values: Mapping[str, Any]) -> int:
    for value in values.values():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return value.device.index
    return torch.cuda.current_device()


def _mask(members: tuple[str, ...], values: Mapping[str, Any]) -> int:
    return sum(1 << bit for bit, member in enumerate(members) if values[member])


class _CudaImplementation(KernelImplementation):
    """One validated :class:`CudaKernel` on a resolved spec."""

    def __init__(
        self, spec: KernelSpec, backend: Backend, declaration: CudaKernel
    ) -> None:
        super().__init__(spec, backend)
        canonical = set(spec.parameters)
        masks = declaration.masks
        masked: set[str] = set()
        for name, members in masks.items():
            invalid = [
                member for member in members if spec.compile_time.get(member) != "bool"
            ]
            if name in canonical or invalid or len(members) > 32:
                raise ValueError(
                    f"{spec.name}: CUDA mask {name!r} must map a non-parameter "
                    "name to at most 32 bool compile-time parameters; "
                    f"invalid={invalid}"
                )
            shared = masked.intersection(members)
            if shared:
                raise ValueError(
                    f"{spec.name}: compile-time features {sorted(shared)} belong "
                    "to more than one mask"
                )
            masked.update(members)
        ignored = set(declaration.ignore)
        if not ignored <= canonical:
            raise ValueError(
                f"{spec.name}: CUDA kernel ignores non-canonical values "
                f"{sorted(ignored - canonical)}"
            )
        fixed = declaration.fixed
        launch = declaration.launch
        plan_names = () if launch is None else named_parameters(spec, launch)
        # A source-specialized program defines its compile-time values.
        defined = (
            set(spec.compile_time).difference(plan_names)
            if declaration.constants == "source"
            else set()
        )
        steps = None
        if launch is None:
            declared = declaration.steps or (CudaCall(kernel=declaration.kernel),)
            skipped = masked | ignored | set(fixed) | defined
            steps = _compile_steps(spec, declared, declaration.workspace, skipped)
            plan_names = (
                *(name for name in spec.parameters if name not in skipped),
                "BLOCK_SIZE",
            )
        elif "BLOCK_SIZE" not in plan_names:
            raise ValueError(
                f"{spec.name}: CUDA launch plan must take compiler-owned BLOCK_SIZE"
            )
        omitted = canonical.difference(plan_names)
        unknown_fixed = set(fixed).difference(omitted)
        if unknown_fixed:
            raise ValueError(
                f"{spec.name}: CUDA kernel fixes values that are still consumed "
                f"by its launches or absent from KernelSpec: {sorted(unknown_fixed)}"
            )
        unbound = omitted.difference(fixed, masked, ignored, defined)
        if unbound:
            raise ValueError(
                f"{spec.name}: CUDA launches omit canonical inputs "
                f"{sorted(unbound)}; declare every omitted value in fixed, "
                "ignore or a mask instead of inferring semantics from its absence"
            )
        for name, value in fixed.items():
            if name in spec.buffers:
                if name not in spec.optional:
                    raise ValueError(
                        f"{spec.name}: CUDA launch plan omits required canonical "
                        f"buffer {name!r}"
                    )
                if value is not None:
                    raise ValueError(
                        f"{spec.name}: omitted optional CUDA buffer {name!r} "
                        "must be fixed to None"
                    )
            elif not host_scalar_is_valid(value, _scalar_kind(spec, name)):
                raise ValueError(
                    f"{spec.name}: fixed CUDA value {name!r} must be an "
                    f"{expected_host_scalar(_scalar_kind(spec, name))}, got "
                    f"{value!r} ({type(value).__name__})"
                )
        self.declaration = declaration
        self.steps = steps
        self.plan_names = plan_names
        self.check_names = (
            ()
            if declaration.check is None
            else named_parameters(spec, declaration.check)
        )
        self.options = rtc.program_options(declaration.source.options, physics=True)
        self._programs: dict[tuple[str, ...], rtc.RtcProgram] = {}

    def _validate(self, call: KernelCall) -> None:
        values = call.arguments
        if self.steps is not None:
            rendering = _Rendering(
                self.spec, values, frozenset(self.spec.buffers), call.buffer_dtypes
            )
            for step in self.steps:
                if not rendering.enabled(step.condition):
                    continue
                if isinstance(step, _CallStep):
                    launch_extent(self.spec.name, step.size, values)
                    axes = tuple(
                        axis
                        for axis in (
                            *step.declaration.parallel_axes,
                            step.declaration.batch_axis,
                        )
                        if isinstance(axis, str)
                    )
                    launch_extent(self.spec.name, axes, values)
                    for name in step.declaration.shared_memory:
                        if name in values and values[name] is None:
                            raise ValueError(f"shared memory buffer {name!r} is absent")
                elif (
                    step.buffer in values
                    and (target := values[step.buffer]) is not None
                ):
                    dtype = target.dtype
                    limit = (
                        torch.finfo(dtype)
                        if dtype.is_floating_point
                        else torch.iinfo(dtype)
                        if dtype != torch.bool
                        else None
                    )
                    if limit is not None and not limit.min <= step.value <= limit.max:
                        raise ValueError(f"CUDA fill value exceeds {dtype} range")
        mismatched = {
            name: (values[name], expected)
            for name, expected in self.declaration.fixed.items()
            if (
                type(values[name]) is not type(expected)
                or values[name] != expected
                or (type(expected) is float and values[name].hex() != expected.hex())
            )
        }
        if mismatched:
            detail = ", ".join(
                f"{name}={observed!r}, required={expected!r}"
                for name, (observed, expected) in sorted(mismatched.items())
            )
            raise ValueError(f"{self.spec.name}: CUDA fixed value mismatch: {detail}")
        check = self.declaration.check
        if check is not None:
            check(**{name: values[name] for name in self.check_names})

    def _program(self, values: Mapping[str, Any], entries: str) -> rtc.RtcProgram:
        """The compiled program of one call: masks, source and generated entries."""

        spec = self.spec
        declaration = self.declaration
        if declaration.constants == "source":
            key = tuple(
                constant_literal(kind, values[name])
                for name, kind in spec.compile_time.items()
            )
        else:
            key = ()
        program = self._programs.get(key)
        if program is None:
            source = declaration.source.read()
            if declaration.constants == "source":
                source = compile_time_source(spec, values) + source
            program = self._programs[key] = rtc.RtcProgram(
                source, self.options, spec.name
            )
        if not declaration.masks and not entries:
            return program
        prefix = "".join(
            f"#define HYDROFORGE_{name} {_mask(members, values)}u\n"
            for name, members in declaration.masks.items()
        )
        return rtc.RtcProgram(
            prefix + program.source + entries, program.options, program.name
        )

    def _plan(
        self, call: KernelCall, *, materialize: bool = True
    ) -> tuple[rtc.RtcRequest | None, tuple, int, dict[str, torch.Tensor]]:
        values = call.arguments
        device = _launch_device(values)
        workspace = {}
        entries = ""
        if self.steps is not None:
            workspace = _allocate_workspace(
                self.declaration.workspace, values, device, materialize=materialize
            )
            steps, entries = _render_steps(
                self.spec, self.steps, values, workspace, call.buffer_dtypes
            )
        else:
            steps = tuple(
                self.declaration.launch(
                    **{name: values[name] for name in self.plan_names}
                )
            )
        request = (
            rtc.request_for(self._program(values, entries), steps)
            if any(isinstance(step, CudaLaunch) for step in steps)
            else None
        )
        return request, steps, device, workspace

    def source(self, call: KernelCall) -> str:
        """The generated program source of one call, for audits and tests."""

        request, _steps, _device, _workspace = self._plan(call, materialize=False)
        return "" if request is None else request.program.source

    def _requests(self, call: KernelCall) -> tuple[CompileRequest, ...]:
        request, _steps, device, _workspace = self._plan(call, materialize=False)
        return () if request is None else (rtc.precompile_request(request, device),)

    def _compile(self, call: KernelCall) -> Launch:
        request, steps, device, workspace = self._plan(call, materialize=True)
        if request is None:

            def run() -> None:
                for step in steps:
                    step()

            return run
        launch = rtc.prepare(request, steps, device)
        # Prepared pointers borrow storage; the public launcher owns its inputs.
        launch.arguments = call.arguments
        launch.workspace = workspace
        return launch


class CudaKernel(KernelDeclaration):
    """One kernel of a runtime-compiled :class:`CudaSource`.

    Exactly one of ``kernel`` (shorthand for ``steps=(CudaCall(kernel=...),)``),
    ``steps`` or ``launch`` declares the launches.  A ``launch`` plan receives
    the canonical values it names (plus ``BLOCK_SIZE``) and returns
    :class:`CudaLaunch` launches and stream-ordered callables; it runs once
    per specialization.  ``workspace`` names scratch buffers of declared
    steps; ``ignore`` lists canonical values no declared launch takes;
    ``fixed`` requires the value of canonical inputs a launch plan omits;
    ``check``, named like a plan's parameters, raises on canonical inputs the
    kernels cannot serve.  ``constants`` and ``masks`` are described in the
    module documentation.
    """

    toolchain: ClassVar[str] = "rtc"

    source: CudaSource
    kernel: str | None = Field(default=None, min_length=1)
    steps: tuple[CudaStep, ...] = ()
    launch: NamedCallable = None
    constants: Literal["arguments", "source"] = "arguments"
    masks: FrozenMapping[Identifier, tuple[Identifier, ...]] = Field(
        default_factory=dict
    )
    workspace: FrozenMapping[Identifier, CudaWorkspace] = Field(default_factory=dict)
    ignore: tuple[Identifier, ...] = ()
    fixed: FrozenMapping[Identifier, bool | int | FiniteFloat | None] = Field(
        default_factory=dict
    )
    check: NamedCallable = None

    def __init__(self, source: CudaSource, /, **fields: Any) -> None:
        super().__init__(source=source, **fields)

    @model_validator(mode="after")
    def _one_entry(self) -> Self:
        if (
            sum((self.launch is not None, self.kernel is not None, bool(self.steps)))
            != 1
        ):
            raise ValueError(
                "CUDA kernel requires exactly one of kernel, steps or launch"
            )
        if self.launch is not None and (self.workspace or self.ignore):
            raise ValueError(
                "workspace and ignore describe declared launches; a launch "
                "plan builds its own"
            )
        for name, members in self.masks.items():
            if not members or len(members) != len(set(members)):
                raise ValueError(f"CUDA mask {name!r} requires unique member names")
        return self

    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        return _CudaImplementation(spec, backend, self)


__all__ = [
    "CudaLaunch",
    "blocks",
    "boolean",
    "ctype",
    "float32",
    "float64",
    "int32",
    "int64",
    "pointer",
    "scalar",
    "struct",
    "uint32",
    "uint64",
]

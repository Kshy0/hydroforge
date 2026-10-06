# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Metal kernel declarations and the typed programs they compile to.

:class:`MetalKernel` is a physics kernel written as MSL statements: its
source holds value-only helpers and one named body per kernel, and HydroForge
generates the argument buffer, function constants and entry from the spec.
A :class:`MetalProgram` is one complete MSL entry together with its typed
argument ABI; physics kernels, framework control and ATen kernels, step
fields and statistics all dispatch through it, so no ABI is parsed back out of
MSL text.  Libraries compile on first launch.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, ClassVar

import torch
from pydantic import Field

from hydroforge.core.errors import ResourceCleanupError
from hydroforge.core.naming import Identifier
from hydroforge.io.files import atomic_write_text
from hydroforge.kernels.codegen.types import SCALARS, element, scalar
from hydroforge.kernels.registry import (
    KernelCall,
    KernelDeclaration,
    KernelImplementation,
    Launch,
    empty_launch,
)
from hydroforge.kernels.spec import (
    KernelSpec,
    buffer_access_semantics,
    host_scalar_is_valid,
    launch_extent,
)
from hydroforge.kernels.toolchain import metal as toolchain
from hydroforge.platform.backend import METAL, Backend

# The native bridge sets bool, int and float function constants only.
COMPILE_SCALAR_KINDS = frozenset(("bool", "int32", "float32"))
# Pointee types a kernel may declare for each tensor dtype; some reductions
# use an atomic uint view of float32 bits.
NATIVE_BUFFER_DTYPES = MappingProxyType(
    {
        "float": frozenset({torch.float32}),
        "hf_hp": frozenset({torch.float64}),
        "atomic_float": frozenset({torch.float32}),
        "int": frozenset({torch.int32}),
        "atomic_int": frozenset({torch.int32}),
        "long": frozenset({torch.int64}),
        "uchar": frozenset({torch.bool}),
        "uint": frozenset({torch.int32}),
        "atomic_uint": frozenset({torch.float32, torch.int32}),
    }
)
# Scalar type names of the native bridge by scalar kind.
_BRIDGE_SCALARS = MappingProxyType(
    {
        kind: str(native.dtype).removeprefix("torch.")
        for kind, native in SCALARS.items()
        if native.msl is not None
    }
)


@dataclass(frozen=True, slots=True)
class MetalArgument:
    """One argument-buffer field: a buffer with its access and pointee type,
    or a scalar (``access=None``) of a scalar kind."""

    name: str
    access: str | None
    native: str
    optional: bool = False


def _physical(value):
    from hydroforge.kernels.emulated import EmulatedTensor

    return value.carrier if isinstance(value, EmulatedTensor) else value


def _constant_key(kind: str, value: Any) -> tuple[str, Any]:
    """A type- and bit-exact pipeline cache component."""

    return (kind, struct.pack("=f", value)) if kind == "float32" else (kind, value)


class MetalProgram:
    """One MSL kernel entry and its typed argument ABI.

    ``arguments`` follow the argument buffer's ``[[id(i)]]`` order and
    ``constants`` its ``[[function_constant(i)]]`` order; the product of the
    ``extent`` values is the thread count.  ``origin`` selects the math
    policy of the library (:func:`hydroforge.kernels.toolchain.metal.library`).
    """

    def __init__(
        self,
        source: str,
        kernel: str,
        arguments: tuple[MetalArgument, ...],
        *,
        extent: tuple[str, ...],
        constants: tuple[tuple[str, str], ...] = (),
        origin: toolchain.MetalOrigin = "framework",
        encoding: str = "native",
    ) -> None:
        names = [argument.name for argument in arguments]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(
                f"{kernel}: Metal argument names must be distinct and nonempty"
            )
        constant_names = [name for name, _kind in constants]
        if len(set(constant_names)) != len(constant_names):
            raise ValueError(f"{kernel}: Metal function constants must be distinct")
        if any(kind not in COMPILE_SCALAR_KINDS for _name, kind in constants):
            raise TypeError(f"{kernel}: unsupported Metal function constant kind")
        if origin not in {"physics", "aten", "framework"}:
            raise ValueError("invalid Metal library origin")
        for argument in arguments:
            if argument.native == "hf_hp" and encoding != "float32x2":
                raise TypeError("hf_hp buffer ABI requires explicit float32x2 encoding")
            if argument.access is None:
                if argument.native not in _BRIDGE_SCALARS:
                    raise TypeError(
                        f"{kernel}: Metal scalar {argument.name!r} has no native "
                        f"kind {argument.native!r}"
                    )
            elif argument.native not in NATIVE_BUFFER_DTYPES:
                raise TypeError(
                    f"{kernel}: unsupported Metal buffer pointee type "
                    f"{argument.native!r} of {argument.name!r}"
                )
            elif buffer_access_semantics(argument.access).atomic != (
                argument.native.startswith("atomic_")
            ):
                raise TypeError(
                    f"{kernel}: {argument.access} buffer {argument.name!r} "
                    f"cannot use Metal pointee type {argument.native!r}"
                )
        self.source = toolchain.prepare_source(source, origin=origin, encoding=encoding)
        self.kernel = kernel
        self.arguments = arguments
        self.extent = extent
        self.constants = constants
        self.origin = origin
        self.encoding = encoding
        self.names = tuple(argument.name for argument in arguments)
        self.native_types = tuple(
            "buffer"
            if argument.access is not None
            else _BRIDGE_SCALARS[argument.native]
            for argument in arguments
        )
        self.dependencies = MappingProxyType(
            {
                argument.name: buffer_access_semantics(argument.access).dependency
                for argument in arguments
                if argument.access is not None
            }
        )
        self._runtime = None
        self._pipelines: dict[tuple[Any, ...], int] = {}

    def _native(self):
        if self._runtime is None:
            self._runtime = toolchain.load_metal_kernel()
        return self._runtime

    def save_source(self, path: str | Path) -> Path:
        """Persist this exact effective compilation unit, including hp helpers.

        A JSON sidecar records the required non-source compile setting. Saving
        does not compile or execute the program, and changes no cache key.
        """
        path = Path(path)
        fast_math = self.origin == "physics" and METAL.math.physics_fast_math
        atomic_write_text(path, self.source)
        atomic_write_text(
            path.with_suffix(path.suffix + ".json"),
            json.dumps(
                {
                    "origin": self.origin,
                    "encoding": self.encoding,
                    "fastMathEnabled": fast_math,
                    "sha256": hashlib.sha256(self.source.encode()).hexdigest(),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        return path

    def pipeline(self, values: Mapping[str, Any]) -> tuple[Any, int]:
        """The native bridge and the pipeline of these constant values."""

        for name, kind in self.constants:
            if not host_scalar_is_valid(values[name], kind):
                raise TypeError(
                    f"{self.kernel}.{name} is not a representable exact {kind}"
                )
        key = tuple(_constant_key(kind, values[name]) for name, kind in self.constants)
        native = self._native()
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            pipeline = native.create_pipeline(
                toolchain.library(
                    self.source, origin=self.origin, encoding=self.encoding
                ),
                self.kernel,
                [
                    (index, kind, float(values[name]))
                    for index, (name, kind) in enumerate(self.constants)
                ],
                self.native_types,
                [self.dependencies.get(name, "none") for name in self.names],
            )
            self._pipelines[key] = pipeline
        return native, pipeline

    def validate(
        self,
        values: Mapping[str, Any],
        group_size: int,
        buffer_dtypes: Mapping[str, torch.dtype | None] | None = None,
    ) -> None:
        """Check buffer element types, the grid extent and the group width."""

        required = {
            *self.names,
            *self.extent,
            *(name for name, _kind in self.constants),
        }
        if missing := required.difference(values):
            raise ValueError(f"{self.kernel}: missing Metal values {sorted(missing)}")
        for name, kind in self.constants:
            if not host_scalar_is_valid(values[name], kind):
                raise TypeError(
                    f"{self.kernel}.{name} is not a representable exact {kind}"
                )
        for argument in self.arguments:
            if argument.access is None:
                if not host_scalar_is_valid(values[argument.name], argument.native):
                    raise TypeError(
                        f"{self.kernel}.{argument.name} has an invalid scalar value"
                    )
                continue
            if buffer_dtypes is None:
                value = values[argument.name]
                if value is not None and (
                    not isinstance(value, torch.Tensor)
                    or value.layout != torch.strided
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"{self.kernel}.{argument.name} must be a contiguous tensor"
                    )
                dtype = None if value is None else value.dtype
            else:
                dtype = buffer_dtypes[argument.name]
            if dtype is None and argument.optional:
                continue
            allowed = NATIVE_BUFFER_DTYPES[argument.native]
            if dtype not in allowed:
                choices = ", ".join(sorted(str(item) for item in allowed))
                raise ValueError(
                    f"{self.kernel}.{argument.name} field declares {dtype}, but "
                    f"the Metal source uses {argument.native}; expected {choices}"
                )
            if argument.native == "hf_hp" and buffer_dtypes is None:
                value = values[argument.name]
                if getattr(value, "encoding", None) != "float32x2":
                    raise TypeError(
                        f"{self.kernel}.{argument.name} requires encoded float32x2 storage, "
                        "not native float64 bytes"
                    )
        METAL.validate_extent(
            self.kernel, launch_extent(self.kernel, self.extent, values), 0
        )
        if type(group_size) is not int or not 1 <= group_size <= 1024:
            raise ValueError("Metal group size must be an exact int in [1, 1024]")

    def launch(self, values: Mapping[str, Any], group_size: int) -> Launch:
        """The launch of validated values; it owns its argument binding."""

        self.validate(values, group_size)
        threads = 1
        for name in self.extent:
            threads *= values[name]
        if threads == 0:
            return empty_launch
        return _MetalLaunch(self, dict(values), threads, group_size)

    def specialize(self, values: Mapping[str, Any], group_size: int) -> Launch:
        """The validated launch of ``values`` (:meth:`launch` validates)."""
        return self.launch(values, group_size)

    def _prepare(self, values: dict[str, Any], threads: int, group_size: int):
        native, pipeline = self.pipeline(values)
        binding = native.create_argument_binding(
            pipeline, [_physical(values[name]) for name in self.names]
        )
        return native, pipeline, binding, threads, group_size

    def _submit(self, prepared, values: dict[str, Any]) -> None:
        sequence = toolchain.recording_metal_sequence()
        native, pipeline, binding, threads, group_size = prepared
        scope = (
            "Metal command enqueue"
            if sequence is not None
            else "Metal command dispatch"
        )
        try:
            if sequence is not None:
                reads, writes = [], []
                for name, access in self.dependencies.items():
                    value = values[name]
                    if value is None:
                        continue
                    if access in {"read", "read_write"}:
                        reads.append(value)
                    if access in {"write", "read_write"}:
                        writes.append(value)
                sequence.add_prepared(
                    prepared, barrier=False, reads=tuple(reads), writes=tuple(writes)
                )
            else:
                native.dispatch(pipeline, binding, threads, group_size)
        except BaseException as primary:
            try:
                native.release_argument_binding(binding)
            except BaseException as cleanup_error:
                raise ResourceCleanupError(scope, (primary, cleanup_error)) from primary
            raise
        if sequence is None:
            native.release_argument_binding(binding)


class _MetalLaunch:
    """One specialized Metal launch owning a reusable argument binding.

    Recording hands each command sequence a fresh binding it owns. Direct
    dispatch reuses one binding, released natively after GPU completion once
    this launch is closed or collected.
    """

    __slots__ = (
        "_program",
        "_values",
        "_threads",
        "_group_size",
        "_native",
        "_pipeline",
        "_binding",
        "_release",
        "__weakref__",
    )

    def __init__(
        self,
        program: MetalProgram,
        values: dict[str, Any],
        threads: int,
        group_size: int,
    ) -> None:
        self._program = program
        self._values = values
        self._threads = threads
        self._group_size = group_size
        self._native = None
        self._pipeline = None
        self._binding = None
        self._release = None

    def __call__(self) -> None:
        program = self._program
        if toolchain.recording_metal_sequence() is not None:
            prepared = program._prepare(self._values, self._threads, self._group_size)
            program._submit(prepared, self._values)
            return
        if self._binding is None:
            native, pipeline = program.pipeline(self._values)
            binding = native.create_argument_binding(
                pipeline, [_physical(self._values[name]) for name in program.names]
            )
            self._native, self._pipeline, self._binding = native, pipeline, binding
            self._release = weakref.finalize(
                self, native.release_argument_binding, binding
            )
            self._release.atexit = False
        self._native.dispatch(
            self._pipeline, self._binding, self._threads, self._group_size
        )

    def close(self) -> None:
        release, self._release = self._release, None
        self._binding = None
        if release is not None:
            release()


class MetalCommandNode(ABC):
    """An explicitly recordable node accepted by Metal ICB construction."""

    reads: tuple[Any, ...]
    writes: tuple[Any, ...]

    @abstractmethod
    def record(self) -> None:
        """Record this node into the active Metal command sequence."""


@dataclass(frozen=True, slots=True)
class MetalCommand(MetalCommandNode):
    """One framework program launch whose dependencies come from its ABI."""

    program: MetalProgram
    arguments: dict[str, Any]
    errors: tuple[Any, ...] = ()

    def _buffers(self, reads: bool) -> tuple[Any, ...]:
        return tuple(
            self.arguments[argument.name]
            for argument in self.program.arguments
            if argument.access is not None
            and getattr(
                buffer_access_semantics(argument.access), "reads" if reads else "writes"
            )
            and self.arguments.get(argument.name) is not None
        )

    @property
    def reads(self) -> tuple[Any, ...]:
        return self._buffers(True)

    @property
    def writes(self) -> tuple[Any, ...]:
        return self._buffers(False)

    def record(self) -> None:
        self.program.specialize(self.arguments, METAL.block.fixed)()


# ---------------------------------------------------------------------- #
# Generated argument buffers
# ---------------------------------------------------------------------- #

METAL_KERNEL_BODY_MARKER = "// HYDROFORGE METAL KERNEL BODY"
_BODY_PATTERN = re.compile(
    rf"^[ \t]*{re.escape(METAL_KERNEL_BODY_MARKER)}\s*$", re.MULTILINE
)
_NAMED_BODY_PATTERN = re.compile(
    rf"^[ \t]*{re.escape(METAL_KERNEL_BODY_MARKER)}:\s*(?P<name>[A-Za-z_]\w*)\s*$",
    re.MULTILINE,
)


def _render(
    spec: KernelSpec,
    buffer_dtypes: Mapping[str, torch.dtype | None],
    *,
    helpers: str,
    body: str,
    group_size: int | None,
    extent: tuple[str, ...],
    origin: toolchain.MetalOrigin,
    emulation: str | None = None,
) -> MetalProgram:
    """Generate the argument buffer, constants and entry of one spec body."""

    if emulation is not None:
        from hydroforge.kernels.emulated import source

        helpers = "#define HF_HP_ENABLED 1\n" + source() + "\n" + helpers
    constants: list[tuple[str, str]] = []
    arguments: list[MetalArgument] = []
    constant_lines: list[str] = []
    field_lines: list[str] = []
    for name in spec.parameters:
        if name in spec.compile_time:
            kind = spec.compile_time[name]
            constant_lines.append(
                f"constant {scalar(kind, 'msl')} {name} "
                f"[[function_constant({len(constants)})]];"
            )
            constants.append((name, kind))
            continue
        if name in spec.buffers:
            access = spec.buffers[name]
            dtype = buffer_dtypes[name]
            try:
                native = (
                    "hf_hp"
                    if dtype == torch.float64 and emulation
                    else element(dtype, "msl")
                )
            except TypeError as error:
                raise TypeError(
                    f"Metal buffer {name!r} has unsupported dtype {dtype}; "
                    "there is no implicit precision conversion"
                ) from error
            if native == "hf_hp" and buffer_access_semantics(access).atomic:
                access = (
                    "read_write"  # Emulated accumulators use destination-owned writes.
                )
            if buffer_access_semantics(access).atomic:
                if native not in {"float", "int"}:
                    raise TypeError(
                        f"Metal {access} buffer {name!r} requires float32 or "
                        f"int32, got {dtype}"
                    )
                if access in {"atomic_min", "atomic_max"} and native == "float":
                    native = "atomic_uint"
                else:
                    native = f"atomic_{native}"
            qualifier = "device const" if access == "read" else "device"
            arguments.append(MetalArgument(name, access, native, name in spec.optional))
        else:
            kind = spec.runtime_scalars[name]
            native = scalar(kind, "msl")
            qualifier = "constant"
            arguments.append(MetalArgument(name, None, kind))
        field_lines.append(
            f"    {qualifier} {native}* {name} [[id({len(field_lines)})]];"
        )
    constant_source = "\n".join(constant_lines)
    field_source = "\n".join(field_lines)
    group_source = (
        ""
        if group_size is None
        else f"constant constexpr uint HF_BLOCK_SIZE = {group_size};"
    )
    source = f"""
#include <metal_stdlib>
using namespace metal;
{constant_source}
{group_source}
{helpers}
struct {spec.name}_args {{
{field_source}
}};
kernel void {spec.name}(
    constant {spec.name}_args& args [[buffer(0)]],
    uint i [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {{
{body}
}}
"""
    return MetalProgram(
        source,
        spec.name,
        tuple(arguments),
        extent=extent,
        constants=tuple(constants),
        origin=origin,
        encoding=emulation or "native",
    )


def _split_template(spec: KernelSpec, source: str) -> tuple[str, str]:
    """Select the single or named body for ``spec`` from a Metal source."""

    named = tuple(_NAMED_BODY_PATTERN.finditer(source))
    bodies = tuple(_BODY_PATTERN.finditer(source))
    markers = tuple(
        line.strip() for line in source.splitlines() if METAL_KERNEL_BODY_MARKER in line
    )
    if not named:
        if len(bodies) == 1 and len(markers) == 1:
            boundary = bodies[0]
            return source[: boundary.start()], source[boundary.end() :]
        raise ValueError(
            "Metal kernel source requires exactly one body marker or one "
            "or more named body markers"
        )
    if bodies or len(markers) != len(named):
        raise ValueError(
            "Metal kernel source may not mix unnamed and named body markers"
        )
    names = tuple(match.group("name") for match in named)
    if len(names) != len(set(names)):
        raise ValueError(f"Metal kernel source has duplicate bodies: {names}")
    try:
        selected = names.index(spec.name)
    except ValueError as error:
        raise ValueError(
            f"Metal kernel source has no body for {spec.name!r}; available={names}"
        ) from error
    start = named[selected].end()
    end = named[selected + 1].start() if selected + 1 < len(named) else len(source)
    return source[: named[0].start()], source[start:end]


def _without_comments(source: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_blocks)


class _MetalImplementation(KernelImplementation):
    def __init__(
        self, spec: KernelSpec, backend: Backend, declaration: MetalKernel
    ) -> None:
        super().__init__(spec, backend)
        if spec.uses_precision:
            raise TypeError(
                f"{spec.name}: a Metal precision-dependent KernelSpec must be "
                "resolved to float32 before it is built"
            )
        unsupported = {
            name: kind
            for name, kind in spec.compile_time.items()
            if kind not in COMPILE_SCALAR_KINDS
        }
        if unsupported:
            raise TypeError(
                f"{spec.name}: Metal has no function constant for {unsupported}; "
                f"compile-time kinds are {sorted(COMPILE_SCALAR_KINDS)}"
            )
        source = declaration.source
        text = source.read_text() if isinstance(source, Path) else source
        helpers, body = _split_template(spec, declaration.prelude + text)
        if not body.strip():
            raise ValueError(f"{spec.name}: Metal kernel body must be non-empty")
        for fragment, scope, structure in (
            (
                body,
                "body must contain physics statements only",
                r"\bstruct\s+\w+_args\b",
            ),
            (
                helpers,
                "helpers may define only backend helper functions/types",
                r"\bstruct\s+[A-Za-z_]\w*_args\b",
            ),
        ):
            forbidden = tuple(
                token
                for token in (
                    "#include",
                    "function_constant",
                    "kernel void",
                    "thread_position_in_grid",
                    "using namespace",
                    "[[",
                )
                if token in fragment
            )
            if forbidden or re.search(structure, fragment):
                raise ValueError(
                    f"{spec.name}: Metal kernel {scope}; forbidden wrapper "
                    f"syntax={forbidden}"
                )
        physics = _without_comments(f"{helpers}\n{body}")
        fields = set(re.findall(r"\bargs\.([A-Za-z_]\w*)", physics))
        runtime = set(spec.parameters).difference(spec.compile_time)
        unknown = fields.difference(runtime)
        if unknown:
            raise ValueError(
                f"{spec.name}: Metal body references fields outside KernelSpec "
                f"runtime ABI: {sorted(unknown)}"
            )
        axis = declaration.batch_axis
        extent = spec.size_keys
        if axis is not None:
            if spec.runtime_scalars.get(axis) != "index":
                raise ValueError(
                    f"{spec.name}: batch_axis {axis!r} must be an index runtime scalar"
                )
            if axis not in extent:
                extent = (*extent, axis)
        unused = runtime.difference(fields, extent)
        if unused:
            raise ValueError(
                f"{spec.name}: Metal body does not consume declared runtime ABI "
                f"fields: {sorted(unused)}"
            )
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", physics))
        unused = set(spec.compile_time).difference(identifiers)
        if unused:
            raise ValueError(
                f"{spec.name}: Metal body does not consume declared compile-time "
                f"ABI fields: {sorted(unused)}"
            )
        self.group_size = (
            spec.block_sizes.get("metal") if "HF_BLOCK_SIZE" in body else None
        )
        if "HF_BLOCK_SIZE" in body and self.group_size is None:
            raise ValueError(
                f"{spec.name}: Metal body uses HF_BLOCK_SIZE but KernelSpec does "
                "not define block_sizes['metal']"
            )
        self.helpers = helpers
        self.body = body
        self.extent = extent
        self._programs: dict[tuple[Any, ...], MetalProgram] = {}

    def program(
        self, buffer_dtypes: Mapping[str, torch.dtype | None], emulation=None
    ) -> MetalProgram:
        """The generated program of one buffer dtype signature."""

        key = (emulation, *(buffer_dtypes[name] for name in self.spec.buffers))
        program = self._programs.get(key)
        if program is None:
            program = self._programs[key] = _render(
                self.spec,
                buffer_dtypes,
                helpers=self.helpers,
                body=self.body,
                group_size=self.group_size,
                extent=self.extent,
                origin="physics",
                emulation=emulation,
            )
        return program

    @staticmethod
    def _emulation(call):
        modes = {
            getattr(value, "encoding", None)
            for value in call.arguments.values()
            if getattr(value, "encoding", None) is not None
        }
        if len(modes) > 1:
            raise TypeError(
                "one Metal kernel cannot mix encoded storage representations"
            )
        return next(iter(modes), None)

    def _validate(self, call: KernelCall) -> None:
        block = call.arguments["BLOCK_SIZE"]
        if self.group_size is not None and block != self.group_size:
            raise ValueError(
                f"{self.spec.name}: Metal reduction requires BLOCK_SIZE="
                f"{self.group_size}, got {block!r}"
            )
        self.program(call.buffer_dtypes, self._emulation(call)).validate(
            call.arguments, block, call.buffer_dtypes
        )

    def _compile(self, call: KernelCall) -> Launch:
        return self.program(call.buffer_dtypes, self._emulation(call)).launch(
            call.arguments, call.arguments["BLOCK_SIZE"]
        )


class MetalKernel(KernelDeclaration):
    """A physics kernel written as MSL statements in a named body.

    ``source`` (a path or text), after ``prelude``, holds value-only helpers
    followed by bodies marked ``// HYDROFORGE METAL KERNEL BODY: <name>``;
    the spec name selects the body.  The body reads parameters as ``args.X``
    (pointers) and compile-time values by name, with ``i`` the thread's item.
    ``batch_axis`` multiplies the launch extent by that member count.
    """

    toolchain: ClassVar[str] = "metal"

    source: Annotated[Path, Field(strict=False)] | str
    prelude: str = ""
    batch_axis: Identifier | None = None

    def __init__(self, source: Path | str, /, **fields: Any) -> None:
        super().__init__(source=source, **fields)

    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        return _MetalImplementation(spec, backend, self)


# ---------------------------------------------------------------------- #
# Framework programs
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MetalBuffer:
    name: str
    dtype: torch.dtype
    access: str


@dataclass(frozen=True, slots=True)
class MetalScalar:
    name: str
    kind: str


def online_program(
    name: str,
    *,
    buffers: tuple[MetalBuffer, ...],
    scalars: tuple[MetalScalar, ...],
    size: str,
    body: str,
    origin: toolchain.MetalOrigin = "framework",
) -> MetalProgram:
    """One framework or ATen program: the same generated ABI around ``body``."""

    spec = KernelSpec(
        name=name,
        size=size,
        parameters={
            **{buffer.name: buffer.access for buffer in buffers},
            **{field.name: field.kind for field in scalars},
        },
    )
    if len(spec.parameters) != len(buffers) + len(scalars):
        raise ValueError(f"online Metal program {name!r} has duplicate fields")
    METAL.validate_scalars(name, spec.runtime_scalars)
    helpers, body = _split_template(spec, f"{METAL_KERNEL_BODY_MARKER}: {name}\n{body}")
    return _render(
        spec,
        {buffer.name: buffer.dtype for buffer in buffers},
        helpers=helpers,
        body=body,
        group_size=None,
        extent=spec.size_keys,
        origin=origin,
    )


__all__ = ["MetalKernel"]

"""KernelSpec-first delayed Metal source generation."""

from __future__ import annotations

import re
from typing import Any, Literal

import torch
from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.kernels import (
    BackendLoweringSpec, BufferDTypeABI, KernelSpec,
    buffer_access_semantics,
)
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.backends.metal.dispatcher import MetalDispatcher
from hydroforge.kernels.backends.metal.types import (
    COMPILE_SCALAR_TYPES, RUNTIME_SCALAR_TYPES, tensor_type,
)
from hydroforge.kernels.context import active_kernel_spec


METAL_KERNEL_BODY_MARKER = "// HYDROFORGE METAL KERNEL BODY"
_BODY_PATTERN = re.compile(
    rf"^[ \t]*{re.escape(METAL_KERNEL_BODY_MARKER)}\s*$",
    re.MULTILINE,
)
_NAMED_BODY_PATTERN = re.compile(
    rf"^[ \t]*{re.escape(METAL_KERNEL_BODY_MARKER)}:\s*"
    r"(?P<name>[A-Za-z_]\w*)\s*$",
    re.MULTILINE,
)


def _split_template_source(
    spec: KernelSpec, source: str,
) -> tuple[str, str]:
    """Select the single or named body for ``spec`` from a Metal source."""

    named = tuple(_NAMED_BODY_PATTERN.finditer(source))
    bodies = tuple(_BODY_PATTERN.finditer(source))
    marker_lines = tuple(
        line.strip() for line in source.splitlines()
        if METAL_KERNEL_BODY_MARKER in line
    )
    if not named:
        if len(bodies) == 1 and len(marker_lines) == 1:
            boundary = bodies[0]
            return source[:boundary.start()], source[boundary.end():]
        raise ValueError(
            "Metal template source requires exactly one body marker or one "
            "or more named body markers"
        )
    if bodies or len(marker_lines) != len(named):
        raise ValueError(
            "Metal template source may not mix unnamed and named body markers"
        )
    names = tuple(match.group("name") for match in named)
    if len(names) != len(set(names)):
        raise ValueError(
            f"Metal template source has duplicate bodies: {names}"
        )
    try:
        selected = names.index(spec.name)
    except ValueError as error:
        raise ValueError(
            f"Metal template source has no body for {spec.name!r}; "
            f"available={names}"
        ) from error
    start = named[selected].end()
    end = (
        named[selected + 1].start()
        if selected + 1 < len(named) else len(source)
    )
    return source[:named[0].start()], source[start:end]


def _physics_source(source: str) -> str:
    """Remove comments before deriving the body-to-Spec dependency contract."""

    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_blocks)


class SpecMetalTemplateDispatcher:
    """Generate one exact Metal ABI after canonical arguments are bound.

    One source contains local value-only helpers and named operator bodies.
    Buffer/scalar declarations, function constants, argument order and access
    are emitted from KernelSpec; no Python-side source assembly is part of the
    downstream adapter.
    """

    def __init__(
        self,
        spec: KernelSpec,
        source: str,
        *,
        parallel_axes: tuple[str, ...] = (),
        variant_role: Literal["standalone", "shared", "batched"] = "standalone",
        batch_key: str | None = None,
    ) -> None:
        if not isinstance(spec, KernelSpec):
            raise TypeError("Metal template requires a KernelSpec")
        if type(source) is not str or not source.strip():
            raise ValueError("Metal template source must be a non-empty string")
        canonical_spec = spec
        if variant_role != "standalone" and batch_key is None:
            raise ValueError("Metal template variant requires batch_key")
        spec = (
            canonical_spec.project(omit=(batch_key,))
            if variant_role == "shared" else canonical_spec
        )
        helpers, body = _split_template_source(spec, source)
        if not body.strip():
            raise ValueError("Metal template kernel body must be non-empty")
        if type(parallel_axes) is not tuple:
            raise TypeError("Metal template parallel_axes must be an exact tuple")
        forbidden_body = tuple(
            token for token in (
                "#include", "function_constant", "kernel void",
                "thread_position_in_grid", "using namespace", "[[",
            )
            if token in body
        )
        if forbidden_body or re.search(r"\bstruct\s+\w+_args\b", body):
            raise ValueError(
                "Metal template body must contain physics statements only; "
                f"forbidden wrapper syntax={forbidden_body}"
            )
        forbidden_helpers = tuple(
            token for token in (
                "#include", "function_constant", "kernel void",
                "thread_position_in_grid", "using namespace", "[[",
            )
            if token in helpers
        )
        if forbidden_helpers or re.search(
            r"\bstruct\s+[A-Za-z_]\w*_args\b", helpers,
        ):
            raise ValueError(
                "Metal template helpers may define only backend helper "
                f"functions/types; forbidden wrapper syntax={forbidden_helpers}"
            )
        physics_source = _physics_source(f"{helpers}\n{body}")
        argument_fields = set(re.findall(
            r"\bargs\.([A-Za-z_]\w*)", physics_source,
        ))
        runtime_fields = set(spec.parameters).difference(spec.compile_time)
        unknown = argument_fields.difference(runtime_fields)
        if unknown:
            raise ValueError(
                f"{spec.name}: Metal body references fields outside "
                f"KernelSpec runtime ABI: {sorted(unknown)}"
            )
        if variant_role == "batched":
            base_axes = (
                (spec.size_key,) if isinstance(spec.size_key, str)
                else tuple(spec.size_key)
            )
            effective_axes = () if batch_key in base_axes else (batch_key,)
        else:
            effective_axes = parallel_axes
        execution_size_key = spec._execution_size_key(effective_axes)
        execution_axes = {
            execution_size_key
        } if isinstance(execution_size_key, str) else set(execution_size_key)
        unused_runtime = runtime_fields.difference(
            argument_fields, execution_axes,
        )
        if unused_runtime:
            raise ValueError(
                f"{spec.name}: Metal body does not consume declared runtime "
                f"ABI fields: {sorted(unused_runtime)}"
            )
        identifiers = set(re.findall(
            r"\b[A-Za-z_]\w*\b", physics_source,
        ))
        unused_constants = set(spec.compile_time).difference(identifiers)
        if unused_constants:
            raise ValueError(
                f"{spec.name}: Metal body does not consume declared "
                f"compile-time ABI fields: {sorted(unused_constants)}"
            )
        self.spec = spec
        self.canonical_spec = canonical_spec
        self.source = source
        self.body = body
        self.helpers = helpers
        self.parallel_axes = parallel_axes
        self.variant_role = variant_role
        self.batch_key = batch_key
        self.fixed_group_size = (
            spec.block_sizes.get("metal") if "HF_BLOCK_SIZE" in body else None
        )
        if "HF_BLOCK_SIZE" in body and self.fixed_group_size is None:
            raise ValueError(
                f"{spec.name}: Metal body uses HF_BLOCK_SIZE but KernelSpec "
                "does not define block_sizes['metal']"
            )
        self.kernel_name = spec.name
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="specialized",
        )
        self._dispatchers: dict[tuple[Any, ...], Any] = {}
        self._validated_signatures: dict[
            tuple[Any, ...], tuple[Any, ...]
        ] = {}

    def _signature_key(
        self, buffer_dtypes: BufferDTypeABI,
    ) -> tuple[Any, ...]:
        return tuple(buffer_dtypes[name] for name in self.spec.buffers)

    def _type_signature(
        self, buffer_dtypes: BufferDTypeABI,
    ) -> tuple[Any, ...]:
        expected_buffers = set(self.spec.buffers)
        if set(buffer_dtypes) != expected_buffers:
            raise TypeError(
                f"{self.spec.name}: Metal template buffer dtype ABI mismatch: "
                f"missing={sorted(expected_buffers - set(buffer_dtypes))}, "
                f"extra={sorted(set(buffer_dtypes) - expected_buffers)}"
            )
        result = []
        for name in self.spec.parameters:
            if name in self.spec.compile_time:
                result.append((
                    "constant", COMPILE_SCALAR_TYPES[
                        self.spec.compile_time[name]
                    ],
                ))
            elif name in self.spec.buffers:
                dtype = buffer_dtypes[name]
                try:
                    native_type = tensor_type(dtype)
                except TypeError as error:
                    raise TypeError(
                        f"Metal buffer {name!r} has unsupported dtype {dtype}; "
                        "there is no implicit precision conversion"
                    ) from error
                if (
                    buffer_access_semantics(self.spec.buffers[name]).atomic
                    and native_type not in {"float", "int"}
                ):
                    raise TypeError(
                        f"Metal {self.spec.buffers[name]} buffer {name!r} "
                        f"requires float32 or int32, got {dtype}"
                    )
                result.append(("buffer", native_type))
            else:
                result.append((
                    "scalar", RUNTIME_SCALAR_TYPES[
                        self.spec.runtime_scalars[name]
                    ],
                ))
        return tuple(result)

    def _signature(
        self, arguments: dict[str, Any], buffer_dtypes: BufferDTypeABI,
    ) -> tuple[Any, ...]:
        missing = set(self.spec.parameters).difference(arguments)
        if missing:
            raise TypeError(
                f"{self.spec.name}: missing Metal specialization values "
                f"{sorted(missing)}"
            )
        self.spec._validate_host_arguments(arguments)
        signature = self._type_signature(buffer_dtypes)
        for name, dtype in buffer_dtypes.items():
            value = arguments[name]
            if isinstance(value, torch.Tensor) and value.dtype != dtype:
                raise TypeError(
                    f"Metal buffer {name!r} has dtype {value.dtype}, "
                    f"but specialization declares {dtype}"
                )
        return signature

    def _validate_specialization_input(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> None:
        """Build and validate the concrete Metal ABI inside Pydantic."""

        self._validate_group_size(arguments)
        signature = self._signature(arguments, buffer_dtypes)
        self._validated_signatures[
            self._signature_key(buffer_dtypes)
        ] = signature
        dispatcher = self._dispatcher_for_signature(signature)
        dispatcher._validate_specialization_input(
            arguments, buffer_dtypes=buffer_dtypes,
        )

    def source_for_types(self, buffer_dtypes: BufferDTypeABI) -> str:
        """Render the ABI from declared buffer dtypes without runtime values."""

        return self._render_source(self._type_signature(buffer_dtypes))

    def source_for(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI | None = None,
    ) -> str:
        """Return deterministic generated MSL; useful for cold-path audit."""

        if buffer_dtypes is None:
            buffer_dtypes = {
                name: value.dtype
                for name, value in arguments.items()
                if name in self.spec.buffers and isinstance(value, torch.Tensor)
            }
        missing_types = set(self.spec.buffers).difference(buffer_dtypes)
        if missing_types:
            raise TypeError(
                f"{self.spec.name}: missing declared Metal buffer dtype(s): "
                f"{sorted(missing_types)}"
            )
        signature = self._signature(arguments, buffer_dtypes)
        return self._render_source(signature)

    def _render_source(self, signature: tuple[Any, ...]) -> str:
        constants = []
        fields = []
        field_index = 0
        for name, (_kind, native_type) in zip(
            self.spec.parameters, signature, strict=True,
        ):
            if name in self.spec.compile_time:
                constant_index = len(constants)
                constants.append(
                    f"constant {native_type} {name} "
                    f"[[function_constant({constant_index})]];"
                )
                continue
            if name in self.spec.buffers:
                access = self.spec.buffers[name]
                atomic = buffer_access_semantics(access).atomic
                if access in {"atomic_min", "atomic_max"} and native_type == "float":
                    dtype = "atomic_uint"
                else:
                    dtype = f"atomic_{native_type}" if atomic else native_type
                qualifier = "device const" if access == "read" else "device"
            else:
                dtype = native_type
                qualifier = "constant"
            fields.append(
                f"    {qualifier} {dtype}* {name} [[id({field_index})]];"
            )
            field_index += 1
        constant_source = "\n".join(constants)
        field_source = "\n".join(fields)
        group_source = (
            "" if self.fixed_group_size is None
            else f"constant constexpr uint HF_BLOCK_SIZE = "
                 f"{self.fixed_group_size};"
        )
        return f"""
#include <metal_stdlib>
using namespace metal;
{constant_source}
{group_source}
{self.helpers}
struct {self.spec.name}_args {{
{field_source}
}};
kernel void {self.spec.name}(
    constant {self.spec.name}_args& args [[buffer(0)]],
    uint i [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {{
{self.body}
}}
"""

    def _validate_group_size(self, arguments: dict[str, Any]) -> None:
        if self.fixed_group_size is None:
            return
        actual = arguments.get("BLOCK_SIZE")
        if actual != self.fixed_group_size:
            raise ValueError(
                f"{self.spec.name}: Metal reduction requires BLOCK_SIZE="
                f"{self.fixed_group_size}, got {actual!r}"
            )

    def _dispatcher_for_signature(self, signature: tuple[Any, ...]):
        dispatcher = self._dispatchers.get(signature)
        if dispatcher is None:
            dispatcher = MetalDispatcher(
                self._render_source(signature),
                self.spec.name,
                spec=self.canonical_spec,
                parallel_axes=self.parallel_axes,
                variant_role=self.variant_role,
                batch_key=self.batch_key,
            )
            self._dispatchers[signature] = dispatcher
        return dispatcher

    def specialize(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ):
        signature = self._validated_signatures[
            self._signature_key(buffer_dtypes)
        ]
        dispatcher = self._dispatchers[signature]
        return dispatcher.specialize(
            arguments, buffer_dtypes=buffer_dtypes,
        )


class _SpecMetalDispatcherDeclaration(HydroForgeModel):
    spec: KernelSpec | None = None
    source: str
    parallel_axes: tuple[str, ...] = ()
    variant_role: Literal["standalone", "shared", "batched"] = "standalone"
    batch_key: str | None = None

    _dispatcher: SpecMetalTemplateDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        active = active_kernel_spec()
        if active is not None:
            if self.spec is not None:
                raise ValueError(
                    "Metal template factory may not repeat active KernelSpec "
                    "metadata"
                )
            spec = active
        elif self.spec is None:
            raise ValueError(
                "make_spec_metal_dispatcher requires a KernelSpec outside a "
                "BackendRegistry factory"
            )
        else:
            spec = self.spec
        try:
            self._dispatcher = SpecMetalTemplateDispatcher(
                spec,
                self.source,
                parallel_axes=self.parallel_axes,
                variant_role=self.variant_role,
                batch_key=self.batch_key,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self

    @property
    def dispatcher(self) -> SpecMetalTemplateDispatcher:
        return self._dispatcher


def make_spec_metal_dispatcher(
    spec: KernelSpec | None = None,
    *,
    source: str,
    parallel_axes: tuple[str, ...] = (),
    variant_role: Literal["standalone", "shared", "batched"] = "standalone",
    batch_key: str | None = None,
) -> SpecMetalTemplateDispatcher:
    """Create a lazy Metal implementation from the active canonical Spec."""

    return _SpecMetalDispatcherDeclaration(
        spec=spec,
        source=source,
        parallel_axes=parallel_axes,
        variant_role=variant_role,
        batch_key=batch_key,
    ).dispatcher


__all__ = [
    "METAL_KERNEL_BODY_MARKER", "make_spec_metal_dispatcher",
]

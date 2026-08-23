"""Metal specialization, argument packing, and dispatch."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from hydroforge.contracts.kernels import (
    BackendLoweringSpec, BufferDTypeABI, KernelSpec, buffer_access_semantics,
)
from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.contracts.kernels import validate_launch_extent
from hydroforge.kernels.backends.metal.limits import (
    validate_metal_launch_extent,
)
from hydroforge.kernels.backends.metal.types import NATIVE_BUFFER_DTYPES
from hydroforge.kernels.context import active_kernel_spec

_MSL_SCALARS = {
    "bool": "bool", "int": "int32", "uint": "uint32",
    "long": "int64", "float": "float32",
}
_PACK_FORMATS = {"int": "i", "uint": "I", "long": "q", "float": "f"}
_PACK_LAYOUTS = {
    "int": (4, 4), "uint": (4, 4), "long": (8, 8), "float": (4, 4),
}
_SPEC_TO_NATIVE_SCALAR = {
    "bool": "bool", "int32": "int32", "uint32": "uint32",
    "index": "int64",
    "float32": "float32",
}


def _validated_group_size(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 1024:
        raise ValueError("Metal group size must be an exact int in [1, 1024]")
    return value


@dataclass(frozen=True, slots=True)
class _MetalArgument:
    index: int
    name: str
    kind: str
    native_type: str
    is_const: bool
    is_atomic: bool


@dataclass(frozen=True, slots=True)
class _MetalABI:
    arguments: tuple[_MetalArgument, ...]
    constants: dict[str, tuple[int, str]]


def _parse_metal_abi(source: str, kernel_name: str) -> _MetalABI:
    """Parse one argument struct exactly once into one native ABI record."""

    struct_match = re.search(
        rf"struct\s+{re.escape(kernel_name)}_args\s*\{{(.*?)\}}\s*;",
        source, re.DOTALL,
    )
    if struct_match is None:
        raise ValueError(
            f"Metal source must define struct {kernel_name}_args"
        )
    if re.search(
        rf"\bkernel\s+void\s+{re.escape(kernel_name)}\s*\(", source,
    ) is None:
        raise ValueError(f"Metal source does not define kernel {kernel_name!r}")
    body = re.sub(
        r"//.*?$|/\*.*?\*/", "", struct_match.group(1),
        flags=re.MULTILINE | re.DOTALL,
    )
    fields: list[_MetalArgument] = []
    pattern = re.compile(
        r"\b(device|constant)\s+(const\s+)?"
        r"(atomic_(?:u?int|float)|[A-Za-z_]\w*)\s*\*\s*"
        r"([A-Za-z_]\w*)\s*\[\[id\((\d+)\)\]\]\s*;"
    )
    for address_space, const, native_type, name, index in pattern.findall(body):
        is_atomic = native_type.startswith("atomic_")
        kind = (
            "buffer"
            if address_space == "device" or is_atomic
            else _MSL_SCALARS.get(native_type, "buffer")
        )
        fields.append(_MetalArgument(
            index=int(index), name=name, kind=kind, native_type=native_type,
            is_const=address_space == "constant" or bool(const),
            is_atomic=is_atomic,
        ))
    if not fields:
        raise ValueError(f"Metal argument struct {kernel_name}_args is empty")
    residual = pattern.sub("", body).strip()
    if residual:
        raise TypeError(
            f"Metal argument struct {kernel_name}_args contains a field "
            "declaration outside the supported pointer ABI: "
            f"{residual!r}"
        )
    fields.sort(key=lambda field: field.index)
    indices = [field.index for field in fields]
    if indices != list(range(len(fields))):
        raise ValueError(
            f"Metal argument ids for {kernel_name!r} must be contiguous from zero"
        )
    names = [field.name for field in fields]
    if len(names) != len(set(names)):
        raise ValueError(
            f"Metal argument names for {kernel_name!r} must be unique"
        )
    constant_fields = re.findall(
        r"constant\s+(bool|int|float)\s+([A-Za-z_]\w*)\s*"
        r"\[\[function_constant\((\d+)\)\]\]",
        source,
    )
    constant_names = [name for _kind, name, _index in constant_fields]
    constant_indices = [int(index) for _kind, _name, index in constant_fields]
    if len(constant_names) != len(set(constant_names)):
        raise ValueError("Metal function-constant names must be unique")
    if len(constant_indices) != len(set(constant_indices)):
        raise ValueError("Metal function-constant indices must be unique")
    constants = {
        name: (int(index), _MSL_SCALARS[kind])
        for kind, name, index in constant_fields
    }
    return _MetalABI(tuple(fields), constants)


def _parse_packed_struct(
    source: str,
    kernel_name: str,
    target: str,
    name: str,
    spec: KernelSpec,
) -> tuple[str, list[str]]:
    """Derive one host packing layout from its authoritative MSL struct."""
    match = re.search(
        rf"struct\s+{re.escape(name)}\s*\{{(.*?)\}}\s*;",
        source, re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Metal source does not define packed struct {name!r}")
    body = re.sub(r"//.*?$|/\*.*?\*/", "", match.group(1), flags=re.MULTILINE | re.DOTALL)
    fields = re.findall(
        r"\b(int|uint|long|float)\s+([A-Za-z_]\w*)\s*;", body,
    )
    if not fields or body.count(";") != len(fields):
        raise TypeError(
            f"Metal packed struct {name!r} must contain only int, uint, long, "
            "or float scalar fields"
        )
    native_names = {field_name for _kind, field_name in fields}
    entry = re.search(
        rf"\bkernel\s+void\s+{re.escape(kernel_name)}\s*\(", source,
    )
    if entry is None:
        raise ValueError(f"Metal source does not define kernel {kernel_name!r}")
    following = re.search(r"\bkernel\s+void\s+", source[entry.end():])
    end = len(source) if following is None else entry.end() + following.start()
    kernel_source = source[entry.start():end]
    referenced = set(re.findall(
        rf"\bargs\.{re.escape(target)}\s*->\s*([A-Za-z_]\w*)",
        kernel_source,
    ))
    unknown_references = referenced.difference(native_names)
    if unknown_references:
        raise ValueError(
            f"Metal packed pointer {target!r} reads fields absent from struct "
            f"{name!r}: {sorted(unknown_references)}"
        )
    sources = [
        _match_canonical_parameter(
            field_name, spec.parameters,
            context=f"Metal packed struct {name!r} field",
        )
        for _kind, field_name in fields
    ]
    if len(sources) != len(set(sources)):
        raise ValueError(
            f"Metal packed struct {name!r} maps multiple fields to one "
            "canonical parameter"
        )
    for (native_kind, native_name), source_name in zip(
        fields, sources, strict=True,
    ):
        expected = spec.runtime_scalars.get(source_name)
        if expected is None:
            expected = spec.compile_time.get(source_name)
        observed = _MSL_SCALARS[native_kind]
        required = None if expected is None else _SPEC_TO_NATIVE_SCALAR[expected]
        if required is None or observed != required:
            raise TypeError(
                f"{kernel_name}: packed field {name}.{native_name} is "
                f"{observed}, but KernelSpec requires {required!r}; packed "
                "configuration may not narrow or reinterpret scalars"
            )

    # Metal structs use natural member alignment. Python's '<' format is
    # deliberately alignment-free, so emit every padding byte explicitly and
    # round the final allocation to the struct's maximum alignment.
    format_parts = ["<"]
    offset = 0
    maximum_alignment = 1
    for kind, _field_name in fields:
        size, alignment = _PACK_LAYOUTS[kind]
        maximum_alignment = max(maximum_alignment, alignment)
        padding = (-offset) % alignment
        if padding:
            format_parts.append(f"{padding}x")
            offset += padding
        format_parts.append(_PACK_FORMATS[kind])
        offset += size
    trailing = (-offset) % maximum_alignment
    if trailing:
        format_parts.append(f"{trailing}x")
    return "".join(format_parts), sources


def _normalized_identifier(name: str) -> str:
    """Return the only permitted native/canonical spelling normalization."""

    return name.strip("_").casefold()


def _match_canonical_parameter(
    native_name: str,
    parameters: tuple[str, ...],
    *,
    context: str,
) -> str:
    """Resolve a native spelling only when the canonical source is unique."""

    if native_name in parameters:
        return native_name
    normalized = _normalized_identifier(native_name)
    matches = tuple(
        name for name in parameters
        if _normalized_identifier(name) == normalized
    )
    if len(matches) != 1:
        detail = "no match" if not matches else f"ambiguous matches {matches}"
        raise TypeError(
            f"{context} {native_name!r} cannot be inferred from KernelSpec: "
            f"{detail}; rename the native field/token to the canonical name"
        )
    return matches[0]


def _packed_layouts(
    source: str,
    kernel_name: str,
    parsed_args: tuple[str, ...],
    parsed_buffer_types: dict[str, str],
    parsed_qualifiers: dict[str, tuple[bool, bool]],
    spec: KernelSpec,
):
    """Infer every backend-private constant struct from the shader ABI."""

    layouts = {}
    for target in parsed_args:
        if target in spec.parameters:
            continue
        is_const, _is_atomic = parsed_qualifiers[target]
        if not is_const:
            raise TypeError(
                f"{kernel_name}: backend-private Metal field {target!r} must "
                "be a const scalar configuration struct"
            )
        struct_name = parsed_buffer_types[target]
        layouts[target] = _parse_packed_struct(
            source, kernel_name, target, struct_name, spec,
        )
    return layouts


def _template_variables(source: str, spec: KernelSpec) -> dict[str, str]:
    """Infer source-template tokens from exact compile-time Spec names."""

    tokens = tuple(dict.fromkeys(re.findall(
        r"__[A-Z][A-Z0-9_]*__", source,
    )))
    result = {
        token: _match_canonical_parameter(
            token, tuple(spec.compile_time), context="Metal template token",
        )
        for token in tokens
    }
    sources = tuple(result.values())
    if len(sources) != len(set(sources)):
        duplicates = sorted({name for name in sources if sources.count(name) > 1})
        raise TypeError(
            "Metal source contains multiple template tokens for canonical "
            f"compile-time parameter(s) {duplicates}"
        )
    return result


def _scalar_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _constant_kind(value: Any) -> str:
    value = _scalar_value(value)
    if value.__class__ is bool:
        return "bool"
    if isinstance(value, int):
        return "int32"
    if isinstance(value, float):
        return "float32"
    raise TypeError(f"unsupported Metal constant type {type(value).__name__}")


def _specialization_cache_value(value: Any) -> tuple[str, Any]:
    """Return a type- and bit-exact Metal specialization cache component."""

    value = _scalar_value(value)
    kind = _constant_kind(value)
    if kind == "float32":
        return kind, struct.pack("=f", value)
    return kind, value


class MetalDispatcher:
    """Own one Metal kernel's immutable ABI and specialization caches."""

    def __init__(
        self,
        msl_source: str | os.PathLike[str],
        kernel_name: str,
        *,
        spec: KernelSpec,
        parallel_axes: tuple[str, ...] = (),
        variant_role: Literal["standalone", "shared", "batched"] = "standalone",
        batch_key: str | None = None,
    ) -> None:
        self.source = (
            Path(msl_source).read_text()
            if isinstance(msl_source, (os.PathLike, Path)) else msl_source
        )
        self.kernel_name = kernel_name
        native_abi = _parse_metal_abi(self.source, kernel_name)
        parsed_args = tuple(field.name for field in native_abi.arguments)
        parsed_types = {
            field.name: field.kind for field in native_abi.arguments
        }
        parsed_constants = native_abi.constants
        parsed_qualifiers = {
            field.name: (field.is_const, field.is_atomic)
            for field in native_abi.arguments
        }
        parsed_buffer_types = {
            field.name: field.native_type for field in native_abi.arguments
        }
        canonical_spec = spec
        packed_layouts = _packed_layouts(
            self.source, kernel_name, parsed_args, parsed_buffer_types,
            parsed_qualifiers, canonical_spec,
        )
        template_vars = _template_variables(self.source, canonical_spec)
        for name, access in canonical_spec.buffers.items():
            if name not in parsed_args:
                continue
            try:
                is_const, is_atomic = parsed_qualifiers[name]
            except KeyError as error:
                raise TypeError(
                    f"{kernel_name}: KernelSpec buffer {name!r} is not a "
                    "device pointer in the Metal argument struct"
                ) from error
            if access == "read" and not is_const:
                raise TypeError(
                    f"{kernel_name}: read-only KernelSpec buffer {name!r} "
                    "must be const-qualified in Metal"
                )
            if access != "read" and is_const:
                raise TypeError(
                    f"{kernel_name}: writable KernelSpec buffer {name!r} "
                    "may not be const-qualified in Metal"
                )
            atomic_access = buffer_access_semantics(access).atomic
            if atomic_access != is_atomic:
                raise TypeError(
                    f"{kernel_name}: {access} KernelSpec buffer {name!r} "
                    f"requires {'an atomic' if atomic_access else 'a non-atomic'} "
                    "Metal pointer type"
                )
        # One MSL translation unit may host several logical kernels. Global
        # function constants belonging to sibling entries are irrelevant to
        # this pipeline; the canonical Spec selects this entry's exact subset.
        shader_constants = set(parsed_constants).intersection(
            canonical_spec.compile_time,
        )
        function_constants = {
            name: parsed_constants[name][0] for name in shader_constants
        }
        for name in shader_constants:
            _index, kind = parsed_constants[name]
            if canonical_spec.compile_time[name] != kind:
                raise TypeError(
                    f"{kernel_name}: function constant {name!r} is {kind}, "
                    f"KernelSpec declares {canonical_spec.compile_time[name]}"
                )
        packed_names = set(packed_layouts)
        shader_fields = set(parsed_args)
        missing_packs = packed_names.difference(shader_fields)
        if missing_packs:
            raise TypeError(
                f"{kernel_name}: packed target(s) are absent from Metal "
                f"argument struct: {sorted(missing_packs)}"
            )
        mutable_packs = sorted(
            name for name in packed_names
            if not parsed_qualifiers.get(name, (False, False))[0]
        )
        if mutable_packs:
            raise TypeError(
                f"{kernel_name}: packed Metal configuration pointers must be "
                "const-qualified: " + ", ".join(mutable_packs)
            )
        packed_sources = [
            source_name
            for _target, (_fmt, source_names) in packed_layouts.items()
            for source_name in source_names
        ]
        if len(packed_sources) != len(set(packed_sources)):
            duplicates = sorted({
                name for name in packed_sources
                if packed_sources.count(name) > 1
            })
            raise TypeError(
                f"{kernel_name}: canonical values feed multiple packed "
                f"fields: {duplicates}"
            )
        template_sources = set(template_vars.values())
        unknown_sources = (
            set(packed_sources) | template_sources
        ).difference(canonical_spec.parameters)
        if unknown_sources:
            raise TypeError(
                f"{kernel_name}: Metal lowering sources are outside "
                f"KernelSpec: {sorted(unknown_sources)}"
            )
        compile_time_paths = (
            shader_constants,
            template_sources,
            set(packed_sources).intersection(canonical_spec.compile_time),
        )
        duplicate_compile_time = sorted(
            name for name in canonical_spec.compile_time
            if sum(name in path for path in compile_time_paths) > 1
        )
        if duplicate_compile_time:
            raise TypeError(
                f"{kernel_name}: compile-time parameters have multiple Metal "
                f"lowerings: {duplicate_compile_time}"
            )
        lowered_compile_time = set().union(*compile_time_paths)
        missing_compile_time = set(canonical_spec.compile_time).difference(
            lowered_compile_time,
        )
        if missing_compile_time:
            raise TypeError(
                f"{kernel_name}: compile-time KernelSpec parameters are absent "
                f"from Metal function constants/templates/config: "
                f"{sorted(missing_compile_time)}"
            )
        consumed = (
            shader_fields.intersection(canonical_spec.parameters)
            | set(packed_sources)
            | shader_constants
            | template_sources
        )
        omitted = tuple(
            name for name in canonical_spec.parameters if name not in consumed
        )
        spec = canonical_spec.project(omit=omitted)
        observed_types = tuple(parsed_types[name] for name in parsed_args)
        for name, kind in zip(parsed_args, observed_types, strict=True):
            expects_buffer = name in spec.buffers or name in packed_names
            if not expects_buffer and kind == "buffer":
                raise TypeError(
                    f"{kernel_name}: Metal argument position for {name!r} is "
                    "a buffer, expected a scalar from KernelSpec"
                )
            if expects_buffer:
                continue
            if name in spec.runtime_scalars:
                expected_kind = {
                    "bool": "bool", "int32": "int32",
                    "uint32": "uint32", "index": "int64",
                    "float32": "float32",
                }[spec.runtime_scalars[name]]
            else:
                raise TypeError(
                    f"{kernel_name}: Metal scalar field {name!r} is not a "
                    "runtime scalar in KernelSpec"
                )
            if kind != expected_kind:
                raise TypeError(
                    f"{kernel_name}: Metal scalar {name!r} is {kind}, "
                    f"KernelSpec requires {expected_kind}"
                )
        buffer_access = {
            name: buffer_access_semantics(spec.buffers[name]).dependency
            for name in parsed_args if name in spec.buffers
        }
        buffer_access.update({name: "read" for name in packed_names})
        scalar_types = {
            name: kind
            for name, kind in zip(parsed_args, observed_types, strict=True)
            if name not in spec.buffers and name not in packed_names
        }
        if variant_role != "standalone" and batch_key is None:
            raise ValueError("Metal variant construction requires batch_key")
        expected_omitted = (
            frozenset({batch_key})
            if variant_role == "shared" else frozenset()
        )
        if frozenset(omitted) != expected_omitted:
            raise TypeError(
                f"{kernel_name}: {variant_role} Metal construction has an "
                "invalid canonical projection: "
                f"omitted={sorted(omitted)}, "
                f"expected={sorted(expected_omitted)}"
            )
        if variant_role != "standalone" and parallel_axes:
            raise ValueError(
                f"{kernel_name}: variant role and explicit parallel axes "
                "cannot be combined"
            )
        self.args = tuple(parsed_args)
        if variant_role == "batched":
            base_axes = (
                (spec.size_key,) if isinstance(spec.size_key, str)
                else tuple(spec.size_key)
            )
            axes = () if batch_key in base_axes else (batch_key,)
        else:
            axes = parallel_axes
        self.size_key = spec._execution_size_key(axes)
        self.spec = spec
        self.template_vars = MappingProxyType(dict(template_vars))
        self.pack_info = MappingProxyType(dict(packed_layouts))
        self.function_constants = MappingProxyType(dict(function_constants))
        self._runtime = None
        self._pipeline_cache: dict[tuple[Any, ...], int] = {}
        self.variant_role = variant_role
        self.batch_key = batch_key

        invalid = set(buffer_access.values()).difference(
            {"read", "write", "read_write"},
        )
        if invalid:
            raise ValueError(f"Invalid Metal buffer access modes: {sorted(invalid)}")
        self.buffer_access = MappingProxyType(dict(buffer_access))
        self.buffer_args = frozenset(buffer_access)
        packed_targets = frozenset(self.pack_info)
        buffer_native_types = {
            name: parsed_buffer_types[name]
            for name in self.buffer_args
            if name not in packed_targets
        }
        bool_buffers = sorted(
            name for name, native in buffer_native_types.items()
            if native == "bool"
        )
        if bool_buffers:
            raise TypeError(
                f"{kernel_name}: Metal tensor bool pointers must use uchar; "
                f"device bool* is forbidden for {bool_buffers}"
            )
        unsupported_buffers = {
            name: native
            for name, native in buffer_native_types.items()
            if native not in NATIVE_BUFFER_DTYPES
        }
        if unsupported_buffers:
            raise TypeError(
                f"{kernel_name}: unsupported public Metal buffer pointee "
                f"types {unsupported_buffers}"
            )
        self.buffer_native_types = MappingProxyType(buffer_native_types)
        unknown = self.buffer_args.difference(parsed_args)
        if unknown:
            raise ValueError(
                f"Metal kernel {kernel_name!r} has unknown buffer_args: {sorted(unknown)}"
            )
        scalar_kinds = dict(scalar_types)
        self.scalar_types = MappingProxyType(scalar_kinds)
        missing = set(parsed_args).difference(self.buffer_args, scalar_kinds)
        if missing:
            raise TypeError(
                f"Metal kernel {kernel_name!r} must declare scalar_types for: {sorted(missing)}"
            )
        self.native_types = tuple(
            "buffer" if name in self.buffer_args else scalar_kinds[name]
            for name in parsed_args
        )
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="specialized",
        )

    @staticmethod
    def _literal(value: Any) -> str:
        value = _scalar_value(value)
        if value.__class__ is bool:
            return "true" if value else "false"
        return str(value)

    def _native(self):
        if self._runtime is None:
            from hydroforge.kernels.backends.metal.runtime import load_metal_kernel

            self._runtime = load_metal_kernel()
        return self._runtime

    def _packed_values(self, values: dict[str, Any]) -> dict[str, Any]:
        if not self.pack_info:
            return {}
        import torch

        result = {}
        for target, (fmt, source_names) in self.pack_info.items():
            packed_values = tuple(
                _scalar_value(values[name])
                for name in source_names
            )
            result[target] = torch.frombuffer(
                bytearray(struct.pack(fmt, *packed_values)), dtype=torch.uint8,
            ).to("mps")
        return result

    def _constants(self, values: dict[str, Any]):
        resolved = []
        constants = []
        for name, index in self.function_constants.items():
            value = values[name]
            value = _scalar_value(value)
            kind = _constant_kind(value)
            resolved.append(value)
            constants.append((index, kind, float(value)))
        return resolved, constants

    def _templates(self, values: dict[str, Any]) -> list[Any]:
        result = []
        for name in self.template_vars.values():
            result.append(_scalar_value(values[name]))
        return result

    def _validate_specialization_input(
        self, values: dict[str, Any], *, buffer_dtypes: BufferDTypeABI,
    ) -> None:
        """Validate Metal-only ABI limits inside the Pydantic call request."""

        for name, native_type in self.buffer_native_types.items():
            dtype = buffer_dtypes[name]
            allowed = NATIVE_BUFFER_DTYPES.get(native_type)
            if dtype not in allowed:
                choices = ", ".join(sorted(str(item) for item in allowed))
                raise ValueError(
                    f"{self.kernel_name}.{name} field declares {dtype}, but "
                    f"the Metal source uses {native_type}; expected {choices}"
                )
        validate_metal_launch_extent(
            self.kernel_name,
            validate_launch_extent(self.kernel_name, self.size_key, values),
        )
        _validated_group_size(values["BLOCK_SIZE"])

    def _trusted_launch_geometry(
        self, values: dict[str, Any],
    ) -> tuple[int, int]:
        keys = (self.size_key,) if isinstance(self.size_key, str) else self.size_key
        threads = 1
        for name in keys:
            threads *= values[name]
        return threads, values["BLOCK_SIZE"]

    def _prepare_values(
        self,
        values: dict[str, Any],
        *,
        threads: int,
        group_size: int,
    ):
        arguments = [values[name] for name in self.args]
        constant_values, constants = self._constants(values)
        template_values = self._templates(values)
        cache_key = tuple(
            _specialization_cache_value(value)
            for value in (*template_values, *constant_values)
        )
        native = self._native()
        pipeline = self._pipeline_cache.get(cache_key)
        if pipeline is None:
            source = self.source
            for (token, _), value in zip(
                self.template_vars.items(), template_values, strict=True,
            ):
                source = source.replace(token, self._literal(value))
            pipeline = native.compile_pipeline(
                source, self.kernel_name, constants, self.native_types,
                [self.buffer_access.get(name, "none") for name in self.args],
            )
            self._pipeline_cache[cache_key] = pipeline
        binding = native.create_argument_binding(pipeline, arguments)
        return native, pipeline, binding, threads, group_size

    def _submit(self, prepared, values: dict[str, Any]) -> None:
        from hydroforge.kernels.backends.metal.runtime import recording_metal_sequence

        sequence = recording_metal_sequence()
        if sequence is not None:
            buffers = {
                name: values[name]
                for name in self.buffer_access
            }
            reads = tuple(
                value for name, value in buffers.items()
                if value is not None
                and self.buffer_access[name] in {"read", "read_write"}
            )
            writes = tuple(
                value for name, value in buffers.items()
                if value is not None
                and self.buffer_access[name] in {"write", "read_write"}
            )
            try:
                sequence.add_prepared(
                    prepared, barrier=False, reads=reads, writes=writes,
                )
            except BaseException as error:
                native, _pipeline, binding, _threads, _group_size = prepared
                try:
                    native.release_argument_binding(binding)
                except BaseException as cleanup_error:
                    combined = ResourceCleanupError(
                        "Metal command enqueue", (error, cleanup_error),
                    )
                    raise combined from error
                raise
            return
        native, pipeline, binding, threads, group_size = prepared
        try:
            native.dispatch(pipeline, binding, threads, group_size)
        finally:
            native.release_argument_binding(binding)

    def specialize(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ):
        """Own packed tensors in the model-local specialized launch."""

        values = dict(arguments)
        packed = self._packed_values(values)
        static = values | packed
        threads, group_size = self._trusted_launch_geometry(static)

        def launch() -> None:
            prepared = self._prepare_values(
                static,
                threads=threads,
                group_size=group_size,
            )
            self._submit(prepared, static)

        return launch

def make_metal_dispatcher(
    msl_source: str, kernel_name: str, *,
    spec: KernelSpec | None = None,
    parallel_axes: tuple[str, ...] = (),
) -> MetalDispatcher:
    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(
                "Metal factory may not repeat active KernelSpec metadata"
            )
        spec = active
    if spec is None:
        raise TypeError(
            f"{kernel_name}: Metal dispatch requires one canonical KernelSpec"
        )
    return MetalDispatcher(
        msl_source, kernel_name, spec=spec, parallel_axes=parallel_axes,
    )

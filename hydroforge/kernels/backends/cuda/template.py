"""KernelSpec-first CUDA launcher generation."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import PrivateAttr, model_validator

from hydroforge.contracts.kernels import (
    BackendLoweringSpec, BufferDTypeABI, KernelSpec,
)
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.backends.cuda.dispatcher import (
    CudaDispatcher, CudaNativeProjection, CudaRoute, _compile_cuda_route,
)
from hydroforge.kernels.backends.cuda.spec import cuda_declarations
from hydroforge.kernels.context import active_kernel_spec


_SCALAR_TYPES = {
    "bool": "bool",
    "int32": "int",
    "uint32": "uint32_t",
    "index": "long",
    "float32": "float",
    "float64": "double",
}
CUDA_LAUNCH_BODY_MARKER = "// HYDROFORGE CUDA LAUNCH BODY"


def _without_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def _constant_literal(kind: str, value: Any) -> str:
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


def _split_source(source: str) -> tuple[str, str]:
    count = source.count(CUDA_LAUNCH_BODY_MARKER)
    if count != 1:
        raise ValueError(
            "Spec-first CUDA source must contain exactly one "
            f"{CUDA_LAUNCH_BODY_MARKER!r} marker, got {count}"
        )
    prelude, body = source.split(CUDA_LAUNCH_BODY_MARKER, 1)
    if not prelude.strip():
        raise ValueError(
            "Spec-first CUDA source requires its device kernel before the "
            "launch-body marker"
        )
    if not body.strip():
        raise ValueError("Spec-first CUDA launch body must not be empty")
    return prelude, body


class _TemplateCudaGroup:
    """Minimal lazy module provider consumed by :class:`CudaDispatcher`."""

    def __init__(
        self, source: str, launch: str, *, cflags: tuple[str, ...],
        env_prefix: str,
    ) -> None:
        digest = hashlib.sha256(
            (source + "\0" + "\0".join(cflags)).encode(),
        ).hexdigest()[:16]
        self.name = f"hydroforge_cuda_template_{digest}"
        self.source = source
        self.launch = launch
        self.cflags = cflags
        self.env_prefix = env_prefix
        self._module = None
        self.declaration = cuda_declarations(source, (launch,))[0]

    def _load(self, extension: str):
        del extension
        if self._module is None:
            from hydroforge.kernels.backends.cuda.build import load_inline_cu_module

            self._module = load_inline_cu_module(
                self.name,
                cpp_sources=(
                    "#include <torch/extension.h>\n"
                    "#include <optional>\n"
                    f"{self.declaration}"
                ),
                cuda_sources=self.source,
                functions=(self.launch,),
                extra_cuda_cflags=self.cflags,
                env_prefix=self.env_prefix,
            )
        return self._module

    def _load_variant(
        self, extension: str, masks: tuple[tuple[str, int], ...],
    ):
        """Return the module whose masks are already rendered in source."""

        del masks
        return self._load(extension)

    def ensure_precompiled(self):
        return {"template": self._load("template")}


class SpecCudaTemplateDispatcher:
    """Generate the host CUDA launcher ABI from one canonical KernelSpec.

    One source owns one physical kernel completely: device functions and local
    inline launch helpers precede ``CUDA_LAUNCH_BODY_MARKER``; statements for
    the generated exported launcher follow it.  The parameter list, optional
    tensor forms and scalar types come only from KernelSpec.
    """

    def __init__(
        self, spec: KernelSpec, source: str, *,
        cflags: tuple[str, ...] = ("-O3", "--use_fast_math"),
        env_prefix: str = "HYDROFORGE",
    ) -> None:
        if not isinstance(spec, KernelSpec):
            raise TypeError("CUDA template requires a KernelSpec")
        if type(source) is not str or not source.strip():
            raise ValueError("CUDA template source must be a non-empty string")
        prelude, body = _split_source(source)
        if type(cflags) is not tuple or not cflags or any(
            type(flag) is not str or not flag for flag in cflags
        ):
            raise TypeError("CUDA template cflags must be a non-empty string tuple")
        if (
            type(env_prefix) is not str or not env_prefix
            or not env_prefix.isidentifier()
        ):
            raise ValueError("CUDA template env_prefix must be an identifier")
        forbidden = tuple(
            token for token in (
                "#include", "__global__", "PYBIND", "TORCH_LIBRARY",
            )
            if token in body
        )
        if forbidden or re.search(r"\bvoid\s+[A-Za-z_]\w*\s*\(", body):
            raise ValueError(
                "CUDA template body must contain exported-launch statements "
                f"only; forbidden wrapper syntax={forbidden}"
            )
        physics = _without_comments(body)
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", physics))
        unknown_ptrs = {
            name for name in identifiers
            if name.endswith("_ptr") and name not in spec.parameters
        }
        if unknown_ptrs:
            raise ValueError(
                f"{spec.name}: CUDA body references pointer fields outside "
                f"KernelSpec: {sorted(unknown_ptrs)}"
            )
        unused = set(spec.parameters).difference(identifiers)
        grouped_members = {
            member
            for mask, members in spec.compile_time_masks.items()
            if mask in identifiers
            for member in members
        }
        unused.difference_update(grouped_members)
        if unused:
            raise ValueError(
                f"{spec.name}: CUDA body does not consume declared ABI "
                f"fields: {sorted(unused)}"
            )
        self.spec = spec
        self.source = source
        self.prelude = prelude
        self.body = body
        self.cflags = cflags
        self.env_prefix = env_prefix
        self.launch = f"hf_launch_{spec.name}"
        self._dispatchers: dict[tuple[Any, ...], CudaDispatcher] = {}
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    def _parameter(self, name: str) -> str:
        if name in self.spec.buffers:
            native = (
                "std::optional<at::Tensor>"
                if name in self.spec.optional_buffers else "at::Tensor"
            )
        else:
            native = _SCALAR_TYPES[self.spec.runtime_scalars[name]]
        return f"{native} {name}"

    def _render_source(self, constants: dict[str, Any]) -> str:
        parameters = ",\n    ".join(
            self._parameter(name) for name in self.spec.parameters
            if name not in self.spec.compile_time
        )
        constant_source = "\n".join(
            f"static constexpr {_SCALAR_TYPES[kind]} {name} = "
            f"{_constant_literal(kind, constants[name])};"
            for name, kind in self.spec.compile_time.items()
        )
        mask_source = "\n".join(
            f"static constexpr uint32_t {name} = "
            f"{self.spec.compile_time_mask(name, constants)}u;"
            for name in self.spec.compile_time_masks
        )
        return f"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <optional>
{constant_source}
{mask_source}
{self.prelude}
void {self.launch}(
    {parameters},
    long BLOCK_SIZE)
{{
{self.body}
}}
"""

    def _constants(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        arguments = {} if arguments is None else arguments
        expected = set(self.spec.compile_time)
        supplied = set(arguments)
        if supplied != expected:
            raise TypeError(
                f"{self.spec.name}: CUDA source specialization requires "
                f"exact compile-time values; missing="
                f"{sorted(expected - supplied)}, extra="
                f"{sorted(supplied - expected)}"
            )
        self.spec._validate_compile_time(arguments)
        return {
            name: arguments[name] for name in self.spec.compile_time
        }

    def source_for(
        self, compile_time: dict[str, Any] | None = None,
    ) -> str:
        """Return the deterministic generated CUDA source for cold-path audit."""

        return self._render_source(self._constants(compile_time))

    def _specialization_key(
        self, arguments: dict[str, Any],
    ) -> tuple[Any, ...]:
        constants = {
            name: arguments[name] for name in self.spec.compile_time
        }
        return tuple(
            (type(constants[name]), constants[name])
            for name in self.spec.compile_time
        )

    def _dispatcher_for(self, arguments: dict[str, Any]) -> CudaDispatcher:
        constants = {
            name: arguments[name] for name in self.spec.compile_time
        }
        key = self._specialization_key(arguments)
        dispatcher = self._dispatchers.get(key)
        if dispatcher is None:
            source = self._render_source(constants)
            group = _TemplateCudaGroup(
                source, self.launch,
                cflags=self.cflags, env_prefix=self.env_prefix,
            )
            route = _compile_cuda_route(
                CudaRoute(
                    extension="template",
                    launch=self.launch,
                    spec=self.spec,
                    projection=CudaNativeProjection(fixed=constants),
                ),
                source,
            )
            dispatcher = CudaDispatcher(
                group, route, spec=self.spec,
            )
            self._dispatchers[key] = dispatcher
        return dispatcher

    def _validate_specialization_input(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ) -> None:
        """Build and validate the concrete source ABI inside Pydantic."""

        dispatcher = self._dispatcher_for(arguments)
        dispatcher._validate_specialization_input(
            arguments, buffer_dtypes=buffer_dtypes,
        )

    def specialize(
        self, arguments: dict[str, Any], *,
        buffer_dtypes: BufferDTypeABI,
    ):
        dispatcher = self._dispatchers[self._specialization_key(arguments)]
        return dispatcher.specialize(
            arguments, buffer_dtypes=buffer_dtypes,
        )


class _SpecCudaDispatcherDeclaration(HydroForgeModel):
    spec: KernelSpec | None = None
    source: str
    cflags: tuple[str, ...] = ("-O3", "--use_fast_math")
    env_prefix: str = "HYDROFORGE"

    _dispatcher: SpecCudaTemplateDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        active = active_kernel_spec()
        if active is not None:
            if self.spec is not None:
                raise ValueError(
                    "CUDA template factory may not repeat active KernelSpec "
                    "metadata"
                )
            spec = active
        elif self.spec is None:
            raise ValueError(
                "make_spec_cuda_dispatcher requires a KernelSpec outside a "
                "BackendRegistry factory"
            )
        else:
            spec = self.spec
        try:
            self._dispatcher = SpecCudaTemplateDispatcher(
                spec,
                self.source,
                cflags=self.cflags,
                env_prefix=self.env_prefix,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self

    @property
    def dispatcher(self) -> SpecCudaTemplateDispatcher:
        return self._dispatcher


def make_spec_cuda_dispatcher(
    spec: KernelSpec | None = None, *, source: str,
    cflags: tuple[str, ...] = ("-O3", "--use_fast_math"),
    env_prefix: str = "HYDROFORGE",
) -> SpecCudaTemplateDispatcher:
    """Create a lazy CUDA implementation from the active canonical Spec."""

    return _SpecCudaDispatcherDeclaration(
        spec=spec,
        source=source,
        cflags=cflags,
        env_prefix=env_prefix,
    ).dispatcher


__all__ = [
    "CUDA_LAUNCH_BODY_MARKER", "make_spec_cuda_dispatcher",
]

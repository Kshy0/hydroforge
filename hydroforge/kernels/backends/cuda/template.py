"""KernelSpec-first CUDA launcher generation."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from hydroforge.contracts import BackendLoweringSpec, BufferDTypeABI, KernelSpec
from hydroforge.kernels.backends.cuda.dispatcher import (
    CudaDispatcher, CudaNativeProjection,
)
from hydroforge.kernels.backends.cuda.spec import (
    cuda_declarations, cuda_function_signature,
)
from hydroforge.kernels.context import active_kernel_spec


_SCALAR_TYPES = {
    "bool": "bool",
    "int32": "int",
    "index": "long",
    "float32": "float",
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
    if kind == "float32":
        return f"{repr(value)}f"
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

    def load(self, extension: str):
        if extension != "template":
            raise KeyError(f"unknown generated CUDA extension {extension!r}")
        if self._module is None:
            from hydroforge.kernels.backends.cuda.build import load_inline_cu_module

            declaration = cuda_declarations(self.source, (self.launch,))[0]
            self._module = load_inline_cu_module(
                self.name,
                cpp_sources=(
                    "#include <torch/extension.h>\n"
                    "#include <optional>\n"
                    f"{declaration}"
                ),
                cuda_sources=self.source,
                functions=(self.launch,),
                extra_cuda_cflags=self.cflags,
                env_prefix=self.env_prefix,
            )
        return self._module

    def ensure_precompiled(self):
        return {"template": self.load("template")}


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
        self.__hydroforge_kernel__ = spec.metadata
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
        return f"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <optional>
{constant_source}
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
        self.spec.validate_compile_time(arguments)
        return {
            name: arguments[name] for name in self.spec.compile_time
        }

    def source_for(
        self, compile_time: dict[str, Any] | None = None,
    ) -> str:
        """Return the deterministic generated CUDA source for cold-path audit."""

        return self._render_source(self._constants(compile_time))

    def _dispatcher_for(self, arguments: dict[str, Any]) -> CudaDispatcher:
        constants = self._constants({
            name: arguments[name] for name in self.spec.compile_time
        })
        key = tuple(
            (type(constants[name]), constants[name])
            for name in self.spec.compile_time
        )
        dispatcher = self._dispatchers.get(key)
        if dispatcher is None:
            source = self._render_source(constants)
            group = _TemplateCudaGroup(
                source, self.launch,
                cflags=self.cflags, env_prefix=self.env_prefix,
            )
            dispatcher = CudaDispatcher(
                group, "template", self.launch, spec=self.spec,
                native_signature=cuda_function_signature(source, self.launch),
                projection=CudaNativeProjection(fixed=constants),
            )
            self._dispatchers[key] = dispatcher
        return dispatcher

    def specialize(
        self, arguments: dict[str, Any], dynamic: frozenset[str], *,
        buffer_dtypes: BufferDTypeABI,
    ):
        return self._dispatcher_for(arguments).specialize(
            arguments, dynamic, buffer_dtypes=buffer_dtypes,
        )

    def __call__(self, **arguments):
        return self._dispatcher_for(arguments)(**arguments)


def make_spec_cuda_dispatcher(
    spec: KernelSpec | None = None, *, source: str,
    cflags: tuple[str, ...] = ("-O3", "--use_fast_math"),
    env_prefix: str = "HYDROFORGE",
) -> SpecCudaTemplateDispatcher:
    """Create a lazy CUDA implementation from the active canonical Spec."""

    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(
                "CUDA template factory may not repeat active KernelSpec metadata"
            )
        spec = active
    elif spec is None:
        raise TypeError(
            "make_spec_cuda_dispatcher requires a KernelSpec outside a "
            "BackendRegistry factory"
        )
    return SpecCudaTemplateDispatcher(
        spec, source, cflags=cflags, env_prefix=env_prefix,
    )


__all__ = [
    "CUDA_LAUNCH_BODY_MARKER", "SpecCudaTemplateDispatcher",
    "make_spec_cuda_dispatcher",
]

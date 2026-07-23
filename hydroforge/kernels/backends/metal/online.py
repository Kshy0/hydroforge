"""Single-source emitter for small online-compiled Metal operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hydroforge.contracts import KernelSpec, buffer_access_semantics
from hydroforge.kernels.backends.metal.dispatcher import make_metal_dispatcher
from hydroforge.kernels.backends.metal.protocol import MetalCommandNode
from hydroforge.kernels.backends.metal.types import (
    TENSOR_TYPES,
)
from hydroforge.kernels.backends.metal.template import (
    METAL_KERNEL_BODY_MARKER, make_spec_metal_dispatcher,
)

@dataclass(frozen=True, slots=True)
class MetalCommand(MetalCommandNode):
    """One command whose dependencies come only from its dispatcher Spec."""

    dispatcher: Any
    arguments: dict[str, Any]
    errors: tuple[Any, ...] = ()

    @property
    def reads(self) -> tuple[Any, ...]:
        metadata = self.dispatcher.__hydroforge_kernel__
        return tuple(
            self.arguments[name]
            for name, access in metadata.buffers.items()
            if buffer_access_semantics(access).reads
            and self.arguments.get(name) is not None
        )

    @property
    def writes(self) -> tuple[Any, ...]:
        metadata = self.dispatcher.__hydroforge_kernel__
        return tuple(
            self.arguments[name]
            for name, access in metadata.buffers.items()
            if buffer_access_semantics(access).writes
            and self.arguments.get(name) is not None
        )

    def record(self) -> None:
        # Online framework kernels are not bound through a model KernelBinder,
        # but the Metal dispatcher still requires an explicit threadgroup size.
        self.dispatcher(BLOCK_SIZE=256, **self.arguments)


@dataclass(frozen=True, slots=True)
class MetalBuffer:
    name: str
    dtype: torch.dtype
    access: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("Metal buffer name must be a Python identifier")
        if self.dtype not in TENSOR_TYPES:
            raise ValueError(f"unsupported online Metal dtype {self.dtype!r}")
        semantics = buffer_access_semantics(self.access)
        if semantics.atomic and self.dtype not in {torch.float32, torch.int32}:
            raise ValueError(
                f"Metal buffer access {self.access!r} requires float32 or "
                f"int32 storage, got {self.dtype}"
            )

@dataclass(frozen=True, slots=True)
class MetalScalar:
    name: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in {"bool", "int32", "index", "float32"}:
            raise ValueError(f"invalid Metal scalar kind {self.kind!r}")


def make_online_metal_dispatcher(
    name: str,
    *,
    buffers: tuple[MetalBuffer, ...],
    scalars: tuple[MetalScalar, ...],
    size_key: str,
    body: str,
):
    """Emit one argument ABI into both MSL and its canonical KernelSpec."""

    fields = (*buffers, *scalars)
    names = tuple(field.name for field in fields)
    if len(names) != len(set(names)):
        raise ValueError(f"online Metal operator {name!r} has duplicate fields")
    if size_key not in {scalar.name for scalar in scalars}:
        raise ValueError(
            f"online Metal operator {name!r} size_key must be a scalar field"
        )
    spec = KernelSpec(
        name=name,
        parameters=names,
        size_key=size_key,
        buffers={field.name: field.access for field in buffers},
        runtime_scalars={
            field.name: field.kind
            for field in scalars
        },
    )
    template = make_spec_metal_dispatcher(
        spec=spec, source=f"{METAL_KERNEL_BODY_MARKER}\n{body}",
    )
    source = template.source_for_types({
        field.name: field.dtype
        for field in buffers
    })
    return make_metal_dispatcher(source, name, spec=spec)

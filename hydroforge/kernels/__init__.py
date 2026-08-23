"""Stable physical-kernel authoring API."""

from hydroforge.kernels.registry import (
    BackendRegistry,
    make_spec_cuda_dispatcher,
    make_spec_metal_dispatcher,
    make_torch_dispatcher,
    make_triton_dispatcher,
    make_triton_program_dispatcher,
    make_triton_sequence_dispatcher,
    registry_factory,
    resolve_model_backend,
)

__all__ = [
    "BackendRegistry",
    "make_spec_cuda_dispatcher",
    "make_spec_metal_dispatcher",
    "make_torch_dispatcher",
    "make_triton_dispatcher",
    "make_triton_program_dispatcher",
    "make_triton_sequence_dispatcher",
    "registry_factory",
    "resolve_model_backend",
]

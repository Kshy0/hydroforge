"""Native Metal backend."""

from hydroforge.kernels.backends.metal.dispatcher import make_metal_dispatcher
from hydroforge.kernels.backends.metal.template import (
    METAL_KERNEL_BODY_MARKER, SpecMetalTemplateDispatcher,
    make_spec_metal_dispatcher,
)

__all__ = [
    "METAL_KERNEL_BODY_MARKER", "SpecMetalTemplateDispatcher",
    "make_metal_dispatcher", "make_spec_metal_dispatcher",
]

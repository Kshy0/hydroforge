"""Public declarative model API."""

from hydroforge.execution.step import managed_step
from hydroforge.execution.substeps import SubstepFrame
from hydroforge.contracts.events import (
    ConsoleEventSink,
    EventSink,
    ModelEvent,
    NullEventSink,
)
from hydroforge.data.input import InputProxy
from hydroforge.model.model import AbstractModel
from hydroforge.contracts.kernel_field import KernelField, kernel_field
from hydroforge.model.module import (
    AbstractModule,
    ModuleReference,
    TensorField,
    computed_tensor_field,
    module_ref,
    optional_module_ref,
)

__all__ = [
    "AbstractModel",
    "AbstractModule",
    "ConsoleEventSink",
    "EventSink",
    "InputProxy",
    "KernelField",
    "ModelEvent",
    "ModuleReference",
    "NullEventSink",
    "TensorField",
    "SubstepFrame",
    "computed_tensor_field",
    "kernel_field",
    "managed_step",
    "module_ref",
    "optional_module_ref",
]

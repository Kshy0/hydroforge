"""Public declarative physical-model API."""

from hydroforge.contracts.events import NullEventSink
from hydroforge.contracts.kernel_field import kernel_field
from hydroforge.model.model import AbstractModel
from hydroforge.model.module import (
    AbstractModule,
    CoordinateField,
    ReferenceField,
    ReferenceIndexField,
    SelectionField,
    TensorField,
    computed_tensor_field,
    module_ref,
    optional_module_ref,
)
from hydroforge.model.structure import (
    StructuralUpdateContext,
    StructuralUpdateResult,
)
from hydroforge.model.tensors import copy_tensor_inputs

__all__ = [
    "AbstractModel",
    "AbstractModule",
    "CoordinateField",
    "NullEventSink",
    "ReferenceField",
    "ReferenceIndexField",
    "SelectionField",
    "StructuralUpdateContext",
    "StructuralUpdateResult",
    "TensorField",
    "computed_tensor_field",
    "copy_tensor_inputs",
    "kernel_field",
    "module_ref",
    "optional_module_ref",
]

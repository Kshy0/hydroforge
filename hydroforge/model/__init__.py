"""Public declarative physical-model API."""

from hydroforge.contracts.events import NullEventSink
from hydroforge.model.model import AbstractModel
from hydroforge.contracts.kernel_field import kernel_field
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

__all__ = [
    "AbstractModel",
    "AbstractModule",
    "CoordinateField",
    "NullEventSink",
    "ReferenceField",
    "ReferenceIndexField",
    "SelectionField",
    "TensorField",
    "computed_tensor_field",
    "kernel_field",
    "module_ref",
    "optional_module_ref",
]

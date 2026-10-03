"""Public declarative physical-model API."""

from hydroforge.declare.kernel_field import kernel_field
from hydroforge.declare.model import OutputConfig
from hydroforge.declare.module import (
    AbstractModule,
    CoordinateField,
    ReferenceField,
    ReferenceIndexField,
    SelectionField,
    TensorField,
    computed_tensor_field,
    module_ref,
    module_schema,
    optional_module_ref,
)
from hydroforge.declare.spec import FieldSpec
from hydroforge.declare.tensors import copy_tensor_inputs
from hydroforge.model.model import AbstractModel

__all__ = [
    "AbstractModel",
    "AbstractModule",
    "CoordinateField",
    "FieldSpec",
    "OutputConfig",
    "ReferenceField",
    "ReferenceIndexField",
    "SelectionField",
    "TensorField",
    "computed_tensor_field",
    "copy_tensor_inputs",
    "kernel_field",
    "module_ref",
    "module_schema",
    "optional_module_ref",
]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Public declarative physical-model API."""

from hydroforge.contracts.conditions import always, module, never, opt
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
    "always",
    "computed_tensor_field",
    "copy_tensor_inputs",
    "kernel_field",
    "module",
    "module_ref",
    "module_schema",
    "never",
    "opt",
    "optional_module_ref",
]

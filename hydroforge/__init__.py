# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
hydroforge: Generic framework for GPU-accelerated hydrological modelling.

Subpackages
-----------
contracts   Immutable field, input, and kernel contracts.
model       Declarative AbstractModule/AbstractModel API.
compiler    Cold-path model specialization and immutable plans.
data        Datasets, distributed loading, and spatial mappings.
serialization Shared file-format contracts and atomic serialization primitives.
output      Checkpoint, NetCDF, and multi-rank output facilities.
statistics  Statistics IR, emitters, and runtime.
kernels     Kernel registration and Torch/Triton/CUDA/Metal backends.
execution   Compiled step orchestration, input staging, and backend capture.
"""

from hydroforge.data.distributed import (
    find_indices_in,
    find_indices_in_torch,
    get_world_size,
    is_rank_zero,
    setup_distributed,
    torch_to_numpy_dtype,
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
from hydroforge.contracts.fields import (
    ModuleFieldSchema,
    ModuleSchema,
    parse_module_schema,
)
from hydroforge.contracts.temporal import (
    CalendarWindow,
    DatasetTemporalContract,
    EveryStep,
    ExplicitWindow,
    ExplicitWindows,
    SimulationSchedule,
    SimulationStep,
    SpinupSchedule,
    StatisticsPlan,
)

__all__ = [
    "AbstractModel",
    "AbstractModule",
    "InputProxy",
    "KernelField",
    "ModuleFieldSchema",
    "ModuleSchema",
    "ModuleReference",
    "TensorField",
    "CalendarWindow",
    "DatasetTemporalContract",
    "EveryStep",
    "ExplicitWindow",
    "ExplicitWindows",
    "SimulationSchedule",
    "SimulationStep",
    "SpinupSchedule",
    "StatisticsPlan",
    "computed_tensor_field",
    "module_ref",
    "optional_module_ref",
    "kernel_field",
    "find_indices_in",
    "find_indices_in_torch",
    "get_world_size",
    "is_rank_zero",
    "parse_module_schema",
    "setup_distributed",
    "torch_to_numpy_dtype",
]

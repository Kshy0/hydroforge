"""Public driver-side input and distributed data utilities."""

from hydroforge.data.distributed import (
    DistributedContext,
    ProcessTopology,
    binread,
    find_indices_in,
    find_indices_in_torch,
    is_rank_zero,
    read_map,
    setup_distributed,
)
from hydroforge.data.input import InputProxy
from hydroforge.data.aggregation import (
    aggregate_field_to_nc,
    build_cama_mapping,
    build_point_mapping,
)
from hydroforge.data.netcdf import (
    monthly_time_to_key,
    single_file_key,
    yearly_time_to_key,
)
from hydroforge.serialization.netcdf import (
    BOOL_LOGICAL_DTYPE,
    LOGICAL_DTYPE_ATTR,
    atomic_netcdf_dataset,
)

__all__ = [
    "InputProxy",
    "BOOL_LOGICAL_DTYPE",
    "DistributedContext",
    "LOGICAL_DTYPE_ATTR",
    "ProcessTopology",
    "aggregate_field_to_nc",
    "atomic_netcdf_dataset",
    "binread",
    "build_cama_mapping",
    "build_point_mapping",
    "find_indices_in",
    "find_indices_in_torch",
    "is_rank_zero",
    "monthly_time_to_key",
    "read_map",
    "setup_distributed",
    "single_file_key",
    "yearly_time_to_key",
]

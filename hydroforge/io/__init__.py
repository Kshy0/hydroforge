"""File formats and atomic publication: NetCDF, binary maps, and rank output.

Importing this package neither probes HDF5 filter plugins nor loads the rank
output reader: plugins are configured before HydroForge first opens a NetCDF
file, and :class:`MultiRankStatsReader` is imported on first access.
"""

from typing import Any

from hydroforge.io.binary import binread, read_map
from hydroforge.io.construction_input import write_construction_input
from hydroforge.io.netcdf.encoding import BOOL_LOGICAL_DTYPE, LOGICAL_DTYPE_ATTR
from hydroforge.io.netcdf.write import atomic_netcdf_dataset

__all__ = [
    "BOOL_LOGICAL_DTYPE",
    "LOGICAL_DTYPE_ATTR",
    "MultiRankStatsReader",
    "atomic_netcdf_dataset",
    "binread",
    "read_map",
    "write_construction_input",
]


def __getattr__(name: str) -> Any:
    if name == "MultiRankStatsReader":
        from hydroforge.io.rank_output.reader import MultiRankStatsReader

        return MultiRankStatsReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

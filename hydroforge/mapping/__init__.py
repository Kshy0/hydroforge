"""Sparse spatial mapping utilities for regular-grid forcing.

The core object is a CSR :class:`MappingTable` with rows as target supports and
columns as flattened source grid cells.  It is independent of any particular
model: catchments, glacier cells and regular cells are all just target
supports.  Two overlap engines feed it: analytic separable area overlap between
regular grids, and area-weighted high-resolution pixel aggregation.  Offline
builders produce CaMa catchment and point mappings and aggregate fields.
"""

from hydroforge.mapping.aggregation import (
    aggregate_field_to_nc,
    build_cama_mapping,
    build_point_mapping,
)
from hydroforge.mapping.build import build_regular_grid_mapping
from hydroforge.mapping.grid import RegularGrid
from hydroforge.mapping.table import MappingTable
from hydroforge.mapping.target import TargetSupport

__all__ = [
    "MappingTable",
    "RegularGrid",
    "TargetSupport",
    "aggregate_field_to_nc",
    "build_cama_mapping",
    "build_point_mapping",
    "build_regular_grid_mapping",
]

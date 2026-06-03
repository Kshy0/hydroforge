"""Sparse spatial mapping utilities for regular-grid forcing.

The core object is a CSR :class:`MappingTable` with rows as target supports and
columns as flattened source grid cells.  It is independent of any particular
model: catchments, glacier cells and regular cells are all just target
supports.  Two overlap engines feed it: analytic separable area overlap between
regular grids, and area-weighted high-resolution pixel aggregation.
"""
from hydroforge.io.spatial_mapping.build import (
    MappingMethod,
    Normalization,
    build_hires_aggregate_mapping,
    build_regular_grid_mapping,
)
from hydroforge.io.spatial_mapping.cama import (
    read_cama_catchments,
    read_cama_hires_pixels,
)
from hydroforge.io.spatial_mapping.grid import RegularGrid
from hydroforge.io.spatial_mapping.table import MappingTable
from hydroforge.io.spatial_mapping.target import TargetSupport

__all__ = [
    "MappingMethod",
    "MappingTable",
    "Normalization",
    "RegularGrid",
    "TargetSupport",
    "build_hires_aggregate_mapping",
    "build_regular_grid_mapping",
    "read_cama_catchments",
    "read_cama_hires_pixels",
]

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from hydroforge.data.datasets.base import DatasetExpression, ForcingDataset
from hydroforge.data.datasets.composite import MultiVariableDataset, open_multivariable
from hydroforge.data.datasets.daily_bin import DailyBinDataset
from hydroforge.data.datasets.era5_land import ERA5LandAccumDataset
from hydroforge.data.datasets.export import (
    export_catchment_data,
    export_climatology,
    export_quantiles,
    generate_mapping_table,
)
from hydroforge.data.datasets.exported import ExportedDataset
from hydroforge.data.datasets.keys import (
    daily_time_to_key,
    monthly_time_to_key,
    single_file_key,
    yearly_time_to_key,
)
from hydroforge.data.datasets.netcdf import NetCDFDataset
from hydroforge.data.datasets.plan import SourceChunk
from hydroforge.data.datasets.space import GridSpace, PointSpace

__all__ = [
    "DailyBinDataset",
    "DatasetExpression",
    "ERA5LandAccumDataset",
    "ExportedDataset",
    "ForcingDataset",
    "GridSpace",
    "MultiVariableDataset",
    "NetCDFDataset",
    "PointSpace",
    "SourceChunk",
    "daily_time_to_key",
    "export_catchment_data",
    "export_climatology",
    "export_quantiles",
    "generate_mapping_table",
    "monthly_time_to_key",
    "open_multivariable",
    "single_file_key",
    "yearly_time_to_key",
]

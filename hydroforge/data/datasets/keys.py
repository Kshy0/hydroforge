"""File-key functions that map one timestamp to its source file."""

from datetime import datetime

import cftime
from pydantic import validate_call

from hydroforge.core.validation import HydroForgeModel

_KEY = validate_call(config=HydroForgeModel.model_config)


@_KEY
def single_file_key(dt: datetime | cftime.datetime) -> str:
    """Constant key for single-file mode."""
    del dt
    return ""


@_KEY
def daily_time_to_key(dt: datetime | cftime.datetime) -> str:
    """Default time-to-file key: one file per day (YYYYMMDD)."""
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"


@_KEY
def yearly_time_to_key(dt: datetime | cftime.datetime) -> str:
    """Default time-to-file key: one file per year."""
    return f"{dt.year}"


@_KEY
def monthly_time_to_key(dt: datetime | cftime.datetime) -> str:
    """Default time-to-file key: one file per month (YYYY_MM)."""
    return f"{dt.year:04d}_{dt.month:02d}"

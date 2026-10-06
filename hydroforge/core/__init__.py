# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Dependency-free primitives shared by every HydroForge layer."""

from hydroforge.core.arrays import find_indices_in, find_indices_in_torch
from hydroforge.core.errors import ResourceCleanupError, cleanup_on_exit
from hydroforge.core.events import ConsoleEventSink, EventSink, NullEventSink
from hydroforge.core.time import timedelta_quotient
from hydroforge.core.units import check_units, normalize_units, units_equal
from hydroforge.core.validation import HydroForgeModel

__all__ = [
    "ConsoleEventSink",
    "EventSink",
    "HydroForgeModel",
    "NullEventSink",
    "ResourceCleanupError",
    "check_units",
    "cleanup_on_exit",
    "find_indices_in",
    "find_indices_in_torch",
    "normalize_units",
    "timedelta_quotient",
    "units_equal",
]

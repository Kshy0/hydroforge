"""Dependency-free primitives shared by every HydroForge layer."""

from hydroforge.core.arrays import find_indices_in, find_indices_in_torch
from hydroforge.core.errors import ResourceCleanupError, cleanup_on_exit
from hydroforge.core.events import ConsoleEventSink, EventSink, NullEventSink
from hydroforge.core.time import timedelta_quotient
from hydroforge.core.validation import HydroForgeModel

__all__ = [
    "ConsoleEventSink",
    "EventSink",
    "HydroForgeModel",
    "NullEventSink",
    "ResourceCleanupError",
    "cleanup_on_exit",
    "find_indices_in",
    "find_indices_in_torch",
    "timedelta_quotient",
]

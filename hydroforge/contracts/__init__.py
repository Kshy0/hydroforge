"""Evidence-backed public declarations for physical-model authoring."""

from hydroforge.contracts.fields import (
    FieldDemandPlan,
    ModuleFieldSchema,
    parse_module_schema,
)
from hydroforge.contracts.kernels import (
    KernelSpec,
    module_enabled,
    module_flag,
    output_requested,
)
from hydroforge.contracts.parameters import ParameterChange
from hydroforge.contracts.runtime import (
    BackendRequirement,
    ModuleRequirement,
)
from hydroforge.contracts.temporal import (
    CalendarWindow,
    EveryStep,
    ExplicitWindow,
    ExplicitWindows,
    SimulationSchedule,
    SpinupSchedule,
    StatisticsPlan,
    timedelta_quotient,
)
from hydroforge.contracts.validation import HydroForgeModel

__all__ = [
    "BackendRequirement",
    "CalendarWindow",
    "FieldDemandPlan",
    "EveryStep",
    "ExplicitWindow",
    "ExplicitWindows",
    "HydroForgeModel",
    "KernelSpec",
    "ModuleFieldSchema",
    "ModuleRequirement",
    "ParameterChange",
    "SimulationSchedule",
    "SpinupSchedule",
    "StatisticsPlan",
    "module_enabled",
    "module_flag",
    "output_requested",
    "parse_module_schema",
    "timedelta_quotient",
]

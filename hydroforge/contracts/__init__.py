"""Evidence-backed public declarations for physical-model authoring."""

from hydroforge.contracts.fields import FieldDemandPlan
from hydroforge.contracts.options import (
    ForcingOptionField,
    OptionField,
    OptionsConfig,
    build_options_group,
    validate_option_groups,
)
from hydroforge.contracts.parameters import ParameterChange
from hydroforge.contracts.runtime import (
    BackendRequirement,
    ModuleRequirement,
)
from hydroforge.contracts.schedule import SimulationSchedule, SpinupSchedule
from hydroforge.contracts.step_fields import StepField, StepTime, step_field
from hydroforge.contracts.windows import (
    CalendarWindow,
    EveryStep,
    ExplicitWindow,
    ExplicitWindows,
    StatisticsPlan,
)

__all__ = [
    "BackendRequirement",
    "CalendarWindow",
    "FieldDemandPlan",
    "ForcingOptionField",
    "EveryStep",
    "ExplicitWindow",
    "ExplicitWindows",
    "ModuleRequirement",
    "ParameterChange",
    "OptionField",
    "OptionsConfig",
    "SimulationSchedule",
    "SpinupSchedule",
    "StatisticsPlan",
    "StepField",
    "StepTime",
    "step_field",
    "build_options_group",
    "validate_option_groups",
]

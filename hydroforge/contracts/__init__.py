"""Evidence-backed public declarations for physical-model authoring."""

from hydroforge.contracts.fields import (
    FieldDemandPlan,
    ModuleFieldSchema,
    parse_module_schema,
)
from hydroforge.contracts.kernels import (
    KernelSpec,
    config_value,
    literal_value,
    module_enabled,
    module_flag,
    option_code,
    output_requested,
)
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
from hydroforge.contracts.step_fields import StepField, StepTime, step_field
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
    "ForcingOptionField",
    "EveryStep",
    "ExplicitWindow",
    "ExplicitWindows",
    "HydroForgeModel",
    "KernelSpec",
    "ModuleFieldSchema",
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
    "config_value",
    "literal_value",
    "build_options_group",
    "module_enabled",
    "module_flag",
    "output_requested",
    "parse_module_schema",
    "option_code",
    "timedelta_quotient",
    "validate_option_groups",
]

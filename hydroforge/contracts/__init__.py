# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Evidence-backed public declarations for physical-model authoring."""

from hydroforge.contracts.conditions import (
    Condition,
    ConditionContext,
    OptionRef,
    always,
    as_condition,
    module,
    never,
    opt,
)
from hydroforge.contracts.fields import FieldDemandPlan
from hydroforge.contracts.options import (
    ForcingOptionField,
    HydroForgeOptionWarning,
    OptionField,
    OptionsConfig,
    build_options_group,
    option,
    validate_option_groups,
)
from hydroforge.contracts.parameters import ParameterChange
from hydroforge.contracts.runtime import (
    BackendRequirement,
    ModuleRequirement,
)
from hydroforge.contracts.schedule import SimulationSchedule, SpinupSchedule
from hydroforge.contracts.slab import SlabAxis
from hydroforge.contracts.step_fields import (
    SUBSTEP_TICK_SECONDS,
    StepField,
    StepTime,
    SubstepField,
    step_field,
    substep_field,
)
from hydroforge.contracts.views import (
    GridAxis,
    Indices,
    Out,
    OutputView,
    Range,
)
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
    "Condition",
    "ConditionContext",
    "EveryStep",
    "ExplicitWindow",
    "ExplicitWindows",
    "FieldDemandPlan",
    "ForcingOptionField",
    "GridAxis",
    "SlabAxis",
    "HydroForgeOptionWarning",
    "Indices",
    "ModuleRequirement",
    "OptionField",
    "OptionRef",
    "OptionsConfig",
    "Out",
    "OutputView",
    "ParameterChange",
    "Range",
    "SimulationSchedule",
    "SpinupSchedule",
    "StatisticsPlan",
    "SUBSTEP_TICK_SECONDS",
    "StepField",
    "StepTime",
    "SubstepField",
    "always",
    "as_condition",
    "build_options_group",
    "module",
    "never",
    "opt",
    "option",
    "step_field",
    "substep_field",
    "validate_option_groups",
]

"""Pure compilation of one model declaration into its frozen plan."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from hydroforge.compiler.fields import FieldPlan, plan_fields
from hydroforge.compiler.outputs import OutputPlan, plan_output
from hydroforge.compiler.parameters import ParameterTarget, plan_parameters
from hydroforge.compiler.selection import Selection, select
from hydroforge.declare.spec import ModelSpec, StepFieldPlan

if TYPE_CHECKING:
    from hydroforge.declare.model import ModelDeclaration


@dataclass(frozen=True, slots=True)
class ModelPlan(Selection):
    """Every compile-time decision of one model declaration.

    Compilation reads the declaration and the process environment (backend,
    device capabilities, process topology) but never the input data, never
    performs file I/O and never allocates model tensors.
    """

    fields: FieldPlan
    output: OutputPlan
    step_fields: StepFieldPlan
    parameters: tuple[ParameterTarget, ...]


def compile_model(spec: ModelSpec, declaration: ModelDeclaration) -> ModelPlan:
    """Compile ``declaration`` in stages: selection, fields, output, parameters."""

    selection = select(spec, declaration)
    field_plan = plan_fields(spec, selection, declaration)
    return ModelPlan(
        **{item.name: getattr(selection, item.name) for item in fields(Selection)},
        fields=field_plan,
        output=plan_output(selection, field_plan, declaration),
        step_fields=spec.step_fields,
        parameters=plan_parameters(
            selection, field_plan, declaration.parameter_changes
        ),
    )

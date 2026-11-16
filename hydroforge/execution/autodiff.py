# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Reverse-mode derivatives of a model's managed steps.

A model whose kernels have a hand-written adjoint declares it once
(``StepAdjoint`` and ``adjoint_step`` come from :mod:`hydroforge.execution`)::

    class Model(AbstractModel):
        step_adjoint_fields: ClassVar[StepAdjoint] = StepAdjoint(
            inputs={"base.storage": "base.grad_storage_start",
                    "base.runoff": "base.grad_runoff"},
            outputs={"base.storage": "base.grad_storage"},
            tape=("base.jacobian",),
        )

        @adjoint_step
        def step_adjoint(self) -> None:
            solve_adjoint()          # registered kernels, bound like a step

``inputs`` maps each differentiable input field (``init_state``, ``state``,
``forcing`` or ``param``) to the field the adjoint writes its adjoint into; ``outputs`` maps
each result field to the field the adjoint reads the result's adjoint from;
``tape`` lists the fields a step leaves for its own adjoint (for instance a
converged Jacobian, or one snapshot per sub-step and whatever else the adjoint
needs to replay the step's sub-steps backwards). Adjoint and tape fields are
ordinary ``state`` tensors. A field that is both an input and a result (a
state carried from step to step) may name one adjoint field for both: the
adjoint reads the result's adjoint there and overwrites it, in place, with the
input's; all other adjoint and tape fields are distinct. Declaring a ``param``
input states that the model's kernels read it directly, so no derived field
caches it. Every key names a module reference of the model; entries that
name an ``optional_module_ref`` module are dropped while that module is closed,
so one declaration covers every combination of opened modules.

Differentiable mode is a switch of the model, not another way to step it::

    model.set_differentiable(True)
    model.assign_state({"base.storage": s0})       # graph tensors are kept
    for runoff in forcing:
        model.set_inputs(runoff=runoff)            # the model's own setters
        model.step_advance(time_step=timedelta(days=1))
        s = model.capture_state(["base.storage"])["base.storage"]
    loss(s).backward()                             # runs the @adjoint_step

While it is on, every value a model setter stages through
``copy_tensor_inputs`` (and every :meth:`assign_state`) into a declared input
is remembered as the field's graph value; after each managed step the step's
results become the graph values of the result fields, so state carries its
derivatives from step to step without being passed again, and
:meth:`capture_state` returns the graph values. A step whose remembered inputs
need no gradient (or that runs with gradients disabled) is a plain step: no
autograd node, copies or tape. Otherwise the step becomes one autograd node
that keeps copies of the tape fields; its backward restores them, stages the
results' adjoints and runs the model's ``@adjoint_step``.

Switching the mode off drops every graph value, so plain steps keep no tape
copies and no graph (the model's own adjoint and tape fields stay allocated as
built, and its kernels may keep filling them). While gradients are enabled, a
value that requires grad is then rejected by ``@between_steps`` setters and
:meth:`assign_state` before anything is copied, rather than silently cut; in
differentiable mode so is one assigned to a field without an adjoint, and a
setter body that stages one into such a field fails (which poisons the model,
like any failed between-steps body). Model code that writes a declared input
with its own in-place copy instead of ``copy_tensor_inputs`` leaves a stale
graph value behind. A graph recorded while the mode was on can still be
backpropagated after it is switched off.

Recomputation (``torch.utils.checkpoint``) repeats steps, so a differentiable
model must not have a simulation schedule: its clock only counts the steps.
A recomputed segment must assign its own start state, and the mode must stay
on until its backward has finished; afterwards the model's fields hold the
state of the last recomputed segment, so assign the state before stepping on.
Every call is a
between-steps operation, in the same order on every rank.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import torch
from pydantic import model_validator
from torch.autograd.function import once_differentiable

from hydroforge.core.validation import FrozenMapping, HydroForgeModel
from hydroforge.declare.module import ModuleReference
from hydroforge.declare.tensors import install_differentiable_staging
from hydroforge.execution.boundaries import between_steps
from hydroforge.execution.context import ACTIVE_MODEL
from hydroforge.execution.state_assignment import STATE_CATEGORIES, stage_fields, state_field

_F = TypeVar("_F", bound=Callable[..., Any])
_MARKER = "__hydroforge_adjoint_step__"
INPUT_CATEGORIES = STATE_CATEGORIES | {"forcing", "param"}


class StepAdjoint(HydroForgeModel):
    """Fields that connect one managed step to its adjoint (see module docs)."""

    inputs: FrozenMapping[str, str]
    outputs: FrozenMapping[str, str]
    tape: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _distinct(self) -> StepAdjoint:
        # A field that is both input and result may share one adjoint field:
        # the adjoint reads the result's adjoint there and overwrites it.
        adjoints = (
            *(v for k, v in self.inputs.items() if self.outputs.get(k) != v),
            *self.outputs.values(),
            *self.tape,
        )
        if len(set(adjoints)) != len(adjoints):
            raise ValueError(
                "adjoint and tape fields must be distinct, except that an input that is "
                "also a result may name that result's adjoint field"
            )
        if not self.outputs:
            raise ValueError("a step adjoint needs at least one result")
        return self


def adjoint_step(function: _F) -> _F:
    """Declare the model method that runs a step's adjoint (no arguments).

    The method runs between steps, so registered kernels bind as in a step.
    """

    guarded = between_steps(function)
    setattr(guarded, _MARKER, True)
    return guarded  # type: ignore[return-value]


_DECLARATIONS: dict[type, tuple[StepAdjoint, str, frozenset[str]]] = {}


def _keys(spec: StepAdjoint) -> tuple[str, ...]:
    return (*spec.inputs, *spec.inputs.values(), *spec.outputs, *spec.outputs.values(), *spec.tape)


def _class_attributes(cls: type) -> dict[str, Any]:
    """Class attributes along the MRO, most derived first."""

    attributes: dict[str, Any] = {}
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            attributes.setdefault(name, value)
    return attributes


def _declaration(cls: type) -> tuple[StepAdjoint, str, frozenset[str]]:
    """The class's StepAdjoint, its @adjoint_step method and its optional modules."""

    found = _DECLARATIONS.get(cls)
    if found is None:
        attributes = _class_attributes(cls)
        specs = [v for v in attributes.values() if isinstance(v, StepAdjoint)]
        methods = [name for name, v in attributes.items() if getattr(v, _MARKER, False)]
        if len(specs) != 1 or len(methods) != 1:
            raise TypeError(
                f"{cls.__name__} declares no step adjoint: differentiable mode needs one "
                "StepAdjoint class attribute and one @adjoint_step method"
            )
        refs = {name: v for name, v in attributes.items() if isinstance(v, ModuleReference)}
        for key in _keys(specs[0]):
            module_name, _, name = key.partition(".")
            if module_name not in refs or not name:
                raise TypeError(
                    f"{cls.__name__} StepAdjoint key {key!r} must be 'module.field' of a "
                    "module reference of the model"
                )
        optional = frozenset(name for name, ref in refs.items() if ref.optional)
        found = _DECLARATIONS[cls] = (specs[0], methods[0], optional)
    return found


def _open_declaration(model: Any) -> StepAdjoint:
    """The declaration restricted to the model's open modules."""

    spec, _, optional = _declaration(type(model))
    closed = frozenset(name for name in optional if getattr(model, name) is None)
    if not closed:
        return spec

    def live(*keys: str) -> bool:
        return not any(k.partition(".")[0] in closed for k in keys)

    return StepAdjoint(
        inputs={k: v for k, v in spec.inputs.items() if live(k, v)},
        outputs={k: v for k, v in spec.outputs.items() if live(k, v)},
        tape=tuple(k for k in spec.tape if live(k)),
    )


def _tensor(model: Any, key: str) -> torch.Tensor:
    module_name, _, name = key.partition(".")
    return getattr(getattr(model, module_name), name)


class StepTape:
    """Graph values of one model in differentiable mode (runtime-owned).

    ``values`` maps ``module.field`` of declared inputs and results to the
    tensor the autograd graph knows the field by.
    """

    def __init__(self, model: Any, spec: StepAdjoint, adjoint: Callable[[], None]) -> None:
        self.model = model
        self.spec = spec
        self.adjoint = adjoint
        self.inputs = tuple(spec.inputs)
        self.outputs = tuple(spec.outputs)
        # Every field a graph value can name; a staged copy into one replaces it.
        self.tracked = tuple(dict.fromkeys((*self.inputs, *self.outputs)))
        self.values: dict[str, torch.Tensor] = {}
        # Set while the framework itself stages tape and adjoint fields.
        self.suspended = False
        # Set while a recorded step runs (ManagedStep.differentiable).
        self.recording = False

    def _key(self, target: torch.Tensor) -> str | None:
        for key in self.tracked:
            if _tensor(self.model, key) is target:
                return key
        return None

    def stage(self, inputs: Iterable[tuple[str, tuple[Any, torch.Tensor]]]) -> None:
        """``copy_tensor_inputs`` in differentiable mode: copy and remember."""

        staged = []
        for name, (value, target) in inputs:
            key = self._key(target)
            grad = isinstance(value, torch.Tensor) and value.requires_grad
            if grad and key not in self.spec.inputs:
                raise ValueError(
                    f"input {name!r} requires grad, but {type(self.model).__name__}'s step "
                    "adjoint has no adjoint of its field; its gradient would be lost"
                )
            staged.append((key, value if grad else None, target, value))
        for key, graph, target, value in staged:
            with torch.no_grad():
                target.copy_(value)
            if key is not None:
                if graph is None:
                    self.values.pop(key, None)
                else:
                    self.values[key] = graph

    def record(self, advance: Callable[[], Any]) -> Any:
        """Run one managed step, as an autograd node when its inputs need it."""

        live = [key for key in self.inputs if key in self.values]
        if torch.is_grad_enabled() and any(self.values[key].requires_grad for key in live):
            returned: list[Any] = []
            results = _Step.apply(self, lambda: returned.append(advance()), tuple(live),
                                  *(self.values[k] for k in live))
            self.values.update(zip(self.outputs, results))
            return returned[-1]
        result = advance()
        for key in self.outputs:
            self.values.pop(key, None)
        return result


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tape, advance, keys, *values):
        tape.recording = True
        try:
            advance()
        finally:
            tape.recording = False
        model, spec = tape.model, tape.spec
        ctx.model, ctx.spec, ctx.adjoint, ctx.keys = model, spec, tape.adjoint, keys
        ctx.set_materialize_grads(False)
        if any(ctx.needs_input_grad[3:]):
            ctx.save_for_backward(*(_tensor(model, key).clone() for key in spec.tape))
        return tuple(_tensor(model, key).clone() for key in spec.outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        model, spec = ctx.model, ctx.spec
        staged = dict(zip(spec.tape, ctx.saved_tensors))
        for adjoint_key, grad in zip(spec.outputs.values(), grads):
            staged[adjoint_key] = (
                torch.zeros_like(_tensor(model, adjoint_key)) if grad is None else grad
            )
        tape = model.__pydantic_private__["_runtime"].step_tape
        if tape is not None:
            tape.suspended = True
        try:
            stage_fields(model, staged, STATE_CATEGORIES, "step adjoint")
        finally:
            if tape is not None:
                tape.suspended = False
        ctx.adjoint()
        return (None, None, None, *(
            _tensor(model, spec.inputs[key]).clone() if needed else None
            for key, needed in zip(ctx.keys, ctx.needs_input_grad[3:])
        ))


def _stage(inputs: Any) -> bool:
    """``copy_tensor_inputs`` of the model executing on this thread."""

    execution = ACTIVE_MODEL.get()
    tape = None if execution is None else execution.runtime.step_tape
    if tape is None or tape.suspended:
        return False
    tape.stage(inputs.items())
    return True


install_differentiable_staging(_stage)


@between_steps
def _switch(self: Any, *, tape: StepTape | None) -> None:
    self.__pydantic_private__["_runtime"].step_tape = tape


def set_differentiable(model: Any, enabled: bool) -> None:
    """Switch ``model`` into or out of differentiable mode between steps."""

    from hydroforge.execution.session import ModelRuntime

    runtime = ModelRuntime.of(model)
    if not enabled:
        if runtime.step_tape is not None:
            _switch(model, tape=None)
        return
    if runtime.step_tape is not None:
        return
    runtime.ensure_materialized(f"{type(model).__name__}.set_differentiable")
    spec = _open_declaration(model)
    if model.plan.schedule is not None:
        raise ValueError(
            "differentiable mode repeats steps under recomputation; build the model "
            "with initial_time instead of a simulation_schedule"
        )
    if len(runtime.execution.step_policies) != 1:
        raise TypeError(
            f"{type(model).__name__} has several @managed_step methods; differentiable "
            "mode needs one, the step its adjoint reverses"
        )
    for key in spec.inputs:
        state_field(model, key, INPUT_CATEGORIES)
    for key in (*spec.outputs, *spec.inputs.values(), *spec.outputs.values(), *spec.tape):
        try:
            state_field(model, key, STATE_CATEGORIES)
        except ValueError as error:
            raise ValueError(
                f"{type(model).__name__} is not differentiable as built: {error}"
            ) from None
    adjoint = getattr(model, _declaration(type(model))[1])
    _switch(model, tape=StepTape(model, spec, adjoint))


__all__ = ["StepAdjoint", "StepTape", "adjoint_step", "set_differentiable"]

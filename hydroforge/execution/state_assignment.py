# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""In-memory capture and assignment of model state and clock between steps.

Checkpoints (:meth:`AbstractModel.save_state`) persist a complete construction
input; these calls instead copy state tensors in device memory and move the
clock, for drivers that rerun, roll back or reverse steps (truncated
backpropagation through time, recomputation, line searches over whole runs).
The model keeps no history for them: whatever a driver wants to return to, it
captures itself.

State keys are ``module.field``. Only ``init_state`` and ``state`` tensors of
open modules are accepted. A value must have the field's local shape (this
rank's rows); it may live on any device, and dtype conversion follows
``Tensor.copy_``. Halo-exchanged fields are rejected while the model exchanges
halos: assigned owned rows would leave their ghost copies stale. Assigning
state does not touch the clock, statistics windows or derived fields.

A :class:`ModelClock` returned by :func:`capture_clock` may be assigned back
on the same timeline (until the clock is reset or the model rematerialized)
as long as it does not lie beyond the furthest point steps have reached:
steps from an earlier clock replay time already simulated and write no
output (see :mod:`hydroforge.execution.step`). Every call is a between-steps
operation, in the same order on every rank.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from hydroforge.core.identity import canonical
from hydroforge.core.time import DateLike
from hydroforge.declare.tensors import copy_tensor_inputs
from hydroforge.execution.boundaries import between_steps
from hydroforge.execution.context import managed_step_active

STATE_CATEGORIES = frozenset({"init_state", "state"})


@dataclass(frozen=True, slots=True)
class ModelClock:
    """One position of a model's clock (opaque to drivers apart from ``time``).

    ``time`` is the date of the next step (``None`` without one);
    ``schedule_index`` the next scheduled step; ``elapsed_microseconds`` the
    time stepped on this timeline.
    """

    time: DateLike | None
    schedule_index: int
    elapsed_microseconds: int
    timeline: int


def state_field(
    model: Any, key: str, categories: frozenset[str] = STATE_CATEGORIES
) -> tuple[torch.Tensor, Any]:
    """Tensor and declaration of the allocated ``module.field`` of an open module.

    Raises ``ValueError`` unless its category is one of ``categories``.
    """
    module_name, _, name = key.partition(".")
    if not name or "." in name:
        raise ValueError(f"state key {key!r} must be 'module.field'")
    module = getattr(model, module_name, None)
    if module is None or not hasattr(type(module), "spec"):
        raise ValueError(f"state key {key!r}: module {module_name!r} is not open")
    spec = type(module).spec().tensor_fields.get(name)
    if spec is None or spec.category not in categories:
        raise ValueError(
            f"state key {key!r} is not a {' or '.join(sorted(categories))} tensor"
        )
    tensor = getattr(module, name)
    if tensor is None:
        raise ValueError(f"state key {key!r} is not allocated in this model")
    return tensor, spec


def state_keys(
    model: Any, categories: Iterable[str] = ("init_state",)
) -> tuple[str, ...]:
    """``module.field`` of every allocated state tensor of the given categories."""

    wanted = frozenset(categories)
    if not wanted <= STATE_CATEGORIES:
        raise ValueError(f"state categories must be among {sorted(STATE_CATEGORIES)}")
    keys = []
    for module_name in model.opened_modules:
        module = getattr(model, module_name)
        for name, spec in type(module).spec().tensor_fields.items():
            if spec.category in wanted and getattr(module, name) is not None:
                keys.append(f"{module_name}.{name}")
    return tuple(keys)


def _require_readable(model: Any, what: str) -> Any:
    from hydroforge.execution.session import ModelRuntime

    runtime = ModelRuntime.of(model)
    runtime.require_materialized(f"{type(model).__name__}.{what}")
    failure = runtime.execution.failure
    if failure is not None:
        raise runtime.execution.poisoned_error(failure)
    return runtime


def capture_state(
    model: Any, keys: Iterable[str] | None = None
) -> dict[str, torch.Tensor]:
    """Detached copies of state tensors (default: every ``init_state`` field)."""

    if managed_step_active():
        raise RuntimeError("capture_state is allowed only between managed steps")
    _require_readable(model, "capture_state")
    names = state_keys(model) if keys is None else tuple(keys)
    return {key: state_field(model, key)[0].detach().clone() for key in names}


def _targets(
    model: Any, values: Mapping[str, Any]
) -> tuple[dict[str, tuple[Any, torch.Tensor]], tuple[torch.Tensor, ...]]:
    targets = {}
    halo_tensors: list[torch.Tensor] = []
    seen_halo: set[int] = set()
    halo_enabled = model.halo_enabled
    for key, value in values.items():
        tensor, spec = state_field(model, key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state {key!r} must be a tensor")
        if value.requires_grad and torch.is_grad_enabled():
            raise ValueError(
                f"state {key!r} requires grad: assigning it would cut its gradient "
                "(assign a detached value under the driver's own autograd node)"
            )
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(
                f"state {key!r} has shape {tuple(value.shape)}; the local field is "
                f"{tuple(tensor.shape)}"
            )
        targets[key] = (value, tensor)
        if spec.tensor.halo and halo_enabled and id(tensor) not in seen_halo:
            seen_halo.add(id(tensor))
            halo_tensors.append(tensor)
    return targets, tuple(halo_tensors)


@between_steps
def _assign(
    self: Any,
    *,
    targets: Mapping[str, tuple[Any, torch.Tensor]],
    halo_tensors: tuple[torch.Tensor, ...] = (),
) -> None:
    from hydroforge.execution.session import ModelRuntime
    from hydroforge.parallel.halo import MAX_FIELDS

    with torch.no_grad():
        copy_tensor_inputs(targets)
        if halo_tensors:
            runtime = ModelRuntime.of(self)
            halo = runtime.halo
            if halo is not None:
                layouts = [runtime.halo_layout(tensor) for tensor in halo_tensors]
                for start in range(0, len(halo_tensors), MAX_FIELDS):
                    stop = start + MAX_FIELDS
                    halo.exchange(
                        halo_tensors[start:stop], layouts[start:stop], "mirror"
                    )


def assign_state(model: Any, values: Mapping[str, torch.Tensor]) -> None:
    """Copy ``values`` into the model's state tensors, in place.

    Every rank validates its values before any rank copies; halo-exchanged
    fields refresh their ghost rows across ranks inside the same transaction.
    """

    from hydroforge.execution.session import ModelRuntime

    if managed_step_active():
        raise RuntimeError("assign_state is allowed only between managed steps")
    runtime = ModelRuntime.of(model)
    error: BaseException | None = None
    targets: dict[str, tuple[Any, torch.Tensor]] = {}
    halo_tensors: tuple[torch.Tensor, ...] = ()
    signature = None
    try:
        runtime.ensure_materialized(f"{type(model).__name__}.assign_state")
        targets, halo_tensors = _targets(model, values)
        signature = tuple(values.keys())
    except BaseException as caught:
        error = caught
    runtime.channel.preflight(
        error,
        phase="state.assign_state.validation",
        scope="distributed state assignment validation",
        signature=signature,
    )
    _assign(model, targets=targets, halo_tensors=halo_tensors)


def capture_clock(model: Any) -> ModelClock:
    """The model's clock, to assign back later with :func:`assign_clock`."""

    if managed_step_active():
        raise RuntimeError("capture_clock is allowed only between managed steps")
    clock = _require_readable(model, "capture_clock").clock
    return ModelClock(
        clock.time, clock.schedule_index, clock.elapsed_microseconds, clock.timeline
    )


def assign_clock(model: Any, value: ModelClock) -> None:
    """Move the clock to a captured position no later than its frontier."""

    from hydroforge.execution.session import ModelRuntime

    if managed_step_active():
        raise RuntimeError("assign_clock is allowed only between managed steps")
    runtime = ModelRuntime.of(model)
    error: BaseException | None = None
    signature = None
    try:
        runtime.require_healthy(f"{type(model).__name__}.assign_clock")
        if not isinstance(value, ModelClock):
            raise TypeError("assign_clock takes a ModelClock from capture_clock")
        clock = runtime.clock
        if value.timeline != clock.timeline:
            raise ValueError(
                "the clock was captured on another timeline (before a clock reset "
                "or rematerialization)"
            )
        if value.elapsed_microseconds > clock.frontier:
            raise ValueError("the clock cannot move beyond the furthest step reached")
        if (
            runtime.plan.parameters
            and value.elapsed_microseconds != clock.elapsed_microseconds
        ):
            raise ValueError(
                "a model with scheduled parameter changes cannot move its clock"
            )
        signature = canonical(
            (value.time, value.schedule_index, value.elapsed_microseconds)
        )
    except BaseException as caught:
        error = caught
    runtime.channel.preflight(
        error,
        phase="clock.assign.validation",
        scope="distributed clock assignment validation",
        signature=signature,
    )
    clock = runtime.clock
    clock.time = value.time
    clock.schedule_index = value.schedule_index
    clock.elapsed_microseconds = value.elapsed_microseconds


__all__ = [
    "ModelClock",
    "assign_clock",
    "assign_state",
    "capture_clock",
    "capture_state",
    "state_field",
    "state_keys",
]

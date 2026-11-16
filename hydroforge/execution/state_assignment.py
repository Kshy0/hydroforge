# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""In-memory capture and assignment of model state between managed steps.

Checkpoints (:meth:`AbstractModel.save_state`) persist a complete construction
input; these two calls instead copy state tensors in device memory, for
drivers that rerun or roll back steps (truncated backpropagation through time,
recomputation, line searches over whole runs). Keys are ``module.field``. In
differentiable mode both carry autograd graph values (see
:mod:`hydroforge.execution.autodiff`).

Only ``init_state`` and ``state`` tensors of open modules are accepted. A value
must have the field's local shape (this rank's rows); dtype conversion follows
``Tensor.copy_``. Halo-exchanged fields are rejected while the model exchanges
halos: assigned owned rows would leave their ghost copies stale. The model's
clock, statistics windows and derived fields are not touched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from hydroforge.declare.tensors import copy_tensor_inputs
from hydroforge.execution.boundaries import between_steps
from hydroforge.execution.context import managed_step_active

STATE_CATEGORIES = frozenset({"init_state", "state"})


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
        raise ValueError(f"state key {key!r} is not a {' or '.join(sorted(categories))} tensor")
    tensor = getattr(module, name)
    if tensor is None:
        raise ValueError(f"state key {key!r} is not allocated in this model")
    return tensor, spec


def state_keys(model: Any, categories: Iterable[str] = ("init_state",)) -> tuple[str, ...]:
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


def capture_state(model: Any, keys: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    """Copies of state tensors (default: every ``init_state`` field).

    Detached, except in differentiable mode, where a field with an autograd
    graph value (a step result, or a value assigned with its graph) returns
    that value (see :mod:`hydroforge.execution.autodiff`).
    """

    if managed_step_active():
        raise RuntimeError("capture_state is allowed only between managed steps")
    names = state_keys(model) if keys is None else tuple(keys)
    tape = model.__pydantic_private__["_runtime"].step_tape
    graph = {} if tape is None else tape.values
    captured = {}
    for key in names:
        tensor = state_field(model, key)[0]
        value = graph.get(key)
        captured[key] = tensor.detach().clone() if value is None else value
    return captured


def _targets(
    model: Any, values: Mapping[str, Any], categories: frozenset[str]
) -> dict[str, tuple[Any, torch.Tensor]]:
    tape = model.__pydantic_private__["_runtime"].step_tape
    targets = {}
    for key, value in values.items():
        tensor, spec = state_field(model, key, categories)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state {key!r} must be a tensor")
        if (value.requires_grad and torch.is_grad_enabled()
                and (tape is None or key not in tape.spec.inputs)):
            raise ValueError(
                f"state {key!r} requires grad, but "
                + ("the model is not in differentiable mode" if tape is None
                   else "the model's step adjoint has no adjoint of it")
                + ": assigning it would cut its gradient"
            )
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(
                f"state {key!r} has shape {tuple(value.shape)}; the local field is "
                f"{tuple(tensor.shape)}"
            )
        if spec.tensor.halo and model.halo_enabled:
            raise ValueError(
                f"state {key!r} is halo-exchanged; assigning it would leave ghost rows stale"
            )
        targets[key] = (value, tensor)
    return targets


@between_steps
def _assign(self: Any, *, targets: Mapping[str, tuple[Any, torch.Tensor]]) -> None:
    with torch.no_grad():
        copy_tensor_inputs(targets)


def assign_state(model: Any, values: Mapping[str, torch.Tensor]) -> None:
    """Copy ``values`` into the model's state tensors, in place.

    Every rank validates its values before any rank copies; a failure while
    copying poisons the model like any between-steps mutation.
    """

    stage_fields(model, values, STATE_CATEGORIES, "assign_state")


def stage_fields(
    model: Any, values: Mapping[str, torch.Tensor], categories: frozenset[str], what: str
) -> None:
    """:func:`assign_state` for fields of ``categories`` (framework-internal)."""

    from hydroforge.execution.session import ModelRuntime

    if managed_step_active():
        raise RuntimeError(f"{what} is allowed only between managed steps")
    runtime = ModelRuntime.of(model)
    error: BaseException | None = None
    targets: dict[str, tuple[Any, torch.Tensor]] = {}
    try:
        runtime.ensure_materialized(f"{type(model).__name__}.{what}")
        targets = _targets(model, values, categories)
    except BaseException as caught:
        error = caught
    runtime.channel.preflight(
        error,
        phase=f"state.{what}.validation",
        scope="distributed state assignment validation",
    )
    _assign(model, targets=targets)


__all__ = ["assign_state", "capture_state", "stage_fields", "state_field", "state_keys"]

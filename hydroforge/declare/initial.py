# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""How a model's state is initialized once its modules are selected.

1. The plan tabulates the fields of the opened modules; the input fields the
   model input lacks are ``AbstractModel.missing_inputs``.
2. Construction reads the inputs (``param``, ``topology`` and ``init_state``
   fields, defaults where absent), evaluates the computed ``topology``,
   ``derived_param`` and ``state`` tensors, and allocates the buffer
   virtuals declaring ``materialize=``. Inputs stay as read: a checkpoint
   saves the same parameters and topology it was built from.
3. ``AbstractModel.initialize_model_state()`` updates state in place, in the
   order it chooses, and may run registered kernels; it branches on
   ``missing_inputs`` to tell a cold start from a restart.
4. Computed state declared ``after_init=True`` is evaluated from the
   initialized state, so a value that follows from the state (a depth from
   a storage) needs no code in the hook.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import Any

_MISSING = object()


class AfterInitProperty(cached_property):
    """A computed state tensor evaluated from the initialized state.

    Evaluated on first read once every module is constructed: a read inside
    ``initialize_model_state`` sees the state at that point, and the model
    evaluates the rest after the hook.
    """

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        cached = instance.__dict__.get(self.attrname, _MISSING)
        if cached is not _MISSING:
            return cached
        private = getattr(instance, "__pydantic_private__", None)
        tensors = None if private is None else private.get("_tensors")
        if tensors is None or not tensors.constructed:
            raise RuntimeError(
                f"{type(instance).__name__}.{self.attrname} is computed after "
                "initialization; it cannot be read while modules are "
                "constructed (for example by a derived_param tensor or a "
                "validator)"
            )
        field = instance.spec().tensor_fields[self.attrname]
        value = tensors._finish_computed(field, self.func(instance))
        instance.__dict__[self.attrname] = value
        return value


def materialize_declared(modules: Iterable[Any]) -> None:
    """Allocate the buffer virtuals whose ``materialize`` conditions hold."""

    for module in modules:
        module.materialize_fields(*sorted(module._binding.plan.materialized))


def evaluate_after_init(modules: Iterable[Any]) -> None:
    """Evaluate every active ``after_init`` tensor not read by the hook."""

    for module in modules:
        active = module._binding.plan.active
        for field in module.spec().tensor_fields.values():
            if field.computed and field.tensor.after_init and field.name in active:
                getattr(module, field.name)


__all__ = ["AfterInitProperty", "evaluate_after_init", "materialize_declared"]

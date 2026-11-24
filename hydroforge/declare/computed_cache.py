# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""One module value shared by several declared fields, evaluated once.

A ``computed_cache`` is not a field: it is not stored as model state, has no
shape or metadata, and is not output or checkpointed. It holds an
intermediate several fields derive from (a decomposition several layout
tensors and their dimensions are cut from, or a solve several derived
parameters read), so the work runs once per module rather than once per
field.

Lifecycle:

- Each object caches its own value: the payload view that resolves declared
  dimensions before construction, and the constructed module.
- Derived parameters see through it: one reading a cache depends on every
  field the cache reads (``hydroforge.execution.parameters``).
- A parameter change re-evaluates a cache it leaves stale once for all
  derived parameters reading it; the module's value is dropped, restored if
  the transaction rolls back, and evaluated again on its next read. Other
  values reading it (dimensions, kernel fields, bound or not) must keep
  their value: the change is rejected otherwise.
- A structural or content update re-derives what reads a cache it leaves
  stale (``hydroforge.execution.derivation``) before anything mutates.
- A cache read from a discarded input is fixed: an update reaching it fails.
- Like computed tensors, it is not re-evaluated by specialization updates.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from typing import Any, TypeVar

T = TypeVar("T")

_TRANSIENT: ContextVar[dict[tuple[int, str], Any] | None] = ContextVar(
    "hydroforge_transient_computed_caches", default=None
)


class ComputedCache(cached_property):
    """Descriptor of :func:`computed_cache`; see the module documentation."""

    __hydroforge_computed_cache__ = True

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        name = self.attrname
        cache = instance.__dict__
        if name in cache:
            return cache[name]
        transient = _TRANSIENT.get()
        if transient is not None:
            # Inside a parameter refresh: shared by this refresh's consumers,
            # never kept (it may be an inference tensor of a refresh that
            # rolls back).
            key = (id(instance), name)
            if key not in transient:
                transient[key] = self.func(instance)
            return transient[key]
        value = self.func(instance)
        cache[name] = value
        return value


def computed_cache(function: Callable[[Any], T]) -> ComputedCache:
    """Cache one value several declared fields of a module derive from.

    Decorate a method taking only the module; read it as an attribute
    (``self._plan["outlet"]``). The value may be any object. Derived
    parameters may read it: their dependencies, refresh and rollback follow
    the fields it reads (see :mod:`hydroforge.declare.computed_cache`).
    Its body must be pure, like a derived parameter formula.
    """

    if isinstance(function, (property, cached_property)):
        raise TypeError(
            "computed_cache caches its getter; decorate the function itself"
        )
    try:
        inspect.signature(function).bind(None)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "computed_cache requires a callable taking only its module"
        ) from error
    return ComputedCache(function)


@contextmanager
def transient_computed_caches() -> Iterator[None]:
    """Share caches that are not kept across the reads inside this block."""

    token = _TRANSIENT.set({})
    try:
        yield
    finally:
        _TRANSIENT.reset(token)


__all__ = [
    "ComputedCache",
    "computed_cache",
    "transient_computed_caches",
]

"""Kernel-recording context shared by logical and backend dispatchers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, TypeVar

from pydantic import InstanceOf

from hydroforge.contracts.kernels import KernelSpec
from hydroforge.contracts.validation import HydroForgeModel


_F = TypeVar("_F", bound=Callable[..., Any])


class _RegistryFactoryDeclaration(HydroForgeModel):
    function: Callable[..., Any]


class _RegistryFactoryInvocation(HydroForgeModel):
    """One registry-owned factory invocation with its canonical ABI."""

    spec: InstanceOf[KernelSpec]


_ACTIVE_OPERATOR_RECORDER: ContextVar[Any | None] = ContextVar(
    "hydroforge_operator_recorder", default=None,
)
_ACTIVE_KERNEL_SPEC: ContextVar[Any | None] = ContextVar(
    "hydroforge_kernel_factory_spec", default=None,
)


@contextmanager
def kernel_factory_contract(spec: Any):
    """Expose one registry's canonical Spec only while its factory builds."""
    token = _ACTIVE_KERNEL_SPEC.set(spec)
    try:
        yield
    finally:
        _ACTIVE_KERNEL_SPEC.reset(token)


def active_kernel_spec() -> Any | None:
    return _ACTIVE_KERNEL_SPEC.get()


def registry_factory(function: _F) -> _F:
    """Declare a helper that is valid only while a registry builds a backend.

    This is the explicit source form for lazy native catalogs that consume the
    enclosing registry's KernelSpec instead of repeating it.
    """
    declaration = _RegistryFactoryDeclaration(function=function)
    function = declaration.function

    @wraps(function)
    def guarded(*args, **kwargs):
        _RegistryFactoryInvocation(spec=active_kernel_spec())
        return function(*args, **kwargs)

    guarded.__hydroforge_registry_factory__ = True
    return guarded


@contextmanager
def native_component_factory():
    """Build a private backend component outside the logical ABI context.

    A sequence component has its own launch extent but is not a separately
    registered logical kernel.  Suspending the enclosing Spec prevents its
    native geometry from being mistaken for duplicate public ABI metadata.
    """
    token = _ACTIVE_KERNEL_SPEC.set(None)
    try:
        yield
    finally:
        _ACTIVE_KERNEL_SPEC.reset(token)


def active_operator_recorder() -> Any | None:
    return _ACTIVE_OPERATOR_RECORDER.get()


def compiled_operator_entry(function: _F) -> _F:
    """Mark a framework function as one nominal substep IR operator entry."""

    function.__hydroforge_compiled_operator__ = True
    return function

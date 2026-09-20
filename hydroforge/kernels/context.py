"""Kernel-recording context shared by logical and backend dispatchers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar

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
    "hydroforge_operator_recorder",
    default=None,
)
_ACTIVE_KERNEL_SPEC: ContextVar[Any | None] = ContextVar(
    "hydroforge_kernel_factory_spec",
    default=None,
)
_ACTIVE_TRITON_PRECISION: ContextVar[tuple[str, frozenset[str]] | None] = ContextVar(
    "hydroforge_triton_precision",
    default=None,
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


def resolve_factory_spec(spec: KernelSpec | None, *, factory: str) -> KernelSpec:
    """Choose the sole explicit or registry-owned declaration for a factory."""
    active = active_kernel_spec()
    if active is not None:
        if spec is not None:
            raise TypeError(f"{factory} may not repeat active KernelSpec metadata")
        return active
    if spec is None:
        raise TypeError(
            f"{factory} requires a KernelSpec outside a BackendRegistry factory"
        )
    return spec


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
    with kernel_factory_contract(None):
        yield


@contextmanager
def triton_precision_context(
    precision: str,
    scalar_names: frozenset[str] = frozenset(),
):
    """Expose one resolved Triton scalar ABI while a program is launching.

    Compound programs are allowed to contain ordinary Python launch helpers,
    so their inner Triton kernels cannot receive the logical ``KernelSpec``
    through the normal factory context.  This small context carries only the
    resolved floating-point ABI needed by :func:`launch_triton_kernel`.
    """

    token = _ACTIVE_TRITON_PRECISION.set((precision, scalar_names))
    try:
        yield
    finally:
        _ACTIVE_TRITON_PRECISION.reset(token)


def active_triton_precision() -> tuple[str, frozenset[str]] | None:
    """Return the active compound-program Triton precision contract."""

    return _ACTIVE_TRITON_PRECISION.get()


def active_operator_recorder() -> Any | None:
    return _ACTIVE_OPERATOR_RECORDER.get()


def compiled_operator_entry(function: _F) -> _F:
    """Mark a framework function as one nominal substep IR operator entry."""

    function.__hydroforge_compiled_operator__ = True
    return function

"""Kernel-recording context shared by logical and backend dispatchers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
from typing import Any


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


def require_active_kernel_spec() -> Any:
    """Return the registry contract or fail outside backend construction."""
    spec = active_kernel_spec()
    if spec is None:
        raise RuntimeError("no active BackendRegistry KernelSpec")
    return spec


def registry_factory(function):
    """Declare a helper that is valid only while a registry builds a backend.

    This is the explicit source form for lazy native catalogs that consume the
    enclosing registry's KernelSpec instead of repeating it.
    """
    @wraps(function)
    def guarded(*args, **kwargs):
        if active_kernel_spec() is None:
            raise RuntimeError(
                f"registry factory {function.__qualname__} was called outside "
                "BackendRegistry.resolve()"
            )
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


def compiled_operator_entry(function):
    """Mark a framework function as one nominal substep IR operator entry."""

    if not function.__module__.startswith("hydroforge."):
        raise TypeError(
            "compiled_operator_entry is reserved for HydroForge-owned IR "
            "operators; downstream physics must use BackendRegistry"
        )
    function.__hydroforge_compiled_operator__ = True
    return function


def is_compiled_operator_entry(value: Any) -> bool:
    """Return only the explicit nominal marker; names carry no meaning."""

    try:
        marker = inspect.getattr_static(
            value, "__hydroforge_compiled_operator__",
        )
    except AttributeError:
        return False
    return marker is True


def reject_direct_kernel_launch(name: str) -> None:
    """Reject native dispatchers that bypass the registered kernel entry."""
    if active_operator_recorder() is not None:
        raise RuntimeError(
            f"kernel {name!r} was launched through a backend dispatcher while "
            "a compiled substep was being recorded; expose it through "
            "BackendRegistry + KernelSpec and call the registered entry"
        )

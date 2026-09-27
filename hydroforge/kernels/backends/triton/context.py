"""Resolved scalar ABI for nested Triton program launches."""

from contextlib import contextmanager
from contextvars import ContextVar

_ACTIVE_TRITON_PRECISION: ContextVar[tuple[str, frozenset[str]] | None] = ContextVar(
    "hydroforge_triton_precision",
    default=None,
)


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

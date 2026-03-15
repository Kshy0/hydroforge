# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Kernel backend selection and adapter for hydroforge.

Set ``HYDROFORGE_BACKEND`` to choose the backend explicitly::

    export HYDROFORGE_BACKEND=cuda     # CUDA C++ kernels (NVIDIA)
    export HYDROFORGE_BACKEND=metal    # Metal shaders (Apple Silicon)
    export HYDROFORGE_BACKEND=triton   # Triton JIT kernels (NVIDIA/AMD)
    export HYDROFORGE_BACKEND=torch    # Pure-PyTorch fallback

When unset, auto-detection picks the best available backend:
``cuda`` → ``metal`` → ``triton`` → ``torch``.

When 'torch' is selected, :class:`KernelAdapter` wraps each PyTorch kernel
so it can be called with the unified hydroforge kwargs convention:

    kernel(arg_ptr=tensor, scalar_arg=value, BLOCK_SIZE=bs)

The adapter:
  1. Strips the ``_ptr`` suffix from tensor argument names.
  2. Drops unknown kwargs (``BLOCK_SIZE``, batch flags, etc.).
  3. Passes scalars directly (no buffer conversion needed).
"""

import os
from typing import Any, Callable


def _resolve_backend() -> str:
    """Resolve kernel backend from HYDROFORGE_BACKEND environment variable.

    When the variable is unset, auto-detect in priority order:
    ``triton`` → ``cuda`` → ``metal`` → ``torch``.
    The CUDA backend is still experimental.
    """
    env = os.environ.get("HYDROFORGE_BACKEND", "").strip().lower()
    if env:
        return env
    try:
        import triton  # noqa: F401
        return "triton"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None) is not None:
                return "hip"
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "metal"
    except ImportError:
        pass
    return "torch"


KERNEL_BACKEND: str = _resolve_backend()
os.environ.setdefault("HYDROFORGE_BACKEND", KERNEL_BACKEND)


def _torch_compile(fn: Callable) -> Callable:
    """Apply torch.compile with inference-optimized settings.

    All physics kernels mutate inputs via ``.copy_()`` / indexed assignment,
    so ``reduce-overhead`` mode (which relies on internal CUDA graphs) can
    never actually use its main optimisation and only produces warnings.
    We use ``fullgraph=True`` so that compilation errors surface at the
    first call rather than lazily on a rare code-path hours later.
    """
    import torch
    return torch.compile(fn, fullgraph=True)


class KernelAdapter:
    """Adapt a kernel to accept the unified hydroforge kwargs convention.

    Transforms caller kwargs to match the kernel's actual signature:
      1. Strips the ``_ptr`` suffix from tensor argument names.
      2. Drops unknown kwargs (``BLOCK_SIZE``, batch flags, etc.).
      3. Passes scalars directly (no buffer conversion needed).

    On the first call a specialised caller function is generated via ``exec``
    that maps caller kwargs directly to kernel parameters **without** building
    an intermediate dict.  All subsequent calls go through the cached fast-path
    with zero overhead beyond one pointer comparison.
    """

    __slots__ = (
        "_kernel",
        "_kernel_raw",
        "_param_names",
        "_key_map",
        "_fast_caller",
    )

    def __init__(self, kernel_func: Callable, *, compile: bool = True):
        import inspect
        self._kernel_raw = kernel_func
        sig = inspect.signature(kernel_func)
        self._param_names: set[str] = set(sig.parameters.keys())
        self._key_map: dict[str, str] = {}  # caller key -> kernel param name
        self._fast_caller: Callable | None = None
        if compile:
            self._kernel = _torch_compile(kernel_func)
        else:
            self._kernel = kernel_func

    # ------------------------------------------------------------------
    # Key resolution (lazy-cached)
    # ------------------------------------------------------------------

    def _resolve_key(self, key: str) -> str:
        """Map a caller kwarg name to the kernel parameter name.

        Returns the mapped name, or an empty string if the key should be
        dropped (e.g. ``BLOCK_SIZE``).
        """
        try:
            return self._key_map[key]
        except KeyError:
            pass
        base_key = key[:-4] if key.endswith("_ptr") else key
        if base_key in self._param_names:
            self._key_map[key] = base_key
            return base_key
        if key in self._param_names:
            self._key_map[key] = key
            return key
        self._key_map[key] = ""  # sentinel: drop this kwarg
        return ""

    # ------------------------------------------------------------------
    # Fast-caller generation
    # ------------------------------------------------------------------

    def _build_fast_caller(self, kwargs_keys: tuple[str, ...]) -> Callable:
        """Generate a specialised function that avoids per-call dict building.

        The generated function has the form::

            def _fast(kw, _k=_kernel):
                return _k(param_a=kw['a_ptr'], param_b=kw['b_ptr'], ...)
        """
        pairs = []
        for k in kwargs_keys:
            mapped = self._resolve_key(k)
            if mapped:
                pairs.append((k, mapped))

        args_str = ", ".join(f"{m}=_kw['{k}']" for k, m in pairs)
        code = f"def _fast(_kw, _k=_kernel): return _k({args_str})"
        ns: dict[str, Any] = {"_kernel": self._kernel}
        exec(code, ns)  # noqa: S102 – generated code is fully controlled
        return ns["_fast"]

    def __call__(self, **kwargs: Any):
        caller = self._fast_caller
        if caller is None:
            caller = self._build_fast_caller(tuple(kwargs.keys()))
            self._fast_caller = caller
        return caller(kwargs)


def adapt_kernel(kernel_func: Callable, *, compile: bool = True) -> KernelAdapter:
    """Create a :class:`KernelAdapter` for the given kernel function.

    Args:
        compile: If False the kernel is used as-is (useful for log / diagnostic
                 kernels that contain ``.item()`` calls which break the graph).
    """
    return KernelAdapter(kernel_func, compile=compile)


# ── Triton dispatcher factory ─────────────────────────────────────────────

def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def make_triton_dispatcher(
    kernel,
    *,
    batched_kernel=None,
    size_key: str = "num_catchments",
    batched_grid: str = "parallel",
    batched_drop: tuple[str, ...] = (),
    non_batched_drop: tuple[str, ...] = (),
) -> Callable:
    """Create a unified dispatch function for a Triton kernel pair.

    The returned function accepts ``**kw`` and handles grid computation,
    ``BLOCK_SIZE`` / ``num_trials`` extraction, and batched / non-batched
    parameter filtering automatically.

    Args:
        kernel: Non-batched Triton JIT kernel.
        batched_kernel: Batched variant (or ``None``).
        size_key: kwarg name whose value determines the grid size
            (e.g. ``"num_catchments"``, ``"num_bifurcation_paths"``).
        batched_grid: ``"parallel"`` → ``cdiv(n*nt, BS)``; ``"loop"`` → ``cdiv(n, BS)``.
        batched_drop: Keys to pop before calling the *batched* kernel
            (params present in non-batched but absent in batched).
        non_batched_drop: Extra keys to pop before calling the *non-batched*
            kernel (besides automatic ``batched_*`` stripping).
    """
    def dispatch(**kw):
        bs = kw.pop("BLOCK_SIZE", 256)
        nt = kw.pop("num_trials", None)
        n = kw[size_key]
        if nt is not None and batched_kernel is not None:
            for k in batched_drop:
                kw.pop(k, None)
            if batched_grid == "parallel":
                grid = lambda META: (_cdiv(n * nt, META["BLOCK_SIZE"]),)
            else:
                grid = lambda META: (_cdiv(n, META["BLOCK_SIZE"]),)
            batched_kernel[grid](BLOCK_SIZE=bs, num_trials=nt, **kw)
        else:
            for k in list(kw):
                if k.startswith("batched_"):
                    del kw[k]
            for k in non_batched_drop:
                kw.pop(k, None)
            grid = lambda META: (_cdiv(n, META["BLOCK_SIZE"]),)
            kernel[grid](BLOCK_SIZE=bs, **kw)
    return dispatch


# ── Metal dispatcher factory ──────────────────────────────────────────────

_MISSING = object()

# Singleton 1-element int32 tensor on MPS, allocated once.
_metal_dummy: dict[str, "torch.Tensor"] = {}


def _metal_dummy_buf(device):
    """Return a 1-element int32 MPS tensor to stand in for a None buffer."""
    key = str(device)
    buf = _metal_dummy.get(key)
    if buf is None:
        import torch
        buf = torch.zeros(1, dtype=torch.int32, device=device)
        _metal_dummy[key] = buf
    return buf


def make_metal_dispatcher(
    msl_source,
    kernel_name: str,
    args: tuple[str, ...],
    *,
    size_key: str = "num_catchments",
    template_vars: dict[str, str] | None = None,
    arg_defaults: dict[str, Any] | None = None,
    group_size: int = 256,
) -> Callable:
    """Create a unified dispatch function for a Metal shader kernel.

    Parallel to :func:`make_triton_dispatcher` but for Metal (MPS) shaders
    compiled via ``torch.mps.compile_shader()``.

    The returned function accepts ``**kw`` and handles shader compilation
    (with optional template specialization), ``bool``→``int`` conversion,
    and positional argument extraction automatically.

    On the **first call** a specialised caller function is generated via
    ``exec`` that maps caller kwargs directly to kernel positional args
    without any per-call loop or type checks.  All subsequent calls go
    through the cached fast-path.

    Args:
        msl_source: MSL source code as a ``str``, **or** a ``pathlib.Path`` /
            ``os.PathLike`` pointing to a ``.metal`` file loaded at import time.
        kernel_name: Entry-point function name in the MSL source.
        args: Ordered tuple of kwarg names matching ``[[buffer(N)]]`` bindings.
            ``bool`` values are automatically converted to ``int`` (Metal has
            no native ``bool`` constant-buffer type).
        size_key: kwarg name providing the thread count (grid size).
        template_vars: ``{placeholder: kwarg_key}`` for compile-time
            specialization.  The shader is recompiled and cached per unique
            combination of template values.
        arg_defaults: Default values for optional kwargs (e.g.
            ``{"HAS_RESERVOIR": False}``).  Missing kwargs without a default
            raise ``KeyError``.
        group_size: Threads per threadgroup (default 256).

    Returns:
        A callable accepting ``**kw`` that compiles (first call), extracts
        args in buffer-binding order, and dispatches the Metal kernel.

    Examples::

        # Inline MSL — simplest case
        compute_kernel = make_metal_dispatcher(
            _MSL_SOURCE, "my_kernel",
            args=("buf_a_ptr", "buf_b_ptr", "num_elements"),
            size_key="num_elements",
        )

        # .metal file + template specialization
        from pathlib import Path
        compute_kernel = make_metal_dispatcher(
            Path(__file__).parent / "my_kernel.metal",
            "my_kernel",
            args=("buf_ptr", "num_elements"),
            size_key="num_elements",
            template_vars={"__BLOCK__": "block_size"},
        )
    """
    from pathlib import Path

    source: str
    if isinstance(msl_source, (os.PathLike, Path)):
        source = Path(msl_source).read_text()
    else:
        source = msl_source

    _defaults = arg_defaults or {}
    _tpl_keys = tuple(template_vars.values()) if template_vars else ()

    # ---- code-gen: build the arg-extraction body once ----
    # The body is shared between template and non-template variants;
    # only the lib acquisition differs.
    _ns: dict[str, Any] = {}
    _body_lines: list[str] = []
    _arg_names: list[str] = []

    for i, k in enumerate(args):
        a = f"_a{i}"
        _arg_names.append(a)
        dv = _defaults.get(k, _MISSING)

        if dv is not _MISSING and isinstance(dv, bool):
            # Bool arg with default — store pre-converted int default.
            # Still need a runtime check because callers *may* override
            # with a Python bool.
            _ns[f"_d{i}"] = int(dv)
            _body_lines.append(
                f"  {a} = _kw.get('{k}', _d{i}); "
                f"{a} = int({a}) if {a}.__class__ is bool else {a}"
            )
        elif dv is not _MISSING:
            _ns[f"_d{i}"] = dv
            _body_lines.append(f"  {a} = _kw.get('{k}', _d{i})")
        else:
            _body_lines.append(f"  {a} = _kw['{k}']")

        # Nullable tensor → substitute dummy buffer
        if k.endswith("_ptr"):
            _body_lines.append(f"  if {a} is None: {a} = _dummy")

    _call_args = ", ".join(_arg_names)
    _fn_body = "\n".join(_body_lines)
    _call_line = (
        f"  _lib.{kernel_name}({_call_args}, "
        f"threads=_kw['{size_key}'], group_size={group_size})"
    )

    # _state[0] holds the compiled fast-path callable (None until first call)
    _state: list = [None]

    if template_vars:
        # ---- template path: lib varies per template key ----
        _shader_cache: dict[Any, Any] = {}

        def _init():
            import torch
            _ns["_dummy"] = _metal_dummy_buf(torch.device("mps"))
            code = f"def _fast(_kw, _lib):\n{_fn_body}\n{_call_line}"
            exec(code, _ns)  # noqa: S102 – generated code is fully controlled
            return _ns["_fast"], torch

        def dispatch(**kw):
            s = _state[0]
            if s is None:
                s = _init()
                _state[0] = s
            fast_fn, torch_mod = s

            cache_key = tuple(kw[k] for k in _tpl_keys)
            lib = _shader_cache.get(cache_key)
            if lib is None:
                src = source
                for placeholder, kwarg_key in template_vars.items():
                    src = src.replace(placeholder, str(kw[kwarg_key]))
                lib = torch_mod.mps.compile_shader(src)
                _shader_cache[cache_key] = lib

            fast_fn(kw, lib)

    else:
        # ---- non-template path: lib is fixed → bake into closure ----
        def dispatch(**kw):
            fn = _state[0]
            if fn is None:
                import torch
                lib = torch.mps.compile_shader(source)
                _ns["_dummy"] = _metal_dummy_buf(torch.device("mps"))
                _ns["_lib"] = lib
                code = f"def _fast(_kw):\n{_fn_body}\n{_call_line}"
                exec(code, _ns)  # noqa: S102 – generated code is fully controlled
                fn = _ns["_fast"]
                _state[0] = fn
            fn(kw)

    return dispatch
